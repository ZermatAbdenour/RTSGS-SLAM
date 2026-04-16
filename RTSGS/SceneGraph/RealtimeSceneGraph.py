import os
import time
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


class GRUResidualUpdater(nn.Module):
    def __init__(self, emb_dim: int, hidden_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(emb_dim * 2, hidden_dim)
        self.gru_cell = nn.GRUCell(hidden_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, prev_e: torch.Tensor, obs_e: torch.Tensor, hidden: torch.Tensor):
        x = torch.cat([prev_e, obs_e], dim=-1)
        x = F.gelu(self.input_proj(x))
        hidden = self.gru_cell(x, hidden)
        hidden = self.dropout(hidden)
        residual = self.output_proj(hidden)
        return residual, hidden


class MPNNLayer(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.msg_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.upd_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, h: torch.Tensor, edges: torch.Tensor):
        if edges.numel() == 0:
            return h
        src, dst = edges[:, 0], edges[:, 1]
        msg = self.msg_mlp(torch.cat([h[src], h[dst]], dim=-1))
        agg = torch.zeros_like(h)
        agg.index_add_(0, dst, msg)
        h_new = self.upd_mlp(torch.cat([h, agg], dim=-1))
        return h + h_new


class RelationMPGNN(nn.Module):
    def __init__(self, clip_dim: int, geom_dim: int, hidden_dim: int, rel_classes: int, num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.clip_proj = nn.Sequential(
            nn.Linear(clip_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        self.node_fuse = nn.Sequential(
            nn.Linear(256 + geom_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.layers = nn.ModuleList([MPNNLayer(hidden_dim) for _ in range(num_layers)])
        self.rel_feat_dim = 8
        self.edge_head = nn.Sequential(
            nn.Linear(2 * hidden_dim + self.rel_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, rel_classes),
        )

    def encode_nodes(self, clip_x: torch.Tensor, geom_x: torch.Tensor, graph_edges: torch.Tensor):
        clip_z = self.clip_proj(clip_x)
        h = self.node_fuse(torch.cat([clip_z, geom_x], dim=-1))
        for layer in self.layers:
            h = layer(h, graph_edges)
        return h

    def build_edge_features(self, geom_raw_x: torch.Tensor, query_pairs: torch.Tensor):
        s, o = query_pairs[:, 0], query_pairs[:, 1]

        c_s = geom_raw_x[s, :3]
        c_o = geom_raw_x[o, :3]
        delta = c_o - c_s

        dist = torch.norm(delta, dim=-1, keepdim=True)
        log_dist = torch.log1p(dist)

        size_s = geom_raw_x[s, 3:6].clamp_min(1e-6)
        size_o = geom_raw_x[o, 3:6].clamp_min(1e-6)
        log_size_ratio = torch.log(size_o / size_s)

        n_s = geom_raw_x[s, 15:18]
        n_o = geom_raw_x[o, 15:18]
        normal_cos = F.cosine_similarity(n_s, n_o, dim=-1, eps=1e-8).unsqueeze(-1)

        return torch.cat([delta, log_dist, log_size_ratio, normal_cos], dim=-1)

    def forward(self, clip_x: torch.Tensor, geom_x: torch.Tensor, graph_edges: torch.Tensor, query_pairs: torch.Tensor, geom_raw_x: Optional[torch.Tensor] = None):
        h = self.encode_nodes(clip_x, geom_x, graph_edges)
        s, o = query_pairs[:, 0], query_pairs[:, 1]
        geom_for_edges = geom_x if geom_raw_x is None else geom_raw_x
        rel_feat = self.build_edge_features(geom_for_edges, query_pairs)
        edge_feat = torch.cat([h[s], h[o], rel_feat], dim=-1)
        return self.edge_head(edge_feat)


class RealtimeSceneGraphRuntime:
    def __init__(self, pcd, config, project_root: str):
        self.pcd = pcd
        self.config = config
        self.project_root = project_root

        self.enabled = bool(config.get("scenegraph_enabled", True))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.clip_model_name = str(config.get("scenegraph_clip_model_name", "ViT-B/32"))
        self.max_objects_per_keyframe = int(config.get("scenegraph_max_objects_per_keyframe", 12))
        self.update_stride = max(1, int(config.get("scenegraph_update_stride", 1)))
        self.max_nodes = int(config.get("scenegraph_max_nodes", 48))
        self.max_relations = int(config.get("scenegraph_max_relations", 256))
        self.rel_threshold = float(config.get("scenegraph_rel_threshold", 0.5))
        self.k_neighbors = int(config.get("scenegraph_knn_k", 6))
        self.instance_ttl = int(config.get("scenegraph_instance_ttl", 30))

        self.residual_ckpt = self._abs_path(
            str(config.get("scenegraph_residual_checkpoint", "SceneGraph/checkpoints/temporal_residual/best.pt"))
        )
        self.gnn_ckpt = self._abs_path(
            str(config.get("scenegraph_gnn_checkpoint", "SceneGraph/checkpoints/scenegraph_multiscan/best.pt"))
        )
        self.relationships_path = self._abs_path(
            str(config.get("scenegraph_relationships_path", "Datasets/3DSSG/3DSSG/relationships.txt"))
        )

        self._clip_model = None
        self._clip_preprocess = None
        self._residual_model = None
        self._gnn_model = None

        self.relationship_names: List[str] = []
        self.instance_state: Dict[int, Dict[str, torch.Tensor]] = {}
        self.version = 0

        self._ready = False
        self._last_error = ""

    def _abs_path(self, p: str) -> str:
        if os.path.isabs(p):
            return p
        return os.path.join(self.project_root, p)

    def _read_relationship_names(self) -> List[str]:
        if not os.path.exists(self.relationships_path):
            return []
        out = []
        with open(self.relationships_path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s:
                    out.append(s)
        return out

    def _load_clip(self) -> bool:
        if self._clip_model is not None:
            return True
        try:
            import clip  # type: ignore

            model, preprocess = clip.load(self.clip_model_name, device=self.device)
            model.eval()
            self._clip_model = model
            self._clip_preprocess = preprocess
            return True
        except Exception as e:
            self._last_error = f"Failed to load CLIP model '{self.clip_model_name}': {e}"
            print(f"[SceneGraph] {self._last_error}")
            return False

    def _load_residual(self) -> bool:
        if self._residual_model is not None:
            return True
        if not os.path.exists(self.residual_ckpt):
            self._last_error = f"Residual checkpoint not found: {self.residual_ckpt}"
            print(f"[SceneGraph] {self._last_error}")
            return False

        ckpt = torch.load(self.residual_ckpt, map_location=self.device)
        state = ckpt.get("model_state_dict", {})
        if "input_proj.weight" not in state:
            self._last_error = f"Invalid residual checkpoint format: {self.residual_ckpt}"
            print(f"[SceneGraph] {self._last_error}")
            return False

        emb_dim = int(state["input_proj.weight"].shape[1] // 2)
        hidden_dim = int(state["input_proj.weight"].shape[0])
        model = GRUResidualUpdater(emb_dim=emb_dim, hidden_dim=hidden_dim, dropout=0.1).to(self.device)
        model.load_state_dict(state, strict=True)
        model.eval()
        self._residual_model = model
        return True

    def _load_gnn(self) -> bool:
        if self._gnn_model is not None:
            return True
        if not os.path.exists(self.gnn_ckpt):
            self._last_error = f"GNN checkpoint not found: {self.gnn_ckpt}"
            print(f"[SceneGraph] {self._last_error}")
            return False

        ckpt = torch.load(self.gnn_ckpt, map_location=self.device)
        state = ckpt.get("model_state_dict", {})
        required = ["clip_proj.0.weight", "node_fuse.0.weight", "edge_head.2.weight"]
        if not all(k in state for k in required):
            self._last_error = f"Invalid GNN checkpoint format: {self.gnn_ckpt}"
            print(f"[SceneGraph] {self._last_error}")
            return False

        clip_dim = int(state["clip_proj.0.weight"].shape[1])
        hidden_dim = int(state["node_fuse.0.weight"].shape[0])
        geom_dim = int(state["node_fuse.0.weight"].shape[1] - 256)
        rel_classes = int(state["edge_head.2.weight"].shape[0])
        num_layers = len([k for k in state.keys() if k.startswith("layers.") and k.endswith("msg_mlp.0.weight")])
        num_layers = max(1, int(num_layers))

        model = RelationMPGNN(
            clip_dim=clip_dim,
            geom_dim=geom_dim,
            hidden_dim=hidden_dim,
            rel_classes=rel_classes,
            num_layers=num_layers,
            dropout=0.1,
        ).to(self.device)
        model.load_state_dict(state, strict=True)
        model.eval()
        self._gnn_model = model

        if not self.relationship_names:
            self.relationship_names = self._read_relationship_names()
        return True

    def start(self):
        if not self.enabled:
            return
        ok = self._load_clip() and self._load_residual() and self._load_gnn()
        self._ready = bool(ok)
        if self._ready:
            print("[SceneGraph] Runtime initialized (CLIP + residual + MP-GNN).")
        else:
            self.enabled = False
            print("[SceneGraph] Disabled due to initialization error.")

    def stop(self):
        return

    @staticmethod
    def _normalize_embeddings(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)

    @staticmethod
    def _standardize_features(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        mu = x.mean(dim=0, keepdim=True)
        sigma = x.std(dim=0, keepdim=True, unbiased=False).clamp_min(eps)
        return (x - mu) / sigma

    @staticmethod
    def _build_knn_graph(centroids: torch.Tensor, k: int = 6) -> torch.Tensor:
        n = int(centroids.shape[0])
        if n <= 1:
            return torch.zeros((0, 2), dtype=torch.long, device=centroids.device)
        k_eff = min(int(k), n - 1)
        d = torch.cdist(centroids, centroids)
        d.fill_diagonal_(float("inf"))
        nn_idx = torch.topk(d, k_eff, largest=False, dim=1).indices
        src = torch.arange(n, device=centroids.device).unsqueeze(1).repeat(1, k_eff).reshape(-1)
        dst = nn_idx.reshape(-1)
        e = torch.stack([src, dst], dim=1)
        e_rev = e[:, [1, 0]]
        return torch.unique(torch.cat([e, e_rev], dim=0), dim=0)

    def _encode_clip_batch(self, obs_list: List[dict]) -> List[torch.Tensor]:
        if not obs_list:
            return []
        imgs = []
        valid_ids = []
        for i, obs in enumerate(obs_list):
            crop_bgr = obs.get("crop_bgr", None)
            if crop_bgr is None:
                continue
            arr = np.asarray(crop_bgr, dtype=np.uint8)
            if arr.ndim != 3 or arr.shape[2] != 3:
                continue
            pil_img = Image.fromarray(arr[..., ::-1])
            imgs.append(self._clip_preprocess(pil_img))
            valid_ids.append(i)

        if not imgs:
            return []

        with torch.inference_mode():
            batch = torch.stack(imgs, dim=0).to(self.device)
            emb = self._clip_model.encode_image(batch)
            # CLIP on CUDA may emit fp16; downstream residual/GNN checkpoints are fp32.
            emb = self._normalize_embeddings(emb.float())

        out: List[torch.Tensor] = [None] * len(obs_list)  # type: ignore
        for j, idx in enumerate(valid_ids):
            out[idx] = emb[j]
        return out

    def _prune_stale_instances(self, current_kf: int):
        if self.instance_ttl <= 0:
            return
        stale_ids = []
        for iid, st in self.instance_state.items():
            last_seen = int(st.get("last_seen_kf", -1))
            if last_seen >= 0 and (current_kf - last_seen) > self.instance_ttl:
                stale_ids.append(iid)
        for iid in stale_ids:
            self.instance_state.pop(iid, None)

    def update_from_segmenter(self, seg_result: Optional[dict], kf_index: int):
        if not self.enabled or not self._ready:
            return
        if seg_result is None:
            return
        if (int(kf_index) % self.update_stride) != 0:
            return

        observations = seg_result.get("scenegraph_observations", [])
        if not isinstance(observations, list) or len(observations) == 0:
            return

        t0 = time.perf_counter()

        observations = sorted(
            [o for o in observations if isinstance(o, dict)],
            key=lambda x: float(x.get("confidence", 0.0)),
            reverse=True,
        )[: self.max_objects_per_keyframe]

        clip_embeddings = self._encode_clip_batch(observations)

        with torch.inference_mode():
            for obs, emb in zip(observations, clip_embeddings):
                if emb is None:
                    continue
                inst_id = int(obs.get("instance_id", -1))
                if inst_id < 0:
                    continue

                center = torch.tensor(obs.get("center", [0.0, 0.0, 0.0]), dtype=torch.float32, device=self.device)
                bmin = torch.tensor(obs.get("bbox_min", [0.0, 0.0, 0.0]), dtype=torch.float32, device=self.device)
                bmax = torch.tensor(obs.get("bbox_max", [0.0, 0.0, 0.0]), dtype=torch.float32, device=self.device)
                obb = obs.get("obb", {}) if isinstance(obs.get("obb", {}), dict) else {}
                if all(k in obb for k in ("centroid", "axesLengths", "normalizedAxes")):
                    center = torch.tensor(obb.get("centroid", center.detach().cpu().tolist()), dtype=torch.float32, device=self.device)
                    size = torch.tensor(obb.get("axesLengths", (bmax - bmin).detach().cpu().tolist()), dtype=torch.float32, device=self.device).clamp_min(1e-6)
                    norm_axes = torch.tensor(obb.get("normalizedAxes", [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]), dtype=torch.float32, device=self.device)
                else:
                    size = (bmax - bmin).clamp_min(1e-6)
                    norm_axes = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], dtype=torch.float32, device=self.device)

                dom = obs.get("dominantNormal", [0.0, 0.0, 1.0])
                if isinstance(dom, (list, tuple)) and len(dom) >= 3:
                    normal = torch.tensor([float(dom[0]), float(dom[1]), float(dom[2])], dtype=torch.float32, device=self.device)
                else:
                    normal = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=self.device)

                if inst_id in self.instance_state:
                    st = self.instance_state[inst_id]
                    prev_e = st["embedding"].unsqueeze(0).float()
                    obs_e = emb.unsqueeze(0).float()
                    hidden = st["hidden"].unsqueeze(0).float()
                    residual, hidden_new = self._residual_model(prev_e, obs_e, hidden)
                    upd = self._normalize_embeddings(prev_e + residual).squeeze(0)
                    st["embedding"] = upd
                    st["hidden"] = hidden_new.squeeze(0)
                    st["center"] = center
                    st["size"] = size
                    st["norm_axes"] = norm_axes
                    st["normal"] = normal
                    st["class_id"] = int(obs.get("class_id", -1))
                    st["confidence"] = float(obs.get("confidence", 0.0))
                    st["last_seen_kf"] = int(kf_index)
                else:
                    hidden_dim = int(self._residual_model.gru_cell.hidden_size)
                    self.instance_state[inst_id] = {
                        "embedding": emb,
                        "hidden": torch.zeros((hidden_dim,), dtype=torch.float32, device=self.device),
                        "center": center,
                        "size": size,
                        "norm_axes": norm_axes,
                        "normal": normal,
                        "class_id": int(obs.get("class_id", -1)),
                        "confidence": float(obs.get("confidence", 0.0)),
                        "last_seen_kf": int(kf_index),
                    }

            self._prune_stale_instances(int(kf_index))

            active = sorted(
                self.instance_state.items(),
                key=lambda kv: (int(kv[1].get("last_seen_kf", -1)), float(kv[1].get("confidence", 0.0))),
                reverse=True,
            )[: self.max_nodes]

            if len(active) < 2:
                self._publish_state(int(kf_index), [], [], scenegraph_ms=(time.perf_counter() - t0) * 1000.0)
                return

            inst_ids = [iid for iid, _ in active]
            clip_x = torch.stack([st["embedding"] for _, st in active], dim=0).float()
            clip_x = self._standardize_features(clip_x)

            geom_raw = []
            for _, st in active:
                c = st["center"]
                s = st["size"]
                a = st["norm_axes"]
                n = st["normal"]
                geom_raw.append(torch.cat([c, s, a, n], dim=0))
            geom_raw_x = torch.stack(geom_raw, dim=0).float()
            geom_x = self._standardize_features(geom_raw_x)

            graph_edges = self._build_knn_graph(geom_raw_x[:, :3], k=self.k_neighbors)

            n_nodes = int(clip_x.shape[0])
            query_pairs = []
            for i in range(n_nodes):
                for j in range(n_nodes):
                    if i != j:
                        query_pairs.append((i, j))
            query_pairs_t = torch.tensor(query_pairs, dtype=torch.long, device=self.device)

            logits = self._gnn_model(clip_x, geom_x, graph_edges, query_pairs_t, geom_raw_x=geom_raw_x)
            probs = torch.sigmoid(logits)

            relations = []
            rel_thresh = float(self.rel_threshold)
            rel_classes = int(probs.shape[1])
            for e_idx in range(probs.shape[0]):
                s_i = int(query_pairs_t[e_idx, 0].item())
                o_i = int(query_pairs_t[e_idx, 1].item())
                active_rel = torch.where(probs[e_idx] >= rel_thresh)[0]
                for r_i_t in active_rel:
                    r_i = int(r_i_t.item())
                    if r_i < 0 or r_i >= rel_classes:
                        continue
                    score = float(probs[e_idx, r_i].item())
                    rel_name = self.relationship_names[r_i] if r_i < len(self.relationship_names) else f"rel_{r_i}"
                    relations.append(
                        {
                            "subject_instance_id": int(inst_ids[s_i]),
                            "object_instance_id": int(inst_ids[o_i]),
                            "predicate_id": r_i,
                            "predicate": rel_name,
                            "score": score,
                        }
                    )

            relations.sort(key=lambda x: x["score"], reverse=True)
            if self.max_relations > 0:
                relations = relations[: self.max_relations]

            nodes = []
            for iid, st in active:
                nodes.append(
                    {
                        "instance_id": int(iid),
                        "class_id": int(st.get("class_id", -1)),
                        "confidence": float(st.get("confidence", 0.0)),
                        "center": st["center"].detach().cpu().tolist(),
                        "size": st["size"].detach().cpu().tolist(),
                    }
                )

            self._publish_state(
                int(kf_index),
                nodes,
                relations,
                scenegraph_ms=(time.perf_counter() - t0) * 1000.0,
            )

    def _publish_state(self, kf_index: int, nodes: List[dict], relations: List[dict], scenegraph_ms: float):
        self.version += 1
        state = {
            "version": int(self.version),
            "kf_index": int(kf_index),
            "timestamp": float(time.time()),
            "num_nodes": int(len(nodes)),
            "num_relations": int(len(relations)),
            "nodes": nodes,
            "relations": relations,
            "runtime_ms": float(scenegraph_ms),
        }

        with self.pcd.lock:
            self.pcd.scene_graph_state = state
            self.pcd.scene_graph_version = int(self.version)
            self.pcd.scene_graph_last_error = ""

