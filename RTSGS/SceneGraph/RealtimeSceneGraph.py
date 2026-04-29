import os
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


class RelationHead(nn.Module):
    def __init__(self, hidden_dim: int = 512, num_rel_classes: int = 41, nhead: int = 8, num_rel_layers: int = 3):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=1024,
            batch_first=True,
            dropout=0.1,
        )

        self.rel_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_rel_layers,
        )

        self.rel_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_rel_classes),
        )

    def forward(self, object_features: torch.Tensor) -> torch.Tensor:
        refined = self.rel_transformer(object_features)
        bsz, k, d = refined.shape

        q_i = refined.unsqueeze(2).expand(bsz, k, k, d)
        q_j = refined.unsqueeze(1).expand(bsz, k, k, d)
        pair_features = torch.cat([q_i, q_j, q_i - q_j, q_i * q_j], dim=-1)
        return self.rel_predictor(pair_features)


class RealtimeSceneGraphRuntime:
    def __init__(self, pcd, config, project_root: str):
        self.pcd = pcd
        self.config = config
        self.project_root = project_root

        self.enabled = bool(config.get("scenegraph_enabled", True))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.max_objects_per_keyframe = int(config.get("scenegraph_max_objects_per_keyframe", 12))
        self.update_stride = max(1, int(config.get("scenegraph_update_stride", 1)))
        self.max_nodes = int(config.get("scenegraph_max_nodes", 48))
        # Keep all relations above the fixed threshold (RelationHead only, no MP-GNN).
        self.max_relations = int(config.get("scenegraph_max_relations", 0))
        self.rel_threshold = 0.5
        self.instance_ttl = int(config.get("scenegraph_instance_ttl", 30))
        self.instance_merge_dist = float(config.get("scenegraph_instance_merge_dist", 0.6))

        self.relation_head_ckpt = self._abs_path(
            str(
                config.get(
                    "scenegraph_relation_head_checkpoint",
                    "Datasets/3RScan/3RScan/data/scans/relation_branch_hard_negative_bce.pt",
                )
            )
        )
        self.relationships_path = self._abs_path(
            str(config.get("scenegraph_relationships_path", "Datasets/3DSSG/3DSSG/relationships.txt"))
        )

        self._relation_head = None
        self._obj_proj = None
        self._hidden_dim = None
        self._rel_dim = None

        self.relationship_names: List[str] = []
        self.instance_state: Dict[int, Dict[str, torch.Tensor]] = {}
        self.relation_cache: Dict[Tuple[int, int, int], Dict[str, object]] = {}
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

    def _load_relation_head(self) -> bool:
        if self._relation_head is not None:
            return True
        if not os.path.exists(self.relation_head_ckpt):
            self._last_error = f"RelationHead checkpoint not found: {self.relation_head_ckpt}"
            print(f"[SceneGraph] {self._last_error}")
            return False

        ckpt = torch.load(self.relation_head_ckpt, map_location=self.device)
        state = ckpt.get("model_state_dict", ckpt.get("relation_head_state_dict", {}))
        if "rel_predictor.0.weight" not in state:
            self._last_error = f"Invalid RelationHead checkpoint format: {self.relation_head_ckpt}"
            print(f"[SceneGraph] {self._last_error}")
            return False

        hidden_dim = int(ckpt.get("hidden_dim", ckpt.get("object_feature_dim", 512)))
        num_rel_classes = int(ckpt.get("num_rel_classes", state["rel_predictor.3.weight"].shape[0]))

        model = RelationHead(
            hidden_dim=hidden_dim,
            num_rel_classes=num_rel_classes,
            nhead=8,
            num_rel_layers=3,
        ).to(self.device)
        model.load_state_dict(state, strict=True)
        model.eval()
        self._relation_head = model
        self._hidden_dim = hidden_dim
        self._rel_dim = num_rel_classes
        return True

    def start(self):
        if not self.enabled:
            return
        ok = self._load_relation_head()
        self._ready = bool(ok)
        if self._ready:
            if not self.relationship_names:
                self.relationship_names = self._read_relationship_names()
            print("[SceneGraph] Runtime initialized (RelationHead only).")
        else:
            self.enabled = False
            print("[SceneGraph] Disabled due to initialization error.")

    def stop(self):
        return

    @staticmethod
    def _standardize_features(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        mu = x.mean(dim=0, keepdim=True)
        sigma = x.std(dim=0, keepdim=True, unbiased=False).clamp_min(eps)
        return (x - mu) / sigma

    def _maybe_build_projection(self, in_dim: int):
        if self._hidden_dim is None:
            return
        if int(in_dim) == int(self._hidden_dim):
            self._obj_proj = None
            return
        if self._obj_proj is None or int(getattr(self._obj_proj, "in_features", -1)) != int(in_dim):
            self._obj_proj = nn.Linear(int(in_dim), int(self._hidden_dim)).to(self.device)
            self._obj_proj.eval()
            print(
                f"[SceneGraph] Warning: object feature dim {in_dim} != relation head dim {self._hidden_dim}. "
                "Using a fresh projection layer."
            )

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
        if stale_ids:
            stale_set = set(int(x) for x in stale_ids)
            if self.relation_cache:
                self.relation_cache = {
                    key: value
                    for key, value in self.relation_cache.items()
                    if key[0] not in stale_set and key[1] not in stale_set
                }

    def _merge_instance_id(self, inst_id: int, class_id: int, center: torch.Tensor) -> int:
        if inst_id in self.instance_state:
            return inst_id
        if self.instance_merge_dist <= 0.0:
            return inst_id

        best_id = inst_id
        best_dist = float("inf")
        for iid, st in self.instance_state.items():
            if int(st.get("class_id", -1)) != int(class_id):
                continue
            prev_center = st.get("center", None)
            if not isinstance(prev_center, torch.Tensor):
                continue
            dist = float(torch.linalg.norm(center - prev_center).item())
            if dist < best_dist:
                best_dist = dist
                best_id = int(iid)

        if best_dist <= float(self.instance_merge_dist):
            return int(best_id)
        return int(inst_id)

    def update_from_segmenter(self, seg_result: Optional[dict], kf_index: int):
        if not self.enabled or not self._ready:
            return
        if seg_result is None:
            return
        publish_state = (int(kf_index) % self.update_stride) == 0

        observations = seg_result.get("scenegraph_observations", [])
        if not isinstance(observations, list):
            observations = []

        t0 = time.perf_counter()

        self._prune_stale_instances(int(kf_index))

        # Ensure every current segmentation instance exists in the scene-graph state.
        # This keeps old objects alive even when they are not in the per-frame observations.
        with self.pcd.lock:
            seg_instances = list(getattr(self.pcd, "segmentation_instances", []) or [])
        for inst in seg_instances:
            if not isinstance(inst, dict):
                continue
            iid = int(inst.get("instance_id", -1))
            if iid < 0:
                continue

            bmin = torch.tensor(inst.get("bbox_min", [0.0, 0.0, 0.0]), dtype=torch.float32, device=self.device)
            bmax = torch.tensor(inst.get("bbox_max", [0.0, 0.0, 0.0]), dtype=torch.float32, device=self.device)
            center = torch.tensor(inst.get("center", [0.0, 0.0, 0.0]), dtype=torch.float32, device=self.device)
            obb = inst.get("obb", {}) if isinstance(inst.get("obb", {}), dict) else {}
            if all(k in obb for k in ("centroid", "axesLengths")):
                center = torch.tensor(obb.get("centroid", center.detach().cpu().tolist()), dtype=torch.float32, device=self.device)
                size = torch.tensor(obb.get("axesLengths", (bmax - bmin).detach().cpu().tolist()), dtype=torch.float32, device=self.device).clamp_min(1e-6)
            else:
                size = (bmax - bmin).clamp_min(1e-6)

            if iid in self.instance_state:
                st = self.instance_state[iid]
                st["center"] = center
                st["size"] = size
                st["class_id"] = int(inst.get("class_id", -1))
                st["confidence"] = float(inst.get("score", st.get("confidence", 0.0)))
                st["last_seen_kf"] = int(kf_index)
            else:
                if self._hidden_dim is None:
                    continue
                self.instance_state[iid] = {
                    "node_sum": torch.zeros(int(self._hidden_dim), dtype=torch.float32, device=self.device),
                    "node_count": 1,
                    "center": center,
                    "size": size,
                    "class_id": int(inst.get("class_id", -1)),
                    "confidence": float(inst.get("score", 0.0)),
                    "last_seen_kf": int(kf_index),
                }

        observations = sorted(
            [o for o in observations if isinstance(o, dict)],
            key=lambda x: float(x.get("confidence", 0.0)),
            reverse=True,
        )

        # Keep only the best observation per instance in this frame.
        best_by_id: Dict[int, dict] = {}
        for obs in observations:
            iid = int(obs.get("instance_id", -1))
            if iid < 0:
                continue
            if iid not in best_by_id or float(obs.get("confidence", 0.0)) > float(best_by_id[iid].get("confidence", 0.0)):
                best_by_id[iid] = obs

        observations = list(best_by_id.values())
        observations = sorted(
            observations,
            key=lambda x: float(x.get("confidence", 0.0)),
            reverse=True,
        )
        max_per_frame = int(self.max_objects_per_keyframe)
        if max_per_frame > 0:
            observations = observations[:max_per_frame]

        frame_feats = []
        frame_inst_ids = []
        debug = {
            "frame_observations": int(len(observations)),
            "frame_features": 0,
            "frame_pairs": 0,
            "edge_pairs_total": 0,
            "relations_raw": 0,
            "relations_kept": 0,
            "max_relation_score": 0.0,
            "max_relation_prob": 0.0,
        }

        with torch.inference_mode():
            for obs in observations:
                inst_id = int(obs.get("instance_id", -1))
                if inst_id < 0:
                    continue

                feat = obs.get("object_feature", None)
                if feat is None:
                    continue
                if isinstance(feat, torch.Tensor):
                    feat_t = feat.to(self.device, dtype=torch.float32).reshape(-1)
                elif isinstance(feat, (list, tuple)):
                    feat_t = torch.tensor(feat, dtype=torch.float32, device=self.device).reshape(-1)
                else:
                    continue

                self._maybe_build_projection(int(feat_t.shape[0]))
                if self._obj_proj is not None:
                    feat_t = self._obj_proj(feat_t)

                center = torch.tensor(obs.get("center", [0.0, 0.0, 0.0]), dtype=torch.float32, device=self.device)
                bmin = torch.tensor(obs.get("bbox_min", [0.0, 0.0, 0.0]), dtype=torch.float32, device=self.device)
                bmax = torch.tensor(obs.get("bbox_max", [0.0, 0.0, 0.0]), dtype=torch.float32, device=self.device)

                obb = obs.get("obb", {}) if isinstance(obs.get("obb", {}), dict) else {}
                if all(k in obb for k in ("centroid", "axesLengths")):
                    center = torch.tensor(obb.get("centroid", center.detach().cpu().tolist()), dtype=torch.float32, device=self.device)
                    size = torch.tensor(obb.get("axesLengths", (bmax - bmin).detach().cpu().tolist()), dtype=torch.float32, device=self.device).clamp_min(1e-6)
                else:
                    size = (bmax - bmin).clamp_min(1e-6)

                inst_id = self._merge_instance_id(inst_id, int(obs.get("class_id", -1)), center)

                if inst_id in self.instance_state:
                    st = self.instance_state[inst_id]
                    st["node_sum"] = st["node_sum"] + feat_t
                    st["node_count"] = int(st["node_count"]) + 1
                    st["center"] = center
                    st["size"] = size
                    st["class_id"] = int(obs.get("class_id", -1))
                    st["confidence"] = float(obs.get("confidence", 0.0))
                    st["last_seen_kf"] = int(kf_index)
                else:
                    self.instance_state[inst_id] = {
                        "node_sum": feat_t.clone(),
                        "node_count": 1,
                        "center": center,
                        "size": size,
                        "class_id": int(obs.get("class_id", -1)),
                        "confidence": float(obs.get("confidence", 0.0)),
                        "last_seen_kf": int(kf_index),
                    }

                frame_feats.append(feat_t)
                frame_inst_ids.append(inst_id)
                debug["frame_features"] += 1

            if len(frame_feats) >= 2:
                obj_feats = torch.stack(frame_feats, dim=0).unsqueeze(0)
                frame_logits = self._relation_head(obj_feats)[0]
                frame_probs = torch.sigmoid(frame_logits)
                k = int(frame_probs.shape[0])
                debug["frame_pairs"] = int(k * (k - 1))

                rel_thresh = float(self.rel_threshold)
                for i in range(k):
                    src_id = int(frame_inst_ids[i])
                    for j in range(k):
                        if i == j:
                            continue
                        dst_id = int(frame_inst_ids[j])
                        active_rel = torch.where(frame_probs[i, j] >= rel_thresh)[0]
                        for r_i_t in active_rel:
                            r_i = int(r_i_t.item())
                            score = float(frame_probs[i, j, r_i].item())
                            rel_name = (
                                self.relationship_names[r_i]
                                if r_i < len(self.relationship_names)
                                else f"rel_{r_i}"
                            )
                            key = (src_id, dst_id, r_i)
                            entry = self.relation_cache.get(key)
                            if entry is None or float(entry.get("score", 0.0)) < score:
                                self.relation_cache[key] = {
                                    "score": score,
                                    "predicate": rel_name,
                                    "last_seen_kf": int(kf_index),
                                }
                            else:
                                entry["last_seen_kf"] = int(kf_index)

                            debug["relations_raw"] += 1
                            if score > debug["max_relation_score"]:
                                debug["max_relation_score"] = score

                debug["max_relation_prob"] = debug["max_relation_score"]

            if not publish_state:
                return

            # Keep all historical instances in the scene graph.
            active = sorted(
                self.instance_state.items(),
                key=lambda kv: (int(kv[1].get("last_seen_kf", -1)), float(kv[1].get("confidence", 0.0))),
                reverse=True,
            )

            if len(active) < 1:
                self._publish_state(
                    int(kf_index),
                    [],
                    [],
                    scenegraph_ms=(time.perf_counter() - t0) * 1000.0,
                    debug=debug,
                )
                return

            inst_ids = [iid for iid, _ in active]
            obj_to_idx = {iid: i for i, iid in enumerate(inst_ids)}

            # Extract relations first, then collect only the instance IDs involved in relations
            relations = []
            relation_instances = set()
            for (src_id, dst_id, rel_id), entry in self.relation_cache.items():
                if src_id not in obj_to_idx or dst_id not in obj_to_idx:
                    continue
                score = float(entry.get("score", 0.0))
                rel_name = entry.get("predicate")
                if not isinstance(rel_name, str):
                    rel_name = self.relationship_names[rel_id] if rel_id < len(self.relationship_names) else f"rel_{rel_id}"
                relations.append(
                    {
                        "subject_instance_id": int(src_id),
                        "object_instance_id": int(dst_id),
                        "predicate_id": int(rel_id),
                        "predicate": rel_name,
                        "score": score,
                    }
                )
                relation_instances.add(int(src_id))
                relation_instances.add(int(dst_id))

            # Only include nodes that are involved in relations (no orphaned nodes)
            node_feats = []
            nodes = []
            for iid, st in active:
                if iid not in relation_instances:
                    continue
                cnt = max(int(st.get("node_count", 1)), 1)
                node_feats.append(st["node_sum"] / float(cnt))
                nodes.append(
                    {
                        "instance_id": int(iid),
                        "class_id": int(st.get("class_id", -1)),
                        "confidence": float(st.get("confidence", 0.0)),
                        "center": st["center"].detach().cpu().tolist(),
                        "size": st["size"].detach().cpu().tolist(),
                    }
                )

            # No need to check len(node_feats) < 2 anymore since we're filtering by relations
            relations.sort(key=lambda x: x["score"], reverse=True)
            debug["edge_pairs_total"] = int(len(self.relation_cache))
            debug["relations_kept"] = int(len(relations))

            self._publish_state(
                int(kf_index),
                nodes,
                relations,
                scenegraph_ms=(time.perf_counter() - t0) * 1000.0,
                debug=debug,
            )

    def _publish_state(self, kf_index: int, nodes: List[dict], relations: List[dict], scenegraph_ms: float, debug: Optional[dict] = None):
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
        if isinstance(debug, dict):
            state["debug"] = dict(debug)

        with self.pcd.lock:
            self.pcd.scene_graph_state = state
            self.pcd.scene_graph_version = int(self.version)
            self.pcd.scene_graph_last_error = ""

