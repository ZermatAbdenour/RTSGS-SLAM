from __future__ import annotations

from imgui_bundle import imgui


class SceneGraphWindow:
    def __init__(self, pcd, title: str = "Scene Graph"):
        self.pcd = pcd
        self.title = title
        self.is_open = True
        self.show_labels = True
        self.show_rel_scores = False
        self.node_radius_px = 6.0
        self.selected_iid = None
        self.zoom = 1.0
        self.zoom_min = 0.35
        self.zoom_max = 4.0

    @staticmethod
    def _class_name(class_id: int, class_names):
        if isinstance(class_names, list) and 0 <= int(class_id) < len(class_names):
            return str(class_names[int(class_id)])
        return f"class_{int(class_id)}"

    def _color(self, r: float, g: float, b: float, a: float = 1.0):
        if hasattr(imgui, "get_color_u32_rgba"):
            return imgui.get_color_u32_rgba(float(r), float(g), float(b), float(a))
        if hasattr(imgui, "color_convert_float4_to_u32"):
            return imgui.color_convert_float4_to_u32((float(r), float(g), float(b), float(a)))
        try:
            return imgui.get_color_u32((float(r), float(g), float(b), float(a)))
        except Exception:
            return 0xFFFFFFFF

    @staticmethod
    def _safe_center(node: dict):
        c = node.get("center", [0.0, 0.0, 0.0])
        if not isinstance(c, (list, tuple)) or len(c) < 3:
            return 0.0, 0.0
        return float(c[0]), float(c[2])

    def draw(self):
        if not self.is_open:
            return

        opened, self.is_open = imgui.begin(self.title, self.is_open)
        if not opened:
            imgui.end()
            return

        with self.pcd.lock:
            state = dict(getattr(self.pcd, "scene_graph_state", {}) or {})
            version = int(getattr(self.pcd, "scene_graph_version", 0))
            err = str(getattr(self.pcd, "scene_graph_last_error", "") or "")
            seg_meta = dict(getattr(self.pcd, "segmentation_metadata", {}) or {})

        nodes = state.get("nodes", []) if isinstance(state.get("nodes", []), list) else []
        relations = state.get("relations", []) if isinstance(state.get("relations", []), list) else []
        class_names = list(seg_meta.get("class_names", []))

        imgui.text_disabled(
            f"version={version} | kf={int(state.get('kf_index', -1))} | nodes={len(nodes)} | relations={len(relations)} | runtime={float(state.get('runtime_ms', 0.0)):.1f} ms"
        )
        if err:
            imgui.text_colored((0.95, 0.35, 0.35, 1.0), f"Last error: {err}")

        changed_labels, v_labels = imgui.checkbox("Show node labels", self.show_labels)
        if changed_labels:
            self.show_labels = bool(v_labels)

        imgui.same_line()
        changed_scores, v_scores = imgui.checkbox("Show relation scores", self.show_rel_scores)
        if changed_scores:
            self.show_rel_scores = bool(v_scores)

        imgui.same_line()
        _, z = imgui.slider_float("Zoom", float(self.zoom), float(self.zoom_min), float(self.zoom_max), "%.2fx")
        self.zoom = max(self.zoom_min, min(self.zoom_max, float(z)))

        imgui.same_line()
        if imgui.button("Reset zoom"):
            self.zoom = 1.0

        imgui.same_line()
        if imgui.button("Clear selection"):
            self.selected_iid = None

        if self.selected_iid is None:
            imgui.text_disabled("Tip: left-click a node to select and isolate its neighborhood.")
        else:
            imgui.text_disabled(f"Selection: {int(self.selected_iid)} (left-click selected node again to unselect)")

        avail = imgui.get_content_region_avail()
        canvas_w = max(220.0, float(avail.x))
        canvas_h = max(220.0, float(avail.y))

        p0 = imgui.get_cursor_screen_pos()
        p1 = imgui.ImVec2(p0.x + canvas_w, p0.y + canvas_h)

        draw_list = imgui.get_window_draw_list()
        bg_col = self._color(0.10, 0.11, 0.13, 1.0)
        border_col = self._color(0.35, 0.37, 0.42, 1.0)
        draw_list.add_rect_filled(p0, p1, bg_col, 4.0)
        draw_list.add_rect(p0, p1, border_col, 4.0, 0, 1.0)

        imgui.invisible_button("scenegraph_canvas", imgui.ImVec2(canvas_w, canvas_h))
        canvas_hovered = bool(imgui.is_item_hovered())
        if canvas_hovered:
            wheel = float(getattr(imgui.get_io(), "mouse_wheel", 0.0))
            if abs(wheel) > 1e-6:
                self.zoom *= 1.0 + 0.12 * wheel
                self.zoom = max(self.zoom_min, min(self.zoom_max, self.zoom))

        if len(nodes) == 0:
            draw_list.add_text(
                imgui.ImVec2(p0.x + 12.0, p0.y + 12.0),
                self._color(0.75, 0.76, 0.80, 1.0),
                "Waiting for scene graph updates...",
            )
            imgui.end()
            return

        min_x = float("inf")
        max_x = float("-inf")
        min_z = float("inf")
        max_z = float("-inf")
        for n in nodes:
            x, z = self._safe_center(n)
            min_x = min(min_x, x)
            max_x = max(max_x, x)
            min_z = min(min_z, z)
            max_z = max(max_z, z)

        if max_x <= min_x:
            max_x = min_x + 1.0
        if max_z <= min_z:
            max_z = min_z + 1.0

        pad = 24.0
        sx = (canvas_w - 2.0 * pad) / (max_x - min_x)
        sz = (canvas_h - 2.0 * pad) / (max_z - min_z)
        scale = max(1e-6, min(sx, sz) * float(self.zoom))

        cx_world = 0.5 * (min_x + max_x)
        cz_world = 0.5 * (min_z + max_z)
        canvas_cx = 0.5 * (p0.x + p1.x)
        canvas_cz = 0.5 * (p0.y + p1.y)

        def to_canvas(x: float, z: float):
            cx = canvas_cx + (x - cx_world) * scale
            cz = canvas_cz - (z - cz_world) * scale
            return cx, cz

        node_pos = {}
        node_map = {}
        for n in nodes:
            iid = int(n.get("instance_id", -1))
            x, z = self._safe_center(n)
            node_pos[iid] = to_canvas(x, z)
            node_map[iid] = n

        if self.selected_iid is not None and int(self.selected_iid) not in node_pos:
            self.selected_iid = None

        hovered_iid = None
        if canvas_hovered:
            mouse = imgui.get_mouse_pos()
            mx, my = float(mouse.x), float(mouse.y)
            best_d2 = float("inf")
            hit_r = float(self.node_radius_px + 5.0)
            hit_r2 = hit_r * hit_r
            for iid, (nx, ny) in node_pos.items():
                dx = mx - nx
                dy = my - ny
                d2 = dx * dx + dy * dy
                if d2 <= hit_r2 and d2 < best_d2:
                    best_d2 = d2
                    hovered_iid = iid

        if canvas_hovered and imgui.is_mouse_clicked(0):
            if hovered_iid is None:
                self.selected_iid = None
            elif self.selected_iid is not None and int(self.selected_iid) == int(hovered_iid):
                self.selected_iid = None
            else:
                self.selected_iid = int(hovered_iid)

        focus_iid = int(self.selected_iid) if self.selected_iid is not None else hovered_iid
        selected_mode = self.selected_iid is not None

        active_node_ids = set()
        if focus_iid is not None:
            active_node_ids.add(int(focus_iid))
            for rel in relations:
                s = int(rel.get("subject_instance_id", -1))
                o = int(rel.get("object_instance_id", -1))
                if s == focus_iid:
                    active_node_ids.add(o)
                if (not selected_mode) and o == focus_iid:
                    active_node_ids.add(s)

        grid_col = self._color(0.20, 0.22, 0.26, 1.0)
        for i in range(1, 6):
            tx = p0.x + (canvas_w / 6.0) * i
            tz = p0.y + (canvas_h / 6.0) * i
            draw_list.add_line(imgui.ImVec2(tx, p0.y), imgui.ImVec2(tx, p1.y), grid_col, 1.0)
            draw_list.add_line(imgui.ImVec2(p0.x, tz), imgui.ImVec2(p1.x, tz), grid_col, 1.0)

        edge_groups = {}
        for rel in relations:
            s = int(rel.get("subject_instance_id", -1))
            o = int(rel.get("object_instance_id", -1))
            if s not in node_pos or o not in node_pos:
                continue
            edge_groups.setdefault((s, o), []).append(rel)

        edge_col_active = self._color(0.96, 0.72, 0.22, 0.95)
        for (s, o), rel_group in edge_groups.items():
            if selected_mode:
                rel_is_active = focus_iid is None or s == focus_iid
            else:
                rel_is_active = focus_iid is None or s == focus_iid or o == focus_iid
            if self.selected_iid is not None and not rel_is_active:
                continue

            x1, y1 = node_pos[s]
            x2, y2 = node_pos[o]
            rel_col = edge_col_active
            rel_thickness = 2.0 if rel_is_active else 1.0
            draw_list.add_line(imgui.ImVec2(x1, y1), imgui.ImVec2(x2, y2), rel_col, rel_thickness)

            vx = x2 - x1
            vy = y2 - y1
            norm = (vx * vx + vy * vy) ** 0.5
            ux, uy = 0.0, 0.0
            px, py = 0.0, 0.0
            if norm > 1e-6:
                ux = vx / norm
                uy = vy / norm
                px = -uy
                py = ux
                head = 7.0
                wing = 4.0
                hx = x2 - ux * (self.node_radius_px + 1.5)
                hy = y2 - uy * (self.node_radius_px + 1.5)
                left = imgui.ImVec2(hx - ux * head - uy * wing, hy - uy * head + ux * wing)
                right = imgui.ImVec2(hx - ux * head + uy * wing, hy - uy * head - ux * wing)
                tip = imgui.ImVec2(hx, hy)
                draw_list.add_triangle_filled(tip, left, right, rel_col)

            show_labels_for_edge = self.show_rel_scores or (focus_iid is not None and rel_is_active)
            if show_labels_for_edge:
                mx = 0.5 * (x1 + x2)
                my = 0.5 * (y1 + y2)
                n_rel = len(rel_group)
                step = 12.0
                for i, rel in enumerate(rel_group):
                    if self.show_rel_scores:
                        label = f"{str(rel.get('predicate', 'rel'))} ({float(rel.get('score', 0.0)):.2f})"
                    else:
                        label = str(rel.get("predicate", "rel"))
                    offset = (i - (n_rel - 1) * 0.5) * step
                    lx = mx + px * offset + ux * 4.0
                    ly = my + py * offset + uy * 4.0
                    txt_col = self._color(0.97, 0.97, 0.99, 0.98)
                    draw_list.add_text(imgui.ImVec2(lx + 3.0, ly + 2.0), txt_col, label)

        node_fill_active = self._color(0.18, 0.80, 0.57, 1.0)
        node_fill_bg = self._color(0.36, 0.41, 0.46, 0.40)
        node_border = self._color(0.05, 0.05, 0.05, 1.0)
        for n in nodes:
            iid = int(n.get("instance_id", -1))
            if iid not in node_pos:
                continue
            if self.selected_iid is not None and iid not in active_node_ids:
                continue
            x, y = node_pos[iid]
            is_active = focus_iid is None or iid in active_node_ids
            is_hovered = focus_iid is not None and iid == focus_iid
            rad = float(self.node_radius_px + 2.0) if is_hovered else float(self.node_radius_px)
            fill_col = node_fill_active if is_active else node_fill_bg
            draw_list.add_circle_filled(imgui.ImVec2(x, y), rad, fill_col, 20)
            draw_list.add_circle(imgui.ImVec2(x, y), rad, node_border, 20, 1.0)
            if self.show_labels:
                cls = int(n.get("class_id", -1))
                conf = float(n.get("confidence", 0.0))
                cls_name = self._class_name(cls, class_names)
                label = f"{iid} | {cls_name} | {conf:.2f}"
                txt_col = self._color(0.96, 0.96, 0.98, 1.0) if is_active else self._color(0.70, 0.70, 0.74, 0.45)
                draw_list.add_text(imgui.ImVec2(x + 8.0, y - 8.0), txt_col, label)

        if focus_iid is not None and focus_iid in node_map:
            n = node_map[focus_iid]
            cls_id = int(n.get("class_id", -1))
            cls_name = self._class_name(cls_id, class_names)
            conf = float(n.get("confidence", 0.0))
            imgui.separator()
            title = "Selected object" if self.selected_iid is not None else "Hovered object"
            imgui.text(f"{title}: id={focus_iid}")
            imgui.text(f"Class: {cls_name} (id={cls_id}) | confidence={conf:.2f}")
            if selected_mode:
                imgui.text("Outgoing relationships:")
            else:
                imgui.text("Relationships:")
            shown = 0
            for rel in relations:
                s = int(rel.get("subject_instance_id", -1))
                o = int(rel.get("object_instance_id", -1))
                if s != focus_iid and o != focus_iid:
                    continue
                if selected_mode and s != focus_iid:
                    continue
                pred = str(rel.get("predicate", "rel"))
                score = float(rel.get("score", 0.0))
                if s == focus_iid:
                    other = int(o)
                    other_cls = int(node_map.get(other, {}).get("class_id", -1))
                    other_name = self._class_name(other_cls, class_names)
                    imgui.bullet_text(f"out -> {other} ({other_name}) : {pred} ({score:.2f})")
                else:
                    other = int(s)
                    other_cls = int(node_map.get(other, {}).get("class_id", -1))
                    other_name = self._class_name(other_cls, class_names)
                    imgui.bullet_text(f"in <- {other} ({other_name}) : {pred} ({score:.2f})")
                shown += 1
            if shown == 0:
                if selected_mode:
                    imgui.text_disabled("No outgoing relations for this object in current graph state.")
                else:
                    imgui.text_disabled("No relations for this object in current graph state.")

        imgui.end()
