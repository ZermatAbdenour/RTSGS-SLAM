from __future__ import annotations

import numpy as np
from imgui_bundle import imgui
from RTSGS.GUI.ImageWidget import ImageWidget


class SegmentationLegendWindow:
    def __init__(self, pcd, renderer=None, title: str = "Segmentation"):
        self.pcd = pcd
        self.renderer = renderer
        self.title = title
        self.is_open = False
        self.only_present_classes = True
        self.show_segmented_image = True
        self.class_enabled = {}
        self._last_applied_filter = object()
        self._img_widget = None

    def _sync_class_flags(self, class_count: int):
        existing = set(self.class_enabled.keys())
        expected = set(range(class_count))
        for k in (existing - expected):
            del self.class_enabled[k]
        for k in (expected - existing):
            self.class_enabled[k] = False

    def _apply_renderer_filter(self):
        if self.renderer is None or not hasattr(self.renderer, "set_segmentation_class_filter"):
            return

        selected = [cid for cid, enabled in self.class_enabled.items() if bool(enabled)]
        class_filter = selected if len(selected) > 0 else None
        try:
            setattr(self.pcd, "segmentation_class_filter", class_filter)
        except Exception:
            pass
        if class_filter != self._last_applied_filter:
            self.renderer.set_segmentation_class_filter(class_filter)
            self._last_applied_filter = class_filter

    def _draw_color_chip(self, color_rgb):
        r, g, b = [float(c) for c in color_rgb]
        if hasattr(imgui, "color_button"):
            imgui.color_button(
                "##chip",
                (r, g, b, 1.0),
                size=imgui.ImVec2(18.0, 18.0),
            )
        else:
            imgui.text_colored((r, g, b, 1.0), "###")

    def draw(self):
        if not self.is_open:
            return

        opened, self.is_open = imgui.begin(self.title, self.is_open)
        if not opened:
            imgui.end()
            return

        changed, value = imgui.checkbox("Only classes present in map", self.only_present_classes)
        if changed:
            self.only_present_classes = bool(value)

        with self.pcd.lock:
            metadata = dict(getattr(self.pcd, "segmentation_metadata", {}) or {})
            labels_t = getattr(self.pcd, "segmentation_labels", None)
            dbg_img = getattr(self.pcd, "segmentation_debug_image_bgr", None)
            dbg_ts = float(getattr(self.pcd, "segmentation_debug_timestamp", 0.0))

        class_names = list(metadata.get("class_names", []))
        class_palette = np.asarray(metadata.get("class_palette", []), dtype=np.float32)

        if len(class_names) == 0 or class_palette.size == 0:
            self.class_enabled = {}
            self._apply_renderer_filter()
            imgui.separator()
            imgui.text_disabled("No class-color mapping yet. Wait for YOLO semantic output.")
            imgui.end()
            return

        self._sync_class_flags(len(class_names))

        present = None
        if labels_t is not None:
            labels_np = labels_t.detach().cpu().numpy().astype(np.int64)
            present = set(int(v) for v in np.unique(labels_np) if int(v) >= 0)

        imgui.separator()
        if metadata:
            ts = float(metadata.get("timestamp", 0.0))
            inf_ms = float(metadata.get("inference_ms", 0.0))
            n_seg = int(metadata.get("num_points_segmented", 0))
            n_tot = int(metadata.get("num_points_total", 0))
            imgui.text_disabled(
                f"Last update: t={ts:.1f}, inference={inf_ms:.1f} ms, points={n_seg}/{n_tot}"
            )
            imgui.separator()

        changed_img, show_img = imgui.checkbox("Show YOLO segmented image", self.show_segmented_image)
        if changed_img:
            self.show_segmented_image = bool(show_img)

        if self.show_segmented_image:
            imgui.text_disabled(f"YOLO image timestamp: {dbg_ts:.3f}")
            if dbg_img is not None and getattr(dbg_img, "size", 0) > 0:
                if self._img_widget is None:
                    self._img_widget = ImageWidget(dbg_img)
                else:
                    self._img_widget.set_image_rgb(dbg_img)
                self._img_widget.draw(fit_to_window=True)
            else:
                imgui.text_disabled("No YOLO segmented image available yet.")
            imgui.separator()

        n_selected = int(sum(1 for v in self.class_enabled.values() if bool(v)))
        if n_selected == 0:
            imgui.text_disabled("Render mode: all classes (no selection)")
        else:
            imgui.text_disabled(f"Render mode: only selected classes ({n_selected} selected)")
        imgui.separator()

        for class_id, class_name in enumerate(class_names):
            if class_id >= class_palette.shape[0]:
                break
            if self.only_present_classes and present is not None and class_id not in present:
                continue

            changed, enabled = imgui.checkbox(f"##seg_class_{class_id}", bool(self.class_enabled.get(class_id, True)))
            if changed:
                self.class_enabled[class_id] = bool(enabled)
            imgui.same_line()
            color = class_palette[class_id]
            self._draw_color_chip(color)
            imgui.same_line()
            imgui.text(
                f"{class_id:02d} | {class_name} | rgb=({int(color[0]*255)}, {int(color[1]*255)}, {int(color[2]*255)})"
            )

        self._apply_renderer_filter()

        imgui.end()
