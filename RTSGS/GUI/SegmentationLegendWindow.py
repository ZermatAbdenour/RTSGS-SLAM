from __future__ import annotations

import numpy as np
from imgui_bundle import imgui


class SegmentationLegendWindow:
    def __init__(self, pcd, title: str = "Segmentation Legend"):
        self.pcd = pcd
        self.title = title
        self.is_open = False
        self.only_present_classes = True

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

        class_names = list(metadata.get("class_names", []))
        class_palette = np.asarray(metadata.get("class_palette", []), dtype=np.float32)

        if len(class_names) == 0 or class_palette.size == 0:
            imgui.separator()
            imgui.text_disabled("No class-color mapping yet. Wait for YOLO semantic output.")
            imgui.end()
            return

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

        for class_id, class_name in enumerate(class_names):
            if class_id >= class_palette.shape[0]:
                break
            if self.only_present_classes and present is not None and class_id not in present:
                continue

            color = class_palette[class_id]
            self._draw_color_chip(color)
            imgui.same_line()
            imgui.text(
                f"{class_id:02d} | {class_name} | rgb=({int(color[0]*255)}, {int(color[1]*255)}, {int(color[2]*255)})"
            )

        imgui.end()
