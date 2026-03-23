from imgui_bundle import imgui
import glfw
from RTSGS.GUI import context

class ViewportWindow:
    def __init__(
        self,
        renderer,
        title: str = "Viewport",
        flip_y: bool = True,
        min_size=(1, 1),
    ):
        self.renderer = renderer
        self.fb = renderer.fb
        self.title = title
        self.flip_y = flip_y
        self.min_w, self.min_h = min_size
        self.is_open = True

    def draw(self, delta_time: float):

        expended,_ = imgui.begin(self.title)
        if(not expended):
            imgui.end()
            return int(self.fb.width), int(self.fb.height)

        changed, use_seg = imgui.checkbox(
            "Use segmentation colors",
            bool(getattr(self.renderer, "use_segmentation_colors", False)),
        )
        if changed and hasattr(self.renderer, "set_use_segmentation_colors"):
            self.renderer.set_use_segmentation_colors(use_seg)

        if bool(getattr(self.renderer, "use_segmentation_colors", False)):
            pcd = getattr(self.renderer, "pcd", None)
            has_seg = pcd is not None and getattr(pcd, "segmentation_labels", None) is not None
            if not has_seg:
                imgui.same_line()
                imgui.text_disabled("(waiting for SoftGroup output)")

        self.renderer.Render()
        avail = imgui.get_content_region_avail()
        target_w, target_h = max(self.min_w, int(avail.x)), max(self.min_h, int(avail.y))
        if target_w != int(self.fb.width) or target_h != int(self.fb.height):
            self.renderer.camera.update_resolution(target_w, target_h)
            self.fb.resize(target_w, target_h)
            self.renderer.on_resize()

        # Draw the Texture
        uv0, uv1 = ((0.0, 1.0), (1.0, 0.0)) if self.flip_y else ((0.0, 0.0), (1.0, 1.0))
        tex_ref = imgui.ImTextureRef(int(self.fb.color_tex))
        img_size = imgui.ImVec2(float(self.fb.width), float(self.fb.height))
        
        imgui.image(tex_ref, img_size, 
                    uv0=imgui.ImVec2(*uv0), 
                    uv1=imgui.ImVec2(*uv1))

        # Process Input via Camera
        self.renderer.camera.process_window_input(
            window_hovered=imgui.is_window_hovered(),
            window_focused=imgui.is_window_focused(),
            delta_time=delta_time
        )

        imgui.end()
        return int(self.fb.width), int(self.fb.height)