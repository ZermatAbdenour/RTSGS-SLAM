import glfw
from imgui_bundle import imgui, implot
from imgui_bundle.python_backends.glfw_backend import GlfwRenderer

from RTSGS.GUI.ViewportWindow import ViewportWindow
from RTSGS.GUI.PerformanceWindow import PerformanceWindow
from RTSGS.GUI.ProfilerWindow import ProfilerWindow
from RTSGS.GUI.GaussianSplattingWindow import GaussianSplattingWindow
from RTSGS.GaussianSplatting.PointCloud import PointCloud
from RTSGS.GaussianSplatting.GaussianSplating import GaussianSplatting
from RTSGS.GaussianSplatting.Renderer.OpenGLRenderer import Renderer
from RTSGS.GaussianSplatting.Renderer.Camera import Camera
from RTSGS.GUI import context
from RTSGS.GUI.theme import apply_theme_v3
class WindowManager:
    def __init__(self,point_cloud:PointCloud,gs:GaussianSplatting, width=1280, height=720, title="Modular Docking"):
        self.width = width
        self.height = height
        self.title = title

        # --- Initialize GLFW ---
        if not glfw.init():
            raise RuntimeError("GLFW could not initialize")

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.RESIZABLE, glfw.TRUE)

        # Fullscreen window on primary monitor (as you had)
        monitor = glfw.get_primary_monitor()
        mode = glfw.get_video_mode(monitor)
        self.window = glfw.create_window(
            mode.size.width,
            mode.size.height,
            self.title,
            None,
            None,
        )
        if not self.window:
            glfw.terminate()
            raise RuntimeError("GLFW window could not be created")
        context.set_window(self.window)

        glfw.make_context_current(self.window)

        # Disable vsync
        glfw.swap_interval(0)

        # Track resize events
        glfw.set_window_size_callback(self.window, self._on_window_resize)

        # --- Initialize ImGui/ImPlot ---
        imgui.create_context()
        implot.create_context()

        io = imgui.get_io()
        io.config_flags |= imgui.ConfigFlags_.docking_enable
        apply_theme_v3(hue07='C', alt07='C', nav07='O', lit01=0, compact01=0, border01=1, shape0123=1)

        self.renderer = GlfwRenderer(self.window)

        # windows
        self.performance_window = PerformanceWindow()
        self.profiler_window = ProfilerWindow()
        camera = Camera()
        self.opengl_renderer = Renderer(point_cloud,camera)
        self.viewport_window = ViewportWindow(self.opengl_renderer)
        self.gaussian_splating_window = GaussianSplattingWindow(point_cloud,camera)
        #time
        self._last_time = None
        self._delta_time = 0.016

    def _on_window_resize(self, window, width, height):
        self.width = max(1, int(width))
        self.height = max(1, int(height))

    def draw_toolbar_mainmenubar(self):
        if imgui.begin_main_menu_bar():
            if imgui.begin_menu("Window", True):
                clicked, _ = imgui.menu_item("Performance", "", self.performance_window.is_open, True)
                if clicked:
                    self.performance_window.is_open = not self.performance_window.is_open

                clicked, _ = imgui.menu_item("Profiler", "", self.profiler_window.is_open, True)
                if clicked:
                    self.profiler_window.is_open = not self.profiler_window.is_open

                clicked, _ = imgui.menu_item("Viewport", "", self.viewport_window.is_open, True)
                if clicked:
                    self.viewport_window.is_open = not self.viewport_window.is_open
                imgui.end_menu()

            imgui.end_main_menu_bar()

    def start_frame(self):
        glfw.poll_events()
        self.renderer.process_inputs()
        imgui.new_frame()

        #  Main menu bar FIRST (so we know its height and we don't cover it)
        self.draw_toolbar_mainmenubar()

        # Dockspace window below the main menu bar
        fb_w, fb_h = glfw.get_framebuffer_size(self.window)
        fb_w = max(1, fb_w)
        fb_h = max(1, fb_h)

        menu_h = imgui.get_frame_height()  # height of main menu bar

        dockspace_flags = (
            imgui.WindowFlags_.no_title_bar
            | imgui.WindowFlags_.no_collapse
            | imgui.WindowFlags_.no_resize
            | imgui.WindowFlags_.no_move
            | imgui.WindowFlags_.no_bring_to_front_on_focus
            | imgui.WindowFlags_.no_nav_focus
        )

        imgui.set_next_window_pos((0.0, float(menu_h)), cond=imgui.Cond_.always)
        imgui.set_next_window_size((float(fb_w), float(fb_h) - float(menu_h)), cond=imgui.Cond_.always)

        # Note: keep this window alive every frame; it hosts the dockspace
        imgui.begin("MainDockSpace", True, dockspace_flags)
        dockspace_id = imgui.get_id("MainDockSpace")
        imgui.dock_space(dockspace_id)
        imgui.end()

        # Other windows
        self.performance_window.update()
        if self.performance_window.is_open:
            self.performance_window.draw()
            
        self.profiler_window.begin() 
        self.profiler_window.render()

        if self.viewport_window.is_open:
            self.viewport_window.draw(self._delta_time)
        
        if self.gaussian_splating_window.is_open:
            self.gaussian_splating_window.draw(self._delta_time)

    def render_frame(self):
        self.update_delta_time()
        imgui.render()
        self.renderer.render(imgui.get_draw_data())
        glfw.swap_buffers(self.window)
        self.profiler_window.end_collect()
    def update_delta_time(self):
        if self._last_time is None:
            self._last_time = glfw.get_time()
            return
        
        current_time = glfw.get_time()
        self._delta_time = current_time - self._last_time
        self._last_time = current_time

    def shutdown(self):
        self.renderer.shutdown()
        glfw.terminate()

    def window_should_close(self):
        return glfw.window_should_close(self.window)