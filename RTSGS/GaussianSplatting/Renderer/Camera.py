import numpy as np
from dataclasses import dataclass, field
from numpy.typing import NDArray
from imgui_bundle import imgui
import glfw
Vec3 = NDArray[np.float32]
Mat4 = NDArray[np.float32]

@dataclass
class Camera():
    position: Vec3 = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    euler_angles: Vec3 = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    projection: Mat4 = field(default_factory=lambda: np.eye(4, dtype=np.float32))
    view: Mat4 = field(default_factory=lambda: np.eye(4, dtype=np.float32))
    
    fov: float = 60.0
    aspect_ratio: float = 1.0
    width: int = 800
    height: int = 600
    
    move_speed: float = 5.0
    sprint_speed: float = 10.0
    scroll_speed: float = 0.2
    mouse_sensitivity: float = 0.1
    
    last_mouse_x: float = 0.0
    last_mouse_y: float = 0.0
    first_mouse: bool = True
    is_looking: bool = False
        
    @staticmethod
    def perspective(fov_y, aspect, near, far):
        f = 1.0 / np.tan(fov_y / 2.0)
        proj = np.zeros((4, 4), dtype=np.float32, order='F') 
        proj[0, 0] = f / aspect
        proj[1, 1] = f
        proj[2, 2] = (far + near) / (near - far)
        proj[3, 2] = (2 * far * near) / (near - far)
        proj[2, 3] = -1.0
        return proj

    
    def get_forward(self):
        pitch, yaw, _ = self.euler_angles
        forward = np.array([
            np.sin(yaw) * np.cos(pitch),
            -np.sin(pitch),
            -np.cos(yaw) * np.cos(pitch)
        ], dtype=np.float32)
        return forward
    
    def get_right(self):
        forward = self.get_forward()
        world_up = np.array([0, 1, 0], dtype=np.float32)
        right = np.cross(forward, world_up)
        right = right / (np.linalg.norm(right) + 1e-8)
        return right
    
    def get_up(self):
        return np.array([0, 1, 0], dtype=np.float32)
    
    def process_mouse(self, mouse_x, mouse_y):
        if self.first_mouse:
            self.last_mouse_x = mouse_x
            self.last_mouse_y = mouse_y
            self.first_mouse = False
            return

        x_offset = mouse_x - self.last_mouse_x
        y_offset = mouse_y - self.last_mouse_y

        self.last_mouse_x = mouse_x
        self.last_mouse_y = mouse_y

        x_offset *= self.mouse_sensitivity
        y_offset *= self.mouse_sensitivity

        pitch, yaw, roll = self.euler_angles
        yaw += np.radians(x_offset)
        pitch += np.radians(y_offset)

        pitch = np.clip(pitch, -np.pi/2 + 0.01, np.pi/2 - 0.01)

        self.euler_angles = np.array([pitch, yaw, roll], dtype=np.float32)
        
    def process_scroll(self, scroll_y):
        forward = self.get_forward()
        self.position += forward * scroll_y * self.scroll_speed
    
    def process_keyboard(self, keys):
        velocity = self.sprint_speed if keys.get('SHIFT', False) else self.move_speed
        
        forward = self.get_forward()
        right = self.get_right()
        up = self.get_up()
        
        if keys.get('W', False):
            self.position += forward * velocity
        if keys.get('S', False):
            self.position -= forward * velocity
        if keys.get('A', False):
            self.position -= right * velocity
        if keys.get('D', False):
            self.position += right * velocity
        if keys.get('E', False):
            self.position += up * velocity
        if keys.get('Q', False):
            self.position -= up * velocity
    

    @staticmethod
    def look_at(eye, target, world_up):
        """
        OpenGL-style lookAt (column-major, column vectors):
        view = R^T * T^-1
        """
        eye = eye.astype(np.float32)
        target = target.astype(np.float32)
        world_up = world_up.astype(np.float32)

        f = target - eye
        f = f / (np.linalg.norm(f) + 1e-8)          # forward

        s = np.cross(f, world_up)                   # right
        s = s / (np.linalg.norm(s) + 1e-8)

        u = np.cross(s, f)                          # up (already normalized)

        # Column-major matrix for OpenGL
        view = np.eye(4, dtype=np.float32, order='F')

        # put basis vectors in COLUMNS
        view[0, 0] = s[0]; view[1, 0] = s[1]; view[2, 0] = s[2]
        view[0, 1] = u[0]; view[1, 1] = u[1]; view[2, 1] = u[2]
        view[0, 2] = -f[0]; view[1, 2] = -f[1]; view[2, 2] = -f[2]

        # translation is in the last COLUMN
        view[3, 0] = -np.dot(s, eye)
        view[3, 1] = -np.dot(u, eye)
        view[3, 2] =  np.dot(f, eye)

        return view

    def update_view(self):
        forward = self.get_forward()
        target = self.position + forward
        world_up = np.array([0, 1, 0], dtype=np.float32)
        
        self.view = Camera.look_at(self.position, target, world_up)
        return self.view

    def update_projection(self, fb):
        self.projection = Camera.perspective(
            np.radians(self.fov), 
            fb.width / fb.height, 
            0.02, 
            50.0
        )
    def update_resolution(self, width: int, height: int):
        self.width = max(width, 1)
        self.height = max(height, 1)
        self.aspect_ratio = self.width / self.height
        # Update projection matrix automatically when size changes
        self.projection = self.perspective(np.radians(self.fov), self.aspect_ratio, 0.02, 50.0)

    def process_window_input(self, window_hovered: bool, window_focused: bool, delta_time: float = 0.0):
        io = imgui.get_io()
        
        # 1. Right-click to look logic
        if imgui.is_mouse_clicked(1) and window_hovered:
            self.is_looking = True

        if not imgui.is_mouse_down(1):
            if self.is_looking:
                self.is_looking = False
                self.first_mouse = True

        if self.is_looking:
            mx, my = imgui.get_mouse_pos()
            self.process_mouse(mx, my)

        # 2. Scroll logic
        if window_hovered and io.mouse_wheel != 0:
            self.process_scroll(io.mouse_wheel)

        # 3. Keyboard logic (using context.window for raw key states)
        if window_focused or self.is_looking:
            from RTSGS.GUI import context
            win = context.window
            keys = {
                'W': glfw.get_key(win, glfw.KEY_W) == glfw.PRESS,
                'A': glfw.get_key(win, glfw.KEY_A) == glfw.PRESS,
                'S': glfw.get_key(win, glfw.KEY_S) == glfw.PRESS,
                'D': glfw.get_key(win, glfw.KEY_D) == glfw.PRESS,
                'Q': glfw.get_key(win, glfw.KEY_Q) == glfw.PRESS,
                'E': glfw.get_key(win, glfw.KEY_E) == glfw.PRESS,
                'SHIFT': glfw.get_key(win, glfw.KEY_LEFT_SHIFT) == glfw.PRESS,}
            self.process_keyboard(keys)