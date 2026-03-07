import torch
import numpy as np
from OpenGL.GL import *
from OpenGL import GL
from .FrameBuffer import FrameBuffer
import RTSGS.GaussianSplatting.Renderer.Resources as res
from .Camera import Camera
from imgui_bundle import imgui
import ctypes

class Renderer:
    def __init__(self, pcd, camera: Camera):
        # Initialize the resources
        res.init_resources()

        self.fb = FrameBuffer(width=800, height=600)
        self.pcd = pcd
        self.vbo_capacity_bytes = 0
        # Setup OpenGL buffers
        self.vbo = None
        self.vao = None
        self._initialized = False
        
        # camera setup
        self.camera = camera

        # Opengl 
        # Enable depth testing for 3D points
        self.fb.bind()
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LESS)
        glEnable(GL_PROGRAM_POINT_SIZE) 
        res.simple_shader.use()
        self.fb.unbind()

        self.pcd_added_size = 0

        print("GL_VENDOR  :", glGetString(GL_VENDOR).decode())
        print("GL_RENDERER:", glGetString(GL_RENDERER).decode())
        print("GL_VERSION :", glGetString(GL_VERSION).decode())

    def _initialize_pcd_rendering(self):
        # UPDATED: Check for all_sh instead of all_colors
        if self._initialized or self.pcd.all_points is None or self.pcd.all_points.numel() == 0:
            return

        # Create VBO
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)

        # Allocate space for (Position: 3 floats + SH0: 3 floats) * num_points
        self.vbo_capacity_bytes = (
            self.pcd.all_points.shape[0] * 6 * 4
        )
        glBufferData(GL_ARRAY_BUFFER, self.vbo_capacity_bytes, None, GL_DYNAMIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

        # Setup VAO
        self.vao = glGenVertexArrays(1)
        stride = 6 * 4  # 6 floats * 4 bytes
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)

        # position (location = 0)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))

        # color/sh0 (location = 1)
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))

        glBindVertexArray(0)
        self._initialized = True
        # UPDATED: Pass all_sh
        self.update_vbo(self.pcd.all_points, self.pcd.all_sh)

    def update_vbo(self, positions, sh_coeffs):
        # The arguments 'positions' and 'sh_coeffs' passed from render_pcd
        # are already detached/referenced under the lock there.
        if positions is None or positions.numel() == 0:
            return

        # Convert to numpy under the assumption these are consistent snapshots
        # We take the 0th order SH (index 0) and the XYZ positions
        sh0_data = sh_coeffs[:, 0, :].detach().cpu().numpy().astype(np.float32)
        positions_data = positions.detach().cpu().numpy().astype(np.float32)
        
        # This will now succeed because they were grabbed under the lock together
        interleaved = np.hstack([positions_data, sh0_data])
        required_bytes = interleaved.nbytes

        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        if required_bytes > self.vbo_capacity_bytes:
            self.vbo_capacity_bytes = required_bytes
            glBufferData(GL_ARRAY_BUFFER, self.vbo_capacity_bytes, interleaved, GL_DYNAMIC_DRAW)
        else:
            glBufferSubData(GL_ARRAY_BUFFER, 0, required_bytes, interleaved)
        
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        self.pcd_added_size = positions.shape[0]

    def Render(self):
        self.fb.bind()
        glViewport(0, 0, self.fb.width, self.fb.height)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glClearColor(0.08, 0.1, 0.13, 1.0)
        self.render_pcd()
        self.fb.unbind()

    def render_pcd(self):
        # 1. Thread-safe check and data extraction
        with self.pcd.lock:
            # Skip if data isn't ready
            if self.pcd.all_points is None or self.pcd.all_sh is None:
                return

            current_count = self.pcd.all_points.shape[0]
            
            # 2. Check if we need to update the OpenGL buffers
            # Only run the heavy upload if the point count changed
            if not self._initialized:
                self._initialize_pcd_rendering()
            elif self.pcd_added_size != current_count:
                # We pass the tensors directly while inside the lock
                self.update_vbo(self.pcd.all_points, self.pcd.all_sh)

        # 3. Standard OpenGL Drawing (Outside the lock to keep it fast)
        res.simple_shader.use()
        self.camera.update_view()
        
        glUniformMatrix4fv(
            glGetUniformLocation(res.simple_shader.program, 'u_view'),
            1, GL_FALSE, self.camera.view           
        )
        glUniformMatrix4fv(
            glGetUniformLocation(res.simple_shader.program, 'u_projection'),
            1, GL_FALSE, self.camera.projection           
        )

        glBindVertexArray(self.vao)
        glDrawArrays(GL_POINTS, 0, self.pcd_added_size)
        glBindVertexArray(0)

    def cleanup(self):
        """Release resources"""
        if self._initialized:
            glDeleteBuffers(1, [self.vbo])
            glDeleteVertexArrays(1, [self.vao])

    def on_resize(self):
        self.camera.update_projection(self.fb)