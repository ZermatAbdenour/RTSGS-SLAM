import numpy as np
import ctypes
from OpenGL import GL
from imgui_bundle import imgui


class ImageWidget:
    def __init__(self, rgb: np.ndarray | None = None, pbo_count: int = 3):
        self._tex_id: int | None = None
        self._tex_ref: imgui.ImTextureRef | None = None

        self._pbo_count = int(pbo_count)
        self._pbo_ids: list[int] = []
        self._pbo_index = 0

        self._img_w = 0
        self._img_h = 0
        self._byte_size = 0

        if rgb is not None:
            self.set_image_rgb(rgb)

    def set_image_rgb(self, rgb: np.ndarray):
        if rgb is None or rgb.size == 0:
            return

        # Make it exactly HxWx3 uint8 contiguous (avoid hidden copies later)
        if rgb.dtype != np.uint8:
            rgb = rgb.astype(np.uint8, copy=False)
        if not rgb.flags["C_CONTIGUOUS"]:
            rgb = np.ascontiguousarray(rgb)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"Expected HxWx3 uint8 image, got shape={rgb.shape}, dtype={rgb.dtype}")

        h, w = rgb.shape[:2]

        if self._tex_id is None:
            self._create_texture(w, h)
            self._create_pbos(w, h)
        elif w != self._img_w or h != self._img_h:
            self._resize_texture(w, h)
            self._delete_pbos()
            self._create_pbos(w, h)

        self._img_w, self._img_h = w, h
        self._byte_size = w * h * 3

        self._upload_frame_pbo_orphan(rgb)

    def _create_texture(self, w: int, h: int):
        self._tex_id = int(GL.glGenTextures(1))
        self._tex_ref = imgui.ImTextureRef(self._tex_id)

        GL.glBindTexture(GL.GL_TEXTURE_2D, self._tex_id)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)

        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGB, w, h, 0, GL.GL_BGR, GL.GL_UNSIGNED_BYTE, None)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

    def _resize_texture(self, w: int, h: int):
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._tex_id)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGB, w, h, 0, GL.GL_BGR, GL.GL_UNSIGNED_BYTE, None)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

    def _create_pbos(self, w: int, h: int):
        self._pbo_ids = list(GL.glGenBuffers(self._pbo_count))
        byte_size = w * h * 3
        for pbo in self._pbo_ids:
            GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, pbo)
            GL.glBufferData(GL.GL_PIXEL_UNPACK_BUFFER, byte_size, None, GL.GL_STREAM_DRAW)
        GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, 0)

    def _delete_pbos(self):
        if self._pbo_ids:
            GL.glDeleteBuffers(len(self._pbo_ids), self._pbo_ids)
            self._pbo_ids = []

    def _upload_frame_pbo_orphan(self, rgb: np.ndarray):
        pbo = self._pbo_ids[self._pbo_index]
        self._pbo_index = (self._pbo_index + 1) % self._pbo_count

        GL.glBindTexture(GL.GL_TEXTURE_2D, self._tex_id)
        GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, pbo)

        # Orphan the buffer: gives driver fresh storage, avoids waiting on GPU usage
        GL.glBufferData(GL.GL_PIXEL_UNPACK_BUFFER, self._byte_size, None, GL.GL_STREAM_DRAW)

        # Map (simple write mapping; often sufficient after orphaning)
        flags = GL.GL_MAP_WRITE_BIT | GL.GL_MAP_INVALIDATE_BUFFER_BIT
        ptr = GL.glMapBufferRange(GL.GL_PIXEL_UNPACK_BUFFER, 0, self._byte_size, flags)
        if ptr:
            ctypes.memmove(ptr, rgb.ctypes.data, self._byte_size)
            GL.glUnmapBuffer(GL.GL_PIXEL_UNPACK_BUFFER)

        GL.glTexSubImage2D(
            GL.GL_TEXTURE_2D, 0, 0, 0, self._img_w, self._img_h,
            GL.GL_BGR, GL.GL_UNSIGNED_BYTE, None
        )

        GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, 0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

    def draw(self, fit_to_window: bool = True):
        if self._tex_ref is None or self._img_w <= 0 or self._img_h <= 0:
            imgui.text("No image loaded.")
            return

        if fit_to_window:
            avail_w, avail_h = imgui.get_content_region_avail()
            scale = min(avail_w / self._img_w, avail_h / self._img_h) if avail_w > 0 and avail_h > 0 else 1.0
            disp_w = max(1.0, self._img_w * scale)
            disp_h = max(1.0, self._img_h * scale)
        else:
            disp_w = float(self._img_w)
            disp_h = float(self._img_h)

        imgui.image(self._tex_ref, (disp_w, disp_h))

    def destroy(self):
        if self._tex_id is not None:
            GL.glDeleteTextures([self._tex_id])
            self._tex_id = None
            self._tex_ref = None
        self._delete_pbos()