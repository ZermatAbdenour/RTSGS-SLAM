from __future__ import annotations

import os
import time
import tracemalloc
from dataclasses import dataclass

from imgui_bundle import imgui

try:
    import psutil  # optional, recommended for RAM/process info
except Exception:
    psutil = None


@dataclass
class PerformanceStats:
    fps: float = 0.0
    frame_ms: float = 0.0
    python_current_mb: float = 0.0
    python_peak_mb: float = 0.0
    rss_mb: float = 0.0  # process resident set size (if psutil available)
    vms_mb: float = 0.0  # process virtual memory size (if psutil available)
    cpu_percent: float = 0.0  # process cpu% (if psutil available)


class PerformanceWindow:
    """
    ImGui window that shows FPS, frame time, and memory usage.

    - FPS/frame time: computed from wall clock (smoothed)
    - Python memory: tracemalloc (current/peak)
    - Process memory / CPU: psutil (optional)
    """

    def __init__(self, title: str = "Performance", sample_hz: float = 4.0, fps_smooth: float = 0.10):
        self.title = title
        self.is_open = False

        self._sample_period = 1.0 / max(sample_hz, 0.1)
        self._fps_smooth = min(max(fps_smooth, 0.0), 1.0)

        self._last_frame_t = time.perf_counter()
        self._last_sample_t = 0.0

        self._ema_dt = 1.0 / 60.0
        self.stats = PerformanceStats()

        self._proc = psutil.Process(os.getpid()) if psutil is not None else None
        self._psutil_cpu_primed = False

        # Start tracemalloc once (safe ato call multiple times, but we guard anyway)
        if not tracemalloc.is_tracing():
            tracemalloc.start()

    def update(self) -> None:
        """Call once per frame before draw()."""
        now = time.perf_counter()
        dt = max(now - self._last_frame_t, 1e-9)
        self._last_frame_t = now

        # Exponential moving average for stable FPS
        a = self._fps_smooth
        self._ema_dt = (1.0 - a) * self._ema_dt + a * dt

        self.stats.frame_ms = self._ema_dt * 1000.0
        self.stats.fps = 1.0 / self._ema_dt if self._ema_dt > 0 else 0.0

        # Sample heavier metrics at lower rate
        if now - self._last_sample_t >= self._sample_period:
            self._last_sample_t = now

            cur, peak = tracemalloc.get_traced_memory()
            self.stats.python_current_mb = cur / (1024.0 * 1024.0)
            self.stats.python_peak_mb = peak / (1024.0 * 1024.0)

            if self._proc is not None:
                mi = self._proc.memory_info()
                self.stats.rss_mb = mi.rss / (1024.0 * 1024.0)
                self.stats.vms_mb = mi.vms / (1024.0 * 1024.0)

                # cpu_percent() needs one warmup call to avoid always-0.0
                if not self._psutil_cpu_primed:
                    self._proc.cpu_percent(interval=None)
                    self._psutil_cpu_primed = True
                else:
                    self.stats.cpu_percent = self._proc.cpu_percent(interval=None)

    def draw(self) -> None:
        """Call every frame after update()."""
        if not self.is_open:
            return

        # open state is returned as second value in many imgui-bundle bindings
        opened, self.is_open = imgui.begin(self.title, self.is_open, flags=imgui.WindowFlags_.always_auto_resize)

        if not opened:
            imgui.end()
            return

        imgui.text(f"FPS: {self.stats.fps:6.1f}")
        imgui.text(f"Frame time: {self.stats.frame_ms:6.2f} ms")
        # Gaussian Splatting optimization steps/sec (if available)
        gs = getattr(self, 'gs', None)
        if gs is not None:
            try:
                sps = float(getattr(gs, 'steps_per_sec', 0.0))
                imgui.text(f"GS steps/s: {sps:6.1f}")
            except Exception:
                pass

        imgui.separator()
        imgui.text("Python memory (tracemalloc)")
        imgui.text(f"  Current: {self.stats.python_current_mb:8.2f} MB")
        imgui.text(f"  Peak:    {self.stats.python_peak_mb:8.2f} MB")

        imgui.separator()
        if psutil is None:
            imgui.text("Process metrics: psutil not installed")
            imgui.text("  Install: pip install psutil")
        else:
            imgui.text("Process metrics (psutil)")
            imgui.text(f"  RSS: {self.stats.rss_mb:8.2f} MB")
            imgui.text(f"  VMS: {self.stats.vms_mb:8.2f} MB")
            imgui.text(f"  CPU: {self.stats.cpu_percent:6.1f} %")

        imgui.end()