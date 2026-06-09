"""Single-thread camera capture for the project task (hardware only).

Never raises during startup — the Flask server must bind :5000 even if the
camera is not ready yet (dashboard may still be releasing nvargus).
"""

from __future__ import annotations

import queue
import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np

try:
    from duckiebot.camera_driver import CameraDriver
except ImportError:
    CameraDriver = None


class SharedCaptureCamera:
    """One background reader feeds the agent queue and the MJPEG preview."""

    def __init__(
        self,
        source,
        frame_queue: queue.Queue,
        stop_event: threading.Event,
    ):
        self._source = source
        self._queue = frame_queue
        self._stop_event = stop_event
        self._latest: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._ready = threading.Event()

    def start(self):
        self._thread = threading.Thread(target=self._capture_loop, daemon=True, name='CameraCapture')
        self._thread.start()

    def stop(self):
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        with self._lock:
            self._latest = None
        stop_fn = getattr(self._source, 'stop', None)
        if callable(stop_fn):
            try:
                stop_fn()
            except Exception:
                pass

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        with self._lock:
            if self._latest is None:
                return False, None
            return True, self._latest.copy()

    def _capture_loop(self):
        while not self._stop_event.is_set():
            ok, frame = self._source.read()
            if not ok or frame is None:
                time.sleep(0.01)
                continue

            self._ready.set()
            with self._lock:
                self._latest = frame

            try:
                self._queue.put_nowait(frame.copy())
            except queue.Full:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._queue.put_nowait(frame.copy())
                except queue.Full:
                    pass


class WaitingCaptureCamera:
    """Placeholder until hardware camera opens; retries in the background."""

    def __init__(self, frame_queue: queue.Queue, stop_event: threading.Event):
        self._queue = frame_queue
        self._stop_event = stop_event
        self._shared: Optional[SharedCaptureCamera] = None
        self._thread = threading.Thread(target=self._retry_loop, daemon=True, name='CameraRetry')
        self._thread.start()

    def stop(self):
        if self._shared is not None:
            self._shared.stop()
        self._thread.join(timeout=2.0)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self._shared is not None:
            return self._shared.read()
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(blank, "Camera starting...", (150, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 80), 2)
        return True, blank

    def _retry_loop(self):
        while not self._stop_event.is_set() and self._shared is None:
            if CameraDriver is None:
                time.sleep(2.0)
                continue
            try:
                hw = CameraDriver()
                hw.start()
                shared = SharedCaptureCamera(hw, self._queue, self._stop_event)
                shared.start()
                self._shared = shared
                print('[Project] Camera: hardware ready (retry succeeded)')
                return
            except Exception as e:
                print(f'[Project] Camera retry failed: {e}')
                time.sleep(2.0)


def open_project_camera(
    frame_queue: queue.Queue,
    stop_event: threading.Event,
) -> Tuple[object, str]:
    """Never raises. Returns (camera_handle, mode_label)."""
    if CameraDriver is None:
        print('[Project] Camera: driver not available on this platform')
        return WaitingCaptureCamera(frame_queue, stop_event), 'waiting'

    try:
        hw = CameraDriver()
        hw.start()
        shared = SharedCaptureCamera(hw, frame_queue, stop_event)
        shared.start()
        print('[Project] Camera: hardware')
        return shared, 'hardware'
    except Exception as e:
        print(f'[Project] Camera: hardware not ready ({e}); will retry in background')
        return WaitingCaptureCamera(frame_queue, stop_event), 'waiting'
