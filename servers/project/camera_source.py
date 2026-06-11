"""Single-thread camera capture for the project task (hardware only).

Jetson nvargus allows one CaptureSession. This module keeps a single live
session and never opens a second one while a stale session is still alive.
"""

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

_camera_open_lock = threading.Lock()
_live_shared: Optional["SharedCaptureCamera"] = None
_OPEN_RETRY_DELAY_S = 2.0
_BACKGROUND_RETRY_BASE_S = 8.0
_BACKGROUND_RETRY_MAX_S = 45.0
_BACKGROUND_GIVE_UP_AFTER = 6
_BACKGROUND_PAUSE_S = 90.0
_STALE_WARN_S = 6.0
_HEALTH_FRAME_GAP_S = 8.0
_HEALTH_BOOT_GRACE_S = 12.0


def _blank_frame(message: str = "Camera starting...") -> np.ndarray:
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(
        blank,
        message,
        (120, 240),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (80, 80, 80),
        2,
    )
    return blank


def _release_hw_camera(hw) -> None:
    if hw is None:
        return
    try:
        if getattr(hw, "is_active", False):
            hw.stop()
    except Exception:
        pass


def _pop_live_shared() -> Optional["SharedCaptureCamera"]:
    global _live_shared
    live = _live_shared
    _live_shared = None
    return live


def _stop_dead_live_session() -> None:
    """Ensure no zombie capture thread holds nvargus before a new open."""
    with _camera_open_lock:
        live = _live_shared
        if live is None or live.is_healthy():
            return
        _pop_live_shared()
    print(
        "[Project] Camera: stopping stale session before retry "
        "(nvargus allows one CaptureSession)",
        flush=True,
    )
    live.shutdown()


def _get_live_shared() -> Optional["SharedCaptureCamera"]:
    with _camera_open_lock:
        live = _live_shared
    if live is not None and live.is_healthy():
        return live
    return None


def stop_project_camera() -> None:
    """Release nvargus session on task shutdown."""
    with _camera_open_lock:
        live = _pop_live_shared()
    if live is not None:
        live.shutdown()


def _register_live(shared: Optional["SharedCaptureCamera"]) -> None:
    global _live_shared
    old = None
    with _camera_open_lock:
        if shared is None:
            old = _pop_live_shared()
        elif _live_shared is not None and _live_shared is not shared:
            old = _pop_live_shared()
            _live_shared = shared
        else:
            _live_shared = shared
    if old is not None:
        old.shutdown()


class PlaceholderCamera:
    """MJPEG placeholder only — never touches nvargus."""

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        return True, _blank_frame()

    def stop(self) -> None:
        return None


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
        self._last_frame_ts = 0.0
        self._stale_warned = False
        self._stopped = False

    def start(self):
        self._thread = threading.Thread(
            target=self._capture_loop, daemon=True, name="CameraCapture"
        )
        self._thread.start()

    def shutdown(self):
        if self._stopped:
            return
        self._stopped = True
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        with self._lock:
            self._latest = None
        _release_hw_camera(self._source)

    def stop(self):
        self.shutdown()
        with _camera_open_lock:
            global _live_shared
            if _live_shared is self:
                _live_shared = None

    def is_healthy(self) -> bool:
        if self._stopped:
            return False
        if self._last_frame_ts <= 0.0:
            return (time.time() - getattr(self, "_started_ts", time.time())) < _HEALTH_BOOT_GRACE_S
        return (time.time() - self._last_frame_ts) < _HEALTH_FRAME_GAP_S

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        with self._lock:
            if self._latest is None:
                return False, None
            return True, self._latest.copy()

    def _capture_loop(self):
        self._started_ts = time.time()
        stale_since: Optional[float] = None
        while not self._stop_event.is_set() and not self._stopped:
            ok, frame = self._source.read()
            if ok and frame is not None:
                stale_since = None
                self._stale_warned = False
                self._last_frame_ts = time.time()
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
                continue

            if stale_since is None:
                stale_since = time.time()
            elif (
                not self._stale_warned
                and time.time() - stale_since >= _STALE_WARN_S
            ):
                self._stale_warned = True
                print(
                    "[Project] Camera: no frames — keeping session open "
                    "(restart task if video stays black; do not auto-reopen nvargus)",
                    flush=True,
                )
            time.sleep(0.02)


def _try_open_shared_camera(
    frame_queue: queue.Queue,
    stop_event: threading.Event,
) -> SharedCaptureCamera:
    """Open exactly one CameraDriver, or return the existing healthy session."""
    if CameraDriver is None:
        raise RuntimeError("CameraDriver not available on this platform")

    existing = _get_live_shared()
    if existing is not None:
        return existing

    _stop_dead_live_session()

    with _camera_open_lock:
        existing = _live_shared
        if existing is not None and existing.is_healthy():
            return existing

    hw = CameraDriver()
    try:
        hw.start()
        shared = SharedCaptureCamera(hw, frame_queue, stop_event)
        shared.start()
        _register_live(shared)
        return shared
    except Exception:
        _release_hw_camera(hw)
        raise


class WaitingCaptureCamera:
    """Placeholder + background retry only when init open failed."""

    def __init__(self, frame_queue: queue.Queue, stop_event: threading.Event):
        self._queue = frame_queue
        self._stop_event = stop_event
        self._shared: Optional[SharedCaptureCamera] = None
        self._stop_retry = threading.Event()
        self._thread = threading.Thread(
            target=self._retry_loop, daemon=True, name="CameraRetry"
        )
        self._thread.start()

    def stop(self):
        self._stop_retry.set()
        if self._shared is not None:
            self._shared.stop()
            self._shared = None
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        live = self._shared or _get_live_shared()
        if live is not None:
            ok, frame = live.read()
            if ok and frame is not None:
                return ok, frame
        return True, _blank_frame("Camera starting...")

    def _retry_loop(self):
        failures = 0
        delay = _BACKGROUND_RETRY_BASE_S
        while not self._stop_event.is_set() and not self._stop_retry.is_set():
            live = _get_live_shared()
            if live is not None:
                self._shared = live
                print("[Project] Camera: using existing live session", flush=True)
                return

            _stop_dead_live_session()

            try:
                self._shared = _try_open_shared_camera(self._queue, self._stop_event)
                print("[Project] Camera: hardware ready (background retry succeeded)", flush=True)
                return
            except Exception as e:
                failures += 1
                print(f"[Project] Camera retry failed ({failures}): {e}", flush=True)
                if failures >= _BACKGROUND_GIVE_UP_AFTER:
                    print(
                        f"[Project] Camera: pausing retries for {_BACKGROUND_PAUSE_S:.0f}s "
                        "(nvargus busy — stop task, wait 10s, redeploy; reboot if needed)",
                        flush=True,
                    )
                    time.sleep(_BACKGROUND_PAUSE_S)
                    failures = 0
                    delay = _BACKGROUND_RETRY_BASE_S
                    continue
                time.sleep(delay)
                delay = min(_BACKGROUND_RETRY_MAX_S, delay * 1.5)


def open_project_camera(
    frame_queue: queue.Queue,
    stop_event: threading.Event,
) -> Tuple[object, str]:
    """Never raises. Returns (camera_handle, mode_label)."""
    live = _get_live_shared()
    if live is not None:
        print("[Project] Camera: reusing live session", flush=True)
        return live, "hardware"

    if CameraDriver is None:
        print("[Project] Camera: driver not available on this platform")
        return WaitingCaptureCamera(frame_queue, stop_event), "waiting"

    last_err = None
    for attempt in range(2):
        _stop_dead_live_session()
        try:
            shared = _try_open_shared_camera(frame_queue, stop_event)
            print("[Project] Camera: hardware ok (shared capture)", flush=True)
            return shared, "hardware"
        except Exception as e:
            last_err = e
            print(
                f"[Project] Camera open attempt {attempt + 1}/2 failed ({e})",
                flush=True,
            )
            if attempt < 1:
                time.sleep(_OPEN_RETRY_DELAY_S)

    print(
        f"[Project] Camera: hardware not ready ({last_err}); "
        "background retry will continue with backoff",
        flush=True,
    )
    return WaitingCaptureCamera(frame_queue, stop_event), "waiting"
