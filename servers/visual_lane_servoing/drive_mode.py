"""Manual drive mode for lane servoing UI — ramps like project convoy leader."""

import threading
import time
from typing import Any, Dict, Tuple

MODE_CRUISING = "CRUISING"
MODE_SLOW = "SLOW"
MODE_STOPPED = "STOPPED"
_VALID_MODES = {MODE_CRUISING, MODE_SLOW, MODE_STOPPED}


def _ramp_toward(current: float, target: float, max_delta: float) -> float:
    if max_delta <= 0.0:
        return float(target)
    diff = float(target) - float(current)
    if abs(diff) <= max_delta:
        return float(target)
    return float(current) + (max_delta if diff > 0.0 else -max_delta)


class DriveModeController(object):
    def __init__(
        self,
        cruise_speed: float = 0.32,
        slow_speed: float = 0.16,
        speed_ramp_s: float = 1.25,
        decel_time_s: float = 1.5,
    ):
        self._lock = threading.Lock()
        self._mode = MODE_CRUISING
        self.cruise_speed = float(cruise_speed)
        self.slow_speed = float(slow_speed)
        self.speed_ramp_s = float(speed_ramp_s)
        self.decel_time_s = float(decel_time_s)
        self.current_speed_cap = float(cruise_speed)
        self._last_ts = time.time()

    def get(self) -> str:
        with self._lock:
            return self._mode

    def set(self, mode: str) -> str:
        mode = str(mode).strip().upper()
        if mode not in _VALID_MODES:
            raise ValueError(f"mode must be one of {sorted(_VALID_MODES)}")
        with self._lock:
            self._mode = mode
            return self._mode

    def reset(self) -> None:
        with self._lock:
            self._mode = MODE_CRUISING
            self.current_speed_cap = 0.0

    def _target_speed(self) -> float:
        if self._mode == MODE_STOPPED:
            return 0.0
        if self._mode == MODE_SLOW:
            return self.slow_speed
        return self.cruise_speed

    def _ramp_delta(self, target: float, frame_dt: float) -> float:
        dt = max(1e-3, float(frame_dt))
        current = self.current_speed_cap
        if target <= 0.0 and current > 0.0:
            decel_s = max(0.05, self.decel_time_s)
            return current / decel_s * dt
        ramp_s = max(0.05, self.speed_ramp_s)
        span = max(0.05, abs(self.cruise_speed - self.slow_speed))
        if target >= self.cruise_speed - 1e-6:
            span = max(span, abs(self.cruise_speed - current))
        elif target <= self.slow_speed + 1e-6:
            span = max(span, abs(current - self.slow_speed))
        return span / ramp_s * dt

    def apply_pwm(self, left: float, right: float, *, running: bool) -> Tuple[float, float]:
        now = time.time()
        frame_dt = max(1e-3, now - self._last_ts)
        self._last_ts = now

        if not running:
            with self._lock:
                self.current_speed_cap = 0.0
            return 0.0, 0.0

        with self._lock:
            target = self._target_speed()
            max_delta = self._ramp_delta(target, frame_dt)
            self.current_speed_cap = _ramp_toward(
                self.current_speed_cap, target, max_delta,
            )
            cap = max(0.0, self.current_speed_cap)

        if cap <= 1e-4:
            return 0.0, 0.0

        left_f = float(left)
        right_f = float(right)
        peak = max(1e-6, max(abs(left_f), abs(right_f)))
        scale = min(1.0, cap / peak)
        return left_f * scale, right_f * scale

    def status_dict(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "drive_mode": self._mode,
                "speed_cap": round(self.current_speed_cap, 3),
                "target_speed": self._target_speed(),
            }
