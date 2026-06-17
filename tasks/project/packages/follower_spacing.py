"""Circle-grid follower spacing: span_px + range-rate + leader speed estimate."""

from typing import Any, Dict, Optional


class GridSpacingController(object):
    """Maintain follow gap using raw grid span_px (higher span = leader closer)."""

    __slots__ = ("_last_span", "_last_ts", "_span_dot", "_v_leader")

    def __init__(self):
        self.reset()

    def reset(self):
        self._last_span = None  # type: Optional[float]
        self._last_ts = None  # type: Optional[float]
        self._span_dot = 0.0
        self._v_leader = 0.0

    @property
    def span_px(self):
        return self._last_span

    @property
    def span_dot(self):
        return self._span_dot

    @property
    def v_leader_est(self):
        return self._v_leader

    def observe(self, span_px, ts, cfg, current_cmd):
        """Ingest a grid span sample (call at grid_detect_hz when leader found)."""
        span = float(span_px)
        alpha = float(cfg.get("spacing_span_dot_alpha", 0.35))
        ff = float(cfg.get("spacing_leader_ff_gain", 0.012))

        if self._last_span is not None and self._last_ts is not None:
            dt = max(1e-3, float(ts) - self._last_ts)
            raw_dot = (span - self._last_span) / dt
            self._span_dot = (1.0 - alpha) * self._span_dot + alpha * raw_dot

        self._last_span = span
        self._last_ts = float(ts)

        span_target = float(cfg.get("span_target_px", 32.0))
        steady_span = float(cfg.get("spacing_steady_span_px", 5.0))
        steady_dot = float(cfg.get("spacing_steady_span_dot", 10.0))
        leader_alpha = float(cfg.get("spacing_leader_alpha", 0.12))
        error = span_target - span

        v_obs = max(0.0, float(current_cmd) - ff * self._span_dot)
        if abs(error) < steady_span and abs(self._span_dot) < steady_dot:
            self._v_leader = (1.0 - leader_alpha) * self._v_leader + leader_alpha * v_obs
        else:
            self._v_leader = (1.0 - leader_alpha * 0.5) * self._v_leader + leader_alpha * 0.5 * v_obs

    def compute_target_speed(self, cfg, min_speed, max_speed):
        """Same spacing law in cruise, turn, and recovery."""
        if self._last_span is None:
            return max(min_speed, min(max_speed, float(cfg.get("slow_speed", 0.15))))

        span_target = float(cfg.get("span_target_px", 32.0))
        kp = float(cfg.get("spacing_kp", 0.010))
        kd = float(cfg.get("spacing_kd", 0.018))
        error = span_target - self._last_span
        v = self._v_leader + kp * error - kd * self._span_dot
        v = max(min_speed, min(max_speed, v))

        too_close = float(cfg.get("span_too_close_px", 44.0))
        if self._last_span >= too_close:
            v = min(v, float(cfg.get("span_too_close_speed", 0.06)))

        return max(min_speed, min(max_speed, v))
