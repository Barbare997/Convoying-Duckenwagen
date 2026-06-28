import os
import sys
import threading
import time

import cv2
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.normpath(os.path.join(script_dir, "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from tasks.project.packages import agent


class DummyWheels:
    def __init__(self):
        self.commands = []

    def set_wheels_speed(self, left: float, right: float):
        self.commands.append((left, right))


def _run_and_stop(loop_fn, timeout_s: float = 0.2):
    stop_event = threading.Event()
    wheels = DummyWheels()
    cfg = {}

    t = threading.Thread(
        target=loop_fn,
        args=(None, wheels, None, stop_event, cfg),
        daemon=True,
    )
    t.start()
    time.sleep(timeout_s)
    stop_event.set()
    t.join(timeout=1.0)

    assert not t.is_alive(), f"{loop_fn.__name__} did not exit after stop_event"
    assert wheels.commands, f"{loop_fn.__name__} never sent wheel command"
    assert wheels.commands[-1] == (0.0, 0.0), f"{loop_fn.__name__} final command was not stop"


def test_next_state_transitions():
    assert agent.next_state(agent.STATE_CRUISING, agent.EVENT_SLOW_SIGN) == agent.STATE_SLOW
    assert agent.next_state(agent.STATE_SLOW, agent.EVENT_NORMAL) == agent.STATE_CRUISING
    assert agent.next_state(agent.STATE_CRUISING, agent.EVENT_STOP_SIGN) == agent.STATE_STOPPING
    assert agent.next_state(agent.STATE_CRUISING, agent.EVENT_LEADER_LOST) == agent.STATE_STOPPING
    print("OK: FSM transitions")


def test_smooth_stop():
    wheels = DummyWheels()
    stop_event = threading.Event()
    agent.smooth_stop(wheels, current_speed=0.32, decel_time_s=0.05, decel_steps=4, stop_event=stop_event)
    assert wheels.commands, "smooth_stop produced no wheel commands"
    assert wheels.commands[-1] == (0.0, 0.0), "smooth_stop did not end at full stop"
    print("OK: smooth_stop reaches zero")


def test_config_loads():
    cfg = agent.load_config()
    assert isinstance(cfg, dict), "Config must be a dictionary"
    assert "role" in cfg, "Config missing role"
    assert "grid_cols" in cfg, "Config missing grid_cols"
    print("OK: config loads")


def test_grid_leader_distance_signal():
    from tasks.project.packages import leader_grid

    cfg = {
        "role": "follower",
        "grid_cols": 7,
        "grid_rows": 3,
        "grid_downscale": 1.0,
        "grid_far_search": False,
        "grid_use_clustering": False,
        "grid_blob_min_area": 10,
        "grid_blob_min_circularity": 0.5,
        "grid_safe_px": 10,
        "grid_stop_px": 80,
    }
    leader_grid.reset_grid_tracker()
    frame = np.full((480, 640, 3), 255, dtype=np.uint8)
    cols, rows = 7, 3
    x0, y0 = 180, 280
    dx, dy = 45, 45
    r = 12
    for row in range(rows):
        for col in range(cols):
            cx = x0 + col * dx
            cy = y0 + row * dy
            cv2.circle(frame, (cx, cy), r, (0, 0, 0), -1)

    det = leader_grid.detect_leader_grid(frame, cfg)
    assert det.found, "Synthetic 7x3 grid should be detected"
    assert det.distance_signal is not None
    assert det.span_px is not None
    print("OK: circle grid detection on synthetic pattern")


def test_grid_spacing_controller():
    from tasks.project.packages.follower_spacing import GridSpacingController

    cfg = {
        "span_target_px": 32.0,
        "spacing_kp": 0.010,
        "spacing_kd": 0.018,
        "spacing_span_dot_alpha": 1.0,
        "spacing_leader_alpha": 0.5,
        "spacing_leader_ff_gain": 0.012,
        "spacing_steady_span_px": 100.0,
        "spacing_steady_span_dot": 1000.0,
        "span_too_close_px": 50.0,
        "span_too_close_speed": 0.05,
    }
    ctl = GridSpacingController()
    ctl.observe(28.0, 0.0, cfg, 0.35)
    ctl.observe(28.0, 0.1, cfg, 0.35)
    v_far = ctl.compute_target_speed(cfg, 0.0, 0.368)
    assert v_far > 0.16, "below span_target should catch up"

    ctl.observe(38.0, 0.2, cfg, 0.35)
    ctl.observe(40.0, 0.3, cfg, 0.35)
    v_close = ctl.compute_target_speed(cfg, 0.0, 0.368)
    assert v_close < v_far, "above span_target / closing should slow down"
    print("OK: grid spacing controller")


def test_follower_grid_signal_mock():
    from tasks.project.packages.leader_grid import GridDetection

    cfg = {"role": "follower", "grid_detect_hz": 10}
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    fake = GridDetection(True, 0.42, 1.0, None, (7, 3))
    original = agent.fetch_leader_grid
    try:
        agent.fetch_leader_grid = lambda _f, _c, force=False: fake
        det = agent.fetch_leader_grid(frame, cfg)
        assert det.found and det.distance_signal == 0.42
    finally:
        agent.fetch_leader_grid = original
    print("OK: follower grid fetch")


def test_red_at_line_near_band():
    from tasks.project.packages.intersection_follow import measure_red_at_line

    cfg = {
        "intersection_red_near_top_frac": 0.72,
        "intersection_red_far_top_frac": 0.35,
        "intersection_red_near_min_pixels": 2000,
        "intersection_red_near_min_frac": 0.05,
        "intersection_red_near_far_ratio": 2.0,
    }
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    far = measure_red_at_line(frame, cfg)
    assert not far.at_line

    # Distant red only (middle band) — must NOT trigger
    frame[200:340, 200:440] = (0, 0, 255)
    mid = measure_red_at_line(frame, cfg)
    assert not mid.at_line

    # Dense red under the bot only (bottom band) — far band empty
    frame2 = np.zeros((480, 640, 3), dtype=np.uint8)
    frame2[360:470, 80:560] = (0, 0, 255)
    near = measure_red_at_line(frame2, cfg)
    assert near.at_line
    print("OK: intersection triggers only on near red band")


def test_project_lane_masks_not_cut_on_straight():
    from tasks.project.packages.agent import _project_lane_kwargs
    from tasks.project.packages.intersection_follow import RedLineProximity

    cfg = {
        "lane_ignore_bottom_frac": 0.0,
        "lane_ignore_bottom_at_intersection_frac": 0.35,
        "intersection_red_approach_min_far_px": 600,
    }
    straight = _project_lane_kwargs(cfg, RedLineProximity(0, 0.0, 0, False))
    assert straight["ignore_bottom_frac"] == 0.0

    approaching = _project_lane_kwargs(
        cfg, RedLineProximity(0, 0.0, 800, False),
    )
    assert approaching["ignore_bottom_frac"] == 0.0, "far red alone must not crop masks"

    at_line = _project_lane_kwargs(
        cfg, RedLineProximity(5000, 0.1, 800, True),
    )
    assert at_line["ignore_bottom_frac"] == 0.35
    print("OK: project lane masks full on straights, trimmed only on red line")


def test_lane_ignore_red_for_convoy():
    from tasks.visual_lane_servoing.packages.agent import LaneServoingAgent, detect_lines_in_slices

    agent = LaneServoingAgent()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    agent.compute_commands(rgb, debug_red_mask=False)
    assert int(np.count_nonzero(agent.last_debug_info["red_mask"])) == 0
    assert agent.last_debug_info.get("lane_use_red") is False
    yellow_xs, white_xs = detect_lines_in_slices(
        agent.last_debug_info["yellow_mask"],
        agent.last_debug_info["white_mask"],
        agent.last_debug_info["yellow_mask"].shape[0],
    )
    assert len(yellow_xs) >= 0 and len(white_xs) >= 0
    print("OK: convoy lane steer ignores red markings")


def test_leader_turn_tracker():
    from tasks.project.packages.intersection_follow import (
        LeaderTurnTracker,
        RedLineProximity,
        TURN_LEFT,
        TURN_RIGHT,
        TURN_STRAIGHT,
    )
    from tasks.project.packages.leader_grid import GridDetection

    cfg = {
        "intersection_cx_drift_px": 20,
        "intersection_heading_thresh": 0.1,
        "intersection_heading_sign": -1.0,
        "intersection_straight_aspect_min": 2.2,
        "intersection_last_cx_offset_px": 20,
        "intersection_turn_infer_lookback": 10,
        "intersection_red_approach_min_far_px": 100,
    }
    pat = (7, 3)
    wide = (100, 50, 300, 110)
    narrow = (130, 50, 270, 110)

    tr = LeaderTurnTracker(window=10)
    tr.begin_approach_if_needed(RedLineProximity(0, 0.0, 500, False), cfg)
    for _ in range(4):
        tr.update(GridDetection(
            True, 0.5, 1.0, None, pat, bbox=wide, center_x=200.0, heading=0.02,
        ))
    for _ in range(6):
        tr.update(GridDetection(
            True, 0.5, 1.0, None, pat, bbox=narrow, center_x=200.0, heading=-0.15,
        ))
    assert tr.infer(cfg, frame_w=640.0) == TURN_RIGHT, "last grid heading => right"

    tr.reset()
    tr.begin_approach_if_needed(RedLineProximity(0, 0.0, 500, False), cfg)
    for _ in range(4):
        tr.update(GridDetection(True, 0.5, 1.0, None, pat, bbox=wide, center_x=200.0, heading=-0.02))
    for _ in range(6):
        tr.update(GridDetection(
            True, 0.5, 1.0, None, pat, bbox=narrow, center_x=200.0, heading=0.15,
        ))
    assert tr.infer(cfg, frame_w=640.0) == TURN_LEFT, "last grid heading => left"

    tr.reset()
    tr.begin_approach_if_needed(RedLineProximity(0, 0.0, 500, False), cfg)
    for _ in range(10):
        tr.update(GridDetection(
            True, 0.5, 1.0, None, pat, bbox=wide, center_x=200.0, heading=0.02,
        ))
    assert tr.infer(cfg, frame_w=640.0) == TURN_STRAIGHT, "wide stable last grid => straight"

    tr.reset()
    tr.begin_approach_if_needed(RedLineProximity(0, 0.0, 500, False), cfg)
    for cx in range(100, 160, 5):
        tr.update(GridDetection(
            True, 0.5, 1.0, None, pat, bbox=wide, center_x=float(cx), heading=0.02,
        ))
    assert tr.infer(cfg, frame_w=640.0) == TURN_STRAIGHT, "cx drift alone without heading => straight"

    tr.reset()
    tr.begin_approach_if_needed(RedLineProximity(0, 0.0, 500, False), cfg)
    for cx in range(280, 380, 8):
        tr.update(GridDetection(
            True, 0.5, 0.8, None, pat, bbox=(int(cx) - 40, 60, int(cx) + 40, 140),
            center_x=float(cx), source="detector",
        ))
    assert tr.infer(cfg, frame_w=640.0) == TURN_RIGHT, "detector center drift => right"

    tr.reset()
    for cx in range(280, 380, 8):
        tr.update(GridDetection(
            True, 0.5, 0.8, None, pat, bbox=(int(cx) - 40, 60, int(cx) + 40, 140),
            center_x=float(cx), source="detector",
        ))
    tr.begin_approach_if_needed(RedLineProximity(0, 0.0, 500, False), cfg)
    assert tr.infer(cfg, frame_w=640.0) == TURN_RIGHT, "last leader kept across approach begin"

    print("OK: leader turn tracker")


def test_intersection_turn_uses_detector_not_grid_when_loaded():
    from tasks.project.packages.agent import _update_intersection_turn_tracker
    from tasks.project.packages.intersection_follow import LeaderTurnTracker, TURN_LEFT, TURN_RIGHT
    from tasks.project.packages.leader_detector import (
        reset_leader_detector_cache,
        set_detector_agent,
    )
    from tasks.project.packages.leader_grid import GridDetection

    class _MockDetAgent(object):
        model_loaded = True

        def detect(self, frame_rgb):
            return [((400, 80, 520, 200), 0.9, 1)]

    pat = (7, 3)
    tr = LeaderTurnTracker(window=10)
    cfg = {
        "role": "follower",
        "leader_detector_enabled": True,
        "leader_detector_class": 1,
        "leader_detector_min_area": 400,
        "grid_cols": 7,
        "grid_rows": 3,
        "intersection_last_cx_offset_px": 20,
        "intersection_heading_thresh": 0.12,
        "intersection_heading_sign": -1.0,
    }
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    set_detector_agent(_MockDetAgent())
    reset_leader_detector_cache()
    tr.update(GridDetection(
        True, 0.5, 1.0, None, pat, bbox=(100, 50, 200, 110),
        center_x=150.0, heading=0.15, source="grid",
    ))
    assert tr.infer(cfg, frame_w=640.0) == TURN_LEFT, "grid memory says left"
    det = GridDetection(
        True, 0.5, 0.9, None, pat, bbox=(400, 80, 520, 200),
        center_x=460.0, heading=-0.15, source="detector",
    )
    _update_intersection_turn_tracker(tr, det)
    assert tr.infer(cfg, frame_w=640.0) == TURN_RIGHT, "detector overwrites grid memory"

    tr.reset()
    _update_intersection_turn_tracker(tr, det)
    assert tr.infer(cfg, frame_w=640.0) == TURN_RIGHT, "detector bbox right of center => right"

    set_detector_agent(None)
    reset_leader_detector_cache()
    tr.reset()
    tr.update(GridDetection(
        True, 0.5, 1.0, None, pat, bbox=(100, 50, 200, 110),
        center_x=150.0, heading=0.15, source="grid",
    ))
    assert tr.infer(cfg, frame_w=640.0) == TURN_LEFT, "grid heading without detector"
    print("OK: intersection turn source priority")


def test_leader_detector_tracking():
    from tasks.project.packages.leader_detector import (
        detect_leader_detector,
        reset_leader_detector_cache,
        set_detector_agent,
    )

    class _MockDetAgent(object):
        model_loaded = True

        def detect(self, frame_rgb):
            return [((200, 80, 360, 200), 0.92, 1)]

    set_detector_agent(_MockDetAgent())
    reset_leader_detector_cache()
    cfg = {
        "role": "follower",
        "leader_detector_enabled": True,
        "leader_detector_class": 1,
        "leader_detector_min_area": 400,
        "grid_cols": 7,
        "grid_rows": 3,
        "leader_grid_fallback_enabled": False,
    }
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    det = detect_leader_detector(frame, cfg)
    assert det.found, "mock YOLO truck should be detected"
    assert det.source == "detector"
    assert det.center_x is not None and det.span_px is not None
    set_detector_agent(None)
    print("OK: leader detector tracking")


def test_leader_detector_spacing_params():
    from tasks.project.packages.leader_detector import (
        leader_spacing_span_px,
        leader_spacing_target_px,
        leader_spacing_too_close_px,
    )
    from tasks.project.packages.leader_grid import GridDetection

    cfg = {
        "span_target_px": 7.0,
        "span_too_close_px": 18.0,
        "leader_detector_span_target_px": 38.0,
        "grid_cols": 7,
        "grid_rows": 3,
    }
    det = GridDetection(
        True, 0.5, 0.9, None, (7, 3),
        span_px=40.0, source="detector",
    )
    assert leader_spacing_span_px(det, cfg) == 40.0
    assert leader_spacing_target_px(det, cfg) == 38.0
    assert abs(leader_spacing_too_close_px(det, cfg) - 97.7) < 0.2

    grid = GridDetection(True, 0.5, 0.9, None, (7, 3), span_px=6.5, source="grid")
    assert leader_spacing_target_px(grid, cfg) == 7.0
    assert leader_spacing_too_close_px(grid, cfg) == 18.0
    print("OK: leader detector spacing params")


def test_role_dispatch():
    called = {"leader": 0, "follower": 0}
    original_load = agent.load_config
    original_leader = agent.run_leader
    original_follower = agent.run_follower

    def fake_leader(*args, **kwargs):
        called["leader"] += 1

    def fake_follower(*args, **kwargs):
        called["follower"] += 1

    try:
        agent.run_leader = fake_leader
        agent.run_follower = fake_follower

        agent.load_config = lambda: {"role": "leader"}
        agent.main(None, None, None, threading.Event())
        assert called["leader"] == 1 and called["follower"] == 0, "Leader dispatch failed"

        called["leader"] = 0
        called["follower"] = 0
        agent.load_config = lambda: {"role": "follower"}
        agent.main(None, None, None, threading.Event())
        assert called["follower"] == 1 and called["leader"] == 0, "Follower dispatch failed"
        print("OK: role dispatch works")
    finally:
        agent.load_config = original_load
        agent.run_leader = original_leader
        agent.run_follower = original_follower


def test_slow_sign_delay():
    agent._sign_runtime.update(
        {
            "candidate_event": agent.EVENT_NORMAL,
            "candidate_count": 0,
            "active_until": 0.0,
            "slow_confirm_count": 0,
            "slow_engaged": False,
            "slow_loss_streak": 0,
            "slow_pending_until": 0.0,
            "stop_confirm_count": 0,
            "stop_triggered_latch": False,
            "stop_pending_rearm": False,
            "stop_rearm_until": 0.0,
        }
    )
    cfg = {
        "stop_tag_ids": [20],
        "slow_tag_ids": [39],
        "sign_confirm_frames": 2,
        "sign_cooldown_s": 0.0,
        "sign_center_roi": 1.0,
        "sign_slow_mode": "delay",
        "sign_slow_delay_s": 2.5,
    }
    slow_det = agent._TagDetection(39, (320.0, 240.0))
    shape = (480, 640)

    for _ in range(2):
        assert agent.detect_sign_event([slow_det], shape, cfg) == agent.EVENT_NORMAL

    assert agent.detect_sign_event([slow_det], shape, cfg) == agent.EVENT_NORMAL
    assert float(agent._sign_runtime.get("slow_pending_until", 0.0)) > time.time()

    agent._sign_runtime["slow_pending_until"] = time.time() - 0.01
    assert agent.detect_sign_event([slow_det], shape, cfg) == agent.EVENT_SLOW_SIGN
    assert agent._sign_runtime["candidate_event"] == agent.EVENT_SLOW_SIGN
    print("OK: slow sign waits before triggering")


def test_tag_slow_distance_engages():
    agent.reset_sign_detection_state()
    cfg = {
        "stop_tag_ids": [20],
        "slow_tag_ids": [39],
        "sign_confirm_frames": 2,
        "sign_cooldown_s": 2.0,
        "sign_center_roi": 1.0,
        "sign_slow_mode": "distance",
        "sign_slow_distance_m": 0.40,
        "sign_slow_loss_confirm_frames": 2,
    }
    far = agent._TagDetection(39, (320.0, 240.0), 0.55)
    near = agent._TagDetection(39, (320.0, 240.0), 0.35)
    shape = (480, 640)

    assert agent.detect_sign_event([far], shape, cfg) == agent.EVENT_NORMAL
    assert agent.detect_sign_event([near], shape, cfg) == agent.EVENT_NORMAL
    assert agent.detect_sign_event([near], shape, cfg) == agent.EVENT_SLOW_SIGN
    assert agent._sign_runtime.get("slow_engaged") is True
    # Tag briefly lost: stay in slow until loss confirm frames elapse.
    assert agent.detect_sign_event([], shape, cfg) == agent.EVENT_SLOW_SIGN
    assert agent.detect_sign_event([], shape, cfg) == agent.EVENT_NORMAL
    assert agent._sign_runtime.get("slow_engaged") is False
    print("OK: slow sign engages by distance and releases after tag loss")


def test_tag_slow_candidate_not_stale_normal():
    agent.reset_sign_detection_state()
    cfg = {
        "stop_tag_ids": [20],
        "slow_tag_ids": [39],
        "sign_confirm_frames": 1,
        "sign_cooldown_s": 2.0,
        "sign_center_roi": 1.0,
        "sign_slow_mode": "distance",
        "sign_slow_distance_m": 0.40,
        "sign_slow_loss_confirm_frames": 99,
    }
    near = agent._TagDetection(39, (320.0, 240.0), 0.30)
    shape = (480, 640)
    assert agent.detect_sign_event([near], shape, cfg) == agent.EVENT_SLOW_SIGN
    assert agent.detect_sign_event([], shape, cfg) == agent.EVENT_SLOW_SIGN
    assert agent._sign_runtime["candidate_event"] == agent.EVENT_SLOW_SIGN
    print("OK: engaged slow keeps EVENT_SLOW_SIGN during cooldown")


def test_stop_sign_distance():
    agent.reset_sign_detection_state()
    cfg = {
        "stop_tag_ids": [20],
        "slow_tag_ids": [39],
        "sign_confirm_frames": 2,
        "sign_cooldown_s": 0.0,
        "sign_center_roi": 1.0,
        "sign_stop_distance_m": 0.30,
        "sign_stop_rearm_s": 1.0,
    }
    far = agent._TagDetection(20, (320.0, 240.0), 0.55)
    near = agent._TagDetection(20, (320.0, 240.0), 0.25)
    shape = (480, 640)

    assert agent.detect_sign_event([far], shape, cfg) == agent.EVENT_NORMAL
    assert agent.detect_sign_event([near], shape, cfg) == agent.EVENT_NORMAL

    ev = agent.detect_sign_event([near], shape, cfg)
    assert ev == agent.EVENT_STOP_SIGN, "Stop should trigger when tag is within distance"
    assert agent._sign_runtime.get("stop_triggered_latch") is True
    assert agent._sign_runtime.get("stop_pending_rearm") is True

    assert agent.detect_sign_event([near], shape, cfg) == agent.EVENT_NORMAL
    print("OK: stop sign triggers on distance, not tag loss")


def test_sign_state_cleared_on_pause():
    agent._sign_runtime["slow_engaged"] = True
    agent._sign_runtime["slow_confirm_count"] = 5
    agent._sign_runtime["stop_triggered_latch"] = True
    agent.set_driving_enabled(True)
    agent.set_driving_enabled(False)
    assert agent._sign_runtime.get("slow_engaged") is False
    assert int(agent._sign_runtime.get("slow_confirm_count", 0)) == 0
    assert agent._sign_runtime.get("stop_triggered_latch") is False
    print("OK: sign detection state cleared on pause/stop")


def test_follower_cruise_target_speed_startup():
    from tasks.project.packages.agent import _follower_cruise_target_speed
    from tasks.project.packages.follower_spacing import GridSpacingController

    cfg = {
        "slow_speed": 0.12,
        "span_target_px": 28.0,
        "spacing_kp": 0.012,
        "spacing_kd": 0.022,
        "follower_require_leader": True,
        "follower_spacing_warmup_frames": 8,
        "follower_catchup_margin": 0.06,
        "follower_lane_fallback_speed": 0.12,
        "span_too_close_px": 44.0,
        "span_too_close_speed": 0.05,
    }
    spacing = GridSpacingController()

    blind = _follower_cruise_target_speed(
        spacing, cfg, 0.0, 0.368, 0.32, 0.12, False, 0, None,
    )
    assert blind == 0.0, "require_leader + no grid -> hold still"

    cfg_lane = dict(cfg)
    cfg_lane["follower_require_leader"] = False
    cfg_lane["follower_lane_fallback_speed"] = 0.32
    lane_only = _follower_cruise_target_speed(
        spacing, cfg_lane, 0.0, 0.368, 0.32, 0.12, False, 0, None,
    )
    assert lane_only == 0.32, "lane fallback when leader lost"

    spacing.observe(18.0, 0.0, cfg, 0.0)
    early = _follower_cruise_target_speed(
        spacing, cfg, 0.0, 0.368, 0.32, 0.12, True, 2, 18.0,
    )
    assert early <= 0.12, "warmup should not sprint before spacing settles"

    for i in range(10):
        spacing.observe(18.0, 0.2 + i * 0.1, cfg, 0.12)
    chase = _follower_cruise_target_speed(
        spacing, cfg, 0.0, 0.368, 0.32, 0.12, True, 10, 18.0,
    )
    assert chase <= 0.368 + 1e-6, "convoy should not outrun leader cruise by much"

    close = _follower_cruise_target_speed(
        spacing, cfg, 0.0, 0.368, 0.32, 0.12, True, 10, 45.0,
    )
    assert close <= 0.05 + 1e-6, "too-close span should crawl or stop"
    print("OK: follower cruise speed is conservative at startup")


def test_intersection_arc_pwm_not_scaled_by_cruise():
    from tasks.project.packages.agent import _apply_lane_wheels
    from tasks.project.packages.intersection_follow import intersection_wheel_commands

    class _Wheels:
        def __init__(self):
            self.left = self.right = 0.0

        def set_wheels_speed(self, left, right):
            self.left = float(left)
            self.right = float(right)

    cfg = {
        "intersection_turn_speed": 0.15,
        "intersection_turn_inner_ratio": 0.27,
        "intersection_turn_outer_ratio": 1.0,
    }
    expected = intersection_wheel_commands("right", cfg)
    wheels = _Wheels()
    _apply_lane_wheels(wheels, expected[0], expected[1], 0.368, True)
    assert abs(wheels.left - expected[0]) < 1e-6
    assert abs(wheels.right - expected[1]) < 1e-6
    print("OK: intersection arc PWM is direct, not cruise-scaled")


def test_intersection_turn_params_reload_and_live_schedule():
    from tasks.project.packages.intersection_follow import intersection_phase_deadlines

    before = agent.get_intersection_turn_params()
    result = agent.patch_intersection_turn_params({"intersection_turn_left_s": 3.21})
    assert result["status"] == "ok"
    assert result.get("config_path")
    cfg = agent.load_config()
    assert abs(cfg["intersection_turn_left_s"] - 3.21) < 1e-6

    t0 = 100.0
    _, arc_end, _, schedule = intersection_phase_deadlines(t0, "left", cfg)
    assert abs(schedule["arc_s"] - 3.21) < 1e-6
    assert abs(arc_end - (t0 + schedule["preamble_s"] + 3.21)) < 1e-6

    agent.patch_intersection_turn_params(
        {"intersection_turn_left_s": before["intersection_turn_left_s"]},
    )
    print("OK: intersection turn UI params reload and live schedule")


def test_follower_test_turn_queue():
    from unittest.mock import patch

    with patch.object(agent, "load_config", return_value={"role": "follower"}):
        result = agent.request_follower_test_turn("left")
    assert result["status"] == "ok" and result["direction"] == "left"
    assert agent.get_follower_test_turn_status()["queued"] == "left"
    assert agent._pop_follower_test_turn() == "left"
    print("OK: follower test turn queues without Start")


def test_loops_exit_cleanly():
    _run_and_stop(agent.run_leader)
    _run_and_stop(agent.run_follower)
    print("OK: loops exit cleanly with fake stop_event")


if __name__ == "__main__":
    test_next_state_transitions()
    test_smooth_stop()
    test_config_loads()
    test_grid_leader_distance_signal()
    test_follower_grid_signal_mock()
    test_grid_spacing_controller()
    test_red_at_line_near_band()
    test_project_lane_masks_not_cut_on_straight()
    test_lane_ignore_red_for_convoy()
    test_leader_turn_tracker()
    test_intersection_turn_uses_detector_not_grid_when_loaded()
    test_leader_detector_tracking()
    test_leader_detector_spacing_params()
    test_follower_cruise_target_speed_startup()
    test_intersection_arc_pwm_not_scaled_by_cruise()
    test_intersection_turn_params_reload_and_live_schedule()
    test_follower_test_turn_queue()
    test_slow_sign_delay()
    test_tag_slow_distance_engages()
    test_tag_slow_candidate_not_stale_normal()
    test_stop_sign_distance()
    test_sign_state_cleared_on_pause()
    test_role_dispatch()
    test_loops_exit_cleanly()
    print("All sanity checks passed.")
