import sys
import os
import threading
import queue
import socket
import traceback
import time

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, '..', '..')
sys.path.insert(0, project_root)

import cv2
import numpy as np
import yaml
from flask import Flask, Response, jsonify, render_template_string, request

from duckiebot.camera_driver.godot_camera_driver import GodotCameraDriver, GodotCameraConfig
from duckiebot.wheel_driver.godot_wheels_driver import GodotWheelsDriver
from duckiebot.wheel_driver.wheels_driver_abs import WheelPWMConfiguration
from launcher.ports import find_available_port
from servers.common import make_frame_generator, shutdown_cleanup, suppress_http_logs
from servers.templates.project import get_template
from servers.visual_lane_servoing.visualization import create_lane_visualization
from servers.visual_lane_servoing.color_sample import sample_pixel_from_frame_bgr
from tasks.visual_lane_servoing.packages.agent import LaneServoingAgent
from tasks.visual_lane_servoing.packages import visual_servoing_activity as _lane_activity
from tasks.project.packages.leader_grid import reset_grid_tracker
import tasks.project.packages.agent as agent

LANE_CONFIG_FILE = os.path.join(project_root, 'config', 'lane_servoing_config.yaml')
LANE_HSV_CONFIG_FILE = os.path.join(project_root, 'config', 'lane_servoing_hsv_config.yaml')

app = Flask(__name__)
camera = None
wheels = None
stop_event = threading.Event()
_lane_agent = None
_frame_queue = queue.Queue(maxsize=1)


class QueuedCamera:
    def __init__(self, frame_queue: queue.Queue):
        self._q = frame_queue

    def start(self):
        return None

    def stop(self):
        return None

    def read(self):
        try:
            frame = self._q.get(timeout=0.25)
            return True, frame
        except queue.Empty:
            return False, None


def _feed_convoy_frame(frame_bgr):
    if frame_bgr is None:
        return
    try:
        _frame_queue.put_nowait(frame_bgr.copy())
    except queue.Full:
        try:
            _frame_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            _frame_queue.put_nowait(frame_bgr.copy())
        except queue.Full:
            pass


def _refresh_display_masks(frame_bgr):
    """Dashboard masks: full frame like visual_lane_servoing (no bottom crop)."""
    if _lane_agent is None or frame_bgr is None:
        return
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    _lane_agent.compute_commands(frame_rgb, ignore_bottom_frac=0.0, debug_red_mask=True)


def _apply_follower_grid_overlay(frame_bgr):
    if frame_bgr is None:
        return frame_bgr
    try:
        cfg = agent.load_config()
        if str(cfg.get('role', 'leader')).lower() != 'follower':
            return frame_bgr
        return agent.render_follower_grid_overlay(frame_bgr, cfg)
    except Exception as e:
        print(f'[Project][Sim] Grid overlay error: {e}')
        return frame_bgr


def _visualize(frame_bgr):
    if frame_bgr is None:
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(blank, 'Waiting for Godot camera...', (120, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 80), 2)
        return blank

    _feed_convoy_frame(frame_bgr)
    display_bgr = _apply_follower_grid_overlay(frame_bgr)

    if _lane_agent is None:
        return display_bgr

    _refresh_display_masks(frame_bgr)
    debug = _lane_agent.last_debug_info or {}
    pwm_left = float(getattr(_lane_agent, '_last_left', 0.0))
    pwm_right = float(getattr(_lane_agent, '_last_right', 0.0))
    try:
        return create_lane_visualization(display_bgr, debug, pwm_left, pwm_right)
    except Exception as e:
        print(f'[Project][Sim] visualize error: {e}')
        return display_bgr


generate_frames = make_frame_generator(lambda: camera, _visualize, quality=50, rgb=False)


@app.route('/')
def index():
    cfg = agent.load_config()
    role = str(cfg.get('role', 'leader'))
    lane_cfg = _lane_agent if _lane_agent is not None else LaneServoingAgent()
    html = get_template(
        title=f'Convoy Project — {socket.gethostname()}',
        subtitle=f'{role} — Godot simulation',
    )
    return render_template_string(html, config=lane_cfg, hostname=socket.gethostname())


@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/reset', methods=['POST'])
def reset():
    agent.set_driving_enabled(False)
    if wheels is not None:
        wheels.reset_game()
        time.sleep(0.2)
        wheels.clear_state()
    reset_grid_tracker()
    return jsonify({'status': 'ok'})


@app.route('/update_config', methods=['POST'])
def update_config():
    if _lane_agent is None:
        return jsonify({'status': 'error', 'message': 'Lane agent not ready'}), 503
    data = request.json or {}
    _lane_agent.p_gain = float(data.get('k_d', _lane_agent.p_gain))
    _lane_agent.d_gain = float(data.get('k_phi', _lane_agent.d_gain))
    _lane_agent.base_speed = float(data.get('const', _lane_agent.base_speed))
    try:
        with open(LANE_CONFIG_FILE, 'r') as f:
            saved = yaml.safe_load(f) or {}
        saved['p_gain'] = _lane_agent.p_gain
        saved['d_gain'] = _lane_agent.d_gain
        saved['base_speed'] = _lane_agent.base_speed
        with open(LANE_CONFIG_FILE, 'w') as f:
            yaml.dump(saved, f, default_flow_style=False)
    except Exception as e:
        print(f'[Project][Sim] Could not save config: {e}')
    return jsonify({'status': 'ok'})


@app.route('/start', methods=['POST'])
def start():
    agent.set_driving_enabled(True)
    return jsonify({'status': 'running'})


@app.route('/stop', methods=['POST'])
def stop():
    agent.set_driving_enabled(False)
    if wheels is not None:
        wheels.set_wheels_speed(0.0, 0.0)
    return jsonify({'status': 'stopped'})


@app.route('/running')
def get_running():
    return jsonify({'running': agent.is_driving_enabled()})


@app.route('/convoy/manual', methods=['POST'])
def convoy_manual():
    cfg = agent.load_config()
    if str(cfg.get('role', 'leader')).lower() != 'leader':
        return jsonify({'ok': False, 'error': 'Leader bot only'}), 403
    data = request.json or {}
    command = data.get('command') or data.get('mode')
    if not command:
        return jsonify({'ok': False, 'error': 'Missing command'}), 400
    try:
        return jsonify(agent.set_manual_convoy_command(command))
    except ValueError as e:
        return jsonify({'ok': False, 'error': str(e)}), 400


@app.route('/status')
def status():
    cfg = agent.load_config()
    payload = agent.get_convoy_ui_status()
    payload['role'] = cfg.get('role', 'leader')
    payload.update(agent.get_runtime_status(cfg))
    payload['simulation'] = True
    gs = wheels.game_state if wheels is not None else None
    if gs is not None and gs.npc_x is not None:
        payload['leader_pose'] = {'x': gs.npc_x, 'z': gs.npc_z}
    return jsonify(payload)


@app.route('/get_hsv')
def get_hsv():
    return jsonify(_lane_activity.get_hsv_bounds())


@app.route('/update_hsv', methods=['POST'])
def update_hsv():
    data = request.json or {}
    current = _lane_activity.get_hsv_bounds()
    current.update({k: int(v) for k, v in data.items()})
    _lane_activity.set_hsv_bounds(
        [current['yellow_lower_h'], current['yellow_lower_s'], current['yellow_lower_v']],
        [current['yellow_upper_h'], current['yellow_upper_s'], current['yellow_upper_v']],
        [current['white_lower_h'], current['white_lower_s'], current['white_lower_v']],
        [current['white_upper_h'], current['white_upper_s'], current['white_upper_v']],
        [current['red_lower_h'], current['red_lower_s'], current['red_lower_v']],
        [current['red_upper_h'], current['red_upper_s'], current['red_upper_v']],
    )
    try:
        with open(LANE_HSV_CONFIG_FILE, 'w') as f:
            yaml.dump(current, f, default_flow_style=False)
    except Exception as e:
        print(f'[Project][Sim] Could not save HSV config: {e}')
    return jsonify({'status': 'ok'})


@app.route('/sample_pixel', methods=['POST'])
def sample_pixel():
    if _lane_agent is None:
        return jsonify({'ok': False, 'error': 'Lane agent not initialized'}), 503
    data = request.json or {}
    try:
        stream_x = float(data['stream_x'])
        stream_y = float(data['stream_y'])
    except (KeyError, TypeError, ValueError):
        return jsonify({'ok': False, 'error': 'Need stream_x and stream_y'}), 400
    debug = _lane_agent.last_debug_info or {}
    frame_bgr = debug.get('frame_bgr')
    if frame_bgr is None and debug.get('roi') is not None:
        frame_bgr = cv2.cvtColor(debug['roi'], cv2.COLOR_RGB2BGR)
    result = sample_pixel_from_frame_bgr(frame_bgr, stream_x, stream_y)
    return jsonify(result), 200 if result.get('ok') else 400


def _run_agent_thread(queued_cam):
    try:
        agent.main(queued_cam, wheels, None, stop_event)
    except Exception:
        print('[Project][Sim] Agent thread crashed:')
        traceback.print_exc()


def main():
    global camera, wheels, _lane_agent

    import argparse
    ap = argparse.ArgumentParser(description='Virtual Project Convoy Server')
    ap.add_argument('--port', type=int, default=5000)
    ap.add_argument('--frame-port', type=int, default=5001)
    ap.add_argument('--wheel-port', type=int, default=5002)
    ap.add_argument('--godot-host', type=str, default='localhost')
    args = ap.parse_args()

    suppress_http_logs()
    cfg = agent.load_config()
    role = str(cfg.get('role', 'leader'))

    print('=' * 60)
    print('VIRTUAL PROJECT CONVOY SERVER')
    print('=' * 60)
    print(f'Role: {role}  (set in config/project_config.yaml)')
    if role == 'follower':
        print('Follower sim: NPC leader with MarkerGridBoard on project_convoy.tscn')
        print('  1. Wait ~4s after load for NPC to depart (camera grace)')
        print('  2. leader_visible=no still lane-follows; yes enables convoy spacing')
        print('  3. Click Start to drive')
    else:
        print('Leader sim: lane follow on convoy map (no NPC needed)')
    print('=' * 60)

    print('\n[1/4] Wheels...')
    wheels = GodotWheelsDriver(
        WheelPWMConfiguration(pwm_min=0), WheelPWMConfiguration(pwm_min=0),
        godot_host=args.godot_host, godot_port=args.wheel_port,
    )
    wheels.trim = 0

    print('\n[2/4] Camera...')
    camera = GodotCameraDriver(godot_config=GodotCameraConfig(host='0.0.0.0', port=args.frame_port))
    camera.start()

    print('\n[3/4] Lane agent...')
    _lane_agent = LaneServoingAgent()
    _lane_agent._last_left = 0.0
    _lane_agent._last_right = 0.0
    agent.set_lane_agent(_lane_agent)
    agent.set_driving_enabled(False)

    print('\n[4/4] Convoy agent thread...')
    threading.Thread(
        target=_run_agent_thread,
        args=(QueuedCamera(_frame_queue),),
        daemon=True,
        name='ProjectAgentThread',
    ).start()

    web_port = find_available_port(args.port)
    print(f'\nWeb Interface: http://localhost:{web_port}')
    print('=' * 60 + '\n')

    try:
        app.run(host='127.0.0.1', port=web_port, debug=False, threaded=True)
    except KeyboardInterrupt:
        print('\nShutting down...')
    finally:
        shutdown_cleanup(wheels, camera, stop_event)


if __name__ == '__main__':
    sys.exit(main())
