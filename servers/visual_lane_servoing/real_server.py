import sys
import os
import signal
import argparse

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, '..', '..')
sys.path.insert(0, project_root)

from flask import Flask, Response, render_template_string, request, jsonify
import cv2
import socket
import yaml

from tasks.visual_lane_servoing.packages.agent import LaneServoingAgent
from servers.visual_lane_servoing.visualization import create_lane_visualization
from servers.visual_lane_servoing.color_sample import sample_pixel_from_frame_bgr
from servers.templates.lane_servoing import LANE_SERVOING_TEMPLATE as HTML_TEMPLATE

from duckiebot.camera_driver import CameraDriver
from duckiebot.wheel_driver import DaguWheelsDriver
from duckiebot.wheel_driver.wheels_driver_abs import WheelPWMConfiguration
from launcher.ports import find_available_port
from servers.common import make_frame_generator, shutdown_cleanup, suppress_http_logs

LANE_CONFIG_FILE = os.path.join(project_root, 'config', 'lane_servoing_config.yaml')
LANE_HSV_CONFIG_FILE = os.path.join(project_root, 'config', 'lane_servoing_hsv_config.yaml')


def _get_student_module():
    from tasks.visual_lane_servoing.packages import visual_servoing_activity
    return visual_servoing_activity


app = Flask(__name__)

camera = None
wheels = None
agent = None
running = False
stop_event = __import__('threading').Event()


def visualize(frame_bgr):
    global running
    if agent is None or wheels is None or frame_bgr is None:
        if frame_bgr is not None:
            return frame_bgr
        import numpy as np
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(blank, "Waiting for camera...", (160, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 80, 80), 2)
        return blank

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pwm_left, pwm_right = agent.compute_commands(frame_rgb)
    if running:
        wheels.set_wheels_speed(pwm_left, pwm_right)
    else:
        wheels.set_wheels_speed(0.0, 0.0)

    return create_lane_visualization(
        frame_bgr, agent.last_debug_info, pwm_left, pwm_right
    )


generate_frames = make_frame_generator(lambda: camera, visualize, quality=50, rgb=False)


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE, config=agent, hostname=socket.gethostname())


@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/reset', methods=['POST'])
def reset():
    if wheels is not None:
        wheels.set_wheels_speed(0.0, 0.0)
    return jsonify({'status': 'ok'})


@app.route('/update_config', methods=['POST'])
def update_config():
    data = request.json
    agent.p_gain = float(data.get('k_d', agent.p_gain))
    agent.d_gain = float(data.get('k_phi', agent.d_gain))
    agent.base_speed = float(data.get('const', agent.base_speed))
    try:
        with open(LANE_CONFIG_FILE, 'r') as f:
            saved = yaml.safe_load(f) or {}
        saved['p_gain'] = agent.p_gain
        saved['d_gain'] = agent.d_gain
        saved['base_speed'] = agent.base_speed
        with open(LANE_CONFIG_FILE, 'w') as f:
            yaml.dump(saved, f, default_flow_style=False)
    except Exception as e:
        print(f"[LaneServoing] Could not save config: {e}")
    return jsonify({'status': 'ok'})


@app.route('/get_hsv')
def get_hsv():
    return jsonify(_get_student_module().get_hsv_bounds())


@app.route('/sample_pixel', methods=['POST'])
def sample_pixel():
    if agent is None:
        return jsonify({'ok': False, 'error': 'Agent not initialized'}), 503

    data = request.json or {}
    try:
        stream_x = float(data['stream_x'])
        stream_y = float(data['stream_y'])
    except (KeyError, TypeError, ValueError):
        return jsonify({'ok': False, 'error': 'Need stream_x and stream_y (float)'}), 400

    debug = agent.last_debug_info or {}
    frame_bgr = debug.get('frame_bgr')
    if frame_bgr is None and debug.get('roi') is not None:
        frame_bgr = cv2.cvtColor(debug['roi'], cv2.COLOR_RGB2BGR)

    result = sample_pixel_from_frame_bgr(frame_bgr, stream_x, stream_y)
    status = 200 if result.get('ok') else 400
    return jsonify(result), status


@app.route('/update_hsv', methods=['POST'])
def update_hsv():
    data = request.json
    mod = _get_student_module()
    current = mod.get_hsv_bounds()
    current.update({k: int(v) for k, v in data.items()})
    mod.set_hsv_bounds(
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
        print(f"[LaneServoing] Could not save HSV config: {e}")
    return jsonify({'status': 'ok'})


@app.route('/start', methods=['POST'])
def start():
    global running
    running = True
    print("[Control] Started")
    return jsonify({'status': 'running'})


@app.route('/stop', methods=['POST'])
def stop():
    global running
    running = False
    if wheels:
        wheels.set_wheels_speed(0.0, 0.0)
    print("[Control] Stopped")
    return jsonify({'status': 'stopped'})


@app.route('/running')
def get_running():
    return jsonify({'running': running})


@app.route('/status')
def status():
    if agent is None:
        return jsonify({'status': 'not_initialized'})
    return jsonify({
        'status': 'active',
        'frame_count': agent.frame_count,
        'config': {
            'p_gain': agent.p_gain,
            'd_gain': agent.d_gain,
            'base_speed': agent.base_speed,
            'detection_threshold': agent.detection_threshold,
        },
    })


def main():
    global camera, wheels, agent

    ap = argparse.ArgumentParser(description='Lane Servoing Server — Real Hardware')
    ap.add_argument('--port', type=int, default=5000)
    args = ap.parse_args()

    suppress_http_logs()
    print('=' * 60)
    print('LANE SERVOING SERVER — REAL HARDWARE')
    print('=' * 60)

    print('\n[1/3] Initializing wheels driver...')
    wheels = DaguWheelsDriver(WheelPWMConfiguration(), WheelPWMConfiguration())
    print('  Wheels: ok')

    print('\n[2/3] Initializing camera driver...')
    camera = CameraDriver()
    camera.start()
    print('  Camera: ok')

    print('\n[3/3] Creating agent...')
    agent = LaneServoingAgent()
    print(f'  p_gain={agent.p_gain}, d_gain={agent.d_gain}, base_speed={agent.base_speed}')

    def _shutdown(signum, frame):
        print('\nShutting down...')
        shutdown_cleanup(wheels, camera, stop_event)
        sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    web_port = find_available_port(args.port)
    print(f'\nWeb Interface: http://localhost:{web_port}')
    print('Click the Camera panel (top-left) to sample HSV.\n')

    try:
        app.run(host='0.0.0.0', port=web_port, debug=False, threaded=True)
    except KeyboardInterrupt:
        print('\nShutting down...')
    finally:
        shutdown_cleanup(wheels, camera, stop_event)


if __name__ == '__main__':
    sys.exit(main())
