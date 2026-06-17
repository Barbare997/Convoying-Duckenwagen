import sys

import os

import signal

import threading

import argparse

import queue

import socket

import time

import traceback



script_dir   = os.path.dirname(os.path.abspath(__file__))

project_root = os.path.join(script_dir, '..', '..')

sys.path.insert(0, project_root)



from flask import Flask, Response, jsonify, render_template_string, request

import numpy as np

import cv2

import yaml



from duckiebot.camera_driver import CameraDriver

from duckiebot.wheel_driver import DaguWheelsDriver

from duckiebot.wheel_driver.wheels_driver_abs import WheelPWMConfiguration

from duckiebot.led_driver import LEDDriver

from launcher.ports import find_available_port

from servers.common import make_frame_generator, shutdown_cleanup, suppress_http_logs



try:

    from servers.project.camera_source import (
        open_project_camera,
        PlaceholderCamera,
        WaitingCaptureCamera,
        stop_project_camera,
    )

except Exception as e:

    open_project_camera = None

    WaitingCaptureCamera = None

    PlaceholderCamera = None

    stop_project_camera = None

    print(f'[Project] camera_source unavailable ({e}), using direct CameraDriver')



try:

    from servers.templates.project import get_template

except ModuleNotFoundError:

    get_template = None



try:

    from servers.visual_lane_servoing.visualization import create_lane_visualization

except ModuleNotFoundError:

    create_lane_visualization = None

try:

    from servers.visual_lane_servoing.color_sample import sample_pixel_from_frame_bgr

except ModuleNotFoundError:

    sample_pixel_from_frame_bgr = None

from tasks.visual_lane_servoing.packages.agent import (
    LaneServoingAgent,
    detect_lines_in_slices,
    _NUM_SLICES,
    _ROI_START,
)
from tasks.visual_lane_servoing.packages import visual_servoing_activity as _lane_activity

import tasks.project.packages.agent as agent



LANE_CONFIG_FILE = os.path.join(project_root, 'config', 'lane_servoing_config.yaml')

LANE_HSV_CONFIG_FILE = os.path.join(project_root, 'config', 'lane_servoing_hsv_config.yaml')



app        = Flask(__name__)

camera     = None

wheels     = None

leds       = None

stop_event = threading.Event()

_lane_agent = None

_frame_queue = queue.Queue(maxsize=1)

_hw_ready    = threading.Event()
_debug_lock  = threading.Lock()
_direct_camera_fallback = False


def _feed_convoy_frame(frame_bgr):
    """When direct CameraDriver fallback is used, MJPEG is the only reader — feed the convoy queue."""
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


class QueuedCamera:

    """Convoy agent reads frames from the shared capture queue."""



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





def _refresh_display_masks(frame_bgr):
    """Re-run HSV masks on the live frame (same as visual_lane_servoing video loop)."""
    if _lane_agent is None or frame_bgr is None:
        return

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    try:
        mask_y, mask_w, mask_r = _lane_activity.detect_lane_markings(frame_bgr)
    except Exception as e:
        print(f'[Project] detect_lane_markings error: {e}')
        return

    mask_y_u8 = (mask_y * 255).astype(np.uint8)
    mask_r_u8 = (mask_r * 255).astype(np.uint8)
    mask_w_u8 = (mask_w * 255).astype(np.uint8)
    mask_left = np.clip(mask_y + mask_r, 0, 1)
    combined = np.clip(mask_left + mask_w, 0, 1)
    h, w = mask_y_u8.shape
    mask_left_u8 = (mask_left * 255).astype(np.uint8)
    yellow_xs, white_xs, red_xs = detect_lines_in_slices(mask_left_u8, mask_w_u8, h, mask_r_u8)
    slice_height = int(h * 0.35 / _NUM_SLICES)
    start_y = int(h * _ROI_START)
    total_pixels = int(np.count_nonzero(mask_y_u8) + np.count_nonzero(mask_r_u8) + np.count_nonzero(mask_w_u8))

    with _debug_lock:
        debug = dict(_lane_agent.last_debug_info or _lane_agent._empty_debug_info(h, w))
        debug.update({
            'frame_bgr': frame_bgr,
            'roi': frame_rgb,
            'lane_mask': (combined * 255).astype(np.uint8),
            'white_mask': mask_w_u8,
            'yellow_mask': mask_y_u8,
            'red_mask': mask_r_u8,
            'total_lane_pixels': total_pixels,
            'lane_detected': total_pixels >= _lane_agent.detection_threshold,
            'yellow_xs': yellow_xs,
            'white_xs': white_xs,
            'red_xs': red_xs,
            'slice_ys': [start_y + i * slice_height + slice_height // 2 for i in range(_NUM_SLICES)],
        })
        _lane_agent.last_debug_info = debug


def _apply_follower_grid_overlay(frame_bgr):
    """Draw circle-grid dots on the camera panel (follower only)."""
    if frame_bgr is None:
        return frame_bgr
    try:
        cfg = agent.load_config()
        if str(cfg.get("role", "leader")).lower() != "follower":
            return frame_bgr
        return agent.render_follower_grid_overlay(frame_bgr, cfg)
    except Exception as e:
        print(f'[Project] Grid overlay error: {e}')
        return frame_bgr


def _visualize(frame_bgr):
    if frame_bgr is None:
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(blank, "Waiting for camera...", (160, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 80, 80), 2)
        return blank

    if _direct_camera_fallback:
        _feed_convoy_frame(frame_bgr)

    display_bgr = _apply_follower_grid_overlay(frame_bgr)

    if _lane_agent is None or create_lane_visualization is None:
        return display_bgr

    _refresh_display_masks(frame_bgr)
    debug = _lane_agent.last_debug_info or {}
    pwm_left = float(getattr(_lane_agent, '_last_left', 0.0))
    pwm_right = float(getattr(_lane_agent, '_last_right', 0.0))
    try:
        return create_lane_visualization(display_bgr, debug, pwm_left, pwm_right)
    except Exception as e:
        print(f'[Project] visualize error: {e}')
        return display_bgr





generate_frames = make_frame_generator(lambda: camera, _visualize, quality=50, rgb=False)





def _fallback_index_html(role: str) -> str:

    host = socket.gethostname()

    return f"""<!DOCTYPE html>

<html><head><meta charset="UTF-8"><title>Convoy — {host}</title>

<style>body{{font-family:sans-serif;background:#1a1d23;color:#e6edf3;margin:12px}}

.grid{{display:grid;grid-template-columns:1fr 320px;gap:12px;max-width:1400px;margin:0 auto}}

img.stream{{width:100%;border-radius:6px}} .card{{background:#13161a;border:1px solid #30363d;padding:12px;border-radius:6px}}

.row{{display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid #30363d;font-size:13px}}</style></head>

<body><h2>Convoy Project — {host}</h2><p>{role} — real hardware</p>

<div class="grid"><div><img src="/video" class="stream"></div>

<div class="card"><div id="statusTable">Loading…</div></div></div>

<script>

function refresh(){{fetch('/status').then(r=>r.json()).then(d=>{{

  const rows=[['role',d.role],['state',d.state],['speed',d.speed!=null?Number(d.speed).toFixed(2):'-'],

    ['event',d.event||'-'],['tags',(d.tag_ids&&d.tag_ids.length)?d.tag_ids.join(', '):'-']];

  document.getElementById('statusTable').innerHTML=rows.map(([k,v])=>

    '<div class="row"><span>'+k+'</span><span>'+v+'</span></div>').join('');

}}).catch(()=>{{}});}}

refresh(); setInterval(refresh,400);

</script></body></html>"""





@app.route('/health')

def health():

    return jsonify({'status': 'ok', 'hardware': _hw_ready.is_set()})





@app.route('/')

def index():

    cfg = agent.load_config()

    role = str(cfg.get('role', 'leader'))

    lane_cfg = _lane_agent if _lane_agent is not None else LaneServoingAgent()

    if get_template is not None:

        if _hw_ready.is_set():

            subtitle = f"{role} — real hardware"

        elif camera is not None:

            subtitle = f"{role} — camera connecting…"

        else:

            subtitle = f"{role} — starting…"

        html = get_template(

            title=f"Convoy Project — {socket.gethostname()}",

            subtitle=subtitle,

        )

        return render_template_string(html, config=lane_cfg, hostname=socket.gethostname())

    return _fallback_index_html(role)





@app.route('/video')

def video():

    return Response(generate_frames(),

                    mimetype='multipart/x-mixed-replace; boundary=frame')





@app.route('/reset', methods=['POST'])

def reset():

    agent.set_driving_enabled(False)

    if wheels is not None:

        wheels.set_wheels_speed(0.0, 0.0)

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

        print(f"[Project] Could not save config: {e}")

    return jsonify({'status': 'ok'})





@app.route('/start', methods=['POST'])

def start():

    agent.set_driving_enabled(True)

    print('[Project][Control] Started')

    return jsonify({'status': 'running'})





@app.route('/stop', methods=['POST'])

def stop():

    agent.set_driving_enabled(False)

    if wheels is not None:

        wheels.set_wheels_speed(0.0, 0.0)

    print('[Project][Control] Stopped')

    return jsonify({'status': 'stopped'})





@app.route('/running')

def get_running():

    return jsonify({'running': agent.is_driving_enabled()})





@app.route('/shutdown')

def shutdown():

    agent.set_driving_enabled(False)

    if camera is not None:

        camera.stop()

    if stop_project_camera is not None:

        stop_project_camera()

    shutdown_cleanup(wheels, None, stop_event)

    return jsonify({'status': 'ok'})





@app.route('/convoy/manual', methods=['POST'])

def convoy_manual():

    cfg = agent.load_config()

    if str(cfg.get("role", "leader")).lower() != "leader":

        return jsonify({'ok': False, 'error': 'Leader bot only'}), 403

    data = request.json or {}

    command = data.get('command') or data.get('mode')

    if not command:

        return jsonify({'ok': False, 'error': 'Missing command (CRUISING, SLOW, STOPPED)'}), 400

    try:

        result = agent.set_manual_convoy_command(command)

        return jsonify(result)

    except ValueError as e:

        return jsonify({'ok': False, 'error': str(e)}), 400





@app.route('/status')

def status():

    cfg = agent.load_config()

    payload = agent.get_convoy_ui_status()

    payload['role'] = cfg.get('role', 'leader')

    payload.update(agent.get_runtime_status(cfg))

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

        [current['white_lower_h'],  current['white_lower_s'],  current['white_lower_v']],

        [current['white_upper_h'],  current['white_upper_s'],  current['white_upper_v']],

        [current['red_lower_h'], current['red_lower_s'], current['red_lower_v']],

        [current['red_upper_h'], current['red_upper_s'], current['red_upper_v']],

    )

    try:

        with open(LANE_HSV_CONFIG_FILE, 'w') as f:

            yaml.dump(current, f, default_flow_style=False)

    except Exception as e:

        print(f"[Project] Could not save HSV config: {e}")

    return jsonify({'status': 'ok'})





@app.route('/sample_pixel', methods=['POST'])

def sample_pixel():

    if _lane_agent is None:

        return jsonify({'ok': False, 'error': 'Lane agent not initialized'}), 503

    if sample_pixel_from_frame_bgr is None:

        return jsonify({'ok': False, 'error': 'color_sample module not available'}), 503



    data = request.json or {}

    try:

        stream_x = float(data['stream_x'])

        stream_y = float(data['stream_y'])

    except (KeyError, TypeError, ValueError):

        return jsonify({'ok': False, 'error': 'Need stream_x and stream_y (float)'}), 400



    debug = _lane_agent.last_debug_info or {}

    frame_bgr = debug.get('frame_bgr')

    if frame_bgr is None and debug.get('roi') is not None:

        frame_bgr = cv2.cvtColor(debug['roi'], cv2.COLOR_RGB2BGR)



    result = sample_pixel_from_frame_bgr(frame_bgr, stream_x, stream_y)

    status = 200 if result.get('ok') else 400

    return jsonify(result), status





def _replace_camera(new_cam) -> None:
    global camera
    old = camera
    camera = new_cam
    if old is not None and old is not new_cam:
        try:
            old.stop()
        except Exception:
            pass


def _open_camera():

    global camera, _direct_camera_fallback

    _direct_camera_fallback = False

    if open_project_camera is not None:

        cam, mode = open_project_camera(_frame_queue, stop_event)

        _replace_camera(cam)

        if mode == "hardware":
            print("  Camera: ok (shared capture)", flush=True)
        else:
            print("  Camera: will retry in background (preview active)", flush=True)

        return

    for attempt in range(3):

        try:

            hw = CameraDriver()

            hw.start()

            camera = hw

            _direct_camera_fallback = True

            print('  Camera: ok (direct)', flush=True)

            return

        except Exception as e:

            print(f'  Camera: direct attempt {attempt + 1}/3 failed ({e})', flush=True)

            time.sleep(0.5)

    print('  Camera: FAILED (nvargus busy?). Stop dashboard camera or restart bot.', flush=True)





def _run_agent_thread(queued_cam):

    try:

        agent.main(queued_cam, wheels, leds, stop_event)

    except Exception:

        print('[Project] Agent thread crashed:')

        traceback.print_exc()





def _init_hardware():

    global wheels, leds, _lane_agent

    try:

        if _lane_agent is None:

            _lane_agent = LaneServoingAgent()

            _lane_agent._last_left = 0.0

            _lane_agent._last_right = 0.0

            agent.set_lane_agent(_lane_agent)

        agent.set_driving_enabled(False)

        print('\n[1/5] Initializing LED driver...')

        try:

            leds = LEDDriver()

            leds.all_off()

            print('  LEDs: ok')

        except Exception as e:

            print(f'  LEDs: not available ({e})')

            leds = None



        print('\n[2/5] Initializing wheels driver...')

        wheels = DaguWheelsDriver(WheelPWMConfiguration(), WheelPWMConfiguration())

        print('  Wheels: ok')



        print('\n[3/5] Initializing camera...')

        stop_event.clear()

        _open_camera()



        print('\n[4/5] Lane agent...')

        if _lane_agent is None:

            _lane_agent = LaneServoingAgent()

            _lane_agent._last_left = 0.0

            _lane_agent._last_right = 0.0

            agent.set_lane_agent(_lane_agent)

        agent.set_driving_enabled(False)

        print(f'  p_gain={_lane_agent.p_gain}, base_speed={_lane_agent.base_speed}')

        _hw_ready.set()

        print('\n[5/5] Starting convoy agent...')

        queued_cam = QueuedCamera(_frame_queue)

        threading.Thread(

            target=_run_agent_thread,

            args=(queued_cam,),

            daemon=True,

            name='AgentThread',

        ).start()

        print('  agent.main() running')

    except Exception:

        print('[Project] Hardware init error:')

        traceback.print_exc()

    finally:

        _hw_ready.set()





def _run_server(web_port: int):

    print(f'\nWeb UI:  http://0.0.0.0:{web_port}/')

    print(f'Video:   http://0.0.0.0:{web_port}/video')

    print('Click Start to drive. Click the Camera panel to sample HSV.\n')

    try:

        app.run(host='0.0.0.0', port=web_port, debug=False, threaded=True, use_reloader=False)

    except (KeyboardInterrupt, SystemExit):

        pass





def main():

    global camera, wheels, leds, stop_event, _lane_agent

    wheels = None

    leds = None

    camera = None

    _lane_agent = LaneServoingAgent()

    _lane_agent._last_left = 0.0

    _lane_agent._last_right = 0.0

    agent.set_lane_agent(_lane_agent)

    agent.set_driving_enabled(False)

    if PlaceholderCamera is not None:
        camera = PlaceholderCamera()
        print("[Project] Camera placeholder active (no nvargus until hardware init)", flush=True)



    ap = argparse.ArgumentParser(description='Project Server — Real Hardware')

    ap.add_argument('--port', type=int, default=5000)

    args = ap.parse_args()



    suppress_http_logs()

    print('=' * 60)

    print('PROJECT SERVER — REAL HARDWARE')

    print('=' * 60)



    web_port = find_available_port(args.port)

    if web_port != args.port:

        print(f'[Project] Port {args.port} busy, using {web_port}')



    flask_thread = threading.Thread(

        target=_run_server,

        args=(web_port,),

        name='FlaskThread',

        daemon=False,

    )

    hw_thread = threading.Thread(target=_init_hardware, name='HardwareInit', daemon=True)
    hw_thread.start()
    print(f'[Project] Web UI starting on port {web_port} (hardware init in background)...', flush=True)
    flask_thread.start()



    def _shutdown(signum, frame):

        print('\nShutting down...')

        agent.set_driving_enabled(False)

        if leds:

            try:

                leds.all_off()

                leds.release()

            except Exception:

                pass

        if camera is not None:

            camera.stop()

        if stop_project_camera is not None:

            stop_project_camera()

        shutdown_cleanup(wheels, None, stop_event)

        sys.exit(0)



    signal.signal(signal.SIGTERM, _shutdown)

    signal.signal(signal.SIGINT,  _shutdown)



    try:

        flask_thread.join()

    finally:

        agent.set_driving_enabled(False)

        if leds:

            try:

                leds.all_off()

                leds.release()

            except Exception:

                pass

        if camera is not None:

            camera.stop()

        if stop_project_camera is not None:

            stop_project_camera()

        shutdown_cleanup(wheels, None, stop_event)



    return 0





if __name__ == '__main__':

    try:

        sys.exit(main())

    except Exception:

        print('[Project] FATAL:')

        traceback.print_exc()

        sys.exit(1)


