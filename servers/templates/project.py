from .base import render_template
from .hsv_controls import HSV_CARD_HTML, HSV_EXTRA_CSS, HSV_EXTRA_JS, HSV_VIDEO_HINT

_EXTRA_CSS = HSV_EXTRA_CSS + '''
.info-box {
    background: var(--bg-sidebar);
    border: 1px solid var(--border-color);
    border-radius: 6px;
    padding: 16px;
}
.control-group { margin-bottom: 20px; }
.control-group:last-child { margin-bottom: 0; }
.control-group label { display: block; margin-bottom: 8px; font-size: 14px; font-weight: 600; }
.control-row { display: flex; align-items: center; gap: 12px; }
.value-display { min-width: 60px; text-align: right; font-family: monospace; font-size: 13px; color: var(--text-secondary); }
#statusTable .row {
    display: flex;
    justify-content: space-between;
    padding: 6px 0;
    border-bottom: 1px solid var(--border-color);
    align-items: baseline;
}
#statusTable .row:last-child { border-bottom: none; }
#statusTable .key  { color: var(--text-secondary); font-size: 12px; }
#statusTable .val  { color: var(--text-primary); font-weight: 500; font-size: 13px; font-family: monospace; }
#statusTable .val.link-normal  { color: var(--accent-green, #2ecc71); }
#statusTable .val.link-visual  { color: #5dade2; }
#statusTable .val.link-fallback { color: var(--accent-orange, #e67e22); }
#statusTable .val.link-timeout  { color: #e74c3c; }
.convoy-btn-active { outline: 2px solid var(--accent-green, #2ecc71); outline-offset: 2px; }
#yoloStatusLine { font-size: 13px; line-height: 1.5; }
#yoloStatusLine .ok   { color: var(--accent-green, #2ecc71); font-weight: 600; }
#yoloStatusLine .warn { color: var(--accent-orange, #e67e22); font-weight: 600; }
#yoloStatusLine .err  { color: #e74c3c; font-weight: 600; }
'''

_CONTENT = f'''
    <div class="container">
        <div class="video-section video-stack">
            <img src="{{{{ url_for('video') }}}}" id="videoStream" class="stream stream-pickable" alt="Project stream">
{HSV_VIDEO_HINT}
        </div>

        <div class="controls-section">

{HSV_CARD_HTML}

            <div class="card">
                <div class="card-header">Drive Control</div>
                <p style="font-size:12px;color:var(--text-muted);margin:0 0 12px">
                    Start/Pause motors. Pause to tune HSV on the road without moving.
                </p>
                <div style="display:flex;align-items:center;gap:16px;margin-bottom:12px">
                    <span id="run-indicator" style="display:inline-block;width:14px;height:14px;border-radius:50%;background:#e74c3c;flex-shrink:0"></span>
                    <span id="run-label" style="font-size:14px;font-weight:600;color:var(--text-secondary)">PAUSED — camera learning lane width</span>
                </div>
                <div style="display:flex;gap:10px">
                    <button id="btn-start" onclick="driveStart()" class="button success" style="flex:1">Start</button>
                    <button id="btn-pause" onclick="drivePause()" class="button" style="flex:1;background:var(--accent-orange,#e67e22)">Pause</button>
                </div>
            </div>

            <div class="card" id="convoySignCard" style="display:none">
                <div class="card-header">Convoy Sign Control</div>
                <p style="font-size:12px;color:var(--text-muted);margin:0 0 12px">
                    Replaces AprilTags for now. Slow and Sign Stop auto-resume to Normal after a few seconds.
                    Follower follows leader state over HTTP. Lane follow continues during Slow.
                </p>
                <div style="display:flex;gap:8px;flex-wrap:wrap">
                    <button type="button" id="btn-convoy-normal" onclick="convoyCommand('CRUISING')" class="button success" style="flex:1;min-width:90px">Normal</button>
                    <button type="button" id="btn-convoy-slow" onclick="convoyCommand('SLOW')" class="button" style="flex:1;min-width:90px;background:#3498db;color:#fff">Slow</button>
                    <button type="button" id="btn-convoy-stop" onclick="convoyCommand('STOPPED')" class="button" style="flex:1;min-width:90px;background:#c0392b;color:#fff">Sign Stop</button>
                </div>
                <div id="convoy-manual-hint" style="font-size:12px;margin-top:10px;color:var(--text-muted)">Command: CRUISING</div>
            </div>

            <div class="card" id="followerYoloCard" style="display:none">
                <div class="card-header">Leader Spacing (YOLO)</div>
                <div id="yoloStatusLine">Checking model…</div>
            </div>

            <div class="card">
                <div class="card-header">Control Parameters</div>

                <div class="control-group">
                    <label for="k_d">Lateral Gain (k_d)</label>
                    <div class="control-row">
                        <input type="range" id="k_d" class="slider" min="0" max="1" step="0.01" value="{{{{ config.p_gain }}}}">
                        <span class="value-display" id="k_d_value">{{{{ config.p_gain }}}}</span>
                    </div>
                </div>

                <div class="control-group">
                    <label for="k_phi">Heading Gain (k_phi)</label>
                    <div class="control-row">
                        <input type="range" id="k_phi" class="slider" min="0" max="2" step="0.01" value="{{{{ config.d_gain }}}}">
                        <span class="value-display" id="k_phi_value">{{{{ config.d_gain }}}}</span>
                    </div>
                </div>

                <div class="control-group">
                    <label for="const">Base Speed <span style="color:var(--text-muted);font-weight:400;font-size:12px">(PWM 0–1)</span></label>
                    <div class="control-row">
                        <input type="range" id="const" class="slider" min="0" max="1" step="0.01" value="{{{{ config.base_speed }}}}">
                        <span class="value-display" id="const_value">{{{{ config.base_speed }}}}</span>
                    </div>
                </div>

                <button onclick="updateConfig()" class="button success">Apply Changes</button>
                <button onclick="resetPosition()" class="button" style="margin-top:8px;background:var(--accent-orange,#e67e22)">Reset Position</button>
                <div id="config-status" class="status"></div>
            </div>

            <div class="card">
                <div class="card-header">
                    Convoy Status
                    <span id="statusDot" style="width:8px;height:8px;border-radius:50%;
                        background:var(--accent-green);display:inline-block;"></span>
                </div>
                <div id="statusTable" style="font-size:12px;">
                    <div style="color:var(--text-muted);text-align:center;padding:12px 0;">
                        Waiting for data...
                    </div>
                </div>
            </div>

        </div>
    </div>
'''

_JS = HSV_EXTRA_JS + '''
    let _manualCommand = 'CRUISING';

    function setRunningUI(isRunning) {
        const indicator = document.getElementById('run-indicator');
        const label     = document.getElementById('run-label');
        indicator.style.background = isRunning ? '#2ecc71' : '#e74c3c';
        label.textContent = isRunning ? 'RUNNING — lane follow active' : 'PAUSED — camera learning lane width';
        label.style.color = isRunning ? '#2ecc71' : 'var(--text-secondary)';
    }

    function driveStart() {
        postJSON('/start', {}).then(() => setRunningUI(true))
            .catch(() => showStatus('config-status', 'Start failed!', 'error'));
    }

    function drivePause() {
        postJSON('/stop', {}).then(() => setRunningUI(false))
            .catch(() => showStatus('config-status', 'Pause failed!', 'error'));
    }

    fetch('/running').then(r => r.json()).then(d => setRunningUI(d.running)).catch(() => {});

    function highlightConvoyButtons(cmd) {
        const map = {
            CRUISING: 'btn-convoy-normal',
            SLOW: 'btn-convoy-slow',
            STOPPED: 'btn-convoy-stop',
        };
        Object.values(map).forEach(id => {
            const el = document.getElementById(id);
            if (el) el.classList.remove('convoy-btn-active');
        });
        const active = document.getElementById(map[cmd] || map.CRUISING);
        if (active) active.classList.add('convoy-btn-active');
        const hint = document.getElementById('convoy-manual-hint');
        if (hint) hint.textContent = 'Command: ' + cmd + ' (follower sees leader state over HTTP)';
    }

    function convoyCommand(cmd) {
        postJSON('/convoy/manual', { command: cmd })
            .then(() => {
                _manualCommand = cmd;
                highlightConvoyButtons(cmd);
                showStatus('config-status', 'Convoy: ' + cmd, 'success');
            })
            .catch(() => showStatus('config-status', 'Convoy command failed!', 'error'));
    }

    function renderYoloStatus(yolo) {
        const el = document.getElementById('yoloStatusLine');
        if (!el) return;
        if (!yolo || !yolo.enabled) {
            el.innerHTML = '<span class="warn">YOLO spacing disabled in project_config.yaml</span>';
            return;
        }
        if (yolo.pending) {
            let msg = 'Loading spacing model…';
            if (yolo.trt_building) {
                const sec = yolo.trt_elapsed_s != null ? yolo.trt_elapsed_s : 0;
                msg = 'TensorRT compiling (~1 min)… ' + sec + 's elapsed';
            }
            el.innerHTML = '<span class="warn">' + msg + '</span>';
            return;
        }
        if (yolo.ready) {
            el.innerHTML = '<span class="ok">YOLO truck model loaded — leader spacing active</span>';
            return;
        }
        const err = yolo.error ? String(yolo.error) : 'unknown error';
        el.innerHTML = '<span class="err">YOLO not loaded</span><br><span style="font-size:11px;color:var(--text-muted)">' + err + '</span>';
    }

    function updateRolePanels(data) {
        const role = (data.role || 'leader').toLowerCase();
        const leaderCard = document.getElementById('convoySignCard');
        const yoloCard = document.getElementById('followerYoloCard');
        if (leaderCard) leaderCard.style.display = role === 'leader' ? 'block' : 'none';
        if (yoloCard) yoloCard.style.display = role === 'follower' ? 'block' : 'none';
        if (role === 'leader') {
            const cmd = data.manual_command || _manualCommand || 'CRUISING';
            _manualCommand = cmd;
            highlightConvoyButtons(cmd);
        }
        if (role === 'follower') {
            renderYoloStatus(data.yolo);
        }
    }

    document.getElementById('k_d').oninput = function() {
        document.getElementById('k_d_value').textContent = this.value;
    };
    document.getElementById('k_phi').oninput = function() {
        document.getElementById('k_phi_value').textContent = this.value;
    };
    document.getElementById('const').oninput = function() {
        document.getElementById('const_value').textContent = this.value;
    };

    function resetPosition() {
        postJSON('/reset', {})
            .then(() => showStatus('config-status', 'Position Reset!', 'success'))
            .catch(() => showStatus('config-status', 'Reset Failed!', 'error'));
    }

    function updateConfig() {
        postJSON('/update_config', {
            k_d:   parseFloat(document.getElementById('k_d').value),
            k_phi: parseFloat(document.getElementById('k_phi').value),
            const: parseFloat(document.getElementById('const').value)
        })
        .then(() => showStatus('config-status', 'Config Updated!', 'success'))
        .catch(() => showStatus('config-status', 'Update Failed!', 'error'));
    }

    function linkPhaseClass(phase) {
        if (phase === 'visual') return 'link-visual';
        if (phase === 'fallback') return 'link-fallback';
        if (phase === 'timeout') return 'link-timeout';
        return 'link-normal';
    }

    function refreshConvoyStatus() {
        fetch('/status')
            .then(r => r.json())
            .then(data => {
                updateRolePanels(data);
                const role = (data.role || 'leader').toLowerCase();
                const rows = [];
                if (role === 'follower') {
                    rows.push(['spacing_mode', data.follower_spacing_mode || 'http']);
                    rows.push(['follower_mode', data.follower_mode || data.state || '-']);
                    rows.push(['http_link', data.http_link_label || data.http_link || '-']);
                    rows.push(['http_age_s', data.http_age_s != null ? Number(data.http_age_s).toFixed(2) : '-']);
                    rows.push(['target_speed', data.target_speed != null ? Number(data.target_speed).toFixed(2) : '-']);
                    rows.push(['cmd_speed', data.speed != null ? Number(data.speed).toFixed(2) : '-']);
                    rows.push(['yolo_signal', data.distance_signal != null ? Number(data.distance_signal).toFixed(3) : '-']);
                } else {
                    rows.push(['role', data.role]);
                    rows.push(['state', data.state]);
                    rows.push(['speed', data.speed != null ? Number(data.speed).toFixed(2) : '-']);
                    rows.push(['event', data.event || '-']);
                    rows.push(['manual', data.manual_command || '-']);
                }
                if (role === 'leader') {
                    rows.push(['sign_source', data.sign_source || '-']);
                    rows.push(['apriltag', data.apriltag_available ? 'yes' : 'no']);
                    rows.push(['tags', (data.tag_ids && data.tag_ids.length) ? data.tag_ids.join(', ') : '-']);
                }
                if (role === 'follower' && data.leader) {
                    rows.push(['leader_state', data.leader.state || '-']);
                    rows.push(['leader_speed', data.leader.speed != null ? Number(data.leader.speed).toFixed(2) : '-']);
                    const leaderDrv = data.leader.driving_enabled;
                    rows.push([
                        'leader_drive',
                        leaderDrv === true ? 'RUNNING' : (leaderDrv === false ? 'PAUSED' : '-'),
                    ]);
                    rows.push(['leader_cmd', data.leader.manual_command || '-']);
                }
                document.getElementById('statusTable').innerHTML = rows.map(([k, v]) => {
                    const cls = k === 'http_link' ? linkPhaseClass(data.http_link) : '';
                    return `<div class="row"><span class="key">${k}</span><span class="val ${cls}">${v}</span></div>`;
                }).join('');
                document.getElementById('statusDot').style.background = 'var(--accent-green)';
            })
            .catch(() => {
                document.getElementById('statusDot').style.background = 'var(--accent-red)';
            });
    }

    refreshConvoyStatus();
    setInterval(refreshConvoyStatus, 400);
'''


def get_template(title='Project', subtitle='Real Duckiebot'):
    return render_template(
        title=title,
        subtitle=subtitle,
        content_html=_CONTENT,
        extra_css=_EXTRA_CSS,
        extra_js=_JS,
    )
