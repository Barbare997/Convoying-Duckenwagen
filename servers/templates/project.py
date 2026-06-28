from .base import render_template
from .hsv_controls import HSV_CARD_HTML, HSV_EXTRA_CSS, HSV_EXTRA_JS, HSV_VIDEO_HINT
from .project_leader_detector_ui import LEADER_DETECTOR_CARD, LEADER_DETECTOR_JS
from .project_intersection_turn_ui import INTERSECTION_TURN_CARD, INTERSECTION_TURN_JS
from .project_spacing_ui import SPACING_CARD, SPACING_JS

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
.convoy-btn-active { outline: 2px solid var(--accent-green, #2ecc71); outline-offset: 2px; }
'''

_CONTENT = f'''
    <div class="container">
        <div class="video-section video-stack">
            <img src="{{{{ url_for('video') }}}}" id="videoStream" class="stream stream-pickable" alt="Project stream">
{HSV_VIDEO_HINT}
        </div>

        <div class="controls-section">

{HSV_CARD_HTML}

{LEADER_DETECTOR_CARD}

{SPACING_CARD}

{INTERSECTION_TURN_CARD}

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
                    Follower tracks the leader's rear dot grid when visible; otherwise lane follow at cruise speed.
                </p>
                <div style="display:flex;gap:8px;flex-wrap:wrap">
                    <button type="button" id="btn-convoy-normal" onclick="convoyCommand('CRUISING')" class="button success" style="flex:1;min-width:90px">Normal</button>
                    <button type="button" id="btn-convoy-slow" onclick="convoyCommand('SLOW')" class="button" style="flex:1;min-width:90px;background:#3498db;color:#fff">Slow</button>
                    <button type="button" id="btn-convoy-stop" onclick="convoyCommand('STOPPED')" class="button" style="flex:1;min-width:90px;background:#c0392b;color:#fff">Sign Stop</button>
                </div>
                <div id="convoy-manual-hint" style="font-size:12px;margin-top:10px;color:var(--text-muted)">Command: CRUISING</div>
            </div>

            <div class="card" id="followerGridCard" style="display:none">
                <div class="card-header">Leader Tracking</div>
                <div id="gridStatusLine">Ready</div>
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

_JS = HSV_EXTRA_JS + LEADER_DETECTOR_JS + SPACING_JS + INTERSECTION_TURN_JS + '''
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
        if (hint) hint.textContent = 'Command: ' + cmd + ' (leader bot only)';
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

    function renderGridStatus(grid) {
        const el = document.getElementById('gridStatusLine');
        if (!el) return;
        if (!grid || !grid.ready) {
            el.innerHTML = '<span class="warn">Grid detector not ready</span>';
            return;
        }
        const src = grid.method || 'none';
        const seen = grid.last_found ? (src + ' seen') : 'searching…';
        const score = grid.score != null ? ' conf=' + Number(grid.score).toFixed(2) : '';
        const span = grid.span_px != null ? ' span=' + grid.span_px + 'px' : '';
        el.innerHTML = '<span class="ok">' + seen + score + span + '</span>';
    }

    function updateRolePanels(data) {
        const role = (data.role || 'leader').toLowerCase();
        const leaderCard = document.getElementById('convoySignCard');
        const gridCard = document.getElementById('followerGridCard');
        const detCard = document.getElementById('leaderDetectorCard');
        const ixCard = document.getElementById('intersectionTurnCard');
        const spCard = document.getElementById('spacingCard');
        if (leaderCard) leaderCard.style.display = role === 'leader' ? 'block' : 'none';
        if (gridCard) gridCard.style.display = role === 'follower' ? 'block' : 'none';
        if (detCard) detCard.style.display = role === 'follower' ? 'block' : 'none';
        if (spCard) spCard.style.display = role === 'follower' ? 'block' : 'none';
        if (ixCard) ixCard.style.display = role === 'follower' ? 'block' : 'none';
        if (role === 'leader') {
            const cmd = data.manual_command || _manualCommand || 'CRUISING';
            _manualCommand = cmd;
            highlightConvoyButtons(cmd);
        }
        if (role === 'follower') {
            renderGridStatus(data.grid);
            renderDetectorStatus(data.detector);
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

    function refreshConvoyStatus() {
        fetch('/status')
            .then(r => r.json())
            .then(data => {
                updateRolePanels(data);
                const rows = [
                    ['role', data.role],
                    ['state', data.state],
                    ['speed', data.speed != null ? Number(data.speed).toFixed(2) : '-'],
                    ['event', data.event || '-'],
                    ['manual', data.manual_command || '-'],
                ];
                if (data.role === 'leader') {
                    rows.push(['sign_source', data.sign_source || '-']);
                    rows.push(['apriltag', data.apriltag_available ? 'yes' : 'no']);
                    rows.push(['tags', (data.tag_ids && data.tag_ids.length) ? data.tag_ids.join(', ') : '-']);
                }
                if (data.role === 'follower') {
                    rows.push(['leader_visible', data.leader_visible ? 'yes' : 'no']);
                    rows.push(['track_source', (data.grid && data.grid.method) || '-']);
                    rows.push(['model_loaded', (data.detector && data.detector.model_loaded) ? 'yes' : 'no']);
                    rows.push(['follow_mode', data.follow_mode || '-']);
                    rows.push(['intersection', data.intersection_phase || '-']);
                    rows.push(['turn', data.intersection_turn || '-']);
                    rows.push(['red_near_px', data.red_near_px != null ? data.red_near_px : '-']);
                    rows.push(['at_line', data.red_at_line ? 'yes' : 'no']);
                    rows.push(['dist_signal', data.dist_signal != null ? Number(data.dist_signal).toFixed(3) : '-']);
                }
                document.getElementById('statusTable').innerHTML = rows.map(([k, v]) =>
                    `<div class="row"><span class="key">${k}</span><span class="val">${v}</span></div>`
                ).join('');
                document.getElementById('statusDot').style.background = 'var(--accent-green)';
            })
            .catch(() => {
                document.getElementById('statusDot').style.background = 'var(--accent-red)';
            });
    }

    refreshConvoyStatus();
    setInterval(refreshConvoyStatus, 400);
    bindSpacingSliders();
    loadSpacingParams();
'''


def get_template(title='Project', subtitle='Real Duckiebot'):
    return render_template(
        title=title,
        subtitle=subtitle,
        content_html=_CONTENT,
        extra_css=_EXTRA_CSS,
        extra_js=_JS,
    )
