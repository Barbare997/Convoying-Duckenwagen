from .base import render_template
from .hsv_controls import HSV_CARD_HTML, HSV_EXTRA_CSS, HSV_EXTRA_JS, HSV_VIDEO_HINT

_EXTRA_CSS = HSV_EXTRA_CSS + '''
.info-box {
    background: var(--bg-sidebar);
    border: 1px solid var(--border-color);
    border-radius: 6px;
    padding: 16px;
}
.info-box h2 { font-size: 16px; font-weight: 600; margin-bottom: 12px; }
.info-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; }
.info-item { display: flex; justify-content: space-between; padding: 8px; background: var(--bg-darker); border-radius: 4px; }
.info-label { color: var(--text-secondary); font-size: 13px; }
.control-group { margin-bottom: 20px; }
.control-group:last-child { margin-bottom: 0; }
.control-group label { display: block; margin-bottom: 8px; font-size: 14px; font-weight: 600; }
.control-row { display: flex; align-items: center; gap: 12px; }
.value-display { min-width: 60px; text-align: right; font-family: monospace; font-size: 13px; color: var(--text-secondary); }
'''

_CONTENT = f'''
    <div class="container">
        <div class="video-section video-stack">
            <img src="{{{{ url_for('video') }}}}" id="videoStream" class="stream stream-pickable" alt="Lane Servoing Stream">
{HSV_VIDEO_HINT}
        </div>

        <div class="controls-section">

{HSV_CARD_HTML}

            <!-- Drive Control card -->
            <div class="card">
                <div class="card-header">Drive Control</div>
                <div style="display:flex;align-items:center;gap:16px;margin-bottom:12px">
                    <span id="run-indicator" style="display:inline-block;width:14px;height:14px;border-radius:50%;background:#e74c3c;flex-shrink:0"></span>
                    <span id="run-label" style="font-size:14px;font-weight:600;color:var(--text-secondary)">STOPPED — camera learning lane width</span>
                </div>
                <div style="display:flex;gap:10px">
                    <button id="btn-start" onclick="driveStart()" class="button success" style="flex:1">Start</button>
                    <button id="btn-stop"  onclick="driveStop()"  class="button" style="flex:1;background:var(--accent-orange,#e67e22)">Pause</button>
                </div>
                <div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:10px">
                    <button type="button" id="btn-mode-normal" onclick="setDriveMode('CRUISING')" class="button" style="flex:1;min-width:90px">Normal</button>
                    <button type="button" id="btn-mode-slow" onclick="setDriveMode('SLOW')" class="button" style="flex:1;min-width:90px;background:#3498db;color:#fff">Slow Down</button>
                    <button type="button" id="btn-mode-sign-stop" onclick="setDriveMode('STOPPED')" class="button" style="flex:1;min-width:90px;background:#c0392b;color:#fff">Stop Sign</button>
                </div>
                <div id="drive-mode-hint" style="font-size:12px;margin-top:10px;color:var(--text-muted)">Mode: CRUISING</div>
            </div>

            <!-- Control Parameters card -->
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

        </div>
    </div>
'''

_JS = HSV_EXTRA_JS + '''
    function setRunningUI(isRunning) {
        const indicator = document.getElementById('run-indicator');
        const label     = document.getElementById('run-label');
        indicator.style.background = isRunning ? '#2ecc71' : '#e74c3c';
        label.textContent = isRunning ? 'RUNNING' : 'STOPPED — camera learning lane width';
        label.style.color = isRunning ? '#2ecc71' : 'var(--text-secondary)';
    }

    function driveStart() {
        postJSON('/start', {}).then(() => setRunningUI(true))
            .catch(() => showStatus('config-status', 'Start failed!', 'error'));
    }

    function driveStop() {
        postJSON('/stop', {}).then(() => setRunningUI(false))
            .catch(() => showStatus('config-status', 'Stop failed!', 'error'));
    }

    fetch('/running').then(r => r.json()).then(d => setRunningUI(d.running));

    let _driveMode = 'CRUISING';

    function highlightDriveMode(mode) {
        const ids = {
            CRUISING: 'btn-mode-normal',
            SLOW: 'btn-mode-slow',
            STOPPED: 'btn-mode-sign-stop',
        };
        Object.keys(ids).forEach(m => {
            const el = document.getElementById(ids[m]);
            if (el) el.style.outline = (m === mode) ? '2px solid #2ecc71' : '';
        });
        const hint = document.getElementById('drive-mode-hint');
        if (hint) hint.textContent = 'Mode: ' + mode;
    }

    function setDriveMode(mode) {
        postJSON('/drive_mode', { mode: mode })
            .then(d => {
                _driveMode = d.mode || mode;
                highlightDriveMode(_driveMode);
            })
            .catch(() => showStatus('config-status', 'Mode change failed!', 'error'));
    }

    function refreshDriveMode() {
        fetch('/status').then(r => r.json()).then(d => {
            if (d.drive_mode) {
                _driveMode = d.drive_mode;
                highlightDriveMode(_driveMode);
                const hint = document.getElementById('drive-mode-hint');
                if (hint) {
                    const cap = (d.speed_cap != null) ? ` (${Number(d.speed_cap).toFixed(2)} PWM)` : '';
                    hint.textContent = 'Mode: ' + _driveMode + cap;
                }
            }
        }).catch(() => {});
    }
    setInterval(refreshDriveMode, 1000);
    refreshDriveMode();

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
'''

LANE_SERVOING_TEMPLATE = render_template(
    'Lane Servoing — {{ hostname }}',
    '{{ hostname }} — Lane Following with Computer Vision',
    _CONTENT,
    extra_css=_EXTRA_CSS,
    extra_js=_JS,
)
