"""Follower convoy spacing sliders for project UI."""

SPACING_CARD = '''
            <div class="card" id="spacingCard" style="display:none">
                <div class="card-header">Convoy Spacing (follower)</div>
                <p style="font-size:12px;color:var(--text-muted);margin:0 0 12px">
                    Higher detector span target = follow closer. Saved to <code>project_config.yaml</code>;
                    applies on the <strong>next frame</strong> while the leader is visible.
                </p>

                <div class="slider-group">
                    <div class="slider-label"><span>Detector span target (px)</span></div>
                    <div class="slider-controls">
                        <input type="range" id="spDetTarget" min="12" max="200" step="1" value="140" class="slider">
                        <input type="number" id="spDetTarget-input" min="12" max="200" step="1" value="140" class="input-box">
                    </div>
                </div>
                <div class="slider-group">
                    <div class="slider-label"><span>Grid span target (px)</span></div>
                    <div class="slider-controls">
                        <input type="range" id="spGridTarget" min="3" max="50" step="0.5" value="26" class="slider">
                        <input type="number" id="spGridTarget-input" min="3" max="50" step="0.5" value="26" class="input-box">
                    </div>
                </div>
                <div class="slider-group">
                    <div class="slider-label"><span>Spacing Kp</span></div>
                    <div class="slider-controls">
                        <input type="range" id="spKp" min="0.001" max="0.08" step="0.001" value="0.02" class="slider">
                        <input type="number" id="spKp-input" min="0.001" max="0.08" step="0.001" value="0.02" class="input-box">
                    </div>
                </div>
                <div class="slider-group">
                    <div class="slider-label"><span>Catch-up margin over cruise</span></div>
                    <div class="slider-controls">
                        <input type="range" id="spCatchup" min="0" max="0.35" step="0.01" value="0.18" class="slider">
                        <input type="number" id="spCatchup-input" min="0" max="0.35" step="0.01" value="0.18" class="input-box">
                    </div>
                </div>

                <div id="spacing-status" class="status"></div>
            </div>
'''

SPACING_JS = '''
    const spacingKeys = {
        'spDetTarget': 'leader_detector_span_target_px',
        'spGridTarget': 'span_target_px',
        'spKp': 'spacing_kp',
        'spCatchup': 'follower_catchup_margin',
    };

    function loadSpacingParams() {
        fetch('/get_spacing')
            .then(r => r.json())
            .then(d => {
                setSliderValue('spDetTarget', d.leader_detector_span_target_px);
                setSliderValue('spGridTarget', d.span_target_px);
                setSliderValue('spKp', d.spacing_kp);
                setSliderValue('spCatchup', d.follower_catchup_margin);
            })
            .catch(() => {});
    }

    function pushSpacingParams() {
        const payload = {};
        Object.entries(spacingKeys).forEach(([sliderId, key]) => {
            payload[key] = parseFloat(document.getElementById(sliderId).value);
        });
        postJSON('/update_spacing', payload)
            .then(d => {
                if (d.status === 'error') {
                    showStatus('spacing-status', d.message || 'Spacing save failed', 'error');
                    return;
                }
                if (d.leader_detector_span_target_px != null) {
                    setSliderValue('spDetTarget', d.leader_detector_span_target_px);
                    setSliderValue('spGridTarget', d.span_target_px);
                    setSliderValue('spKp', d.spacing_kp);
                    setSliderValue('spCatchup', d.follower_catchup_margin);
                }
                const where = d.config_path ? (' → ' + d.config_path) : '';
                showStatus('spacing-status', 'Spacing saved' + where, 'success');
            })
            .catch(() => showStatus('spacing-status', 'Spacing save failed', 'error'));
    }

    function bindSpacingSliders() {
        Object.keys(spacingKeys).forEach(id => {
            syncSliderInput(id, pushSpacingParams);
        });
    }
'''
