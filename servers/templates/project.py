from .base import render_template

_CONTENT = '''
    <div class="container">
        <div class="video-section">
            <img src="{{ url_for('video') }}" class="stream" id="videoStream">
        </div>

        <div class="controls-section">

            <div class="card">
                <div class="card-header">
                    Status
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

_EXTRA_CSS = '''
#statusTable .row {
    display: flex;
    justify-content: space-between;
    padding: 6px 0;
    border-bottom: 1px solid var(--border-color);
    align-items: baseline;
}
#statusTable .row:last-child { border-bottom: none; }
#statusTable .key  { color: var(--text-secondary); font-size: 12px; }
#statusTable .val  { color: var(--text-primary);   font-weight: 500; font-size: 13px; font-family: monospace; }
'''

_EXTRA_JS = '''
function refreshStatus() {
    fetch('/status')
        .then(r => r.json())
        .then(data => {
            const rows = [
                ['role', data.role],
                ['state', data.state],
                ['speed', data.speed != null ? Number(data.speed).toFixed(2) : '-'],
                ['event', data.event || '-'],
                ['tags', (data.tag_ids && data.tag_ids.length) ? data.tag_ids.join(', ') : '-'],
            ];
            document.getElementById('statusTable').innerHTML = rows.map(([k, v]) =>
                `<div class="row"><span class="key">${k}</span><span class="val">${v}</span></div>`
            ).join('');
            document.getElementById('statusDot').style.background = 'var(--accent-green)';
        })
        .catch(() => {
            document.getElementById('statusDot').style.background = 'var(--accent-red)';
        });
}

refreshStatus();
setInterval(refreshStatus, 400);
'''


def get_template(title='Project', subtitle='Real Duckiebot'):
    return render_template(
        title=title,
        subtitle=subtitle,
        content_html=_CONTENT,
        extra_css=_EXTRA_CSS,
        extra_js=_EXTRA_JS,
    )
