import os
import sqlite3
from html import escape
from urllib.parse import quote
from flask import Flask, request, Response, abort, render_template_string
from twilio.twiml.voice_response import VoiceResponse, Connect, Gather, Play
from twilio.request_validator import RequestValidator
from dotenv import load_dotenv
import stats

load_dotenv()

app = Flask(__name__)

TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
VALIDATE_TWILIO = os.getenv("VALIDATE_TWILIO_SIGNATURE", "true").lower() == "true"

# Resolve database path to pull metrics directly for the dashboard
if os.path.exists("/data"):
    DB_PATH = "/data/voice_assistant.db"
else:
    DB_PATH = os.getenv("SQLITE_DB_PATH", "voice_assistant.db")

LANG_MAP = {
    # Strict IVR enforcement: only these three digits may enter the voice pipeline.
    "1": ("en", "English"),
    "2": ("fr", "French"),
    "3": ("sw", "Swahili"),
}

HELP_GREETINGS = {
    "en": "How can I help you?",
    "fr": "Comment puis-je vous aider?",
    "sw": "Nawezaje kukusaidia?",
}

LANGUAGE_SELECTED_PROMPTS = {
    "en": "English selected. How can I help you?",
    "fr": "French selected. Comment puis-je vous aider?",
    "sw": "Swahili selected. Nawezaje kukusaidia?",
}

IVR_GREETING = (
    "Hello! Please select your language. "
    "Press 1 for English. "
    "Appuyez sur 2 pour le français. "
    "Bonyeza 3 kwa Kiswahili."
)

IVR_INVALID = (
    "Sorry, that was not a valid option. Please try again. "
    "Press 1 for English, 2 for French, or 3 for Swahili."
)

# Static IVR audio generated offline with Soniox Grace voice.
# These files must exist under: static/ivr/
#   static/ivr/ivr.wav
#   static/ivr/ivr_invalid.wav
#
# Only the IVR menu uses static files. After a valid digit is selected,
# routes.py connects directly to the WebSocket. The app pipeline should then
# speak HELP_GREETINGS[lang] through TTSModule + PhoneAudioOutput, not provider TTS.
IVR_GREETING_AUDIO_FILE = "ivr.wav"
IVR_INVALID_AUDIO_FILE = "ivr_invalid.wav"


def _public_base_url() -> str:
    """Return the HTTPS public base URL for Twilio/Telnyx callbacks and static media."""
    host = _public_host()
    scheme = os.getenv("PUBLIC_SCHEME", "https").strip() or "https"
    return f"{scheme}://{host}"


def _static_ivr_url(filename: str) -> str:
    """Return an absolute URL to a static IVR audio file."""
    return f"{_public_base_url()}/static/ivr/{quote(filename, safe='')}"


def _telnyx_play(filename: str) -> str:
    """Return a Telnyx TeXML <Play> tag for a static IVR audio file."""
    return f"<Play>{escape(_static_ivr_url(filename))}</Play>"


def _caller_param_from_request() -> str:
    """Return a URL-safe caller value for media-stream query parameters."""
    return quote(request.form.get("From", "UNKNOWN"), safe="")


def _media_stream_url(provider: str, lang: str) -> str:
    """Build the WebSocket URL used by Twilio and Telnyx media streams.

    The InitialGreeting is passed to the WebSocket so sockets.py/VoiceAssistant
    can speak it through TTSModule. Do not use Twilio/Telnyx <Say> after IVR.
    """
    from_param = _caller_param_from_request()
    greeting_text = LANGUAGE_SELECTED_PROMPTS.get(lang, LANGUAGE_SELECTED_PROMPTS["en"])
    greeting_param = quote(greeting_text, safe="")
    return (
        f"wss://{_public_host()}/media-stream/{provider}/{lang}"
        f"?From={from_param}&InitialGreeting={greeting_param}"
    )


def _twilio_audio_gather(audio_filename: str, action: str = "/twilio/language") -> Gather:
    """Play a static WAV inside <Gather> so Twilio still collects 1/2/3."""
    gather = Gather(
        num_digits=1,
        action=action,
        method="POST",
        timeout=int(os.getenv("IVR_GATHER_TIMEOUT", "3")),
        finish_on_key="",
    )
    gather.append(Play(_static_ivr_url(audio_filename)))
    return gather


def _twilio_language_menu_response(audio_filename: str = IVR_GREETING_AUDIO_FILE,
                                   redirect_to: str = "/twilio/voice") -> Response:
    """Return TwiML that plays a static IVR WAV and collects a digit."""
    resp = VoiceResponse()
    resp.append(_twilio_audio_gather(audio_filename))
    resp.redirect(redirect_to, method="POST")
    return Response(str(resp), mimetype="text/xml")


def _twilio_stream_response(lang: str) -> Response:
    """Connect the call to the Twilio stream immediately after IVR selection."""
    resp = VoiceResponse()
    connect = Connect()
    connect.stream(url=_media_stream_url("twilio", lang))
    resp.append(connect)
    return Response(str(resp), mimetype="text/xml")


def _telnyx_language_menu_response(audio_filename: str = IVR_GREETING_AUDIO_FILE) -> Response:
    """Return TeXML that plays a static IVR WAV and collects a digit."""
    host = escape(_public_host())
    gather_body = _telnyx_play(audio_filename)

    texml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Gather numDigits="1" action="https://{host}/telnyx/language" method="POST" timeout="{escape(os.getenv('IVR_GATHER_TIMEOUT', '3'))}">
        {gather_body}
    </Gather>
    <Redirect method="POST">https://{host}/telnyx/voice</Redirect>
</Response>"""
    return Response(texml, mimetype="text/xml")


def _telnyx_stream_response(lang: str) -> Response:
    """Connect the call to the Telnyx stream immediately after IVR selection."""
    host = escape(_public_host())
    stream_url = escape(_media_stream_url("telnyx", lang))

    texml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Connect>
        <Stream
            url="{stream_url}"
            bidirectionalMode="rtp"
            statusCallback="https://{host}/telnyx/stream-status"
            statusCallbackMethod="POST" />
    </Connect>
</Response>"""

    print("[Telnyx TeXML /language]", texml)
    return Response(texml, mimetype="text/xml")


def _get_db_connection():
    """Helper to open a quick, read-only sync connection for Flask metrics."""
    conn = sqlite3.connect(DB_PATH, timeout=5.0)
    conn.row_factory = sqlite3.Row
    return conn


def _public_host() -> str:
    return os.getenv("PUBLIC_HOST") or request.host


def _validate_twilio():
    if VALIDATE_TWILIO and TWILIO_AUTH_TOKEN:
        validator = RequestValidator(TWILIO_AUTH_TOKEN)
        if not validator.validate(
                request.url,
                request.form,
                request.headers.get("X-Twilio-Signature", ""),
        ):
            abort(403, description="Invalid Twilio signature")


def _format_language_counts(counts: dict) -> str:
    """Return public-friendly language counts for dashboard display."""
    return (
        f"English: {counts.get('en', 0)} &nbsp;|&nbsp; "
        f"French: {counts.get('fr', 0)} &nbsp;|&nbsp; "
        f"Swahili: {counts.get('sw', 0)}"
    )


# ===========================================================================
# DASHBOARD — shell page
# ===========================================================================

DASHBOARD_SHELL = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Voice Assistant — System Dashboard</title>
    <script src="https://unpkg.com/htmx.org@1.9.10"></script>
    <style>
        * { box-sizing: border-box; }
        body { font-family: 'Courier New', monospace; margin: 0; display: flex; height: 100vh; color: #1a1a1a; }
        #sidebar { width: 180px; flex-shrink: 0; border-right: 1px solid #ccc; padding: 1rem 0; background: #fafafa; transition: margin-left 0.2s ease; }
        #sidebar.collapsed { margin-left: -180px; }
        #sidebar strong { display: block; padding: 0.5rem 1rem; border-bottom: 1px solid #eee; margin-bottom: 0.5rem; font-size: 0.85rem; letter-spacing: 0.02em; }
        #sidebar a { display: block; padding: 0.5rem 1rem; text-decoration: none; color: #333; font-size: 0.85rem; }
        #sidebar a:hover { background: #f0f0f0; color: #000; }
        #main { flex: 1; overflow-y: auto; padding: 1rem; scroll-behavior: smooth; }

        #topbar { display: flex; align-items: center; margin-bottom: 0.75rem; }
        #hamburger-btn { width: 32px; height: 32px; border: 1px solid #e2e2e2; background: #fff; cursor: pointer; display: flex; align-items: center; justify-content: center; padding: 0; }
        #hamburger-btn:hover { background: #f5f5f5; }
        #hamburger-btn span { display: block; width: 16px; height: 2px; background: #444; position: relative; }
        #hamburger-btn span::before, #hamburger-btn span::after { content: ""; position: absolute; left: 0; width: 16px; height: 2px; background: #444; }
        #hamburger-btn span::before { top: -5px; }
        #hamburger-btn span::after { top: 5px; }

        .section { border: 1px solid #e2e2e2; margin-bottom: 0.75rem; padding: 0.75rem 1rem; background: #fff; }
        .section h2 { margin: 0 0 0.5rem 0; font-size: 0.8rem; font-weight: bold; letter-spacing: 0.04em; text-transform: uppercase; color: #444; }
        .section h2 small { text-transform: none; font-weight: normal; letter-spacing: normal; color: #999; font-size: 0.75rem; }

        .live-dot { width: 6px; height: 6px; border-radius: 50%; background: #22c55e; display: inline-block; margin-left: 6px; }

        table { border-collapse: collapse; width: 100%; font-size: 0.82rem; }
        td, th { padding: 3px 0; text-align: left; border-bottom: 1px solid #f0f0f0; }
        th { font-weight: bold; color: #777; background: none; text-transform: uppercase; font-size: 0.7rem; letter-spacing: 0.03em; }
        tr:last-child td { border-bottom: none; }

        .transcript-box { max-height: 200px; overflow-y: auto; background: #fff; padding: 0; font-size: 0.8rem; }
        .user-turn, .assistant-turn { margin: 0; padding: 3px 0; border-bottom: 1px solid #f5f5f5; }
        .user-turn strong, .assistant-turn strong { font-weight: bold; text-transform: lowercase; }
        .user-turn strong { color: #0066cc; }
        .assistant-turn strong { color: #cc3300; }

        .log-box { max-height: 250px; overflow-y: auto; background: #222; color: #22c55e; padding: 0.6rem; font-size: 0.78rem; }
        .log-line { margin: 0; border-bottom: 1px solid #2d2d2d; padding: 3px 0; }
        .log-time { color: #a3a3a3; font-size: 0.72rem; margin-right: 8px; }
        .log-dev { color: #38bdf8; margin-left: 6px; font-size: 0.72rem; }
        .language-summary { line-height: 1.6; }
    </style>
</head>
<body>

<nav id="sidebar">
    <strong>Dashboard</strong>
    <a href="#section-calls">Call Metrics</a>
    <a href="#section-live-feed">Live Transcript Feed</a>
    <a href="#section-latency">Pipeline Latency</a>
    <a href="#section-resources">System Resources</a>
    <a href="#section-concurrency">Concurrency</a>
    <a href="#section-transcripts">Live Transcripts</a>
    <a href="#section-system-logs">⚙️ System Logs</a> </nav>

<div id="main">

    <div id="topbar">
        <button id="hamburger-btn" aria-label="Toggle sidebar" onclick="document.getElementById('sidebar').classList.toggle('collapsed')"><span></span></button>
    </div>

    <div id="section-calls" hx-get="/dashboard/calls" hx-trigger="load, every 2s" hx-swap="innerHTML">
        Loading call metrics...
    </div>

    <div id="section-live-feed" hx-get="/dashboard/live-feed" hx-trigger="load, every 2s" hx-swap="innerHTML">
        Loading live transcript feed...
    </div>

    <div id="section-latency" hx-get="/dashboard/latency" hx-trigger="load, every 2s" hx-swap="innerHTML">
        Loading latency...
    </div>


    <div id="section-resources" hx-get="/dashboard/resources" hx-trigger="load, every 2s" hx-swap="innerHTML">
        Loading system resources...
    </div>

    <div id="section-concurrency" hx-get="/dashboard/concurrency" hx-trigger="load, every 2s" hx-swap="innerHTML">
        Loading concurrency info...
    </div>

    <div id="section-transcripts" hx-get="/dashboard/transcripts" hx-trigger="load, every 3s" hx-swap="innerHTML">
        Loading persistent conversations...
    </div>

    <div id="section-system-logs" hx-get="/dashboard/system-logs" hx-trigger="load, every 5s" hx-swap="innerHTML">
        Loading initialization and lifecycle logs from database...
    </div>

</div>
</body>
</html>
"""


@app.route("/", methods=["GET"])
def dashboard():
    return render_template_string(DASHBOARD_SHELL)


# ===========================================================================
# DASHBOARD FRAGMENTS
# ===========================================================================

@app.route("/dashboard/calls", methods=["GET"])
def dashboard_calls():
    s = stats.get_cached("calls_snap")

    try:
        with _get_db_connection() as conn:
            historical_total = conn.execute("SELECT COUNT(*) as cnt FROM call_logs").fetchone()["cnt"]
            failed_total = conn.execute("SELECT COUNT(*) as cnt FROM call_logs WHERE status='failed'").fetchone()["cnt"]
    except Exception:
        historical_total = s["total"]
        failed_total = s["failed"]

    html = """
    <div class="section">
        <h2>Call Metrics</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Active calls</td><td>{active_count}</td></tr>
            <tr><td>Total Historical Records (SQLite)</td><td>{total}</td></tr>
            <tr><td>Total Persistent Failed Logs</td><td>{failed}</td></tr>
            <tr><td>Peak concurrent</td><td>{peak}</td></tr>
            <tr><td>Avg call duration</td><td>{avg_dur}</td></tr>
            <tr><td>Calls by language</td><td class="language-summary">{language_counts}</td></tr>
        </table>
    </div>
    """.format(
        active_count=s["active_count"],
        total=historical_total,
        failed=failed_total,
        peak=s["peak_concurrent"],
        avg_dur=f"{s['avg_duration_s']}s" if s["avg_duration_s"] else "—",
        language_counts=_format_language_counts(s["by_lang"]),
    )
    return html


@app.route("/dashboard/transcripts", methods=["GET"])
def dashboard_transcripts():
    html_lines = ["<div class='section'><h2>Recent Voice Conversations</h2>"]

    try:
        with _get_db_connection() as conn:
            sessions = conn.execute("""
                                    SELECT DISTINCT session_id
                                    FROM transcript_logs
                                    ORDER BY id DESC LIMIT 3
                                    """).fetchall()

            if not sessions:
                html_lines.append(
                    "<p style='color:gray;'>No recorded text exchanges yet. Speak into the line to populate.</p>")

            for sess in sessions:
                sid = sess["session_id"]
                html_lines.append(f"<strong>Session: {sid}</strong>")
                html_lines.append("<div class='transcript-box'>")

                turns = conn.execute("""
                                     SELECT role, text, timestamp
                                     FROM transcript_logs
                                     WHERE session_id = ?
                                     ORDER BY timestamp ASC
                                     """, (sid,)).fetchall()

                for turn in turns:
                    cls = "user-turn" if turn["role"] == "user" else "assistant-turn"
                    label = "👤 User (Whisper)" if turn["role"] == "user" else "🤖 Assistant"
                    html_lines.append(f"<p class='{cls}'><strong>{label}:</strong> {escape(str(turn['text']))}</p>")

                html_lines.append("</div><br>")
    except Exception as e:
        html_lines.append(f"<p style='color:red;'>Failed to load transcripts from DB: {e}</p>")

    html_lines.append("</div>")
    return "".join(html_lines)


@app.route("/dashboard/latency", methods=["GET"])
def dashboard_latency():
    lat = stats.get_cached("latency")

    def fmt(v):
        return f"{v}s" if v is not None else "—"

    html = """
    <div class="section">
        <h2>Pipeline Latency <small>(rolling avg, last 20)</small></h2>
        <table>
            <tr><th>Stage</th><th>Avg Latency</th></tr>
            <tr><td>STT (audio → transcript)</td><td>{stt}</td></tr>
            <tr><td>Response generation (transcript → first token)</td><td>{llm}</td></tr>
            <tr><td>TTS (first token → first audio)</td><td>{tts}</td></tr>
            <tr><td>End-to-end</td><td>{e2e}</td></tr>
        </table>
    </div>
    """.format(
        stt=fmt(lat["stt_avg_s"]),
        llm=fmt(lat["llm_avg_s"]),
        tts=fmt(lat["tts_avg_s"]),
        e2e=fmt(lat["e2e_avg_s"]),
    )
    return html


@app.route("/dashboard/models", methods=["GET"])
def dashboard_models():
    """Model details are intentionally hidden from the public dashboard."""
    return ""


@app.route("/dashboard/resources", methods=["GET"])
def dashboard_resources():
    r = stats.get_cached("resources")

    scope_warning = ""
    if not r.get("scoped", True):
        scope_warning = (
            "<tr><td colspan='2' style='color:orange'>"
            "⚠️ cgroup unavailable — showing host-level figures, not container"
            "</td></tr>"
        )

    ram_total = r["ram_total_gb"]
    ram_pct = r["ram_pct"]
    ram_str = (
        f"{r['ram_used_gb']} GB / {ram_total} GB ({ram_pct}%)"
        if ram_total else
        f"{r['ram_used_gb']} GB (limit unknown)"
    )

    if r["gpu_total_mb"] is not None:
        gpu_rows = """
            <tr><td>GPU VRAM used</td><td>{used} MB / {total} MB ({pct}%)</td></tr>
        """.format(
            used=r["gpu_used_mb"],
            total=r["gpu_total_mb"],
            pct=r["gpu_pct"],
        )
    else:
        gpu_rows = "<tr><td>GPU VRAM</td><td>— (CPU only)</td></tr>"

    html = """
    <div class="section">
        <h2>System Resources <small>(container-scoped)</small></h2>
        <table>
            <tr><th>Resource</th><th>Value</th></tr>
            {scope_warning}
            <tr><td>RAM used</td><td>{ram_str}</td></tr>
            {gpu_rows}
            <tr><td>CPU usage</td><td>{cpu}%</td></tr>
        </table>
    </div>
    """.format(
        scope_warning=scope_warning,
        ram_str=ram_str,
        gpu_rows=gpu_rows,
        cpu=r["cpu_pct"],
    )
    return html


@app.route("/dashboard/concurrency", methods=["GET"])
def dashboard_concurrency():
    c = stats.get_cached("concurrency")
    tts_c = c.get("tts_contention", {})

    html = """
    <div class="section">
        <h2>Concurrency</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Active calls</td><td>{active}</td></tr>
            <tr><td>Peak concurrent calls</td><td>{peak}</td></tr>
            <tr><td>Whisper queue depth</td><td>{whisper_q}</td></tr>
            <tr><td>TTS contention by language</td><td class="language-summary">{tts_contention}</td></tr>
        </table>
    </div>
    """.format(
        active=c.get("active_count", 0),
        peak=c.get("peak_concurrent", 0),
        whisper_q=c.get("whisper_queue", 0),
        tts_contention=_format_language_counts(tts_c),
    )
    return html


@app.route("/dashboard/live-feed", methods=["GET"])
def dashboard_live_feed():
    """Compact live view of the most recent session's transcript turns,
    sitting between Concurrency and System Logs on the dashboard."""
    html_lines = [
        "<div class='section'>",
        "<h2>Live transcript feed<span class='live-dot'></span> <small>(most recent session)</small></h2>",
    ]

    try:
        with _get_db_connection() as conn:
            latest_session = conn.execute("""
                                          SELECT session_id
                                          FROM transcript_logs
                                          ORDER BY id DESC LIMIT 1
                                          """).fetchone()

            if not latest_session:
                html_lines.append(
                    "<p style='color:gray; font-size:0.85rem;'>No active session yet.</p>")
            else:
                sid = latest_session["session_id"]
                turns = conn.execute("""
                                     SELECT role, text, timestamp
                                     FROM transcript_logs
                                     WHERE session_id = ?
                                     ORDER BY timestamp DESC LIMIT 8
                                     """, (sid,)).fetchall()
                turns = list(reversed(turns))

                html_lines.append("<div class='transcript-box'>")
                for turn in turns:
                    cls = "user-turn" if turn["role"] == "user" else "assistant-turn"
                    label = "user" if turn["role"] == "user" else "assistant"
                    html_lines.append(f"<p class='{cls}'><strong>{label}</strong> &nbsp;{escape(str(turn['text']))}</p>")
                html_lines.append("</div>")
    except Exception as e:
        html_lines.append(f"<p style='color:red; font-size:0.85rem;'>Failed to load live feed: {e}</p>")

    html_lines.append("</div>")
    return "".join(html_lines)


# ===========================================================================
# Raw JSON endpoint
# ===========================================================================

@app.route("/metrics", methods=["GET"])
def metrics():
    import json
    return Response(json.dumps(stats.snapshot(), indent=2), mimetype="application/json")


# ===========================================================================
# TWILIO
# ===========================================================================

@app.route("/twilio/voice", methods=["POST"])
def twilio_voice():
    _validate_twilio()
    return _twilio_language_menu_response(IVR_GREETING_AUDIO_FILE)


@app.route("/twilio/language", methods=["POST"])
def twilio_language():
    _validate_twilio()

    digit = (request.form.get("Digits", "") or "").strip()
    if digit not in LANG_MAP:
        return _twilio_language_menu_response(IVR_INVALID_AUDIO_FILE)

    lang, lang_name = LANG_MAP[digit]
    print(f"[Twilio IVR] Caller selected: {lang_name} ({lang})")
    return _twilio_stream_response(lang)


# ===========================================================================
# TELNYX
# ===========================================================================

@app.route("/telnyx/voice", methods=["POST"])
def telnyx_voice():
    return _telnyx_language_menu_response(IVR_GREETING_AUDIO_FILE)


@app.route("/telnyx/language", methods=["POST"])
def telnyx_language():
    digit = (request.form.get("Digits", "") or "").strip()
    if digit not in LANG_MAP:
        return _telnyx_language_menu_response(IVR_INVALID_AUDIO_FILE)

    lang, lang_name = LANG_MAP[digit]
    print(f"[Telnyx IVR] Caller selected: {lang_name} ({lang})")
    return _telnyx_stream_response(lang)


@app.route("/telnyx/stream-status", methods=["POST"])
def telnyx_stream_status():
    print("[Telnyx Stream Status]", dict(request.form))
    return Response("", status=204)


# ===========================================================================
# Health check
# ===========================================================================

@app.route("/health", methods=["GET"])
def health():
    return {"status": "ok"}, 200


@app.route("/dashboard/system-logs", methods=["GET"])
def dashboard_system_logs():
    """NEW COMPATIBLE ENDPOINT: Reads model startup performance profiles from the SQLite database."""
    html_lines = ["<div class='section'><h2>System Logs (Database Startup Lifecycle)</h2>"]
    html_lines.append("<div class='log-box'>")

    try:
        with _get_db_connection() as conn:
            logs = conn.execute("""
                                SELECT model_name, duration_s, device, loaded_at
                                FROM model_load_logs
                                ORDER BY id DESC LIMIT 15
                                """).fetchall()

            if not logs:
                html_lines.append(
                    "<div class='log-line'><span class='log-time'>—</span> No hardware configuration records found in SQLite.</div>")

            for log in logs:
                timestamp_str = log["loaded_at"]
                if hasattr(timestamp_str, "strftime"):
                    timestamp_str = timestamp_str.strftime("%Y-%m-%d %H:%M:%S UTC")

                html_lines.append(
                    f"<div class='log-line'>"
                    f"<span class='log-time'>[{escape(str(timestamp_str))}]</span>"
                    f"INIT_STAGE: Component <strong style='color:#fff;'>{escape(str(log['model_name']))}</strong> "
                    f"allocated inside engine space in <strong style='color:#facc15;'>{log['duration_s']}s</strong>"
                    f"<span class='log-dev'>[Target: {escape(str(log['device']).upper())}]</span>"
                    f"</div>"
                )
    except Exception as e:
        html_lines.append(
            f"<div class='log-line' style='color:#ef4444;'>Failed to query database engine logs: {e}</div>")

    html_lines.append("</div></div>")
    return "".join(html_lines)