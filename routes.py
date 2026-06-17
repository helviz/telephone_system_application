import os
import sqlite3
from flask import Flask, request, Response, abort, render_template_string
from twilio.twiml.voice_response import VoiceResponse, Connect, Gather
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
    "1": ("en", "English"),
    "2": ("fr", "French"),
    "3": ("sw", "Swahili"),
}

HELP_GREETINGS = {
    "en": "How can I help you?",
    "fr": "Comment puis-je vous aider?",
    "sw": "Nawezaje kukusaidia?",
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
        body { font-family: monospace; margin: 0; display: flex; height: 100vh; }
        #sidebar { width: 180px; flex-shrink: 0; border-right: 1px solid #ccc; padding: 1rem 0; background: #fafafa; }
        #sidebar strong { display: block; padding: 0.5rem 1rem; border-bottom: 1px solid #eee; margin-bottom: 0.5rem; }
        #sidebar a { display: block; padding: 0.5rem 1rem; text-decoration: none; color: #333; }
        #sidebar a:hover { background: #f0f0f0; color: #000; }
        #main { flex: 1; overflow-y: auto; padding: 1rem; scroll-behavior: smooth; }
        .section { border: 1px solid #ccc; margin-bottom: 1.5rem; padding: 1rem; background: #fff; box-shadow: 1px 1px 3px rgba(0,0,0,0.05); }
        .section h2 { margin: 0 0 0.75rem 0; font-size: 1rem; border-bottom: 1px solid #eee; padding-bottom: 0.25rem; }
        table { border-collapse: collapse; width: 100%; font-size: 0.9rem; }
        td, th { padding: 0.4rem 0.5rem; text-align: left; border-bottom: 1px solid #eee; }
        th { font-weight: bold; background: #f5f5f5; }
        .transcript-box { max-height: 200px; overflow-y: auto; background: #f9f9f9; padding: 0.5rem; border: 1px solid #eee; font-size: 0.85rem; }
        .user-turn { color: #0066cc; margin: 4px 0; }
        .assistant-turn { color: #cc3300; margin: 4px 0; }
        .log-box { max-height: 250px; overflow-y: auto; background: #222; color: #22c55e; padding: 0.75rem; border-radius: 4px; font-size: 0.85rem; }
        .log-line { margin: 2px 0; border-bottom: 1px solid #2d2d2d; padding-bottom: 2px; }
        .log-time { color: #a3a3a3; font-size: 0.75rem; margin-right: 8px; }
        .log-dev { color: #38bdf8; margin-left: 6px; font-size: 0.75rem; }
    </style>
</head>
<body>

<nav id="sidebar">
    <strong>Dashboard</strong>
    <a href="#section-calls">Call Metrics</a>
    <a href="#section-transcripts">Live Transcripts</a>
    <a href="#section-latency">Pipeline Latency</a>
    <a href="#section-models">Model Health</a>
    <a href="#section-resources">System Resources</a>
    <a href="#section-concurrency">Concurrency</a>
    <a href="#section-system-logs">⚙️ System Logs</a> </nav>

<div id="main">

    <div id="section-calls" hx-get="/dashboard/calls" hx-trigger="load, every 2s" hx-swap="innerHTML">
        Loading call metrics...
    </div>

    <div id="section-transcripts" hx-get="/dashboard/transcripts" hx-trigger="load, every 3s" hx-swap="innerHTML">
        Loading persistent conversations...
    </div>

    <div id="section-latency" hx-get="/dashboard/latency" hx-trigger="load, every 2s" hx-swap="innerHTML">
        Loading latency...
    </div>

    <div id="section-models" hx-get="/dashboard/models" hx-trigger="load, every 10s" hx-swap="innerHTML">
        Loading model info...
    </div>

    <div id="section-resources" hx-get="/dashboard/resources" hx-trigger="load, every 2s" hx-swap="innerHTML">
        Loading system resources...
    </div>

    <div id="section-concurrency" hx-get="/dashboard/concurrency" hx-trigger="load, every 2s" hx-swap="innerHTML">
        Loading concurrency info...
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
    active = s.get("active", [])

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
            <tr><td>Active calls (in memory)</td><td>{active_count}</td></tr>
            <tr><td>&nbsp;&nbsp;↳ Twilio</td><td>{twilio_active}</td></tr>
            <tr><td>&nbsp;&nbsp;↳ Telnyx</td><td>{telnyx_active}</td></tr>
            <tr><td>Total Historical Records (SQLite)</td><td>{total}</td></tr>
            <tr><td>Total Persistent Failed Logs</td><td>{failed}</td></tr>
            <tr><td>Peak concurrent</td><td>{peak}</td></tr>
            <tr><td>Avg call duration</td><td>{avg_dur}</td></tr>
            <tr><td>Calls by language</td><td>en={en} fr={fr} sw={sw}</td></tr>
            <tr><td>Calls by provider</td><td>Twilio={twilio} Telnyx={telnyx}</td></tr>
        </table>
    </div>
    """.format(
        active_count=s["active_count"],
        twilio_active=sum(1 for c in active if c["provider"] == "twilio"),
        telnyx_active=sum(1 for c in active if c["provider"] == "telnyx"),
        total=historical_total,
        failed=failed_total,
        peak=s["peak_concurrent"],
        avg_dur=f"{s['avg_duration_s']}s" if s["avg_duration_s"] else "—",
        en=s["by_lang"].get("en", 0),
        fr=s["by_lang"].get("fr", 0),
        sw=s["by_lang"].get("sw", 0),
        twilio=s["by_provider"].get("twilio", 0),
        telnyx=s["by_provider"].get("telnyx", 0),
    )
    return html


@app.route("/dashboard/transcripts", methods=["GET"])
def dashboard_transcripts():
    html_lines = ["<div class='section'><h2>Recent Whispers & LLM Conversations</h2>"]

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
                    label = "👤 User (Whisper)" if turn["role"] == "user" else "🤖 Assistant (LLM)"
                    html_lines.append(f"<p class='{cls}'><strong>{label}:</strong> {turn['text']}</p>")

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
            <tr><td>LLM (transcript → first token)</td><td>{llm}</td></tr>
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
    m = stats.get_cached("models")
    tts_langs = ", ".join(m["tts_languages"]) if m["tts_languages"] else "—"
    preload_status = "✅ OK" if m["preload_ok"] else "⚠️ fallback / pending"
    preload_dur = f"{m['preload_duration_s']}s" if m["preload_duration_s"] else "—"

    html = """
    <div class="section">
        <h2>Model Health</h2>
        <table>
            <tr><th>Item</th><th>Value</th></tr>
            <tr><td>Whisper model size</td><td>{whisper_size}</td></tr>
            <tr><td>Whisper device</td><td>{whisper_device}</td></tr>
            <tr><td>TTS languages loaded</td><td>{tts_langs}</td></tr>
            <tr><td>LLM provider</td><td>{llm_provider}</td></tr>
            <tr><td>LLM model</td><td>{llm_model}</td></tr>
            <tr><td>Preload status</td><td>{preload_status}</td></tr>
            <tr><td>Preload duration</td><td>{preload_dur}</td></tr>
        </table>
    </div>
    """.format(
        whisper_size=m["whisper_size"],
        whisper_device=m["whisper_device"],
        tts_langs=tts_langs,
        llm_provider=m["llm_provider"],
        llm_model=m["llm_model"],
        preload_status=preload_status,
        preload_dur=preload_dur,
    )
    return html


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
            <tr><td>TTS contention — en</td><td>{tts_en}</td></tr>
            <tr><td>TTS contention — fr</td><td>{tts_fr}</td></tr>
            <tr><td>TTS contention — sw</td><td>{tts_sw}</td></tr>
        </table>
    </div>
    """.format(
        active=c.get("active_count", 0),
        peak=c.get("peak_concurrent", 0),
        whisper_q=c.get("whisper_queue", 0),
        tts_en=tts_c.get("en", 0),
        tts_fr=tts_c.get("fr", 0),
        tts_sw=tts_c.get("sw", 0),
    )
    return html


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

    resp = VoiceResponse()
    gather = Gather(
        num_digits=1,
        action="/twilio/language",
        method="POST",
        timeout=10,
        finish_on_key="",
    )
    # Use premium neural voice to present initial language options
    gather.say(IVR_GREETING, voice="Polly.Joanna-Neural")
    resp.append(gather)
    resp.redirect("/twilio/voice", method="POST")

    return Response(str(resp), mimetype="text/xml")


@app.route("/twilio/language", methods=["POST"])
def twilio_language():
    _validate_twilio()

    digit = request.form.get("Digits", "")
    host = _public_host()

    if digit not in LANG_MAP:
        resp = VoiceResponse()
        gather = Gather(
            num_digits=1,
            action="/twilio/language",
            method="POST",
            timeout=10,
            finish_on_key="",
        )
        gather.say(IVR_INVALID, voice="Polly.Joanna-Neural")
        resp.append(gather)
        resp.redirect("/twilio/voice", method="POST")
        return Response(str(resp), mimetype="text/xml")

    lang, lang_name = LANG_MAP[digit]
    print(f"[Twilio IVR] Caller selected: {lang_name} ({lang})")

    resp = VoiceResponse()

    # Map your selected language to a premium, clear voice profile
    voice_profiles = {
        "en": "Polly.Joanna-Neural",
        "fr": "Polly.Celine-Neural",
        "sw": "Polly.Jambo"
    }
    selected_voice = voice_profiles.get(lang, "Polly.Joanna-Neural")

    # Use the advanced voice engine to speak your greeting text cleanly
    resp.say(
        HELP_GREETINGS[lang],
        language=lang,
        voice=selected_voice
    )

    connect = Connect()
    caller = request.form.get("From", "UNKNOWN")
    call_sid = request.form.get("CallSid", caller)
    connect.stream(
        url=f"wss://{host}/media-stream/twilio/{lang}?From={caller}&CallSid={call_sid}"
    )
    resp.append(connect)

    return Response(str(resp), mimetype="text/xml")


# ===========================================================================
# TELNYX
# ===========================================================================

@app.route("/telnyx/voice", methods=["POST"])
def telnyx_voice():
    host = _public_host()
    texml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Gather numDigits="1" action="https://{host}/telnyx/language" method="POST" timeout="10">
        <Say>{IVR_GREETING}</Say>
    </Gather>
    <Redirect method="POST">https://{host}/telnyx/voice</Redirect>
</Response>"""
    return Response(texml, mimetype="text/xml")


@app.route("/telnyx/language", methods=["POST"])
def telnyx_language():
    digit = request.form.get("Digits", "")
    host = _public_host()

    if digit not in LANG_MAP:
        texml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Gather numDigits="1" action="https://{host}/telnyx/language" method="POST" timeout="10">
        <Say>{IVR_INVALID}</Say>
    </Gather>
    <Redirect method="POST">https://{host}/telnyx/voice</Redirect>
</Response>"""
        return Response(texml, mimetype="text/xml")

    lang, lang_name = LANG_MAP[digit]
    print(f"[Telnyx IVR] Caller selected: {lang_name} ({lang})")

    caller = request.form.get("From", "UNKNOWN")
    texml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say>{HELP_GREETINGS[lang]}</Say>
    <Stream url="wss://{host}/media-stream/telnyx/{lang}?From={caller}" />
</Response>"""
    return Response(texml, mimetype="text/xml")


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
                    f"<span class='log-time'>[{timestamp_str}]</span>"
                    f"INIT_STAGE: Component <strong style='color:#fff;'>{log['model_name']}</strong> "
                    f"allocated inside engine space in <strong style='color:#facc15;'>{log['duration_s']}s</strong>"
                    f"<span class='log-dev'>[Target: {log['device'].upper()}]</span>"
                    f"</div>"
                )
    except Exception as e:
        html_lines.append(
            f"<div class='log-line' style='color:#ef4444;'>Failed to query database engine logs: {e}</div>")

    html_lines.append("</div></div>")
    return "".join(html_lines)