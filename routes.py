import os
from flask import Flask, request, Response, abort
from twilio.twiml.voice_response import VoiceResponse, Connect, Gather, Say
from twilio.request_validator import RequestValidator
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
VALIDATE_TWILIO = os.getenv("VALIDATE_TWILIO_SIGNATURE", "true").lower() == "true"

# Maps DTMF digit → (language code, human-readable name)
LANG_MAP = {
    "1": ("en", "English"),
    "2": ("fr", "French"),
    "3": ("sw", "Swahili"),
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


def _public_host() -> str:
    """
    Returns the public hostname (no scheme).
    Reads PUBLIC_HOST from .env; falls back to request.host for local dev.
    """
    return os.getenv("PUBLIC_HOST") or request.host


def _validate_twilio():
    """Abort 403 if Twilio signature validation is enabled and fails."""
    if VALIDATE_TWILIO and TWILIO_AUTH_TOKEN:
        validator = RequestValidator(TWILIO_AUTH_TOKEN)
        if not validator.validate(
            request.url,
            request.form,
            request.headers.get("X-Twilio-Signature", ""),
        ):
            abort(403, description="Invalid Twilio signature")


# ===========================================================================
# TWILIO
# ===========================================================================

@app.route("/twilio/voice", methods=["POST"])
def twilio_voice():
    """
    Step 1 — Twilio calls this on inbound call.
    Plays the IVR menu and waits for a single DTMF digit.
    """
    _validate_twilio()

    resp = VoiceResponse()
    gather = Gather(
        num_digits=1,
        action="/twilio/language",   # Step 2
        method="POST",
        timeout=10,
        finish_on_key="",            # Don't need # — single digit is enough
    )
    gather.say(IVR_GREETING)
    resp.append(gather)

    # If the caller doesn't press anything, replay the menu
    resp.redirect("/twilio/voice", method="POST")

    return Response(str(resp), mimetype="text/xml")


@app.route("/twilio/language", methods=["POST"])
def twilio_language():
    """
    Step 2 — Twilio posts the pressed digit here.
    Validates it, then connects the Media Stream with the chosen language
    embedded in the WebSocket URL path.
    """
    _validate_twilio()

    digit = request.form.get("Digits", "")
    host  = _public_host()

    if digit not in LANG_MAP:
        # Invalid key — replay the IVR menu
        resp = VoiceResponse()
        gather = Gather(
            num_digits=1,
            action="/twilio/language",
            method="POST",
            timeout=10,
            finish_on_key="",
        )
        gather.say(IVR_INVALID)
        resp.append(gather)
        resp.redirect("/twilio/voice", method="POST")
        return Response(str(resp), mimetype="text/xml")

    lang, lang_name = LANG_MAP[digit]
    print(f"[Twilio IVR] Caller selected: {lang_name} ({lang})")

    resp = VoiceResponse()
    connect = Connect()
    # Language is passed in the URL so the WebSocket handler knows it immediately
    connect.stream(url=f"wss://{host}/media-stream/twilio/{lang}")
    resp.append(connect)

    return Response(str(resp), mimetype="text/xml")


# ===========================================================================
# TELNYX
# ===========================================================================

@app.route("/telnyx/voice", methods=["POST"])
def telnyx_voice():
    """
    Step 1 — Telnyx calls this on inbound call.
    Plays the IVR menu and waits for a single DTMF digit via TeXML <Gather>.
    """
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
    """
    Step 2 — Telnyx posts the pressed digit here.
    Validates it, then opens a Media Stream with the chosen language in the URL.
    """
    digit = request.form.get("Digits", "")
    host  = _public_host()

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

    texml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Stream url="wss://{host}/media-stream/telnyx/{lang}" />
</Response>"""
    return Response(texml, mimetype="text/xml")


# ===========================================================================
# Health check
# ===========================================================================

@app.route("/health", methods=["GET"])
def health():
    return {"status": "ok"}, 200