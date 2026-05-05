"""
Twilio Voice Webhook — API Gateway routes
==========================================
Mount this router in your main FastAPI app:

    from twilio_routes import router as twilio_router
    app.include_router(twilio_router, prefix="/twilio")

Required .env.example vars (see ..env.example.example):
    TWILIO_ACCOUNT_SID
    TWILIO_AUTH_TOKEN
    TWILIO_PHONE_NUMBER
    BASE_URL          — public root URL of this deployment
    STT_URL           — internal URL of the STT service
    LLM_URL           — internal URL of the LLM service
    TTS_URL           — internal URL of the TTS service
    REDIS_URL         — Redis connection string

Call flow
---------
1. Twilio calls POST /twilio/voice when a call arrives.
   We validate the Twilio signature and return TwiML that:
     a. Greets the caller with a short <Say>.
     b. Opens a Media Stream back to  wss://<BASE_URL>/stream/<CallSid>
        (handled by the STT service WebSocket endpoint).

2. The STT service runs VAD + Whisper on the live audio and publishes
   transcripts to Redis channel  twilio:transcript:<call_sid>.

3. POST /twilio/transcript is called by the STT service (or a Redis
   subscriber) whenever a complete utterance is ready.
   We call the LLM service → get a reply → call the TTS service →
   get an audio URL → instruct Twilio to play it via the REST API.

4. POST /twilio/status receives Twilio call-status callbacks so we
   can log / clean up Redis keys when the call ends.
"""

from __future__ import annotations

import asyncio
import os
from typing import Optional

import httpx
import redis.asyncio as aioredis
import structlog
from fastapi import APIRouter, Form, Header, HTTPException, Request, Response
from twilio.request_validator import RequestValidator
from twilio.rest import Client as TwilioClient
from twilio.twiml.voice_response import Connect, Say, Start, Stream, VoiceResponse

log = structlog.get_logger()

# ── Config ────────────────────────────────────────────────────────────────────
TWILIO_ACCOUNT_SID  = os.environ["TWILIO_ACCOUNT_SID"]
TWILIO_AUTH_TOKEN   = os.environ["TWILIO_AUTH_TOKEN"]
TWILIO_PHONE_NUMBER = os.environ["TWILIO_PHONE_NUMBER"]
BASE_URL            = os.environ["BASE_URL"].rstrip("/")
STT_URL             = os.getenv("STT_URL", "http://stt:8001")
LLM_URL             = os.getenv("LLM_URL", "http://llm:8002")
TTS_URL             = os.getenv("TTS_URL", "http://tts:8003")
REDIS_URL           = os.getenv("REDIS_URL", "redis://redis:6379/0")

STREAM_WS_URL = f"{BASE_URL.replace('https://', 'wss://').replace('http://', 'ws://')}/stream"

twilio_client   = TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
twilio_validator = RequestValidator(TWILIO_AUTH_TOKEN)

router = APIRouter()

# ── Redis (shared pool) ───────────────────────────────────────────────────────
_redis: Optional[aioredis.Redis] = None


async def get_redis() -> aioredis.Redis:
    global _redis
    if _redis is None:
        _redis = await aioredis.from_url(REDIS_URL, decode_responses=True)
    return _redis


# ── Signature validation ───────────────────────────────────────────────────────

def _validate_twilio(request: Request, form_data: dict) -> None:
    """
    Raises HTTP 403 if the X-Twilio-Signature header does not match.
    In development (BASE_URL starts with http://localhost) validation is skipped.
    """
    if BASE_URL.startswith("http://localhost"):
        return  # skip in local dev

    signature = request.headers.get("X-Twilio-Signature", "")
    url        = str(request.url)
    if not twilio_validator.validate(url, form_data, signature):
        raise HTTPException(status_code=403, detail="Invalid Twilio signature")


# ── Endpoint 1: inbound call webhook ─────────────────────────────────────────

@router.post("/voice", response_class=Response)
async def voice_webhook(
    request: Request,
    CallSid: str  = Form(...),
    From:    str  = Form(...),
    To:      str  = Form(...),
):
    """
    Twilio calls this URL when a call is answered.
    We return TwiML that:
      1. Says a greeting.
      2. Starts a bidirectional Media Stream to our STT WebSocket.

    Configure this URL in the Twilio Console:
      Phone Numbers → Manage → Active Numbers → <your number>
        Voice & Fax → A Call Comes In → Webhook → POST
        URL: https://your-domain.com/twilio/voice
    """
    form_data = dict(await request.form())
    _validate_twilio(request, form_data)

    log.info("call_inbound", call_sid=CallSid, from_=From, to=To)

    # Store caller info in Redis for the duration of the call
    redis = await get_redis()
    await redis.hset(f"call:{CallSid}", mapping={"from": From, "to": To, "status": "active"})
    await redis.expire(f"call:{CallSid}", 3600)  # 1-hour TTL

    # Build TwiML
    response = VoiceResponse()
    response.say(
        "Hello! I'm your AI assistant. How can I help you today?",
        voice="Polly.Joanna",
    )

    # Open a Media Stream — Twilio will pipe real-time µ-law audio here
    start = Start()
    start.stream(
        url=f"{STREAM_WS_URL}/{CallSid}",
        track="inbound_track",            # capture caller audio only
    )
    response.append(start)

    # Keep the call alive while we process audio
    response.pause(length=60)

    log.info("twiml_stream_started", call_sid=CallSid, stream_url=f"{STREAM_WS_URL}/{CallSid}")
    return Response(content=str(response), media_type="application/xml")


# ── Endpoint 2: transcript ready (called internally) ──────────────────────────

@router.post("/transcript")
async def handle_transcript(
    call_sid:    str,
    transcript:  str,
    stream_sid:  str = "",
):
    """
    Internal endpoint called by the Redis subscriber (see subscribe_transcripts)
    or directly by any service that has a completed transcript.

    Pipeline: transcript → LLM → TTS → Twilio play
    """
    log.info("transcript_received", call_sid=call_sid, text=transcript)

    async with httpx.AsyncClient(timeout=30) as client:
        # 1. LLM — get AI reply
        llm_resp = await client.post(
            f"{LLM_URL}/generate",
            json={"prompt": transcript, "call_sid": call_sid},
        )
        llm_resp.raise_for_status()
        reply_text = llm_resp.json()["text"]
        log.info("llm_reply", call_sid=call_sid, text=reply_text)

        # 2. TTS — synthesise audio
        tts_resp = await client.post(
            f"{TTS_URL}/synthesize",
            json={"text": reply_text, "call_sid": call_sid},
        )
        tts_resp.raise_for_status()
        audio_url = tts_resp.json()["url"]   # publicly reachable audio URL
        log.info("tts_audio", call_sid=call_sid, url=audio_url)

    # 3. Instruct Twilio to play the audio on the live call
    try:
        twilio_client.calls(call_sid).update(
            twiml=f"""
            <Response>
              <Play>{audio_url}</Play>
              <Pause length="30"/>
            </Response>
            """
        )
        log.info("twilio_play_sent", call_sid=call_sid)
    except Exception as exc:
        log.error("twilio_play_failed", call_sid=call_sid, error=str(exc))
        raise HTTPException(status_code=502, detail=f"Twilio update failed: {exc}")

    return {"status": "ok", "call_sid": call_sid}


# ── Endpoint 3: call status callback ─────────────────────────────────────────

@router.post("/status")
async def call_status(
    request:    Request,
    CallSid:    str = Form(...),
    CallStatus: str = Form(...),
):
    """
    Twilio sends status updates (initiated, ringing, in-progress, completed, etc.)
    Configure as the Status Callback URL in the Twilio Console.
    URL: https://your-domain.com/twilio/status
    """
    form_data = dict(await request.form())
    _validate_twilio(request, form_data)

    log.info("call_status", call_sid=CallSid, status=CallStatus)

    if CallStatus in ("completed", "failed", "busy", "no-answer", "canceled"):
        redis = await get_redis()
        await redis.hset(f"call:{CallSid}", "status", CallStatus)
        # Clean up after a short delay so in-flight tasks can read the record
        await asyncio.sleep(5)
        await redis.delete(f"call:{CallSid}")
        log.info("call_cleaned_up", call_sid=CallSid)

    return Response(content="<Response/>", media_type="application/xml")


# ── Redis subscriber — bridges STT pub/sub to the transcript pipeline ─────────

async def subscribe_transcripts():
    """
    Long-running coroutine: subscribe to all  twilio:transcript:*  Redis channels
    and forward each transcript to handle_transcript().

    Start this as a background task from your app lifespan:

        @asyncio.on_event("startup")
        async def startup():
            asyncio.create_task(subscribe_transcripts())
    """
    redis = await aioredis.from_url(REDIS_URL, decode_responses=True)
    pubsub = redis.pubsub()
    await pubsub.psubscribe("twilio:transcript:*")
    log.info("redis_subscribed", pattern="twilio:transcript:*")

    import json

    async for message in pubsub.listen():
        if message["type"] != "pmessage":
            continue
        try:
            data = json.loads(message["data"])
            asyncio.create_task(
                handle_transcript(
                    call_sid=data["call_sid"],
                    transcript=data["transcript"],
                    stream_sid=data.get("stream_sid", ""),
                )
            )
        except Exception as exc:
            log.error("subscriber_error", error=str(exc))
