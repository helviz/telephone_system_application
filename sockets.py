import os
import sys
import time
import asyncio
import logging
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.wsgi import WSGIMiddleware
from dotenv import load_dotenv

from routes import app as flask_app
from VoiceAssistant import VoiceAssistant
from Audio.PhoneStreamSource import PhoneStreamSource
from Audio.PhoneAudioOutput import PhoneAudioOutput
import stats

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")

# ---------------------------------------------------------------------------
# In-process model store — populated by lifespan, read by WebSocket handler
# ---------------------------------------------------------------------------
_preloaded_tts: dict     = {}
_preloaded_whisper       = None


def _load_models():
    global _preloaded_tts, _preloaded_whisper

    print("\n" + "=" * 60)
    print("🔥 INITIALIZING COMPACT SERVICE PROFILE — HUGGING FACE FREE TIER")
    print("=" * 60 + "\n")

    total_start = time.time()

    # ------------------------------------------------------------------
    # STT — Faster-Whisper (Load this immediately)
    # ------------------------------------------------------------------
    print("📦 [1/2] Loading Faster-Whisper...")
    t = time.time()
    from faster_whisper import WhisperModel
    import os

    resolved_size = os.getenv("WHISPER_MODEL_SIZE", "small").strip()
    resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if resolved_device == "cuda" else "int8"

    _preloaded_whisper = WhisperModel(
        resolved_size,
        device=resolved_device,
        compute_type=compute_type,
        download_root=os.getenv("HF_HOME")
    )
    print(f"✅ Whisper loaded in {time.time() - t:.2f}s")

    # ------------------------------------------------------------------
    # TTS — Kept Empty for Lazy Evaluation
    # ------------------------------------------------------------------
    print("📦 [2/2] TTS Language Array Initialized [Deferred Mode]...")
    _preloaded_tts = {}  # Leave empty! TTSModule will handle loading per-call.

    print(f"\n🚀 System initialized successfully in {time.time() - total_start:.2f}s! Ready for inbound audio traffic.")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# FastAPI lifespan — runs _load_models() before the server accepts requests
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Run the blocking model-load in a thread so we don't block the loop
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _load_models)
    yield
    # (shutdown hook — nothing to clean up for these models)


fastapi_app = FastAPI(lifespan=lifespan)


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------
@fastapi_app.websocket("/media-stream/{provider}/{lang}")
async def handle_media_stream(websocket: WebSocket, provider: str, lang: str):
    supported_providers = {"twilio", "telnyx"}
    supported_langs     = {"en", "fr", "sw"}

    if provider not in supported_providers or lang not in supported_langs:
        print(f"[REJECTED] Provider={provider}, Lang={lang}")
        await websocket.close(code=1008)
        return

    await websocket.accept()

    session_id = f"{provider}:{id(websocket)}"
    stats.call_started(session_id, provider, lang)

    print("\n" + "=" * 60)
    print(f"   INBOUND CALL CONNECTED")
    print(f"   Provider:   {provider.upper()}")
    print(f"   Language:   {lang.upper()}")
    _llm_label = stats.model_info.get("llm_provider", LLM_PROVIDER)
    print(f"   LLM Engine: {_llm_label.upper()}")
    print("=" * 60 + "\n")

    source = PhoneStreamSource(provider=provider)
    output = PhoneAudioOutput(websocket, provider=provider)

    assistant = VoiceAssistant(
        source=source,
        output=output,
        provider=LLM_PROVIDER,
        lang=lang,
        preloaded_tts=_preloaded_tts,
        preloaded_whisper=_preloaded_whisper,
    )

    async def websocket_receiver():
        try:
            while True:
                message = await websocket.receive_text()
                await source.add_data(message)

                # Forward streamSid to output as soon as source captures it
                # from Twilio's 'start' event. No-op once set.
                if source.stream_sid and not output.stream_sid:
                    output.set_stream_sid(source.stream_sid)

        except WebSocketDisconnect:
            print(f"\n❌ [{provider.upper()}] Caller hung up")
        except Exception as e:
            print(f"\n[{provider.upper()}] Receiver error: {e}")

    receiver_task = asyncio.create_task(websocket_receiver())
    assistant_task = asyncio.create_task(assistant.start())

    try:
        done, pending = await asyncio.wait(
            [receiver_task, assistant_task],
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
    finally:
        stats.call_ended(session_id)
        print(f"[{provider.upper()}] Session cleaned up.\n")


fastapi_app.mount("/", WSGIMiddleware(flask_app))

class _DashboardFilter(logging.Filter):
    """
    Drop uvicorn access log records for /dashboard/* and /metrics routes.
    These are polled every 1-2s by the browser and produce hundreds of
    log lines per minute that bury actual call activity.
    """
    _SUPPRESSED = ("/dashboard/", "/metrics")

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return not any(path in msg for path in self._SUPPRESSED)


def _configure_logging():
    """Attach the dashboard filter to uvicorn's access logger."""
    logging.getLogger("uvicorn.access").addFilter(_DashboardFilter())


# Call at module level so the filter is active when HF Spaces launches
# uvicorn externally (the __main__ block won't run in that case).
_configure_logging()


if __name__ == "__main__":
    import uvicorn

    _configure_logging()
    port = int(os.getenv("PORT", 7860))
    print(f"Starting voice assistant on port {port}...")
    uvicorn.run(fastapi_app, host="0.0.0.0", port=port)