import os
import sys
import time
import asyncio
import logging
from contextlib import asynccontextmanager
from urllib.parse import unquote

import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.wsgi import WSGIMiddleware
from dotenv import load_dotenv

from routes import app as flask_app
from VoiceAssistant import VoiceAssistant
from Audio.PhoneStreamSource import PhoneStreamSource
from Audio.PhoneAudioOutput import PhoneAudioOutput
import stats
import database  # persistent layer — tables created on import

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gguf").strip().lower()



# ---------------------------------------------------------------------------
# In-process runtime store — populated by lifespan, read by WebSocket handler
# ---------------------------------------------------------------------------
_preloaded_tts: dict = {}
_preloaded_stt: dict = {}


def _safe_log_model_load(model_name: str, duration_s: float, device: str = "cpu") -> None:
    """Best-effort startup profiling; never fail boot because dashboard logging failed."""
    try:
        database.db_log_model_load(model_name, duration_s, device)
    except Exception as exc:
        print(f"[startup-log] Could not persist model load record for {model_name}: {exc}")


def _load_models():
    global _preloaded_tts, _preloaded_stt

    print("\n" + "=" * 60)
    print("🔥 INITIALIZING SYSTEM INFRASTRUCTURE — CACHING LOCAL STRATEGIES")
    print("=" * 60 + "\n")

    total_start = time.time()

    # ------------------------------------------------------------------
    # STT — OpenAI Whisper via faster-whisper for all IVR-selected languages
    #   1 -> en, 2 -> fr, 3 -> sw are enforced in routes.py.
    #   The same model is shared across languages; language is forced per call.
    # ------------------------------------------------------------------
    print("📦 [1/3] Loading STT: OpenAI Whisper/faster-whisper for EN/FR/SW...")
    t = time.time()

    from faster_whisper import WhisperModel

    whisper_model_name = (
        os.getenv("OPENAI_WHISPER_MODEL")
        or os.getenv("WHISPER_MODEL_SIZE")
        or "large-v3"
    ).strip()

    resolved_device = os.getenv(
        "WHISPER_DEVICE",
        "cuda" if torch.cuda.is_available() else "cpu",
    ).strip()
    if resolved_device == "cuda" and not torch.cuda.is_available():
        print("   ⚠️ WHISPER_DEVICE=cuda requested, but CUDA is unavailable. Falling back to CPU.")
        resolved_device = "cpu"

    faster_compute_type = os.getenv(
        "WHISPER_COMPUTE_TYPE",
        "float16" if resolved_device == "cuda" else "int8",
    ).strip()

    whisper_model = WhisperModel(
        whisper_model_name,
        device=resolved_device,
        compute_type=faster_compute_type,
        download_root=os.getenv("HF_HOME"),
    )

    _preloaded_stt.clear()
    for lang in ("en", "fr", "sw"):
        _preloaded_stt[lang] = {
            "engine": "faster_whisper",
            "model": whisper_model,
            "model_name": whisper_model_name,
            "device": resolved_device,
            "forced_language": lang,
        }

    stt_duration = time.time() - t
    _safe_log_model_load(f"stt:openai_whisper:{whisper_model_name}", stt_duration, resolved_device)
    print(
        "   ✅ STT ready: "
        f"model=[{whisper_model_name}] backend=[faster-whisper] "
        f"forced_languages=[en, fr, sw] device=[{resolved_device}] "
        f"compute_type=[{faster_compute_type}] — {stt_duration:.1f}s\n"
    )

    # ------------------------------------------------------------------
    # TTS — Hybrid: local Kokoro for EN/FR, Soniox API config for SW
    # ------------------------------------------------------------------
    print("📦 [2/3] Configuring TTS: Kokoro(en/fr) + Soniox(sw)...")
    t = time.time()

    try:
        from kokoro import KPipeline
        from soniox import SonioxClient  # noqa: F401

        if not os.getenv("SONIOX_API_KEY"):
            raise RuntimeError("SONIOX_API_KEY is not set in environment/secrets. It is still needed for Swahili TTS.")

        kokoro_sample_rate = int(os.getenv("KOKORO_SAMPLE_RATE", "24000"))
        kokoro_speed = float(os.getenv("KOKORO_SPEED", "0.95"))
        kokoro_voice_en = os.getenv("KOKORO_VOICE_EN", "af_heart")
        kokoro_voice_fr = os.getenv("KOKORO_VOICE_FR", "ff_siwis")
        kokoro_lang_en = os.getenv("KOKORO_LANG_CODE_EN", "a")
        kokoro_lang_fr = os.getenv("KOKORO_LANG_CODE_FR", "f")

        soniox_model = os.getenv("SONIOX_TTS_MODEL", "tts-rt-v1")
        soniox_voice = os.getenv("SONIOX_TTS_VOICE", "Grace")
        soniox_format = os.getenv("SONIOX_TTS_AUDIO_FORMAT", "wav")
        soniox_sample_rate = int(os.getenv("SONIOX_TTS_SAMPLE_RATE", "16000"))

        en_start = time.time()
        kokoro_en = KPipeline(lang_code=kokoro_lang_en)
        _safe_log_model_load("tts:kokoro:en", time.time() - en_start, "cpu")

        fr_start = time.time()
        kokoro_fr = KPipeline(lang_code=kokoro_lang_fr)
        _safe_log_model_load("tts:kokoro:fr", time.time() - fr_start, "cpu")

        _preloaded_tts.clear()
        _preloaded_tts.update({
            "en": {
                "engine": "kokoro",
                "pipeline": kokoro_en,
                "voice": kokoro_voice_en,
                "sample_rate": kokoro_sample_rate,
                "speed": kokoro_speed,
            },
            "fr": {
                "engine": "kokoro",
                "pipeline": kokoro_fr,
                "voice": kokoro_voice_fr,
                "sample_rate": kokoro_sample_rate,
                "speed": kokoro_speed,
            },
            "sw": {
                "engine": "soniox",
                "model": soniox_model,
                "voice": soniox_voice,
                "language": os.getenv("SONIOX_TTS_LANG_SW", "sw"),
                "audio_format": soniox_format,
                "sample_rate": soniox_sample_rate,
            },
        })
        print(
            "   ✅ TTS configured: "
            f"en=Kokoro[{kokoro_voice_en}], fr=Kokoro[{kokoro_voice_fr}], "
            f"sw=Soniox[{soniox_voice}/{soniox_model}] — {time.time() - t:.1f}s\n"
        )
    except Exception as exc:
        print(f"   ❌ Hybrid TTS configuration failed: {exc}")
        print("   Add `kokoro==0.9.4` and `soniox` to requirements.txt, then rebuild/redeploy.")
        raise

    # ------------------------------------------------------------------
    # LLM — GGUF Warm Up Engine Singleton
    # ------------------------------------------------------------------
    if LLM_PROVIDER in ("qwen", "gguf", "local"):
        print("📦 [3/3] Warming up local GGUF engine singleton...")
        llm_start = time.time()
        try:
            from llmModule.LLM import LLM
            # Trigger the standard factory setup to instantiate and cache the shared Llama object
            LLM.get_model(provider="gguf", lang="en")
            llm_duration = time.time() - llm_start
            _safe_log_model_load("llm:gguf", llm_duration, os.getenv("WHISPER_DEVICE", "cpu"))
            print(f"   ✅ GGUF engine compiled successfully — {llm_duration:.1f}s\n")
        except Exception as e:
            print(f"   ❌ Critical failure initializing GGUF engine: {e}")
            sys.exit(1)
    else:
        print("📦 [3/3] Using Cloud Provider. Skipping local LLM engine loading.\n")

    total_duration = time.time() - total_start
    stats.model_info["preload_ok"] = True
    stats.model_info["preload_duration_s"] = round(total_duration, 2)
    print(f"🚀 All strategies warmed up successfully! Total application boot timeline: {total_duration:.1f}s\n")


@asynccontextmanager
async def lifespan(fastapi_app: FastAPI):
    # Pass the function reference itself, NOT the evaluated output
    await asyncio.to_thread(_load_models)

    # Save active runtime engine attributes back to stats tracker
    stats.model_info["llm_provider"] = LLM_PROVIDER
    stats.model_info["stt"] = os.getenv("OPENAI_WHISPER_MODEL", os.getenv("WHISPER_MODEL_SIZE", "large-v3"))
    stats.model_info["stt_languages"] = "1=en, 2=fr, 3=sw"
    yield
    # Free memory maps on down-scale/shutdown sequences
    _preloaded_tts.clear()
    _preloaded_stt.clear()


fastapi_app = FastAPI(lifespan=lifespan)


# ---------------------------------------------------------------------------
# WebSocket endpoint — Multi-provider support (Twilio & Telnyx)
# ---------------------------------------------------------------------------
@fastapi_app.websocket("/media-stream/{provider}/{lang}")
async def handle_media_stream(websocket: WebSocket, provider: str, lang: str):
    supported_providers = {"twilio", "telnyx"}
    supported_langs = {"en", "fr", "sw"}

    if provider not in supported_providers or lang not in supported_langs:
        print(f"[REJECTED] Provider={provider}, Lang={lang}")
        await websocket.close(code=1008)
        return

    await websocket.accept()

    # Caller identity and initial post-IVR greeting are passed from routes.py.
    caller_number = websocket.query_params.get("From", "UNKNOWN")
    initial_greeting = unquote(websocket.query_params.get("InitialGreeting", "") or "").strip()
    session_id = f"{provider}:{id(websocket)}"

    # Record the call once. stats.call_started() already persists the session to SQLite.
    # Do not also call database.async_start_call(), or the DB does duplicate work.
    stats.call_started(session_id, provider, lang, caller_number)

    print("\n" + "=" * 60)
    print(f"   INBOUND CALL CONNECTED & INITIALIZED IN DB")
    print(f"   Session ID: {session_id}")
    print(f"   Provider:   {provider.upper()}")
    print(f"   Caller:     {caller_number}")
    print(f"   Language:   {lang.upper()}")
    _llm_label = stats.model_info.get("llm_provider", LLM_PROVIDER)
    print(f"   LLM Engine: {_llm_label.upper()}")
    print("=" * 60 + "\n")

    source = PhoneStreamSource(provider=provider)
    output = PhoneAudioOutput(websocket, provider=provider)

    # COMPATIBILITY FIX: Asynchronous turn logger hook fed into the VoiceAssistant pipeline
    async def transcript_logger_hook(role: str, text: str):
        if text and text.strip():
            await database.async_log_transcript(session_id, role, text)

    # Wire the callback wrapper parameter directly into your assistant pipeline
    assistant = VoiceAssistant(
        source=source,
        output=output,
        provider=LLM_PROVIDER,
        lang=lang,
        preloaded_tts=_preloaded_tts,
        preloaded_stt=_preloaded_stt,
        on_turn_logged=transcript_logger_hook,  # Ensure your instance intercepts text turns
    )

    async def websocket_receiver():
        try:
            while True:
                message = await websocket.receive_text()
                await source.add_data(message)

                if source.stream_sid and not output.stream_sid:
                    output.set_stream_sid(source.stream_sid)

        except WebSocketDisconnect:
            print(f"\n❌ [{provider.upper()}] Caller hung up")
        except Exception as e:
            print(f"\n[{provider.upper()}] Receiver error: {e}")

    async def wait_for_outbound_stream_ready(timeout_s: float = 3.0):
        """
        Give the provider's first websocket/start event a moment to arrive before
        sending the TTS greeting. Twilio needs streamSid on outbound media;
        Telnyx is less strict, but waiting is still harmless.
        """
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if output.stream_sid or output.stream_id:
                return True
            if getattr(source, "stream_sid", None):
                output.set_stream_sid(source.stream_sid)
                return True
            if getattr(source, "stream_id", None):
                output.set_stream_id(source.stream_id)
                return True
            await asyncio.sleep(0.02)

        print(
            f"[{provider.upper()}] Outbound stream id not observed after {timeout_s:.1f}s; "
            "attempting greeting anyway."
        )
        return False

    async def assistant_runner():
        await wait_for_outbound_stream_ready()

        if initial_greeting:
            print(f"[{provider.upper()}] Speaking initial greeting through TTS: {initial_greeting!r}")
            await assistant.speak_greeting(initial_greeting)
        else:
            print(f"[{provider.upper()}] No InitialGreeting query parameter received.")

        await assistant.start()

    receiver_task = asyncio.create_task(websocket_receiver())
    assistant_task = asyncio.create_task(assistant_runner())

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
        # End the call once. stats.call_ended() already persists duration/status to SQLite.
        stats.call_ended(session_id)
        print(f"[{provider.upper()}] Session cleaned up and written to SQLite.\n")


# Mount the Flask Application WSGI Layer for the Admin Dashboard UI
fastapi_app.mount("/", WSGIMiddleware(flask_app))


class _DashboardFilter(logging.Filter):
    """Suppress noisy /dashboard/* and /metrics poll lines in uvicorn access log."""
    _SUPPRESSED = ("/dashboard/", "/metrics")

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return not any(path in msg for path in self._SUPPRESSED)


def _configure_logging():
    logging.getLogger("uvicorn.access").addFilter(_DashboardFilter())


if __name__ == "__main__":
    import uvicorn

    _configure_logging()
    port = int(os.getenv("PORT", "7860"))
    print(f"Starting voice assistant on port {port}...")
    uvicorn.run("sockets:fastapi_app", host="0.0.0.0", port=port, log_level="info")
