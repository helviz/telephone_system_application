import os
import random
import sys
import time
import asyncio
import logging
from contextlib import asynccontextmanager

import numpy as np
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


def seed_omnivoice(seed: int | None = None) -> int:
    """Seed RNGs before loading/generating with OmniVoice."""
    seed = int(seed if seed is not None else os.getenv("OMNIVOICE_SEED", "12345"))
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    try:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    except Exception:
        pass

    return seed


# ---------------------------------------------------------------------------
# In-process model store — populated by lifespan, read by WebSocket handler
# ---------------------------------------------------------------------------
_preloaded_tts: dict = {}
_preloaded_whisper = None


def _load_models():
    global _preloaded_tts, _preloaded_whisper

    print("\n" + "=" * 60)
    print("🔥 INITIALIZING SYSTEM INFRASTRUCTURE — CACHING LOCAL STRATEGIES")
    print("=" * 60 + "\n")

    total_start = time.time()

    # ------------------------------------------------------------------
    # STT — Faster-Whisper
    # ------------------------------------------------------------------
    print("📦 [1/3] Loading Faster-Whisper...")
    t = time.time()
    from faster_whisper import WhisperModel

    resolved_size = os.getenv("WHISPER_MODEL_SIZE", "medium").strip()
    resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if resolved_device == "cuda" else "int8"

    _preloaded_whisper = WhisperModel(
        resolved_size,
        device=resolved_device,
        compute_type=compute_type,
        download_root=os.getenv("HF_HOME"),
    )
    print(f"   ✅ Faster-Whisper loaded on {resolved_device} in {time.time() - t:.1f}s\n")

    # ------------------------------------------------------------------
    # TTS — Kokoro for English/French, OmniVoice for Swahili
    # ------------------------------------------------------------------
    print("📦 [2/3] Loading TTS: Kokoro(en=af_heart, fr=ff_siwis) + OmniVoice(sw)...")
    t = time.time()

    try:
        from kokoro import KPipeline
    except Exception as exc:
        print(
            "   ❌ Kokoro failed to import. Add `kokoro==0.9.4` to requirements.txt "
            "and rebuild/redeploy the app."
        )
        raise exc

    try:
        from omnivoice import OmniVoice
    except Exception as exc:
        print(
            "   ❌ OmniVoice failed to import. Add `omnivoice` to requirements.txt "
            "and rebuild/redeploy the app."
        )
        raise exc

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    device_map = "cuda:0" if device == "cuda" else "cpu"
    omnivoice_seed = seed_omnivoice()
    print(f"   🎚️  OmniVoice deterministic seed: {omnivoice_seed}")

    kokoro_slots = {
        "en": {"voice": "af_heart", "lang_code": "a"},
        "fr": {"voice": "ff_siwis", "lang_code": "f"},
    }

    for lang, cfg in kokoro_slots.items():
        lt = time.time()
        print(f"   Loading language slot [{lang}] via Kokoro voice {cfg['voice']}...")
        _preloaded_tts[lang] = {
            "engine": "kokoro",
            "pipeline": KPipeline(lang_code=cfg["lang_code"]),
            "voice": cfg["voice"],
            "sample_rate": 24000,
        }
        print(f"   ✅ Language component [{lang}] Kokoro ready — {time.time() - lt:.1f}s")

    lt = time.time()
    sw_model_name = "k2-fsa/OmniVoice"
    print(f"   Loading language slot [sw] via {sw_model_name}...")
    seed_omnivoice(omnivoice_seed)
    sw_model = OmniVoice.from_pretrained(
        sw_model_name,
        device_map=device_map,
        dtype=dtype,
    )
    _preloaded_tts["sw"] = {
        "engine": "omnivoice",
        "model": sw_model,
        "sample_rate": 24000,
        "instruct": "female, middle-aged, moderate pitch",
        "num_step": 16,
        "speed": 1.0,
        "language_id": "sw",
        "seed": omnivoice_seed,
    }
    print(f"   ✅ Language component [sw] OmniVoice ready — {time.time() - lt:.1f}s")

    print(f"   ✅ All TTS configurations allocated on {device} — {time.time() - t:.1f}s\n")

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
            print(f"   ✅ GGUF engine compiled successfully — {time.time() - llm_start:.1f}s\n")
        except Exception as e:
            print(f"   ❌ Critical failure initializing GGUF engine: {e}")
            sys.exit(1)
    else:
        print("📦 [3/3] Using Cloud Provider. Skipping local LLM engine loading.\n")

    print(
        f"🚀 All strategies warmed up successfully! Total application boot timeline: {time.time() - total_start:.1f}s\n")


@asynccontextmanager
async def lifespan(fastapi_app: FastAPI):
    # Pass the function reference itself, NOT the evaluated output
    await asyncio.to_thread(_load_models)

    # Save active runtime engine attributes back to stats tracker
    stats.model_info["llm_provider"] = LLM_PROVIDER
    stats.model_info["whisper_size"] = os.getenv("WHISPER_MODEL_SIZE", "medium")
    yield
    # Free memory maps on down-scale/shutdown sequences
    _preloaded_tts.clear()


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

    # Caller identity query param parsed
    caller_number = websocket.query_params.get("From", "UNKNOWN")
    session_id = f"{provider}:{id(websocket)}"

    # COMPATIBILITY FIX: Write to memory Dashboard AND persist session entry to SQLite
    stats.call_started(session_id, provider, lang, caller_number)
    await database.async_start_call(session_id, provider, lang, caller_number)

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
        preloaded_whisper=_preloaded_whisper,
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
        # COMPATIBILITY FIX: Clear memory slots AND calculate and update persistent duration state
        stats.call_ended(session_id)
        await database.async_end_call(session_id, completed=True)
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
    port = int(os.getenv("PORT", 7860))
    print(f"Starting voice assistant on port {port}...")
    uvicorn.run("sockets:fastapi_app", host="0.0.0.0", port=port, log_level="info")