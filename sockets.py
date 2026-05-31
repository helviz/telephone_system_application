import os
import sys
import time
import asyncio
from contextlib import asynccontextmanager

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
    """
    Runs synchronously at startup, before the event loop starts accepting
    connections. Mirrors preload.py but populates module-level variables in
    THIS process so they are visible to every WebSocket session.
    """
    global _preloaded_tts, _preloaded_whisper

    print("\n" + "=" * 60)
    print("🔥 PRELOADING ALL MODELS — server will start after this")
    print("=" * 60 + "\n")

    total_start = time.time()

    # ------------------------------------------------------------------
    # STT — Faster-Whisper
    # ------------------------------------------------------------------
    print("📦 [1/3] Loading Faster-Whisper ...")
    t = time.time()
    try:
        from faster_whisper import WhisperModel

        whisper_device     = os.getenv("WHISPER_DEVICE", "cpu").strip()
        whisper_model_size = os.getenv("WHISPER_MODEL_SIZE", "medium").strip()
        compute_type       = "float16" if whisper_device == "cuda" else "int8"

        _preloaded_whisper = WhisperModel(
            whisper_model_size,
            device=whisper_device,
            compute_type=compute_type,
            download_root=os.getenv("HF_HOME"),
        )
        print(f"   ✅ Whisper [{whisper_model_size}] on [{whisper_device}] — {time.time()-t:.1f}s\n")
    except Exception as e:
        print(f"   ❌ Whisper failed: {e}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # TTS — Facebook MMS-TTS (en / fr / sw)
    # ------------------------------------------------------------------
    print("📦 [2/3] Loading MMS-TTS models (en / fr / sw) ...")
    t = time.time()
    try:
        import torch
        from transformers import VitsModel, AutoTokenizer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        tts_models = {
            "en": "facebook/mms-tts-eng",
            "fr": "facebook/mms-tts-fra",
            "sw": "facebook/mms-tts-swh",
        }

        for lang, model_name in tts_models.items():
            lt = time.time()
            print(f"   Loading {lang} → {model_name} ...")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model     = VitsModel.from_pretrained(model_name).to(device)
            model.eval()
            _preloaded_tts[lang] = (model, tokenizer)
            print(f"   ✅ {lang} ready — {time.time()-lt:.1f}s")

        print(f"   ✅ All TTS models on [{device}] — {time.time()-t:.1f}s\n")
    except Exception as e:
        print(f"   ❌ TTS failed: {e}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # LLM — GGUF singleton (only when LLM_PROVIDER=qwen/gguf/local)
    # ------------------------------------------------------------------
    llm_provider = os.getenv("LLM_PROVIDER", "gemini").strip().lower()

    if llm_provider in ("qwen", "gguf", "local"):
        print("📦 [3/3] Loading GGUF LLM ...")
        t = time.time()
        try:
            from llmModule.GGUFStrategy import _get_llm
            _get_llm()
            print(f"   ✅ GGUF LLM ready — {time.time()-t:.1f}s\n")
        except Exception as e:
            print(f"   ❌ GGUF LLM failed: {e}")
            sys.exit(1)
    else:
        print(f"📦 [3/3] LLM provider [{llm_provider}] — skipping GGUF preload.\n")

    elapsed = time.time() - total_start
    print("=" * 60)
    print(f"✅ ALL MODELS READY — {elapsed:.1f}s total")
    print("🚀 Handing off to uvicorn...\n")
    print("=" * 60 + "\n")

    # Write model metadata into the shared stats store
    stats.model_info.update({
        "whisper_size":       os.getenv("WHISPER_MODEL_SIZE", "medium"),
        "whisper_device":     os.getenv("WHISPER_DEVICE", "cpu"),
        "tts_languages":      list(_preloaded_tts.keys()),
        "llm_provider":       os.getenv("LLM_PROVIDER", "gemini"),
        "llm_model":          "gemma-4-26b-a4b-it",
        "preload_ok":         True,
        "preload_duration_s": round(elapsed, 1),
    })


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
    print(f"   LLM Engine: {LLM_PROVIDER.upper()}")
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

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 7860))
    print(f"Starting voice assistant on port {port}...")
    uvicorn.run(fastapi_app, host="0.0.0.0", port=port)