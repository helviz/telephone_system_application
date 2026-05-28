import os
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.wsgi import WSGIMiddleware
from dotenv import load_dotenv

from routes import app as flask_app
from VoiceAssistant import VoiceAssistant
from Audio.PhoneStreamSource import PhoneStreamSource
from Audio.PhoneAudioOutput import PhoneAudioOutput

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")

fastapi_app = FastAPI()

# ---------------------------------------------------------------------------
# Pull preloaded models from preload.py if they are already in memory.
# preload.py stores them as module-level variables; we import them here so
# every WebSocket session reuses the same loaded objects — no per-call I/O.
# ---------------------------------------------------------------------------
_preloaded_tts: dict = {}
_preloaded_whisper = None

try:
    import preload
    _preloaded_tts     = preload._tts_store       # {lang: (model, tokenizer)}
    _preloaded_whisper = preload._whisper          # WhisperModel instance
    print("[sockets] ✅ Preloaded models attached.")
except Exception as e:
    print(f"[sockets] Could not attach preloaded models: {e} — will load on demand.")


@fastapi_app.websocket("/media-stream/{provider}/{lang}")
async def handle_media_stream(websocket: WebSocket, provider: str, lang: str):
    supported_providers = {"twilio", "telnyx"}
    supported_langs     = {"en", "fr", "sw"}

    if provider not in supported_providers or lang not in supported_langs:
        print(f"[REJECTED] Unsupported connection attempt: Provider={provider}, Lang={lang}")
        await websocket.close(code=1008)
        return

    await websocket.accept()

    print("\n" + "=" * 60)
    print(f"   INBOUND CALL CONNECTED SUCCESSFULLY")
    print(f"   Provider:  {provider.upper()}")
    print(f"   Language:  {lang.upper()}")
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
            print(f"\n❌ [{provider.upper()}] Call Ended (Caller hung up)")
        except Exception as e:
            print(f"\n [{provider.upper()}] Receiver error: {e}")

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
        print(f" [{provider.upper()}] Session cleaned up.\n")


fastapi_app.mount("/", WSGIMiddleware(flask_app))

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 7860))
    print(f" Starting voice assistant server on port {port}...")
    uvicorn.run(fastapi_app, host="0.0.0.0", port=port)