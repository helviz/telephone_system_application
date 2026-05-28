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

# Set the fallback model provider in your .env ("gemini" or "qwen")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")

fastapi_app = FastAPI()


@fastapi_app.websocket("/media-stream/{provider}/{lang}")
async def handle_media_stream(websocket: WebSocket, provider: str, lang: str):
    """
    Accepts a WebSocket media stream from Twilio or Telnyx with a dynamic language path.
    The {provider} must be 'twilio' or 'telnyx', and {lang} must be 'en', 'fr', or 'sw'.
    """
    supported_providers = {"twilio", "telnyx"}
    supported_langs = {"en", "fr", "sw"}

    if provider not in supported_providers or lang not in supported_langs:
        print(f"[REJECTED] Unsupported connection attempt: Provider={provider}, Lang={lang}")
        await websocket.close(code=1008)  # Policy Violation
        return

    await websocket.accept()

    # Clear console logs showing a successful connection
    print("\n" + "=" * 60)
    print(f"📞 INBOUND CALL CONNECTED SUCCESSFULY")
    print(f"   Provider: {provider.upper()}")
    print(f"   Language: {lang.upper()}")
    print(f"   LLM Engine: {LLM_PROVIDER.upper()}")
    print("=" * 60 + "\n")

    source = PhoneStreamSource(provider=provider)
    output = PhoneAudioOutput(websocket, provider=provider)

    # Pass the dynamic language chosen by the user in the IVR menu
    assistant = VoiceAssistant(
        source=source,
        output=output,
        provider=LLM_PROVIDER,
        lang=lang,
    )

    async def websocket_receiver():
        """Reads raw WebSocket packets and feeds them into the audio source queue."""
        try:
            while True:
                message = await websocket.receive_text()
                await source.add_data(message)
        except WebSocketDisconnect:
            print(f"\n❌ [{provider.upper()}] Call Ended (Caller hung up) ---")
        except Exception as e:
            print(f"\n⚠️ [{provider.upper()}] Receiver error: {e} ---")

    # Run receiver and AI pipeline concurrently
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
        print(f"🧹 [{provider.upper()}] Session cleaned up and resources released.\n")


# Mount Flask under "/" so Twilio/Telnyx HTTP webhooks and WebSockets share one port
fastapi_app.mount("/", WSGIMiddleware(flask_app))

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 7860))
    print(f"🚀 Starting voice assistant server on port {port}...")
    uvicorn.run(fastapi_app, host="0.0.0.0", port=port)