import asyncio
from fastapi import FastAPI, WebSocket
from fastapi.middleware.wsgi import WSGIMiddleware
from routes import app as flask_app
from VoiceAssistant import VoiceAssistant
from Audio.PhoneStreamSource import PhoneStreamSource
from Audio.PhoneAudioOutput import PhoneAudioOutput

fastapi_app = FastAPI()

@fastapi_app.websocket("/media-stream/{provider}")
async def handle_media_stream(websocket: WebSocket, provider: str):
    await websocket.accept()
    print(f"--- {provider.upper()} Call Connected ---")

    # 1. Initialize bridges for phone audio logic
    source = PhoneStreamSource(provider=provider)
    output = PhoneAudioOutput(websocket, provider=provider)

    # 2. Initialize the Assistant (Provider can be 'gemini' or 'qwen')
    assistant = VoiceAssistant(
        source=source,
        output=output,
        provider="gemini",
        lang="en"
    )

    async def websocket_receiver():
        """Feeds raw WebSocket JSON packets into the Assistant's source."""
        try:
            while True:
                message = await websocket.receive_text()
                await source.add_data(message)
        except Exception:
            print(f"--- {provider.upper()} Call Ended ---")

    # 3. Run the receiver and the AI pipeline concurrently
    await asyncio.gather(
        websocket_receiver(),
        assistant.start()
    )

# 4. Mount the Flask app so everything runs on one port
# Flask routes will be available under the '/web' prefix or as defined
fastapi_app.mount("/", WSGIMiddleware(flask_app))

if __name__ == "__main__":
    import uvicorn
    # Run the unified server
    uvicorn.run(fastapi_app, host="0.0.0.0", port=5000)