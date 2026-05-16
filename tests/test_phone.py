import os
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.wsgi import WSGIMiddleware
from dotenv import load_dotenv

from routes import app as flask_app
from transcribe.SSTModule import STTModule
from tts.TTSModule import TTSModule
from Audio.PhoneStreamSource import PhoneStreamSource
from Audio.PhoneAudioOutput import PhoneAudioOutput

load_dotenv()

fastapi_app = FastAPI()


# ===========================================================================
# LIGHTWEIGHT BYPASS ASSISTANT FOR TESTING
# ===========================================================================
class TestVoiceAssistant:
    """A direct loopback assistant that skips the LLM and mocks responses."""

    def __init__(self, source, output, lang="en"):
        self.lang = lang
        self.source = source

        # Initialize STT and TTS directly
        print(f"[TestMode] Initializing Whisper STT (medium) for language: {self.lang}...")
        self.stt = STTModule(model_size="medium", lang=self.lang)
        self.audio_output = output
        self.tts = TTSModule(output=self.audio_output)

        self._tasks: list[asyncio.Task] = []

    async def handle_text(self, text: str):
        try:
            clean_text = text.strip()
            print(f"🎙️  [STT Detected]: \"{clean_text}\"")

            # --- BYPASS LLM GENERATION ---
            # Create a hardcoded response template based on what the user said
            if self.lang == "fr":
                demo_response = f"J'ai entendu : {clean_text}. Ceci est une réponse de démonstration sans IA."
            elif self.lang == "sw":
                demo_response = f"Nimesikia: {clean_text}. Hili ni jibu la jaribio bila mfumo wa akili bandia."
            else:
                demo_response = f"I heard you say: {clean_text}. This is a demo response without the LLM."

            print(f"🤖 [TTS Playback]: \"{demo_response}\"")

            # Convert our raw string into an async generator to mimic an LLM stream
            async def mock_llm_stream():
                yield demo_response

            # Stream the demo text straight to the TTS module
            await self.tts.speak_stream(mock_llm_stream(), lang=self.lang)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"[TestVoiceAssistant] Error processing stream: {e}")

    async def start(self):
        print(f"\n🚀 --- TEST VOICE ASSISTANT ACTIVE [{self.lang.upper()}] (LLM BYPASSED) ---")
        try:
            # Continuously read speech fragments from the incoming phone audio source
            async for text in self.stt.transcribe_stream(self.source):
                if text.strip():
                    task = asyncio.create_task(self.handle_text(text))
                    self._tasks.append(task)
                    self._tasks = [t for t in self._tasks if not t.done()]
        except Exception as e:
            print(f"[TestVoiceAssistant] Stream Error: {e}")
        finally:
            for task in self._tasks:
                task.cancel()


# ===========================================================================
# WEBSOCKET STREAM ROUTER
# ===========================================================================
@fastapi_app.websocket("/media-stream/{provider}/{lang}")
async def handle_media_stream(websocket: WebSocket, provider: str, lang: str):
    supported_providers = {"twilio", "telnyx"}
    supported_langs = {"en", "fr", "sw"}

    if provider not in supported_providers or lang not in supported_langs:
        print(f"[REJECTED] Unsupported connection: Provider={provider}, Lang={lang}")
        await websocket.close(code=1008)
        return

    await websocket.accept()

    print("\n" + "=" * 60)
    print(f"📞 TEST CALL CONNECTED | BYPASSING LLM")
    print(f"   Provider: {provider.upper()}")
    print(f"   Language: {lang.upper()}")
    print("=" * 60 + "\n")

    source = PhoneStreamSource(provider=provider)
    output = PhoneAudioOutput(websocket, provider=provider)

    # Instantiate our local Test Assistant instead of production VoiceAssistant
    assistant = TestVoiceAssistant(
        source=source,
        output=output,
        lang=lang,
    )

    async def websocket_receiver():
        try:
            while True:
                message = await websocket.receive_text()
                await source.add_data(message)
        except WebSocketDisconnect:
            print(f"\n❌ [{provider.upper()}] Call Ended (Caller hung up) ---")
        except Exception as e:
            print(f"\n⚠️ [{provider.upper()}] Receiver error: {e} ---")

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
        print(f"🧹 [{provider.upper()}] Test session cleaned up.\n")


# Mount Flask under "/" so webhooks and WebSockets coexist seamlessly
fastapi_app.mount("/", WSGIMiddleware(flask_app))

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 5000))
    print(f"🚀 Starting LLM-Bypass Test Server on port {port}...")
    uvicorn.run(fastapi_app, host="0.0.0.0", port=port)