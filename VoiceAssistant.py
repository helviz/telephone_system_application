import asyncio
from llmModule.LLM import LLM
from transcribe.SSTModule import STTModule
from tts.TTSModule import TTSModule


class VoiceAssistant:
    def __init__(self, source, output, provider="gemini", lang="en"):
        allowed_langs = ["en", "fr", "sw"]
        if lang not in allowed_langs:
            raise ValueError(f"Unsupported language: {lang}. Choose from {allowed_langs}")

        self.lang = lang
        self.source = source

        self.stt = STTModule(model_size="medium", lang=self.lang)
        self.llm = LLM.get_model(provider=provider, lang=self.lang)

        self.audio_output = output
        self.tts = TTSModule(output=self.audio_output)

        # Tracks in-flight LLM/TTS tasks so we can cancel them on hangup
        self._tasks: list[asyncio.Task] = []

    async def handle_text(self, text: str):
        try:
            print(f"--- Processing [{self.lang}]: {text} ---")
            llm_stream = self.llm.generate_stream(text)
            await self.tts.speak_stream(llm_stream, lang=self.lang)
        except asyncio.CancelledError:
            pass  # Call ended mid-response — that's fine
        except Exception as e:
            print(f"[VoiceAssistant] Pipeline error: {e}")

    async def start(self):
        print(f"--- Voice Assistant Active [{self.lang.upper()}] ---")
        try:
            # transcribe_stream is now an async generator (fixed in STTModule)
            async for text in self.stt.transcribe_stream(self.source):
                if text.strip():
                    # create_task keeps listening while the previous response plays
                    task = asyncio.create_task(self.handle_text(text))
                    self._tasks.append(task)
                    # Clean up finished tasks to avoid unbounded list growth
                    self._tasks = [t for t in self._tasks if not t.done()]
        except Exception as e:
            print(f"[VoiceAssistant] Error: {e}")
        finally:
            # Cancel any still-running TTS tasks when the call ends
            for task in self._tasks:
                task.cancel()