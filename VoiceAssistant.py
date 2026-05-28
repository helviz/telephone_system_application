import asyncio
from llmModule.LLM import LLM
from transcribe.SSTModule import STTModule
from tts.TTSModule import TTSModule


class VoiceAssistant:
    def __init__(
        self,
        source,
        output,
        provider="gemini",
        lang="en",
        preloaded_tts: dict = None,
        preloaded_whisper=None,
    ):
        allowed_langs = ["en", "fr", "sw"]
        if lang not in allowed_langs:
            raise ValueError(f"Unsupported language: {lang}. Choose from {allowed_langs}")

        self.lang   = lang
        self.source = source

        # STT — reuse preloaded WhisperModel if available
        self.stt = STTModule(
            model_size="small",
            lang=self.lang,
            preloaded_model=preloaded_whisper,
        )

        # LLM — GGUF singleton is already warm if provider=qwen
        self.llm = LLM.get_model(provider=provider, lang=self.lang)

        # TTS — pass preloaded models so no disk I/O happens mid-call
        self.audio_output = output
        self.tts = TTSModule(
            output=self.audio_output,
            preloaded_models=preloaded_tts,
        )

        self._tasks: list[asyncio.Task] = []

    async def handle_text(self, text: str):
        try:
            print(f"--- Processing [{self.lang}]: {text} ---")
            llm_stream = self.llm.generate_stream(text)
            await self.tts.speak_stream(llm_stream, lang=self.lang)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"[VoiceAssistant] Pipeline error: {e}")

    async def start(self):
        print(f"--- Voice Assistant Active [{self.lang.upper()}] ---")
        try:
            async for text in self.stt.transcribe_stream(self.source):
                if text.strip():
                    task = asyncio.create_task(self.handle_text(text))
                    self._tasks.append(task)
                    self._tasks = [t for t in self._tasks if not t.done()]
        except Exception as e:
            print(f"[VoiceAssistant] Error: {e}")
        finally:
            for task in self._tasks:
                task.cancel()