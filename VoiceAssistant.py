import asyncio
import time
import stats
from llmModule.LLM import LLM
from transcribe.SSTModule import STTModule
from tts.TTSModule import TTSModule


class VoiceAssistant:
    """
    FIX 4: The original start() spawned a new asyncio.Task for every
    transcribed utterance without cancelling the previous one. If the caller
    spoke again while the assistant was still generating audio, two (or more)
    LLM→TTS pipelines ran concurrently, producing overlapping audio and
    wasting compute.

    Fix: keep a reference to the single active handle_text task. When a new
    utterance arrives, cancel the in-flight task first (barge-in behaviour),
    then start a fresh one. CancelledError is suppressed inside handle_text
    so the cancellation is clean.
    """

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

        self.lang = lang
        self.source = source

        self.stt = STTModule(
            model_size="small",
            lang=self.lang,
            preloaded_model=preloaded_whisper,
        )

        self.llm = LLM.get_model(provider=provider, lang=self.lang)

        self.audio_output = output

        # Pass the preloaded_tts dict safely. TTSModule handles the lazy-loading fallback
        # seamlessly if this dictionary comes in empty from sockets.py.
        self.tts = TTSModule(
            output=self.audio_output,
            preloaded_models=preloaded_tts
        )

        self._active_task: asyncio.Task | None = None

    async def handle_text(self, text: str):
        print(f"[User]: {text}")
        t_e2e = time.time()

        try:
            # Gather stream responses from LLM layer
            llm_stream = self.llm.generate_stream(text)

            async def _measured_llm_stream():
                _first = True
                async for chunk in llm_stream:
                    if _first:
                        stats.record_llm_latency(time.time() - t_e2e)
                        _first = False
                    yield chunk

            def _on_first_audio():
                stats.record_e2e_latency(time.time() - t_e2e)

            # Modified: Send the active language context downstream so TTSModule
            # knows exactly which weights to allocate in RAM if it hasn't cached them yet.
            await self.tts.speak_stream(
                _measured_llm_stream(),
                lang=self.lang,
                on_first_audio=_on_first_audio,
            )
        except asyncio.CancelledError:
            # Barge-in: caller spoke while we were responding — clean exit
            print(f"[VoiceAssistant] 🔇 Barge-in detected — response cancelled.")
        except Exception as e:
            print(f"[VoiceAssistant] Pipeline error: {e}")

    async def _cancel_active(self):
        """Cancel the in-flight LLM→TTS task and wait for it to finish."""
        if self._active_task and not self._active_task.done():
            self._active_task.cancel()
            try:
                await self._active_task
            except (asyncio.CancelledError, Exception):
                pass
        self._active_task = None

    async def start(self):
        print(f"--- Voice Assistant Active [{self.lang.upper()}] ---")
        try:
            async for text in self.stt.transcribe_stream(self.source):
                if text.strip():
                    # FIX 4: cancel whatever is playing before starting a new response
                    await self._cancel_active()
                    self._active_task = asyncio.create_task(self.handle_text(text))
        except Exception as e:
            print(f"[VoiceAssistant] Error: {e}")
        finally:
            await self._cancel_active()