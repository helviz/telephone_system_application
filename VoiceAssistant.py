import asyncio
import time
import stats
from llmModule.LLM import LLM
from transcribe.SSTModule import STTModule
from tts.TTSModule import TTSModule


class VoiceAssistant:

    def __init__(
            self,
            source,
            output,
            provider="gguf",
            lang="en",
            preloaded_tts: dict = None,
            preloaded_stt=None,
            preloaded_whisper=None,  # backward compatibility with older sockets.py
            on_turn_logged=None,
    ):
        allowed_langs = ["en", "fr", "sw"]
        if lang not in allowed_langs:
            raise ValueError(f"Unsupported language: {lang}. Choose from {allowed_langs}")

        self.lang = lang
        self.source = source
        self.on_turn_logged = on_turn_logged  #  Saved as instance variable

        # Prefer the language-keyed OpenAI Whisper STT store from sockets.py.
        # If an older caller still passes preloaded_whisper, STTModule remains backward compatible.
        self.stt = STTModule(
            model_size=None,
            lang=self.lang,
            preloaded_model=preloaded_stt if preloaded_stt is not None else preloaded_whisper,
            on_speech_start=self._on_caller_speech_start,
        )

        self.llm = LLM.get_model(provider=provider, lang=self.lang)

        self.audio_output = output

        # Pass Soniox TTS configuration from sockets.py. Soniox is API-backed,
        # so this dict contains config only, not local Kokoro/OmniVoice weights.
        self.tts = TTSModule(
            output=self.audio_output,
            preloaded_models=preloaded_tts
        )

        self._active_task: asyncio.Task | None = None

    async def handle_text(self, text: str):
        print(f"[User]: {text}")
        t_e2e = time.time()

        # 1. LOG THE USER TURN (STT Transcript) ASYNCHRONOUSLY
        if self.on_turn_logged:
            try:
                await self.on_turn_logged(role="user", text=text)
            except Exception as log_err:
                print(f"[VoiceAssistant] Error logging user turn to DB: {log_err}")

        # FIX: Initialize the list OUTSIDE the try block so it is guaranteed
        # to be assigned before any exception blocks run.
        full_assistant_response = []

        try:
            # Gather stream responses from LLM layer
            llm_stream = self.llm.generate_stream(text)

            async def _measured_llm_stream():
                _first = True
                async for chunk in llm_stream:
                    if _first:
                        stats.record_llm_latency(time.time() - t_e2e)
                        _first = False

                    # Accumulate text tokens as they stream from the LLM engine
                    if chunk:
                        full_assistant_response.append(chunk)
                    yield chunk

            def _on_first_audio():
                stats.record_e2e_latency(time.time() - t_e2e)

            # Send the active language context downstream so TTSModule knows exactly
            # which weights to allocate in RAM if it hasn't cached them yet.
            await self.tts.speak_stream(
                _measured_llm_stream(),
                lang=self.lang,
                on_first_audio=_on_first_audio,
            )

            # 2. LOG THE ASSISTANT TURN (LLM Output) ONCE THE AUDIO STREAM FINISHES
            final_text = "".join(full_assistant_response).strip()
            if final_text and self.on_turn_logged:
                try:
                    await self.on_turn_logged(role="assistant", text=final_text)
                except Exception as log_err:
                    print(f"[VoiceAssistant] Error logging assistant turn to DB: {log_err}")

        except asyncio.CancelledError:
            # Barge-in: caller spoke while we were responding — clean exit
            print(f"[VoiceAssistant] 🔇 Barge-in detected — response cancelled.")

            # Safe to read now because it was initialized before the try block
            partial_text = "".join(full_assistant_response).strip()
            if partial_text and self.on_turn_logged:
                try:
                    await self.on_turn_logged(role="assistant", text=f"{partial_text}... [Interrupted]")
                except Exception as log_err:
                    print(f"[VoiceAssistant] Error logging partial response: {log_err}")
            raise  # Re-raise so the task wrapper exits cleanly

        except Exception as e:
            print(f"[VoiceAssistant] Pipeline error: {e}")

    async def speak_greeting(self, text: str):
        """
        Speak the post-IVR greeting through the configured TTS module.

        This bypasses the LLM and is used immediately after the caller selects
        a language from the static IVR WAV menu.
        """
        text = (text or "").strip()
        if not text:
            return

        print(f"[Assistant Greeting]: {text}")

        async def _one_chunk():
            yield text

        try:
            await self.tts.speak_stream(
                _one_chunk(),
                lang=self.lang,
                on_first_audio=None,
            )

            if self.on_turn_logged:
                try:
                    await self.on_turn_logged(role="assistant", text=text)
                except Exception as log_err:
                    print(f"[VoiceAssistant] Error logging greeting turn to DB: {log_err}")

        except asyncio.CancelledError:
            print("[VoiceAssistant] Greeting playback cancelled.")
            raise
        except Exception as e:
            print(f"[VoiceAssistant] Greeting TTS error: {e}")

    async def _clear_audio_output(self):
        """Stop queued/playing assistant audio when the caller barges in."""
        for method_name in ("clear", "clear_buffer", "interrupt", "stop_playback"):
            method = getattr(self.audio_output, method_name, None)
            if callable(method):
                result = method()
                if asyncio.iscoroutine(result):
                    await result
                return

    async def _on_caller_speech_start(self):
        """
        Called by STT immediately when WebRTC VAD detects new caller speech.
        This stops TTS before waiting for Whisper to finish transcription.
        """
        if self._active_task and not self._active_task.done():
            print("[VoiceAssistant] 🎙️ Caller barge-in — stopping assistant audio now.")
            await self._clear_audio_output()
            await self._cancel_active()

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