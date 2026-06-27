import asyncio
import os
import time

import stats
from llmModule.LLM import LLM
from safety.ASRConfidence import ASRConfidenceChecker, TranscriptionResult
from safety.EscalationManager import EscalationManager
from safety.SafetyFilter import SafetyFilter, SafetyResult
from safety.SafetyMessages import get_safety_message
from transcribe.SSTModule import STTModule
from tts.TTSModule import TTSModule


def _env_bool(name: str, default: bool = True) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int, min_value: int | None = None, max_value: int | None = None) -> int:
    try:
        value = int(os.getenv(name, str(default)).strip())
    except Exception:
        value = default
    if min_value is not None:
        value = max(min_value, value)
    if max_value is not None:
        value = min(max_value, value)
    return value


class SafetyEscalationTriggered(Exception):
    def __init__(self, message_key: str, result: SafetyResult | None = None):
        super().__init__(message_key)
        self.message_key = message_key
        self.result = result


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
        self.on_turn_logged = on_turn_logged

        self.safety_enabled = _env_bool("SAFETY_ENABLED", True)
        self.safety_filter = SafetyFilter(enabled=self.safety_enabled)
        self.asr_checker = ASRConfidenceChecker()
        self.escalation = EscalationManager()
        self.asr_failure_count = 0
        self.max_asr_failures = _env_int("SAFETY_STT_FAILURE_MAX_RETRIES", 2, 1, 10)
        self._transfer_in_progress = False

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

        print(
            f"[Safety] enabled={self.safety_enabled} "
            f"transfer_enabled={self.escalation.enabled} "
            f"max_stt_failures={self.max_asr_failures}"
        )

    @staticmethod
    def _text_from_transcription(result: TranscriptionResult | str) -> str:
        if isinstance(result, TranscriptionResult):
            return (result.text or "").strip()
        return str(result or "").strip()

    async def _log_turn(self, role: str, text: str):
        if self.on_turn_logged and text and text.strip():
            try:
                await self.on_turn_logged(role=role, text=text)
            except Exception as log_err:
                print(f"[VoiceAssistant] Error logging {role} turn to DB: {log_err}")

    async def _speak_fixed_message(self, key: str):
        message = get_safety_message(self.lang, key)
        print(f"[Safety Message][{self.lang}] {key}: {message}")

        async def _one_chunk():
            yield message

        await self.tts.speak_stream(_one_chunk(), lang=self.lang, on_first_audio=None)
        await self._log_turn("assistant", message)

    async def _transfer_to_operator(self, reason: str) -> bool:
        if self._transfer_in_progress:
            return True

        self._transfer_in_progress = True
        provider = getattr(self.source, "provider", "")
        call_id = getattr(self.source, "call_control_id", None)

        ok = await self.escalation.transfer_to_operator(
            provider=provider,
            lang=self.lang,
            reason=reason,
            twilio_call_sid=call_id if provider == "twilio" else None,
            telnyx_call_control_id=call_id if provider == "telnyx" else None,
        )
        if not ok:
            await self._speak_fixed_message("transfer_failed")
        return ok

    async def _handle_safety_escalation(self, message_key: str, reason: str):
        await self._cancel_active()
        await self._clear_audio_output()
        await self._speak_fixed_message(message_key)
        await self._transfer_to_operator(reason)

    async def _handle_asr_failure(self, reasons: list[str]):
        self.asr_failure_count += 1
        print(f"[Safety][ASR] Low-confidence transcript #{self.asr_failure_count}: {reasons}")

        if self.asr_failure_count >= self.max_asr_failures:
            await self._handle_safety_escalation("asr_failure_final", reason="asr_failure")
            return

        await self._speak_fixed_message("asr_failure")

    async def _check_user_safety_or_transfer(self, text: str) -> bool:
        if not self.safety_enabled:
            return False

        result = self.safety_filter.detect(text, self.lang)
        if not result:
            return False

        print(
            f"[Safety][ASR] category={result.category} severity={result.severity} "
            f"matched={result.matched_terms}"
        )
        await self._handle_safety_escalation(result.category, reason=result.category)
        return True

    async def handle_text(self, text: str):
        print(f"[User]: {text}")
        t_e2e = time.time()

        await self._log_turn(role="user", text=text)

        if await self._check_user_safety_or_transfer(text):
            return

        full_assistant_response = []

        try:
            llm_stream = self.llm.generate_stream(text)
            safety_scan_buffer = ""

            async def _measured_llm_stream():
                nonlocal safety_scan_buffer
                _first = True
                async for chunk in llm_stream:
                    if _first:
                        stats.record_llm_latency(time.time() - t_e2e)
                        _first = False

                    if chunk:
                        full_assistant_response.append(chunk)

                        if self.safety_enabled:
                            safety_scan_buffer = (safety_scan_buffer + chunk)[-600:]
                            detected = self.safety_filter.detect(safety_scan_buffer, self.lang)
                            if detected and detected.category != "operator_request":
                                print(
                                    f"[Safety][LLM] Unsafe assistant output detected: "
                                    f"category={detected.category} matched={detected.matched_terms}"
                                )
                                raise SafetyEscalationTriggered("unsafe_llm_output", detected)

                    yield chunk

            def _on_first_audio():
                stats.record_e2e_latency(time.time() - t_e2e)

            await self.tts.speak_stream(
                _measured_llm_stream(),
                lang=self.lang,
                on_first_audio=_on_first_audio,
            )

            final_text = "".join(full_assistant_response).strip()
            await self._log_turn(role="assistant", text=final_text)

        except SafetyEscalationTriggered as safety_event:
            await self._handle_safety_escalation(
                safety_event.message_key,
                reason=safety_event.result.category if safety_event.result else safety_event.message_key,
            )

        except asyncio.CancelledError:
            print(f"[VoiceAssistant] 🔇 Barge-in detected — response cancelled.")
            partial_text = "".join(full_assistant_response).strip()
            if partial_text:
                await self._log_turn(role="assistant", text=f"{partial_text}... [Interrupted]")
            raise

        except Exception as e:
            print(f"[VoiceAssistant] Pipeline error: {e}")

    async def speak_greeting(self, text: str):
        """Speak the post-IVR greeting through the configured TTS module."""
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
            await self._log_turn(role="assistant", text=text)

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
        """Called by STT immediately when WebRTC VAD detects new caller speech."""
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
            async for result in self.stt.transcribe_stream(self.source):
                text = self._text_from_transcription(result)

                if self.safety_enabled:
                    low_confidence, reasons = self.asr_checker.check(result)
                    if low_confidence:
                        await self._cancel_active()
                        await self._handle_asr_failure(reasons)
                        continue

                if text:
                    self.asr_failure_count = 0
                    await self._cancel_active()
                    self._active_task = asyncio.create_task(self.handle_text(text))

        except asyncio.CancelledError:
            raise
        except Exception as e:
            print(f"[VoiceAssistant] Error: {e}")
        finally:
            await self._cancel_active()
