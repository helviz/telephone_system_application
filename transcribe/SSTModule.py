import asyncio
import collections
import json
import os
import time
from contextlib import suppress

import numpy as np
import webrtcvad
from dotenv import load_dotenv

from Audio.AudioSource import AudioSource

load_dotenv()

# PhoneStreamSource normalizes Twilio/Telnyx phone audio to 16 kHz PCM16.
SAMPLE_RATE = 16000
FRAME_MS = 20
FRAME_BYTES = int(SAMPLE_RATE * FRAME_MS / 1000) * 2  # int16 = 2 bytes

SONIOX_WS_URL = "wss://stt-rt.soniox.com/transcribe-websocket"


def _env_int(name: str, default: int, min_value: int | None = None, max_value: int | None = None) -> int:
    """Read an integer env var safely and optionally clamp it."""
    try:
        value = int(os.getenv(name, str(default)).strip())
    except Exception:
        value = default

    if min_value is not None:
        value = max(min_value, value)
    if max_value is not None:
        value = min(max_value, value)
    return value


def _env_float(name: str, default: float, min_value: float | None = None, max_value: float | None = None) -> float:
    """Read a float env var safely and optionally clamp it."""
    try:
        value = float(os.getenv(name, str(default)).strip())
    except Exception:
        value = default

    if min_value is not None:
        value = max(min_value, value)
    if max_value is not None:
        value = min(max_value, value)
    return value


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


# Existing Whisper/VAD defaults. They are still used when USE_WHISPER=true.
VAD_AGGRESSIVENESS = _env_int("WEBRTC_VAD_AGGRESSIVENESS", 1, 0, 3)
END_SILENCE_MS = _env_int("STT_END_SILENCE_MS", 500, 200, 3000)
END_SILENCE_FRAMES = max(1, END_SILENCE_MS // FRAME_MS)
FINAL_GRACE_MS = _env_int("STT_FINAL_GRACE_MS", 1000, 0, 3000)
FINAL_GRACE_FRAMES = max(0, FINAL_GRACE_MS // FRAME_MS)
PADDING_MS = _env_int("STT_PADDING_MS", 250, 0, 2000)
PADDING_FRAMES = max(1, PADDING_MS // FRAME_MS)
MAX_UTTERANCE_MS = _env_int("STT_MAX_UTTERANCE_MS", 20_000, 1000, 120_000)
MAX_UTTERANCE_FRAMES = max(1, MAX_UTTERANCE_MS // FRAME_MS)
MIN_SAMPLES = _env_int("STT_MIN_SAMPLES", 6400, 1600, 160000)
MIN_RMS = _env_float("STT_MIN_RMS", 0.006, 0.0, 1.0)
FRAME_SPEECH_MIN_RMS = _env_float("STT_FRAME_SPEECH_MIN_RMS", 0.0015, 0.0, 1.0)
SPEECH_START_CONFIRM_MS = _env_int("STT_SPEECH_START_CONFIRM_MS", 120, FRAME_MS, 1000)
SPEECH_START_CONFIRM_FRAMES = max(1, SPEECH_START_CONFIRM_MS // FRAME_MS)
SPEECH_START_MIN_RMS = _env_float("STT_SPEECH_START_MIN_RMS", 0.003, 0.0, 1.0)
BARGE_IN_MIN_RMS = _env_float("STT_BARGE_IN_MIN_RMS", 0.006, 0.0, 1.0)
HALLUCINATION_MAX_SAMPLES = _env_int("STT_HALLUCINATION_MAX_SAMPLES", 24000, 1600, 160000)
BAD_SILENCE_PHRASES = {
    "thank you", "thank you.", "thanks", "thanks.", "you", "you.",
    "asante", "asante.", "merci", "merci.",
}


class STTModule:
    """
    Streaming STT for phone calls.

    Backend selection:
      - USE_WHISPER=true  -> local faster-whisper, using your existing VAD/final-grace logic.
      - USE_WHISPER=false -> Soniox realtime STT WebSocket, model stt-rt-v5 by default.

    The caller's IVR digit remains the source of truth for language selection:
      - digit 1 in routes.py -> lang="en"
      - digit 2 in routes.py -> lang="fr"
      - digit 3 in routes.py -> lang="sw"
    """

    ALLOWED_LANGUAGES = ("en", "fr", "sw")
    WHISPER_LANGUAGE_CODES = {"en": "en", "fr": "fr", "sw": "sw"}
    SONIOX_LANGUAGE_HINTS = {
        "en": ["en"],
        "fr": ["fr"],
        # Soniox uses ISO-style language hints. "sw" is the standard code for Swahili.
        "sw": ["sw"],
    }

    def __init__(self, model_size=None, device=None, lang="en", preloaded_model=None, on_speech_start=None):
        if lang not in self.ALLOWED_LANGUAGES:
            raise ValueError(f"Language '{lang}' not supported. Choose from {list(self.ALLOWED_LANGUAGES)}")

        self.lang = lang
        self.on_speech_start = on_speech_start
        self._last_speech_start_notify = 0.0
        self.use_whisper = _env_bool("USE_WHISPER", True)
        self.engine = "faster_whisper" if self.use_whisper else "soniox_realtime"
        self.model = None
        self.model_name = "unknown"
        self.device = None

        if not self.use_whisper:
            self._configure_soniox()
            return

        # Preferred path: sockets.py passes a language-keyed STT store. Each
        # language entry points to the same preloaded OpenAI Whisper model.
        if isinstance(preloaded_model, dict) and lang in preloaded_model:
            entry = preloaded_model[lang]
            if isinstance(entry, dict):
                self.engine = entry.get("engine", "faster_whisper")
                self.model = entry.get("model")
                self.model_name = entry.get("model_name", "unknown")
                self.device = entry.get("device")
                forced_language = entry.get("forced_language", self.WHISPER_LANGUAGE_CODES[lang])
            else:
                self.model = entry
                self.model_name = "preloaded-openai-whisper"
                forced_language = self.WHISPER_LANGUAGE_CODES[lang]

            if self.model is None:
                raise RuntimeError(f"No OpenAI Whisper model found for language [{lang}] in preloaded_model store.")
            if self.engine != "faster_whisper":
                raise RuntimeError(
                    f"Unsupported STT engine [{self.engine}] for [{lang}]. Expected faster_whisper/OpenAI Whisper."
                )

            print(
                f"[STT] ♻️  Reusing preloaded OpenAI Whisper [{self.model_name}] "
                f"for [{lang}] forced_language=[{forced_language}]"
            )
            return

        if preloaded_model is not None:
            self.model = preloaded_model
            self.model_name = "preloaded-openai-whisper"
            print(
                f"[STT] ♻️  Reusing legacy OpenAI Whisper model for [{lang}] "
                f"forced_language=[{self.WHISPER_LANGUAGE_CODES[lang]}]"
            )
            return

        self._load_faster_whisper_fallback(model_size=model_size, device=device)

    # ------------------------------------------------------------------
    # Common helpers
    # ------------------------------------------------------------------
    async def _notify_speech_start(self):
        """Notify the assistant immediately when caller speech starts."""
        if not self.on_speech_start:
            return

        now = time.time()
        if now - self._last_speech_start_notify < 0.5:
            return

        self._last_speech_start_notify = now

        try:
            result = self.on_speech_start()
            if asyncio.iscoroutine(result):
                await result
        except Exception as e:
            print(f"[STT] Speech-start callback failed: {e}")

    @staticmethod
    def _rms(audio_data: np.ndarray) -> float:
        if audio_data.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(np.square(audio_data))))

    @staticmethod
    def _normalize_text(text: str) -> str:
        return " ".join(text.strip().lower().split())

    @staticmethod
    def _bytes_to_float_audio(audio_bytes: bytes) -> np.ndarray:
        if len(audio_bytes) % 2 != 0:
            audio_bytes = audio_bytes[:-1]
        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        return np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

    @classmethod
    def _frame_rms(cls, frame: bytes) -> float:
        return cls._rms(cls._bytes_to_float_audio(frame))

    @staticmethod
    def _frame_generator(byte_stream_buffer: bytearray) -> list[bytes]:
        """Consume complete 20 ms PCM16 frames and keep any partial remainder."""
        frames = []
        offset = 0
        while offset + FRAME_BYTES <= len(byte_stream_buffer):
            frames.append(bytes(byte_stream_buffer[offset:offset + FRAME_BYTES]))
            offset += FRAME_BYTES
        del byte_stream_buffer[:offset]
        return frames

    # ------------------------------------------------------------------
    # Soniox realtime backend
    # ------------------------------------------------------------------
    def _configure_soniox(self):
        self.engine = "soniox_realtime"
        self.model_name = os.getenv("SONIOX_STT_MODEL", "stt-rt-v5").strip() or "stt-rt-v5"
        self.soniox_api_key = os.getenv("SONIOX_API_KEY", "").strip()
        if not self.soniox_api_key:
            raise RuntimeError("USE_WHISPER=false requires SONIOX_API_KEY in env/secrets.")

        self.soniox_url = os.getenv("SONIOX_STT_WS_URL", SONIOX_WS_URL).strip() or SONIOX_WS_URL
        self.soniox_audio_format = os.getenv("SONIOX_STT_AUDIO_FORMAT", "pcm_s16le").strip() or "pcm_s16le"
        self.soniox_sample_rate = _env_int("SONIOX_STT_SAMPLE_RATE", SAMPLE_RATE, 8000, 48000)
        self.soniox_num_channels = _env_int("SONIOX_STT_NUM_CHANNELS", 1, 1, 2)
        self.soniox_endpoint_detection = _env_bool("SONIOX_STT_ENDPOINT_DETECTION", True)
        self.soniox_language_hints_strict = _env_bool("SONIOX_STT_LANGUAGE_HINTS_STRICT", True)
        self.soniox_max_endpoint_delay_ms = _env_int("SONIOX_STT_MAX_ENDPOINT_DELAY_MS", 900, 500, 3000)
        self.soniox_endpoint_sensitivity = _env_float("SONIOX_STT_ENDPOINT_SENSITIVITY", 0.2, -1.0, 1.0)
        self.soniox_endpoint_latency_level = _env_int("SONIOX_STT_ENDPOINT_LATENCY_LEVEL", 1, 0, 3)
        self.soniox_keepalive_seconds = _env_int("SONIOX_STT_KEEPALIVE_SECONDS", 10, 5, 19)
        self.soniox_manual_finalize = _env_bool("SONIOX_STT_MANUAL_FINALIZATION", False)

        hints_env = os.getenv(f"SONIOX_STT_LANG_HINTS_{self.lang.upper()}", "").strip()
        self.soniox_language_hints = (
            [x.strip() for x in hints_env.split(",") if x.strip()]
            if hints_env else self.SONIOX_LANGUAGE_HINTS[self.lang]
        )

        print(
            f"[STT] 🌐 Soniox realtime configured: model=[{self.model_name}] "
            f"lang_hints={self.soniox_language_hints} strict={self.soniox_language_hints_strict} "
            f"audio={self.soniox_audio_format}/{self.soniox_sample_rate}Hz "
            f"endpoint_detection={self.soniox_endpoint_detection} "
            f"manual_finalize={self.soniox_manual_finalize}"
        )

    def _soniox_config(self) -> dict:
        config = {
            "api_key": self.soniox_api_key,
            "model": self.model_name,
            "audio_format": self.soniox_audio_format,
            "language_hints": self.soniox_language_hints,
            "language_hints_strict": self.soniox_language_hints_strict,
            "enable_endpoint_detection": self.soniox_endpoint_detection,
            "client_reference_id": f"voice-assistant-{self.lang}",
        }

        if self.soniox_audio_format != "auto":
            config["sample_rate"] = self.soniox_sample_rate
            config["num_channels"] = self.soniox_num_channels

        if self.soniox_endpoint_detection:
            config["max_endpoint_delay_ms"] = self.soniox_max_endpoint_delay_ms
            config["endpoint_sensitivity"] = self.soniox_endpoint_sensitivity
            config["endpoint_latency_adjustment_level"] = self.soniox_endpoint_latency_level

        return config

    @staticmethod
    def _tokens_to_text(tokens: list[dict]) -> str:
        return "".join(str(t.get("text", "")) for t in tokens if t.get("text") not in {"<end>", "<fin>"}).strip()

    async def _soniox_sender(self, ws, audio_source: AudioSource, result_queue: asyncio.Queue):
        """Send PCM16 chunks to Soniox and optionally run local VAD for barge-in/manual finalize."""
        vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
        byte_buffer = bytearray()
        last_audio_at = time.time()
        last_keepalive_at = time.time()
        in_speech = False
        trailing_silence = 0
        speech_confirm_frames = 0
        speech_rms_total = 0.0
        sent_any_audio = False

        async def maybe_keepalive():
            nonlocal last_keepalive_at
            now = time.time()
            if now - last_audio_at >= self.soniox_keepalive_seconds and now - last_keepalive_at >= self.soniox_keepalive_seconds:
                await ws.send(json.dumps({"type": "keepalive"}))
                last_keepalive_at = now
                print(f"[STT][Soniox] keepalive sent after {now - last_audio_at:.1f}s without audio")

        try:
            async for chunk in audio_source.get_stream():
                if not chunk or not isinstance(chunk, (bytes, bytearray)):
                    await maybe_keepalive()
                    await asyncio.sleep(0)
                    continue

                # Send every chunk to Soniox so semantic endpointing has full context.
                await ws.send(bytes(chunk))
                sent_any_audio = True
                last_audio_at = time.time()

                # Local VAD is only for fast barge-in notification and optional manual finalization.
                byte_buffer.extend(chunk)
                frames = self._frame_generator(byte_buffer)
                for frame in frames:
                    try:
                        vad_speech = vad.is_speech(frame, SAMPLE_RATE)
                    except Exception:
                        vad_speech = False

                    frame_rms = self._frame_rms(frame)
                    is_speech = vad_speech and frame_rms >= FRAME_SPEECH_MIN_RMS

                    if not in_speech:
                        if is_speech:
                            speech_confirm_frames += 1
                            speech_rms_total += frame_rms
                            avg_rms = speech_rms_total / max(1, speech_confirm_frames)
                            if speech_confirm_frames >= SPEECH_START_CONFIRM_FRAMES and avg_rms >= SPEECH_START_MIN_RMS:
                                in_speech = True
                                trailing_silence = 0
                                print(f"[STT][Soniox] Local speech start confirmed avg_rms={avg_rms:.5f}")
                                if avg_rms >= BARGE_IN_MIN_RMS:
                                    await self._notify_speech_start()
                        else:
                            speech_confirm_frames = 0
                            speech_rms_total = 0.0
                        continue

                    if is_speech:
                        trailing_silence = 0
                    else:
                        trailing_silence += 1

                    if self.soniox_manual_finalize and in_speech and trailing_silence >= END_SILENCE_FRAMES:
                        # Soniox recommends finalizing only after ~200ms silence after speech.
                        await ws.send(json.dumps({"type": "finalize"}))
                        print(f"[STT][Soniox] manual finalize sent after {END_SILENCE_MS}ms silence")
                        in_speech = False
                        trailing_silence = 0
                        speech_confirm_frames = 0
                        speech_rms_total = 0.0

                await maybe_keepalive()

            # Audio source ended. Gracefully end Soniox stream.
            if sent_any_audio:
                await ws.send(b"")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await result_queue.put({"type": "error", "message": f"Soniox sender error: {exc}"})

    async def _soniox_receiver(self, ws, result_queue: asyncio.Queue):
        final_tokens: list[dict] = []
        turn_started_at: float | None = None

        try:
            async for raw in ws:
                try:
                    event = json.loads(raw)
                except Exception:
                    print(f"[STT][Soniox] Non-JSON response ignored: {raw!r}")
                    continue

                if event.get("error_code") or event.get("error_type"):
                    await result_queue.put({
                        "type": "error",
                        "message": f"Soniox error {event.get('error_code')} {event.get('error_type')}: {event.get('error_message')}",
                    })
                    return

                if event.get("finished"):
                    if final_tokens:
                        await result_queue.put({"type": "text", "text": self._tokens_to_text(final_tokens), "started_at": turn_started_at})
                    await result_queue.put({"type": "done"})
                    return

                tokens = event.get("tokens") or []
                if not tokens:
                    continue

                has_endpoint = any(t.get("text") == "<end>" for t in tokens)
                has_finalize = any(t.get("text") == "<fin>" for t in tokens)
                real_tokens = [t for t in tokens if t.get("text") not in {"<end>", "<fin>"}]

                if real_tokens and turn_started_at is None:
                    turn_started_at = time.time()
                    # Backup speech-start signal if local VAD missed it.
                    await self._notify_speech_start()

                for token in real_tokens:
                    if token.get("is_final"):
                        final_tokens.append(token)

                if has_endpoint or has_finalize:
                    text = self._tokens_to_text(final_tokens)
                    final_tokens = []
                    if text:
                        await result_queue.put({"type": "text", "text": text, "started_at": turn_started_at})
                    turn_started_at = None

        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await result_queue.put({"type": "error", "message": f"Soniox receiver error: {exc}"})

    async def _transcribe_soniox_stream(self, audio_source: AudioSource):
        try:
            import websockets
        except Exception as exc:
            raise RuntimeError(
                "USE_WHISPER=false requires the `websockets` package. Add `websockets` to requirements.txt."
            ) from exc

        print(
            f"--- STT Active: Soniox realtime [{self.lang}] model={self.model_name}, "
            f"endpoint_delay={self.soniox_max_endpoint_delay_ms}ms, "
            f"endpoint_sensitivity={self.soniox_endpoint_sensitivity}, "
            f"latency_level={self.soniox_endpoint_latency_level} ---"
        )

        result_queue: asyncio.Queue = asyncio.Queue()
        async with websockets.connect(self.soniox_url, ping_interval=15, ping_timeout=15, max_size=None) as ws:
            await ws.send(json.dumps(self._soniox_config()))
            sender_task = asyncio.create_task(self._soniox_sender(ws, audio_source, result_queue))
            receiver_task = asyncio.create_task(self._soniox_receiver(ws, result_queue))

            try:
                while True:
                    item = await result_queue.get()
                    kind = item.get("type")

                    if kind == "text":
                        text = (item.get("text") or "").strip()
                        if not text:
                            continue
                        started_at = item.get("started_at")
                        if started_at:
                            with suppress(Exception):
                                import stats
                                stats.record_stt_latency(time.time() - started_at)
                        print(f"[STT][Soniox] Final turn -> {text!r}")
                        yield text

                    elif kind == "error":
                        print(f"[STT][Soniox] ⚠️ {item.get('message')}")
                        break

                    elif kind == "done":
                        break
            finally:
                for task in (sender_task, receiver_task):
                    task.cancel()
                    with suppress(asyncio.CancelledError, Exception):
                        await task

    # ------------------------------------------------------------------
    # faster-whisper backend (your existing implementation)
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_model_name(model_size=None) -> str:
        return (
            model_size
            or os.getenv("OPENAI_WHISPER_MODEL")
            or os.getenv("WHISPER_MODEL_SIZE")
            or "large-v3"
        ).strip()

    @staticmethod
    def _resolve_device(device=None) -> str:
        return (device or os.getenv("WHISPER_DEVICE") or "cpu").strip()

    @staticmethod
    def _resolve_compute_type(device: str) -> str:
        return (os.getenv("WHISPER_COMPUTE_TYPE") or ("float16" if device == "cuda" else "int8")).strip()

    def _load_faster_whisper_fallback(self, model_size=None, device=None):
        from faster_whisper import WhisperModel

        resolved_size = self._resolve_model_name(model_size)
        resolved_device = self._resolve_device(device)
        compute_type = self._resolve_compute_type(resolved_device)

        print(
            f"[STT] 📦 Loading OpenAI Whisper/faster-whisper [{resolved_size}] on [{resolved_device}] "
            f"(compute_type={compute_type}) forced_language=[{self.WHISPER_LANGUAGE_CODES[self.lang]}]..."
        )
        self.engine = "faster_whisper"
        self.model_name = resolved_size
        self.device = resolved_device
        self.model = WhisperModel(
            resolved_size,
            device=resolved_device,
            compute_type=compute_type,
            download_root=os.getenv("HF_HOME"),
        )

    def _is_likely_silence_hallucination(self, text: str, audio_len_samples: int, rms: float) -> bool:
        normalized = self._normalize_text(text)
        if not normalized:
            return True
        if normalized in BAD_SILENCE_PHRASES:
            if audio_len_samples <= HALLUCINATION_MAX_SAMPLES or rms < (MIN_RMS * 1.5):
                return True
        return False

    def _transcribe_blocking(self, audio_data: np.ndarray) -> list[str]:
        try:
            use_whisper_vad = _env_bool("WHISPER_INTERNAL_VAD", False)
            forced_language = self.WHISPER_LANGUAGE_CODES[self.lang]

            kwargs = {
                "language": forced_language,
                "beam_size": _env_int("WHISPER_BEAM_SIZE", 1, 1, 10),
                "best_of": _env_int("WHISPER_BEST_OF", 1, 1, 10),
                "condition_on_previous_text": False,
                "vad_filter": use_whisper_vad,
                "temperature": _env_float("WHISPER_TEMPERATURE", 0.0, 0.0, 1.0),
            }

            if use_whisper_vad:
                kwargs["vad_parameters"] = {
                    "min_silence_duration_ms": _env_int("WHISPER_VAD_MIN_SILENCE_MS", 900, 100, 3000),
                    "threshold": _env_float("WHISPER_VAD_THRESHOLD", 0.35, 0.0, 1.0),
                    "min_speech_duration_ms": _env_int("WHISPER_VAD_MIN_SPEECH_MS", 250, 50, 2000),
                }

            segments, _ = self.model.transcribe(audio_data, **kwargs)
            return [seg.text.strip() for seg in segments if seg.text and seg.text.strip()]
        except Exception as e:
            print(f"[STT] ⚠️  Transcription error (chunk discarded): {e}")
            return []

    async def _transcribe_whisper_stream(self, audio_source: AudioSource):
        print(
            f"--- STT Active: Whisper [{self.lang}] "
            f"VAD={VAD_AGGRESSIVENESS}, silence={END_SILENCE_MS}ms, "
            f"final_grace={FINAL_GRACE_MS}ms, "
            f"padding={PADDING_MS}ms, min_samples={MIN_SAMPLES}, min_rms={MIN_RMS}, "
            f"frame_rms_gate={FRAME_SPEECH_MIN_RMS}, start_confirm={SPEECH_START_CONFIRM_MS}ms, "
            f"start_min_rms={SPEECH_START_MIN_RMS}, barge_min_rms={BARGE_IN_MIN_RMS} ---"
        )

        loop = asyncio.get_running_loop()
        vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)

        byte_buffer = bytearray()
        ring_pad = collections.deque(maxlen=PADDING_FRAMES)
        utterance_frames: list[bytes] = []
        in_speech = False
        trailing_silence = 0
        maybe_done = False
        grace_frames = 0
        grace_resets = 0
        start_candidate_frames = 0
        start_candidate_rms_total = 0.0

        async def _flush(frames: list[bytes]):
            t_flush_start = time.time()
            if not frames:
                return

            audio_bytes = b"".join(frames)
            audio_data = self._bytes_to_float_audio(audio_bytes)
            utterance_ms = len(audio_data) / SAMPLE_RATE * 1000

            if len(audio_data) < MIN_SAMPLES:
                print(
                    f"[STT][{t_flush_start:.3f}] Flush skipped: below MIN_SAMPLES "
                    f"({len(audio_data)} < {MIN_SAMPLES}, {utterance_ms:.0f}ms audio)"
                )
                return

            rms = self._rms(audio_data)
            if rms < MIN_RMS:
                print(
                    f"[STT][{t_flush_start:.3f}] Flush skipped: below MIN_RMS "
                    f"(rms={rms:.5f} < {MIN_RMS}, {utterance_ms:.0f}ms audio)"
                )
                return

            print(
                f"[STT][{t_flush_start:.3f}] Flush start: {utterance_ms:.0f}ms audio, "
                f"rms={rms:.5f} — invoking Whisper..."
            )

            t0 = time.time()
            texts = await loop.run_in_executor(None, self._transcribe_blocking, audio_data)
            stt_elapsed = time.time() - t0

            print(
                f"[STT][{time.time():.3f}] Whisper transcribe done in {stt_elapsed:.3f}s "
                f"(total flush-to-text: {time.time() - t_flush_start:.3f}s) -> {texts!r}"
            )

            cleaned_texts: list[str] = []
            for text in texts:
                if self._is_likely_silence_hallucination(text, len(audio_data), rms):
                    print(f"[STT] 🧹 Dropped likely silence hallucination: {text!r} | rms={rms:.5f}")
                    continue
                cleaned_texts.append(text)

            if cleaned_texts:
                with suppress(Exception):
                    import stats
                    stats.record_stt_latency(stt_elapsed)

            for text in cleaned_texts:
                yield text

        try:
            async for chunk in audio_source.get_stream():
                if not chunk or not isinstance(chunk, (bytes, bytearray)):
                    continue

                byte_buffer.extend(chunk)
                frames = self._frame_generator(byte_buffer)

                for frame in frames:
                    try:
                        vad_speech = vad.is_speech(frame, SAMPLE_RATE)
                    except Exception:
                        vad_speech = False

                    frame_rms = self._frame_rms(frame)
                    is_speech = vad_speech and frame_rms >= FRAME_SPEECH_MIN_RMS

                    if not in_speech:
                        ring_pad.append(frame)

                        if is_speech:
                            start_candidate_frames += 1
                            start_candidate_rms_total += frame_rms
                            avg_start_rms = start_candidate_rms_total / max(1, start_candidate_frames)

                            if (
                                start_candidate_frames >= SPEECH_START_CONFIRM_FRAMES
                                and avg_start_rms >= SPEECH_START_MIN_RMS
                            ):
                                in_speech = True
                                trailing_silence = 0
                                maybe_done = False
                                grace_frames = 0
                                grace_resets = 0
                                t_speech_start = time.time()
                                print(
                                    f"[STT][{t_speech_start:.3f}] Speech start confirmed "
                                    f"({SPEECH_START_CONFIRM_MS}ms, avg_rms={avg_start_rms:.5f})."
                                )

                                if avg_start_rms >= BARGE_IN_MIN_RMS:
                                    await self._notify_speech_start()
                                else:
                                    print(
                                        f"[STT][{time.time():.3f}] Speech start kept for STT but not used "
                                        f"for barge-in (avg_rms={avg_start_rms:.5f} < {BARGE_IN_MIN_RMS})."
                                    )

                                utterance_frames = list(ring_pad)
                        else:
                            start_candidate_frames = 0
                            start_candidate_rms_total = 0.0

                        continue

                    utterance_frames.append(frame)

                    if is_speech:
                        if maybe_done:
                            grace_resets += 1
                            print(
                                f"[STT][{time.time():.3f}] Speech resumed during grace window "
                                f"(reset #{grace_resets}) — restarting {END_SILENCE_MS}ms silence count."
                            )
                        trailing_silence = 0
                        maybe_done = False
                        grace_frames = 0
                    else:
                        trailing_silence += 1

                    if trailing_silence >= END_SILENCE_FRAMES and not maybe_done:
                        maybe_done = True
                        grace_frames = 0
                        print(
                            f"[STT][{time.time():.3f}] Possible end of speech detected "
                            f"(after {grace_resets} prior reset(s)); entering final grace window."
                        )

                    if maybe_done:
                        grace_frames += 1

                    reached_max_length = len(utterance_frames) >= MAX_UTTERANCE_FRAMES
                    grace_expired = maybe_done and grace_frames >= FINAL_GRACE_FRAMES

                    if reached_max_length or grace_expired:
                        reason = "max_length" if reached_max_length else "grace_expired"
                        print(
                            f"[STT][{time.time():.3f}] Flush triggered ({reason}), "
                            f"{len(utterance_frames)} frames, {grace_resets} grace reset(s) total."
                        )

                        frames_to_flush = utterance_frames
                        utterance_frames = []
                        ring_pad.clear()
                        in_speech = False
                        trailing_silence = 0
                        maybe_done = False
                        grace_frames = 0
                        grace_resets = 0
                        start_candidate_frames = 0
                        start_candidate_rms_total = 0.0

                        async for text in _flush(frames_to_flush):
                            yield text

        except asyncio.CancelledError:
            raise
        except Exception as e:
            print(f"[STT] Stream error: {e}")

    async def transcribe_stream(self, audio_source: AudioSource):
        if self.use_whisper:
            async for text in self._transcribe_whisper_stream(audio_source):
                yield text
        else:
            async for text in self._transcribe_soniox_stream(audio_source):
                yield text
