import asyncio
import collections
import os
import time

import numpy as np
import webrtcvad
from dotenv import load_dotenv
from faster_whisper import WhisperModel

from Audio.AudioSource import AudioSource

load_dotenv()

# WebRTC VAD requires mono PCM16 frames at 8, 16, 32, or 48 kHz.
# PhoneStreamSource normalizes Twilio/Telnyx phone audio to 16 kHz PCM16.
SAMPLE_RATE = 16000
FRAME_MS = 20
FRAME_BYTES = int(SAMPLE_RATE * FRAME_MS / 1000) * 2  # int16 = 2 bytes


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


# Defaults tuned for low-latency narrowband phone speech after upsampling to 16 kHz.
# These can still be overridden from Hugging Face / Docker env variables.
VAD_AGGRESSIVENESS = _env_int("WEBRTC_VAD_AGGRESSIVENESS", 1, 0, 3)

# Two-stage turn finalization for phone calls:
#   1. END_SILENCE_MS marks a possible end-of-speech.
#   2. FINAL_GRACE_MS waits a little longer before yielding text to the LLM.
# If the caller resumes during the grace window, the pending flush is cancelled
# and the new audio is merged into the same utterance.
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

# Extra protection against Whisper hallucinating common phrases on silence/noise.
# The length gate means real longer utterances containing these words are not blocked.
HALLUCINATION_MAX_SAMPLES = _env_int("STT_HALLUCINATION_MAX_SAMPLES", 24000, 1600, 160000)
BAD_SILENCE_PHRASES = {
    "thank you",
    "thank you.",
    "thanks",
    "thanks.",
    "you",
    "you.",
    "asante",
    "asante.",
    "merci",
    "merci.",
}


class STTModule:
    """
    Streaming STT with language-routed backends.

    Routing:
      - en/sw: Sunbird/asr-whisper-large-v3-salt using the official
        WhisperProcessor + WhisperForConditionalGeneration path. The SALT
        language token is forced from the IVR-selected language.
      - fr: env-configured faster-whisper model, usually WHISPER_MODEL_SIZE.

    Language is always forced from the IVR-selected `lang`; no auto-detection.
    """

    # Official Sunbird/SALT Whisper language token IDs. Do not use these with
    # base/faster-whisper models; they are for the Sunbird SALT tokenizer/model.
    SALT_LANGUAGE_TOKENS_WHISPER = {
        "eng": 50259,  # English (Ugandan)
        "swa": 50318,  # Swahili
        "ach": 50357,  # Acholi
        "lgg": 50356,  # Lugbara
        "lug": 50355,  # Luganda
        "nyn": 50354,  # Runyankole
        "teo": 50353,  # Ateso
        "xog": 50352,  # Lusoga
        "ttj": 50351,  # Rutooro
        "kin": 50350,  # Kinyarwanda
        "myx": 50349,  # Lumasaba
    }

    APP_TO_SALT_LANG = {
        "en": "eng",
        "sw": "swa",
    }

    FASTER_WHISPER_LANGUAGE_CODES = {
        "en": "en",
        "sw": "sw",
        "fr": "fr",
    }

    def __init__(self, model_size=None, device=None, lang="en", preloaded_model=None, on_speech_start=None):
        allowed_languages = ["en", "fr", "sw"]
        if lang not in allowed_languages:
            raise ValueError(f"Language '{lang}' not supported. Choose from {allowed_languages}")

        self.lang = lang
        self.on_speech_start = on_speech_start
        self._last_speech_start_notify = 0.0
        self.engine = "faster_whisper"
        self.model = None
        self.processor = None
        self.device = None
        self.salt_lang = None

        # New preferred path: sockets.py/preload.py passes a language-keyed STT store:
        # {
        #   "en": {"engine": "sunbird_salt", "processor": processor, "model": model, "salt_lang": "eng", ...},
        #   "sw": {"engine": "sunbird_salt", "processor": processor, "model": model, "salt_lang": "swa", ...},
        #   "fr": {"engine": "faster_whisper", "model": french_whisper, ...},
        # }
        if isinstance(preloaded_model, dict) and lang in preloaded_model:
            entry = preloaded_model[lang]
            if isinstance(entry, dict):
                self.engine = entry.get("engine", "faster_whisper")
                self.model = entry.get("model")
                self.processor = entry.get("processor")
                self.model_name = entry.get("model_name", "unknown")
                self.device = entry.get("device")
                self.salt_lang = entry.get("salt_lang")
            else:
                # Backward compatibility for a raw language-keyed model dict.
                self.engine = "faster_whisper"
                self.model = entry
                self.model_name = "preloaded"

            if self.model is None:
                raise RuntimeError(f"No STT model found for language [{lang}] in preloaded_model store.")
            if self.engine == "sunbird_salt" and self.processor is None:
                raise RuntimeError(f"Sunbird SALT STT for [{lang}] requires a preloaded processor.")

            forced = self._forced_language_for_engine()
            print(f"[STT] ♻️  Reusing preloaded {self.engine} STT for [{lang}] forced_language=[{forced}]")
            return

        # Backward compatibility for your old single-model preload.
        if preloaded_model is not None:
            self.engine = "faster_whisper"
            self.model = preloaded_model
            self.model_name = "preloaded-faster-whisper"
            forced = self._forced_language_for_engine()
            print(f"[STT] ♻️  Reusing legacy Faster-Whisper model for [{lang}] forced_language=[{forced}]")
            return

        # Fallback only if no preload is provided. This keeps local dev usable.
        if lang in {"en", "sw"} and _env_bool("USE_SUNBIRD_FOR_EN_SW", True):
            self._load_transformers_sunbird_fallback(device=device)
        else:
            self._load_faster_whisper_fallback(model_size=model_size, device=device)

    def _forced_language_for_engine(self) -> str:
        if self.engine == "sunbird_salt":
            salt_lang = self.salt_lang or self.APP_TO_SALT_LANG.get(self.lang)
            return f"{salt_lang}:{self.SALT_LANGUAGE_TOKENS_WHISPER.get(salt_lang)}"
        return self.FASTER_WHISPER_LANGUAGE_CODES[self.lang]

    def _load_faster_whisper_fallback(self, model_size=None, device=None):
        resolved_size = model_size or os.getenv("WHISPER_MODEL_SIZE", "medium").strip()
        resolved_device = device or os.getenv("WHISPER_DEVICE", "cpu").strip()
        compute_type = "float16" if resolved_device == "cuda" else "int8"

        print(
            f"[STT] 📦 Loading Faster-Whisper [{resolved_size}] on [{resolved_device}] "
            f"(compute_type={compute_type}) forced_language=[{self.FASTER_WHISPER_LANGUAGE_CODES[self.lang]}]..."
        )
        self.engine = "faster_whisper"
        self.model_name = resolved_size
        self.model = WhisperModel(
            resolved_size,
            device=resolved_device,
            compute_type=compute_type,
            download_root=os.getenv("HF_HOME"),
        )

    def _load_transformers_sunbird_fallback(self, device=None):
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        model_id = os.getenv("SUNBIRD_ASR_MODEL", "Sunbird/asr-whisper-large-v3-salt").strip()
        resolved_device = device or os.getenv("SUNBIRD_ASR_DEVICE", "cuda" if torch.cuda.is_available() else "cpu").strip()
        if resolved_device == "cuda" and not torch.cuda.is_available():
            print("[STT] ⚠️ SUNBIRD_ASR_DEVICE=cuda requested, but CUDA is unavailable. Falling back to CPU.")
            resolved_device = "cpu"

        torch_dtype = torch.float16 if resolved_device == "cuda" else torch.float32
        salt_lang = self.APP_TO_SALT_LANG[self.lang]

        print(
            f"[STT] 📦 Loading Sunbird SALT ASR [{model_id}] on [{resolved_device}] "
            f"forced_salt_language=[{salt_lang}:{self.SALT_LANGUAGE_TOKENS_WHISPER[salt_lang]}]..."
        )

        self.processor = WhisperProcessor.from_pretrained(model_id, cache_dir=os.getenv("HF_HOME"))
        self.model = WhisperForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            cache_dir=os.getenv("HF_HOME"),
        ).to(resolved_device)
        self.model.eval()

        self.engine = "sunbird_salt"
        self.model_name = model_id
        self.device = resolved_device
        self.salt_lang = salt_lang

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

    def _is_likely_silence_hallucination(self, text: str, audio_len_samples: int, rms: float) -> bool:
        normalized = self._normalize_text(text)
        if not normalized:
            return True

        # Whisper often emits these on silence/noise. Only suppress them for short
        # and low-energy clips so genuine user speech is not removed.
        if normalized in BAD_SILENCE_PHRASES:
            if audio_len_samples <= HALLUCINATION_MAX_SAMPLES or rms < (MIN_RMS * 1.5):
                return True

        return False

    def _transcribe_blocking(self, audio_data: np.ndarray) -> list[str]:
        try:
            if self.engine == "sunbird_salt":
                import torch

                salt_lang = self.salt_lang or self.APP_TO_SALT_LANG[self.lang]
                salt_token_id = self.SALT_LANGUAGE_TOKENS_WHISPER[salt_lang]

                # Official Sunbird/SALT forcing style:
                #   language=processor.tokenizer.decode(SALT_LANGUAGE_TOKENS_WHISPER[lang])
                #   forced_decoder_ids=None
                language_token = self.processor.tokenizer.decode(salt_token_id)

                input_features = self.processor(
                    audio_data.astype(np.float32),
                    sampling_rate=SAMPLE_RATE,
                    return_tensors="pt",
                ).input_features

                target_device = self.device or next(self.model.parameters()).device
                input_features = input_features.to(target_device)

                with torch.no_grad():
                    predicted_ids = self.model.generate(
                        input_features,
                        language=language_token,
                        forced_decoder_ids=None,
                    )

                texts = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)
                return [text.strip() for text in texts if text and text.strip()]

            use_whisper_vad = _env_bool("WHISPER_INTERNAL_VAD", False)
            forced_language = self.FASTER_WHISPER_LANGUAGE_CODES[self.lang]

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

    @staticmethod
    def _bytes_to_float_audio(audio_bytes: bytes) -> np.ndarray:
        if len(audio_bytes) % 2 != 0:
            audio_bytes = audio_bytes[:-1]
        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        return np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

    async def transcribe_stream(self, audio_source: AudioSource):
        print(
            f"--- STT Active: Listening [{self.lang}] "
            f"VAD={VAD_AGGRESSIVENESS}, silence={END_SILENCE_MS}ms, "
            f"final_grace={FINAL_GRACE_MS}ms, "
            f"padding={PADDING_MS}ms, min_samples={MIN_SAMPLES}, min_rms={MIN_RMS} ---"
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
                try:
                    import stats
                    stats.record_stt_latency(stt_elapsed)
                except Exception:
                    pass

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
                        is_speech = vad.is_speech(frame, SAMPLE_RATE)
                    except Exception:
                        is_speech = False

                    if not in_speech:
                        ring_pad.append(frame)

                        if is_speech:
                            in_speech = True
                            trailing_silence = 0
                            maybe_done = False
                            grace_frames = 0
                            grace_resets = 0
                            t_speech_start = time.time()
                            print(f"[STT][{t_speech_start:.3f}] Speech start detected.")
                            await self._notify_speech_start()

                            # Include a small amount of pre-speech padding so Whisper
                            # does not lose the beginning of the caller's first word.
                            utterance_frames = list(ring_pad)
                            utterance_frames.append(frame)

                        continue

                    # We are inside one caller utterance. Keep collecting frames until
                    # both the normal silence threshold and the final grace window pass.
                    utterance_frames.append(frame)

                    if is_speech:
                        # Caller resumed speech during silence/grace. This is the key
                        # merge step: cancel the pending flush and keep the same buffer.
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
                        async for text in _flush(utterance_frames):
                            yield text

                        utterance_frames = []
                        in_speech = False
                        trailing_silence = 0
                        maybe_done = False
                        grace_frames = 0
                        grace_resets = 0
                        ring_pad.clear()

        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"[STT] ❌ Stream error: {e}")
        finally:
            if utterance_frames:
                async for text in _flush(utterance_frames):
                    yield text