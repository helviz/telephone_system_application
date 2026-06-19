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


# Defaults tuned for narrowband 8 kHz phone speech after upsampling to 16 kHz.
# These can still be overridden from Hugging Face / Docker env variables.
VAD_AGGRESSIVENESS = _env_int("WEBRTC_VAD_AGGRESSIVENESS", 2, 0, 3)

END_SILENCE_MS = _env_int("STT_END_SILENCE_MS", 900, 200, 3000)
END_SILENCE_FRAMES = max(1, END_SILENCE_MS // FRAME_MS)

PADDING_MS = _env_int("STT_PADDING_MS", 400, 0, 2000)
PADDING_FRAMES = max(1, PADDING_MS // FRAME_MS)

MAX_UTTERANCE_MS = _env_int("STT_MAX_UTTERANCE_MS", 20_000, 1000, 120_000)
MAX_UTTERANCE_FRAMES = max(1, MAX_UTTERANCE_MS // FRAME_MS)

MIN_SAMPLES = _env_int("STT_MIN_SAMPLES", 12000, 1600, 160000)
MIN_RMS = _env_float("STT_MIN_RMS", 0.008, 0.0, 1.0)

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
    def __init__(self, model_size=None, device=None, lang="en", preloaded_model=None):
        allowed_languages = ["en", "fr", "sw"]
        if lang not in allowed_languages:
            raise ValueError(f"Language '{lang}' not supported. Choose from {allowed_languages}")

        self.lang = lang

        if preloaded_model is not None:
            self.model = preloaded_model
            print(f"[STT] ♻️  Reusing preloaded Whisper model for [{lang}]")
            return

        resolved_size = model_size or os.getenv("WHISPER_MODEL_SIZE", "medium").strip()
        resolved_device = device or os.getenv("WHISPER_DEVICE", "cpu").strip()
        compute_type = "float16" if resolved_device == "cuda" else "int8"

        print(f"[STT] 📦 Loading Whisper [{resolved_size}] on [{resolved_device}]...")
        self.model = WhisperModel(
            resolved_size,
            device=resolved_device,
            compute_type=compute_type,
            download_root=os.getenv("HF_HOME"),
        )

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
            use_whisper_vad = _env_bool("WHISPER_INTERNAL_VAD", False)

            kwargs = {
                "language": self.lang,
                "beam_size": _env_int("WHISPER_BEAM_SIZE", 5, 1, 10),
                "best_of": _env_int("WHISPER_BEST_OF", 5, 1, 10),
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
            f"padding={PADDING_MS}ms, min_samples={MIN_SAMPLES}, min_rms={MIN_RMS} ---"
        )

        loop = asyncio.get_running_loop()
        vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)

        byte_buffer = bytearray()
        ring_pad = collections.deque(maxlen=PADDING_FRAMES)
        utterance_frames: list[bytes] = []
        in_speech = False
        trailing_silence = 0

        async def _flush(frames: list[bytes]):
            if not frames:
                return

            audio_bytes = b"".join(frames)
            audio_data = self._bytes_to_float_audio(audio_bytes)

            if len(audio_data) < MIN_SAMPLES:
                return

            rms = self._rms(audio_data)
            if rms < MIN_RMS:
                return

            t0 = time.time()
            texts = await loop.run_in_executor(None, self._transcribe_blocking, audio_data)
            stt_elapsed = time.time() - t0

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
                            utterance_frames = list(ring_pad)
                            utterance_frames.append(frame)
                        continue

                    utterance_frames.append(frame)

                    if is_speech:
                        trailing_silence = 0
                    else:
                        trailing_silence += 1

                    should_flush = (
                        trailing_silence >= END_SILENCE_FRAMES
                        or len(utterance_frames) >= MAX_UTTERANCE_FRAMES
                    )

                    if should_flush:
                        async for text in _flush(utterance_frames):
                            yield text
                        utterance_frames = []
                        in_speech = False
                        trailing_silence = 0
                        ring_pad.clear()

        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"[STT] ❌ Stream error: {e}")
        finally:
            if utterance_frames:
                async for text in _flush(utterance_frames):
                    yield text
