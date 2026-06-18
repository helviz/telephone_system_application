import base64
import audioop
import json
from math import gcd
from typing import Any

import numpy as np
import torch

try:
    from scipy.signal import resample_poly
except Exception:  # pragma: no cover - fallback for minimal deployments
    resample_poly = None

from Audio.AudioOutput import AudioOutput


class PhoneAudioOutput(AudioOutput):
    """
    Sends synthesized audio back into a Twilio or Telnyx phone call.

    Output format expected by phone media streams:
      - 8 kHz
      - mono
      - G.711 mu-law
      - base64 inside a JSON media event

    Audio-quality improvements over audioop.ratecv-only output:
      - keeps audio in float until the final quantization step
      - removes DC offset
      - peak-normalizes safely
      - uses anti-aliased polyphase resampling when scipy is available
      - avoids int16 wraparound crackle by clipping before conversion
      - chunks outbound media payloads to avoid very large websocket messages
    """

    PROVIDERS = {"twilio", "telnyx"}
    TARGET_SAMPLE_RATE = 8000
    MULAW_BYTES_PER_SAMPLE = 1
    DEFAULT_CHUNK_MS = 100
    _mark_counter = 0

    def __init__(self, websocket: Any, provider: str = "twilio", chunk_ms: int = DEFAULT_CHUNK_MS):
        if provider not in self.PROVIDERS:
            raise ValueError(f"Unsupported provider: '{provider}'. Choose from {self.PROVIDERS}")
        if chunk_ms <= 0:
            raise ValueError("chunk_ms must be greater than 0")

        self.ws = websocket
        self.provider = provider
        self.stream_sid: str | None = None
        self.chunk_ms = chunk_ms

    def set_stream_sid(self, sid: str | None):
        """Called once the provider's websocket 'start' event is received."""
        self.stream_sid = sid

    @staticmethod
    def _to_float_mono(audio: Any) -> np.ndarray:
        """Convert torch/bytes/numpy/list audio into mono float32 in [-1, 1]."""
        if isinstance(audio, torch.Tensor):
            audio_np = audio.detach().cpu().numpy()
        elif isinstance(audio, bytes):
            # bytes are assumed to be PCM16 little-endian
            audio_np = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
        else:
            audio_np = np.asarray(audio)

        audio_np = np.squeeze(audio_np).astype(np.float32, copy=False)

        # Convert stereo/multi-channel to mono.
        if audio_np.ndim > 1:
            audio_np = np.mean(audio_np, axis=-1)

        # Remove NaN/Inf values that can produce pops during conversion.
        audio_np = np.nan_to_num(audio_np, nan=0.0, posinf=0.0, neginf=0.0)

        # If input looks like PCM16 numeric values, scale it down.
        if audio_np.size and np.max(np.abs(audio_np)) > 2.0:
            audio_np = audio_np / 32768.0

        return audio_np

    @staticmethod
    def _prepare_for_phone(audio_np: np.ndarray) -> np.ndarray:
        """Basic mastering for narrowband telephone playback."""
        if audio_np.size == 0:
            return audio_np.astype(np.float32)

        # Remove DC offset. This reduces clicks and wasted headroom.
        audio_np = audio_np - float(np.mean(audio_np))

        # Soft safety normalization. Keep headroom before mu-law conversion.
        peak = float(np.max(np.abs(audio_np)))
        if peak > 1e-6:
            audio_np = (audio_np / peak) * 0.90

        # Tiny fade to avoid clicks at start/end of one-shot TTS clips.
        fade_len = min(int(0.005 * 8000), audio_np.size // 2)  # about 5 ms after resampling
        if fade_len > 1:
            fade = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
            audio_np[:fade_len] *= fade
            audio_np[-fade_len:] *= fade[::-1]

        return np.clip(audio_np, -0.98, 0.98).astype(np.float32)

    @classmethod
    def _resample_to_phone_rate(cls, audio_np: np.ndarray, sample_rate: int) -> np.ndarray:
        """Resample to 8 kHz with anti-aliasing where possible."""
        if sample_rate == cls.TARGET_SAMPLE_RATE:
            return audio_np.astype(np.float32, copy=False)

        if sample_rate <= 0:
            raise ValueError("sample_rate must be a positive integer")

        if resample_poly is not None:
            factor = gcd(sample_rate, cls.TARGET_SAMPLE_RATE)
            up = cls.TARGET_SAMPLE_RATE // factor
            down = sample_rate // factor
            return resample_poly(audio_np, up, down).astype(np.float32)

        # Fallback only: audioop.ratecv is lower quality but avoids crashing if scipy is absent.
        pcm = np.clip(audio_np, -0.98, 0.98)
        pcm_bytes = (pcm * 32767.0).astype(np.int16).tobytes()
        pcm_8k, _ = audioop.ratecv(pcm_bytes, 2, 1, sample_rate, cls.TARGET_SAMPLE_RATE, None)
        return np.frombuffer(pcm_8k, dtype=np.int16).astype(np.float32) / 32768.0

    def _media_message(self, payload: str) -> dict:
        message = {
            "event": "media",
            "media": {"payload": payload},
        }
        if self.stream_sid:
            message["streamSid"] = self.stream_sid
        return message

    def _mark_message(self) -> dict:
        PhoneAudioOutput._mark_counter += 1
        message = {
            "event": "mark",
            "mark": {"name": f"tts_{PhoneAudioOutput._mark_counter}"},
        }
        if self.stream_sid:
            message["streamSid"] = self.stream_sid
        return message

    async def send_audio(self, audio: Any, sample_rate: int = 16000):
        audio_np = self._to_float_mono(audio)
        audio_8k = self._resample_to_phone_rate(audio_np, sample_rate)
        audio_8k = self._prepare_for_phone(audio_8k)

        pcm_8k = (audio_8k * 32767.0).astype(np.int16).tobytes()
        mulaw = audioop.lin2ulaw(pcm_8k, 2)

        chunk_size = int(self.TARGET_SAMPLE_RATE * self.MULAW_BYTES_PER_SAMPLE * self.chunk_ms / 1000)
        chunk_size = max(chunk_size, 160)  # never below 20 ms at 8 kHz mu-law

        for start in range(0, len(mulaw), chunk_size):
            chunk = mulaw[start:start + chunk_size]
            payload = base64.b64encode(chunk).decode("utf-8")
            await self.ws.send_text(json.dumps(self._media_message(payload)))

        # One mark after the full utterance, not after every small media packet.
        await self.ws.send_text(json.dumps(self._mark_message()))

    @classmethod
    def twilio(cls, websocket, chunk_ms: int = DEFAULT_CHUNK_MS) -> "PhoneAudioOutput":
        return cls(websocket, provider="twilio", chunk_ms=chunk_ms)

    @classmethod
    def telnyx(cls, websocket, chunk_ms: int = DEFAULT_CHUNK_MS) -> "PhoneAudioOutput":
        return cls(websocket, provider="telnyx", chunk_ms=chunk_ms)
