import asyncio
import audioop
import base64
import json
from typing import Any

try:
    from scipy.signal import resample_poly
except Exception:  # pragma: no cover - fallback for minimal deployments
    resample_poly = None

import numpy as np

from Audio.AudioSource import AudioSource


class PhoneStreamSource(AudioSource):
    """
    Incoming phone media stream normalizer for Twilio and Telnyx.

    Both providers are normalized into:
      - PCM16
      - mono
      - 16 kHz

    This is the format expected by transcribe/SSTModule.py, where WebRTC VAD
    works on exact 16 kHz PCM16 frames.
    """

    PROVIDERS = {"twilio", "telnyx"}
    TARGET_STT_SAMPLE_RATE = 16000
    DEFAULT_PHONE_SAMPLE_RATE = 8000

    def __init__(self, provider: str = "twilio"):
        provider = (provider or "twilio").strip().lower()
        if provider not in self.PROVIDERS:
            raise ValueError(f"Unsupported provider: '{provider}'. Choose from {self.PROVIDERS}")

        self.provider = provider
        self.queue: asyncio.Queue[bytes] = asyncio.Queue()

        # Twilio uses streamSid; Telnyx uses stream_id. Keep both names so the
        # rest of the app can remain provider-neutral.
        self.stream_sid: str | None = None
        self.stream_id: str | None = None
        self.call_control_id: str | None = None

        self.input_sample_rate = self.DEFAULT_PHONE_SAMPLE_RATE
        self.input_channels = 1

    async def add_data(self, websocket_message: str | bytes):
        """Parse one provider websocket frame and enqueue PCM16/16k audio."""
        if isinstance(websocket_message, bytes):
            websocket_message = websocket_message.decode("utf-8", errors="ignore")

        try:
            packet: dict[str, Any] = json.loads(websocket_message)
        except json.JSONDecodeError as exc:
            print(f"[{self.provider.upper()}] Ignoring malformed websocket JSON: {exc}")
            return

        event = packet.get("event")

        if event == "start":
            self._handle_start(packet)
            return

        if event == "media":
            await self._handle_media(packet)
            return

        if event == "stop":
            print(f"[{self.provider.upper()}] Stream stop event received.")
            return

        if event == "mark":
            name = packet.get("mark", {}).get("name")
            print(f"[{self.provider.upper()}] Mark received: {name}")
            return

        if event == "dtmf":
            digit = packet.get("dtmf", {}).get("digit")
            print(f"[{self.provider.upper()}] DTMF received over stream: {digit}")
            return

        if event == "error":
            print(f"[{self.provider.upper()}] Stream error: {packet.get('payload')}")
            return

        print(f"[{self.provider.upper()}] Ignored websocket event: {event}")

    def _handle_start(self, packet: dict[str, Any]):
        start = packet.get("start", {}) or {}

        if self.provider == "twilio":
            self.stream_sid = (
                start.get("streamSid")
                or packet.get("streamSid")
                or packet.get("stream_sid")
            )
            self.stream_id = self.stream_sid
            self.call_control_id = start.get("callSid") or packet.get("callSid")
            self.input_sample_rate = self.DEFAULT_PHONE_SAMPLE_RATE
            self.input_channels = 1
        else:
            self.stream_id = packet.get("stream_id") or start.get("stream_id")
            self.stream_sid = self.stream_id
            self.call_control_id = start.get("call_control_id")

            try:
                self.input_sample_rate = int(start.get("sample_rate") or self.DEFAULT_PHONE_SAMPLE_RATE)
            except (TypeError, ValueError):
                self.input_sample_rate = self.DEFAULT_PHONE_SAMPLE_RATE

            try:
                self.input_channels = int(start.get("channels") or 1)
            except (TypeError, ValueError):
                self.input_channels = 1

        print(
            f"[{self.provider.upper()}] Stream started — "
            f"stream_id={self.stream_id}, sample_rate={self.input_sample_rate}, "
            f"channels={self.input_channels}"
        )

    async def _handle_media(self, packet: dict[str, Any]):
        media = packet.get("media", {}) or {}
        payload = media.get("payload")

        if not payload:
            print(f"[{self.provider.upper()}] Media event without payload.")
            return

        try:
            encoded_audio = base64.b64decode(payload)
        except Exception as exc:
            print(f"[{self.provider.upper()}] Failed to base64-decode media payload: {exc}")
            return

        # With your TwiML/TeXML config, both providers should send PCMU/G.711
        # mu-law payloads. Telnyx bidirectional RTP mode sends the RTP payload
        # without headers, so direct mu-law decoding is correct.
        try:
            pcm_phone_rate = audioop.ulaw2lin(encoded_audio, 2)
        except Exception as exc:
            print(f"[{self.provider.upper()}] Failed to decode PCMU/mu-law payload: {exc}")
            return

        if self.input_channels > 1:
            try:
                pcm_phone_rate = audioop.tomono(pcm_phone_rate, 2, 0.5, 0.5)
            except Exception as exc:
                print(f"[{self.provider.upper()}] Failed to downmix audio to mono: {exc}")

        try:
            pcm_16k = self._resample_pcm16(
                pcm_phone_rate,
                source_rate=int(self.input_sample_rate or self.DEFAULT_PHONE_SAMPLE_RATE),
                target_rate=self.TARGET_STT_SAMPLE_RATE,
            )
        except Exception as exc:
            print(f"[{self.provider.upper()}] Failed to resample audio to 16 kHz: {exc}")
            return

        await self.queue.put(pcm_16k)

    @staticmethod
    def _resample_pcm16(pcm_bytes: bytes, source_rate: int, target_rate: int) -> bytes:
        """Resample PCM16 mono bytes. Prefer scipy anti-aliasing, fallback to audioop."""
        if source_rate == target_rate:
            return pcm_bytes

        if source_rate <= 0 or target_rate <= 0:
            raise ValueError("source_rate and target_rate must be positive")

        if resample_poly is not None:
            from math import gcd

            audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            factor = gcd(source_rate, target_rate)
            up = target_rate // factor
            down = source_rate // factor
            resampled = resample_poly(audio, up, down)
            resampled = np.clip(resampled, -1.0, 1.0)
            return (resampled * 32767.0).astype(np.int16).tobytes()

        pcm_resampled, _ = audioop.ratecv(pcm_bytes, 2, 1, source_rate, target_rate, None)
        return pcm_resampled

    async def get_stream(self):
        """Async generator consumed by the STT pipeline."""
        while True:
            chunk = await self.queue.get()
            yield chunk

    @classmethod
    def twilio(cls) -> "PhoneStreamSource":
        return cls(provider="twilio")

    @classmethod
    def telnyx(cls) -> "PhoneStreamSource":
        return cls(provider="telnyx")
