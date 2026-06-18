import asyncio
import audioop
import base64
import json
from typing import Any

from Audio.AudioSource import AudioSource


class PhoneStreamSource(AudioSource):
    """
    Handles incoming phone audio from Twilio and Telnyx media streams.

    Twilio and Telnyx are similar but not identical:
      - Twilio uses streamSid and usually sends 8 kHz G.711 mu-law media.
      - Telnyx uses stream_id and sends base64 RTP payloads without RTP headers.
        With TeXML <Stream codec="PCMU" ...>, the payload is 8 kHz PCMU/mu-law.

    This source normalizes both providers into PCM16 mono at 16 kHz for STT.
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

        # Twilio uses streamSid; Telnyx uses stream_id. Keep both names.
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

        # Keep this visible while testing Telnyx because unexpected event names
        # are the fastest way to discover provider-format mismatches.
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
        else:
            self.stream_id = packet.get("stream_id") or start.get("stream_id")
            self.stream_sid = self.stream_id
            self.call_control_id = start.get("call_control_id")

            # Telnyx includes sample_rate/channels in the start message.
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

        # With your TeXML/TwiML config, both providers should be PCMU/G.711 mu-law.
        # Telnyx calls this RTP payload without headers; for PCMU that means raw
        # mu-law bytes, which can be decoded directly.
        try:
            pcm_phone_rate = audioop.ulaw2lin(encoded_audio, 2)
        except Exception as exc:
            print(f"[{self.provider.upper()}] Failed to decode PCMU/mu-law payload: {exc}")
            return

        # If a provider ever sends stereo, downmix to mono before STT.
        if self.input_channels > 1:
            try:
                pcm_phone_rate = audioop.tomono(pcm_phone_rate, 2, 0.5, 0.5)
            except Exception:
                # Continue with original audio rather than dropping the call.
                pass

        # Upsample phone-rate PCM16 to 16 kHz for Whisper/faster-whisper.
        try:
            pcm_16k, _ = audioop.ratecv(
                pcm_phone_rate,
                2,
                1,
                int(self.input_sample_rate or self.DEFAULT_PHONE_SAMPLE_RATE),
                self.TARGET_STT_SAMPLE_RATE,
                None,
            )
        except Exception as exc:
            print(f"[{self.provider.upper()}] Failed to resample audio to 16 kHz: {exc}")
            return

        await self.queue.put(pcm_16k)

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
