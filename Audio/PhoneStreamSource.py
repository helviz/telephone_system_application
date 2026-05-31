import json
import base64
import audioop
import asyncio
from Audio.AudioSource import AudioSource


class PhoneStreamSource(AudioSource):
    """
    Handles incoming audio streams from both Twilio and Telnyx.
    Both platforms send mu-law encoded audio at 8kHz over websocket media events.

    FIX: The original implementation discarded the 'start' event entirely.
    Twilio sends the streamSid in the 'start' event — it must be captured
    here and passed to PhoneAudioOutput so outbound media messages include it.
    Without streamSid, Twilio silently drops all outbound audio.
    """

    PROVIDERS = {"twilio", "telnyx"}

    def __init__(self, provider: str = "twilio"):
        if provider not in self.PROVIDERS:
            raise ValueError(f"Unsupported provider: '{provider}'. Choose from {self.PROVIDERS}")

        self.provider = provider
        self.queue = asyncio.Queue()
        self.stream_sid: str | None = None   # populated on 'start' event

    async def add_data(self, websocket_message: str):
        packet = json.loads(websocket_message)
        event = packet.get("event")

        if event == "start":
            # Twilio sends streamSid here — capture it for outbound audio
            self.stream_sid = packet.get("start", {}).get("streamSid")
            print(f"[{self.provider.upper()}] Stream started — SID: {self.stream_sid}")

        elif event == "media":
            payload = packet["media"]["payload"]

            # Decode base64 μ-law (8kHz)
            chunk = base64.b64decode(payload)

            # μ-law → PCM int16
            pcm_8k = audioop.ulaw2lin(chunk, 2)

            # Upsample 8kHz → 16kHz (Whisper-friendly)
            pcm_16k, _ = audioop.ratecv(pcm_8k, 2, 1, 8000, 16000, None)

            await self.queue.put(pcm_16k)

        elif event == "stop":
            print(f"[{self.provider.upper()}] Stream stop event received.")

    async def get_stream(self):
        """Async generator for STT pipeline consumption."""
        while True:
            chunk = await self.queue.get()
            yield chunk

    @classmethod
    def twilio(cls) -> "PhoneStreamSource":
        return cls(provider="twilio")

    @classmethod
    def telnyx(cls) -> "PhoneStreamSource":
        return cls(provider="telnyx")