import base64
import audioop
import json
import asyncio
import torch
import numpy as np
from Audio.AudioOutput import AudioOutput


class PhoneAudioOutput(AudioOutput):
    """
    Handles audio output for both Twilio and Telnyx websocket streams.

    FIX: The original implementation sent media events without a streamSid.
    Twilio requires every outbound media message to include the streamSid
    (received in the 'start' event) otherwise it drops the audio silently.

    The source (PhoneStreamSource) captures the streamSid — we accept it
    here via set_stream_sid() which sockets.py calls after the start event
    is received.

    Also adds a 'mark' event after each audio chunk so Twilio can sequence
    playback correctly and detect when the assistant has finished speaking.
    """

    PROVIDERS = {"twilio", "telnyx"}
    _mark_counter = 0

    def __init__(self, websocket, provider: str = "twilio"):
        if provider not in self.PROVIDERS:
            raise ValueError(f"Unsupported provider: '{provider}'. Choose from {self.PROVIDERS}")

        self.ws = websocket
        self.provider = provider
        self.stream_sid: str | None = None

    def set_stream_sid(self, sid: str):
        """Called by sockets.py once the 'start' event is received."""
        self.stream_sid = sid

    async def send_audio(self, audio, sample_rate=16000):
        # Handle tensor input
        if isinstance(audio, torch.Tensor):
            audio_np = audio.detach().cpu().numpy().squeeze()
            pcm16k = (audio_np * 32767).astype("int16").tobytes()
        else:
            pcm16k = audio

        # Downsample 16k → 8k
        pcm_8k, _ = audioop.ratecv(pcm16k, 2, 1, sample_rate, 8000, None)

        # PCM → μ-law
        mulaw = audioop.lin2ulaw(pcm_8k, 2)

        # Encode base64
        payload = base64.b64encode(mulaw).decode("utf-8")

        # Build the media message — include streamSid for Twilio
        media_msg: dict = {
            "event": "media",
            "media": {"payload": payload},
        }
        if self.stream_sid:
            media_msg["streamSid"] = self.stream_sid

        await self.ws.send_text(json.dumps(media_msg))

        # Send a mark event so Twilio knows this chunk finished playing
        PhoneAudioOutput._mark_counter += 1
        mark_msg: dict = {
            "event": "mark",
            "mark":  {"name": f"chunk_{PhoneAudioOutput._mark_counter}"},
        }
        if self.stream_sid:
            mark_msg["streamSid"] = self.stream_sid

        await self.ws.send_text(json.dumps(mark_msg))

    @classmethod
    def twilio(cls, websocket) -> "PhoneAudioOutput":
        return cls(websocket, provider="twilio")

    @classmethod
    def telnyx(cls, websocket) -> "PhoneAudioOutput":
        return cls(websocket, provider="telnyx")