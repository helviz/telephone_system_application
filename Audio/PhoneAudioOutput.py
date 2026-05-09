import base64
import audioop
import json
import torch
import numpy as np
from Audio.AudioOutput import AudioOutput


class PhoneAudioOutput(AudioOutput):
    """
    Handles audio output for both Twilio and Telnyx websocket streams.
    Both platforms use the same mu-law encoding over websocket media events.
    """

    PROVIDERS = {"twilio", "telnyx"}

    def __init__(self, websocket, provider: str = "twilio"):
        if provider not in self.PROVIDERS:
            raise ValueError(f"Unsupported provider: '{provider}'. Choose from {self.PROVIDERS}")

        self.ws = websocket
        self.provider = provider

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

        await self.ws.send_text(json.dumps({
            "event": "media",
            "media": {"payload": payload}
        }))

    @classmethod
    def twilio(cls, websocket) -> "PhoneAudioOutput":
        return cls(websocket, provider="twilio")

    @classmethod
    def telnyx(cls, websocket) -> "PhoneAudioOutput":
        return cls(websocket, provider="telnyx")


# # Twilio
# output = PhoneAudioOutput.twilio(websocket)
#
# # Telnyx
# output = PhoneAudioOutput.telnyx(websocket)
#
# # Or directly
# output = PhoneAudioOutput(websocket, provider="twilio")