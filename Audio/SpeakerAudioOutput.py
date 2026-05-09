import torch
import numpy as np
import sounddevice as sd


class SpeakerAudioOutput:
    async def send_audio(self, waveform, sample_rate=16000):
        if isinstance(waveform, torch.Tensor):
            audio = waveform.detach().cpu().numpy().squeeze()
        else:
            audio = np.frombuffer(waveform, dtype=np.int16).astype("float32") / 32767.0

        sd.play(audio.astype("float32"), samplerate=sample_rate)
        sd.wait()