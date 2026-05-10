from faster_whisper import WhisperModel
import numpy as np
from Audio.AudioSource import AudioSource
from dotenv import load_dotenv
import os

load_dotenv()

hf_token = os.getenv("HF_TOKEN")


class STTModule:
    def __init__(self, model_size="base", device="cpu", lang="en"):
        # Restrict the language to the three specified options
        allowed_languages = ["en", "fr", "sw"]
        if lang not in allowed_languages:
            raise ValueError(f"Language '{lang}' not supported. Choose from {allowed_languages}")

        self.lang = lang
        # Initializing with "medium" and int8 for a balance of speed/accuracy
        self.model = WhisperModel(model_size, device=device, compute_type="int8")

    def transcribe_stream(self, audio_source: AudioSource):
        """
        Consumes the audio source stream and yields transcribed text.
        """
        print(f"--- System Active: Listening [{self.lang}] ---")

        buffer = []
        for chunk in audio_source.get_stream():
            buffer.append(chunk)

            # Process buffer every ~100 chunks
            if len(buffer) > 100:
                audio_bytes = b"".join(buffer)
                audio_data = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

                # We pass self.lang to 'language' to force the model to stay in that mode
                segments, _ = self.model.transcribe(
                    audio_data,
                    language=self.lang,
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=500)
                )

                for segment in segments:
                    if segment.text.strip():
                        yield segment.text.strip()

                buffer = []  # Clear buffer after processing