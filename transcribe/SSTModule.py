from faster_whisper import WhisperModel
import numpy as np
from Audio.AudioSource import AudioSource
from dotenv import load_dotenv
import os

load_dotenv()

hf_token = os.getenv("HF_TOKEN")

class STTModule:
    def __init__(self, model_size="base", device="cpu"):
        self.model = WhisperModel(model_size, device=device, compute_type="int8")

    def transcribe_stream(self, audio_source: AudioSource):
        """
        Consumes the audio source stream and yields transcribed text
        whenever a voice segment is completed.
        """
        print("--- System Active: Listening... ---")

        # faster-whisper's transcribe method can take a generator or a buffer.
        # However, for real-time, we typically batch the stream into small
        # manageable segments to avoid high latency.

        buffer = []
        for chunk in audio_source.get_stream():
            buffer.append(chunk)

            # We process the buffer every ~2 seconds of audio collected
            # or you can implement a more complex rolling buffer here.
            if len(buffer) > 100:  # Adjust based on chunk size for timing
                audio_bytes = b"".join(buffer)
                audio_data = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

                # vad_filter=True handles the silence removal internally
                segments, _ = self.model.transcribe(
                    audio_data,
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=500)
                )

                for segment in segments:
                    if segment.text.strip():
                        yield segment.text.strip()

                buffer = []  # Clear buffer after processing