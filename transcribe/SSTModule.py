import asyncio
import numpy as np
from faster_whisper import WhisperModel
from Audio.AudioSource import AudioSource
from dotenv import load_dotenv
import os

load_dotenv()

hf_token = os.getenv("HF_TOKEN")

# How many 20ms chunks (at 16kHz, each PCM chunk ≈ 640 bytes) to buffer before
# sending to Whisper. 150 chunks ≈ 3 seconds of audio — enough for a sentence.
CHUNK_BUFFER_SIZE = 150


class STTModule:
    def __init__(self, model_size="small", device="cpu", lang="en"):
        allowed_languages = ["en", "fr", "sw"]
        if lang not in allowed_languages:
            raise ValueError(f"Language '{lang}' not supported. Choose from {allowed_languages}")

        self.lang = lang
        self.model = WhisperModel(model_size, device=device, compute_type="int8")

    def _transcribe_blocking(self, audio_data: np.ndarray) -> list:
        """
        Runs Whisper synchronously. Called via run_in_executor so it does not
        block the asyncio event loop.
        """
        segments, _ = self.model.transcribe(
            audio_data,
            language=self.lang,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
        )
        return [seg.text.strip() for seg in segments if seg.text.strip()]

    async def transcribe_stream(self, audio_source: AudioSource):
        """
        Async generator. Consumes the async audio source, buffers chunks,
        offloads Whisper inference to a thread pool, and yields transcript strings.
        """
        print(f"--- STT Active: Listening [{self.lang}] ---")
        loop = asyncio.get_running_loop()
        buffer = []

        async for chunk in audio_source.get_stream():
            buffer.append(chunk)

            if len(buffer) >= CHUNK_BUFFER_SIZE:
                audio_bytes = b"".join(buffer)
                audio_data = (
                    np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
                    / 32768.0
                )
                buffer = []

                # Offload blocking Whisper call so the event loop stays free
                texts = await loop.run_in_executor(
                    None, self._transcribe_blocking, audio_data
                )

                for text in texts:
                    yield text