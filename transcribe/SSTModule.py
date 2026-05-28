import asyncio
import numpy as np
from faster_whisper import WhisperModel
from Audio.AudioSource import AudioSource
from dotenv import load_dotenv
import os

load_dotenv()

CHUNK_BUFFER_SIZE = 150


class STTModule:
    def __init__(self, model_size=None, device=None, lang="en", preloaded_model=None):
        allowed_languages = ["en", "fr", "sw"]
        if lang not in allowed_languages:
            raise ValueError(f"Language '{lang}' not supported. Choose from {allowed_languages}")

        self.lang = lang

        if preloaded_model is not None:
            # Reuse the already-loaded WhisperModel — zero disk I/O
            self.model = preloaded_model
            print(f"[STT] Reusing preloaded Whisper model for [{lang}]")
        else:
            # Fallback: load now (test mode or preload.py wasn't run)
            resolved_size   = model_size or os.getenv("WHISPER_MODEL_SIZE", "medium").strip()
            resolved_device = device     or os.getenv("WHISPER_DEVICE", "cpu").strip()
            compute_type    = "float16" if resolved_device == "cuda" else "int8"

            print(f"[STT] 📦 Loading Whisper [{resolved_size}] on [{resolved_device}]...")
            self.model = WhisperModel(
                resolved_size,
                device=resolved_device,
                compute_type=compute_type,
                download_root=os.getenv("HF_HOME"),
            )

    def _transcribe_blocking(self, audio_data: np.ndarray) -> list:
        segments, _ = self.model.transcribe(
            audio_data,
            language=self.lang,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
        )
        return [seg.text.strip() for seg in segments if seg.text.strip()]

    async def transcribe_stream(self, audio_source: AudioSource):
        print(f"--- STT Active: Listening [{self.lang}] ---")
        loop   = asyncio.get_running_loop()
        buffer = []

        async for chunk in audio_source.get_stream():
            buffer.append(chunk)

            if len(buffer) >= CHUNK_BUFFER_SIZE:
                audio_bytes = b"".join(buffer)
                audio_data  = (
                    np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
                    / 32768.0
                )
                buffer = []

                texts = await loop.run_in_executor(
                    None, self._transcribe_blocking, audio_data
                )

                for text in texts:
                    yield text