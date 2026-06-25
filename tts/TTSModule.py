import asyncio
import os
import tempfile
import time
import wave
from pathlib import Path
from typing import Any

import numpy as np
import torch

MAX_BUFFER_CHARS = int(os.getenv("TTS_MAX_BUFFER_CHARS", "80"))

SONIOX_MODEL = os.getenv("SONIOX_TTS_MODEL", "tts-rt-v1")
SONIOX_VOICE = os.getenv("SONIOX_TTS_VOICE", "Grace")
SONIOX_AUDIO_FORMAT = os.getenv("SONIOX_TTS_AUDIO_FORMAT", "wav")
SONIOX_SAMPLE_RATE = int(os.getenv("SONIOX_TTS_SAMPLE_RATE", "24000"))

# Keep the same app-level language codes you already use in routes/sockets.
SONIOX_LANGUAGE_MAP = {
    "en": os.getenv("SONIOX_TTS_LANG_EN", "en"),
    "fr": os.getenv("SONIOX_TTS_LANG_FR", "fr"),
    "sw": os.getenv("SONIOX_TTS_LANG_SW", "sw"),
}


class TTSModule:
    """
    Streaming TTS module for phone calls using Soniox Text-to-Speech.

    Public interface is intentionally kept compatible with your current module:
        await speak_stream(text_generator, lang="en", on_first_audio=callback)

    Required environment variable:
        SONIOX_API_KEY=<your Soniox API key>

    Optional environment variables:
        SONIOX_TTS_MODEL=tts-rt-v1
        SONIOX_TTS_VOICE=Grace
        SONIOX_TTS_SAMPLE_RATE=24000
        SONIOX_TTS_AUDIO_FORMAT=wav
        TTS_MAX_BUFFER_CHARS=80
    """

    def __init__(self, output, preloaded_models: dict | None = None):
        self.output = output
        self.client = None

        # `preloaded_models` is now config-only. It lets sockets.py pass the
        # selected Soniox model/voice/language per app language without loading
        # any local TTS weights.
        self.models: dict[str, dict[str, Any]] = preloaded_models or {}

        self.default_config = {
            "engine": "soniox",
            "model": SONIOX_MODEL,
            "voice": SONIOX_VOICE,
            "audio_format": SONIOX_AUDIO_FORMAT,
            "sample_rate": SONIOX_SAMPLE_RATE,
        }
        self.language_map = dict(SONIOX_LANGUAGE_MAP)

    def _ensure_client(self):
        if self.client is not None:
            return self.client

        api_key = os.getenv("SONIOX_API_KEY")
        if not api_key:
            raise RuntimeError(
                "SONIOX_API_KEY is not set. Add it to your Hugging Face Space secrets/env vars."
            )

        try:
            from soniox import SonioxClient
        except Exception as exc:
            raise RuntimeError(
                "Soniox Python SDK is not installed. Add `soniox` to requirements.txt."
            ) from exc

        # The SDK reads SONIOX_API_KEY from the environment. Passing no args keeps
        # compatibility with the official quickstart.
        self.client = SonioxClient()
        return self.client

    def _config_for(self, lang: str) -> dict[str, Any]:
        if lang not in self.language_map:
            raise ValueError(
                f"Unsupported TTS language: {lang}. Choose from {list(self.language_map)}"
            )

        cfg = dict(self.default_config)
        cfg["language"] = self.language_map[lang]

        # Merge sockets/preload config when present. Ignore non-Soniox legacy
        # bundles defensively so rolling deploys do not accidentally use stale
        # Kokoro/OmniVoice objects.
        incoming = self.models.get(lang) or {}
        if isinstance(incoming, dict) and incoming.get("engine", "soniox") == "soniox":
            cfg.update({k: v for k, v in incoming.items() if v is not None})

        return cfg

    def _language_for(self, lang: str) -> str:
        return str(self._config_for(lang)["language"])

    async def speak_stream(self, text_generator, lang="en", on_first_audio=None):
        """
        Consume the LLM's async text stream, split it into natural sentence-sized
        chunks, generate Soniox audio, and send it to PhoneAudioOutput.
        """
        self._language_for(lang)

        buffer = ""
        first_audio_fired = False
        sentence_endings = {".", "!", "?", "\n"}

        async for chunk in text_generator:
            if not chunk:
                continue

            buffer += chunk

            should_speak = (
                    any(p in chunk for p in sentence_endings)
                    or len(buffer) >= MAX_BUFFER_CHARS
            )

            if should_speak:
                to_say, buffer = self._pop_speakable_text(buffer)
                if to_say:
                    cb = None
                    if not first_audio_fired and on_first_audio:
                        cb = on_first_audio
                        first_audio_fired = True

                    await self._generate_audio(to_say, lang, on_first_audio=cb)

        if buffer.strip():
            cb = None
            if not first_audio_fired and on_first_audio:
                cb = on_first_audio
                first_audio_fired = True

            await self._generate_audio(buffer.strip(), lang, on_first_audio=cb)

    @staticmethod
    def _pop_speakable_text(buffer: str) -> tuple[str, str]:
        buffer = buffer.strip()
        if not buffer:
            return "", ""

        if len(buffer) >= MAX_BUFFER_CHARS:
            split_idx = buffer.rfind(" ", 0, MAX_BUFFER_CHARS)
            if split_idx != -1:
                return buffer[:split_idx].strip(), buffer[split_idx:].strip()

        return buffer, ""

    async def _generate_audio(self, text: str, lang: str, on_first_audio=None):
        loop = asyncio.get_running_loop()
        t0 = time.time()

        waveform, sample_rate = await loop.run_in_executor(
            None,
            lambda: self._synthesize_soniox(text, lang),
        )

        tts_elapsed = time.time() - t0
        try:
            import stats
            stats.record_tts_latency(tts_elapsed)
        except Exception:
            pass

        if on_first_audio:
            on_first_audio()

        await self.output.send_audio(waveform, sample_rate=sample_rate)

    def _synthesize_soniox(self, text: str, lang: str) -> tuple[torch.Tensor, int]:
        """
        Generate one chunk of speech through Soniox REST TTS.

        We ask Soniox for WAV because it is simple to decode with Python's
        standard library and matches the existing PhoneAudioOutput contract:
        CPU float32 torch tensor shaped [1, samples] plus sample_rate.
        """
        client = self._ensure_client()
        cfg = self._config_for(lang)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            client.tts.generate_to_file(
                str(tmp_path),
                text=text,
                model=str(cfg.get("model", SONIOX_MODEL)),
                language=str(cfg.get("language", self.language_map.get(lang, "en"))),
                voice=str(cfg.get("voice", SONIOX_VOICE)),
                audio_format="wav",
                sample_rate=int(cfg.get("sample_rate", SONIOX_SAMPLE_RATE)),
            )
            waveform, sample_rate = self._read_wav_as_tensor(tmp_path)
            waveform = self._apply_output_envelope(waveform, sample_rate)
            return waveform, sample_rate
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass

    @staticmethod
    def _read_wav_as_tensor(path: Path) -> tuple[torch.Tensor, int]:
        with wave.open(str(path), "rb") as wf:
            channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            sample_rate = wf.getframerate()
            frames = wf.readframes(wf.getnframes())

        if not frames:
            return torch.zeros(1, 1, dtype=torch.float32), sample_rate

        if sample_width == 1:
            audio = np.frombuffer(frames, dtype=np.uint8).astype(np.float32)
            audio = (audio - 128.0) / 128.0
        elif sample_width == 2:
            audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        elif sample_width == 4:
            audio = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            raise RuntimeError(f"Unsupported WAV sample width: {sample_width} bytes")

        if channels > 1:
            audio = audio.reshape(-1, channels).mean(axis=1)

        waveform = torch.from_numpy(audio).to(dtype=torch.float32).unsqueeze(0).contiguous()
        return waveform.cpu(), int(sample_rate)

    @staticmethod
    def _apply_output_envelope(waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """
        Small fades prevent boundary clicks when many generated chunks are sent
        into an 8 kHz phone codec.
        """
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)

        waveform = waveform.to(dtype=torch.float32, device="cpu").contiguous()
        n = waveform.shape[-1]
        if n <= 1:
            return waveform

        fade_in_samples = min(int(sample_rate * 0.005), n)
        fade_out_samples = min(int(sample_rate * 0.010), n)
        silence_samples = int(sample_rate * 0.005)

        if fade_in_samples > 1:
            fade_in = torch.linspace(0.0, 1.0, fade_in_samples)
            waveform[..., :fade_in_samples] *= fade_in

        if fade_out_samples > 1:
            fade_out = torch.linspace(1.0, 0.0, fade_out_samples)
            waveform[..., -fade_out_samples:] *= fade_out

        if silence_samples > 0:
            silence = torch.zeros(waveform.shape[0], silence_samples, dtype=waveform.dtype)
            waveform = torch.cat([waveform, silence], dim=-1)

        return waveform.contiguous()
