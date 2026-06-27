import asyncio
import os
import re
import tempfile
import time
import wave
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


MAX_BUFFER_CHARS = int(os.getenv("TTS_MAX_BUFFER_CHARS", "120"))

# New recommended mode for phone calls:
#   request 1 = first complete sentence as soon as it is available
#   request 2 = the rest of the completed answer after LLM finishes
FIRST_SENTENCE_THEN_REST = _env_bool("TTS_FIRST_SENTENCE_THEN_REST", False)

# Existing modes kept for compatibility.
WHOLE_RESPONSE_MODE = _env_bool("TTS_WHOLE_RESPONSE_MODE", False)
SENTENCE_CHUNKS = _env_bool("TTS_SENTENCE_CHUNKS", False)
FLUSH_ON_MAX_CHARS = _env_bool("TTS_FLUSH_ON_MAX_CHARS", True)
DROP_INCOMPLETE_TRAILING = _env_bool("TTS_DROP_INCOMPLETE_TRAILING", True)

SONIOX_MODEL = os.getenv("SONIOX_TTS_MODEL", "tts-rt-v1")
SONIOX_VOICE = os.getenv("SONIOX_TTS_VOICE", "Grace")
SONIOX_AUDIO_FORMAT = os.getenv("SONIOX_TTS_AUDIO_FORMAT", "wav")
SONIOX_SAMPLE_RATE = int(os.getenv("SONIOX_TTS_SAMPLE_RATE", "24000"))

SONIOX_LANGUAGE_MAP = {
    "en": os.getenv("SONIOX_TTS_LANG_EN", "en"),
    "fr": os.getenv("SONIOX_TTS_LANG_FR", "fr"),
    "sw": os.getenv("SONIOX_TTS_LANG_SW", "sw"),
}


class TTSModule:
    """
    Soniox TTS for phone calls.

    Supported output strategies:
      1. TTS_FIRST_SENTENCE_THEN_REST=true
         - Send the first complete sentence as soon as it appears.
         - Continue consuming the LLM stream while that first TTS request is running.
         - After the LLM finishes, send the remaining completed text as one second request.

      2. TTS_WHOLE_RESPONSE_MODE=true
         - Wait for the full LLM response, then send one TTS request.

      3. TTS_SENTENCE_CHUNKS=true
         - Send every complete sentence separately.

      4. fallback legacy stream mode
         - Similar to your earlier buffer/punctuation strategy.
    """

    def __init__(self, output, preloaded_models: dict | None = None):
        self.output = output
        self.client = None
        self.models: dict[str, dict[str, Any]] = preloaded_models or {}

        self.default_config = {
            "engine": "soniox",
            "model": SONIOX_MODEL,
            "voice": SONIOX_VOICE,
            "audio_format": SONIOX_AUDIO_FORMAT,
            "sample_rate": SONIOX_SAMPLE_RATE,
        }
        self.language_map = dict(SONIOX_LANGUAGE_MAP)

        print(
            "[TTS] mode="
            f"first_sentence_then_rest={FIRST_SENTENCE_THEN_REST}, "
            f"whole_response={WHOLE_RESPONSE_MODE}, "
            f"sentence_chunks={SENTENCE_CHUNKS}, "
            f"flush_on_max_chars={FLUSH_ON_MAX_CHARS}, "
            f"max_chars={MAX_BUFFER_CHARS}, "
            f"sample_rate={SONIOX_SAMPLE_RATE}"
        )

    def _ensure_client(self):
        if self.client is not None:
            return self.client

        api_key = os.getenv("SONIOX_API_KEY")
        if not api_key:
            raise RuntimeError("SONIOX_API_KEY is not set. Add it to secrets/env vars.")

        try:
            from soniox import SonioxClient
        except Exception as exc:
            raise RuntimeError("Soniox Python SDK is not installed. Add `soniox` to requirements.txt.") from exc

        self.client = SonioxClient()
        return self.client

    def _config_for(self, lang: str) -> dict[str, Any]:
        if lang not in self.language_map:
            raise ValueError(f"Unsupported TTS language: {lang}. Choose from {list(self.language_map)}")

        cfg = dict(self.default_config)
        cfg["language"] = self.language_map[lang]

        incoming = self.models.get(lang) or {}
        if isinstance(incoming, dict) and incoming.get("engine", "soniox") == "soniox":
            cfg.update({k: v for k, v in incoming.items() if v is not None})

        return cfg

    def _language_for(self, lang: str) -> str:
        return str(self._config_for(lang)["language"])

    async def speak_stream(self, text_generator, lang="en", on_first_audio=None):
        self._language_for(lang)

        if FIRST_SENTENCE_THEN_REST:
            await self._speak_first_sentence_then_rest(text_generator, lang, on_first_audio)
            return

        if WHOLE_RESPONSE_MODE:
            await self._speak_whole_response(text_generator, lang, on_first_audio)
            return

        if SENTENCE_CHUNKS:
            await self._speak_sentence_chunks(text_generator, lang, on_first_audio)
            return

        await self._speak_legacy_stream(text_generator, lang, on_first_audio)

    async def _speak_first_sentence_then_rest(self, text_generator, lang: str, on_first_audio=None):
        """
        Two-trip strategy:
          - First Soniox request: first complete sentence only.
          - Second Soniox request: all remaining completed text.

        This lowers perceived latency without making every sentence its own API call.
        """
        t_mode = time.time()
        first_buffer = ""
        rest_buffer = ""
        first_task: asyncio.Task | None = None
        first_audio_fired = False

        def _first_audio_cb_once():
            nonlocal first_audio_fired
            if first_audio_fired:
                return None
            first_audio_fired = True
            return on_first_audio

        try:
            async for chunk in text_generator:
                if not chunk:
                    continue

                if first_task is None:
                    first_buffer += chunk
                    first_sentence, remainder = self._pop_first_complete_sentence(first_buffer)
                    if first_sentence:
                        cb = _first_audio_cb_once()
                        print(
                            f"[TTS] First-sentence mode: request #1 "
                            f"({len(first_sentence)} chars) after {time.time() - t_mode:.2f}s."
                        )
                        first_task = asyncio.create_task(
                            self._generate_audio(first_sentence, lang, on_first_audio=cb)
                        )
                        rest_buffer = remainder
                else:
                    rest_buffer += chunk

            if first_task is None:
                # No sentence boundary arrived. Fall back to one clean request.
                final_text = self._finalize_for_speech(first_buffer, allow_add_period=True)
                if final_text:
                    cb = _first_audio_cb_once()
                    print(f"[TTS] First-sentence mode fallback: one request ({len(final_text)} chars).")
                    await self._generate_audio(final_text, lang, on_first_audio=cb)
                return

            # Preserve playback order: first audio must finish sending before the rest.
            await first_task

            rest_text = self._finalize_for_speech(rest_buffer, allow_add_period=False)
            if rest_text:
                print(f"[TTS] First-sentence mode: request #2 rest ({len(rest_text)} chars).")
                await self._generate_audio(rest_text, lang, on_first_audio=None)
            else:
                print("[TTS] First-sentence mode: no completed rest text to synthesize.")

        except asyncio.CancelledError:
            if first_task and not first_task.done():
                first_task.cancel()
            raise

    async def _speak_whole_response(self, text_generator, lang: str, on_first_audio=None):
        parts = []
        async for chunk in text_generator:
            if chunk:
                parts.append(chunk)

        text = self._finalize_for_speech("".join(parts), allow_add_period=True)
        if not text:
            return

        print(f"[TTS] Whole-response mode: synthesizing one Soniox request ({len(text)} chars).")
        await self._generate_audio(text, lang, on_first_audio=on_first_audio)

    async def _speak_sentence_chunks(self, text_generator, lang: str, on_first_audio=None):
        buffer = ""
        first_audio_fired = False

        async for chunk in text_generator:
            if not chunk:
                continue

            buffer += chunk
            while True:
                sentence, remainder = self._pop_first_complete_sentence(buffer)
                if not sentence:
                    break

                cb = None
                if not first_audio_fired and on_first_audio:
                    cb = on_first_audio
                    first_audio_fired = True

                await self._generate_audio(sentence, lang, on_first_audio=cb)
                buffer = remainder

        tail = self._finalize_for_speech(buffer, allow_add_period=True)
        if tail:
            cb = None
            if not first_audio_fired and on_first_audio:
                cb = on_first_audio
            await self._generate_audio(tail, lang, on_first_audio=cb)

    async def _speak_legacy_stream(self, text_generator, lang: str, on_first_audio=None):
        buffer = ""
        first_audio_fired = False
        sentence_endings = {".", "!", "?", "\n"}

        async for chunk in text_generator:
            if not chunk:
                continue

            buffer += chunk
            should_speak = any(p in chunk for p in sentence_endings) or (
                FLUSH_ON_MAX_CHARS and len(buffer) >= MAX_BUFFER_CHARS
            )

            if should_speak:
                to_say, buffer = self._pop_speakable_text(buffer)
                if to_say:
                    cb = None
                    if not first_audio_fired and on_first_audio:
                        cb = on_first_audio
                        first_audio_fired = True
                    await self._generate_audio(to_say, lang, on_first_audio=cb)

        final_text = self._finalize_for_speech(buffer, allow_add_period=True)
        if final_text:
            cb = None
            if not first_audio_fired and on_first_audio:
                cb = on_first_audio
            await self._generate_audio(final_text, lang, on_first_audio=cb)

    @staticmethod
    def _clean_text(text: str) -> str:
        text = re.sub(r"\s+", " ", (text or "").strip())
        # Fix common stream-join artifact seen in logs: "helpprovide".
        text = text.replace("helpprovide", "help provide")
        return text

    @classmethod
    def _pop_first_complete_sentence(cls, buffer: str) -> tuple[str, str]:
        """Return the first complete sentence and the remaining tail."""
        buffer = cls._clean_text(buffer)
        if not buffer:
            return "", ""

        for idx, ch in enumerate(buffer):
            if ch not in ".!?":
                continue

            # Accept punctuation as a sentence boundary if it is the end of the
            # available buffer or followed by space/quote/closing bracket.
            next_char = buffer[idx + 1] if idx + 1 < len(buffer) else ""
            if not next_char or next_char.isspace() or next_char in "'\")]}":
                return buffer[:idx + 1].strip(), buffer[idx + 1:].strip()

        return "", buffer

    @classmethod
    def _finalize_for_speech(cls, text: str, allow_add_period: bool) -> str:
        """Avoid sending unfinished trailing phrases such as 'If the pain is'."""
        text = cls._clean_text(text)
        if not text:
            return ""

        if text[-1] in ".!?":
            return text

        if DROP_INCOMPLETE_TRAILING:
            last_stop = max(text.rfind("."), text.rfind("!"), text.rfind("?"))
            if last_stop >= 20:
                return text[:last_stop + 1].strip()

            if not allow_add_period:
                print(f"[TTS] Dropping incomplete trailing text: {text!r}")
                return ""

        return text + "." if allow_add_period else ""

    @staticmethod
    def _pop_speakable_text(buffer: str) -> tuple[str, str]:
        buffer = TTSModule._clean_text(buffer)
        if not buffer:
            return "", ""

        if FLUSH_ON_MAX_CHARS and len(buffer) >= MAX_BUFFER_CHARS:
            split_idx = buffer.rfind(" ", 0, MAX_BUFFER_CHARS)
            if split_idx != -1:
                return buffer[:split_idx].strip(), buffer[split_idx:].strip()

        return buffer, ""

    async def _generate_audio(self, text: str, lang: str, on_first_audio=None):
        text = self._clean_text(text)
        if not text:
            return

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
