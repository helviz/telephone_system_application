import asyncio
import inspect
import os
import re
import time
from collections.abc import AsyncIterator
from typing import Any
from uuid import uuid4


def _env_int(name: str, default: int, min_value: int | None = None, max_value: int | None = None) -> int:
    try:
        value = int(str(os.getenv(name, str(default))).strip())
    except Exception:
        value = default
    if min_value is not None:
        value = max(min_value, value)
    if max_value is not None:
        value = min(max_value, value)
    return value


def _env_optional_int(name: str) -> int | None:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return None
    try:
        return int(raw.strip())
    except Exception:
        return None


SONIOX_MODEL = os.getenv("SONIOX_TTS_MODEL", "tts-rt-v1").strip()
SONIOX_VOICE = os.getenv("SONIOX_TTS_VOICE", "Grace").strip()

# Preferred for phone calls. If your existing PhoneAudioOutput does not have
# send_mulaw_audio(), this module automatically falls back to pcm_s16le/16000
# and uses the existing send_audio() path.
SONIOX_AUDIO_FORMAT = os.getenv("SONIOX_TTS_AUDIO_FORMAT", "pcm_mulaw").strip().lower()
SONIOX_SAMPLE_RATE = _env_int("SONIOX_TTS_SAMPLE_RATE", 8000, 8000, 48000)
SONIOX_BITRATE = _env_optional_int("SONIOX_TTS_BITRATE")
SONIOX_CONNECT_TIMEOUT_SEC = float(os.getenv("SONIOX_CONNECT_TIMEOUT_SEC", "10"))

SONIOX_LANGUAGE_MAP = {
    "en": os.getenv("SONIOX_TTS_LANG_EN", "en").strip(),
    "fr": os.getenv("SONIOX_TTS_LANG_FR", "fr").strip(),
    "sw": os.getenv("SONIOX_TTS_LANG_SW", "sw").strip(),
}

# Real-time text batching. This replaces the old first-sentence trick.
TTS_MIN_CHUNK_CHARS = _env_int("TTS_MIN_CHUNK_CHARS", 16, 1, 200)
TTS_MAX_CHUNK_CHARS = _env_int("TTS_MAX_CHUNK_CHARS", 80, 10, 500)
TTS_FLUSH_INTERVAL_MS = _env_int("TTS_FLUSH_INTERVAL_MS", 80, 0, 1000)

RAW_PHONE_FORMATS = {"pcm_mulaw", "pcm_s16le"}


class TTSModule:
    """
    Soniox realtime TTS for Twilio/Telnyx phone calls.

    Public interface stays unchanged:
        await speak_stream(text_generator, lang="en", on_first_audio=callback)

    This version does not require changing PhoneAudioOutput:
      - If PhoneAudioOutput has send_mulaw_audio(), it uses Soniox pcm_mulaw/8000.
      - Otherwise it uses Soniox pcm_s16le/16000 and calls the existing send_audio().

    It also protects deployment from stale env vars such as:
        SONIOX_TTS_AUDIO_FORMAT=wav
        SONIOX_TTS_SAMPLE_RATE=16000
    by automatically switching to a realtime phone-safe raw PCM format.
    """

    def __init__(self, output, preloaded_models: dict | None = None):
        self.output = output
        self.models: dict[str, dict[str, Any]] = preloaded_models or {}
        self.client = None

        self.default_config: dict[str, Any] = {
            "engine": "soniox",
            "model": SONIOX_MODEL,
            "voice": SONIOX_VOICE,
            "language": None,
            "audio_format": SONIOX_AUDIO_FORMAT,
            "sample_rate": SONIOX_SAMPLE_RATE,
            "bitrate": SONIOX_BITRATE,
        }

        print(
            "[TTS] Soniox realtime "
            f"model={SONIOX_MODEL} voice={SONIOX_VOICE} "
            f"env_format={SONIOX_AUDIO_FORMAT} env_sample_rate={SONIOX_SAMPLE_RATE} "
            f"text_batch={TTS_MIN_CHUNK_CHARS}-{TTS_MAX_CHUNK_CHARS} chars "
            f"flush={TTS_FLUSH_INTERVAL_MS}ms"
        )

    async def aclose(self):
        client = self.client
        self.client = None
        if client is None:
            return
        close = getattr(client, "aclose", None) or getattr(client, "close", None)
        if callable(close):
            result = close()
            if inspect.isawaitable(result):
                await result

    async def _ensure_client(self):
        if self.client is not None:
            return self.client

        api_key = os.getenv("SONIOX_API_KEY")
        if not api_key:
            raise RuntimeError("SONIOX_API_KEY is not set. Add it to secrets/env vars.")

        try:
            from soniox import AsyncSonioxClient
        except Exception as exc:
            raise RuntimeError("Soniox Python SDK is not installed. Add `soniox` to requirements.txt.") from exc

        self.client = AsyncSonioxClient(api_key=api_key)
        return self.client

    def _output_supports_direct_mulaw(self) -> bool:
        return callable(getattr(self.output, "send_mulaw_audio", None))

    def _config_for(self, lang: str) -> dict[str, Any]:
        lang = (lang or "en").strip().lower()
        if lang not in SONIOX_LANGUAGE_MAP:
            raise ValueError(f"Unsupported TTS language: {lang}. Choose from {list(SONIOX_LANGUAGE_MAP)}")

        cfg = dict(self.default_config)
        cfg["language"] = SONIOX_LANGUAGE_MAP[lang]

        # sockets.py can pass a per-language config. It may still contain old
        # values from the Hugging Face env, so we sanitize below.
        incoming = self.models.get(lang) or {}
        if isinstance(incoming, dict):
            cfg.update({k: v for k, v in incoming.items() if v is not None})

        cfg["audio_format"] = str(cfg.get("audio_format") or "pcm_mulaw").strip().lower()
        cfg["sample_rate"] = int(cfg.get("sample_rate") or 8000)
        cfg["model"] = str(cfg.get("model") or SONIOX_MODEL).strip()
        cfg["voice"] = str(cfg.get("voice") or SONIOX_VOICE).strip()
        cfg["language"] = str(cfg.get("language") or SONIOX_LANGUAGE_MAP[lang]).strip()

        # Real-time phone playback should not use wav/mp3/opus/flac/aac here.
        # If a stale env var says wav, do not crash the call; switch to a raw
        # format that the phone output can consume immediately.
        if cfg["audio_format"] not in RAW_PHONE_FORMATS:
            old_format = cfg["audio_format"]
            if self._output_supports_direct_mulaw():
                cfg["audio_format"] = "pcm_mulaw"
                cfg["sample_rate"] = 8000
            else:
                cfg["audio_format"] = "pcm_s16le"
                cfg["sample_rate"] = 16000
            print(
                f"[TTS] Overriding realtime audio_format={old_format!r} to "
                f"{cfg['audio_format']}/{cfg['sample_rate']} for phone playback."
            )

        # Direct mu-law passthrough must be 8 kHz. If PhoneAudioOutput does not
        # support send_mulaw_audio(), fall back to PCM16 + send_audio().
        if cfg["audio_format"] == "pcm_mulaw":
            if self._output_supports_direct_mulaw():
                if cfg["sample_rate"] != 8000:
                    print("[TTS] For pcm_mulaw, forcing SONIOX_TTS_SAMPLE_RATE=8000.")
                cfg["sample_rate"] = 8000
            else:
                print(
                    "[TTS] PhoneAudioOutput has no send_mulaw_audio(); "
                    "using pcm_s16le/16000 through existing send_audio()."
                )
                cfg["audio_format"] = "pcm_s16le"
                cfg["sample_rate"] = 16000

        return cfg

    @staticmethod
    async def _maybe_await(result):
        if inspect.isawaitable(result):
            return await result
        return result

    @staticmethod
    def _clean_text_chunk(text: Any) -> str:
        text = str(text or "")
        if not text:
            return ""
        # Keep normal spaces from the LLM stream so words do not join.
        text = text.replace("helpprovide", "help provide")
        text = re.sub(r"[\t\r\n]+", " ", text)
        text = re.sub(r" {2,}", " ", text)
        return text

    @staticmethod
    def _should_flush(buffer: str, elapsed_ms: float) -> bool:
        if not buffer:
            return False
        if len(buffer) >= TTS_MAX_CHUNK_CHARS:
            return True
        if buffer[-1:] in ".!?:;\n" and len(buffer.strip()) >= 4:
            return True
        if len(buffer) >= TTS_MIN_CHUNK_CHARS and (
            buffer[-1].isspace() or elapsed_ms >= TTS_FLUSH_INTERVAL_MS
        ):
            return True
        return False

    async def _iter_tts_text_chunks(self, text_generator) -> AsyncIterator[str]:
        buffer = ""
        last_flush = time.monotonic()

        async for raw in text_generator:
            chunk = self._clean_text_chunk(raw)
            if not chunk:
                continue

            buffer += chunk
            now = time.monotonic()
            elapsed_ms = (now - last_flush) * 1000.0

            if self._should_flush(buffer, elapsed_ms):
                out = buffer
                buffer = ""
                last_flush = now
                if out.strip():
                    yield out

        if buffer.strip():
            yield buffer

    def _build_realtime_config(self, cfg: dict[str, Any]):
        from soniox.types import RealtimeTTSConfig

        kwargs = {
            "stream_id": f"tts-{uuid4()}",
            "model": cfg["model"],
            "language": cfg["language"],
            "voice": cfg["voice"],
            "audio_format": cfg["audio_format"],
            "sample_rate": cfg["sample_rate"],
        }
        if cfg.get("bitrate") is not None:
            kwargs["bitrate"] = int(cfg["bitrate"])
        return RealtimeTTSConfig(**kwargs)

    async def speak_stream(self, text_generator, lang="en", on_first_audio=None):
        cfg = self._config_for(lang)
        print(
            f"[TTS] Realtime session [{lang}] "
            f"format={cfg['audio_format']} sample_rate={cfg['sample_rate']}"
        )

        client = await self._ensure_client()
        realtime_config = self._build_realtime_config(cfg)

        begin_stream = getattr(self.output, "begin_stream", None)
        if callable(begin_stream):
            result = begin_stream()
            if inspect.isawaitable(result):
                await result

        t0 = time.time()
        async with client.realtime.tts.connect(
            config=realtime_config,
            connect_timeout_sec=SONIOX_CONNECT_TIMEOUT_SEC,
        ) as session:
            sender_task = asyncio.create_task(self._send_text_to_soniox(session, text_generator))
            first_audio_box = {"fired": False}
            receiver_task = asyncio.create_task(
                self._receive_audio_chunks(session, cfg, t0, on_first_audio, first_audio_box)
            )

            try:
                await asyncio.gather(sender_task, receiver_task)
            except asyncio.CancelledError:
                sender_task.cancel()
                receiver_task.cancel()
                await self._cancel_soniox_stream(session)
                raise
            except Exception:
                sender_task.cancel()
                receiver_task.cancel()
                await self._cancel_soniox_stream(session)
                raise

    async def _send_text_to_soniox(self, session, text_generator) -> None:
        sent_any = False
        async for text in self._iter_tts_text_chunks(text_generator):
            await self._maybe_await(session.send_text_chunk(text, text_end=False))
            sent_any = True

        finish = getattr(session, "finish", None)
        if callable(finish):
            await self._maybe_await(finish())
        else:
            # Safe finalization even if no text was sent.
            await self._maybe_await(session.send_text_chunk("", text_end=True))

    async def _receive_audio_chunks(
        self,
        session,
        cfg: dict[str, Any],
        t0: float,
        on_first_audio,
        first_audio_box: dict,
    ) -> None:
        async for audio_chunk in session.receive_audio_chunks():
            if not audio_chunk:
                continue

            if not first_audio_box["fired"]:
                first_audio_box["fired"] = True
                elapsed = time.time() - t0
                try:
                    import stats
                    stats.record_tts_latency(elapsed)
                except Exception:
                    pass
                if on_first_audio:
                    on_first_audio()

            await self._send_audio_chunk_to_phone(audio_chunk, cfg)

        send_mark = getattr(self.output, "send_mark", None)
        if callable(send_mark):
            result = send_mark()
            if inspect.isawaitable(result):
                await result

    async def _send_audio_chunk_to_phone(self, audio_chunk: bytes, cfg: dict[str, Any]) -> None:
        audio_format = cfg["audio_format"]
        sample_rate = int(cfg["sample_rate"])

        if audio_format == "pcm_mulaw" and sample_rate == 8000 and self._output_supports_direct_mulaw():
            await self.output.send_mulaw_audio(audio_chunk, mark=False)
            return

        if audio_format == "pcm_s16le":
            # Existing PhoneAudioOutput.send_audio() accepts bytes as PCM16.
            await self.output.send_audio(audio_chunk, sample_rate=sample_rate)
            return

        raise RuntimeError(
            f"Cannot stream Soniox audio_format={audio_format!r}, sample_rate={sample_rate} "
            "to this phone output. Use pcm_mulaw/8000 or pcm_s16le/16000."
        )

    async def _cancel_soniox_stream(self, session) -> None:
        cancel = getattr(session, "cancel", None)
        if callable(cancel):
            try:
                await self._maybe_await(cancel())
            except Exception as exc:
                print(f"[TTS] Soniox cancel failed: {exc}")
