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

# For Twilio/Telnyx phone media, request exactly what the phone stream needs:
# 8 kHz G.711 mu-law. This avoids WAV files, resampling, and re-encoding.
SONIOX_AUDIO_FORMAT = os.getenv("SONIOX_TTS_AUDIO_FORMAT", "pcm_mulaw").strip()
SONIOX_SAMPLE_RATE = _env_int("SONIOX_TTS_SAMPLE_RATE", 8000, 8000, 48000)
SONIOX_BITRATE = _env_optional_int("SONIOX_TTS_BITRATE")
SONIOX_CONNECT_TIMEOUT_SEC = float(os.getenv("SONIOX_CONNECT_TIMEOUT_SEC", "10"))

SONIOX_LANGUAGE_MAP = {
    "en": os.getenv("SONIOX_TTS_LANG_EN", "en").strip(),
    "fr": os.getenv("SONIOX_TTS_LANG_FR", "fr").strip(),
    "sw": os.getenv("SONIOX_TTS_LANG_SW", "sw").strip(),
}

# Real-time text batching. This replaces the old first-sentence trick.
# It sends small natural pieces, not whole sentences/responses.
TTS_MIN_CHUNK_CHARS = _env_int("TTS_MIN_CHUNK_CHARS", 16, 1, 200)
TTS_MAX_CHUNK_CHARS = _env_int("TTS_MAX_CHUNK_CHARS", 80, 10, 500)
TTS_FLUSH_INTERVAL_MS = _env_int("TTS_FLUSH_INTERVAL_MS", 80, 0, 1000)

PHONE_SAFE_FORMATS = {"pcm_mulaw", "pcm_s16le"}


class TTSModule:
    """
    Soniox real-time TTS for phone calls.

    Public interface stays the same:
        await speak_stream(text_generator, lang="en", on_first_audio=callback)

    Design:
      - Send LLM text chunks into Soniox immediately over the realtime SDK.
      - Receive Soniox audio chunks concurrently.
      - Forward each audio chunk to PhoneAudioOutput immediately.
      - Fire on_first_audio when the first audio bytes arrive, not after synthesis ends.
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
            f"format={SONIOX_AUDIO_FORMAT} sample_rate={SONIOX_SAMPLE_RATE} "
            f"text_batch={TTS_MIN_CHUNK_CHARS}-{TTS_MAX_CHUNK_CHARS} chars "
            f"flush={TTS_FLUSH_INTERVAL_MS}ms"
        )

    async def aclose(self):
        """Optional cleanup hook for app shutdown."""
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

    def _config_for(self, lang: str) -> dict[str, Any]:
        lang = (lang or "en").strip().lower()
        if lang not in SONIOX_LANGUAGE_MAP:
            raise ValueError(f"Unsupported TTS language: {lang}. Choose from {list(SONIOX_LANGUAGE_MAP)}")

        cfg = dict(self.default_config)
        cfg["language"] = SONIOX_LANGUAGE_MAP[lang]

        incoming = self.models.get(lang) or {}
        if isinstance(incoming, dict):
            cfg.update({k: v for k, v in incoming.items() if v is not None})

        cfg["audio_format"] = str(cfg.get("audio_format") or "pcm_mulaw").strip()
        cfg["sample_rate"] = int(cfg.get("sample_rate") or 8000)
        cfg["model"] = str(cfg.get("model") or SONIOX_MODEL).strip()
        cfg["voice"] = str(cfg.get("voice") or SONIOX_VOICE).strip()
        cfg["language"] = str(cfg.get("language") or SONIOX_LANGUAGE_MAP[lang]).strip()

        if cfg["audio_format"] not in PHONE_SAFE_FORMATS:
            raise RuntimeError(
                "For realtime phone playback, set SONIOX_TTS_AUDIO_FORMAT to "
                "`pcm_mulaw` preferably, or `pcm_s16le`. Do not use `wav` here."
            )

        if cfg["audio_format"] == "pcm_mulaw" and cfg["sample_rate"] != 8000:
            raise RuntimeError("For direct phone PCMU passthrough, use SONIOX_TTS_SAMPLE_RATE=8000.")

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

        # Keep normal leading/trailing spaces from the LLM stream so words do not join.
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
        if len(buffer) >= TTS_MIN_CHUNK_CHARS and (buffer[-1].isspace() or elapsed_ms >= TTS_FLUSH_INTERVAL_MS):
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
        client = await self._ensure_client()
        realtime_config = self._build_realtime_config(cfg)

        if hasattr(self.output, "begin_stream"):
            self.output.begin_stream()

        t0 = time.time()
        async with client.realtime.tts.connect(
            config=realtime_config,
            connect_timeout_sec=SONIOX_CONNECT_TIMEOUT_SEC,
        ) as session:
            sender_task = asyncio.create_task(self._send_text_to_soniox(session, text_generator))
            first_audio_box = {"fired": False}
            receiver_task = asyncio.create_task(
                self._receive_audio_from_soniox(session, cfg, t0, on_first_audio, first_audio_box)
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

        # Always finish. If no text was sent, this cleanly terminates the stream.
        finish = getattr(session, "finish", None)
        if callable(finish):
            await self._maybe_await(finish())
        elif sent_any:
            await self._maybe_await(session.send_text_chunk("", text_end=True))

    async def _receive_audio_from_soniox(self, session, cfg: dict[str, Any], t0: float, on_first_audio, first_audio_box: dict) -> None:
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

        if hasattr(self.output, "send_mark"):
            result = self.output.send_mark()
            if inspect.isawaitable(result):
                await result

    async def _receive_audio_from_soniox(self, *args, **kwargs) -> None:
        # Kept only to make accidental old references fail safely.
        raise RuntimeError("Use _receive_audio_from_soniox instead.")

    @staticmethod
    def _set_nonlocal_first_audio():
        return True

    async def _send_audio_chunk_to_phone(self, audio_chunk: bytes, cfg: dict[str, Any]) -> None:
        audio_format = cfg["audio_format"]
        sample_rate = int(cfg["sample_rate"])

        if audio_format == "pcm_mulaw" and sample_rate == 8000 and hasattr(self.output, "send_mulaw_audio"):
            await self.output.send_mulaw_audio(audio_chunk, mark=False)
            return

        if audio_format == "pcm_s16le":
            await self.output.send_audio(audio_chunk, sample_rate=sample_rate)
            return

        raise RuntimeError(
            f"Cannot stream Soniox audio_format={audio_format!r}, sample_rate={sample_rate} "
            "to this phone output. Use pcm_mulaw/8000 with PhoneAudioOutput.send_mulaw_audio."
        )

    async def _cancel_soniox_stream(self, session) -> None:
        cancel = getattr(session, "cancel", None)
        if callable(cancel):
            try:
                await self._maybe_await(cancel())
            except Exception as exc:
                print(f"[TTS] Soniox cancel failed: {exc}")
