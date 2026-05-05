"""
STT Service — faster-whisper + Silero VAD
==========================================
Endpoints:

  GET  /health               — liveness probe
  GET  /vad_config           — inspect VAD tuning parameters
  POST /transcribe           — classic: accepts a complete audio file, returns transcript
  POST /transcribe_vad       — VAD-aware: full audio file → VAD → Whisper per segment
  WS   /stream/{call_sid}    — Twilio Media Stream WebSocket
                               Receives mulaw-encoded audio frames from Twilio,
                               runs Silero VAD in real time, flushes speech
                               segments to Whisper, and returns transcripts + TwiML
                               responses over the same socket.

Twilio Media Stream flow
------------------------
1. Twilio calls POST /twilio/voice  (handled in the API gateway service).
   The gateway responds with TwiML <Stream> pointing at wss://<host>/stream/<CallSid>.
2. Twilio opens the WebSocket and sends JSON messages:
      {"event": "connected"}
      {"event": "start",   "start": {"streamSid": ..., "callSid": ...}}
      {"event": "media",   "media": {"payload": "<base64-mulaw>"}}
      {"event": "stop"}
3. Each "media" payload is 20 ms of µ-law 8 kHz mono audio.
   We decode → float32 16 kHz, accumulate, run VAD every VAD_CHUNK_MS ms,
   and when silence is detected we transcribe the buffered speech segment.
4. The transcript is published to Redis so the API gateway can pick it up
   and invoke the LLM + TTS pipeline.

VAD flow
--------
1. Incoming audio is sliced into VAD_CHUNK_MS-ms windows.
2. Silero assigns a speech-probability score to each window.
3. Once the probability drops below VAD_THRESHOLD for VAD_SILENCE_THRESHOLD_S
   seconds we declare end-of-speech and flush the accumulated speech buffer to Whisper.
4. Segments shorter than VAD_MIN_SPEECH_DURATION_S are discarded (breath noise, clicks).
"""

from __future__ import annotations

import asyncio
import audioop  # stdlib µ-law decoder (Python ≤ 3.12; for 3.13 use audioop-lts)
import base64
import io
import json
import os
import time
from typing import Optional

import numpy as np
import redis.asyncio as aioredis
import structlog
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from faster_whisper import WhisperModel
from pydantic import BaseModel

log = structlog.get_logger()

# ── Config ────────────────────────────────────────────────────────────────────
WHISPER_MODEL_SIZE      = os.getenv("WHISPER_MODEL", "base.en")
DEVICE                  = os.getenv("DEVICE", "cpu")
COMPUTE_TYPE            = os.getenv("COMPUTE_TYPE", "int8")

VAD_SILENCE_THRESHOLD_S = float(os.getenv("VAD_SILENCE_THRESHOLD_S", "0.8"))
VAD_MIN_SPEECH_S        = float(os.getenv("VAD_MIN_SPEECH_DURATION_S", "0.3"))
VAD_THRESHOLD           = float(os.getenv("VAD_THRESHOLD", "0.5"))
VAD_CHUNK_MS            = int(os.getenv("VAD_CHUNK_MS", "32"))

REDIS_URL               = os.getenv("REDIS_URL", "redis://redis:6379/0")

# Whisper expects 16 kHz mono; Twilio sends µ-law 8 kHz mono
SAMPLE_RATE             = 16_000
TWILIO_SAMPLE_RATE      = 8_000
VAD_CHUNK_SAMPLES       = int(SAMPLE_RATE * VAD_CHUNK_MS / 1000)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="STT Service (Whisper + Silero VAD + Twilio Stream)", version="3.0.0")

_whisper: Optional[WhisperModel] = None
_vad_model = None
_vad_utils = None
_redis: Optional[aioredis.Redis] = None


# ── Model loading (cached) ────────────────────────────────────────────────────

def get_whisper() -> WhisperModel:
    global _whisper
    if _whisper is None:
        log.info("loading_whisper", model=WHISPER_MODEL_SIZE, device=DEVICE)
        _whisper = WhisperModel(WHISPER_MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
        log.info("whisper_ready")
    return _whisper


def get_vad():
    """Load Silero VAD model (cached after first call)."""
    global _vad_model, _vad_utils
    if _vad_model is None:
        log.info("loading_silero_vad")
        _vad_model, _vad_utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=False,
        )
        _vad_model.eval()
        log.info("silero_vad_ready")
    return _vad_model, _vad_utils


async def get_redis() -> aioredis.Redis:
    global _redis
    if _redis is None:
        _redis = await aioredis.from_url(REDIS_URL, decode_responses=False)
    return _redis


# ── Startup ───────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    get_whisper()
    get_vad()
    await get_redis()


# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "whisper_model": WHISPER_MODEL_SIZE,
        "vad": "silero",
        "vad_silence_threshold_s": VAD_SILENCE_THRESHOLD_S,
        "vad_threshold": VAD_THRESHOLD,
    }


@app.get("/vad_config")
async def vad_config():
    """Inspect current VAD tuning parameters — useful for debugging."""
    return {
        "vad_threshold": VAD_THRESHOLD,
        "vad_silence_threshold_s": VAD_SILENCE_THRESHOLD_S,
        "vad_min_speech_s": VAD_MIN_SPEECH_S,
        "vad_chunk_ms": VAD_CHUNK_MS,
        "sample_rate": SAMPLE_RATE,
    }


# ── Audio helpers ─────────────────────────────────────────────────────────────

def _ulaw_to_float32(ulaw_bytes: bytes) -> np.ndarray:
    """
    Decode Twilio µ-law (8 kHz, mono) → float32 (16 kHz, mono).
    audioop.ulaw2lin decodes to signed 16-bit PCM, then we upsample 2×.
    """
    pcm16 = audioop.ulaw2lin(ulaw_bytes, 2)          # 8 kHz int16
    pcm16_up = audioop.ratecv(                        # 16 kHz int16
        pcm16, 2, 1, TWILIO_SAMPLE_RATE, SAMPLE_RATE, None
    )[0]
    arr = np.frombuffer(pcm16_up, dtype=np.int16).astype(np.float32)
    return arr / 32768.0


def _audio_bytes_to_float32(raw: bytes) -> np.ndarray:
    """
    Accept either:
      • WAV file   — decoded via soundfile
      • Raw PCM16  — interpreted as int16 → float32
    Returns float32 mono array normalised to [-1, 1] at 16 kHz.
    """
    try:
        import soundfile as sf
        audio, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != SAMPLE_RATE:
            factor = SAMPLE_RATE / sr
            new_len = int(len(audio) * factor)
            audio = np.interp(
                np.linspace(0, len(audio) - 1, new_len),
                np.arange(len(audio)),
                audio,
            ).astype(np.float32)
        return audio
    except Exception:
        pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
        return pcm / 32768.0


def _run_vad(audio: np.ndarray) -> list[dict]:
    """
    Run Silero VAD over `audio` and return a list of speech segments:
        [{"start_s": float, "end_s": float, "audio": np.ndarray}, ...]

    Segments shorter than VAD_MIN_SPEECH_S are discarded.
    """
    model, utils = get_vad()
    get_speech_ts = utils[0]

    tensor = torch.from_numpy(audio)
    speech_timestamps = get_speech_ts(
        tensor,
        model,
        threshold=VAD_THRESHOLD,
        sampling_rate=SAMPLE_RATE,
        min_silence_duration_ms=int(VAD_SILENCE_THRESHOLD_S * 1000),
        min_speech_duration_ms=int(VAD_MIN_SPEECH_S * 1000),
    )

    segments = []
    for ts in speech_timestamps:
        seg_audio = audio[ts["start"]: ts["end"]]
        duration  = len(seg_audio) / SAMPLE_RATE
        if duration < VAD_MIN_SPEECH_S:
            continue
        segments.append({
            "start_s": round(ts["start"] / SAMPLE_RATE, 3),
            "end_s":   round(ts["end"]   / SAMPLE_RATE, 3),
            "audio":   seg_audio,
        })
    return segments


def _transcribe_audio(audio: np.ndarray, language: str = "en") -> str:
    model = get_whisper()
    segments, _ = model.transcribe(
        audio,
        language=language,
        beam_size=1,
        vad_filter=False,   # We do our own VAD
    )
    return " ".join(s.text.strip() for s in segments).strip()


# ── Pydantic response models ───────────────────────────────────────────────────

class TranscribeResponse(BaseModel):
    transcript: str
    latency_ms: int
    language: str = "en"


class VADTranscribeResponse(BaseModel):
    transcript: str
    latency_ms: int
    speech_segments: list[dict]
    speaker_done: bool
    language: str = "en"


# ── REST endpoints (unchanged from v2) ────────────────────────────────────────

@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(
    audio: UploadFile = File(...),
    language: str = "en",
):
    """Simple transcription — send a complete audio file, get text back."""
    t0 = time.perf_counter()
    raw = await audio.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty audio file")

    arr  = _audio_bytes_to_float32(raw)
    text = _transcribe_audio(arr, language=language)
    latency_ms = round((time.perf_counter() - t0) * 1000)

    log.info("transcribed", latency_ms=latency_ms, chars=len(text))
    return TranscribeResponse(transcript=text, latency_ms=latency_ms, language=language)


@app.post("/transcribe_vad", response_model=VADTranscribeResponse)
async def transcribe_vad(
    audio: UploadFile = File(...),
    language: str = "en",
):
    """VAD-aware transcription — full file, VAD segments, per-segment Whisper."""
    t0 = time.perf_counter()
    raw = await audio.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty audio file")

    arr         = _audio_bytes_to_float32(raw)
    total_dur_s = len(arr) / SAMPLE_RATE

    vad_t0   = time.perf_counter()
    segments = _run_vad(arr)
    vad_ms   = round((time.perf_counter() - vad_t0) * 1000)

    if not segments:
        latency_ms = round((time.perf_counter() - t0) * 1000)
        log.info("vad_no_speech", vad_ms=vad_ms)
        return VADTranscribeResponse(
            transcript="",
            latency_ms=latency_ms,
            speech_segments=[],
            speaker_done=True,
            language=language,
        )

    result_segments: list[dict] = []
    full_texts: list[str] = []

    for seg in segments:
        seg_text = _transcribe_audio(seg["audio"], language=language)
        duration = round(seg["end_s"] - seg["start_s"], 3)
        result_segments.append({
            "start_s":    seg["start_s"],
            "end_s":      seg["end_s"],
            "duration_s": duration,
            "text":       seg_text,
        })
        if seg_text:
            full_texts.append(seg_text)

    last_speech_end_s = segments[-1]["end_s"]
    silence_tail_s    = total_dur_s - last_speech_end_s
    speaker_done      = silence_tail_s >= VAD_SILENCE_THRESHOLD_S

    latency_ms = round((time.perf_counter() - t0) * 1000)
    transcript = " ".join(full_texts).strip()

    log.info(
        "vad_transcribed",
        latency_ms=latency_ms,
        vad_ms=vad_ms,
        n_segments=len(segments),
        speaker_done=speaker_done,
        chars=len(transcript),
    )

    return VADTranscribeResponse(
        transcript=transcript,
        latency_ms=latency_ms,
        speech_segments=result_segments,
        speaker_done=speaker_done,
        language=language,
    )


# ── Twilio Media Stream WebSocket ─────────────────────────────────────────────

class _StreamSession:
    """
    Accumulates raw float32 audio from Twilio media frames,
    runs Silero VAD in real time, and flushes complete speech
    utterances to Whisper when silence is detected.
    """

    def __init__(self, call_sid: str, stream_sid: str, language: str = "en"):
        self.call_sid   = call_sid
        self.stream_sid = stream_sid
        self.language   = language

        # Rolling audio buffer (float32 @ 16 kHz)
        self._buffer: list[np.ndarray] = []
        self._buffer_samples = 0

        # VAD state
        self._in_speech         = False
        self._speech_buf: list[np.ndarray] = []
        self._silence_samples   = 0
        self._silence_threshold = int(VAD_SILENCE_THRESHOLD_S * SAMPLE_RATE)
        self._min_speech_samples = int(VAD_MIN_SPEECH_S * SAMPLE_RATE)

        self.transcripts: list[dict] = []

    def ingest(self, ulaw_payload: str) -> Optional[str]:
        """
        Feed one Twilio media payload (base64 µ-law).
        Returns a transcript string if an utterance just completed, else None.
        """
        raw      = base64.b64decode(ulaw_payload)
        chunk    = _ulaw_to_float32(raw)         # float32 @ 16 kHz

        # Run VAD on this chunk
        vad_model, vad_utils = get_vad()
        get_speech_ts = vad_utils[0]

        tensor = torch.from_numpy(chunk)
        try:
            timestamps = get_speech_ts(
                tensor,
                vad_model,
                threshold=VAD_THRESHOLD,
                sampling_rate=SAMPLE_RATE,
                min_silence_duration_ms=100,   # coarse — we track silence ourselves
                min_speech_duration_ms=50,
            )
            has_speech = len(timestamps) > 0
        except Exception:
            has_speech = False

        if has_speech:
            self._in_speech = True
            self._silence_samples = 0
            self._speech_buf.append(chunk)
        else:
            if self._in_speech:
                self._speech_buf.append(chunk)   # include trailing silence
                self._silence_samples += len(chunk)

                if self._silence_samples >= self._silence_threshold:
                    # End of utterance — transcribe
                    return self._flush()
        return None

    def _flush(self) -> Optional[str]:
        """Transcribe buffered speech and reset state."""
        if not self._speech_buf:
            return None

        audio = np.concatenate(self._speech_buf)

        # Discard very short segments (noise / breath)
        if len(audio) < self._min_speech_samples:
            self._reset_speech()
            return None

        t0         = time.perf_counter()
        transcript = _transcribe_audio(audio, language=self.language)
        latency_ms = round((time.perf_counter() - t0) * 1000)

        if transcript:
            record = {
                "transcript":  transcript,
                "latency_ms":  latency_ms,
                "call_sid":    self.call_sid,
                "stream_sid":  self.stream_sid,
            }
            self.transcripts.append(record)
            log.info("stream_transcribed", **record)

        self._reset_speech()
        return transcript or None

    def _reset_speech(self):
        self._in_speech       = False
        self._speech_buf      = []
        self._silence_samples = 0

    def flush_final(self) -> Optional[str]:
        """Call when the stream ends — flush whatever is left in the buffer."""
        if self._in_speech and self._speech_buf:
            return self._flush()
        return None


@app.websocket("/stream/{call_sid}")
async def twilio_stream(websocket: WebSocket, call_sid: str, language: str = "en"):
    """
    Twilio Media Stream WebSocket endpoint.

    Twilio connects here after your TwiML response includes:
        <Connect>
          <Stream url="wss://your-host/stream/{CallSid}" />
        </Connect>

    On each completed utterance, the transcript is:
      1. Sent back over the WebSocket as JSON (for the API gateway to consume).
      2. Published to Redis channel  twilio:transcript:{call_sid}
         so the API gateway can trigger the LLM → TTS pipeline.
    """
    await websocket.accept()
    log.info("twilio_stream_connected", call_sid=call_sid)

    redis    = await get_redis()
    session  = None
    channel  = f"twilio:transcript:{call_sid}"

    try:
        async for raw_msg in websocket.iter_text():
            msg = json.loads(raw_msg)
            event = msg.get("event")

            if event == "connected":
                log.info("twilio_stream_event", event="connected", call_sid=call_sid)

            elif event == "start":
                stream_sid = msg["start"]["streamSid"]
                session    = _StreamSession(call_sid, stream_sid, language=language)
                log.info("twilio_stream_event", event="start",
                         stream_sid=stream_sid, call_sid=call_sid)

            elif event == "media":
                if session is None:
                    continue

                payload    = msg["media"]["payload"]
                # Run VAD + possible transcription (CPU-bound — offload to thread)
                transcript = await asyncio.get_event_loop().run_in_executor(
                    None, session.ingest, payload
                )

                if transcript:
                    result = {
                        "event":      "transcript",
                        "call_sid":   call_sid,
                        "stream_sid": session.stream_sid,
                        "transcript": transcript,
                        "speaker_done": True,
                    }
                    # Notify the API gateway via WebSocket message
                    await websocket.send_json(result)
                    # Also pub to Redis for any other consumers
                    await redis.publish(channel, json.dumps(result))
                    log.info("transcript_published", call_sid=call_sid, text=transcript)

            elif event == "stop":
                log.info("twilio_stream_event", event="stop", call_sid=call_sid)
                if session:
                    final = await asyncio.get_event_loop().run_in_executor(
                        None, session.flush_final
                    )
                    if final:
                        result = {
                            "event":        "transcript",
                            "call_sid":     call_sid,
                            "stream_sid":   session.stream_sid,
                            "transcript":   final,
                            "speaker_done": True,
                        }
                        await websocket.send_json(result)
                        await redis.publish(channel, json.dumps(result))
                break

    except WebSocketDisconnect:
        log.info("twilio_stream_disconnected", call_sid=call_sid)
    except Exception as exc:
        log.error("twilio_stream_error", call_sid=call_sid, error=str(exc))
    finally:
        if session:
            log.info(
                "twilio_stream_summary",
                call_sid=call_sid,
                n_transcripts=len(session.transcripts),
            )
