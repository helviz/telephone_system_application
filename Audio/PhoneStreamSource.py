import asyncio
import collections
import time
import numpy as np
import webrtcvad
from faster_whisper import WhisperModel
from Audio.AudioSource import AudioSource
from dotenv import load_dotenv
import os

load_dotenv()

# --- VAD framing constants -------------------------------------------------
# webrtcvad requires 16kHz mono PCM16 frames of exactly 10/20/30ms.
# Our pipeline already normalizes everything to 16kHz upstream
# (PhoneStreamSource upsamples 8k->16k, MicrophoneSource records at 16k),
# so we frame at 20ms here regardless of source.
SAMPLE_RATE = 16000
FRAME_MS = 20
FRAME_BYTES = int(SAMPLE_RATE * FRAME_MS / 1000) * 2  # *2 for int16 bytes

# How aggressive webrtcvad is about classifying a frame as speech.
# 0 = least aggressive (more false positives on noise),
# 3 = most aggressive (only very confident speech passes).
VAD_AGGRESSIVENESS = int(os.getenv("WEBRTC_VAD_AGGRESSIVENESS", "1"))

# Trailing silence required before we consider the utterance "finished"
# and flush it to Whisper. This is the real end-of-utterance signal —
# it replaces the old fixed CHUNK_BUFFER_SIZE cutoff.
END_SILENCE_MS = int(os.getenv("STT_END_SILENCE_MS", "900"))
END_SILENCE_FRAMES = max(1, END_SILENCE_MS // FRAME_MS)

# How much silence to keep *before* detected speech onset and *after*
# detected speech offset, so we don't clip the first/last phoneme.
PADDING_MS = int(os.getenv("STT_PADDING_MS", "500"))
PADDING_FRAMES = max(1, PADDING_MS // FRAME_MS)

# Safety valve: if someone talks for a very long time without a pause,
# force a flush so latency doesn't grow unbounded and memory doesn't balloon.
MAX_UTTERANCE_MS = 20_000
MAX_UTTERANCE_FRAMES = MAX_UTTERANCE_MS // FRAME_MS

MIN_SAMPLES = int(os.getenv("STT_MIN_SAMPLES", "8000"))

# Clamp externally supplied VAD aggressiveness to the valid webrtcvad range.
VAD_AGGRESSIVENESS = max(0, min(3, VAD_AGGRESSIVENESS))


class STTModule:
    def __init__(self, model_size=None, device=None, lang="en", preloaded_model=None):
        allowed_languages = ["en", "fr", "sw"]
        if lang not in allowed_languages:
            raise ValueError(f"Language '{lang}' not supported. Choose from {allowed_languages}")

        self.lang = lang

        if preloaded_model is not None:
            self.model = preloaded_model
            print(f"[STT] ♻️  Reusing preloaded Whisper model for [{lang}]")
        else:
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
        try:
            # We already segment the stream with webrtcvad before calling Whisper.
            # Running Whisper's internal VAD again can clip low-volume or narrowband
            # phone speech, especially Swahili over 8 kHz PCMU. Keep it disabled by
            # default, but allow enabling through env when testing.
            use_whisper_vad = os.getenv("WHISPER_INTERNAL_VAD", "false").lower() == "true"

            kwargs = dict(
                language=self.lang,
                beam_size=int(os.getenv("WHISPER_BEAM_SIZE", "5")),
                condition_on_previous_text=False,
                vad_filter=use_whisper_vad,
            )

            if use_whisper_vad:
                kwargs["vad_parameters"] = dict(
                    min_silence_duration_ms=int(os.getenv("WHISPER_VAD_MIN_SILENCE_MS", "900")),
                    threshold=float(os.getenv("WHISPER_VAD_THRESHOLD", "0.35")),
                    min_speech_duration_ms=int(os.getenv("WHISPER_VAD_MIN_SPEECH_MS", "250")),
                )

            segments, _ = self.model.transcribe(audio_data, **kwargs)
            return [seg.text.strip() for seg in segments if seg.text.strip()]
        except Exception as e:
            print(f"[STT] ⚠️  Transcription error (chunk discarded): {e}")
            return []

    @staticmethod
    def _frame_generator(byte_stream_buffer: bytearray):
        """
        Slices a growing byte buffer into fixed-size 20ms frames as required
        by webrtcvad. Consumes fully-formed frames from the front of the
        buffer in place and yields each as raw bytes; leaves any trailing
        partial frame in the buffer for the next call.
        """
        frames = []
        offset = 0
        while offset + FRAME_BYTES <= len(byte_stream_buffer):
            frames.append(bytes(byte_stream_buffer[offset:offset + FRAME_BYTES]))
            offset += FRAME_BYTES
        # Drop consumed bytes, keep the leftover partial frame
        del byte_stream_buffer[:offset]
        return frames

    def _bytes_to_float_audio(self, audio_bytes: bytes) -> np.ndarray:
        if len(audio_bytes) % 2 != 0:
            audio_bytes = audio_bytes[:-1]
        return (
            np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
            / 32768.0
        )

    async def transcribe_stream(self, audio_source: AudioSource):
        """
        FIX: The previous implementation cut the incoming audio into fixed
        150-chunk windows (~3s) with no overlap and handed each window to
        Whisper's internal VAD blind. Any utterance that straddled a window
        boundary got truncated or, if the speech fragment on either side of
        the cut was shorter than min_speech_duration_ms, silently dropped
        entirely. That's why callers were heard saying only the first few
        words of a sentence, or only the last few — the cut point had
        nothing to do with where they actually paused.

        This version runs webrtcvad frame-by-frame (20ms frames, required
        by webrtcvad) to find real speech boundaries. Audio is accumulated
        into an utterance buffer for as long as speech (or a short pause
        inside speech) continues, and the utterance is only flushed to
        Whisper once a sustained trailing silence (END_SILENCE_MS) confirms
        the person has actually stopped talking — plus a safety cap
        (MAX_UTTERANCE_MS) so an unbroken monologue still gets flushed
        periodically instead of buffering forever.
        """
        print(f"--- STT Active: Listening [{self.lang}] (VAD-segmented) ---")
        loop = asyncio.get_running_loop()
        vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)

        byte_buffer = bytearray()          # raw bytes not yet sliced into frames
        ring_pad = collections.deque(maxlen=PADDING_FRAMES)  # pre-speech padding ring
        utterance_frames = []              # frames belonging to current utterance
        in_speech = False
        trailing_silence = 0               # consecutive non-speech frames since last speech frame

        async def _flush(frames):
            if not frames:
                return
            audio_bytes = b"".join(frames)
            audio_data = self._bytes_to_float_audio(audio_bytes)

            if len(audio_data) < MIN_SAMPLES:
                return

            t0 = time.time()
            texts = await loop.run_in_executor(
                None, self._transcribe_blocking, audio_data
            )
            stt_elapsed = time.time() - t0

            if texts:
                try:
                    import stats
                    stats.record_stt_latency(stt_elapsed)
                except Exception:
                    pass

            for text in texts:
                yield text

        try:
            async for chunk in audio_source.get_stream():
                if not chunk or not isinstance(chunk, (bytes, bytearray)):
                    continue

                byte_buffer.extend(chunk)
                frames = self._frame_generator(byte_buffer)

                for frame in frames:
                    try:
                        is_speech = vad.is_speech(frame, SAMPLE_RATE)
                    except Exception:
                        # Malformed frame (e.g. wrong length) — treat as silence
                        # rather than crashing the stream.
                        is_speech = False

                    if not in_speech:
                        # Keep a short rolling pre-buffer so that when speech
                        # *does* start, we don't lose the syllable right
                        # before webrtcvad first flags it as speech.
                        ring_pad.append(frame)
                        if is_speech:
                            in_speech = True
                            trailing_silence = 0
                            utterance_frames = list(ring_pad)
                            utterance_frames.append(frame)
                    else:
                        utterance_frames.append(frame)

                        if is_speech:
                            trailing_silence = 0
                        else:
                            trailing_silence += 1

                        flushed = False
                        if trailing_silence >= END_SILENCE_FRAMES:
                            # Real pause long enough to count as end-of-utterance.
                            async for text in _flush(utterance_frames):
                                yield text
                            flushed = True
                        elif len(utterance_frames) >= MAX_UTTERANCE_FRAMES:
                            # Safety valve for very long unbroken speech.
                            async for text in _flush(utterance_frames):
                                yield text
                            flushed = True

                        if flushed:
                            utterance_frames = []
                            in_speech = False
                            trailing_silence = 0
                            ring_pad.clear()

        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"[STT] ❌ Stream error: {e}")
        finally:
            # Flush whatever speech was in progress when the stream ended
            # (e.g. caller hung up mid-sentence) instead of discarding it.
            if utterance_frames:
                async for text in _flush(utterance_frames):
                    yield text