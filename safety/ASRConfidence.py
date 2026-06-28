import os
from dataclasses import dataclass, field
from typing import Any


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)).strip())
    except Exception:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)).strip())
    except Exception:
        return default


@dataclass(slots=True)
class TranscriptionResult:
    text: str
    avg_logprob: float | None = None
    no_speech_prob: float | None = None
    compression_ratio: float | None = None
    rms: float | None = None
    duration_ms: float | None = None
    engine: str = "unknown"
    low_confidence_reasons: list[str] = field(default_factory=list)
    raw_segments: list[Any] = field(default_factory=list, repr=False)

    def __str__(self) -> str:
        return self.text


class ASRConfidenceChecker:
    """Reliability gate for Whisper/faster-whisper outputs."""

    def __init__(self):
        self.avg_logprob_threshold = _env_float("ASR_LOW_CONF_AVG_LOGPROB", -0.8)
        self.no_speech_threshold = _env_float("ASR_HIGH_NO_SPEECH_PROB", 0.45)
        self.compression_threshold = _env_float("ASR_HIGH_COMPRESSION_RATIO", 2.2)
        self.min_chars = _env_int("ASR_MIN_TEXT_CHARS", 3)

    def check(self, result: TranscriptionResult | str | None) -> tuple[bool, list[str]]:
        if result is None:
            return True, ["empty_result"]
        if isinstance(result, str):
            text = result.strip()
            return (len(text) < self.min_chars), (["too_short"] if len(text) < self.min_chars else [])

        reasons: list[str] = list(result.low_confidence_reasons or [])
        text = (result.text or "").strip()
        if len(text) < self.min_chars:
            reasons.append("too_short")
        if result.avg_logprob is not None and result.avg_logprob < self.avg_logprob_threshold:
            reasons.append(f"avg_logprob<{self.avg_logprob_threshold}")
        if result.no_speech_prob is not None and result.no_speech_prob > self.no_speech_threshold:
            reasons.append(f"no_speech_prob>{self.no_speech_threshold}")
        if result.compression_ratio is not None and result.compression_ratio > self.compression_threshold:
            reasons.append(f"compression_ratio>{self.compression_threshold}")

        result.low_confidence_reasons = reasons
        return bool(reasons), reasons
