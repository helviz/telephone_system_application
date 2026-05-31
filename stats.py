import time
import os
import psutil


# ---------------------------------------------------------------------------
# Call tracking
# ---------------------------------------------------------------------------

# { session_id: {"provider": str, "lang": str, "started_at": float} }
active_calls: dict[str, dict] = {}

total_calls: int = 0
failed_calls: int = 0
peak_concurrent: int = 0

# Total seconds across all completed calls (for average duration)
_total_duration_seconds: float = 0.0
_completed_calls: int = 0

# Calls per language since start
calls_by_lang: dict[str, int] = {"en": 0, "fr": 0, "sw": 0}

# Calls per provider since start
calls_by_provider: dict[str, int] = {"twilio": 0, "telnyx": 0}


# ---------------------------------------------------------------------------
# Pipeline latency (rolling average over last 20 measurements)
# ---------------------------------------------------------------------------

_WINDOW = 20

_stt_latencies:  list[float] = []
_llm_latencies:  list[float] = []
_tts_latencies:  list[float] = []
_e2e_latencies:  list[float] = []


def _rolling_avg(samples: list[float]) -> float | None:
    return round(sum(samples) / len(samples), 3) if samples else None


# ---------------------------------------------------------------------------
# Model info — written once by sockets.py lifespan after preload completes
# ---------------------------------------------------------------------------

model_info: dict = {
    "whisper_size":       os.getenv("WHISPER_MODEL_SIZE", "medium"),
    "whisper_device":     os.getenv("WHISPER_DEVICE", "cpu"),
    "tts_languages":      [],          # populated after preload
    "llm_provider":       os.getenv("LLM_PROVIDER", "gemini"),
    "llm_model":          "gemma-4-26b-a4b-it",
    "preload_ok":         False,       # set True by lifespan on success
    "preload_duration_s": None,
}

# Concurrency helpers — written by sockets.py when semaphore/locks land
whisper_queue_depth: int = 0
tts_lock_contention: dict[str, int] = {"en": 0, "fr": 0, "sw": 0}


# ---------------------------------------------------------------------------
# Public write API — called by sockets.py
# ---------------------------------------------------------------------------

def call_started(session_id: str, provider: str, lang: str):
    global total_calls, peak_concurrent

    active_calls[session_id] = {
        "provider":   provider,
        "lang":       lang,
        "started_at": time.time(),
    }

    total_calls += 1
    calls_by_lang[lang]         = calls_by_lang.get(lang, 0) + 1
    calls_by_provider[provider] = calls_by_provider.get(provider, 0) + 1

    if len(active_calls) > peak_concurrent:
        peak_concurrent = len(active_calls)


def call_ended(session_id: str, failed: bool = False):
    global failed_calls, _total_duration_seconds, _completed_calls

    call = active_calls.pop(session_id, None)
    if call:
        duration = time.time() - call["started_at"]
        _total_duration_seconds += duration
        _completed_calls += 1

    if failed:
        failed_calls += 1


def record_stt_latency(seconds: float):
    _stt_latencies.append(seconds)
    if len(_stt_latencies) > _WINDOW:
        _stt_latencies.pop(0)


def record_llm_latency(seconds: float):
    _llm_latencies.append(seconds)
    if len(_llm_latencies) > _WINDOW:
        _llm_latencies.pop(0)


def record_tts_latency(seconds: float):
    _tts_latencies.append(seconds)
    if len(_tts_latencies) > _WINDOW:
        _tts_latencies.pop(0)


def record_e2e_latency(seconds: float):
    _e2e_latencies.append(seconds)
    if len(_e2e_latencies) > _WINDOW:
        _e2e_latencies.pop(0)


# ---------------------------------------------------------------------------
# Public read API — called by dashboard routes
# ---------------------------------------------------------------------------

def avg_call_duration() -> float | None:
    if _completed_calls == 0:
        return None
    return round(_total_duration_seconds / _completed_calls, 1)


def get_latencies() -> dict:
    return {
        "stt_avg_s":  _rolling_avg(_stt_latencies),
        "llm_avg_s":  _rolling_avg(_llm_latencies),
        "tts_avg_s":  _rolling_avg(_tts_latencies),
        "e2e_avg_s":  _rolling_avg(_e2e_latencies),
    }


def get_system_resources() -> dict:
    mem   = psutil.virtual_memory()
    cpu   = psutil.cpu_percent(interval=None)

    result = {
        "ram_used_gb":  round(mem.used  / 1024**3, 2),
        "ram_total_gb": round(mem.total / 1024**3, 2),
        "ram_pct":      mem.percent,
        "cpu_pct":      cpu,
        "gpu_used_mb":  None,
        "gpu_total_mb": None,
        "gpu_pct":      None,
    }

    try:
        import torch
        if torch.cuda.is_available():
            used  = torch.cuda.memory_allocated(0)
            total = torch.cuda.get_device_properties(0).total_memory
            result["gpu_used_mb"]  = round(used  / 1024**2, 1)
            result["gpu_total_mb"] = round(total / 1024**2, 1)
            result["gpu_pct"]      = round(used / total * 100, 1)
    except Exception:
        pass

    return result


def snapshot() -> dict:
    """Full stats dict — used by /metrics JSON endpoint."""
    return {
        "calls": {
            "active":          list(active_calls.values()),
            "active_count":    len(active_calls),
            "total":           total_calls,
            "failed":          failed_calls,
            "peak_concurrent": peak_concurrent,
            "avg_duration_s":  avg_call_duration(),
            "by_lang":         dict(calls_by_lang),
            "by_provider":     dict(calls_by_provider),
        },
        "latency":      get_latencies(),
        "models":       dict(model_info),
        "resources":    get_system_resources(),
        "concurrency": {
            "active_count":       len(active_calls),
            "peak_concurrent":    peak_concurrent,
            "whisper_queue":      whisper_queue_depth,
            "tts_contention":     dict(tts_lock_contention),
        },
    }