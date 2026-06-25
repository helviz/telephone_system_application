import os
import sys
import time

"""
Preload / validate runtime dependencies before the server starts.

    STT  — Faster-Whisper, shared by all languages
    TTS  — Soniox API using the configured voice; no local TTS weights loaded
    LLM  — GGUF via llama.cpp singleton, when local provider is enabled

Soniox TTS is API-backed, so this file intentionally does NOT import/load
Kokoro, OmniVoice, torch TTS models, or voice-design seeds.
"""

print("\n" + "=" * 60)
print("🔥 PRELOADING CORE MODELS — server will start after this")
print("=" * 60 + "\n")

total_start = time.time()


# ---------------------------------------------------------------------------
# STT — Faster-Whisper
# ---------------------------------------------------------------------------
print("📦 [1/3] Loading Faster-Whisper ...")
t = time.time()
try:
    from faster_whisper import WhisperModel

    whisper_device = os.getenv("WHISPER_DEVICE", "cpu").strip()
    whisper_model_size = os.getenv("WHISPER_MODEL_SIZE", "medium").strip()
    compute_type = "float16" if whisper_device == "cuda" else "int8"

    _whisper = WhisperModel(
        whisper_model_size,
        device=whisper_device,
        compute_type=compute_type,
        download_root=os.getenv("HF_HOME"),
    )
    print(f"   ✅ Whisper [{whisper_model_size}] ready on [{whisper_device}] — {time.time() - t:.1f}s\n")
except Exception as e:
    print(f"   ❌ Whisper failed to load: {e}")
    sys.exit(1)


# ---------------------------------------------------------------------------
# TTS — Soniox API configuration only
# ---------------------------------------------------------------------------
print("📦 [2/3] Configuring TTS: Soniox API ...")
t = time.time()
try:
    from soniox import SonioxClient  # noqa: F401

    if not os.getenv("SONIOX_API_KEY"):
        raise RuntimeError("SONIOX_API_KEY is not set in environment/secrets.")

    _tts_store = {
        "en": {
            "engine": "soniox",
            "model": os.getenv("SONIOX_TTS_MODEL", "tts-rt-v1"),
            "voice": os.getenv("SONIOX_TTS_VOICE", "Grace"),
            "language": os.getenv("SONIOX_TTS_LANG_EN", "en"),
            "audio_format": os.getenv("SONIOX_TTS_AUDIO_FORMAT", "wav"),
            "sample_rate": int(os.getenv("SONIOX_TTS_SAMPLE_RATE", "24000")),
        },
        "fr": {
            "engine": "soniox",
            "model": os.getenv("SONIOX_TTS_MODEL", "tts-rt-v1"),
            "voice": os.getenv("SONIOX_TTS_VOICE", "Grace"),
            "language": os.getenv("SONIOX_TTS_LANG_FR", "fr"),
            "audio_format": os.getenv("SONIOX_TTS_AUDIO_FORMAT", "wav"),
            "sample_rate": int(os.getenv("SONIOX_TTS_SAMPLE_RATE", "24000")),
        },
        "sw": {
            "engine": "soniox",
            "model": os.getenv("SONIOX_TTS_MODEL", "tts-rt-v1"),
            "voice": os.getenv("SONIOX_TTS_VOICE", "Grace"),
            "language": os.getenv("SONIOX_TTS_LANG_SW", "sw"),
            "audio_format": os.getenv("SONIOX_TTS_AUDIO_FORMAT", "wav"),
            "sample_rate": int(os.getenv("SONIOX_TTS_SAMPLE_RATE", "24000")),
        },
    }
    print(
        "   ✅ Soniox TTS configured "
        f"voice=[{_tts_store['en']['voice']}] model=[{_tts_store['en']['model']}] — {time.time() - t:.1f}s\n"
    )
except Exception as e:
    print(f"   ❌ Soniox TTS configuration failed: {e}")
    print("   Add `soniox` to requirements.txt and set SONIOX_API_KEY in secrets/env vars.")
    sys.exit(1)


# ---------------------------------------------------------------------------
# LLM — GGUF singleton
# ---------------------------------------------------------------------------
llm_provider = os.getenv("LLM_PROVIDER", "gguf").strip().lower()

if llm_provider in ("qwen", "gguf", "local"):
    print("📦 [3/3] Preloading Local GGUF LLM ...")
    t = time.time()
    try:
        from llmModule.LLM import LLM

        LLM.get_model(provider="gguf", lang="en")
        print(f"   ✅ GGUF Model successfully loaded and cached in RAM — {time.time() - t:.1f}s\n")
    except Exception as e:
        print(f"   ❌ GGUF LLM failed to preload: {e}")
        sys.exit(1)
else:
    print("📦 [3/3] Using cloud LLM provider. Skipping local GGUF preload.\n")

print(f"🎉 Preload pipeline successful. Total setup execution window: {time.time() - total_start:.1f}s\n")
