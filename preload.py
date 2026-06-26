import os
import sys
import time

"""
Preload / validate runtime dependencies before the server starts.

    STT  — Mixed backend:
            en/sw -> Sunbird/asr-whisper-large-v3-salt through transformers
            fr    -> Faster-Whisper from WHISPER_MODEL_SIZE env
    TTS  — Soniox API using the configured voice; no local TTS weights loaded
    LLM  — GGUF via llama.cpp singleton, when local provider is enabled
"""

print("\n" + "=" * 60)
print("🔥 PRELOADING CORE MODELS — server will start after this")
print("=" * 60 + "\n")

total_start = time.time()


# ---------------------------------------------------------------------------
# STT — Sunbird SALT for English/Swahili + Faster-Whisper for French
# ---------------------------------------------------------------------------
print("📦 [1/3] Loading STT: Sunbird SALT for EN/SW + Faster-Whisper for FR ...")
t = time.time()
try:
    import torch
    from faster_whisper import WhisperModel
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    stt_device = os.getenv("WHISPER_DEVICE", "cuda" if torch.cuda.is_available() else "cpu").strip()
    if stt_device == "cuda" and not torch.cuda.is_available():
        print("   ⚠️ WHISPER_DEVICE=cuda requested, but CUDA is unavailable. Falling back to CPU.")
        stt_device = "cpu"

    sunbird_model_id = os.getenv("SUNBIRD_ASR_MODEL", "Sunbird/asr-whisper-large-v3-salt").strip()
    torch_dtype = torch.float16 if stt_device == "cuda" else torch.float32
    sunbird_processor = WhisperProcessor.from_pretrained(
        sunbird_model_id,
        cache_dir=os.getenv("HF_HOME"),
    )
    sunbird_model = WhisperForConditionalGeneration.from_pretrained(
        sunbird_model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
        cache_dir=os.getenv("HF_HOME"),
    ).to(stt_device)
    sunbird_model.eval()

    french_whisper_size = os.getenv("WHISPER_MODEL_SIZE", "medium").strip()
    faster_compute_type = "float16" if stt_device == "cuda" else "int8"
    _french_whisper = WhisperModel(
        french_whisper_size,
        device=stt_device,
        compute_type=faster_compute_type,
        download_root=os.getenv("HF_HOME"),
    )

    _stt_store = {
        "en": {
            "engine": "sunbird_salt",
            "processor": sunbird_processor,
            "model": sunbird_model,
            "model_name": sunbird_model_id,
            "device": stt_device,
            "salt_lang": "eng",
            "forced_language": "eng:50259",
        },
        "sw": {
            "engine": "sunbird_salt",
            "processor": sunbird_processor,
            "model": sunbird_model,
            "model_name": sunbird_model_id,
            "device": stt_device,
            "salt_lang": "swa",
            "forced_language": "swa:50318",
        },
        "fr": {
            "engine": "faster_whisper",
            "model": _french_whisper,
            "model_name": french_whisper_size,
            "forced_language": "fr",
        },
    }

    print(
        "   ✅ STT ready: "
        f"EN/SW=[{sunbird_model_id} forced SALT tokens eng/swa], "
        f"FR=[{french_whisper_size} forced fr], "
        f"device=[{stt_device}] — {time.time() - t:.1f}s\n"
    )
except Exception as e:
    print(f"   ❌ STT failed to load: {e}")
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
