import os
import random
import sys
import time

import numpy as np

"""
Loads all model families into memory so the first caller never waits:

    STT  — Faster-Whisper, shared by all languages
    TTS  — Kokoro en=af_heart, fr=ff_siwis; OmniVoice for Swahili
    LLM  — GGUF via llama.cpp singleton, when local provider is enabled
"""

print("\n" + "=" * 60)
print("🔥 PRELOADING ALL MODELS — server will start after this")
print("=" * 60 + "\n")

total_start = time.time()

def seed_omnivoice(seed: int | None = None) -> int:
    """Seed RNGs before loading/generating with OmniVoice."""
    import torch

    seed = int(seed if seed is not None else os.getenv("OMNIVOICE_SEED", "12345"))
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    try:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    except Exception:
        pass

    return seed


# STT — Faster-Whisper
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


# TTS — Kokoro for en/fr, OmniVoice for sw
print("📦 [2/3] Loading TTS: Kokoro(en=af_heart, fr=ff_siwis) + OmniVoice(sw) ...")
t = time.time()
try:
    import torch
    from kokoro import KPipeline
    from omnivoice import OmniVoice

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    device_map = "cuda:0" if device == "cuda" else "cpu"
    omnivoice_seed = seed_omnivoice()
    print(f"   🎚️  OmniVoice deterministic seed: {omnivoice_seed}")

    _tts_store = {}

    kokoro_slots = {
        "en": {"voice": "af_heart", "lang_code": "a"},
        "fr": {"voice": "ff_siwis", "lang_code": "f"},
    }

    for lang, cfg in kokoro_slots.items():
        lt = time.time()
        print(f"   Loading {lang} → Kokoro voice {cfg['voice']} ...")
        _tts_store[lang] = {
            "engine": "kokoro",
            "pipeline": KPipeline(lang_code=cfg["lang_code"]),
            "voice": cfg["voice"],
            "sample_rate": 24000,
        }
        print(f"   ✅ {lang} Kokoro ready — {time.time() - lt:.1f}s")

    lt = time.time()
    sw_model_name = "k2-fsa/OmniVoice"
    print(f"   Loading sw → {sw_model_name} ...")
    seed_omnivoice(omnivoice_seed)
    sw_model = OmniVoice.from_pretrained(
        sw_model_name,
        device_map=device_map,
        dtype=dtype,
    )
    _tts_store["sw"] = {
        "engine": "omnivoice",
        "model": sw_model,
        "sample_rate": 24000,
        "instruct": "female, middle-aged, moderate pitch",
        "num_step": 16,
        "speed": 1.0,
        "language_id": "sw",
        "seed": omnivoice_seed,
    }
    print(f"   ✅ sw OmniVoice ready — {time.time() - lt:.1f}s")

    print(f"   ✅ All TTS engines loaded on [{device}] — {time.time() - t:.1f}s\n")
except Exception as e:
    print(f"   ❌ TTS failed to load: {e}")
    sys.exit(1)


# LLM — GGUF singleton
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

print(f"🎉 Preload pipeline successful. Total setup execution window: {time.time() - total_start:.1f}s\n")
