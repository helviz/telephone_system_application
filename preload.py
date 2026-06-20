import os
import sys
import time

"""
Loads all model families into memory so the first caller never waits:

    STT  — Faster-Whisper, shared by all languages
    TTS  — Kokoro for English/French using af_heart; Facebook MMS for Swahili
    LLM  — GGUF via llama.cpp singleton, when local provider is enabled
"""

print("\n" + "=" * 60)
print("🔥 PRELOADING ALL MODELS — server will start after this")
print("=" * 60 + "\n")

total_start = time.time()


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


# TTS — Kokoro for en/fr, MMS for sw
print("📦 [2/3] Loading TTS: Kokoro(en/fr: af_heart) + MMS(sw) ...")
t = time.time()
try:
    import torch
    from transformers import VitsModel, AutoTokenizer
    from kokoro import KPipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"

    _tts_store = {}

    # Kokoro: English and French use the same requested af_heart voice.
    # af_heart is an American-English voice, so lang_code='a' is used for both.
    # For native French pronunciation later, switch fr to lang_code='f' and a French voice.
    for lang in ("en", "fr"):
        lt = time.time()
        print(f"   Loading {lang} → Kokoro voice af_heart ...")
        pipeline = KPipeline(lang_code="a")
        _tts_store[lang] = {
            "engine": "kokoro",
            "pipeline": pipeline,
            "voice": "af_heart",
            "sample_rate": 24000,
        }
        print(f"   ✅ {lang} Kokoro ready — {time.time() - lt:.1f}s")

    # Swahili remains Facebook MMS exactly as before.
    lt = time.time()
    sw_model_name = "facebook/mms-tts-swh"
    print(f"   Loading sw → {sw_model_name} ...")
    sw_tokenizer = AutoTokenizer.from_pretrained(sw_model_name)
    sw_model = VitsModel.from_pretrained(sw_model_name).to(device)
    sw_model.eval()
    _tts_store["sw"] = {
        "engine": "mms",
        "model": sw_model,
        "tokenizer": sw_tokenizer,
    }
    print(f"   ✅ sw MMS ready — {time.time() - lt:.1f}s")

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
