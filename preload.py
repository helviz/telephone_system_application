import os
import sys
import time

"""
Loads all three model families into memory so the first caller never waits:

    STT  — Faster-Whisper medium (all three languages share one model)
    TTS  — Facebook MMS-TTS  (en / fr / sw — three separate VITS models)
    LLM  — GGUF via llama.cpp singleton (reads GGUF_MODEL env var)

"""

print("\n" + "=" * 60)
print("🔥 PRELOADING ALL MODELS — server will start after this")
print("=" * 60 + "\n")

total_start = time.time()


#STT — Faster-Whisper
print("📦 [1/3] Loading Faster-Whisper (medium) ...")
t = time.time()
try:
    from faster_whisper import WhisperModel

    whisper_device      = os.getenv("WHISPER_DEVICE", "cpu").strip()
    whisper_model_size  = os.getenv("WHISPER_MODEL_SIZE", "medium").strip()
    compute_type        = "float16" if whisper_device == "cuda" else "int8"

    _whisper = WhisperModel(
        whisper_model_size,
        device=whisper_device,
        compute_type=compute_type,
        download_root=os.getenv("HF_HOME"),
    )
    print(f"   ✅ Whisper [{whisper_model_size}] ready on [{whisper_device}] — {time.time()-t:.1f}s\n")
except Exception as e:
    print(f"   ❌ Whisper failed to load: {e}")
    sys.exit(1)


# TTS — Facebook MMS-TTS (en / fr / sw)
print("📦 [2/3] Loading MMS-TTS models (en / fr / sw) ...")
t = time.time()
try:
    import torch
    from transformers import VitsModel, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tts_models = {
        "en": "facebook/mms-tts-eng",
        "fr": "facebook/mms-tts-fra",
        "sw": "facebook/mms-tts-swh",
    }

    _tts_store = {}
    for lang, model_name in tts_models.items():
        lt = time.time()
        print(f"   Loading {lang} → {model_name} ...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model     = VitsModel.from_pretrained(model_name).to(device)
        model.eval()
        _tts_store[lang] = (model, tokenizer)
        print(f"   ✅ {lang} ready — {time.time()-lt:.1f}s")

    print(f"   ✅ All TTS models loaded on [{device}] — {time.time()-t:.1f}s\n")
except Exception as e:
    print(f"   ❌ TTS failed to load: {e}")
    sys.exit(1)

#LLM — GGUF singleton (only if LLM_PROVIDER=qwen/gguf)
llm_provider = os.getenv("LLM_PROVIDER", "gemini").strip().lower()

if llm_provider in ("qwen", "gguf", "local"):
    print("📦 [3/3] Loading GGUF LLM ...")
    t = time.time()
    try:
        from llmModule.GGUFStrategy import _get_llm
        _get_llm()   # populates the module-level singleton
        print(f"   ✅ GGUF LLM ready — {time.time()-t:.1f}s\n")
    except Exception as e:
        print(f"   ❌ GGUF LLM failed to load: {e}")
        sys.exit(1)
else:
    print(f"📦 [3/3] LLM provider is [{llm_provider}] — skipping GGUF preload.\n")


print("=" * 60)
print(f"✅ ALL MODELS READY — total startup time: {time.time()-total_start:.1f}s")
print("🚀 Handing off to uvicorn...\n")
print("=" * 60 + "\n")