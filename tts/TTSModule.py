import asyncio
import os
import random
import threading
import time
from typing import Any

import numpy as np
import torch

MAX_BUFFER_CHARS = 200
KOKORO_SAMPLE_RATE = 24000
OMNIVOICE_SAMPLE_RATE = 24000

KOKORO_VOICE_MAP = {
    "en": "af_heart",
    "fr": "ff_siwis",
}

# English uses Kokoro American English voice/pipeline. French uses the native
# Kokoro French female voice and French pipeline.
KOKORO_LANG_CODE_MAP = {
    "en": "a",
    "fr": "f",
}

OMNIVOICE_MODEL_ID = "k2-fsa/OmniVoice"
OMNIVOICE_INSTRUCT = "female, middle-aged, moderate pitch"
OMNIVOICE_NUM_STEP = 16
OMNIVOICE_SPEED = 1.0
OMNIVOICE_LANGUAGE_ID = "sw"

# Keep OmniVoice voice-design output stable across calls/restarts.
# You can override this in Hugging Face Space secrets/env vars.
OMNIVOICE_SEED = int(os.getenv("OMNIVOICE_SEED", "12345"))


def seed_omnivoice(seed: int = OMNIVOICE_SEED) -> int:
    """Seed all RNGs OmniVoice may use during sampling/generation."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # These flags reduce CUDA/CuDNN nondeterminism. They do not make every
    # possible GPU kernel bit-identical, but they remove the common source of
    # voice-design drift between generations.
    try:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    except Exception:
        pass

    return seed


def _bundle_omnivoice_seed(bundle: dict[str, Any]) -> int:
    return int(bundle.get("seed", OMNIVOICE_SEED))


def _ensure_omnivoice_lock(bundle: dict[str, Any]) -> dict[str, Any]:
    # A shared preloaded OmniVoice model uses global torch RNG state. The lock
    # prevents two calls from reseeding/generating at the same time and changing
    # each other's voice output.
    if bundle.get("engine") == "omnivoice" and "lock" not in bundle:
        bundle["lock"] = threading.Lock()
    return bundle


class TTSModule:
    """
    Streaming TTS module for phone calls.

    Language routing:
      - en: Kokoro TTS with voice af_heart
      - fr: Kokoro TTS with voice ff_siwis
      - sw: OmniVoice voice-design mode using k2-fsa/OmniVoice

    The public interface is unchanged:
        await speak_stream(text_generator, lang="en", on_first_audio=callback)
    """

    def __init__(self, output, preloaded_models: dict | None = None):
        self.output = output
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.engine_map = {
            "en": "kokoro",
            "fr": "kokoro",
            "sw": "omnivoice",
        }

        self.models: dict[str, dict[str, Any]] = {}
        if preloaded_models:
            for lang, bundle in preloaded_models.items():
                self.models[lang] = self._normalize_preloaded_bundle(lang, bundle)

    def _normalize_preloaded_bundle(self, lang: str, bundle: Any) -> dict[str, Any]:
        """
        Accept the new dictionary preload format plus older tuple MMS bundles.
        The MMS tuple fallback is kept only so old cached/injected state does not
        crash the app during a rolling deployment.
        """
        if isinstance(bundle, dict):
            return _ensure_omnivoice_lock(bundle)

        if isinstance(bundle, tuple) and len(bundle) == 2:
            model, tokenizer = bundle
            return {
                "engine": "mms",
                "model": model,
                "tokenizer": tokenizer,
            }

        raise ValueError(f"Unsupported preloaded TTS bundle for lang={lang}: {type(bundle)}")

    def _ensure_model_loaded(self, lang: str) -> dict[str, Any]:
        """Lazily load only the TTS engine needed for the active language."""
        if lang not in self.engine_map:
            raise ValueError(f"Unsupported TTS language: {lang}. Choose from {list(self.engine_map)}")

        if lang in self.models:
            return self.models[lang]

        engine = self.engine_map[lang]
        print(f"\n[TTS] 🧠 Lazy loading [{engine.upper()}] for [{lang.upper()}]...")
        t_load = time.time()

        if engine == "kokoro":
            self.models[lang] = self._load_kokoro(lang)
        elif engine == "omnivoice":
            self.models[lang] = self._load_omnivoice()
        else:
            raise ValueError(f"Unsupported TTS engine for lang={lang}: {engine}")

        print(f"[TTS] [{lang.upper()}] ready in {time.time() - t_load:.2f}s.\n")
        return self.models[lang]

    def _load_kokoro(self, lang: str) -> dict[str, Any]:
        try:
            from kokoro import KPipeline
        except Exception as exc:
            raise RuntimeError(
                "Kokoro is not installed. Add `kokoro==0.9.4` to requirements.txt "
                "or install it with `pip install kokoro==0.9.4`."
            ) from exc

        return {
            "engine": "kokoro",
            "pipeline": KPipeline(lang_code=KOKORO_LANG_CODE_MAP[lang]),
            "voice": KOKORO_VOICE_MAP[lang],
            "sample_rate": KOKORO_SAMPLE_RATE,
        }

    def _load_omnivoice(self) -> dict[str, Any]:
        try:
            from omnivoice import OmniVoice
        except Exception as exc:
            raise RuntimeError(
                "OmniVoice is not installed. Add `omnivoice` to requirements.txt "
                "or install it from the official k2-fsa/OmniVoice package."
            ) from exc

        dtype = torch.float16 if self.device == "cuda" else torch.float32
        device_map = "cuda:0" if self.device == "cuda" else "cpu"

        seed_omnivoice(OMNIVOICE_SEED)
        model = OmniVoice.from_pretrained(
            OMNIVOICE_MODEL_ID,
            device_map=device_map,
            dtype=dtype,
        )

        return _ensure_omnivoice_lock({
            "engine": "omnivoice",
            "model": model,
            "sample_rate": OMNIVOICE_SAMPLE_RATE,
            "instruct": OMNIVOICE_INSTRUCT,
            "num_step": OMNIVOICE_NUM_STEP,
            "speed": OMNIVOICE_SPEED,
            "language_id": OMNIVOICE_LANGUAGE_ID,
            "seed": OMNIVOICE_SEED,
        })

    async def speak_stream(self, text_generator, lang="en", on_first_audio=None):
        bundle = self._ensure_model_loaded(lang)

        buffer = ""
        _first_audio_fired = False

        async for chunk in text_generator:
            buffer += chunk

            if any(p in chunk for p in [".", "!", "?", "\n"]) or len(buffer) >= MAX_BUFFER_CHARS:
                if len(buffer) >= MAX_BUFFER_CHARS:
                    split_idx = buffer.rfind(" ")
                    if split_idx != -1:
                        to_say = buffer[:split_idx].strip()
                        buffer = buffer[split_idx:].strip()
                    else:
                        to_say = buffer.strip()
                        buffer = ""
                else:
                    to_say = buffer.strip()
                    buffer = ""

                if to_say:
                    cb = None
                    if not _first_audio_fired and on_first_audio:
                        cb = on_first_audio
                        _first_audio_fired = True
                    await self._generate_audio(to_say, bundle, on_first_audio=cb)

        if buffer.strip():
            cb = None
            if not _first_audio_fired and on_first_audio:
                cb = on_first_audio
                _first_audio_fired = True
            await self._generate_audio(buffer.strip(), bundle, on_first_audio=cb)

    async def _generate_audio(self, text: str, bundle: dict[str, Any], on_first_audio=None):
        loop = asyncio.get_running_loop()
        t0 = time.time()

        if bundle["engine"] == "kokoro":
            waveform, sample_rate = await loop.run_in_executor(
                None,
                lambda: self._synthesize_kokoro(text, bundle),
            )
        elif bundle["engine"] == "omnivoice":
            waveform, sample_rate = await loop.run_in_executor(
                None,
                lambda: self._synthesize_omnivoice(text, bundle),
            )
        else:
            raise ValueError(f"Unsupported TTS engine: {bundle.get('engine')}")

        tts_elapsed = time.time() - t0
        try:
            import stats
            stats.record_tts_latency(tts_elapsed)
        except Exception:
            pass

        if on_first_audio:
            on_first_audio()

        await self.output.send_audio(waveform, sample_rate=sample_rate)

    def _synthesize_kokoro(self, text: str, bundle: dict[str, Any]) -> tuple[torch.Tensor, int]:
        generator = bundle["pipeline"](
            text,
            voice=bundle.get("voice", KOKORO_VOICE_MAP["en"]),
            speed=0.95,
        )

        chunks = []
        for _, _, audio in generator:
            if audio is not None:
                chunks.append(torch.as_tensor(audio, dtype=torch.float32))

        waveform = self._concat_or_silence(chunks)
        return waveform, int(bundle.get("sample_rate", KOKORO_SAMPLE_RATE))

    def _synthesize_omnivoice(self, text: str, bundle: dict[str, Any]) -> tuple[torch.Tensor, int]:
        model = bundle["model"]

        kwargs = {
            "text": text,
            "instruct": bundle.get("instruct", OMNIVOICE_INSTRUCT),
            "num_step": int(bundle.get("num_step", OMNIVOICE_NUM_STEP)),
            "speed": float(bundle.get("speed", OMNIVOICE_SPEED)),
        }

        # OmniVoice supports language_id as an optional hint. Keep this separate
        # so older package versions that do not accept it can still run.
        language_id = bundle.get("language_id")
        if language_id:
            kwargs["language_id"] = language_id

        seed = _bundle_omnivoice_seed(bundle)
        lock = bundle.get("lock")

        def _run_generate():
            seed_omnivoice(seed)
            with torch.inference_mode():
                try:
                    return model.generate(**kwargs)
                except TypeError:
                    kwargs.pop("language_id", None)
                    return model.generate(**kwargs)

        if lock is not None:
            with lock:
                audio = _run_generate()
        else:
            audio = _run_generate()

        sample_rate = int(bundle.get("sample_rate", OMNIVOICE_SAMPLE_RATE))
        waveform = self._to_mono_tensor(audio)
        waveform = self._apply_omnivoice_envelope(waveform, sample_rate)
        return waveform, sample_rate

    @staticmethod
    def _apply_omnivoice_envelope(waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """
        Eliminate the crackling/clicking artifacts that OmniVoice produces at
        segment boundaries.

        OmniVoice's diffusion decoder does not taper the waveform to zero at
        the start or end of each chunk, so abrupt joins between segments (or
        between a segment and silence) create audible clicks.  Three cheap
        fixes applied in sequence solve this:

          1. 5 ms linear fade-in  — removes click at the leading edge.
          2. 10 ms linear fade-out — removes click/pop at the trailing edge,
             where the artefact is most common.
          3. 50 ms silence pad    — gives the phone codec a clean tail instead
             of an abrupt cut, and gives the listener a natural pause between
             sentences.
        """
        n = waveform.shape[-1]

        fade_in_samples  = int(sample_rate * 0.005)   # 5 ms
        fade_out_samples = int(sample_rate * 0.010)   # 10 ms
        silence_samples  = int(sample_rate * 0.050)   # 50 ms

        if n > fade_in_samples:
            fade_in = torch.linspace(0.0, 1.0, fade_in_samples, device=waveform.device)
            waveform[..., :fade_in_samples] *= fade_in

        if n > fade_out_samples:
            fade_out = torch.linspace(1.0, 0.0, fade_out_samples, device=waveform.device)
            waveform[..., -fade_out_samples:] *= fade_out

        silence = torch.zeros(waveform.shape[0], silence_samples,
                              dtype=waveform.dtype, device=waveform.device)
        waveform = torch.cat([waveform, silence], dim=-1)

        return waveform

    @staticmethod
    def _concat_or_silence(chunks: list[torch.Tensor]) -> torch.Tensor:
        if not chunks:
            waveform = torch.zeros(1, 1, dtype=torch.float32)
        elif len(chunks) == 1:
            waveform = chunks[0]
        else:
            waveform = torch.cat(chunks, dim=-1)

        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        return waveform.to(dtype=torch.float32, device="cpu")

    @staticmethod
    def _to_mono_tensor(audio: Any) -> torch.Tensor:
        """
        Normalize OmniVoice outputs to the existing PhoneAudioOutput contract:
        a CPU float32 tensor shaped as [1, samples].
        """
        if isinstance(audio, torch.Tensor):
            waveform = audio.detach().to(dtype=torch.float32, device="cpu")
        else:
            waveform = torch.as_tensor(audio, dtype=torch.float32, device="cpu")

        # Common OmniVoice return shapes are [samples], [1, samples], or a batch
        # where the first item is the generated utterance.
        if waveform.ndim == 0:
            waveform = waveform.reshape(1, 1)
        elif waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        elif waveform.ndim > 2:
            waveform = waveform.reshape(-1, waveform.shape[-1])

        if waveform.shape[0] > 1:
            waveform = waveform[:1, :]

        return waveform.contiguous()