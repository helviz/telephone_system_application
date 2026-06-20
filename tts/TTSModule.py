import asyncio
import time
from typing import Any

import torch
from transformers import VitsModel, AutoTokenizer

MAX_BUFFER_CHARS = 200
KOKORO_SAMPLE_RATE = 24000
KOKORO_VOICE = "af_heart"


class TTSModule:
    """
    Streaming TTS module for phone calls.

    Language routing:
      - en/fr: Kokoro TTS with voice af_heart
      - sw: Facebook MMS TTS, unchanged

    The public interface is unchanged:
        await speak_stream(text_generator, lang="en", on_first_audio=callback)
    """

    def __init__(self, output, preloaded_models: dict | None = None):
        self.output = output
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # English and French now use Kokoro. Swahili stays on MMS.
        # NOTE: af_heart is an English/American Kokoro voice. We use lang_code="a"
        # for both en and fr because the user explicitly requested af_heart.
        self.engine_map = {
            "en": "kokoro",
            "fr": "kokoro",
            "sw": "mms",
        }
        self.kokoro_voice_map = {
            "en": KOKORO_VOICE,
            "fr": KOKORO_VOICE,
        }
        self.kokoro_lang_code_map = {
            "en": "a",  # American English
            "fr": "a",  # keep af_heart as requested; use "f" only with a French voice
        }
        self.mms_model_map = {
            "sw": "facebook/mms-tts-swh",
        }

        # Cached runtime objects. Entries are normalized dictionaries:
        #   {"engine": "kokoro", "pipeline": KPipeline(...), ...}
        #   {"engine": "mms", "model": VitsModel(...), "tokenizer": AutoTokenizer(...)}
        self.models: dict[str, dict[str, Any]] = {}

        if preloaded_models:
            for lang, bundle in preloaded_models.items():
                self.models[lang] = self._normalize_preloaded_bundle(lang, bundle)

    def _normalize_preloaded_bundle(self, lang: str, bundle: Any) -> dict[str, Any]:
        """
        Accept both the old preload format `(model, tokenizer)` and the new
        dictionary format used by preload.py after the Kokoro refactor.
        """
        if isinstance(bundle, dict):
            return bundle

        # Backwards compatibility: old code stored `(model, tokenizer)` for MMS.
        if isinstance(bundle, tuple) and len(bundle) == 2:
            model, tokenizer = bundle
            return {
                "engine": "mms",
                "model": model,
                "tokenizer": tokenizer,
            }

        raise ValueError(f"Unsupported preloaded TTS bundle for lang={lang}: {type(bundle)}")

    def _ensure_model_loaded(self, lang: str) -> dict[str, Any]:
        """
        Lazily load only the TTS engine needed for the active language.
        """
        if lang not in self.engine_map:
            raise ValueError(f"Unsupported TTS language: {lang}. Choose from {list(self.engine_map)}")

        if lang in self.models:
            return self.models[lang]

        engine = self.engine_map[lang]
        print(f"\n[TTS] 🧠 Lazy loading [{engine.upper()}] for [{lang.upper()}]...")
        t_load = time.time()

        if engine == "kokoro":
            try:
                from kokoro import KPipeline
            except Exception as exc:
                raise RuntimeError(
                    "Kokoro is not installed. Add `kokoro` to requirements.txt "
                    "or install it with `pip install kokoro`."
                ) from exc

            pipeline = KPipeline(lang_code=self.kokoro_lang_code_map[lang])
            self.models[lang] = {
                "engine": "kokoro",
                "pipeline": pipeline,
                "voice": self.kokoro_voice_map[lang],
                "sample_rate": KOKORO_SAMPLE_RATE,
            }

        else:
            repo_id = self.mms_model_map[lang]
            tokenizer = AutoTokenizer.from_pretrained(repo_id)
            model = VitsModel.from_pretrained(repo_id).to(self.device)
            model.eval()
            self.models[lang] = {
                "engine": "mms",
                "model": model,
                "tokenizer": tokenizer,
            }

        print(f"[TTS] [{lang.upper()}] ready in {time.time() - t_load:.2f}s.\n")
        return self.models[lang]

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
        else:
            waveform, sample_rate = await loop.run_in_executor(
                None,
                lambda: self._synthesize_mms(text, bundle),
            )

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
            voice=bundle.get("voice", KOKORO_VOICE),
            speed=0.95,
        )

        chunks = []
        for _, _, audio in generator:
            if audio is not None:
                chunks.append(torch.as_tensor(audio, dtype=torch.float32))

        if not chunks:
            waveform = torch.zeros(1, 1, dtype=torch.float32)
        elif len(chunks) == 1:
            waveform = chunks[0]
        else:
            waveform = torch.cat(chunks, dim=-1)

        # Keep the shape compatible with the existing PhoneAudioOutput path:
        # torch tensor, mono, batch-like first dimension.
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)

        return waveform, int(bundle.get("sample_rate", KOKORO_SAMPLE_RATE))

    def _synthesize_mms(self, text: str, bundle: dict[str, Any]) -> tuple[torch.Tensor, int]:
        model = bundle["model"]
        tokenizer = bundle["tokenizer"]

        inputs = tokenizer(text, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            waveform = model(**inputs).waveform

        return waveform, int(model.config.sampling_rate)
