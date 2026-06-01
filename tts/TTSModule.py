import asyncio
import time
import torch
from transformers import VitsModel, AutoTokenizer

MAX_BUFFER_CHARS = 200


class TTSModule:
    def __init__(self, output, preloaded_models: dict = None):
        self.output = output
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Safe map tracking Hugging Face Model Hub repositories
        self.model_map = {
            "en": "facebook/mms-tts-eng",
            "fr": "facebook/mms-tts-fra",
            "sw": "facebook/mms-tts-swh",
        }

        # Keep these empty at startup to preserve precious Free Space memory!
        self.models = {}
        self.tokenizers = {}

        if preloaded_models:
            for lang, (model, tokenizer) in preloaded_models.items():
                self.models[lang] = model
                self.tokenizers[lang] = tokenizer

    def _ensure_model_loaded(self, lang: str):
        """
        Dynamically extracts and allocates weights only when a phone call
        actively requests a specific language block.
        """
        if lang not in self.models:
            print(
                f"\n[TTS] 🧠 Free Space Optimization: Lazy loading model weights for language context: [{lang.upper()}]...")
            t_load = time.time()

            repo_id = self.model_map[lang]
            self.tokenizers[lang] = AutoTokenizer.from_pretrained(repo_id)
            self.models[lang] = VitsModel.from_pretrained(repo_id).to(self.device)

            print(f"[TTS] Successfully allocated memory for [{lang.upper()}] in {time.time() - t_load:.2f}s.\n")

        return self.models[lang], self.tokenizers[lang]

    async def speak_stream(self, text_generator, lang="en", on_first_audio=None):
        # Dynamically ensure the specific model is loaded without touching the other 2 languages
        model, tokenizer = self._ensure_model_loaded(lang)

        buffer = ""
        _first_audio_fired = False

        async for chunk in text_generator:
            buffer += chunk

            # Sentence splitting logic
            if any(p in chunk for p in [".", "!", "?", "\n"]) or len(buffer) >= MAX_BUFFER_CHARS:
                if len(buffer) >= MAX_BUFFER_CHARS:
                    # Flush safely at closest word boundary
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
                    await self._generate_audio(to_say, model, tokenizer, on_first_audio=cb)

        # Drain remaining content
        if buffer.strip():
            cb = None
            if not _first_audio_fired and on_first_audio:
                cb = on_first_audio
                _first_audio_fired = True
            await self._generate_audio(buffer.strip(), model, tokenizer, on_first_audio=cb)

    async def _generate_audio(self, text, model, tokenizer, on_first_audio=None):
        loop = asyncio.get_running_loop()
        t0 = time.time()

        def _synthesize():
            inputs = tokenizer(text, return_tensors="pt").to(self.device)
            with torch.inference_mode():
                waveform = model(**inputs).waveform
            return waveform

        waveform = await loop.run_in_executor(None, _synthesize)

        tts_elapsed = time.time() - t0
        try:
            import stats
            stats.record_tts_latency(tts_elapsed)
        except Exception:
            pass

        if on_first_audio:
            on_first_audio()

        await self.output.send_audio(waveform, sample_rate=model.config.sampling_rate)