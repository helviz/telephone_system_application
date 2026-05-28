import torch
from transformers import VitsModel, AutoTokenizer


class TTSModule:
    def __init__(self, output, preloaded_models: dict = None):
        self.output = output
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model_map = {
            "en": "facebook/mms-tts-eng",
            "fr": "facebook/mms-tts-fra",
            "sw": "facebook/mms-tts-swh",
        }

        self.models = {}
        self.tokenizers = {}

        if preloaded_models:
            # Accept already-loaded models from preload.py — zero disk I/O
            for lang, (model, tokenizer) in preloaded_models.items():
                self.models[lang] = model
                self.tokenizers[lang] = tokenizer
            print(f"[TTS] Using preloaded models for: {list(preloaded_models.keys())}")
        else:
            # Fallback: load all languages now (first-run or test mode)
            print("[TTS] No preloaded models supplied — loading all languages now...")
            for lang in self.model_map:
                self.load_language(lang)
            print("[TTS]  All language models ready.")

    def load_language(self, lang="en"):
        if lang not in self.model_map:
            raise ValueError(f"Unsupported language: {lang}")

        model_name = self.model_map[lang]
        print(f"[TTS] Loading model: {model_name}")

        self.tokenizers[lang] = AutoTokenizer.from_pretrained(model_name)
        self.models[lang] = VitsModel.from_pretrained(model_name).to(self.device)
        self.models[lang].eval()

        print(f"[TTS] Loaded {lang} on {self.device}")

    async def speak_stream(self, text_generator, lang="en"):
        if lang not in self.models:
            self.load_language(lang)

        model = self.models[lang]
        tokenizer = self.tokenizers[lang]

        buffer = ""

        async for chunk in text_generator:
            buffer += chunk

            if any(p in chunk for p in [".", "!", "?", "\n"]):
                sentence = buffer.strip()
                if sentence:
                    await self._generate_audio(sentence, model, tokenizer)
                    buffer = ""

        if buffer.strip():
            await self._generate_audio(buffer.strip(), model, tokenizer)

    async def _generate_audio(self, text, model, tokenizer):
        inputs = tokenizer(text, return_tensors="pt").to(self.device)

        with torch.inference_mode():
            waveform = model(**inputs).waveform

        await self.output.send_audio(waveform, sample_rate=model.config.sampling_rate)