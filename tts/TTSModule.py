import torch
from transformers import VitsModel, AutoTokenizer


class TTSModule:
    def __init__(self, output):
        self.output = output
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model_map = {
            "en": "facebook/mms-tts-eng",
            "fr": "facebook/mms-tts-fra",
            "sw": "facebook/mms-tts-swh",
        }

        self.models = {}
        self.tokenizers = {}

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


# # English
# wget https://huggingface.co/facebook/mms-tts-eng/resolve/main/model.safetensors -O mms-tts-eng.safetensors
# wget https://huggingface.co/facebook/mms-tts-eng/resolve/main/config.json -O mms-tts-eng-config.json
#
# # French
# wget https://huggingface.co/facebook/mms-tts-fra/resolve/main/model.safetensors -O mms-tts-fra.safetensors
# wget https://huggingface.co/facebook/mms-tts-fra/resolve/main/config.json -O mms-tts-fra-config.json
#
# # Swahili
# wget https://huggingface.co/facebook/mms-tts-swh/resolve/main/model.safetensors -O mms-tts-swh.safetensors
# wget https://huggingface.co/facebook/mms-tts-swh/resolve/main/config.json -O mms-tts-swh-config.json

