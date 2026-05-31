import torch
from transformers import VitsModel, AutoTokenizer

# Flush the buffer when it exceeds this many characters even without punctuation.
# Prevents indefinite blocking on LLM outputs that lack sentence-ending marks
# (e.g. bullet lists, code snippets, single-word answers).
MAX_BUFFER_CHARS = 200


class TTSModule:
    """
    FIX 3: The original speak_stream() only flushed the buffer when a
    sentence-ending punctuation character (. ! ? \\n) appeared in a chunk.
    LLM responses that contain no punctuation — lists, code, numbers, short
    factual replies — would accumulate indefinitely in the buffer and only
    play after the entire LLM stream finished, adding large perceived latency.

    Fix: add a MAX_BUFFER_CHARS hard limit. When the buffer grows beyond that
    threshold we flush at the nearest word boundary (split on the last space)
    so we never cut a word in half mid-synthesis.
    """

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
            for lang, (model, tokenizer) in preloaded_models.items():
                self.models[lang] = model
                self.tokenizers[lang] = tokenizer
            print(f"[TTS] Using preloaded models for: {list(preloaded_models.keys())}")
        else:
            print("[TTS] No preloaded models supplied — loading all languages now...")
            for lang in self.model_map:
                self.load_language(lang)
            print("[TTS] All language models ready.")

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

        model     = self.models[lang]
        tokenizer = self.tokenizers[lang]
        buffer    = ""

        async for chunk in text_generator:
            buffer += chunk

            # Primary flush trigger: sentence-ending punctuation
            if any(p in chunk for p in [".", "!", "?", "\n"]):
                sentence = buffer.strip()
                if sentence:
                    await self._generate_audio(sentence, model, tokenizer)
                buffer = ""

            # FIX 3 — Secondary flush trigger: buffer too long, no punctuation yet.
            # Flush at the last word boundary to avoid cutting words mid-synthesis.
            elif len(buffer) >= MAX_BUFFER_CHARS:
                last_space = buffer.rfind(" ")
                if last_space != -1:
                    # Synthesise everything up to the last complete word
                    sentence = buffer[:last_space].strip()
                    buffer   = buffer[last_space + 1:]   # carry the partial word forward
                else:
                    # No space found — the whole buffer is one giant token; flush as-is
                    sentence = buffer.strip()
                    buffer   = ""

                if sentence:
                    await self._generate_audio(sentence, model, tokenizer)

        # Drain whatever remains after the LLM stream closes
        if buffer.strip():
            await self._generate_audio(buffer.strip(), model, tokenizer)

    async def _generate_audio(self, text, model, tokenizer):
        inputs = tokenizer(text, return_tensors="pt").to(self.device)

        with torch.inference_mode():
            waveform = model(**inputs).waveform

        await self.output.send_audio(waveform, sample_rate=model.config.sampling_rate)