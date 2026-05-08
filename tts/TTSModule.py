import torch
from TTS.api import TTS
from transformers import VitsModel, AutoTokenizer


class TTSModule:
    def __init__(self, lang="en"):
        self.lang = lang
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load models based on language
        if self.lang in ["en", "fr"]:
            # Coqui XTTS v2
            self.model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(self.device)

        elif self.lang == "sw":
            # Facebook MMS Swahili
            self.model_name = "facebook/mms-tts-swh"
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = VitsModel.from_pretrained(self.model_name).to(self.device)

    async def speak_stream(self, text_generator):
        """Processes the LLM stream and sends text to synthesis."""
        buffer = ""
        async for chunk in text_generator:
            buffer += chunk
            if any(punc in chunk for punc in [".", "!", "?", "\n"]):
                sentence = buffer.strip()
                if sentence:
                    await self._generate_audio(sentence)
                    buffer = ""

    async def _generate_audio(self, text: str):
        if self.lang in ["en", "fr"]:
            # Coqui synthesis (Note: uses a reference wav for voice cloning)
            # You need a 5-10 second sample.wav in your project folder
            self.model.tts_to_file(
                text=text,
                speaker_wav="sample.wav",
                language=self.lang,
                file_path="output.wav"
            )
        elif self.lang == "sw":
            # Facebook MMS synthesis
            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
            with torch.no_grad():
                output = self.model(**inputs).waveform
            # Convert output tensor to audio and play (using sounddevice or simpleaudio)
            self._play_tensor(output)

    def _play_tensor(self, waveform):
        # Implementation for playing raw tensor audio
        import sounddevice as sd
        sd.play(waveform.cpu().numpy().squeeze(), samplerate=16000)
        sd.wait()