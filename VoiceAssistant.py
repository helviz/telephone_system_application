import asyncio
from llmModule.LLM import LLM
from transcribe.SSTModule import STTModule
from tts.TTSModule import TTSModule


class VoiceAssistant:
    def __init__(self, source, output, provider="gemini", lang="en"):
        # Validate language early
        allowed_langs = ["en", "fr", "sw"]
        if lang not in allowed_langs:
            raise ValueError(f"Unsupported language: {lang}. Choose from {allowed_langs}")

        self.lang = lang
        self.source = source

        # Pass the lang to STTModule so it locks to the correct Whisper model setting
        self.stt = STTModule(model_size="medium", lang=self.lang)

        # Initialize Brain with the specific language context
        self.llm = LLM.get_model(provider=provider, lang=self.lang)

        # Initialize Voice output (e.g., mms-tts-eng or similar)
        self.audio_output = output
        self.tts = TTSModule(output=self.audio_output)

        self.loop = None

    async def handle_text(self, text):
        try:
            print(f"--- Processing [{self.lang}]: {text} ---")

            # Generate LLM stream
            llm_stream = self.llm.generate_stream(text)

            # Direct stream to TTS and then to output
            # Ensure the TTS module also knows which language to synthesize
            await self.tts.speak_stream(llm_stream, lang=self.lang)
        except Exception as e:
            print(f"Error in pipeline: {e}")

    async def start(self):
        self.loop = asyncio.get_running_loop()

        print(f"--- Voice Assistant Active [{self.lang.upper()}] ---")
        try:
            # transcribe_stream now yields text based on the lang set in __init__
            async for text in self.stt.transcribe_stream(self.source):
                if text.strip():
                    # We use create_task so the assistant can keep listening
                    # while the LLM/TTS is processing the previous sentence
                    asyncio.create_task(self.handle_text(text))

        except Exception as e:
            print(f"Assistant error: {e}")