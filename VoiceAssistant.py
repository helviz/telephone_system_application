import asyncio
from llmModule.LLM import LLM
from transcribe.SSTModule import STTModule
from tts.TTSModule import TTSModule


class VoiceAssistant:
    def __init__(self, source, output, provider="gemini", lang="en"):
        # The source can now be MicrophoneSource OR PhoneStreamSource
        self.source = source
        self.stt = STTModule()
        self.lang = lang

        # Initialize Brain
        self.llm = LLM.get_model(provider=provider, lang=lang)

        # Initialize Voice with the provided output (Speaker or Phone)
        self.audio_output = output
        self.tts = TTSModule(output=self.audio_output)

        self.loop = None

    async def handle_text(self, text):
        try:
            # Generate LLM stream
            llm_stream = self.llm.generate_stream(text)

            # Direct stream to TTS and then to PhoneAudioOutput/SpeakerAudioOutput
            await self.tts.speak_stream(llm_stream, lang=self.lang)
        except Exception as e:
            print(f"Error in pipeline: {e}")

    async def start(self):
        self.loop = asyncio.get_running_loop()

        try:
            # For PhoneStreamSource, we consume the async generator
            # if hasattr(self.source, 'get_stream'):
            async for text in self.stt.transcribe_stream(self.source):
                if text.strip():
                    await self.handle_text(text)
        except Exception as e:
            print(f"Assistant error: {e}")