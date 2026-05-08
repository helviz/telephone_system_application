import asyncio
from Audio.MicrophoneSource import MicrophoneSource
from llmModule.LLM import LLM
from transcribe.SSTModule import STTModule


# from tts.TTSModule import TTSModule # Commented out for testing

class VoiceAssistant:
    def __init__(self, provider="gemini", lang="en"):
        self.source = MicrophoneSource()
        self.stt = STTModule()
        self.lang = lang  # Store lang locally since TTS is disabled

        # 1. Initialize the Brain using your LLM Factory
        self.llm = LLM.get_model(provider=provider, lang=lang)

        # 2. Initialize the Voice (Disabled for testing)
        # self.tts = TTSModule(lang=lang)

        self.loop = None

    async def handle_text(self, text):
        """
        Coordinates the LLM. This runs in the main event loop.
        """
        try:
            # Generate the stream from the LLM
            # Remember: Our Gemini/GGUF strategies print to terminal internally
            llm_stream = self.llm.generate_stream(text)

            # We MUST consume the stream for the print statements inside to trigger
            async for _ in llm_stream:
                pass

        except Exception as e:
            print(f"Error in pipeline: {e}")

    async def start(self):
        self.loop = asyncio.get_running_loop()
        print(f"--- LLM Test Mode Active (Lang: {self.lang}) ---")
        print("Speak into the mic to see the LLM response...")

        try:
            def run_stt():
                # This stays synchronous as faster-whisper is a blocking generator
                for text in self.stt.transcribe_stream(self.source):
                    if text.strip():
                        print(f"\n[STT Captured]: {text}")

                        # Schedule the LLM response
                        asyncio.run_coroutine_threadsafe(
                            self.handle_text(text),
                            self.loop
                        )

            await self.loop.run_in_executor(None, run_stt)

        except KeyboardInterrupt:
            print("\nShutting down...")
            self.source.close()


if __name__ == "__main__":
    # Test with Gemini (Cloud) or Qwen (Local)
    # Languages: 'en', 'fr', 'sw'
    assistant = VoiceAssistant(provider="gemini", lang="en")
    asyncio.run(assistant.start())


# import asyncio
# from Audio.MicrophoneSource import MicrophoneSource
# from llmModule.LLM import LLM
# from transcribe.SSTModule import STTModule
#
# from tts.TTSModule import TTSModule
#
#
# class VoiceAssistant:
#     def __init__(self, provider="gemini", lang="en"):
#         self.source = MicrophoneSource()
#         self.stt = STTModule()
#
#         # 1. Initialize the Brain using your LLM Factory
#         self.llm = LLM.get_model(provider=provider, lang=lang)
#
#         # 2. Initialize the Voice
#         # self.tts = TTSModule(lang=lang)
#
#         self.loop = None
#
#     async def handle_text(self, text):
#         """
#         Coordinates the LLM and TTS. This runs in the main event loop.
#         """
#         try:
#             # Generate the stream from the LLM (which now also prints to terminal)
#             llm_stream = self.llm.generate_stream(text)
#
#             # Pipe the LLM stream into the TTS sentence-buffer speaker
#             # await self.tts.speak_stream(llm_stream)
#
#         except Exception as e:
#             print(f"Error in pipeline: {e}")
#
#     async def start(self):
#         self.loop = asyncio.get_running_loop()
#         print(f"--- Assistant Started (Lang: {self.tts.lang}) ---")
#
#         try:
#             # We run the STT blocking generator in a thread
#             def run_stt():
#                 # self.stt.transcribe_stream handles the VAD and yields text
#                 for text in self.stt.transcribe_stream(self.source):
#                     if text.strip():
#                         # We don't print here anymore as LLM handles terminal output
#                         # We schedule the async handling of this text
#                         asyncio.run_coroutine_threadsafe(
#                             self.handle_text(text),
#                             self.loop
#                         )
#
#             await self.loop.run_in_executor(None, run_stt)
#
#         except KeyboardInterrupt:
#             print("\nShutting down...")
#             self.source.close()
#
#
#
#
# if __name__ == "__main__":
#     # You can change provider to "qwen" and lang to "sw" or "fr"
#     assistant = VoiceAssistant(provider="gemini", lang="en")
#     asyncio.run(assistant.start())