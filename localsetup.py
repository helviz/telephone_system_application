import asyncio
from Audio.MicrophoneSource import MicrophoneSource
from Audio.SpeakerAudioOutput import SpeakerAudioOutput
from VoiceAssistant import VoiceAssistant


async def main():
    # 1. Initialize local hardware components
    # MicrophoneSource handles capturing audio from your mic
    source = MicrophoneSource()

    # SpeakerAudioOutput handles playing audio through your speakers
    output = SpeakerAudioOutput()

    # 2. Initialize the Voice Assistant
    # We pass the local source and output into the refactored class
    # You can change provider to "qwen" to test your local GGUF model
    assistant = VoiceAssistant(
        source=source,
        output=output,
        provider="gemini",
        lang="en"
    )

    # 3. Start the assistant
    print("--- Local Voice Assistant Started ---")
    print("Speak into your microphone...")

    try:
        await assistant.start()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        # Clean up resources if your classes have close methods
        if hasattr(source, 'close'):
            source.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass