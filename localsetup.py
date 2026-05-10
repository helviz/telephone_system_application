import asyncio
import sys
from Audio.MicrophoneSource import MicrophoneSource
from Audio.SpeakerAudioOutput import SpeakerAudioOutput
from VoiceAssistant import VoiceAssistant


def get_user_language():
    """Prompt the user for a language selection."""
    print("\n" + "=" * 30)
    print("  VOICE ASSISTANT SETUP")
    print("=" * 30)
    print("Select your preferred language:")
    print("1. English (en)")
    print("2. French (fr)")
    print("3. Swahili (sw)")
    print("-" * 30)

    choice = input("Enter 1, 2, or 3: ").strip()

    mapping = {
        "1": "en",
        "2": "fr",
        "3": "sw"
    }

    selected_lang = mapping.get(choice)

    if not selected_lang:
        print("Invalid selection. Defaulting to English.")
        return "en"

    return selected_lang


async def main():
    # 1. Prompt for language selection
    lang = get_user_language()

    # 2. Initialize local hardware components
    source = MicrophoneSource()
    output = SpeakerAudioOutput()

    # 3. Initialize the Voice Assistant
    # Now passing the dynamic 'lang' variable
    assistant = VoiceAssistant(
        source=source,
        output=output,
        provider="gemini",
        lang=lang
    )

    print(f"\n--- Local Voice Assistant Started [{lang.upper()}] ---")
    print("Speak into your microphone...")
    print("Press Ctrl+C to exit.")

    try:
        await assistant.start()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        if hasattr(source, 'close'):
            source.close()


if __name__ == "__main__":
    try:
        # Using run() is correct for the main entry point
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(0)