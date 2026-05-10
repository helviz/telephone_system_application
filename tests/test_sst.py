from Audio.MicrophoneSource import MicrophoneSource
from transcribe.SSTModule import STTModule


def get_language_selection():
    """Prompts the user to select a language and returns the code."""
    print("Select Transcription Language:")
    print("1: English")
    print("2: French")
    print("3: Swahili")

    choice = input("\nEnter choice (1, 2, or 3): ").strip()

    mapping = {
        "1": "en",
        "2": "fr",
        "3": "sw"
    }

    return mapping.get(choice)


def run_stt_test():
    # 1. Get user language choice
    lang_code = get_language_selection()

    if not lang_code:
        print("Invalid selection. Please run the script again and choose 1, 2, or 3.")
        return

    # 2. Initialize the Microphone
    print(f"\nInitializing Microphone...")
    mic = MicrophoneSource(sample_rate=16000, chunk_size=1024)

    # 3. Initialize the STT Module
    print(f"Loading Whisper Medium model for code: {lang_code}...")
    try:
        # We explicitly pass the selected language here
        stt = STTModule(model_size="medium", device="cpu", lang=lang_code)
    except Exception as e:
        print(f"Error initializing STT: {e}")
        mic.close()
        return

    print("\n" + "=" * 30)
    print(f" SYSTEM ACTIVE: {lang_code.upper()} ")
    print(" Speak into your mic...")
    print(" Press Ctrl+C to stop.")
    print("=" * 30 + "\n")

    try:
        # 4. Stream and Print
        for transcription in stt.transcribe_stream(mic):
            # Using flush=True to ensure it prints immediately in all terminals
            print(f"[{lang_code.upper()}] >> {transcription}", flush=True)

    except KeyboardInterrupt:
        print("\n\n--- Session Ended by User ---")
    finally:
        mic.close()
        print("Microphone resources released.")


if __name__ == "__main__":
    run_stt_test()