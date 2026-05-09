import asyncio

from Audio.SpeakerAudioOutput import SpeakerAudioOutput
from tts.TTSModule import TTSModule


async def fake_stream(text: str, delay: float = 0.05):
    """Simulates an LLM streaming text chunk by chunk."""
    for word in text.split():
        yield word + " "
        await asyncio.sleep(delay)


async def test_english(tts: TTSModule):
    print("[TEST] English TTS...")
    text = (
        "Hello, this is a test of the English text to speech system. "
        "The model should generate natural sounding speech."
    )
    await tts.speak_stream(fake_stream(text), lang="en")
    print("[DONE] English")


async def test_french(tts: TTSModule):
    print("[TEST] French TTS...")
    text = (
        "Bonjour, ceci est un test du système de synthèse vocale en français. "
        "Le modèle devrait produire une voix naturelle."
    )
    await tts.speak_stream(fake_stream(text), lang="fr")
    print("[DONE] French")


async def test_swahili(tts: TTSModule):
    print("[TEST] Swahili TTS...")
    text = (
        "Habari, huu ni mtihani wa mfumo wa kubadilisha maandishi kuwa sauti kwa Kiswahili. "
        "Mfano huu unapaswa kutoa sauti ya kawaida."
    )
    await tts.speak_stream(fake_stream(text), lang="sw")
    print("[DONE] Swahili")


async def main():
    output = SpeakerAudioOutput()
    tts = TTSModule(output=output)

    # Pre-load all languages before testing
    print("[INIT] Loading models...")
    tts.load_language("en")
    tts.load_language("fr")
    tts.load_language("sw")
    print("[INIT] All models loaded\n")

    print("=" * 50)
    await test_english(tts)

    print("=" * 50)
    await test_french(tts)

    print("=" * 50)
    await test_swahili(tts)

    print("=" * 50)
    print("All tests completed.")


if __name__ == "__main__":
    asyncio.run(main())