from Audio.AudioOutput import AudioOutput


class VoiceSession:
    def __init__(self, output: AudioOutput):
        self.output = output

    async def speak(self, audio):
        await self.output.send_audio(audio)