import abc

class AudioOutput(abc.ABC):
    @abc.abstractmethod
    async def send_audio(self, audio, sample_rate=16000):
        pass