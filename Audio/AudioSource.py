import abc
class AudioSource(abc.ABC):
    @abc.abstractmethod
    async def get_stream(self):
        """Async generator that yields raw PCM bytes."""
        pass