import abc

class AudioSource(abc.ABC):
    @abc.abstractmethod
    def get_stream(self):
        pass