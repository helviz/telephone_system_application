import pyaudio

from Audio.AudioSource import AudioSource


class MicrophoneSource(AudioSource):
    def __init__(self, sample_rate=16000, chunk_size=1024):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size
        )

    def get_stream(self):
        """Generator that yields raw chunks of audio."""
        while True:
            yield self.stream.read(self.chunk_size, exception_on_overflow=False)

    def close(self):
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()