import queue

class ChunkRecord:
    def __init__(self, duration, chunk_len, sample_rate=16000):
        self.duration = duration
        self.chunk_len = chunk_len
        self.sample_rate = sample_rate
        self.total_chunks = int(duration * sample_rate / chunk_len)
        self.audio_queue = queue.Queue()
        self._stream = None
    
    def _callback(self, indata, _frames, _time, _status):
        self.audio_queue.put(indata[:, 0].copy())
    
    def __enter__(self):
        import sounddevice as sd
        self._stream = sd.InputStream(samplerate=self.sample_rate, channels=1, dtype="float32",
                                      blocksize=self.chunk_len, callback=self._callback)
        self._stream.start()
        return self
    
    def __exit__(self, *exc):
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
    
    def __iter__(self):
        for _ in range(self.total_chunks):
            yield self.audio_queue.get()