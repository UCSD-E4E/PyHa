import numpy as np


class Audio:
    def __init__(self, filename: str, signal: np.ndarray[np.int], sample_rate: int):
        self.filename = filename
        self.signal = signal
        self.sample_rate = sample_rate

    @property
    def duration(self):
        return len(self.signal) / self.sample_rate
