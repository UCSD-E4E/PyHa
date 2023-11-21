from __future__ import annotations
from typing import Callable

import scipy.signal as scipy_signal
import pandas as pd
import librosa

import pandas as pd
import logging
import os

from .BaseConfig import BaseConfig
from .Audio import Audio


class BaseAdapter:
    def __init__(self, config: BaseConfig):
        self.config = config

    def generate(self) -> pd.Dataframe:
        raise NotImplementedError()

    def __generate_annotations(
        self, create_entry: Callable[[Audio], pd.DataFrame]
    ) -> pd.Dataframe:
        annotations = pd.DataFrame()

        # generate local scores for every bird file in chosen directory
        for audio_file in os.listdir(self.config.audio_dir):
            if os.path.isdir(self.config.audio_dir + audio_file):
                continue

            # Reading in the audio files using librosa, converting to single channeled data with original sample rate
            # Reason for the factor for the signal is explained here: https://stackoverflow.com/questions/53462062/pyaudio-bytes-data-to-librosa-floating-point-time-series
            # Librosa scales down to [-1, 1], but the models require the range [-32768, 32767]
            try:
                signal, sample_rate = librosa.load(
                    self.config.audio_dir + audio_file, sr=None, mono=True
                )
                signal = signal * 32768
            except KeyboardInterrupt:
                exit("Keyboard interrupt")
            except BaseException:
                logging.info(f"failed to load {audio_file}, skipping")
                continue

            # downsample the audio if the sample rate isn't 44.1 kHz
            # Force everything into the human hearing range.
            # May consider reworking this function so that it upsamples as well
            try:
                if sample_rate != self.config.normalized_sample_rate:
                    rate_ratio = self.config.normalized_sample_rate / sample_rate
                    signal = scipy_signal.resample(
                        signal, int(len(signal) * rate_ratio)
                    )
                    sample_rate = self.config.normalized_sample_rate
            except KeyboardInterrupt:
                exit("Keyboard interrupt")
            except:
                logging.error(f"failed to downsample {audio_file}, skipping")
                continue

            # convert stereo to mono if needed
            # might want to compare to just taking the first set of data.
            if len(signal.shape) == 2:
                signal = signal.sum(axis=1) / 2

            audio = Audio(audio_file, signal, sample_rate)

            try:
                new_entry = create_entry(audio)
            except KeyboardInterrupt:
                exit("Keyboard interrupt")
            except BaseException:
                logging.error(
                    f"could not isolate bird calls from {audio_file}, skipping"
                )
                continue

            annotations = (
                new_entry if annotations.empty else annotations.append(new_entry)
            )

        # Quick fix to indexing
        annotations.reset_index(inplace=True, drop=True)
        return annotations
