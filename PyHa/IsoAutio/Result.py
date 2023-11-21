from typing import Literal

import pandas as pd
import numpy as np

import logging
import math

from .BaseConfig import BaseConfig
from .Audio import Audio

ThresholdMethod = Literal["median", "mean", "avg", "std", "pure"]


class Result:
    def __init__(
        self,
        config: BaseConfig,
        audio: Audio,
        local_scores: np.ndarray[np.float],
    ):
        self.config = config
        self.audio = audio
        self.local_scores = local_scores

    def threshold_median(self) -> float:
        return np.median(self.local_scores) * self.threshold_const

    def threshold_mean(self) -> float:
        return np.mean(self.local_scores) * self.threshold_const

    def threshold_std(self) -> float:
        return np.mean(self.local_scores) + (
            np.std(self.local_scores) * self.threshold_const
        )

    def threshold_pure(self) -> float:
        result = self.threshold_const
        if result < 0:
            logging.info("threshold is less than zero, setting to zero")
            result = 0
        elif result > 1:
            logging.info("threshold is greater than one, setting to one")
            result = 1

        return result

    def threshold(self) -> float:
        """Utility to compute the threshold (among many different options)

        Returns:
            float: how labels are created in current audio clip
        """
        methods = {
            "median": self.threshold_median,
            "mean": self.threshold_mean,
            "avg": self.threshold_mean,
            "std": self.threshold_std,
            "pure": self.threshold_pure,
        }

        method = methods.get(self.config.threshold_method, None)
        if method is None:
            logging.error(
                f"{self.config.threshold_method} is not a valid threshold method"
            )
            return 0.0

        return method()

    def __create_isolate_entry(self, empty: bool = False):
        if empty:
            return {
                "FOLDER": [],
                "IN FILE": [],
                "CHANNEL": [],
                "CLIP LENGTH": [],
                "SAMPLE RATE": [],
                "OFFSET": [],
                "MANUAL ID": [],
            }

        return {
            "FOLDER": self.config.audio_dir,
            "IN FILE": self.audio.file,
            "CHANNEL": 0,
            "CLIP LENGTH": self.audio.duration,
            "SAMPLE RATE": self.audio.sample_rate,
            "OFFSET": [],
            "MANUAL ID": self.config.manual_id,
        }

    @property()
    def samples_per_score(self) -> int:
        """How many samples one local score represents"""
        return len(self.audio.signal) // len(self.local_scores)

    @property()
    def time_per_score(self) -> float:
        return self.samples_per_score / self.audio.sample_rate

    @property()
    def scores_per_second(self) -> float:
        """Number of local scores per second"""
        return len(self.local_scores) / self.audio.duration

    def isolate_steinberg(self, window_size: int) -> pd.DataFrame:
        """Technique developed by Gabriel Steinberg that attempts to take the local
        score array output of a neural network and lump local scores together in
        a way to produce automated labels based on a class across an audio clip.

        Technique Pseudocode:

        Loop through local score array:
            if current local score > (threshold and threshold_min):
                build an annotation with current local score at the center with
                +- window_size/2 seconds around current local score.
            else:
                continue
        extra logic handles overlap if a local score meets the criteria within
        the "window_size" from a prior local score

        Args:
            window_size (int): @audit enter description

        Returns:
            pd.DataFrame: Pandas Dataframe of automated labels for the audio clip.
        """

        # create entry for audio clip
        entry = self.__create_isolate_entry()
        thresh = self.threshold()
        thresh_scores = self.local_scores >= max(thresh, self.config.threshold_min)

        # check if window size is smaller than time between two local scores
        # (as a safeguard against problems that can occur)
        if int(window_size / 2 * self.audio.sample_rate) * 2 >= self.samples_per_score:
            # Set up to find the starts and ends of clips (not considering window)
            thresh_scores = np.append(thresh_scores, [0])
            rolled_scores = np.roll(thresh_scores, 1)
            rolled_scores[0] = 0
            diff_scores = thresh_scores - rolled_scores

            # Logic for finding the starts and ends:
            # If thresh_scores = [1 1 1 1 0 0 0 1 1], then
            # thresh_scores becomes [1 1 1 1 0 0 0 1 1 0] and
            # rolled_scores are [0 1 1 1 1 0 0 0 1 1]. Subtracting
            # yields [1 0 0 0 -1 0 0 1 0 -1]. The 1s are the starts of the clips,
            # and the -1s are 1 past the ends of the clips

            # Adds the "window" to each annotation
            offset = int(window_size / 2 * self.audio.sample_rate)
            starts = np.where(diff_scores == 1)[0] * self.samples_per_score - offset
            ends = np.where(diff_scores == -1)[0] - 1
            ends = ends * self.samples_per_score + offset

            # Does not continue if no annotations exist
            if len(starts) == 0:
                return pd.DataFrame.from_dict(self.__create_isolate_entry(empty=True))

            # Checks annotations for any overlap, and removes if so
            i = 0
            while True:
                if i == len(ends) - 1:
                    break
                if starts[i + 1] < ends[i]:
                    ends = np.delete(ends, i)
                    starts = np.delete(starts, i + 1)
                else:
                    i += 1

            # Correcting bounds
            starts[0] = max(0, starts[0])
            ends[-1] = min(len(self.audio.signal), ends[-1])

            # Calculates offsets and durations from starts and ends
            entry["OFFSET"] = starts * 1.0 / self.audio.sample_rate
            entry["DURATION"] = ends - starts
            entry["DURATION"] = entry["DURATION"] * 1.0 / self.audio.sample_rate

            # Assigns manual ids to all annotations
            entry["MANUAL ID"] = np.full(
                entry["OFFSET"].shape, self.config.self.config.manual_id
            )
        else:
            # Simply assigns each 1 in thresh scores to be its own window if windows are too small
            entry["OFFSET"] = (
                np.where(thresh_scores == 1)[0]
                * self.samples_per_score
                / self.audio.sample_rate
                - window_size / 2
            )
            entry["DURATION"] = np.full(entry["OFFSET"].shape, window_size * 1.0)
            if entry["OFFSET"] < 0:
                entry["OFFSET"][0] = 0
                entry["DURATION"][0] = window_size * 0.5
            entry["MANUAL ID"] = np.full(
                entry["OFFSET"].shape, self.config.self.config.manual_id
            )

        return pd.DataFrame.from_dict(entry)

    def isolate_simple(self) -> pd.DataFrame:
        """Technique suggested by Irina Tolkova, implemented by Jacob Ayers.
        Attempts to produce automated annotations of an audio clip based
        on local score array outputs from a neural network.

        Technique Pseudocode:
        Loop through local score array:
            if current local score > (threshold and threshold_min)
            and annotation start = 0:
                start annotation
            else if current local score < thresh and annotation start = 1:
                end annotation
            else:
                continue

        Returns:
            Pandas Dataframe of automated labels for the audio clip.
        """
        entry = self.__create_isolate_entry()
        thresh = self.threshold()
        thresh_scores = self.local_scores >= max(thresh, self.config.threshold_min)

        # Set up to find the starts and ends of clips
        thresh_scores = np.append(thresh_scores, [0])
        rolled_scores = np.roll(thresh_scores, 1)
        rolled_scores[0] = 0

        # Logic for finding starts and ends given in steinberg isolate
        diff_scores = thresh_scores - rolled_scores

        # Calculates offsets and durations from difference
        entry["OFFSET"] = np.where(diff_scores == 1)[0] * self.time_per_score * 1.0
        entry["DURATION"] = (
            np.where(diff_scores == -1)[0] * self.time_per_score - entry["OFFSET"]
        )

        # Assigns manual ids to all annotations
        entry["MANUAL ID"] = np.full(entry["OFFSET"].shape, self.config.manual_id)

        return pd.DataFrame.from_dict(entry)

    def isolate_stack(self) -> pd.DataFrame:
        """Technique created by Jacob Ayers. Attempts to produce automated
        annotations of an audio clip base on local score array outputs
        from a neural network.

        Technique Pseudocode:

        Loop through local score array:
            if current local score > (threshold and threshold_min):
                if annotation start false:
                    set annotation start true
                push to stack counter
            else if current local score < thresh and annotation start true:
                pop from stack counter
                if stack counter = 0:
                    end annotation
            else:
                continue

        Returns:
            Pandas Dataframe of automated labels for the audio clip.
        """
        entry = self.__create_isolate_entry()
        thresh = self.threshold()
        thresh_scores = self.local_scores >= max(thresh, self.config.threshold_min)

        # Set up to find the starts and ends of clips
        thresh_scores = np.append(thresh_scores, [0])
        rolled_scores = np.roll(thresh_scores, 1)
        rolled_scores[0] = 0

        # Logic for finding starts and ends given in steinberg isolate
        diff_scores = thresh_scores - rolled_scores

        starts = np.where(diff_scores == 1)[0]
        ends = np.where(diff_scores == -1)[0]

        # Stack algorithm: considers a stack counter, and
        # updates stack counter between annotations (+1 for every
        # entry above the threshold, -1 for below); Combines annotations
        # in this way, along with any adjacent annotations (where stack
        # counter is 0 for one value between annotations).
        i = 0
        while i < len(ends):
            stack_counter = ends[i] - starts[i]
            new_end = ends[i] + stack_counter
            while i < len(ends) - 1 and starts[i + 1] <= new_end:
                stack_counter -= starts[i + 1] - ends[i]
                stack_counter += ends[i + 1] - starts[i + 1]
                ends = np.delete(ends, i)
                starts = np.delete(starts, i + 1)
                new_end = ends[i] + stack_counter
            ends[i] = new_end
            i += 1

        # Addressing situation where end goes above max length of local scores
        ends[-1] = min(len(self.local_scores) - 1, ends[-1])

        # Deletes annotation if it starts on the
        # last local score
        if starts[-1] == len(self.local_scores) - 1:
            starts = np.delete(starts, len(starts) - 1)
            ends = np.delete(ends, len(ends) - 1)

        # Calculates offsets and durations from starts/ends
        entry["OFFSET"] = starts * self.time_per_score
        entry["DURATION"] = ends - starts
        entry["DURATION"] = entry["DURATION"] * self.time_per_score

        # Assigns manual ids to all annotations
        entry["MANUAL ID"] = np.full(entry["OFFSET"].shape, self.config.manual_id)

        return pd.DataFrame.from_dict(entry)

    def isolate_chunk(self, chunk_size: int) -> pd.DataFrame:
        """
        Technique created by Jacob Ayers. Attempts to produce automated
        annotations of an audio clip based on local score array outputs
        from a neural network.

        Technique Pseudocode:

        number of chunks = clip length / "chunk_size"
        Loop through number of chunks:
            if max(local score chunk) > (threshold and "threshold_min"):
                set the chunk as an annotation
            else:
                continue

        Returns:
            Pandas Dataframe of automated labels for the audio clip.
        """

        entry = self.__create_isolate_entry()
        thresh = self.threshold()

        # calculating the number of chunks that define an audio clip
        chunk_count = math.ceil(
            len(self.audio.signal) / (chunk_size * self.audio.sample_rate)
        )

        # calculating the chunk size with respect to the local score array
        local_scores_per_chunk = self.scores_per_second * chunk_size

        # Creates indices for starts of chunks using np.linspace
        # which creates even splits across a range, and then is
        # treated as int (rounds down)
        chunk_starts_float = np.linspace(
            start=0,
            stop=chunk_count * local_scores_per_chunk,
            num=chunk_count,
            endpoint=False,
        )
        chunk_starts = chunk_starts_float.copy().astype(int)

        # Deletes the first element of the array (0) to
        # avoid empty array
        chunk_starts = np.delete(chunk_starts, 0)

        # Creates chunked scores based on starts
        # Finds max value of each chunked array
        chunked_scores = np.array(
            list(map(np.amax, np.split(self.local_scores, chunk_starts)))
        )

        # Finds which chunks are above threshold, and creates indices based on that
        thresh_scores = chunked_scores >= max(thresh, self.config.threshold_min)
        chunk_indices = np.where(thresh_scores == 1)[0]

        # Assigns offset values based on float values of the starts
        entry["OFFSET"] = chunk_starts_float[chunk_indices] / self.scores_per_second

        # Creates durations based on float values of chunk starts
        all_chunk_durs = (
            np.roll(chunk_starts_float, -1) / self.scores_per_second
            - chunk_starts_float / self.scores_per_second
        )
        all_chunk_durs[-1] = (
            len(self.local_scores) / self.scores_per_second
            - chunk_starts_float[-1] / self.scores_per_second
        )
        entry["DURATION"] = all_chunk_durs[chunk_indices]

        # Assigns manual ids to all annotations
        entry["MANUAL ID"] = np.full(entry["OFFSET"].shape, self.config.manual_id)

        return pd.DataFrame.from_dict(entry)


# @todo
# Make it so that a user has the option of an overlap between the chunks.
# Make it so that a user can choose how many samples have to be above the
# threshold in order to consider a chunk to be good or not.
# Give the option to combine annotations that follow one-another.
