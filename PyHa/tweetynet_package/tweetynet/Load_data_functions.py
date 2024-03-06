import os
import math
import pickle

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

from .TweetyNetAudio import wav2spc, create_spec
from .CustomAudioDataset import CustomAudioDataset

"""Make function headers for each function."""


def get_frames(
    x, hop_length
):  # Calculates the frame number given the start point and hop length of a spectrogram.
    return ((x) / hop_length) + 1


def frames2seconds(
    x, sr
):  # Calculates the time in seconds from the frame number of a spectrogram.
    return x / sr


def window_spectrograms(spc, Y, uid, time_bin, windowsize):
    """
    Helper function to Window/split the data to the specified window size.

    Args:
        spc: (Numpy Array)
            - Spectrogram of the wav file.

        Y: (Numpy Array)
            - labels of the spectrogram.

        uid: (Numpy Array)
            - Filename of the file.

        time_bin: (Numpy Array)
            - NUmber of time bins in the spectrograms.

        windowsize: (int)
            - Number of seconds in a window of the clip.

    Returns: (Tuple)
        Contains the uids, spectrograms, labels, and time bins of the windowed spectrogram.
    """
    computed = (
        windowsize // time_bin
    )  # verify, big assumption. are time bins consistent?
    time_axis = int(computed * (spc.shape[1] // computed))
    freq_axis = int(spc.shape[1] // computed)  # 31, 2, 19
    spc_split = np.split(spc[:, :time_axis], freq_axis, axis=1)
    Y_split = np.split(Y[:time_axis], freq_axis)
    uid_split = [str(i) + "_" + uid for i in range(freq_axis)]
    return spc_split, Y_split, uid_split


def window_data(spcs, ys, uids, time_bins, windowsize):
    """
    Window/split the data to the specified window size.

    Args:
        spcs: (Numpy Array)
            - Spectrogram of the wav file.

        Ys: (Numpy Array)
            - labels of the spectrogram.

        uids: (Numpy Array)
            - Filename of each file.

        time_bins: (Numpy Array)
            - NUmber of time bins in the spectrograms.

        windowsize: (int)
            - Number of seconds in a window of the clip.

    Returns: (Dictionary)
        Contains the uids, spectrograms, labels, and time bins of the windowed spectrograms.
    """
    windowed_dataset = {"uids": [], "X": [], "Y": []}
    # print("Windowing Spectrogram")
    for i in range(len(uids)):
        spc_split, Y_split, uid_split = window_spectrograms(
            spcs[i], ys[i], uids[i], time_bins[i], windowsize
        )
        windowed_dataset["X"].extend(spc_split)
        windowed_dataset["Y"].extend(Y_split)
        windowed_dataset["uids"].extend(uid_split)
    return windowed_dataset


def create_signal2spec(signal, SR, n_mels, frame_size, hop_length):
    """
    Creates the spectrogram from the signal.

    Args:
        SIGNAL: (list of ints)
            - Samples from the audio clip.

        SR: (int)
            - Sampling rate of the audio clip, usually 44100.

        n_mels: (int)
            - (Mel Spectrogram Parameter)

        frame_size: (int)
            - (Mel Spectrogram Parameter)

        hop_length: (int)
            - (Mel Spectrogram Parameter)

        windowsize: (int)
            - Number of seconds in a window of the clip.

    Returns: (Dictionary)
        Contains the uids, spectrograms, labels, and time bins.
    """
    features = {"uids": [], "X": [], "Y": [], "time_bins": []}
    spc = create_spec(
        signal, fs=SR, n_mels=n_mels, n_fft=frame_size, hop_len=hop_length
    )
    time_bins = (len(signal) / SR) / spc.shape[1]
    Y = np.array([0] * spc.shape[1])
    features["uids"].append("f")
    features["X"].append(spc)
    features["Y"].append(Y)
    features["time_bins"].append(time_bins)
    return features


def load_signal2spec(signal, SR, n_mels, frame_size, hop_length):
    """
    Load signal to spectrogram

    Args:
        SIGNAL: (list of ints)
            - Samples from the audio clip.

        SR: (int)
            - Sampling rate of the audio clip, usually 44100.

        n_mels: (int)
            - (Mel Spectrogram Parameter)

        frame_size: (int)
            - (Mel Spectrogram Parameter)

        hop_length: (int)
            - (Mel Spectrogram Parameter)

        windowsize: (int)
            - Number of seconds in a window of the clip.

    Returns: (Tuples)
        Containing the spectrogram(X), labels(Y), unique file identifier(uids), Number of time bins(time_bins).
    """
    dataset = create_signal2spec(signal, SR, n_mels, frame_size, hop_length)
    X = dataset["X"]
    Y = dataset["Y"]
    uids = dataset["uids"]
    time_bins = dataset["time_bins"]
    return X, Y, uids, time_bins


def compute_features(signal, SR=44100):
    """
    Compute features

    Args:
        SIGNAL: (list of ints)
            - Samples from the audio clip.

        SR: (int)
            - Sampling rate of the audio clip, usually 44100.

    Returns:
        CustomAudioDataset containing, the spectrograms(X), labels(Y), and UIDS(UIDS) of a wav file.
    """
    n_mels = 86
    frame_size = 2048
    hop_length = 1024
    windowsize = 2
    x, y, uids, time_bins = load_signal2spec(
        signal[0], SR, n_mels, frame_size, hop_length
    )
    dataset = window_data(x, y, uids, time_bins, windowsize)
    X = np.array(dataset["X"]).astype(np.float32) / 255
    X = X.reshape(X.shape[0], 1, X.shape[1], X.shape[2])
    Y = np.array(dataset["Y"]).astype(np.longlong)
    UIDS = np.array(dataset["uids"])
    tweetynet_features = CustomAudioDataset(X, Y, UIDS)
    return tweetynet_features


# I imagine that this is broken. because the csv is correct


def predictions_to_kaleidoscope(
    predictions, SIGNAL, audio_dir, audio_file, manual_id, sample_rate
):
    """
    TweetyNet predictions to kaleidoscope

    Args:
        predictions: (Pandas DataFrame)
            Containing the starting time of each time bin and prediction from tweetynet.

        SIGNAL: (list of ints)
            - Samples from the audio clip.

        audio_dir: (str)
            - Directory of the audio clip.

        audio_file: (str)
            - Name of the audio clip file.

        manual_id: (str)
            - controls the name of the class written to the pandas dataframe.

        sample_rate: (int)
            - Sampling rate of the audio clip, usually 44100.

    Returns:
        Pandas Dataframe of automated labels for the audio clipmin Kaleidoscope format.
    """
    time_bin_seconds = predictions.iloc[1]["time_bins"]
    zero_sorted_filtered_df = predictions[predictions["pred"] == 0]
    offset = zero_sorted_filtered_df["time_bins"]
    duration = zero_sorted_filtered_df["time_bins"].diff().shift(-1)
    intermediary_df = pd.DataFrame({"OFFSET": offset, "DURATION": duration})
    kaleidoscope_df = []

    if offset.shape[0] == 0:
        raise BaseException("No birds were detected!!")

    if offset.iloc[0] != 0:
        kaleidoscope_df.append(
            pd.DataFrame({"OFFSET": [0], "DURATION": [offset.iloc[0]]})
        )
    kaleidoscope_df.append(
        intermediary_df[intermediary_df["DURATION"] >= 2 * time_bin_seconds]
    )

    if offset.iloc[-1] < predictions.iloc[-1]["time_bins"]:
        kaleidoscope_df.append(
            pd.DataFrame(
                {
                    "OFFSET": [offset.iloc[-1]],
                    "DURATION": [
                        predictions.iloc[-1]["time_bins"]
                        + predictions.iloc[1]["time_bins"]
                    ],
                }
            )
        )

    kaleidoscope_df = pd.concat(kaleidoscope_df)
    kaleidoscope_df = kaleidoscope_df.reset_index(drop=True)
    kaleidoscope_df["FOLDER"] = audio_dir
    kaleidoscope_df["IN FILE"] = audio_file
    kaleidoscope_df["CHANNEL"] = 0
    kaleidoscope_df["CLIP LENGTH"] = len(SIGNAL) / sample_rate
    kaleidoscope_df["SAMPLE RATE"] = sample_rate
    kaleidoscope_df["MANUAL ID"] = manual_id

    return kaleidoscope_df
