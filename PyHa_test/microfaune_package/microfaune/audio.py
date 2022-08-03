import librosa
import numpy as np
from scipy.io import wavfile
from scipy import signal
from math import *


def load_wav(path, decimate=None):
    """Load audio data.

        Parameters
        ----------
        path: str
            Wav file path.
        decimate: int
            If not None, downsampling by a factor of `decimate` value.

        Returns
        -------
        S: array-like
            Array of shape (Mel bands, time) containing the spectrogram.
    """
    fs, data = wavfile.read(path)

    data = data.astype(np.float32)

    if decimate is not None:
        data = signal.decimate(data, decimate)
        fs /= decimate

    return fs, data

def load_mp3(path):
    """Load audio data mp3 format.

        Parameters
        ----------
        path : str
            mp3 file path.

        Returns:
        -------
        data : array-like
            Audio data.
        fs : int
            Sampling frequency in Hz.
    """
    data, fs = librosa.core.load(path, sr=None)

    return fs, data


def load_audio(path):
    """Load audio data, mp3 or wav format

    Parameters
    ----------
        path : str
            audio file path.

    Returns:
    -------
        data : array-like
            Audio data.
        fs : int
            Sampling frequency in Hz.
    """
    if path[-4:] == ".wav":
        fs, data = load_wav(path)

    elif path[-4:] == ".mp3":
        fs, data = load_mp3(path)

    else:
        raise ValueError("Wrong file format, use mp3 or wav")

    return fs, data
  

def cut_audio(old_path, new_path, start, end):
    """
        Cut audio data to specific starting and end point and save it as a new wav file

        Parameters
        ----------
        old_path : str
            Original wav file path.
        new_path : str
            New wav file path.
        start : float
            Desired start time of new audio in seconds.
        end : float
            Desired end time of new audio in seconds.

    """
    fs, data = wavfile.read(old_path)
    indx_start = int(start*fs)
    indx_end = int(end*fs)+1
    wavfile.write(new_path,fs,data[indx_start:indx_end])

    return True


def create_spec(data, fs, n_mels=32, n_fft=2048, hop_len=1024):
    """Compute the Mel spectrogram from audio data.

        Parameters
        ----------
        data: array-like
            Audio data.
        fs: int
            Sampling frequency in Hz.
        n_mels: int
            Number of Mel bands to generate.
        n_fft: int
            Length of the FFT window.
        hop_len: int
            Number of samples between successive frames.

        Returns
        -------
        S: array-like
            Array of shape (Mel bands, time) containing the spectrogram.
    """
    # Calculate spectrogram
    S = librosa.feature.melspectrogram(
      data, sr=fs, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)
    S = S.astype(np.float32)

    # Convert power to dB
    S = librosa.power_to_db(S)

    return S


def wav2spc(wav_file, fs=44100, n_mels=40, n_fft=2048, hop_len=1024, duration=None):
    """Load a wav file and compute its MEL spectogram.

    Parameters
    ----------
    wav_file: str
        path to a wav file.
    fs: int
        Sampling frequency in Hz.
    n_mels: int
        Number of Mel bands to generate.
    n_fft: int
        Length of the FFT window.
    hop_len: int
        Number of samples between successive frames.
    duration: int
        Duration of the sound to consider (starting at the beginning)
        If None, no truncature is made

    Returns
    --------
    spec: array-like
        Array of shape (Mel bands, time) containing the spectrogram.
    """

    x_fs, x = load_wav(wav_file)
    if duration is not None:
        x = x[:int(x_fs * duration) + 1]

    if x_fs != fs:
        raise ValueError(f"wav file with wrong frequency {x_fs}: {wav_file}")
    spec = create_spec(x, fs, n_mels, n_fft, hop_len)
    return spec


def file2spec(path_file, scale_spec="linear", N_MELS=40, window_length=0.020, overlap=0.5, f_max=15000, duration=None):
    """ Compute spectrogram from a wav or mp3 file.

    Parameters
    ----------
        path_file : str
            path to a wav or mp3 file.
        scale_spec : str
            scale used to use to compute spectrogram, can be "linear" or "MEL".
        N_MELS : int
            Number of Mel bands to generate.
        window_length : float
            Length of the FFT window in seconds.
        overlap : float
            Overlap of the FFT windows.
        f_max : int
            Maximum frequency of the FFT domain.
        duration: int
            Duration of the sound to consider (starting at the beginning)
            If None, no cut is made

    Returns:
    -------
        spec : array-like
            Array of shape (frequency, time) containing the spectrogram.
        t : array-like
            Array of shape (time, 1) containing the time scale of spectogram.
            None if MEL scale is used
        f : array-like
            Array of shape (frequency, 1) containing the frequency scale of spectogram.
            None if MEL scale is used
    """

    # Load audio file
    x_fs, x = load_audio(path_file)

    shape = np.shape(x)
    # If the file contains several channel
    if len(shape) > 1:
        x = np.sum(x, axis=1)
    if duration is not None:
        x = x[:int(x_fs * duration) + 1]

    # Derive FFT parameters
    N_FFT = int(window_length * x_fs) + 1
    HOP_LEN = int(overlap * window_length * x_fs) + 1

    # Compute spectrograms
    if (scale_spec == "linear"):
        frequency_resolution = x_fs / N_FFT
        size_frequency_axis = 1 + floor(f_max / frequency_resolution)
        f, t, spec = signal.stft(x, fs=x_fs, nperseg=N_FFT, noverlap=HOP_LEN)
        # scipy returns a complex array, only the modulus is used in spectograms
        spec = np.abs(spec)
        # remove frequency above f_max
        if f[-1] > f_max:
            fsup_to_fmax = np.where(f > f_max)
            f = f[0:fsup_to_fmax[0][0] + 1]
            spec = spec[0:fsup_to_fmax[0][0] + 1, :]

    elif (scale_spec == "MEL"):
        # librosa library does not give access to t and f
        spec = librosa.feature.melspectrogram(x, sr=x_fs, n_fft=N_FFT, hop_length=HOP_LEN, n_mels=N_MELS)
        spec = np.abs(spec)
        t = None
        f = None

    else:
        raise ValueError(f"Wrong scale_spec parameter {scale_spec}, use linear or MEL")

    # Convert power to dB with the minimum as a reference, only positive dB
    spec = librosa.power_to_db(spec, ref=np.min(spec))

    return spec, t, f, x_fs
