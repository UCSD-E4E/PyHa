import matplotlib.pyplot as plt
import pylab
import numpy as np
import librosa.display

from microfaune import audio, labeling

def plot_spec(spec, t, f, fs, scale_spec="linear", window_length=0.2, overlap=0.5,
              plot_title="", fig_size=(20, 5), save_fig=False, save_path="spec.png", plot_main_frequencies=False):
    """ Plot spectrogram.ss
           Parameters
            ----------
            spec : array-like
                Array of shape (frequency, time) containing the spectrogram.
            t : array-like
                Array of shape (time, 1) containing the time scale of spectrogram.
                None if MEL scale is used
            f : array-like
                Array of shape (frequency, 1) containing the frequency scale of spectrogram.
                None if MEL scale is used
            fs : int
                Sampling frequency in Hz.
            scale_spec : str
                scale used to use to compute spectrogram, can be "linear" or "MEL".
            window_length : float
                Length of the FFT window in seconds.
            overlap : float
                Overlap of the FFT windows.
            plot_title : str
                Title of the plotted figure.
            fig_size : (int, int)
                Size of the plotted figure.
            save_fig : boolean
                True if the plot is saved, wrong otherwise.
            save_path : str
                Path of the saved figure.
            plot_main_frequencies : boolean
                True if median, minimum and maximum frequency are plotted on spectrograms, wrong otherwise.

            Returns:
            -------
            None
        """

    plt.set_cmap('inferno')
    pylab.rcParams['figure.figsize'] = fig_size
    plt.close()

    # Derive FFT parameters
    HOP_LEN = int(overlap * window_length * fs) + 1

    if scale_spec == "linear":
        plt.pcolormesh(t, f, spec)
        plt.xlabel('Time [sec]')
        plt.ylabel('Frequency [Hz]')

        if plot_main_frequencies:
            [freq_median, freq_min, freq_max] = find_most_used_frequencies(f, spec)

            plt.plot([0, t[-1]], [freq_min, freq_min], 'k')
            plt.plot([0, t[-1]], [freq_median, freq_median], 'w')
            plt.plot([0, t[-1]], [freq_max, freq_max], 'k')

    elif scale_spec == "MEL":
        librosa.display.specshow(spec, x_axis='time', y_axis='mel', sr=fs, hop_length=HOP_LEN)

    else:
        raise ValueError(f"Wrong scale_spec parameter {scale_spec}, use linear or MEL")

    plt.colorbar(format='%+2.0f dB')
    plt.title(plot_title)
    if save_fig:
        plt.savefig(save_path)
    plt.show()
    plt.close()

    return None


def find_most_used_frequencies(f, spec):
    """ Detect frequency used by the bird.

           Parameters
            ----------
            f : array-like
                Array of shape (frequency, 1) containing the frequency scale of spectrogram.
            spec : array-like
                Array of shape (frequency, time) containing the spectrogram.

            Returns:
            -------
            freq_median : int
                Median frequency of bird song in Hz.
            freq_min : int
                Minimum frequency of bird song in Hz (10% quantile).
            freq_max : int
                Maximum frequency of bird song in Hz (90% quantile).
        """

    # Removing lower frequencies often saturated
    f = f[4:]
    spec = spec[4:, :]

    quantile_95 = np.quantile(spec, 0.95)
    pixels_above_q95 = np.where(spec > quantile_95)
    freq_most_used = f[pixels_above_q95[0]]

    freq_median = round(np.quantile(freq_most_used, 0.50), 0)
    freq_min = round(np.quantile(freq_most_used, 0.10), 0)
    freq_max = round(np.quantile(freq_most_used, 0.90), 0)

    return [freq_median, freq_min, freq_max]


def plot_audio(fs, data):
    """
        Plot audio data.

        Parameters
        ----------
        data : array-like
            Audio data.
        fs : int
            Sampling frequency in Hz.

        Returns:
        -------
        None
    """
    plt.figure()
    sampled_points = len(data)
    time = [ti * 1. / fs for ti in range(sampled_points)]
    plt.plot(time, data)
    plt.xlabel('t [s]')
    plt.ylabel('P [W]')
    plt.show()
    plt.close()

    return None


def plot_charac_audio(json_file_path, audio_file_path):
    """ Plot the characteritic function derived on audio time scale from the labels in json file
                Parameters
                ----------
                json_file_path : str
                    Path of json file.
                audio_file_path : str
                    Path of audio file.

    """
    charac_func = labeling.charac_function_audio(json_file_path, audio_file_path)

    fs, data = audio.load_audio(audio_file_path)
    t_plot = np.array(range(len(data)))
    t_plot = t_plot / fs

    pylab.rcParams['figure.figsize'] = (20, 2)
    plt.close()

    plt.plot(t_plot, charac_func)
    plt.xlabel('time [s]')
    plt.ylabel('1 bird / 0 no bird')
    plt.title('Characteristic function on audio time scale')
    plt.show()

    return None


def plot_charac_spec(audio_file_path, window_length, overlap, charac_func_audio):
    """ Plot the characteristic function derived on spec time scale from the labels in json file
            Parameters
            ----------
            audio_file_path : str
                Path of audio file.
            window_length : float
                Length of the FFT window in seconds.
            overlap : float
                Overlap of the FFT windows.
            charac_func_audio : numpy array (nb bites in audio,1)
                Characteristic function derived in audio time scale, equal to 1 on labeled segments, 0 elsewhere

    """
    charac_func_spec = labeling.charac_function_spec(audio_file_path, window_length, overlap, charac_func_audio)

    fs, data = audio.load_audio(audio_file_path)
    t_plot = np.array(range(len(charac_func_spec)))
    t_plot = np.array(range(len(charac_func_spec))) * window_length * (1 - overlap)

    pylab.rcParams['figure.figsize'] = (20, 2)
    plt.close()

    plt.plot(t_plot, charac_func_spec)
    plt.xlabel('time [s]')
    plt.ylabel('1 bird / 0 no bird')
    plt.title('Characteristic function on spec time scale')
    plt.show()

    return None

def plot_charac_fs(fs, charac_func_fs):
    """ Plot the characteristic function with known sampling rate
        Parameters
        ----------
            fs: int
                Sampling frequency in Hz.
            charac_func_fs : numpy array (fs*duration, 1)
                Characteristic function derived with the desired sampling rate

    """
    duration = len(charac_func_fs) / fs
    t_fs = np.arange(0, duration, step=1./fs)

    plt.plot(t_fs, charac_func_fs)
    plt.xlabel('time [s]')
    plt.ylabel('1 bird / 0 no bird')
    plt.title('Characteristic function on with sampling rate ' + str(fs) + 'Hz')
    plt.show()

    return None