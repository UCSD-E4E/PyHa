import librosa
from scipy.signal import butter, lfilter, stft
import numpy as np


def generate_specgram(SIGNAL, SAMPLE_RATE):
    """
    Generates a magnitude stft spectrogram normalized [0,1] for the sake of template matching
    SIGNAL (ndarray)
        - Audio signal of which the stft is performed
    SAMPLE_RATE (int)
        - rate at which the audio signal was sampled at
    returns:
        - 2D numpy array representing the stft of the signal using a window length of 1024 and 50% overlap
    """
    assert isinstance(SIGNAL, np.ndarray)
    assert isinstance(SAMPLE_RATE, int)
    assert SAMPLE_RATE > 0

    window_len = 1024
    noverlap = 512
    nsperseg = 1024
    f, t, SIGNAL_stft = stft(SIGNAL, fs=SAMPLE_RATE, window=np.hanning(window_len), noverlap=noverlap, nperseg=nsperseg)
    SIGNAL_stft_mag = np.abs(SIGNAL_stft)
    output = SIGNAL_stft_mag/np.max(SIGNAL_stft_mag)
    return output

def butter_bandpass(lowcut, highcut, fs, order=5):
    """
    
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def filter(data, b, a):
    return lfilter(b, a, data)

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filter(b, a, data)
    return y


def template_matching_local_score_arr(SIGNAL, SAMPLE_RATE, template_spec, n, template_std_dev):
    assert isinstance(SIGNAL, np.ndarray)
    assert isinstance(SAMPLE_RATE, int)
    assert isinstance(template_spec, np.ndarray)
    signal_spec = generate_specgram(SIGNAL, SAMPLE_RATE)

    assert signal_spec.shape[0] == template_spec.shape[0]
    padded_signal = np.zeros((signal_spec.shape[0], signal_spec.shape[1] + 2 * template_spec.shape[1]))
    start_ndx = template_spec.shape[1]
    end_ndx = start_ndx + signal_spec.shape[1]
    padded_signal[0:signal_spec.shape[0],start_ndx:end_ndx] = signal_spec
    local_score_arr = np.zeros((signal_spec.shape[1],))
    local_score_ndx = 0
    for ndx in range(start_ndx, end_ndx):
        # handling even and odd template cases
        if template_spec.shape[1] % 2 == 1:
            active_window = padded_signal[:, (ndx-template_spec.shape[1]//2)-1:ndx+template_spec.shape[1]//2]
        else:
            active_window = padded_signal[:, (ndx-template_spec.shape[1]//2):ndx+template_spec.shape[1]//2]
        active_window_mean = np.mean(active_window)
        active_window -= active_window_mean
        active_window_std = np.std(active_window)
        std_dev_product = template_std_dev*active_window_std
        if std_dev_product == 0:
            local_score_arr[local_score_ndx] = 0
            local_score_ndx += 1
            continue
        #print("STD DEV PRODUCT", std_dev_product)
        product = active_window * template_spec * (1/std_dev_product)
        product_sum = np.sum(product)/n
        local_score_arr[local_score_ndx] = product_sum

        local_score_ndx += 1
    
    return local_score_arr
