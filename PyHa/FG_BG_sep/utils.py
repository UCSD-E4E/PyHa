import librosa
import numpy as np
import scipy.signal as scipy_signal
from scipy import ndimage

def perform_stft(clip_path, SAMPLE_RATE=48000):
    """
    Function that's main purpose is for reverse-engineering the birdnet FG-BG separation technique
    clip_path (string)
        - path to clip to perform stft on
    SAMPLE_RATE (int)
        - sample rate to load the clip in as
    
    returns:
        - Numpy array representing the normalized magnitude stft of the clip from clip_path
    """
    
    assert isinstance(clip_path,str)
    assert isinstance(SAMPLE_RATE, int)
    assert SAMPLE_RATE > 0

    clip, _ = librosa.load(clip_path, sr=SAMPLE_RATE)
    # parameters set by "Audio Based Bird Species Identification using Deep Learning Techniques"
    window_size = 512
    overlap_size = int(window_size*0.75)
    f,t,z = scipy_signal.stft(clip,fs=SAMPLE_RATE,window=np.hanning(window_size),noverlap=overlap_size,nperseg=window_size)
    # normalizing [0,1]
    z = np.abs(z)
    z = z/np.max(z)
    return z

def calculate_medians(stft):
    """
    Function that computes the frequency and temporal medians of a 2D stft spectrogram.
    Used in binary thresholding for FG-BG separation
    stft (ndarray)
        - numpy array of spectrogram being processed 
    """
    assert isinstance(stft,np.ndarray)

    freq_medians = np.median(stft,axis=1)
    time_medians = np.median(stft,axis=0)

    return time_medians, freq_medians

def binary_thresholding(stft, time_medians, freq_medians, multiplier_treshold=3.0):
    """
    Primary Foreground-background separation step used in BirdNET.
    stft (ndarray)
        - numpy array of spectrogram being processed
    time_medians (ndarray)
        - vector of medians wrt time of stft
    freq_medians (ndarray)
        - vector of medians wrt frequency of stft
    multiplier_threshold (int, float)
        - default = 3.0
        - a constant that is multiplied by both the time and frequency medians to decide
        whether or not a pixel is foreground or not
    returns:
        - binary ndarray same size as stft that contains 1's for foreground and 0's for background
    
    """

    assert isinstance(stft, np.ndarray)
    assert isinstance(time_medians, np.ndarray)
    assert isinstance(freq_medians, np.ndarray)
    assert isinstance(multiplier_treshold, float) or isinstance(multiplier_treshold, int)
    assert multiplier_treshold > 0

    binary_mask_time = np.zeros(stft.shape)
    binary_mask_freq = np.zeros(stft.shape)

    # building time mask
    for column in range(stft.shape[1]):
        binary_mask_time[:,column] = stft[:,column] >= multiplier_treshold*time_medians[column]

    # building frequency mask
    for row in range(stft.shape[0]):
        binary_mask_freq[row,:] = stft[row,:] >= multiplier_treshold*freq_medians[row]


    # performing a element-wise and operation
    return (binary_mask_freq*binary_mask_time).astype(np.uint8)

def binary_morph_opening(binary_stft, kernel_shape=(4,4)):
    """
    Function that performs the binary morphological and followed by an or operation, commonly referred to
    as erosion and dilation respectively. Called an opening operation to people familiar with image processing

    binary_stft (ndarray)
        - 
    kernel_shape (tuple)
        - 
    returns:
        - binary stft image after a binary morphological opening operation determined by the kernel shape
    """

    assert isinstance(binary_stft, np.ndarray)
    assert isinstance(kernel_shape, tuple)
    for val in kernel_shape:
        assert val > 0

    kernel = np.ones( kernel_shape, np.uint8)

    erode = ndimage.binary_erosion(binary_stft, kernel, iterations=1)
    dilate = ndimage.binary_dilation(erode, kernel, iterations=1)

    return dilate.astype(np.uint8)


def temporal_thresholding(opened_binary_stft):
    """
    Function that converts the 2D binary thresholded stft into a temporal indicator vector
    
    opened_binary_stft (ndarray)
        - binary foreground-background separated stft
    returns:
        - binary temporal indicator vector that signifies the temporal components with high power 
    """
    time_axis_sum = np.sum(opened_binary_stft, axis=0)
    indicator_vector = time_axis_sum > 0
    return indicator_vector.astype(np.uint8)
