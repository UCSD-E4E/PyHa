import librosa
import numpy as np
import scipy.signal as scipy_signal
from scipy import ndimage

def perform_stft(SIGNAL, SAMPLE_RATE=44100):
    """
    Function that's main purpose is for reverse-engineering the birdnet FG-BG separation technique
    SIGNAL (list, np.ndarray)
        - Audio Signal the STFT is being performed on
    SAMPLE_RATE (int)
        - Nyquist sample rate to load the clip in as
    
    returns:
        - floating point value that is a ratio between the length of the clip and the length of the x-axis of the spectrogram
        - Numpy array representing the normalized magnitude stft of the clip from clip_path
    """
    
    assert isinstance(SIGNAL, list) or isinstance(SIGNAL, np.ndarray)
    assert isinstance(SAMPLE_RATE, int)
    assert SAMPLE_RATE > 0

    # parameters set by "Audio Based Bird Species Identification using Deep Learning Techniques"
    window_size = 512
    overlap_size = int(window_size*0.75)
    f,t,z = scipy_signal.stft(SIGNAL,fs=SAMPLE_RATE,window=np.hanning(window_size),noverlap=overlap_size,nperseg=window_size)
    # normalizing [0,1]
    z = np.abs(z)
    z = z/np.max(z)
    clip_stft_time_ratio = len(SIGNAL)/z.shape[1]
    return clip_stft_time_ratio, z

def calculate_medians(stft):
    """
    Function that computes the frequency and temporal medians of a 2D stft spectrogram.
    Used in binary thresholding for FG-BG separation
    stft (ndarray)
        - numpy array of spectrogram being processed 
    returns:
        - median values of each spectrogram column (time medians)
        - median values of each spectrogram row (frequency medians)
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

def binary_morph_opening(binary_stft, kernel_size=4):
    """
    Function that performs the binary morphological and followed by an or operation, commonly referred to
    as erosion and dilation respectively. Called an opening operation to people familiar with image processing

    binary_stft (ndarray)
        - foreground (high power) pixels represented as 1, background (lower power) represented as 0.
    kernel_shape (int)
        - defines the dimensions of the 2D binary morph kernel.
    returns:
        - binary stft image after a binary morphological opening operation determined by the kernel shape
    """

    assert isinstance(binary_stft, np.ndarray)
    assert isinstance(kernel_size, int)
    assert kernel_size > 0

    kernel = np.ones( (kernel_size, kernel_size), np.uint8)

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

def indicator_vector_processing(indicator_vector, kernel_size=4):
    """
    Function that performs additional dilations to the temporal indicator vector, expands on smaller relevant high-power sections

    indicator_vector (ndarray)
        - Numpy binary vector indicating high power temporal regions from the STFT
    kernel_size (int)
        - default: 4
        - determines the length of the kernel that performs the dilation (1, kernel_size)
    returns:
        - indicator vector that has been subjected to 2 binary morphological dilation (or) operations based on 1D kernel
    """
    assert isinstance(indicator_vector, np.ndarray)
    assert isinstance(kernel_size, int)
    assert kernel_size > 0

    kernel = np.ones((1, kernel_size), np.uint8)
    dilate = ndimage.binary_dilation(indicator_vector.reshape((1,indicator_vector.shape[0])), kernel, iterations=2)

    return dilate.astype(np.uint8)


def FG_BG_local_score_arr(SIGNAL, isolation_parameters, normalized_sample_rate):
    """
    Function that reverse-engineers that uses the BirdNET Signal-to-noise-ratio technique to build local score arrays out of audio clips

    SIGNAL (list, np.ndarray)
        - Audio Signal the STFT is being performed on
    SAMPLE_RATE (int)
        - Nyquist sampling rate at which to process the audio clip
    returns:
        - ratio between the length of the audio clip and the stft time axis
        - Numpy array of the local score array derived from median thresholding
    """
    assert isinstance(SIGNAL, list) or isinstance(SIGNAL, np.ndarray)
    assert isinstance(normalized_sample_rate, int)

    time_ratio, stft = perform_stft(SIGNAL, normalized_sample_rate)
    time_medians, freq_medians = calculate_medians(stft)
    binary_stft = binary_thresholding(stft, time_medians, freq_medians, isolation_parameters["power_threshold"])
    opened_binary_stft = binary_morph_opening(binary_stft, isolation_parameters["kernel_size"])
    temporal_indicator_vector = temporal_thresholding(opened_binary_stft)
    dilated_indicator_vector = indicator_vector_processing(temporal_indicator_vector, isolation_parameters["kernel_size"])

    return time_ratio, dilated_indicator_vector.reshape((dilated_indicator_vector.shape[1],))



# sanity check
#x = np.array([0,1,1,1,1,1,0]).reshape((1,7))
#print(x)
#print(indicator_vector_processing(x))