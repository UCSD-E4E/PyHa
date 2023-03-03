from scipy.signal import butter,filtfilt
from scipy.fft import fft
import matplotlib.pyplot as plt
import numpy as np


def build_low_pass_filter(normalized_cutoff, order):
    """
    Scipy butterworth function wrapper that enables us to generate low pass filter coefficients
    to filter the high frequency noise observed in the CNN-RNN local score arrays.

    Args:
        normalized_cutoff (float)
            - Specifies what percentage of the frequency domain will be in the passband

        order (int)
            - Controls how many coefficients will be produced, the higher the order,
            the more effective the filtering will be, but that comes with a time tradeoff
        
        returns:
            - numerator and denominator coefficients of low pass filter (ndarray)
    """
    assert isinstance(normalized_cutoff, float)
    assert normalized_cutoff > 0.0 and normalized_cutoff < 1.0
    assert isinstance(order, int)
    
    b, a = butter(order, normalized_cutoff, btype='low', analog=False)
    return b, a

def filter_data(local_score_arr,b,a):
    """
    Scipy filtering function wrapper that guarantees that the input is the same length as
    the output after performing convolution on the local score array with the coefficients
    outputted by build_low_pass_filter

    Args:
        local_score_arr (list):
            - Audio timestep classifications that are the usual output of a CNN-RNN model

        b (list):
            - Numerator coefficients of low pass filter

        a (list):
            - Denominator coefficients of low pass filter

        returns:
            - Local score array that has been filtered by a low pass filter

    """
    assert isinstance(local_score_arr,np.ndarray) or isinstance(local_score_arr,list)
    assert isinstance(b, np.ndarray)
    assert isinstance(a, np.ndarray)

    return filtfilt(b,a,local_score_arr)

def local_score_filtering(local_score_arr, normalized_cutoff, order):
    """
    Wrapper function for build_low_pass_filter() and filter_data() functions because not everyone
    has a DSP background.

    Args:
        local_score_arr (list):
            - Audio timestep classifications that are the usual output of a CNN-RNN model

        normalized_cutoff (float):
            - Specifies what percentage of the frequency domain will be in the passband

        order (int):
            - Controls how many coefficients will be produced, the higher the order,
            the more effective the filtering will be, but that comes with a time tradeoff
        
        returns:
            - local score array that has been filtered by a low pass filter
        
    """
    assert isinstance(local_score_arr,np.ndarray) or isinstance(local_score_arr,list)
    assert isinstance(normalized_cutoff,float)
    assert normalized_cutoff > 0 and normalized_cutoff < 1
    b, a = build_low_pass_filter(normalized_cutoff=normalized_cutoff, order=order)

    return filter_data(b=b,a=a,local_score_arr=local_score_arr)

# helper function that can help people understand the frequency domain of their local score arrays.
#def local_score_freq_domain(local_scores,save_fig=False,fig_name=None, a=None, b=None):
#    if a is not None and b is not None:
#        local_scores = filter_data(local_scores,b,a)
#    
#    local_score_freq = fft(local_scores)
#    plt.subplot(2,1,1)
#    plt.plot(local_scores)
#    plt.title("Local Score Array")
#    plt.xlabel("20ms timestep count")
#    plt.ylabel("Timestep Score")
#    plt.subplot(2,1,2)
#    plt.plot(np.log(np.abs(local_score_freq[0:int(len(local_score_freq)/2)])))
#    plt.title("Local Score Array Frequency Representation")
#    plt.ylabel("Log Power")
#    plt.xlabel("FFT")
#    plt.grid()
#    plt.tight_layout()
#    if save_fig and fig_name is not None:
#        plt.savefig(fig_name)
#    else:
#        plt.show()
#    plt.clf()
