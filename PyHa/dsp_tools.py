from scipy.signal import butter,filtfilt
import numpy as np
def build_low_pass_filter(normalized_cutoff, order):
    b, a = butter(order, normalized_cutoff, btype='low', analog=False)
    return b, a

def filter_data(data,b,a):
    return filtfilt(b,a,data)
