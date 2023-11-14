from .birdnet_lite.analyze import analyze
from .microfaune_package.microfaune.detection import RNNDetector
from .microfaune_package.microfaune import audio
from .tweetynet_package.tweetynet.TweetyNetModel import TweetyNetModel
from .tweetynet_package.tweetynet.Load_data_functions import compute_features, predictions_to_kaleidoscope
import os
import torch
import librosa
import pandas as pd
import scipy.signal as scipy_signal
import numpy as np
from math import ceil
from copy import deepcopy
from sys import exit


def checkVerbose(
    errorMessage, 
    isolation_parameters):
    """
    Adds the ability to toggle on/off all error messages and warnings.

    Args:
        errorMessage (string)
            - Error message to be displayed

        isolation_parameters (dict)
            - Python Dictionary that controls the various label creation
              techniques.
    """
    assert isinstance(errorMessage,str)
    assert isinstance(isolation_parameters,dict)
    assert 'verbose' in isolation_parameters.keys()
    
    if(isolation_parameters['verbose']):
        print(errorMessage)

        

def build_isolation_parameters_microfaune(
        technique,
        threshold_type,
        threshold_const,
        threshold_min=0.0,
        window_size=1.0,
        chunk_size=2.0,
        verbose=True):
    """
    Wrapper function for all audio isolation techniques (Steinberg, Simple, 
    Stack, Chunk). Will call the respective function of each technique
    based on isolation_parameters "technique" key.

    Args:
        technique (string)
            - Chooses which of the four isolation techniques to deploy
            - options: "steinberg", "simple", "stack", "chunk"

        threshold_type (string)
            - Chooses how to derive a threshold from local score arrays
            - options: "mean", "median", "standard deviation", "pure"

        threshold_const (float)
            - Multiplier for "mean", "median", and "standard deviation". Acts
              as threshold for "pure"

        threshold_min (float)
            - Serves as a minimum barrier of entry for a local score to be
              considered a positive ID of a class.
            - default: 0

        window_size (float)
            - determines how many seconds around a positive ID local score
              to build an annotation.

        chunk_size (float)
            - determines the length of annotation when using "chunk"
              isolation technique

        verbose (boolean)
            - Whether to display error messages

    Returns:
        isolation_parameters (dict)
            - Python dictionary that controls how to go about isolating
              automated labels from audio.
    """
    technique_options = ["steinberg", "simple", "stack", "chunk"]
    assert isinstance(technique,str)
    assert technique in technique_options
    threshold_options = ["mean", "median", "standard deviation", "pure"]
    assert isinstance(threshold_type,str)
    assert threshold in threshold_options
    assert isinstance(threshold_const,float) or isinstance(threshold_const,int)
    if threshold_type == "pure":
        assert threshold_const > 0 and threshold_const < 1
    assert isinstance(threshold_min,float)
    assert threshold_min > 0 and threshold_min < 1
    assert isinstance(window_size,int) or isinstance(window_size,float)
    assert window_size > 0
    assert isinstance(chunk_size,int) or isinstance(chunk_size,float)
    assert chunk_size > 0
    

    isolation_parameters = {
        "technique": technique,
        "threshold_type": threshold_type,
        "threshold_const": threshold_const,
        "threshold_min": threshold_min,
        "chunk_size": chunk_size,
        "verbose": verbose
    }

    if window_size != 1.0 and technique != "steinberg":
        checkVerbose('''Warning: window_size is dedicated to the steinberg isolation
        technique. Won't affect current technique.''', isolation_parameters)
    if chunk_size != 2.0 and technique != "chunk":
        checkVerbose('''Warning: chunk_size is dedicated to the chunk technique.
        Won't affect current technique.''', isolation_parameters)

    return isolation_parameters


def isolate(
        local_scores,
        SIGNAL,
        SAMPLE_RATE,
        audio_dir,
        filename,
        isolation_parameters,
        manual_id="bird",
        normalize_local_scores=False):
    """
    Wrapper function for all of Microfaune's audio isolation techniques 
    (Steinberg, Simple, Stack, Chunk). Will call the respective function of each technique based on 
    isolation_parameters "technique" key.

    Args:
        local_scores (list of floats)
            - Local scores of the audio clip as determined by
              Microfaune Recurrent Neural Network.

        SIGNAL (list of ints)
            - Samples that make up the audio signal.

        SAMPLE_RATE (int)
            - Sampling rate of the audio clip, usually 44100.

        audio_dir (string)
            - Directory of the audio clip.

        filename (string)
            - Name of the audio clip file.

        isolation_parameters (dict)
            - Python Dictionary that controls the various label creation
              techniques.

    Returns:
        Dataframe of automated labels for the audio clip based on passed in
        isolation technique.
    """

    assert isinstance(local_scores,np.ndarray)
    assert isinstance(SIGNAL,np.ndarray)
    assert isinstance(SAMPLE_RATE,int)
    assert SAMPLE_RATE > 0
    assert isinstance(audio_dir,str)
    assert isinstance(filename,str)
    assert isinstance(isolation_parameters,dict)
    assert isinstance(manual_id,str)
    assert isinstance(normalize_local_scores,bool)
    assert "technique" in dict.fromkeys(isolation_parameters)
    potential_isolation_techniques = {"simple","steinberg","stack","chunk"}
    assert isolation_parameters["technique"] in potential_isolation_techniques

    # normalize the local scores so that the max value is 1.
    #if normalize_local_scores:
    #    local_scores_max = max(local_scores)
    #    for ndx in range(len(local_scores)):
    #        local_scores[ndx] = local_scores[ndx] / local_scores_max
    # initializing the output dataframe that will contain labels across a
    # single clip
    isolation_df = pd.DataFrame()

    # deciding which isolation technique to deploy for a given clip based on
    # the technique isolation parameter
    if isolation_parameters["technique"] == "simple":
        isolation_df = simple_isolate(
            local_scores,
            SIGNAL,
            SAMPLE_RATE,
            audio_dir,
            filename,
            isolation_parameters,
            manual_id=manual_id)
    elif isolation_parameters["technique"] == "steinberg":
        isolation_df = steinberg_isolate(
            local_scores,
            SIGNAL,
            SAMPLE_RATE,
            audio_dir,
            filename,
            isolation_parameters,
            manual_id=manual_id)
    elif isolation_parameters["technique"] == "stack":
        isolation_df = stack_isolate(
            local_scores,
            SIGNAL,
            SAMPLE_RATE,
            audio_dir,
            filename,
            isolation_parameters,
            manual_id=manual_id)
    elif isolation_parameters["technique"] == "chunk":
        isolation_df = chunk_isolate(
            local_scores,
            SIGNAL,
            SAMPLE_RATE,
            audio_dir,
            filename,
            isolation_parameters,
            manual_id=manual_id)

    return isolation_df


def threshold(local_scores, isolation_parameters):
    """
    Takes in the local score array output from a neural network and determines
    the threshold at which we determine a local score to be a positive
    ID of a class of interest. Most proof of concept work is dedicated to bird
    presence. Threshold is determined by "threshold_type" and "threshold_const"
    from the isolation_parameters dictionary.

    Args:
        local_scores (list of floats)
            - Local scores of the audio clip as determined by Microfaune
              Recurrent Neural Network.

        isolation_parameters (dict)
            - Python Dictionary that controls the various label creation
              techniques.

    Returns:
        thresh (float)
            - threshold at which the local scores in the local score array of
              an audio clip will be viewed as a positive ID.
    """

    assert isinstance(local_scores,np.ndarray)
    assert isinstance(isolation_parameters,dict)
    potential_threshold_types = {"median","mean","standard deviation","threshold_const"}
    assert isolation_parameters["threshold_type"] in potential_threshold_types


    if isolation_parameters["threshold_type"] == "median":
        thresh = np.median(local_scores) \
            * isolation_parameters["threshold_const"]
    elif (isolation_parameters["threshold_type"] == "mean" or
            isolation_parameters["threshold_type"] == "average"):
        thresh = np.mean(local_scores) \
            * isolation_parameters["threshold_const"]
    elif isolation_parameters["threshold_type"] == "standard deviation":
        thresh = np.mean(local_scores) + \
            (np.std(local_scores) * isolation_parameters["threshold_const"])
    elif isolation_parameters["threshold_type"] == "pure":
        thresh = isolation_parameters["threshold_const"]
        if thresh < 0:
            print("Threshold is less than zero, setting to zero")
            thresh = 0
        elif thresh > 1:
            print("Threshold is greater than one, setting to one.")
            thresh = 1
    return thresh


def steinberg_isolate(
        local_scores,
        SIGNAL,
        SAMPLE_RATE,
        audio_dir,
        filename,
        isolation_parameters,
        manual_id="bird"):
    """
    Technique developed by Gabriel Steinberg that attempts to take the local 
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
        local_scores (list of floats)
            - Local scores of the audio clip as determined by RNNDetector.

        SIGNAL (list of ints)
            - Samples from the audio clip.

        SAMPLE_RATE (int)
            - Sampling rate of the audio clip, usually 44100.

        audio_dir (string)
            - Directory of the audio clip.

        filename (string)
            - Name of the audio clip file.

        isolation_parameters (dict)
            - Python Dictionary that controls the various label creation
              techniques.

        manual_id (string)
            - controls the name of the class written to the pandas dataframe

    Returns:
        Pandas Dataframe of automated labels for the audio clip.
    """

    assert isinstance(local_scores,np.ndarray)
    assert isinstance(SIGNAL,np.ndarray)
    assert isinstance(SAMPLE_RATE,int)
    assert SAMPLE_RATE > 0
    assert isinstance(audio_dir,str)
    assert isinstance(filename,str)
    assert isinstance(isolation_parameters,dict)
    assert "window_size" in dict.fromkeys(isolation_parameters)

    # calculate original duration
    old_duration = len(SIGNAL) / SAMPLE_RATE
    # create entry for audio clip
    entry = {'FOLDER': audio_dir,
             'IN FILE': filename,
             'CHANNEL': 0,
             'CLIP LENGTH': old_duration,
             'SAMPLE RATE': SAMPLE_RATE,
             'OFFSET': [],
             'MANUAL ID': []}

    # calculating threshold that will define how labels are created in current
    # audio clip
    thresh = threshold(local_scores, isolation_parameters)
    # how many samples one local score represents
    samples_per_score = len(SIGNAL) // len(local_scores)
    
    # Calculating local scores that are at or above threshold
    thresh_scores = local_scores >= max(thresh, isolation_parameters["threshold_min"])
    
    # if statement to check if window size is smaller than time between two local scores
    # (as a safeguard against problems that can occur)
    if (int(isolation_parameters["window_size"] / 2 * SAMPLE_RATE) * 2 >= samples_per_score):
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
        starts = np.where(diff_scores == 1)[0] * samples_per_score - int(isolation_parameters["window_size"] / 2 * SAMPLE_RATE)
        ends = np.where(diff_scores == -1)[0] - 1
        ends = ends * samples_per_score + int(isolation_parameters["window_size"] / 2 * SAMPLE_RATE)
        
        # Does not continue if no annotations exist
        if (len(starts) == 0):
            return pd.DataFrame.from_dict({'FOLDER': [],       \
                                           'IN FILE': [],      \
                                           'CHANNEL': [],      \
                                           'CLIP LENGTH': [],  \
                                           'SAMPLE RATE': [],  \
                                           'OFFSET': [],       \
                                           'MANUAL ID': []})
        
        # Checks annotations for any overlap, and removes if so
        i = 0
        while True:
            if (i == len(ends) - 1):
                break
            if (starts[i + 1] < ends[i]):
                ends = np.delete(ends, i)
                starts = np.delete(starts, i + 1)
            else:
                i += 1
        
        # Correcting bounds
        starts[0] = max(0, starts[0])
        ends[-1] = min(len(SIGNAL), ends[-1])
        
        # Calculates offsets and durations from starts and ends
        entry['OFFSET'] = starts * 1.0 / SAMPLE_RATE
        entry['DURATION'] = ends - starts
        entry['DURATION'] = entry['DURATION'] * 1.0 / SAMPLE_RATE
        
        # Assigns manual ids to all annotations
        entry['MANUAL ID'] = np.full(entry['OFFSET'].shape, manual_id)
    else:
        # Simply assigns each 1 in thresh scores to be its own window if windows are too small
        entry['OFFSET'] = np.where(thresh_scores == 1)[0] * samples_per_score / SAMPLE_RATE - isolation_parameters["window_size"] / 2
        entry['DURATION'] = np.full(entry['OFFSET'].shape, isolation_parameters["window_size"] * 1.0)
        if (entry['OFFSET'] < 0):
            entry['OFFSET'][0] = 0
            entry['DURATION'][0] = isolation_parameters["window_size"] * 0.5
        entry['MANUAL ID'] = np.full(entry['OFFSET'].shape, manual_id)
    
    # returning pandas dataframe from dictionary constructed with all of the
    # annotations
    return pd.DataFrame.from_dict(entry)


def simple_isolate(
        local_scores,
        SIGNAL,
        SAMPLE_RATE,
        audio_dir,
        filename,
        isolation_parameters,
        manual_id="bird"):
    """
    Technique suggested by Irina Tolkova, implemented by Jacob Ayers. 
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

    Args:
        local_scores (list of floats)
            - Local scores of the audio clip as determined by RNNDetector.

        SIGNAL (list of ints)
            - Samples from the audio clip.

        SAMPLE_RATE (int)
            - Sampling rate of the audio clip, usually 44100.

        audio_dir (string)
            - Directory of the audio clip.

        filename (string)
            - Name of the audio clip file.

        isolation_parameters (dict)
            - Python Dictionary that controls the various label creation
              techniques.

        manual_id (string)
            - controls the name of the class written to the pandas dataframe

    Returns:
        Pandas Dataframe of automated labels for the audio clip.
    """

    assert isinstance(local_scores,np.ndarray)
    assert isinstance(SIGNAL,np.ndarray)
    assert isinstance(SAMPLE_RATE,int)
    assert SAMPLE_RATE > 0
    assert isinstance(audio_dir,str)
    assert isinstance(filename,str)
    assert isinstance(isolation_parameters,dict)
    assert isinstance(manual_id,str)

    # Calculating threshold that defines the creation of the automated labels
    # for an audio clip
    threshold_min = 0
    thresh = threshold(local_scores, isolation_parameters)
    if "threshold_min" in dict.fromkeys(isolation_parameters):
        threshold_min = isolation_parameters["threshold_min"]
    # calculate original duration
    old_duration = len(SIGNAL) / SAMPLE_RATE

    entry = {'FOLDER': audio_dir,
             'IN FILE': filename,
             'CHANNEL': 0,
             'CLIP LENGTH': old_duration,
             'SAMPLE RATE': SAMPLE_RATE,
             'OFFSET': [],
             'DURATION': [],
             'MANUAL ID': []}

    # how many samples one score represents
    # Scores meaning local scores
    samples_per_score = len(SIGNAL) // len(local_scores)
    # local_score * samples_per_score / sample_rate
    time_per_score = samples_per_score / SAMPLE_RATE

    # Calculating local scores that are at or above threshold
    thresh_scores = local_scores >= max(thresh, isolation_parameters["threshold_min"])
    
    # Set up to find the starts and ends of clips
    thresh_scores = np.append(thresh_scores, [0])
    rolled_scores = np.roll(thresh_scores, 1)
    rolled_scores[0] = 0
    
    # Logic for finding starts and ends given in steinberg isolate
    diff_scores = thresh_scores - rolled_scores
    
    # Calculates offsets and durations from difference
    entry['OFFSET'] = np.where(diff_scores == 1)[0] * time_per_score * 1.0
    entry['DURATION'] = np.where(diff_scores == -1)[0] * time_per_score - entry['OFFSET']
    
    # Assigns manual ids to all annotations
    entry['MANUAL ID'] = np.full(entry['OFFSET'].shape, manual_id)
    
    # returning pandas dataframe from dictionary constructed with all of the
    # annotations
    return pd.DataFrame.from_dict(entry)

def stack_isolate(
        local_scores,
        SIGNAL,
        SAMPLE_RATE,
        audio_dir,
        filename,
        isolation_parameters,
        manual_id="bird"):
    """
    Technique created by Jacob Ayers. Attempts to produce automated
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

    Args:
        local_scores (list of floats)
            - Local scores of the audio clip as determined by RNNDetector.

        SIGNAL (list of ints)
            - Samples from the audio clip.

        SAMPLE_RATE (int)
            - Sampling rate of the audio clip, usually 44100.

        audio_dir (string)
            - Directory of the audio clip.

        filename (string)
            - Name of the audio clip file.

        isolation_parameters (dict)
            - Python Dictionary that controls the various label creation
              techniques.

        manual_id (string)
            - controls the name of the class written to the pandas dataframe

    Returns:
        Pandas Dataframe of automated labels for the audio clip.
    """

    assert isinstance(local_scores,np.ndarray)
    assert isinstance(SIGNAL,np.ndarray)
    assert isinstance(SAMPLE_RATE,int)
    assert SAMPLE_RATE > 0
    assert isinstance(audio_dir,str)
    assert isinstance(filename,str)
    assert isinstance(isolation_parameters,dict)
    assert isinstance(manual_id,str)

    # configuring the threshold based on isolation parameters
    thresh = threshold(local_scores, isolation_parameters)
    threshold_min = 0
    if "threshold_min" in dict.fromkeys(isolation_parameters):
        threshold_min = isolation_parameters["threshold_min"]
    # calculate original duration
    old_duration = len(SIGNAL) / SAMPLE_RATE

    # initializing a dictionary that will be used to construct the output
    # pandas dataframe.
    entry = {'FOLDER': audio_dir,
             'IN FILE': filename,
             'CHANNEL': 0,
             'CLIP LENGTH': old_duration,
             'SAMPLE RATE': SAMPLE_RATE,
             'OFFSET': [],
             'DURATION': [],
             'MANUAL ID': []}

    # how many samples one score represents
    # Scores meaning local scores
    samples_per_score = len(SIGNAL) // len(local_scores)
    # local_score * samples_per_score / sample_rate
    # constant that will be used to convert from local score indices to
    # annotation start/stop values.
    time_per_score = samples_per_score / SAMPLE_RATE

    # Calculating local scores that are at or above threshold
    thresh_scores = local_scores >= max(thresh, isolation_parameters["threshold_min"])
    
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
        while (i < len(ends) - 1 and starts[i + 1] <= new_end):
            stack_counter -= starts[i + 1] - ends[i]
            stack_counter += ends[i + 1] - starts[i + 1]
            ends = np.delete(ends, i)
            starts = np.delete(starts, i + 1)
            new_end = ends[i] + stack_counter
        ends[i] = new_end
        i += 1
    
    # Addressing situation where end goes above max length of local scores
    ends[-1] = min(len(local_scores) - 1, ends[-1])

    # Deletes annotation if it starts on the
    # last local score
    if (starts[-1] == len(local_scores) - 1):
        starts = np.delete(starts, len(starts) - 1)
        ends = np.delete(ends, len(ends) - 1)
    
    # Calculates offsets and durations from starts/ends
    entry['OFFSET'] = starts * time_per_score
    entry['DURATION'] = ends - starts
    entry['DURATION'] = entry['DURATION'] * time_per_score
    
    # Assigns manual ids to all annotations
    entry['MANUAL ID'] = np.full(entry['OFFSET'].shape, manual_id)
    
    # returning pandas dataframe from dictionary constructed with all of the
    # annotations
    return pd.DataFrame.from_dict(entry)

# TODO
# Make it so that a user has the option of an overlap between the chunks.
# Make it so that a user can choose how many samples have to be above the
# threshold in order to consider a chunk to be good or not.
# Give the option to combine annotations that follow one-another.


def chunk_isolate(
        local_scores,
        SIGNAL,
        SAMPLE_RATE,
        audio_dir,
        filename,
        isolation_parameters,
        manual_id="bird"):
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

    Args:
        local_scores (list of floats)
            - Local scores of the audio clip as determined by RNNDetector.

        SIGNAL (list of ints)
            - Samples from the audio clip.

        SAMPLE_RATE (int)
            - Sampling rate of the audio clip, usually 44100.

        audio_dir (string)
            - Directory of the audio clip.

        filename (string)
            - Name of the audio clip file.

        isolation_parameters (dict)
            - Python Dictionary that controls the various label creation
              techniques.

        manual_id (string)
            - controls the name of the class written to the pandas dataframe

    Returns:
        Pandas Dataframe of automated labels for the audio clip.
    """

    assert isinstance(local_scores,np.ndarray)
    assert isinstance(SIGNAL,np.ndarray)
    assert isinstance(SAMPLE_RATE,int)
    assert SAMPLE_RATE > 0
    assert isinstance(audio_dir,str)
    assert isinstance(filename,str)
    assert isinstance(isolation_parameters,dict)
    assert isinstance(manual_id,str)

    # configuring the threshold based on isolation parameters
    thresh = threshold(local_scores, isolation_parameters)
    threshold_min = 0
    if "threshold_min" in dict.fromkeys(isolation_parameters):
        threshold_min = isolation_parameters["threshold_min"]
    # calculate original duration
    old_duration = len(SIGNAL) / SAMPLE_RATE

    # initializing the dictionary for the output pandas dataframe
    entry = {'FOLDER': audio_dir,
             'IN FILE': filename,
             'CHANNEL': 0,
             'CLIP LENGTH': old_duration,
             'SAMPLE RATE': SAMPLE_RATE,
             'OFFSET': [],
             'DURATION': [],
             'MANUAL ID': manual_id}

    # calculating the number of chunks that define an audio clip
    chunk_count = ceil(
        len(SIGNAL) / (isolation_parameters["chunk_size"] * SAMPLE_RATE))
    # calculating the number of local scores per second
    scores_per_second = len(local_scores) / old_duration
    # calculating the chunk size with respect to the local score array
    local_scores_per_chunk = scores_per_second * \
        isolation_parameters["chunk_size"]
    
    # Creates indices for starts of chunks using np.linspace
    # which creates even splits across a range, and then is
    # treated as int (rounds down)
    chunk_starts_float = np.linspace(start = 0, stop = chunk_count * local_scores_per_chunk, num = chunk_count, endpoint = False)
    chunk_starts = chunk_starts_float.copy().astype(int)
    
    # Deletes the first element of the array (0) to
    # avoid empty array
    chunk_starts = np.delete(chunk_starts, 0)
    
    # Creates chunked scores based on starts
    # Finds max value of each chunked array
    chunked_scores = np.array(list(map(np.amax, np.split(local_scores, chunk_starts))))
    
    # Finds which chunks are above threshold, and creates indices based on that
    thresh_scores = chunked_scores >= max(thresh, isolation_parameters["threshold_min"])
    chunk_indices = np.where(thresh_scores == 1)[0]
    
    # Assigns offset values based on float values of the starts
    entry['OFFSET'] = chunk_starts_float[chunk_indices] / scores_per_second
    
    # Creates durations based on float values of chunk starts
    all_chunk_durs = np.roll(chunk_starts_float, -1) / scores_per_second - chunk_starts_float / scores_per_second
    all_chunk_durs[-1] = len(local_scores) / scores_per_second - chunk_starts_float[-1] / scores_per_second
    entry['DURATION'] = all_chunk_durs[chunk_indices]

    # Assigns manual ids to all annotations
    entry['MANUAL ID'] = np.full(entry['OFFSET'].shape, manual_id)
    
    # returning pandas dataframe from dictionary constructed with all of the
    # annotations
    return pd.DataFrame.from_dict(entry)


def generate_automated_labels_birdnet(audio_dir, isolation_parameters):
    """
    Function that generates the bird labels for an audio file or across a
    folder using the BirdNet-Lite model

    Args:
        audio_dir (string)
            - Directory with wav audio files. Can be an individual file
              as well.

        isolation_parameters (dict)
            - Python Dictionary that controls the various label creation
              techniques. The keys it accepts are :
              - output_path (string)
                - Path to output folder. By default results are written into 
                  the input folder
                - default: None

              - lat (float)
                - Recording location latitude
                - default: -1 (ignore)

              - lon (float)
                - Recording location longitude
                - default: -1 (ignore)

              - week (int)
                - Week of the year when the recording was made
                - Values in [1, 48] (4 weeks per month) 
                - default: -1 (ignore)

              - overlap (float)
                - Overlap in seconds between extracted spectrograms
                - Values in [0.5, 1.5]
                - default: 0.0

              - sensitivity (float)
                - Detection sensitivity. Higher values result in higher sensitivity
                - Values in [0.5, 1.5] 
                - default: 1.0

              - min_conf (float)
                - Minimum confidence threshold
                - Values in [0.01, 0.99]
                - default: 0.1

              - custom_list (string)
                - Path to text file containing a list of species
                - default: '' (not used if not provided)

              - filetype (string)
                - Filetype of soundscape recordings
                - default: 'wav'

              - num_predictions (int)
                - Defines maximum number of written predictions in a given 3s segment
                - default: 10

              - write_to_csv (bool)
                - Set whether or not to write output to CSV
                - default: False

    Returns:
        Dataframe of automated labels for the audio clip(s) in audio_dir.
    """

    annotations = analyze(audio_path=audio_dir, **isolation_parameters)
    return annotations

def generate_automated_labels_microfaune(
        audio_dir,
        isolation_parameters,
        ml_model = "microfaune",
        manual_id="bird",
        weight_path=None,
        normalized_sample_rate=44100,
        normalize_local_scores=False):
    """
    Function that applies isolation technique on the local scores generated
    by the Microfaune mode across a folder of audio clips. It is determined
    by the isolation_parameters dictionary.

    Args:
        audio_dir (string)
            - Directory with wav audio files.

        isolation_parameters (dict)
            - Python Dictionary that controls the various label creation
              techniques.

        manual_id (string)
            - controls the name of the class written to the pandas dataframe

        weight_path (string)
            - File path of weights to be used by the RNNDetector for
              determining presence of bird sounds.

        normalized_sample_rate (int)
            - Sampling rate that the audio files should all be normalized to.

    Returns:
        Dataframe of automated labels for the audio clips in audio_dir.
    """


    assert isinstance(audio_dir,str)
    assert isinstance(isolation_parameters,dict)
    assert isinstance(manual_id,str)
    assert weight_path is None or isinstance(weight_path,str)
    assert isinstance(normalize_local_scores,bool)
    assert isinstance(normalized_sample_rate,int)
    assert normalized_sample_rate > 0
    

    if weight_path is None:
        detector = RNNDetector()
    # Use Custom weights for Microfaune Detector
    else:
        detector = RNNDetector(weight_path)
        # print("model \"{}\" does not exist".format(ml_model))
        # return None

    # init labels dataframe
    annotations = pd.DataFrame()
    # generate local scores for every bird file in chosen directory
    for audio_file in os.listdir(audio_dir):
        # skip directories
        if os.path.isdir(audio_dir + audio_file):
            continue

        # Reading in the audio files using librosa, converting to single channeled data with original sample rate
        # Reason for the factor for the signal is explained here: https://stackoverflow.com/questions/53462062/pyaudio-bytes-data-to-librosa-floating-point-time-series
        # Librosa scales down to [-1, 1], but the models require the range [-32768, 32767]
        try:
            SIGNAL, SAMPLE_RATE = librosa.load(audio_dir + audio_file, sr=None, mono=True)
            SIGNAL = SIGNAL * 32768
        except KeyboardInterrupt:
            exit("Keyboard interrupt")
        except BaseException:
            checkVerbose("Failed to load" + audio_file, isolation_parameters)
            continue

        # downsample the audio if the sample rate isn't 44.1 kHz
        # Force everything into the human hearing range.
        # May consider reworking this function so that it upsamples as well
        try:
            if SAMPLE_RATE != normalized_sample_rate:
                rate_ratio = normalized_sample_rate / SAMPLE_RATE
                SIGNAL = scipy_signal.resample(
                    SIGNAL, int(len(SIGNAL) * rate_ratio))
                SAMPLE_RATE = normalized_sample_rate
        except KeyboardInterrupt:
            exit("Keyboard interrupt")
        except:
            checkVerbose("Failed to Downsample" + audio_file, isolation_parameters)
            # resample produces unreadable float32 array so convert back
            # SIGNAL = np.asarray(SIGNAL, dtype=np.int16)
            
        # print(SIGNAL.shape)
        # convert stereo to mono if needed
        # Might want to compare to just taking the first set of data.
        if len(SIGNAL.shape) == 2:
            SIGNAL = SIGNAL.sum(axis=1) / 2
        # detection
        try:
            microfaune_features = detector.compute_features([SIGNAL])
            global_score, local_scores = detector.predict(microfaune_features)
        except KeyboardInterrupt:
            exit("Keyboard interrupt")
        except BaseException as e:
            checkVerbose(e, isolation_parameters)
            checkVerbose("Error in detection, skipping" + audio_file, isolation_parameters)
            continue
        
            
        # get duration of clip
        duration = len(SIGNAL) / SAMPLE_RATE

        try:
            # Running moment to moment algorithm and appending to a master
            # dataframe.
            new_entry = isolate(
                local_scores[0],
                SIGNAL,
                SAMPLE_RATE,
                audio_dir,
                audio_file,
                isolation_parameters,
                manual_id=manual_id,
                normalize_local_scores=normalize_local_scores)
            # print(new_entry)
            if annotations.empty:
                annotations = new_entry
            else:
                annotations = annotations.append(new_entry)
        except KeyboardInterrupt:
            exit("Keyboard interrupt")
        except BaseException as e:
            checkVerbose(e, isolation_parameters)
            checkVerbose("Error in isolating bird calls from" + audio_file, isolation_parameters)

            continue
    # Quick fix to indexing
    annotations.reset_index(inplace=True, drop=True)
    return annotations

def generate_automated_labels_tweetynet(
        audio_dir,
        isolation_parameters,
        manual_id="bird",
        weight_path=None,
        normalized_sample_rate=44100,
        normalize_local_scores=False):
    """
    Function that applies isolation technique determined by
    isolation_parameters dictionary across a folder of audio clips.

    Args:
        audio_dir (string)
            - Directory with wav audio files.

        isolation_parameters (dict)
            - Python Dictionary that controls the various label creation
              techniques. The only unique key to TweetyNET is tweety_output:
              - tweety_output (bool)
                - Set whether or not to use TweetyNET's original output. 
                - If set to `False`, TweetyNET will use the specified `technique` parameter.
                - default: True

        manual_id (string)
            - controls the name of the class written to the pandas dataframe.

        weight_path (string)
            - File path of weights to be used by TweetyNet for
              determining presence of bird sounds.

        normalized_sample_rate (int)
            - Sampling rate that the audio files should all be normalized to.

        normalize_local_scores (bool) # may want to incorporate into isolation parameters
            - Flag to normalize the local scores.

    Returns:
        Dataframe of automated labels for the audio clips in audio_dir.
    """

    assert isinstance(audio_dir,str)
    assert isinstance(isolation_parameters,dict)
    assert isinstance(manual_id,str)
    assert isinstance(weight_path,str) or weight_path is None
    assert isinstance(normalized_sample_rate,int)
    assert normalized_sample_rate > 0
    assert isinstance(normalize_local_scores,bool)

    
    # init detector
    device = torch.device('cpu')
    detector = TweetyNetModel(2, (1, 86, 86), 86, device)

    # init labels dataframe
    annotations = pd.DataFrame()
    # generate local scores for every bird file in chosen directory
    for audio_file in os.listdir(audio_dir):
        # skip directories
        if os.path.isdir(audio_dir + audio_file):
            continue

        # Reading in the audio files using librosa, converting to single channeled data with original sample rate
        # Reason for the factor for the signal is explained here: https://stackoverflow.com/questions/53462062/pyaudio-bytes-data-to-librosa-floating-point-time-series
        # Librosa scales down to [-1, 1], but the models require the range [-32768, 32767], so the multiplication is required
        try:
            SIGNAL, SAMPLE_RATE = librosa.load(audio_dir + audio_file, sr=None, mono=True)
            SIGNAL = SIGNAL * 32768
        except KeyboardInterrupt:
            exit("Keyboard interrupt")
        except BaseException:
            checkVerbose("Failed to load " + audio_file, isolation_parameters)
            continue
            
        # Resample the audio if it isn't the normalized sample rate
        try:
            if SAMPLE_RATE != normalized_sample_rate:
                rate_ratio = normalized_sample_rate / SAMPLE_RATE
                SIGNAL = scipy_signal.resample(
                    SIGNAL, int(len(SIGNAL) * rate_ratio))
                SAMPLE_RATE = normalized_sample_rate
        except KeyboardInterrupt:
            exit("Keyboard interrupt")
        except:
            checkVerbose("Failed to Downsample " + audio_file, isolation_parameters)
            
        # convert stereo to mono if needed
        # Might want to compare to just taking the first set of data.
        if len(SIGNAL.shape) == 2:
            SIGNAL = SIGNAL.sum(axis=1) / 2
        # detection
        try:
            tweetynet_features = compute_features([SIGNAL])
            predictions, local_scores = detector.predict(tweetynet_features, model_weights=weight_path, norm=normalize_local_scores)
        except KeyboardInterrupt:
            exit("Keyboard interrupt")
        except BaseException as e:
            checkVerbose("Error in detection, skipping " + audio_file, isolation_parameters)
            print(e)
            continue
           
        try:
            # Running moment to moment algorithm and appending to a master
            # dataframe. 
            if isolation_parameters["tweety_output"]:
                new_entry = predictions_to_kaleidoscope(
                    predictions, 
                    SIGNAL, 
                    audio_dir, 
                    audio_file, 
                    manual_id, 
                    SAMPLE_RATE)
            else:
                new_entry = isolate(
                    local_scores[0],
                    SIGNAL,
                    SAMPLE_RATE,
                    audio_dir,
                    audio_file,
                    isolation_parameters,
                    manual_id=manual_id,
                    normalize_local_scores=normalize_local_scores)
            # print(new_entry)
            if annotations.empty:
                annotations = new_entry
            else:
                annotations = annotations.append(new_entry)
        except KeyboardInterrupt:
            exit("Keyboard interrupt")
        except BaseException as e:
            checkVerbose("Error in isolating bird calls from " + audio_file, isolation_parameters)
            print(e)
            continue
    # Quick fix to indexing
    annotations.reset_index(inplace=True, drop=True)
    return annotations


def generate_automated_labels(
        audio_dir,
        isolation_parameters,
        manual_id="bird",
        weight_path=None,
        normalized_sample_rate=44100,
        normalize_local_scores=False):
    """
    Function that generates the bird labels across a folder of audio clips
    given the isolation_parameters

    Args:
        audio_dir (string)
            - Directory with wav audio files.

        isolation_parameters (dict)
            - Python Dictionary that controls the various label creation
              techniques.

        manual_id (string)
            - controls the name of the class written to the pandas dataframe

        weight_path (string)
            - File path of weights to be used by the model for
              determining presence of bird sounds.

        normalized_sample_rate (int)
            - Sampling rate that the audio files should all be normalized to.
              Used only for the Microfaune model.
        
        normalize_local_scores (bool)
            - Set whether or not to normalize the local scores.

    Returns:
        Dataframe of automated labels for the audio clips in audio_dir.
    """

    assert isinstance(audio_dir,str)
    assert isinstance(isolation_parameters,dict)
    assert isinstance(manual_id,str)
    assert weight_path is None or isinstance(weight_path,str)
    assert isinstance(normalized_sample_rate,int)
    assert normalized_sample_rate > 0
    assert isinstance(normalize_local_scores,bool)

    #try:
    if(isolation_parameters["model"] == 'microfaune'):
        annotations = generate_automated_labels_microfaune(
                        audio_dir=audio_dir,
                        isolation_parameters=isolation_parameters,
                        manual_id=manual_id,
                        weight_path=weight_path,
                        normalized_sample_rate=normalized_sample_rate,
                        normalize_local_scores=normalize_local_scores)
    elif(isolation_parameters["model"] == 'birdnet'):
        # We need to delete the some keys from the isolation_parameters
        # because we are unpacking the other arguments
        birdnet_parameters = deepcopy(isolation_parameters)
        keys_to_delete = ['model', 'technique', 'threshold_type',
            'threshold_const', 'chunk_size']
        for key in keys_to_delete:
            birdnet_parameters.pop(key, None)
        annotations = generate_automated_labels_birdnet(
                        audio_dir, birdnet_parameters)
    elif(isolation_parameters['model'] == 'tweetynet'):
        annotations = generate_automated_labels_tweetynet(
                        audio_dir=audio_dir,
                        isolation_parameters=isolation_parameters,
                        manual_id=manual_id,
                        weight_path=weight_path,
                        normalized_sample_rate=normalized_sample_rate,
                        normalize_local_scores=normalize_local_scores)
    else:
        # print("{model_name} model does not exist"\
        #     .format(model_name=isolation_parameters["model"]))
        checkVerbose("{model_name} model does not exist"\
        .format(model_name=isolation_parameters["model"]), isolation_parameters)
        annotations = None
    # except:
    #     print("Error. Check your isolation_parameters")
    #     return None
    return annotations

def kaleidoscope_conversion(df):
    """
    Function that strips away Pandas Dataframe columns necessary for PyHa
    package that aren't compatible with Kaleidoscope software

    Args:
        df (Pandas Dataframe)
            - Dataframe compatible with PyHa package whether it be human labels
              or automated labels.

    Returns:
        Pandas Dataframe compatible with Kaleidoscope.
    """

    assert isinstance(df,pd.DataFrame)
    assert "FOLDER" in df.columns and "IN FILE" in df.columns 
    assert "OFFSET" in df.columns and "DURATION" in df.columns
    assert "CHANNEL" in df.columns and "MANUAL ID" in df.columns

    kaleidoscope_df = [df["FOLDER"].str.rstrip("/\\"), df["IN FILE"], df["CHANNEL"],
                       df["OFFSET"], df["DURATION"], df["MANUAL ID"]]
    headers = ["FOLDER", "IN FILE", "CHANNEL",
               "OFFSET", "DURATION", "MANUAL ID"]

    kaleidoscope_df = pd.concat(kaleidoscope_df, axis=1, keys=headers)
    return kaleidoscope_df
