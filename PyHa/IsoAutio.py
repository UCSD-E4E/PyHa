#from PyHa.tweetynet_package.tweetynet.network import TweetyNet
from .microfaune_package.microfaune.detection import RNNDetector
from .microfaune_package.microfaune import audio
from .tweetynet_package.tweetynet.TweetyNetModel import TweetyNetModel
from .tweetynet_package.tweetynet.Load_data_functions import compute_features, predictions_to_kaleidoscope
import torch
import pandas as pd
import scipy.signal as scipy_signal
import numpy as np
import math
import os
from .birdnet_lite.analyze import analyze
from copy import deepcopy

def build_isolation_parameters_microfaune(
        technique,
        threshold_type,
        threshold_const,
        threshold_min=0,
        window_size=1.0,
        chunk_size=2.0):
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

    Returns:
        isolation_parameters (dict)
            - Python dictionary that controls how to go about isolating
              automated labels from audio.
    """
    isolation_parameters = {
        "technique": technique,
        "threshold_type": threshold_type,
        "threshold_const": threshold_const,
        "threshold_min": threshold_min,
        "window_size": window_size,
        "chunk_size": chunk_size
    }

    if window_size != 1.0 and technique != "steinberg":
        print('''Warning: window_size is dedicated to the steinberg isolation
        technique. Won't affect current technique.''')
    if chunk_size != 2.0 and technique != "chunk":
        print('''Warning: chunk_size is dedicated to the chunk technique.
        Won't affect current technique.''')

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

    # normalize the local scores so that the max value is 1.
    if normalize_local_scores:
        local_scores_max = max(local_scores)
        for ndx in range(len(local_scores)):
            local_scores[ndx] = local_scores[ndx] / local_scores_max
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
    score array output of a neural network and lump local scores together in a
    way to produce automated labels based on a class across an audio clip.

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

    # isolate samples that produce a score above thresh
    isolated_samples = np.empty(0, dtype=np.int16)
    prev_cap = 0        # sample idx of previously captured
    for i in range(len(local_scores)):
        # if a score hits or surpasses thresh, capture 1s on both sides of it
        if (local_scores[i] >= thresh and
                local_scores[i] >= isolation_parameters["threshold_min"]):
            # score_pos is the sample index that the score corresponds to
            score_pos = i * samples_per_score

            # upper and lower bound of captured call
            # sample rate is # of samples in 1 second: +-1 second
            lo_idx = max(
                0,
                score_pos - int(isolation_parameters["window_size"]
                                / 2 * SAMPLE_RATE))
            hi_idx = min(
                len(SIGNAL),
                score_pos + int(isolation_parameters["window_size"]
                                / 2 * SAMPLE_RATE))
            lo_time = lo_idx / SAMPLE_RATE
            hi_time = hi_idx / SAMPLE_RATE

            # calculate start and end stamps
            # create new sample if not overlapping or if first stamp
            if prev_cap < lo_idx or prev_cap == 0:
                # New label
                new_stamp = [lo_time, hi_time]
                # TODO make it so that here we get the duration
                entry['OFFSET'].append(new_stamp)
                entry['MANUAL ID'].append(manual_id)
            # extend same stamp if still overlapping
            else:
                entry['OFFSET'][-1][1] = hi_time

            # mark previously captured to prevent overlap collection
            lo_idx = max(prev_cap, lo_idx)
            prev_cap = hi_idx

            # add to isolated samples
            # sub-clip numpy array
            isolated_samples = np.append(
                isolated_samples, SIGNAL[lo_idx:hi_idx])
    entry = pd.DataFrame.from_dict(entry)
    # TODO, when you go through the process of rebuilding this isolate function
    # as a potential optimization problem
    # rework the algorithm so that it builds the dataframe correctly to save
    # time.

    OFFSET = entry['OFFSET'].str[0]
    DURATION = entry['OFFSET'].str[1]
    DURATION = DURATION - OFFSET
    # Adding a new "DURATION" Column
    # Making compatible with Kaleidoscope
    entry.insert(6, "DURATION", DURATION)
    entry["OFFSET"] = OFFSET
    return entry


def simple_isolate(
        local_scores,
        SIGNAL,
        SAMPLE_RATE,
        audio_dir,
        filename,
        isolation_parameters,
        manual_id="bird"):
    """
    Technique suggested by Irina Tolkova and implemented by Jacob Ayers.
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
    # Calculating threshold that defines the creation of the automated labels
    # for an audio clip
    thresh = threshold(local_scores, isolation_parameters)

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

    annotation_start = 0
    call_start = 0
    call_stop = 0
    # looping through all of the local scores
    for ndx in range(len(local_scores)):
        current_score = local_scores[ndx]
        # Start of a new sequence.
        if (current_score >= thresh and
                annotation_start == 0 and
                current_score >= isolation_parameters["threshold_min"]):
            # signal a start of a new sequence.
            annotation_start = 1
            call_start = float(ndx * time_per_score)
            # print("Call Start",call_start)
        # End of a sequence
        elif current_score < thresh and annotation_start == 1:
            # signal the end of a sequence
            annotation_start = 0
            #
            call_end = float(ndx * time_per_score)
            # print("Call End",call_end)
            entry['OFFSET'].append(call_start)
            entry['DURATION'].append(call_end - call_start)
            entry['MANUAL ID'].append(manual_id)
            call_start = 0
            call_end = 0
        else:
            continue
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
    Technique created by Jacob Ayers. Attempts to produce automated annotations
    of an audio clip base on local score array outputs from a neural network.

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
    # configuring the threshold based on isolation parameters
    thresh = threshold(local_scores, isolation_parameters)

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

    # initializing variables used in master loop
    stack_counter = 0
    annotation_start = 0
    call_start = 0
    call_stop = 0
    # looping through every local score array value
    for ndx in range(len(local_scores)):
        # the case for the end of the local score array and the stack isn't
        # empty.
        if ndx == (len(local_scores) - 1) and stack_counter > 0:
            call_end = float(ndx * time_per_score)
            entry['OFFSET'].append(call_start)
            entry['DURATION'].append(call_end - call_start)
            entry['MANUAL ID'].append(manual_id)
        # pushing onto the stack whenever a sample is above the threshold
        if (local_scores[ndx] >= thresh and
                local_scores[ndx] >= isolation_parameters["threshold_min"]):
            # in case this is the start of a new annotation
            if stack_counter == 0:
                call_start = float(ndx * time_per_score)
                annotation_start = 1
            # increasing this stack counter will be referred to as "pushing"
            stack_counter = stack_counter + 1

        # when a score is below the threshold
        else:
            # the case where it is the end of an annotation
            if stack_counter == 0 and annotation_start == 1:
                # marking the end of a clip
                call_end = float(ndx * time_per_score)

                # adding annotation to dictionary containing all annotations
                entry['OFFSET'].append(call_start)
                entry['DURATION'].append(call_end - call_start)
                entry['MANUAL ID'].append(manual_id)

                # resetting for the next annotation
                call_start = 0
                call_end = 0
                annotation_start = 0
            # the case where the stack is empty and a new annotation hasn't
            # started, you just want to increment the index
            elif stack_counter == 0 and annotation_start == 0:
                continue
            # the case where we are below the threshold and the stack isn't
            # empty. Pop from the stack, which in this case means just
            # subtracting from the counter.
            else:
                stack_counter = stack_counter - 1
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
    Technique created by Jacob Ayers. Attempts to produce automated annotations
    of an audio clip based on local score array outputs from a neural network.

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
    # configuring the threshold based on isolation parameters
    thresh = threshold(local_scores, isolation_parameters)

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
    chunk_count = math.ceil(
        len(SIGNAL) / (isolation_parameters["chunk_size"] * SAMPLE_RATE))
    # calculating the number of local scores per second
    scores_per_second = len(local_scores) / old_duration
    # calculating the chunk size with respect to the local score array
    local_scores_per_chunk = scores_per_second * \
        isolation_parameters["chunk_size"]
    # looping through each chunk
    for ndx in range(chunk_count):
        # finding the start of a chunk
        chunk_start = ndx * local_scores_per_chunk
        # finding the end of a chunk
        chunk_end = min((ndx + 1) * local_scores_per_chunk, len(local_scores))
        # breaking up the local_score array into a chunk.
        chunk = local_scores[int(chunk_start):int(chunk_end)]
        # comparing the largest local score value to the threshold.
        # the case for if we label the chunk as an annotation
        if max(chunk) >= thresh and max(
                chunk) >= isolation_parameters["threshold_min"]:
            # Creating the time stamps for the annotation.
            # Requires converting from local score index to time in seconds.
            annotation_start = chunk_start / scores_per_second
            annotation_end = chunk_end / scores_per_second
            entry["OFFSET"].append(annotation_start)
            entry["DURATION"].append(annotation_end - annotation_start)

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

    # init detector
    # Use Default Microfaune Detector
    # TODO
    # Expand to neural networks beyond just microfaune
    #Add flag to work for creating tweetynet model.
    if weight_path is None:
        detector = RNNDetector()
    # Use Custom weights for Microfaune Detector
    else:
        detector = RNNDetector(weight_path)

    # init labels dataframe
    annotations = pd.DataFrame()
    # generate local scores for every bird file in chosen directory
    for audio_file in os.listdir(audio_dir):
        # skip directories
        if os.path.isdir(audio_dir + audio_file):
            continue

        # It is a bit awkward here to be relying on Microfaune's wave file
        # reading when we want to expand to other frameworks,
        # Likely want to change that in the future. Librosa had some troubles.

        # Reading in the wave audio files
        try:
            SAMPLE_RATE, SIGNAL = audio.load_wav(audio_dir + audio_file)
        except BaseException:
            print("Failed to load", audio_file)
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
        except:
            print("Failed to Downsample" + audio_file)
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
        except BaseException as e:
            print("Error in detection, skipping", audio_file)
            print(e)
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
        except BaseException as e:
            print("Error in isolating bird calls from", audio_file)
            print(e)
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

        # It is a bit awkward here to be relying on Microfaune's wave file
        # reading when we want to expand to other frameworks,
        # Likely want to change that in the future. Librosa had some troubles.

        # Reading in the wave audio files
        try:
            SAMPLE_RATE, SIGNAL = audio.load_wav(audio_dir + audio_file)
        except BaseException:
            print("Failed to load", audio_file)
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
        except:
            print("Failed to Downsample" + audio_file)

        # convert stereo to mono if needed
        # Might want to compare to just taking the first set of data.
        if len(SIGNAL.shape) == 2:
            SIGNAL = SIGNAL.sum(axis=1) / 2
        # detection
        try:
            tweetynet_features = compute_features([SIGNAL])
            predictions, local_scores = detector.predict(tweetynet_features, model_weights=weight_path, norm=normalize_local_scores)
        except BaseException as e:
            print("Error in detection, skipping", audio_file)
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
        except BaseException:
            print("Error in isolating bird calls from", audio_file)
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
        print("{model_name} model does not exist"\
            .format(model_name=isolation_parameters["model"]))
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
    kaleidoscope_df = [df["FOLDER"].str.rstrip("/\\"), df["IN FILE"], df["CHANNEL"],
                       df["OFFSET"], df["DURATION"], df["MANUAL ID"]]
    headers = ["FOLDER", "IN FILE", "CHANNEL",
               "OFFSET", "DURATION", "MANUAL ID"]

    kaleidoscope_df = pd.concat(kaleidoscope_df, axis=1, keys=headers)
    return kaleidoscope_df

# def annotation_combiner(df):
#    # Initializing the output Pandas dataframe
#    combined_annotation_df = pd.DataFrame()
    # looping through each annotation in the passed in dataframe
#    for annotation in df.index:
    # the case for the first iteration.
#        if combined_annotation_df.empty:
#            combined_annotation_df = df.loc[annotation,:]
#        else:
#            combined_annotation_df = combined_annotation_df.append()
# keeps track of how many annotations have been added to the current
# annotation.
#        annotation_chain_count = 0
# Boolean to keep track whether or not an annotation should be combined with
# the current annotation
#        chain_break = False
    # keeping track of where the current annotation starts
#        cur_offset = df.loc[annotation,"OFFSET"]
#        cur_duration = df.loc[annotation,"DURATION"]
#        start_offset = cur_offset+cur_duration
#        while chain_break == False:
#            annotation_chain_count = annotation_chain_count + 1
#            next_offset = df.loc[annotation+annotation_chain_count,"OFFSET"]
#            next_duration df.loc[annotation+annotation_chain_count,"DURATION"]
    # case in which an annotation overlaps
#            if next_offset <= start_offset:


#        annotation = annotation + annotation_chain_count - 1
