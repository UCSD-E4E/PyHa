from microfaune_package.microfaune.detection import RNNDetector
from microfaune_package.microfaune import audio
import pandas as pd
import scipy.signal as scipy_signal
import numpy as np
import math
import os


# function that encapsulates many different isolation techniques to the dictionary isolation_parameters
def isolate(local_scores, SIGNAL, SAMPLE_RATE, audio_dir, filename,isolation_parameters,manual_id = "bird", normalize_local_scores = False):

    # normalize the local scores so that the max value is 1.
    if normalize_local_scores == True:
        local_scores_max = max(local_scores)
        for ndx in range(len(local_scores)):
            local_scores[ndx] = local_scores[ndx]/local_scores_max
    # initializing the output dataframe that will contain labels across a single clip
    isolation_df = pd.DataFrame()

    # deciding which isolation technique to deploy for a given clip
    if isolation_parameters["technique"] == "simple":
        isolation_df = simple_isolate(local_scores, SIGNAL, SAMPLE_RATE, audio_dir, filename, isolation_parameters, manual_id = "bird")
    elif isolation_parameters["technique"] == "steinberg":
        isolation_df = steinberg_isolate(local_scores, SIGNAL, SAMPLE_RATE, audio_dir, filename,isolation_parameters, manual_id = "bird")
    elif isolation_parameters["technique"] == "stack":
        isolation_df = stack_isolate(local_scores, SIGNAL, SAMPLE_RATE, audio_dir, filename, isolation_parameters, manual_id = "bird")
    elif isolation_parameters["technique"] == "chunk":
        isolation_df = chunk_isolate(local_scores, SIGNAL, SAMPLE_RATE, audio_dir, filename, isolation_parameters, manual_id = "bird")

    return isolation_df

def steinberg_isolate(local_scores, SIGNAL, SAMPLE_RATE, audio_dir, filename,isolation_parameters,manual_id = "bird"):
    """
    Returns a dataframe of automated labels for the given audio clip. The automated labels determine intervals of bird noise as
    determined by the local scores given by an RNNDetector.

    Args:
        scores (list of floats) - Local scores of the audio clip as determined by RNNDetector.
        SIGNAL (list of ints) - Samples from the audio clip.
        SAMPLE_RATE (int) - Sampling rate of the audio clip, usually 44100.
        audio_dir (string) - Directory of the audio clip.
        filename (string) - Name of the audio clip file.

    Returns:
        Dataframe of automated labels for the audio clip.
    """
    # calculate original duration
    old_duration = len(SIGNAL) / SAMPLE_RATE

    # create entry for audio clip
    entry = {'FOLDER'  : audio_dir,
             'IN FILE'    : filename,
             'CHANNEL' : 0,
             'CLIP LENGTH': old_duration,
             'SAMPLE RATE': SAMPLE_RATE,
             'OFFSET'  : [],
             'MANUAL ID'  : []}

    # Variable to modulate when encapsulating this function.
    # treshold is 'thresh_mult' times above median score value
    # thresh_mult = 2
    if isolation_parameters["threshold_type"] == "median":
        thresh = np.median(local_scores) * isolation_parameters["threshold_const"]
    elif isolation_parameters["threshold_type"] == "mean" or isolation_parameters["threshold_type"] == "average":
        thresh = np.mean(local_scores) * isolation_parameters["threshold_const"]
    elif isolation_parameters["threshold_type"] == "standard deviation":
        thresh = np.mean(local_scores) + (np.std(local_scores) * isolation_parameters["threshold_const"])
    elif isolation_parameters["threshold_type"] == "pure":
        thresh = isolation_parameters["threshold_const"]
        if thresh < 0:
            print("Threshold is less than zero, setting to zero")
            thresh = 0
        elif thresh > 1:
            print("Threshold is greater than one, setting to one.")
            thresh = 1

    # how many samples one score represents
    # Scores meaning local scores
    samples_per_score = len(SIGNAL) // len(local_scores)

    # isolate samples that produce a score above thresh
    isolated_samples = np.empty(0, dtype=np.int16)
    prev_cap = 0        # sample idx of previously captured
    for i in range(len(local_scores)):
        # if a score hits or surpasses thresh, capture 1s on both sides of it
        if local_scores[i] >= thresh:
            # score_pos is the sample index that the score corresponds to
            score_pos = i * samples_per_score

            # upper and lower bound of captured call
            # sample rate is # of samples in 1 second: +-1 second
            lo_idx = max(0, score_pos - int(isolation_parameters["bi_directional_jump"]*SAMPLE_RATE))
            hi_idx = min(len(SIGNAL), score_pos + int(isolation_parameters["bi_directional_jump"]*SAMPLE_RATE))
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
            isolated_samples = np.append(isolated_samples,SIGNAL[lo_idx:hi_idx])


    entry = pd.DataFrame.from_dict(entry)
    # Making the necessary adjustments to the Pandas Dataframe so that it is compatible with Kaleidoscope.
    ## TODO, when you go through the process of rebuilding this isolate function as a potential optimization problem
    ## rework the algorithm so that it builds the dataframe correctly to save time.
    #print(entry["OFFSET"].tolist())
    # This solution is not system agnostic. The problem is that Gabriel stored the start and stop times as a list under the OFFSET column.
    OFFSET = entry['OFFSET'].str[0]
    DURATION = entry['OFFSET'].str[1]
    DURATION = DURATION - OFFSET
    # Adding a new "DURATION" Column
    # Making compatible with Kaleidoscope
    entry.insert(6,"DURATION",DURATION)
    entry["OFFSET"] = OFFSET
    return entry

def simple_isolate(local_scores, SIGNAL, SAMPLE_RATE, audio_dir, filename, isolation_parameters, manual_id = "bird"):

    #local_scores2 = local_scores
    #threshold = 2*np.median(local_scores)
    if isolation_parameters["threshold_type"] == "median":
        thresh = np.median(local_scores) * isolation_parameters["threshold_const"]
    elif isolation_parameters["threshold_type"] == "mean" or isolation_parameters["threshold_type"] == "average":
        thresh = np.mean(local_scores) * isolation_parameters["threshold_const"]
    elif isolation_parameters["threshold_type"] == "standard deviation":
        thresh = np.mean(local_scores) + (np.std(local_scores) * isolation_parameters["threshold_const"])
    elif isolation_parameters["threshold_type"] == "pure":
        thresh = isolation_parameters["threshold_const"]
        if thresh < 0:
            print("Threshold is less than zero, setting to zero")
            thresh = 0
        elif thresh > 1:
            print("Threshold is greater than one, setting to one.")
            thresh = 1

    # calculate original duration
    old_duration = len(SIGNAL) / SAMPLE_RATE

    entry = {'FOLDER'  : audio_dir,
             'IN FILE'    : filename,
             'CHANNEL' : 0,
             'CLIP LENGTH': old_duration,
             'SAMPLE RATE': SAMPLE_RATE,
             'OFFSET'  : [],
             'DURATION' : [],
             'MANUAL ID'  : []}

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
        if current_score >= thresh and annotation_start == 0:
            # signal a start of a new sequence.
            annotation_start = 1
            call_start = float(ndx*time_per_score)
            #print("Call Start",call_start)
        # End of a sequence
        elif current_score < thresh and annotation_start == 1:
            # signal the end of a sequence
            annotation_start = 0
            #
            call_end = float(ndx*time_per_score)
            #print("Call End",call_end)
            entry['OFFSET'].append(call_start)
            entry['DURATION'].append(call_end - call_start)
            entry['MANUAL ID'].append(manual_id)
            call_start = 0
            call_end = 0
        else:
            continue
    return pd.DataFrame.from_dict(entry)



def stack_isolate(local_scores, SIGNAL, SAMPLE_RATE, audio_dir, filename, isolation_parameters, manual_id = "bird"):

    # configuring the threshold based on isolation parameters
    if isolation_parameters["threshold_type"] == "median":
        thresh = np.median(local_scores) * isolation_parameters["threshold_const"]
    elif isolation_parameters["threshold_type"] == "mean" or isolation_parameters["threshold_type"] == "average":
        thresh = np.mean(local_scores) * isolation_parameters["threshold_const"]
    elif isolation_parameters["threshold_type"] == "standard deviation":
        thresh = np.mean(local_scores) + (np.std(local_scores) * isolation_parameters["threshold_const"])
    elif isolation_parameters["threshold_type"] == "pure":
        thresh = isolation_parameters["threshold_const"]
        if thresh < 0:
            print("Threshold is less than zero, setting to zero")
            thresh = 0
        elif thresh > 1:
            print("Threshold is greater than one, setting to one.")
            thresh = 1

    # calculate original duration
    old_duration = len(SIGNAL) / SAMPLE_RATE

    # initializing a dictionary that will be used to construct the output pandas dataframe.
    entry = {'FOLDER'  : audio_dir,
             'IN FILE'    : filename,
             'CHANNEL' : 0,
             'CLIP LENGTH': old_duration,
             'SAMPLE RATE': SAMPLE_RATE,
             'OFFSET'  : [],
             'DURATION' : [],
             'MANUAL ID'  : []}

    # how many samples one score represents
    # Scores meaning local scores
    samples_per_score = len(SIGNAL) // len(local_scores)
    # local_score * samples_per_score / sample_rate
    # constant that will be used to convert from local score indices to annotation start/stop values.
    time_per_score = samples_per_score / SAMPLE_RATE

    # initializing variables used in master loop
    stack_counter = 0
    annotation_start = 0
    call_start = 0
    call_stop = 0
    # looping through every local score array value
    for ndx in range(len(local_scores)):
        # the case for the end of the local score array and the stack isn't empty.
        if ndx == (len(local_scores) - 1) and stack_counter > 0:
            call_end = float(ndx*time_per_score)
            entry['OFFSET'].append(call_start)
            entry['DURATION'].append(call_end - call_start)
            entry['MANUAL ID'].append(manual_id)
        # pushing onto the stack whenever a sample is above the threshold
        if local_scores[ndx] >= thresh:
            # in case this is the start of a new annotation
            if stack_counter == 0:
                call_start = float(ndx*time_per_score)
                annotation_start = 1
            # increasing this stack counter will be referred to as "pushing"
            stack_counter = stack_counter + 1

        # when a score is below the treshold
        else:
            # the case where it is the end of an annotation
            if stack_counter == 0 and annotation_start == 1:
                # marking the end of a clip
                call_end = float(ndx*time_per_score)

                # adding annotation to dictionary containing all annotations
                entry['OFFSET'].append(call_start)
                entry['DURATION'].append(call_end - call_start)
                entry['MANUAL ID'].append(manual_id)

                # resetting for the next annotation
                call_start = 0
                call_end = 0
                annotation_start = 0
            # the case where the stack is empty and a new annotation hasn't started, you just want to increment the index
            elif stack_counter == 0 and annotation_start == 0:
                continue
            # the case where we are below the threshold and the stack isn't empty. Pop from the stack, which in this case means just subtracting from the counter.
            else:
                stack_counter = stack_counter - 1
    # returning pandas dataframe from dictionary constructed with all of the annotations
    return pd.DataFrame.from_dict(entry)

# Isolation technique that breaks down an audio clip into chunks based on a user-defined duration. It then goes through and finds the max local score
# in those chunks to decide whether or not a chunk contains the vocalization of interest.
# TODO
# Make it so that a user has the option of an overlap between the chunks.
# Make it so that a user can choose how many samples have to be above the threshold in order to consider a chunk to be good or not.
# Give the option to combine annotations that follow one-another.
def chunk_isolate(local_scores, SIGNAL, SAMPLE_RATE, audio_dir, filename, isolation_parameters, manual_id = "bird"):
    # configuring the threshold based on isolation parameters
    if isolation_parameters["threshold_type"] == "median":
        thresh = np.median(local_scores) * isolation_parameters["threshold_const"]
    elif isolation_parameters["threshold_type"] == "mean" or isolation_parameters["threshold_type"] == "average":
        thresh = np.mean(local_scores) * isolation_parameters["threshold_const"]
    elif isolation_parameters["threshold_type"] == "standard deviation":
        thresh = np.mean(local_scores) + (np.std(local_scores) * isolation_parameters["threshold_const"])
    elif isolation_parameters["threshold_type"] == "pure":
        thresh = isolation_parameters["threshold_const"]
        if thresh < 0:
            print("Threshold is less than zero, setting to zero")
            thresh = 0
        elif thresh > 1:
            print("Threshold is greater than one, setting to one.")
            thresh = 1

    # calculate original duration
    old_duration = len(SIGNAL) / SAMPLE_RATE

    # initializing the dictionary for the output pandas dataframe
    entry = {'FOLDER'  : audio_dir,
             'IN FILE'    : filename,
             'CHANNEL' : 0,
             'CLIP LENGTH': old_duration,
             'SAMPLE RATE': SAMPLE_RATE,
             'OFFSET'  : [],
             'DURATION' : [],
             'MANUAL ID'  : manual_id}

    # calculating the number of chunks that define an audio clip
    chunk_count = math.ceil(len(SIGNAL)/(isolation_parameters["chunk_size"]*SAMPLE_RATE))
    # calculating the number of local scores per second
    scores_per_second = len(local_scores)/old_duration
    # calculating the chunk size with respect to the local score array
    local_scores_per_chunk = scores_per_second * isolation_parameters["chunk_size"]
    # looping through each chunk
    for ndx in range(chunk_count):
        # finding the start of a chunk
        chunk_start = ndx*local_scores_per_chunk
        # finding the end of a chunk
        chunk_end = min((ndx+1)*local_scores_per_chunk,len(local_scores))
        # breaking up the local_score array into a chunk.
        chunk = local_scores[int(chunk_start):int(chunk_end)]
        # comparing the largest local score value to the treshold.
        # the case for if we label the chunk as an annotation
        if max(chunk) >= thresh:
            annotation_start = chunk_start/scores_per_second
            annotation_end = chunk_end/scores_per_second
            entry["OFFSET"].append(annotation_start)
            entry["DURATION"].append(annotation_end - annotation_start)

    return pd.DataFrame.from_dict(entry)



## Function that applies the moment to moment labeling system to a directory full of wav files.
def generate_automated_labels(bird_dir, isolation_parameters, weight_path=None, Normalized_Sample_Rate = 44100, normalize_local_scores = False):
    """
    Function that applies the moment to moment labeling system to a directory full of wav files.

    Args:
        bird_dir (string) - Directory with wav audio files.
        weight_path (string) - File path of weights to be used by the RNNDetector for determining presence of bird sounds.
        Normalized_Sample_Rate (int) - Sampling rate that the audio files should all be normalized to.

    Returns:
        Dataframe of automated labels for the audio clips in bird_dir.
    """

    # init detector
    # Use Default Microfaune Detector
    # TODO
    # Expand to neural networks beyond just microfaune
    if weight_path is None:
        detector = RNNDetector()
    # Use Custom weights for Microfaune Detector
    else:
        detector = RNNDetector(weight_path)

    # init labels dataframe
    annotations = pd.DataFrame()
    # generate local scores for every bird file in chosen directory
    for audio_file in os.listdir(bird_dir):
        # skip directories
        if os.path.isdir(bird_dir+audio_file): continue

        # read file
        SAMPLE_RATE, SIGNAL = audio.load_wav(bird_dir + audio_file)

        # downsample the audio if the sample rate > 44.1 kHz
        # Force everything into the human hearing range.
        # May consider reworking this function so that it upsamples as well
        if SAMPLE_RATE > Normalized_Sample_Rate:
            rate_ratio = Normalized_Sample_Rate / SAMPLE_RATE
            SIGNAL = scipy_signal.resample(
                    SIGNAL, int(len(SIGNAL)*rate_ratio))
            SAMPLE_RATE = Normalized_Sample_Rate
            # resample produces unreadable float32 array so convert back
            #SIGNAL = np.asarray(SIGNAL, dtype=np.int16)

        #print(SIGNAL.shape)
        # convert stereo to mono if needed
        # Might want to compare to just taking the first set of data.
        if len(SIGNAL.shape) == 2:
            SIGNAL = SIGNAL.sum(axis=1) / 2

        # detection
        try:
            microfaune_features = detector.compute_features([SIGNAL])
            global_score,local_scores = detector.predict(microfaune_features)
        except:
            print("Error in detection, skipping", audio_file)
            continue

        # get duration of clip
        duration = len(SIGNAL) / SAMPLE_RATE

        try:
            # Running moment to moment algorithm and appending to a master dataframe.
            new_entry = isolate(local_scores[0], SIGNAL, SAMPLE_RATE, bird_dir, audio_file, isolation_parameters, manual_id = "bird", normalize_local_scores=normalize_local_scores)
            #print(new_entry)
            if annotations.empty == True:
                annotations = new_entry
            else:
                annotations = annotations.append(new_entry)
        except:
            print("Error in isolating bird calls from", audio_file)
            continue
    # Quick fix to indexing
    annotations.reset_index(inplace = True, drop = True)
    return annotations
