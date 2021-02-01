# Import Statements
from __future__ import division

from microfaune_package.microfaune.detection import RNNDetector
from microfaune_package.microfaune import audio
import matplotlib.pyplot as plt
import os
from pathlib import Path
import sys
import numpy as np
import pdb
import csv
import argparse
from scipy.io import wavfile
import scipy.signal as scipy_signal
import pandas as pd

# Gabriel's original moment-to-moment classification tool. Reworked to output
# a Pandas DataFrame.
def isolate(scores, samples, sample_rate, audio_dir, filename):
    # calculate original duration
    old_duration = len(samples) / sample_rate

    # create entry for audio clip
    entry = {'FOLDER'  : audio_dir,
             'IN FILE'    : filename,
             'CHANNEL' : 0,
             'CLIP LENGTH': old_duration,
             'SAMPLE RATE': sample_rate,
             'OFFSET'  : [],
             'MANUAL ID'  : []}

    # Variable to modulate when encapsulating this function.
    # treshold is 'thresh_mult' times above median score value
    thresh_mult = 2
    thresh = np.median(scores) * thresh_mult


    # how many samples one score represents
    # Scores meaning local scores
    samples_per_score = len(samples) // len(scores)

    # isolate samples that produce a score above thresh
    isolated_samples = np.empty(0, dtype=np.int16)
    prev_cap = 0        # sample idx of previously captured
    for i in range(len(scores)):
        # if a score hits or surpasses thresh, capture 1s on both sides of it
        if scores[i] >= thresh:
            # score_pos is the sample index that the score corresponds to
            score_pos = i * samples_per_score

            # upper and lower bound of captured call
            # sample rate is # of samples in 1 second: +-1 second
            lo_idx = max(0, score_pos - sample_rate)
            hi_idx = min(len(samples), score_pos + sample_rate)
            lo_time = lo_idx / sample_rate
            hi_time = hi_idx / sample_rate

            # calculate start and end stamps
            # create new sample if not overlapping or if first stamp
            if prev_cap < lo_idx or prev_cap == 0:
                new_stamp = [lo_time, hi_time]
                entry['OFFSET'].append(new_stamp)
                entry['MANUAL ID'].append(1)
            # extend same stamp if still overlapping
            else:
                entry['OFFSET'][-1][1] = hi_time

            # mark previously captured to prevent overlap collection
            lo_idx = max(prev_cap, lo_idx)
            prev_cap = hi_idx

            # add to isolated samples
            # sub-clip numpy array
            isolated_samples = np.append(isolated_samples,samples[lo_idx:hi_idx])


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


## Function that applies the moment to moment labeling system to a directory full of wav files.
def calc_local_scores(bird_dir,weight_path=None, Normalized_Sample_Rate = 44100):
    # init detector
    # Use Default Microfaune Detector
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
            new_entry = isolate(local_scores[0], SIGNAL, SAMPLE_RATE, bird_dir, audio_file)
            #print(new_entry)
            if annotations.empty == True:
                annotations = new_entry
            else:
                annotations = annotations.append(new_entry)
        except:
            print("Error in isolating bird calls from", audio_file)
            continue

    return annotations

# Function that produces graphs with the local score plot and spectrogram of an audio clip. Now integrated with Pandas so you can visualize human and automated annotations.
def local_line_graph(local_scores,clip_name, sample_rate,samples, automated_df=None, human_df=None, save_fig = False):
    # Calculating the length of the audio clip
    duration = samples.shape[0]/sample_rate
    # Calculating the number of local scores outputted by Microfaune
    num_scores = len(local_scores)

    ## Making sure that the local score of the x-axis are the same across the spectrogram and the local score plot
    step = duration / num_scores
    time_stamps = np.arange(0, duration, step)

    if len(time_stamps) > len(local_scores):
        time_stamps = time_stamps[:-1]

    # general graph features
    fig, axs = plt.subplots(2)
    fig.set_figwidth(22)
    fig.set_figheight(10)
    fig.suptitle("Spectrogram and Local Scores for "+clip_name)
    # score line plot - top plot
    axs[0].plot(time_stamps, local_scores)
    axs[0].set_xlim(0,duration)
    axs[0].set_ylim(0,1)
    axs[0].grid(which='major', linestyle='-')
    # Adding in the optional automated labels from a Pandas DataFrame
    if automated_df.empty == False:
        ndx = 0
        for row in automated_df.index:
            minval = automated_df["OFFSET"][row]
            maxval = automated_df["OFFSET"][row] + automated_df["DURATION"][row]
            axs[0].axvspan(xmin=minval,xmax=maxval,facecolor="yellow",alpha=0.4, label = "_"*ndx + "Automated Labels")
            ndx += 1
    # Adding in the optional human labels from a Pandas DataFrame
    if human_df.empty == False:
        ndx = 0
        for row in human_df.index:
            minval = human_df["OFFSET"][row]
            maxval = human_df["OFFSET"][row] + human_df["DURATION"][row]
            axs[0].axvspan(xmin=minval,xmax=maxval,facecolor="red",alpha=0.4, label = "_"*ndx + "Human Labels")
            ndx += 1
    axs[0].legend()

    # spectrogram - bottom plot
    # Will require the input of a pandas dataframe
    Pxx, freqs, bins, im = axs[1].specgram(samples, Fs=sample_rate,
            NFFT=4096, noverlap=2048,
            window=np.hanning(4096), cmap="ocean")
    axs[1].set_xlim(0,duration)
    axs[1].set_ylim(0,22050)
    axs[1].grid(which='major', linestyle='-')

    # save graph
    if save_fig:
        plt.savefig(clip_name + "_Local_Score_Graph.png")

# Wrapper function for the local_line_graph function for ease of use.
def local_score_visualization(clip_path,weight_path = None, human_df = None,automated_df = False, save_fig = False):

    # Loading in the clip with Microfaune's built-in loading function
    SAMPLE_RATE, SIGNAL = audio.load_wav(clip_path)
    # downsample the audio if the sample rate > 44.1 kHz
    # Force everything into the human hearing range.
    if SAMPLE_RATE > 44100:
        rate_ratio = 44100 / SAMPLE_RATE
        SIGNAL = scipy_signal.resample(SIGNAL, int(len(SIGNAL)*rate_ratio))
        SAMPLE_RATE = 44100
        # Converting to Mono if Necessary
    if len(SIGNAL.shape) == 2:
        SIGNAL = SIGNAL.sum(axis=1) / 2

    # Initializing the detector to baseline or with retrained weights
    if weight_path is None:
        detector = RNNDetector()
    else:
        detector = RNNDetector(weight_path)
    try:
        # Computing Mel Spectrogram of the audio clip
        microfaune_features = detector.compute_features([SIGNAL])
        # Running the Mel Spectrogram through the RNN
        global_score,local_score = detector.predict(microfaune_features)
    except:
        print("Error in " + clip_path + " Skipping.")

    # In the case where the user wants to look at automated bird labels
    if human_df is None:
        human_df = pd.DataFrame
    if automated_df == True:
        automated_df = isolate(local_score[0],SIGNAL, SAMPLE_RATE,"Doesn't","Matter")
    else:
        automated_df = pd.DataFrame()

    local_line_graph(local_score[0].tolist(),clip_path,SAMPLE_RATE,SIGNAL,automated_df,human_df, save_fig = save_fig)
def bird_label_scores(automated_df,human_df,plot_fig = False, save_fig = False):

    duration = automated_df["CLIP LENGTH"].to_list()[0]
    SAMPLE_RATE = automated_df["SAMPLE RATE"].to_list()[0]
    # Initializing two arrays that will represent the human labels and automated labels with respect to
    # the audio clip
    #print(SIGNAL.shape)
    human_arr = np.zeros((int(SAMPLE_RATE*duration),))
    bot_arr = np.zeros((int(SAMPLE_RATE*duration),))

    folder_name = automated_df["FOLDER"].to_list()[0]
    clip_name = automated_df["IN FILE"].to_list()[0]
    # Placing 1s wherever the au
    for row in automated_df.index:
        minval = int(round(automated_df["OFFSET"][row]*SAMPLE_RATE,0))
        maxval = int(round((automated_df["OFFSET"][row] + automated_df["DURATION"][row]) *SAMPLE_RATE,0))
        bot_arr[minval:maxval] = 1
    for row in human_df.index:
        minval = int(round(human_df["OFFSET"][row]*SAMPLE_RATE,0))
        maxval = int(round((human_df["OFFSET"][row] + human_df["DURATION"][row])*SAMPLE_RATE,0))
        human_arr[minval:maxval] = 1

    human_arr_flipped = 1 - human_arr
    bot_arr_flipped = 1 - bot_arr

    true_positive_arr = human_arr*bot_arr
    false_negative_arr = human_arr * bot_arr_flipped
    false_positive_arr = human_arr_flipped * bot_arr
    true_negative_arr = human_arr_flipped * bot_arr_flipped
    IoU_arr = human_arr + bot_arr
    IoU_arr[IoU_arr == 2] = 1

    true_positive_count = np.count_nonzero(true_positive_arr == 1)/SAMPLE_RATE
    false_negative_count = np.count_nonzero(false_negative_arr == 1)/SAMPLE_RATE
    false_positive_count = np.count_nonzero(false_positive_arr == 1)/SAMPLE_RATE
    true_negative_count = np.count_nonzero(true_negative_arr == 1)/SAMPLE_RATE
    union_count = np.count_nonzero(IoU_arr == 1)/SAMPLE_RATE

    # Calculating useful values related to tp,fn,fp,tn values

    # Precision = TP/(TP+FP)
    try:
        precision = true_positive_count/(true_positive_count + false_positive_count)


    # Recall = TP/(TP+FP)
        recall = true_positive_count/(true_positive_count + false_negative_count)

    # F1 = 2*(Recall*Precision)/(Recall + Precision)

        f1 = 2*(recall*precision)/(recall + precision)
        IoU = true_positive_count/union_count
    except:
        print("Error calculating statistics, likely due to zero division, setting values to zero")
        f1 = 0
        precision = 0
        recall = 0

    # Creating a Dictionary which will be turned into a Pandas Dataframe
    entry = {'FOLDER'  : folder_name,
             'IN FILE'    : clip_name,
             'TRUE POSITIVE' : true_positive_count,
             'FALSE POSITIVE': false_positive_count,
             'FALSE NEGATIVE'  : false_negative_count,
             'TRUE NEGATIVE'  : true_negative_count,
             'UNION' : union_count,
             'PRECISION' : precision,
             'RECALL' : recall,
             "F1" : f1,
             'IoU' : IoU}
    #print(entry)
    # Plotting the three arrays to visualize where
    if plot_fig == True:
        plt.figure(figsize=(22,10))
        plt.subplot(7,1,1)
        plt.plot(human_arr)
        plt.title("Ground Truth for " + clip_name)
        plt.subplot(7,1,2)
        plt.plot(bot_arr)
        plt.title("Automated Label for " + clip_name)

        #Visualizing True Positives for the Automated Labeling
        plt.subplot(7,1,3)
        plt.plot(true_positive_arr)
        plt.title("True Positive for " + clip_name)

        #Visualizing False Negatives for the Automated Labeling
        plt.subplot(7,1,4)
        plt.plot(false_negative_arr)
        plt.title("False Negative for " + clip_name)

        plt.subplot(7,1,5)
        plt.plot(false_positive_arr)
        plt.title("False Positive for " + clip_name)

        plt.subplot(7,1,6)
        plt.plot(true_negative_arr)
        plt.title("True Negative for " + clip_name)

        plt.subplot(7,1,7)
        plt.plot(IoU_arr)
        plt.title("Union for " + clip_name)

        plt.tight_layout()
        if save_fig == True:
            x = clip_name.split(".")
            clip_name = x[0]
            plt.save_fig(clip_name + "_label_plot.png")

    return pd.DataFrame(entry,index=[0])

# Function that will allow users to easily pass in two dataframes, and it will output statistics on them
# Will have to adjust the isolate function so that it adds a sampling rate onto the dataframes.
def automated_labeling_statistics(automated_df,manual_df):
    # Getting a list of clips
    clips = automated_df["IN FILE"].to_list()
    # Removing duplicates
    clips = list(dict.fromkeys(clips))
    # Initializing the returned dataframe
    statistics_df = pd.DataFrame()
    # Looping through each audio clip
    for clip in clips:
        clip_automated_df = automated_df[automated_df["IN FILE"] == clip]
        clip_manual_df = manual_df[manual_df["IN FILE"] == clip]
        #try:
        clip_stats_df = bird_label_scores(clip_automated_df,clip_manual_df)
        if statistics_df.empty:
            statistics_df = clip_stats_df
        else:
            statistics_df = statistics_df.append(clip_stats_df)
        #except:
        #    print("Something went wrong with: "+clip)
        #    continue
    return statistics_df

# Small function that takes in the statistics and outputs their global values
def global_dataset_statistics(statistics_df):
    tp_sum = statistics_df["TRUE POSITIVE"].sum()
    fp_sum = statistics_df["FALSE POSITIVE"].sum()
    fn_sum = statistics_df["FALSE NEGATIVE"].sum()
    tn_sum = statistics_df["TRUE NEGATIVE"].sum()
    union_sum = statistics_df["UNION"].sum()
    precision = tp_sum/(tp_sum + fp_sum)
    recall = tp_sum/(tp_sum + fn_sum)
    f1 = 2*(precision*recall)/(precision+recall)
    IoU = tp_sum/union_sum
    entry = {'PRECISION'  : round(precision,6),
             'RECALL'    : round(recall,6),
             'F1' : round(f1,6),
             'IoU' : round(IoU,6)}
    return pd.DataFrame.from_dict([entry])
