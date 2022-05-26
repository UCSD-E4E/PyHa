from .microfaune_package.microfaune.detection import RNNDetector
from .microfaune_package.microfaune import audio
import matplotlib.pyplot as plt
import pandas as pd
import scipy.signal as scipy_signal
import numpy as np
import seaborn as sns
from .IsoAutio import *

import torch
from .tweetynet_package.tweetynet.TweetyNetModel import TweetyNetModel
from .tweetynet_package.tweetynet.Load_data_functions import compute_features

def local_line_graph(
        local_scores,
        clip_name,
        sample_rate,
        samples,
        automated_df=None,
        premade_annotations_df=None,
        premade_annotations_label="Human Labels",
        log_scale=False,
        save_fig=False,
        normalize_local_scores=False):
    """
    Function that produces graphs with the local score plot and spectrogram of
    an audio clip. Now integrated with Pandas so you can visualize human and
    automated annotations.

    Args:
        local_scores (list of floats)
            - Local scores for the clip determined by the RNN.

        clip_name (string)
            - Directory of the clip.

        sample_rate (int)
            - Sample rate of the audio clip, usually 44100.

        samples (list of ints)
            - Each of the samples from the audio clip.

        automated_df (Dataframe)
            - Dataframe of automated labelling of the clip.

        premade_annotations_df (Dataframe)
            - Dataframe labels that have been made outside of the scope of this
              function.

        premade_annotations_label (string)
            - Descriptor of premade_annotations_df

        log_scale (boolean)
            - Whether the axis for local scores should be logarithmically
              scaled on the plot.

        save_fig (boolean)
            - Whether the clip should be saved in a directory as a png file.

    Returns:
        None
    """
    # Calculating the length of the audio clip
    duration = samples.shape[0] / sample_rate
    # Calculating the number of local scores outputted by Microfaune
    num_scores = len(local_scores)
    # the case for normalizing the local scores between [0,1]
    if normalize_local_scores:
        local_scores_max = max(local_scores)
        for ndx in range(num_scores):
            local_scores[ndx] = local_scores[ndx] / local_scores_max
    # Making sure that the local score of the x-axis are the same across the
    # spectrogram and the local score plot
    step = duration / num_scores
    time_stamps = np.arange(0, duration, step)

    if len(time_stamps) > len(local_scores):
        time_stamps = time_stamps[:-1]

    # general graph features
    fig, axs = plt.subplots(2)
    fig.set_figwidth(22)
    fig.set_figheight(10)
    fig.suptitle("Spectrogram and Local Scores for " + clip_name)
    # score line plot - top plot
    axs[0].plot(time_stamps, local_scores)
    #Look into this and their relation.
    axs[0].set_xlim(0, duration)
    if log_scale:
        axs[0].set_yscale('log')
    else:
        axs[0].set_ylim(0, 1)
    axs[0].grid(which='major', linestyle='-')
    # Adding in the optional automated labels from a Pandas DataFrame
    # if automated_df is not None:
    if not automated_df.empty:
        ndx = 0
        for row in automated_df.index:
            minval = automated_df["OFFSET"][row]
            maxval = automated_df["OFFSET"][row] + \
                automated_df["DURATION"][row]
            axs[0].axvspan(xmin=minval, xmax=maxval, facecolor="yellow",
                           alpha=0.4, label="_" * ndx + "Automated Labels")
            ndx += 1
    # Adding in the optional premade annotations from a Pandas DataFrame
    if not premade_annotations_df.empty:
        ndx = 0
        for row in premade_annotations_df.index:
            minval = premade_annotations_df["OFFSET"][row]
            maxval = premade_annotations_df["OFFSET"][row] + \
                premade_annotations_df["DURATION"][row]
            axs[0].axvspan(
                xmin=minval,
                xmax=maxval,
                facecolor="red",
                alpha=0.4,
                label="_" *
                ndx +
                premade_annotations_label)
            ndx += 1
    axs[0].legend()

    # spectrogram - bottom plot
    # Will require the input of a pandas dataframe
    Pxx, freqs, bins, im = axs[1].specgram(
                                            samples,
                                            Fs=sample_rate,
                                            NFFT=4096,
                                            noverlap=2048,
                                            window=np.hanning(4096),
                                            cmap="ocean")
    axs[1].set_xlim(0, duration)
    axs[1].set_ylim(0, 22050)
    axs[1].grid(which='major', linestyle='-')

    # save graph
    if save_fig:
        plt.savefig(clip_name + "_Local_Score_Graph.png")

# TODO rework function so that instead of generating the automated labels, it
# takes the automated_df as input same as it does with the manual dataframe.


def local_score_visualization(
        clip_path,
        ml_model="tweetynet",
        tweety_output=False,
        weight_path=None,
        premade_annotations_df=None,
        premade_annotations_label="Human Labels",
        automated_df=False,
        isolation_parameters=None,
        log_scale=False,
        save_fig=False,
        normalize_local_scores=False):
    """
    Wrapper function for the local_line_graph function for ease of use.
    Processes clip for local scores to be used for the local_line_graph
    function.

    Args:
        clip_path (string)
            - Path to an audio clip.

        weight_path (string)
            - Weights to be used for RNNDetector.

        premade_annotations_df (Dataframe)
            - Dataframe of annotations to be displayed that have been created
              outside of the function.
        premade_annotations_label (string)
            - String that serves as the descriptor for the premade_annotations
              dataframe.

        automated_df (Dataframe)
            - Whether the audio clip should be labelled by the isolate function
              and subsequently plotted.

        log_scale (boolean)
            - Whether the axis for local scores should be logarithmically
              scaled on the plot.

        save_fig (boolean)
            - Whether the plots should be saved in a directory as a png file.

    Returns:
        None
    """

    # Loading in the clip with Microfaune's built-in loading function
    try:
        SAMPLE_RATE, SIGNAL = audio.load_wav(clip_path)
    except BaseException:
        print("Failure in loading", clip_path)
        return
    # downsample the audio if the sample rate > 44.1 kHz
    # Force everything into the human hearing range.
    try:
        if SAMPLE_RATE > 44100:
            rate_ratio = 44100 / SAMPLE_RATE
            SIGNAL = scipy_signal.resample(
                SIGNAL, int(len(SIGNAL) * rate_ratio))
            SAMPLE_RATE = 44100
    except BaseException:
        print("Failure in downsampling", clip_path)
        return

    # Converting to Mono if Necessary
    if len(SIGNAL.shape) == 2:
        # averaging the two channels together
        SIGNAL = SIGNAL.sum(axis=1) / 2

    # Initializing the detector to baseline or with retrained weights
    if ml_model == "microfaune":
        if weight_path is None:
            # Microfaune RNNDetector class
            detector = RNNDetector()
        else:
            try:
                # Initializing Microfaune hybrid CNN-RNN with new weights
                detector = RNNDetector(weight_path)
            except BaseException:
                print("Error in weight path:", weight_path)
                return
    elif ml_model == "tweetynet":
        device = torch.device('cpu')
        detector = TweetyNetModel(2, (1, 86, 86), 86, device, binary = False)
    else:
        print("model \"{}\" does not exist".format(ml_model))
        return None
    try:
        if ml_model == "microfaune":
            # Computing Mel Spectrogram of the audio clip
            microfaune_features = detector.compute_features([SIGNAL])
            # Running the Mel Spectrogram through the RNN
            global_score, local_score = detector.predict(microfaune_features)
        elif ml_model == "tweetynet":
            #need a function to convert a signal into a spectrogram and then window it
            tweetynet_features = compute_features([SIGNAL])
            predictions, local_score = detector.predict(tweetynet_features, model_weights=weight_path)
        #if tweety_output:
            #    local_score = [np.array(predictions["pred"].values)]
    except BaseException:
        print(
            "Skipping " +
            clip_path +
            " due to error in Microfaune Prediction")

    # In the case where the user wants to look at automated bird labels
    if premade_annotations_df is None:
        premade_annotations_df = pd.DataFrame()
    if automated_df:
        if tweety_output:
                local_scores = [np.array(predictions["pred"].values)]
                automated_df = predictions_to_kaleidoscope(predictions, SIGNAL, "Doesn't", "Doesn't", "Matter", SAMPLE_RATE)
        else:
            automated_df = isolate(
                local_score[0],
                SIGNAL,
                SAMPLE_RATE,
                "Doesn't",
                "Matter",
                isolation_parameters,
                normalize_local_scores=normalize_local_scores)
    else:
        automated_df = pd.DataFrame()

    local_line_graph(
        local_score[0].tolist(),
        clip_path,
        SAMPLE_RATE,
        SIGNAL,
        automated_df,
        premade_annotations_df,
        premade_annotations_label=premade_annotations_label,
        log_scale=log_scale,
        save_fig=save_fig,
        normalize_local_scores=normalize_local_scores)

def local_score_visualization_tweetynet(
        clip_path,
        tweety_output=False,
        weight_path=None,
        premade_annotations_df=None,
        premade_annotations_label="Human Labels",
        automated_df=False,
        isolation_parameters=None,
        log_scale=False,
        save_fig=False,
        normalize_local_scores=False):
    """
    Wrapper function for the local_line_graph function for ease of use.
    Processes clip for local scores to be used for the local_line_graph
    function.

    Args:
        clip_path (string)
            - Path to an audio clip.

        tweety_output (boolean) # may want to incorporate into isolation parameters
            - True to use tweetynet's original output, or False to use 

        weight_path (string)
            - Weights to be used for RNNDetector.

        premade_annotations_df (Dataframe)
            - Dataframe of annotations to be displayed that have been created
              outside of the function.
        premade_annotations_label (string)
            - String that serves as the descriptor for the premade_annotations
              dataframe.

        automated_df (Dataframe)
            - Whether the audio clip should be labelled by the isolate function
              and subsequently plotted.

        log_scale (boolean)
            - Whether the axis for local scores should be logarithmically
              scaled on the plot.

        save_fig (boolean)
            - Whether the plots should be saved in a directory as a png file.

    Returns:
        None
    """

    # Loading in the clip with Microfaune's built-in loading function
    try:
        SAMPLE_RATE, SIGNAL = audio.load_wav(clip_path)
    except BaseException:
        print("Failure in loading", clip_path)
        return
    # downsample the audio if the sample rate > 44.1 kHz
    # Force everything into the human hearing range.
    try:
        if SAMPLE_RATE > 44100:
            rate_ratio = 44100 / SAMPLE_RATE
            SIGNAL = scipy_signal.resample(
                SIGNAL, int(len(SIGNAL) * rate_ratio))
            SAMPLE_RATE = 44100
    except BaseException:
        print("Failure in downsampling", clip_path)
        return

    # Converting to Mono if Necessary
    if len(SIGNAL.shape) == 2:
        # averaging the two channels together
        SIGNAL = SIGNAL.sum(axis=1) / 2

    # Initializing the detector to baseline or with retrained weights
    device = torch.device('cpu')
    detector = TweetyNetModel(2, (1, 86, 86), 86, device, binary = False)

    try:
        #need a function to convert a signal into a spectrogram and then window it
        tweetynet_features = compute_features([SIGNAL])
        predictions, local_score = detector.predict(tweetynet_features, model_weights=weight_path)
    except BaseException:
        print(
            "Skipping " +
            clip_path +
            " due to error in TweetyNet Prediction")
        return None

    # In the case where the user wants to look at automated bird labels
    if premade_annotations_df is None:
        premade_annotations_df = pd.DataFrame()
    if automated_df:
        if tweety_output:
            automated_df = predictions_to_kaleidoscope(predictions, SIGNAL, "Doesn't", "Doesn't", "Matter", SAMPLE_RATE)
        else:
            automated_df = isolate(
                local_score[0],
                SIGNAL,
                SAMPLE_RATE,
                "Doesn't",
                "Matter",
                isolation_parameters,
                normalize_local_scores=normalize_local_scores)
    else:
        automated_df = pd.DataFrame()

    local_line_graph(
        local_score[0].tolist(),
        clip_path,
        SAMPLE_RATE,
        SIGNAL,
        automated_df,
        premade_annotations_df,
        premade_annotations_label=premade_annotations_label,
        log_scale=log_scale,
        save_fig=save_fig,
        normalize_local_scores=normalize_local_scores)




def plot_bird_label_scores(automated_df, human_df, save_fig=False):
    """
    Function to visualize automated and human annotation scores across an audio
    clip.

    Args:
        automated_df (Dataframe)
            - Dataframe of automated labels for one clip

        human_df (Dataframe)
            - Dataframe of human labels for one clip.

        plot_fig (boolean)
            - Whether or not the efficiency statistics should be displayed.

        save_fig (boolean)
            - Whether or not the plot should be saved within a file.

    Returns:
        Dataframe with statistics comparing the automated and human labeling.
    """
    duration = automated_df["CLIP LENGTH"].to_list()[0]
    SAMPLE_RATE = automated_df["SAMPLE RATE"].to_list()[0]
    # Initializing two arrays that will represent the
    # labels with respect to the audio clip
    # print(SIGNAL.shape)
    human_arr = np.zeros((int(SAMPLE_RATE * duration),))
    bot_arr = np.zeros((int(SAMPLE_RATE * duration),))

    folder_name = automated_df["FOLDER"].to_list()[0]
    clip_name = automated_df["IN FILE"].to_list()[0]
    # Placing 1s wherever the au
    for row in automated_df.index:
        minval = int(round(automated_df["OFFSET"][row] * SAMPLE_RATE, 0))
        maxval = int(
            round(
                (automated_df["OFFSET"][row] +
                 automated_df["DURATION"][row]) *
                SAMPLE_RATE,
                0))
        bot_arr[minval:maxval] = 1
    for row in human_df.index:
        minval = int(round(human_df["OFFSET"][row] * SAMPLE_RATE, 0))
        maxval = int(
            round(
                (human_df["OFFSET"][row] +
                 human_df["DURATION"][row]) *
                SAMPLE_RATE,
                0))
        human_arr[minval:maxval] = 1

    human_arr_flipped = 1 - human_arr
    bot_arr_flipped = 1 - bot_arr

    true_positive_arr = human_arr * bot_arr
    false_negative_arr = human_arr * bot_arr_flipped
    false_positive_arr = human_arr_flipped * bot_arr
    true_negative_arr = human_arr_flipped * bot_arr_flipped
    IoU_arr = human_arr + bot_arr
    IoU_arr[IoU_arr == 2] = 1

    plt.figure(figsize=(22, 10))
    plt.subplot(7, 1, 1)
    plt.plot(human_arr)
    plt.title("Ground Truth for " + clip_name)
    plt.subplot(7, 1, 2)
    plt.plot(bot_arr)
    plt.title("Automated Label for " + clip_name)

    # Visualizing True Positives for the Automated Labeling
    plt.subplot(7, 1, 3)
    plt.plot(true_positive_arr)
    plt.title("True Positive for " + clip_name)

    # Visualizing False Negatives for the Automated Labeling
    plt.subplot(7, 1, 4)
    plt.plot(false_negative_arr)
    plt.title("False Negative for " + clip_name)

    plt.subplot(7, 1, 5)
    plt.plot(false_positive_arr)
    plt.title("False Positive for " + clip_name)

    plt.subplot(7, 1, 6)
    plt.plot(true_negative_arr)
    plt.title("True Negative for " + clip_name)

    plt.subplot(7, 1, 7)
    plt.plot(IoU_arr)
    plt.title("Union for " + clip_name)

    plt.tight_layout()
    if save_fig:
        x = clip_name.split(".")
        clip_name = x[0]
        plt.save_fig(clip_name + "_label_plot.png")

def annotation_duration_histogram(
    annotation_df,
    n_bins = 6,
    min_length = None,
    max_length = None,
    save_fig = False,
    title = "Annotation Length Histogram",
    filename = "annotation_histogram.png"):
    """
    Function to build a histogram so a user can visually see the length of 
    the annotations they are working with. 

    Args:
        annotation_df (Dataframe)
            - Dataframe of automated or human labels

        n_bins (int)
            - number of histogram bins in the final histogram
            - default: 6
            
        min_length (int)
            - minimum length of the audio clip
            - default: 0s

        max_length (int)
            - maximum length of the audio clip
            - default: 60s

        save_fig (boolean)
            - Whether or not the histogram should be saved as a file.
            - default: False

        filename (string)
            - Name of the file to save the histogram to.
            - default: "annotation_histogram.png"

    Returns:
        Histogram of the length of the annotations.
    """
    # Create the initial histogram
    duration = annotation_df["DURATION"].to_list()
    sns_hist = sns.histplot(
        data=duration,
        bins=n_bins,
        line_kws=dict(edgecolor="k", linewidth=2),
        stat="count")

    # Modify the length of the x-axis as specified
    if max_length is not None and min_length is not None:
        if max_length < min_length:
            raise ValueError("max_length cannot be less than `min_length")
        plt.xlim(min_length, max_length)
    elif max_length is not None:
        plt.xlim(right=max_length)
    elif min_length is not None:
        plt.xlim(left=min_length)

    # Set title and the labels
    sns_hist.set_title(title)
    sns_hist.set(xlabel="Annotation Length (s)", ylabel = "Count")

    # Save the histogram if specified
    if save_fig: 
        sns_hist.get_figure().savefig(filename)