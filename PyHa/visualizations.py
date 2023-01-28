from .microfaune_package.microfaune.detection import RNNDetector
from .microfaune_package.microfaune import audio
from .tweetynet_package.tweetynet.TweetyNetModel import TweetyNetModel
from .tweetynet_package.tweetynet.Load_data_functions import compute_features
import torch
import librosa
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import pandas as pd
import scipy.signal as scipy_signal
import numpy as np
import seaborn as sns
from .IsoAutio import *

def checkVerbose(
    errorMessage, 
    verbose):
    """
    Adds the ability to toggle on/off all error messages and warnings.
    
    Args:
        errorMessage (string)
            - Error message to be displayed

        verbose (boolean)
            - Whether to display error messages
    """
    if(verbose):
        print(errorMessage)

def get_clip_name(clip_path):
    clip_path = clip_path.rstrip('/')

    last_sep=clip_path.rfind('/')+1
    if last_sep==-1:
        return clip_path

    return clip_path[last_sep:]

# Returns the signal and sample rate of the passed clip
def clip_info(clip_path, verbose):
    try:
        SIGNAL, SAMPLE_RATE = librosa.load(clip_path, sr=None, mono=True)
        SIGNAL = SIGNAL * 32768
    except BaseException:
        checkVerbose("Failure in loading " + clip_path, verbose)
        return
    # Downsample the audio if the sample rate > 44.1 kHz
    # Force everything into the human hearing range.
    try:
        if SAMPLE_RATE > 44100:
            rate_ratio = 44100 / SAMPLE_RATE
            SIGNAL = scipy_signal.resample(
                SIGNAL, int(len(SIGNAL) * rate_ratio))
            SAMPLE_RATE = 44100
    except BaseException:
        checkVerbose("Failure in downsampling" + clip_path, verbose)
        return

    return SIGNAL, SAMPLE_RATE

# Plot spectrogram from a given clip path
def draw_spectrogram(clip_path, samples, sample_rate, save_fig=False):
    duration = samples.shape[0] / sample_rate
    time_stamps = np.arange(0, duration, step=1)

    # general graph features
    fig, axs = plt.subplots(1)
    fig.set_figwidth(22)
    fig.set_figheight(5)
    fig.suptitle("Spectrogram for " + clip_path)

    # spectrogram plot
    axs.specgram(samples,
                 Fs=sample_rate,
                 NFFT=4096,
                 noverlap=2048,
                 window=np.hanning(4096),
                 cmap="ocean")

    axs.set_xlim(0, duration)
    axs.set_ylim(0, 22050)
    axs.grid(which='major', linestyle='-')

    # save graph
    if save_fig:
        plt.savefig(get_clip_name(clip_path) + "_Spectrogram_Graph.png")

# Get a list of colors
# cmap: colormap to get colors from
# iters: number of colors to generate,
#        given by (2**iters)-1
def get_colors(cmap,iters):
    colors = []
    # For each iteration, add the colors 
    # halfway between all the previously added colors
    for i in range(1,iters+1):
        offset=1/(2**i)
        colors.extend([cmap(n) for n in np.linspace(offset,1-offset,2**(i-1))])
    return colors

# Updates label_colors with the annotation and a new color
def get_label_color(annotation, label_colors):
    cmap = get_cmap('hsv')
    possible_colors = []

    if annotation not in list(label_colors.keys()):
        # If not enough colors, generate more
        # From all the generated colors that are not in label_colors,
        # update label_colors with first
        num_colors=3
        while not possible_colors:
            num_colors+=1
            colors = get_colors(cmap, num_colors)
            possible_colors = list(set(colors)-set(label_colors.values()))
        label_colors.update({annotation : possible_colors[0]})

def draw_labels(
        clip_name,
        df, 
        df_source, 
        samples, 
        sample_rate, 
        label_colors={}, 
        __label_colors = {},
        save_fig=False):
    """
    Draw the labeled sections from a given dataframe
    Args:
        df (dataframe)
            - Dataframe to get labels from

        df_source (str)
            - Where the dataframe came from, used for titling graph

        samples (numpy ndarray)
            - Signal from recording, used to determine duration
            
        sample_rate (int)
            - Sample rate of signal, used to determine duration

        label_colors (dict)
            - Specify what colors any given label should use,
            updates __label_colors with those values

        __label_colors (dict)
            - Because function default args are initialized at the program
            start and dictionaries are mutable, __label_colors serves as a 
            static variable so long as it is never overwritten, but modified
            instead

        save_fig (boolean)
            - Whether to save graph
    Returns:
        __label_colors (dict)
    """
    assert isinstance(df, pd.DataFrame)
    assert isinstance(df_source, str)
    assert isinstance(samples, np.ndarray)
    assert isinstance(sample_rate, int)
    assert isinstance(label_colors, dict)
    assert isinstance(__label_colors, dict)

    __label_colors.update(label_colors)

    # General graph features
    duration = samples.shape[0] / sample_rate
    fig, axs = plt.subplots(1)
    fig.set_figwidth(22)
    fig.set_figheight(5)
    fig.suptitle(df_source + " Labels")
    axs.set_xlim(0, duration)

    ndx = 0
    label_list = []
    for row in df.index:
        # Determine whether to add a new label and give it a color
        annotation = df["MANUAL ID"][row]
        if(annotation not in label_list):
            label_list.append(annotation)
            get_label_color(annotation, __label_colors)
            ndx=0
        else:
            ndx=1

        # Plot clip and its label as a rectangl
        minval = df["OFFSET"][row]
        maxval = df["OFFSET"][row] + \
                 df["DURATION"][row]

        axs.axvspan(xmin=minval, 
                    xmax=maxval,
                    facecolor=__label_colors[annotation],
                    alpha=0.5, 
                    label="_" * ndx + annotation)

    axs.legend()

    # save graph
    if save_fig:
        plt.savefig(clip_name + df_source + "_Label_Graph.png")

    return __label_colors


def line_graph(
        clip_name,
        rnn_scores,
        duration,
        normalize=False,
        log_scale=False,
        save_fig=False):
    """
    Args:
        clip_name (str)
        - Name of clip to draw scores for 

        rnn_scores (list of floats)
        - Scores to plot

        duration (float)
        - Length of signal, determines length of graph's x-axis

        normalize (boolean)
        - Whether to normalize scores

        log_scale (boolean)
        - Whether to draw with log scale

        save_fig (boolean)
        - Whether to save figure

    Returns:
        Nothing
    """
    assert isinstance(clip_name, str)
    assert isinstance(rnn_scores, list)
    assert [isinstance(entry, float) for entry in rnn_scores]
    assert isinstance(duration, float)
    assert isinstance(normalize, bool)
    assert isinstance(log_scale, bool)
    assert isinstance(save_fig, bool)

    # General graph features
    fig, axs = plt.subplots(1)
    fig.set_figwidth(22)
    fig.set_figheight(5)
    fig.suptitle("Local Scores for " + clip_name)
    axs.set_xlim(0, duration)

    # Helper variables
    num_scores=len(rnn_scores)
    step = duration / num_scores
    time_stamps = np.arange(0, duration, step)

    # Normalize scores
    if normalize:
        max_score = max(rnn_scores)
        rnn_scores = [score/max_score for score in rnn_scores]

    # Set vertical scale
    if log_scale:
        axs.set_yscale('log')
    else:
        axs.set_ylim(0,1)

    # Draw graph
    axs.plot(time_stamps, rnn_scores)

    # save graph
    if save_fig:
        plt.savefig(clip_name + "_Line_Graph.png")


def spectrogram_visualization(
        clip_path,
        rnn_scores=None,
        automated_df=None,
        manual_df=None,
        label_colors={},
        normalize_scores=False,
        log_scale=False,
        save_fig=False,
        return_colors=False,
        verbose=True):
    """
    Draws spectrogram for given clip, and the labels for any passed dataframes

    Args:
        clip_path (str)
        - Path to find clip that will be displayed
        scores 

        rnn_scores (list of floats)
        - Local scores for the clip determined by the RNN. Will be graphed
        if passed

        log_scale (boolean)
        - Whether to use log scale for line graph

        normalize_scores (boolean)
        - Whether to normalize the rnn scores

        automated_df (dataframe)
        - Dataframe with automated labels

        manual_df (dataframe)
        - Dataframe with manual labels

        label_colors (dict)
        - Specifies colors that labels should user

        save_fig (bool)
        - Whether to save resulting figures

        verbose (boolean)
        - Whether to display error messages

        Returns:
            label_colors if return_colors is true
    """
    assert isinstance(rnn_scores, list) or rnn_scores is None
    if rnn_scores is not None:
        assert [isinstance(item, float) for item in rnn_scores]
    assert isinstance(log_scale, bool)
    assert isinstance(normalize_scores, bool)
    assert isinstance(manual_df, pd.DataFrame) or manual_df is None
    assert isinstance(automated_df, pd.DataFrame) or automated_df is None
    assert isinstance(label_colors, dict)
    assert isinstance(save_fig, bool)
    assert isinstance(verbose, bool)



    # clip_name = os.path.basename(os.path.normpath(clip_path))
    clip_name = get_clip_name(clip_path)

    SIGNAL, SAMPLE_RATE = clip_info(clip_path, verbose)
    duration = len(SIGNAL)/SAMPLE_RATE

    #Draw spectrogram
    draw_spectrogram(clip_path, SIGNAL, SAMPLE_RATE, save_fig=save_fig)

    #Draw line graph of RNN scores
    if rnn_scores is not None:
        line_graph(
                clip_name,
                rnn_scores,
                duration,
                normalize=normalize_scores,
                log_scale=log_scale,
                save_fig=save_fig)

    # Draw automated labels
    if automated_df is not None:
        automated_df = automated_df[automated_df["IN FILE"]==clip_name]
        label_colors = draw_labels(
                clip_name,
                automated_df, 
                "Model", 
                SIGNAL, 
                SAMPLE_RATE, 
                label_colors=label_colors,
                save_fig=save_fig)

    # Draw manual labels
    if manual_df is not None:
        manual_df = manual_df[manual_df["IN FILE"]==clip_name]
        label_colors = draw_labels(
                clip_name,
                manual_df, 
                "Human", 
                SIGNAL, 
                SAMPLE_RATE, 
                label_colors=label_colors,
                save_fig=save_fig)

    if return_colors:
        return label_colors




def binary_visualization(automated_df, human_df, save_fig=False):
    """
    Function to visualize automated and human annotation scores across an audio
    clip in the form of multiple binary plots that represent different metrics.

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

    assert isinstance(automated_df,pd.DataFrame)
    assert isinstance(human_df,pd.DataFrame)
    assert isinstance(save_fig,bool)
    assert "CLIP LENGTH" in automated_df.columns
    assert "SAMPLE RATE" in automated_df.columns
    assert "OFFSET"      in automated_df.columns
    assert "DURATION"    in human_df.columns
    assert "DURATION"    in automated_df.columns


    duration = automated_df["CLIP LENGTH"].to_list()[0]
    SAMPLE_RATE = automated_df["SAMPLE RATE"].to_list()[0]
    # Initializing two arrays that will represent the
    # labels with respect to the audio clip
    human_arr = np.zeros((int(SAMPLE_RATE * duration),))
    bot_arr = np.zeros((int(SAMPLE_RATE * duration),))

    folder_name = automated_df["FOLDER"].to_list()[0]
    clip_path = automated_df["IN FILE"].to_list()[0]
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
    plt.title("Ground Truth for " + clip_path)
    plt.subplot(7, 1, 2)
    plt.plot(bot_arr)
    plt.title("Automated Label for " + clip_path)

    # Visualizing True Positives for the Automated Labeling
    plt.subplot(7, 1, 3)
    plt.plot(true_positive_arr)
    plt.title("True Positive for " + clip_path)

    # Visualizing False Negatives for the Automated Labeling
    plt.subplot(7, 1, 4)
    plt.plot(false_negative_arr)
    plt.title("False Negative for " + clip_path)

    plt.subplot(7, 1, 5)
    plt.plot(false_positive_arr)
    plt.title("False Positive for " + clip_path)

    plt.subplot(7, 1, 6)
    plt.plot(true_negative_arr)
    plt.title("True Negative for " + clip_path)

    plt.subplot(7, 1, 7)
    plt.plot(IoU_arr)
    plt.title("Union for " + clip_path)

    plt.tight_layout()
    if save_fig:
        x = clip_path.split(".")
        clip_path = x[0]
        plt.save_fig(clip_path + "_label_plot.png")

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
    assert isinstance(annotation_df,pd.DataFrame)
    assert "DURATION" in annotation_df.columns
    assert isinstance(n_bins,int)
    assert n_bins > 0
    assert min_length is None or isinstance(min_length,float) or isinstance(min_length,int)
    assert max_length is None or isinstance(max_length,float) or isinstance(max_length,int)
    assert isinstance(save_fig,bool)
    assert isinstance(title,str)
    assert isinstance(filename,str)

    # Create the initial histogram
    duration = annotation_df["DURATION"].to_list()
    fig, ax = plt.subplots()
    sns_hist = sns.histplot(
        data=duration,
        bins=n_bins,
        line_kws=dict(edgecolor="k", linewidth=2),
        stat="count",
        ax=ax)

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

def get_local_scores(clip_path, isolation_parameters, weight_path=None, verbose=True):

    assert isinstance(clip_path, str)
    assert isinstance(isolation_parameters, dict)
    assert isinstance(weight_path, str)
    assert isinstance(verbose, bool)

    # Reading in the audio file using librosa, converting to single channeled data with original sample rate
    # Reason for the factor for the signal is explained here: https://stackoverflow.com/questions/53462062/pyaudio-bytes-data-to-librosa-floating-point-time-series
    # Librosa scales down to [-1, 1], but the models require the range [-32768, 32767], so the multiplication is required
    SIGNAL, SAMPLE_RATE = clip_info(clip_path, True)
    local_scores = None
    if(isolation_parameters["model"] == 'microfaune'):
        # Initializing the detector to baseline or with retrained weights
        if weight_path is None:
            # Microfaune RNNDetector class
            detector = RNNDetector()
        else:
            try:
            # Initializing Microfaune hybrid CNN-RNN with new weights
                detector = RNNDetector(weight_path)
            except BaseException:
                checkVerbose("Error in weight path:" + weight_path, verbose)
                return
        try:
            # Computing Mel Spectrogram of the audio clip
            microfaune_features = detector.compute_features([SIGNAL])
            # Running the Mel Spectrogram through the RNN
            global_score, local_score = detector.predict(microfaune_features)
            local_scores = local_score[0].tolist()
        except BaseException:
                checkVerbose(
                        "Skipping " +
                        clip_path +

                        " due to error in Microfaune Prediction", verbose)
    elif (isolation_parameters["model"] == 'tweetynet'):
        # Initializing the detector to baseline or with retrained weights
            device = torch.device('cpu')
            detector = TweetyNetModel(2, (1, 86, 86), 86, device, binary = False)

            try:
                #need a function to convert a signal into a spectrogram and then window it
                tweetynet_features = compute_features([SIGNAL])
                predictions, local_score = detector.predict(tweetynet_features, model_weights=weight_path)
                local_scores = local_score[0].tolist()
            except BaseException:

                checkVerbose(
                        "Skipping " +
                        clip_path +
                        " due to error in TweetyNet Prediction", verbose)
                return None
    return local_scores
