from .microfaune_package.microfaune.detection import RNNDetector
from .microfaune_package.microfaune import audio
from .tweetynet_package.tweetynet.TweetyNetModel import TweetyNetModel
from .tweetynet_package.tweetynet.Load_data_functions import compute_features
from .FG_BG_sep.utils import FG_BG_local_score_arr
from .template_matching.utils import filter, butter_bandpass, generate_specgram, template_matching_local_score_arr
import torch
import librosa
import matplotlib.pyplot as plt
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

def spectrogram_graph(
        clip_name,
        sample_rate,
        samples,
        automated_df=None,
        premade_annotations_df=None,
        premade_annotations_label="Human Labels",
        save_fig=False):
    """
    Function that produces graphs with the spectrogram of an audio
    clip. Now integrated with Pandas so you can visualize human and
    automated annotations.

    Args:
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

        save_fig (boolean)
            - Whether the clip should be saved in a directory as a png file.

    Returns:
        None
    """

    assert isinstance(clip_name,str)
    assert isinstance(sample_rate,int)
    assert isinstance(samples,np.ndarray)
    assert automated_df is None or isinstance(automated_df,pd.DataFrame)
    assert premade_annotations_df is None or isinstance(premade_annotations_df,pd.DataFrame)
    assert isinstance(premade_annotations_label,str)
    assert isinstance(save_fig,bool)

    # Calculating the length of the audio clip
    duration = samples.shape[0] / sample_rate
    time_stamps = np.arange(0, duration, step=1)

    # general graph features
    fig, axs = plt.subplots(1)
    fig.set_figwidth(22)
    fig.set_figheight(5)
    fig.suptitle("Spectrogram for " + clip_name)

    # spectrogram plot
    # Will require the input of a pandas dataframe
    Pxx, freqs, bins, im = axs.specgram(
                                            samples,
                                            Fs=sample_rate,
                                            NFFT=4096,
                                            noverlap=2048,
                                            window=np.hanning(4096),
                                            cmap="ocean")
    axs.set_xlim(0, duration)
    axs.set_ylim(0, 22050)
    axs.grid(which='major', linestyle='-')

    # if automated_df is not None:
    if not automated_df.empty:
        ndx = 0
        for row in automated_df.index:
            minval = automated_df["OFFSET"][row]
            maxval = automated_df["OFFSET"][row] + \
                automated_df["DURATION"][row]
            axs.axvspan(xmin=minval, xmax=maxval, facecolor="yellow",
                           alpha=0.4, label="_" * ndx + "Automated Labels")
            ndx += 1

    # Adding in the optional premade annotations from a Pandas DataFrame
    if not premade_annotations_df.empty:
        ndx = 0
        for row in premade_annotations_df.index:
            minval = premade_annotations_df["OFFSET"][row]
            maxval = premade_annotations_df["OFFSET"][row] + \
                premade_annotations_df["DURATION"][row]
            axs.axvspan(
                xmin=minval,
                xmax=maxval,
                facecolor="red",
                alpha=0.4,
                label="_" *
                ndx +
                premade_annotations_label)
            ndx += 1
    axs.legend()

    # save graph
    if save_fig:
        plt.savefig(clip_name + "_Local_Score_Graph.png")

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

        normalize_local_scores (boolean)
            - Whether the local scores will be forced to a range where the max local score is 1.
            All values / max_score

    Returns:
        None
    """

    assert isinstance(local_scores,list) or isinstance(local_scores, np.ndarray)
    assert isinstance(clip_name,str)
    assert isinstance(sample_rate,int)
    assert sample_rate > 0
    assert isinstance(samples,np.ndarray)
    assert automated_df is None or isinstance(automated_df,pd.DataFrame)
    assert premade_annotations_df is None or isinstance(premade_annotations_df,pd.DataFrame)
    assert isinstance(premade_annotations_label,str)
    assert isinstance(log_scale,bool)
    assert isinstance(save_fig,bool)
    assert isinstance(normalize_local_scores,bool)

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
        axs[0].set_ylim(0, 1.05)
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

def spectrogram_visualization(
        clip_path,
        weight_path=None,
        premade_annotations_df=None,
        premade_annotations_label="Human Labels",
        build_automated_df=None,
        isolation_parameters=None,
        log_scale=False,
        save_fig=False,
        normalize_local_scores=False,
        verbose=True):
    """
    Wrapper function for the local_line_graph and spectrogram_graph functions
    for ease of use. Processes clip for local scores to be used for the
    local_line_graph function.

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

        build_automated_df (bool)
            - Whether the audio clip should be labelled by the isolate function
              and subsequently plotted.

        log_scale (boolean)
            - Whether the axis for local scores should be logarithmically
              scaled on the plot.

        save_fig (boolean)
            - Whether the plots should be saved in a directory as a png file.

        verbose (boolean)
            - Whether to display error messages

    Returns:
        None
    """
    assert isinstance(clip_path,str)
    assert weight_path is None or isinstance(weight_path,str)
    assert premade_annotations_df is None or isinstance(premade_annotations_df,pd.DataFrame)
    assert isinstance(premade_annotations_label,str)
    assert build_automated_df is None or isinstance(build_automated_df,bool)
    assert isolation_parameters is None or isinstance(isolation_parameters,dict)
    assert isinstance(log_scale,bool)
    assert isinstance(save_fig,bool)
    assert isinstance(normalize_local_scores,bool)

    # Reading in the audio file using librosa, converting to single channeled data with original sample rate
    # Reason for the factor for the signal is explained here: https://stackoverflow.com/questions/53462062/pyaudio-bytes-data-to-librosa-floating-point-time-series
    # Librosa scales down to [-1, 1], but the models require the range [-32768, 32767], so the multiplication is required
    try:
        SIGNAL, SAMPLE_RATE = librosa.load(clip_path, sr=None, mono=True)
        SIGNAL = SIGNAL * 32768
    except BaseException:
        checkVerbose("Failure in loading" + clip_path, verbose)
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

    # Converting to Mono if Necessary
    if len(SIGNAL.shape) == 2:
        # averaging the two channels together
        SIGNAL = SIGNAL.sum(axis=1) / 2

    # Generate parameters for specific models
    local_scores = None
    if(isolation_parameters is not None):
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
        elif (isolation_parameters["model"] == "fg_bg_dsp_sep"):
            time_ratio, local_scores = FG_BG_local_score_arr(SIGNAL, isolation_parameters, normalized_sample_rate=SAMPLE_RATE)
        elif (isolation_parameters["model"]=="template_matching"):
            bandpass = False
            b = None
            a = None
            if "cutoff_freq_low" in isolation_parameters.keys() and "cutoff_freq_high" in isolation_parameters.keys():
                bandpass = True
                assert isinstance(isolation_parameters["cutoff_freq_low"], int)
                assert isinstance(isolation_parameters["cutoff_freq_high"], int)
                assert isolation_parameters["cutoff_freq_low"] > 0 and isolation_parameters["cutoff_freq_high"] > 0
                assert isolation_parameters["cutoff_freq_high"] > isolation_parameters["cutoff_freq_low"]
                assert isolation_parameters["cutoff_freq_high"] <= int(0.5*SAMPLE_RATE)
            
            TEMPLATE, _ = librosa.load(isolation_parameters["template_path"], sr=SAMPLE_RATE, mono=True)
            if bandpass:
                b, a = butter_bandpass(isolation_parameters["cutoff_freq_low"], isolation_parameters["cutoff_freq_high"], SAMPLE_RATE)
                TEMPLATE = filter(TEMPLATE, b, a)
            
            TEMPLATE_spec = generate_specgram(TEMPLATE, SAMPLE_RATE)
            TEMPLATE_mean = np.mean(TEMPLATE_spec)
            TEMPLATE_spec -= TEMPLATE_mean
            TEMPLATE_std_dev = np.std(TEMPLATE_spec)
            n = TEMPLATE_spec.shape[0] * TEMPLATE_spec.shape[1]

            SIGNAL, SAMPLE_RATE = librosa.load(clip_path, sr=SAMPLE_RATE, mono=True)
            if bandpass:
                SIGNAL = filter(SIGNAL, b, a)
            local_scores = template_matching_local_score_arr(SIGNAL, SAMPLE_RATE, TEMPLATE_spec, n, TEMPLATE_std_dev)

    # In the case where the user wants to look at automated bird labels
    if premade_annotations_df is None:
            premade_annotations_df = pd.DataFrame()
    automated_df = None
    # Generate labels based on the model
    if (build_automated_df is not None):
        if (isinstance(build_automated_df, bool) and not build_automated_df):
            automated_df = pd.DataFrame()
            pass
        # Check if Microfaune or TweetyNET was used to generate local scores
        if (local_scores is not None):
            # TweetyNET techniques and output
            if (isolation_parameters["model"] == "tweetynet"
                and isolation_parameters["tweety_output"]):
                automated_df = predictions_to_kaleidoscope(
                                predictions, 
                                SIGNAL, 
                                "Doesn't", 
                                "Doesn't", 
                                "Matter", 
                                SAMPLE_RATE)
            # Isolation techniques
            else: 
                if isinstance(local_scores,list):
                    local_scores = np.array(local_scores)
                automated_df = isolate(
                        local_scores,
                        SIGNAL,
                        SAMPLE_RATE,
                        audio_dir = "",
                        filename = "",
                        isolation_parameters=isolation_parameters)
        # Catch, generate the labels for other models
        else:
            automated_df = generate_automated_labels(
                audio_dir=clip_path,
                isolation_parameters=isolation_parameters,
                weight_path=weight_path,
                normalized_sample_rate=SAMPLE_RATE,
                normalize_local_scores=normalize_local_scores)

        if (len(automated_df["IN FILE"].unique()) > 1):
            checkVerbose("\nWarning: This function only generates spectrograms for one clip. " +
                  "automated_df has annotations for more than one clip.", verbose)
    else:
        automated_df = pd.DataFrame()

    # If local scores were generated, plot them AND spectrogram
    if (local_scores is not None):
        local_line_graph(
                local_scores,
                clip_path,
                SAMPLE_RATE,
                SIGNAL,
                automated_df,
                premade_annotations_df,
                premade_annotations_label=premade_annotations_label,
                log_scale=log_scale,
                save_fig=save_fig,
                normalize_local_scores=normalize_local_scores)
    else:
        spectrogram_graph(
            clip_path,
            SAMPLE_RATE,
            SIGNAL,
            automated_df=automated_df,
            premade_annotations_df=premade_annotations_df,
            premade_annotations_label=premade_annotations_label,
            save_fig=save_fig)

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
