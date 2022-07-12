from .microfaune_package.microfaune.detection import RNNDetector
from .microfaune_package.microfaune import audio
from .tweetynet_package.tweetynet.TweetyNetModel import TweetyNetModel
from .tweetynet_package.tweetynet.Load_data_functions import compute_features
import torch
import matplotlib.pyplot as plt
import pandas as pd
import scipy.signal as scipy_signal
import numpy as np
import seaborn as sns
from .IsoAutio import *
from .annotation_post_processing import *

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

def spectrogram_visualization(
        clip_path,
        weight_path=None,
        premade_annotations_df=None,
        premade_annotations_label="Human Labels",
        automated_df=None,
        isolation_parameters=None,
        log_scale=False,
        save_fig=False,
        normalize_local_scores=False):
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

    # Downsample the audio if the sample rate > 44.1 kHz
    # Force everything into the human hearing range.
    try:
        if SAMPLE_RATE != 44100:
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
                    print("Error in weight path:", weight_path)
                    return
            try:
                # Computing Mel Spectrogram of the audio clip
                microfaune_features = detector.compute_features([SIGNAL])
                # Running the Mel Spectrogram through the RNN
                global_score, local_score = detector.predict(microfaune_features)
                local_scores = local_score[0].tolist()
            except BaseException:
                print(
                    "Skipping " +
                    clip_path +
                    " due to error in Microfaune Prediction")
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
                print(
                    "Skipping " +
                    clip_path +
                    " due to error in TweetyNet Prediction")
                return None

    # In the case where the user wants to look at automated bird labels
    if premade_annotations_df is None:
            premade_annotations_df = pd.DataFrame()

    # Generate labels based on the model
    if (automated_df is not None):
        if (isinstance(automated_df, bool) and not automated_df):
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
                automated_df = isolate(
                        local_score[0],
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
            print("\nWarning: This function only generates spectrograms for one clip. " +
                  "automated_df has annotations for more than one clip.")
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


## ROC CURVE CODE BELOW
def convert_label_to_local_score(manual_df, size_of_local_score):
    """
    Helper Function For ROC Curve generation. Given a manual dataframe and a spefifed array size,
    coverts the dataframe's anntotions to a array representation with each index corresponding to
    a spefific time in the audio data with values of 1 (annotation present) or 0 (annotation absent)

        Args:
            manual_df (Dataframe)
                - Dataframe of human labels of ONLY A SINGLE CLIP

            size_of_local_score
                - size of array to create

        Returns:
            array representation of annotation locations of a given clip
    """
    #Determine how many seconds each index corresponds to
    duration_of_clip = manual_df.iloc[0]["CLIP LENGTH"]
    seconds_per_index = duration_of_clip/size_of_local_score

    #Init array
    local_score = np.zeros(size_of_local_score)

    #For each index
    for i in range(size_of_local_score):
        #Determine the time that corresponds to this index
        current_seconds = i * seconds_per_index

        #If the manual dataframe has an annotation at that time, set the index value to 1
        annotations_at_time = manual_df[(manual_df["OFFSET"] <= current_seconds) & (manual_df["OFFSET"] +manual_df["DURATION"] >=  current_seconds)]
        if (not annotations_at_time.empty):
            local_score[i] = 1
    
    return local_score


    
def get_target_annotations(chunked_manual_df, chunk_size):
    """
    Helper Function For ROC Curve generation. Converts a manual_df into a array representation of target ground truth values
    for an entire dataset in order of the clips appearing in a dataset

        Args:
            manual_df (Dataframe)
                - Dataframe of human labels

            size_of_local_score
                - size of array to create

        Returns:
            array representation of annotation locations of a given clip
            some debugging array to make sure everything is correctly lining up
    """
    target_score_array = []
    #Iterate through all files in the dataframe
    manual_df = chunked_manual_df.set_index(["FOLDER","IN FILE"])
    chunk_size_list = []
    for item in np.unique(manual_df.index):
        #Get all annotations for a given clip
        clip_df = chunked_manual_df[(chunked_manual_df["FOLDER"] == item[0]) & (chunked_manual_df["IN FILE"] == item[1])]
        clip_duration = clip_df.iloc[0]["CLIP LENGTH"]

        #get the target array for that clip
        number_of_chunks = math.floor(clip_duration/chunk_size)
        target_score_clip = convert_label_to_local_score(clip_df, number_of_chunks)

        #Append that data to the return array
        chunk_size_list.append((number_of_chunks, clip_duration, item[1]))
        target_score_array.extend(target_score_clip)

    return np.array(target_score_array), chunk_size_list


# #Remember to chunk before passing it in
# target_array = get_target_annotations(chunked_df_manual_clip, 3)
# #Returns array -> 1 = bird, 0 = no bird

#instead here get the local scores array from generated_automated_labels dictionary 
def get_confidence_array(local_scores_array,chunked_df, chunk_size_list):
    """
    Helper Function For ROC Curve generation. Converts a directory of local scores into
    a array containing the local score maximums of each chunk in each clip in a dataset

        Args:
            chunked_df (Dataframe)
                - Dataframe of dataset in question

            local_scores_array
                - Directory of local scores with keys being the file name of each local score (values)

            chunk_size_list
                - Debugging directory to ensure everything is lining up correctly  

        Returns:
            array representation of local maximum confidences for an entire dataset
    """
    array_of_max_scores = []
    manual_df = chunked_df.set_index(["FOLDER","IN FILE"])
    k = 0

    for item in np.unique(manual_df.index):
        #For each file in the dataset, obtain the annotation data of that file
        clip_df = chunked_df[(chunked_df["FOLDER"] == item[0]) & (chunked_df["IN FILE"] == item[1])]
        local_score_clip = local_scores_array[item[1]]
        
        #calculate the number of local maximums we need
        duration_of_clip = clip_df.iloc[0]["CLIP LENGTH"] 
        num_chunks = math.floor(duration_of_clip/3)

        #At this point the files should line up with the files appended in order for the target array
        #If no, return an error 
        if (num_chunks != chunk_size_list[k][0]):
            print("BAD CHUNK SIZE, CONFIDENCE SIZE: ", num_chunks, " TARGET: ", chunk_size_list[k] )
            print("duration_of_clip", duration_of_clip, chunk_size_list[i][1])
            print(item[1], chunk_size_list[k][2])
            break
        k += 1


        chunk_length = int(clip_df.iloc[0]["DURATION"]) #3 sec
        #For each chunk in seconds
        for i in range(0, num_chunks * chunk_length,chunk_length):
            # now iterate through the local_score array for each chunk
            clip_length = clip_df.iloc[0]["CLIP LENGTH"]
            #seconds_per_index = clip_length/len(local_score_clip)
            index_per_seconds = len(local_score_clip)/clip_length
            
            #Get the starting and ending index of that chunk as respective to
            #the local score array
            start_index = math.floor(index_per_seconds * i)
            end_index = math.floor(index_per_seconds *(i + chunk_length))
           
            #Compute the local maximum in this chunk in the local scores
            max_score = 0.0
            current_score = 0.0
            chunk_length = int(clip_df.iloc[0]["DURATION"])
            for j in range(start_index, end_index):
                    #TODO use max() here instead?
                    current_score = local_score_clip[j]
                    if (current_score > max_score):
                        max_score = current_score

            #Append local maximum to list of all local maximum local scores            
            array_of_max_scores.append(max_score)
    return array_of_max_scores



def generate_ROC_curves_chunked_local(automated_df, manual_df, local_scoress, chunk_length = 3, label=""):
    """
    Function For ROC Curve generation. Displays the given roc curve for some automated labels
    NOTE: this chunks both automated and manual labels piror to ROC curve generation
    NOTE: This requires the local_score array. If your automated_df has confidence values, please use
    generate_ROC_curves

        Args:
            automated_df (Dataframe)
                - Autoamted Dataframe of dataset in question

            manual_df (Dataframe)
                - Manual Dataframe of dataset in question

            local_scores_array
                - Directory of local scores with keys being the file name of each local score (values)

            chunk_length
                - size in seconds of each chunk  

            label (String)
                - name to display in legend. Doesn't display legend if left blank

        Returns:
            Area Under the ROC Curve
    """

    #Only include files shared by both
    manual_df = manual_df[manual_df['IN FILE'].isin(automated_df["IN FILE"].to_list())]
    automated_df = automated_df[automated_df['IN FILE'].isin(manual_df["IN FILE"].to_list())]

    #chunk the data
    auto_chunked_df = annotation_chunker_no_duplicates(automated_df, chunk_length)
    manual_chunked_df = annotation_chunker_no_duplicates(manual_df, chunk_length)

    #sort the data to ensure all append operations are in order
    auto_chunked_df = auto_chunked_df.sort_values(by="IN FILE")
    manual_chunked_df = manual_chunked_df.sort_values(by="IN FILE")
    
    #GENERATE TARGET AND CONFIDENCE ARRAYS FOR ROC CURVE GENERATION
    target_array, chunk_size_list = get_target_annotations(manual_chunked_df, chunk_length)
    confidence_scores_array = get_confidence_array(local_scoress,manual_chunked_df, chunk_size_list) #auto_chunked_df
    
    #Sanity check code
    print("target", len(target_array))
    print("confidence", len(confidence_scores_array))

    #GENERATE AND PLOT ROC CURVES
    fpr, tpr, thresholds = metrics.roc_curve(target_array, confidence_scores_array) 
    roc_auc = metrics.auc(fpr, tpr)
    plt.plot(fpr, tpr, label=label)
    plt.ylabel("True Postives")
    plt.xlabel("False Positives ")
    if (label != ""):
        plt.legend(loc="lower right")
    plt.show
    return roc_auc


#wrapper function for get_confidence_array()
#i don't think this should be local_scores
def generate_ROC_curves_raw_local(automated_df, manual_df, local_scoress, label=""):
    """
    Function For ROC Curve generation. Displays the given roc curve for some automated labels
    Note: this does not chunk data and relies of local score itself for annotations. Great for comparing
    models who have long periods of high local score array values, less good for models like 
    mircofaune with spikes in data

    NOTE: It is recommend to use generate_ROC_curves() if automated_df has confidence column

        Args:
            automated_df (Dataframe)
                - Autoamted Dataframe of dataset in question

            manual_df (Dataframe)
                - Manual Dataframe of dataset in question

            local_scores_array
                - Directory of local scores with keys being the file name of each local score (values)

            label (String)
                - name to display in legend. Doesn't display legend if left blank

        Returns:
            Area Under the ROC Curve
    """

    #Only include files shared by both
    manual_df = manual_df[manual_df['IN FILE'].isin(automated_df["IN FILE"].to_list())]
    automated_df = automated_df[automated_df['IN FILE'].isin(manual_df["IN FILE"].to_list())]
    
    #Since we don't need chunking we can be a bit more stright forward
    target_array = np.array([])
    confidence_scores_array = np.array([])
    tmp_df = manual_df.set_index(["FOLDER","IN FILE"])
    for item in np.unique(tmp_df.index):
        #Get the data for each indivual clip
        clip_manual_df = manual_df[(manual_df["FOLDER"] == item[0]) & (manual_df["IN FILE"] == item[1])]
        local_score_clip = local_scoress[item[1]]
        
        #Determine how many seconds corresponds to each index of the
        #local score array
        duration_of_clip = clip_manual_df.iloc[0]["CLIP LENGTH"]
        size_of_local_score = len(local_score_clip)
        seconds_per_index = duration_of_clip/size_of_local_score
        
        #Get the target array represetnation of the manual dataframe that is
        #the same size as our local_score array
        target_clip = np.zeros((size_of_local_score))
        for i in range(size_of_local_score):
            current_seconds = i * seconds_per_index
            annotations_at_time = manual_df[(manual_df["OFFSET"] <= current_seconds) & (manual_df["OFFSET"] +manual_df["DURATION"] >=  current_seconds)]
            if (not annotations_at_time.empty):
                target_clip[i] = 1
        
        #Append the target array and the corresponding local score array
        target_array = np.append(target_array, target_clip)
        confidence_scores_array = np.append(confidence_scores_array, local_score_clip)
    
    
    #sanity check code
    print("target", len(target_array.tolist()))
    print("confidence", len(confidence_scores_array.tolist()))

    #GENERATE AND PLOT ROC CURVES
    fpr, tpr, thresholds = metrics.roc_curve(target_array, confidence_scores_array) 
    roc_auc = metrics.auc(fpr, tpr)
    plt.plot(fpr, tpr, label=label)
    plt.ylabel("True Postives")
    plt.xlabel("False Positives ")
    if (label != ""):
        plt.legend(loc="lower right")
    plt.show
    return roc_auc

def generate_ROC_curves(automated_df, manual_df, label="", chunk_length=3):
    """
    Function For ROC Curve generation. Displays the given roc curve for some automated labels
    Primary generate ROC curve function

        Args:
            automated_df (Dataframe)
                - Autoamted Dataframe of dataset in question

            manual_df (Dataframe)
                - Manual Dataframe of dataset in question

            label (String)
                - name to display in legend. Doesn't display legend if left blank

        Returns:
            Area Under the ROC Curve
    """

    #Only include files shared by both
    manual_df = manual_df[manual_df['IN FILE'].isin(automated_df["IN FILE"].to_list())]
    automated_df = automated_df[automated_df['IN FILE'].isin(manual_df["IN FILE"].to_list())]
    
    #chunk the data
    automated_df = annotation_chunker_no_duplicates(automated_df, chunk_length, include_no_bird=True)
    manual_df = annotation_chunker_no_duplicates(manual_df, chunk_length, include_no_bird=True)

    #sort the data to ensure all append operations are in order
    automated_df = automated_df.sort_values(by=["IN FILE", "OFFSET"])
    manual_df = manual_df.sort_values(by=["IN FILE", "OFFSET"])

    #get the true labels and confidence of each chunk, save as 2 arrays
    #each index in both arrays are the confidence and true value for one chunk
    target_array = np.array(manual_df["CONFIDENCE"])#get_target_annotations(manual_df, chunk_length)[0])
    confidence_scores_array = np.array(automated_df["CONFIDENCE"])

    #sanity check code
    print("target", len(target_array.tolist()))
    print("confidence", len(confidence_scores_array.tolist()))
    print("automated df", automated_df.shape[0])

    #GENERATE AND PLOT ROC CURVES
    fpr, tpr, thresholds = metrics.roc_curve(target_array, confidence_scores_array) 
    roc_auc = metrics.auc(fpr, tpr)
    plt.plot(fpr, tpr, label=label)
    plt.ylabel("True Postives")
    plt.xlabel("False Positives ")
    if (label != ""):
        plt.legend(loc="lower right")
    plt.show
    return roc_auc


