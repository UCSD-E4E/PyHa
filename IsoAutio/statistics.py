import pandas as pd
from scipy import stats
import numpy as np


# Function that takes in a pandas dataframe of annotations and outputs a dataframe of the
# mean, median, mode, quartiles, and standard deviation of the annotation durations.
def annotation_duration_statistics(df):
    # Reading in the Duration column of the passed in dataframe as a Python list
    annotation_lengths = df["DURATION"].to_list()
    # converting to numpy array which has more readily available statistics functions
    annotation_lengths = np.asarray(annotation_lengths)
    # Converting the Python list to a numpy array
    entry = {'COUNT' : np.shape(annotation_lengths)[0],
             'MODE'  : stats.mode(np.round(annotation_lengths,2))[0][0],
             'MEAN'    : np.mean(annotation_lengths),
             'STANDARD DEVIATION' : np.std(annotation_lengths),
             'MIN': np.amin(annotation_lengths),
             'Q1': np.percentile(annotation_lengths,25),
             'MEDIAN'  : np.median(annotation_lengths),
             'Q3' : np.percentile(annotation_lengths,75),
             'MAX'  : np.amax(annotation_lengths)}
    # returning the dictionary as a pandas dataframe
    return pd.DataFrame.from_dict([entry])

def bird_label_scores(automated_df,human_df):
    """
    Function to generate a dataframe with statistics relating to the efficiency of the automated label compared to the human label.
    These statistics include true positive, false positive, false negative, true negative, union, precision, recall, F1, and Global IoU.

    Args:
        automated_df (Dataframe) - Dataframe of automated labels for one clip
        human_df (Dataframe) - Dataframe of human labels for one clip.
        plot_fig (boolean) - Whether or not the efficiency statistics should be displayed.
        save_fig (boolean) - Whether or not the plot should be saved within a file.

    Returns:
        Dataframe with statistics comparing the automated and human labeling.
    """
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


    # Recall = TP/(TP+FN)
        recall = true_positive_count/(true_positive_count + false_negative_count)

    # F1 = 2*(Recall*Precision)/(Recall + Precision)

        f1 = 2*(recall*precision)/(recall + precision)
        IoU = true_positive_count/union_count
    except:
        print("Error calculating statistics, likely due to zero division, setting values to zero")
        f1 = 0
        precision = 0
        recall = 0
        IoU = 0

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
             'Global IoU' : IoU}

    return pd.DataFrame(entry,index=[0])

def plot_bird_Label_scores(automated_df,human_df,save_fig = False):
    """
    Function to visualize automated and human annotation scores across an audio clip.

    Args:
        automated_df (Dataframe) - Dataframe of automated labels for one clip
        human_df (Dataframe) - Dataframe of human labels for one clip.
        plot_fig (boolean) - Whether or not the efficiency statistics should be displayed.
        save_fig (boolean) - Whether or not the plot should be saved within a file.

    Returns:
        Dataframe with statistics comparing the automated and human labeling.
    """
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





# Will have to adjust the isolate function so that it adds a sampling rate onto the dataframes.
def automated_labeling_statistics(automated_df,manual_df):
    """
    Function that will allow users to easily pass in two dataframes of manual labels and automated labels,
    and a dataframe is returned with statistics examining the efficiency of the automated labelling system compared
    to the human labels for multiple clips.

    Calls bird_local_scores on corresponding audio clips to generate the efficiency statistics for one specific clip which is then all put into one
    dataframe of statistics for multiple audio clips.

    Args:
        automated_df (Dataframe) - Dataframe of automated labels of multiple clips.
        manual_df (Dataframe) - Dataframe of human labels of multiple clips.

    Returns:
        Dataframe of statistics comparing automated labels and human labels for multiple clips.
    """
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
        statistics_df.reset_index(inplace = True, drop = True)
    return statistics_df


def global_dataset_statistics(statistics_df):
    """
    Function that takes in a dataframe of efficiency statistics for multiple clips and outputs their global values.

    Args:
        statistics_df (Dataframe) - Dataframe of statistics value for multiple audio clips as returned by the function automated_labelling_statistics.

    Returns:
        Dataframe of global statistics for the multiple audio clips' labelling.
    """
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
             'Global IoU' : round(IoU,6)}
    return pd.DataFrame.from_dict([entry])

# TODO rework this function to implement some linear algebra, right now the nested for loop won't handle larger loads well
# To make a global matrix, find the clip with the most amount of automated labels and set that to the number of columns
def clip_IoU(automated_df,manual_df):
    """
    Function that takes in the manual and automated labels for a clip and outputs human label-by-label IoU Scores.

    Args:
        automated_df (Dataframe) - Dataframe of automated labels for an audio clip.
        manual_df (Dataframe) - Dataframe of human labels for an audio clip.

    Returns:
        Numpy Array of human label IoU scores.
    """

    automated_df.reset_index(inplace = True, drop = True)
    manual_df.reset_index(inplace = True, drop = True)
    # Determining the number of rows in the output numpy array
    manual_row_count = manual_df.shape[0]
    # Determining the number of columns in the output numpy array
    automated_row_count = automated_df.shape[0]

    # Determining the length of the input clip
    duration = automated_df["CLIP LENGTH"].to_list()[0]
    # Determining the sample rate of the input clip
    SAMPLE_RATE = automated_df["SAMPLE RATE"].to_list()[0]

    # Initializing the output array that will contain the clip-by-clip Intersection over Union percentages.
    IoU_Matrix = np.zeros((manual_row_count,automated_row_count))
    #print(IoU_Matrix.shape)

    # Initializing arrays that will represent each of the human and automated labels
    bot_arr = np.zeros((int(duration * SAMPLE_RATE)))
    human_arr = np.zeros((int(duration * SAMPLE_RATE)))

    # Looping through each human label
    for row in manual_df.index:
        #print(row)
        # Determining the beginning of a human label
        minval = int(round(manual_df["OFFSET"][row]*SAMPLE_RATE,0))
        # Determining the end of a human label
        maxval = int(round((manual_df["OFFSET"][row] + manual_df["DURATION"][row]) *SAMPLE_RATE,0))
        # Placing the label relative to the clip
        human_arr[minval:maxval] = 1
        # Looping through each automated label
        for column in automated_df.index:
            # Determining the beginning of an automated label
            minval = int(round(automated_df["OFFSET"][column]*SAMPLE_RATE,0))
            # Determining the ending of an automated label
            maxval = int(round((automated_df["OFFSET"][column] + automated_df["DURATION"][column]) *SAMPLE_RATE,0))
            # Placing the label relative to the clip
            bot_arr[minval:maxval] = 1
            # Determining the overlap between the human label and the automated label
            intersection = human_arr * bot_arr
            # Determining the union between the human label and the automated label
            union = human_arr + bot_arr
            union[union == 2] = 1
            # Determining how much of the human label and the automated label overlap with respect to time
            intersection_count = np.count_nonzero(intersection == 1)/SAMPLE_RATE
            # Determining the span of the human label and the automated label with respect to time.
            union_count = np.count_nonzero(union == 1)/SAMPLE_RATE
            # Placing the Intersection over Union Percentage into it's respective position in the array.
            IoU_Matrix[row,column] = round(intersection_count/union_count,4)
            # Resetting the automated label to zero
            bot_arr[bot_arr == 1] = 0
        # Resetting the human label to zero
        human_arr[human_arr == 1] = 0

    return IoU_Matrix

def matrix_IoU_Scores(IoU_Matrix,manual_df,threshold):
    """
    Function that takes in the IoU Matrix from the clip_IoU function and ouputs the number of true positives and false positives,
    as well as calculating the precision.

    Args:
        IoU_Matrix (Numpy Array) - Matrix of human label IoU scores.
        manual_df (Dataframe) - Dataframe of human labels for an audio clip.
        threshold (float) - Threshold for determining true positives and false negatives.

    Returns:
        Dataframe of clip statistics such as True Positive, False Negative, False Positive, Precision, Recall, and F1 values.
    """

    audio_dir = manual_df["FOLDER"][0]
    filename = manual_df["IN FILE"][0]
    # TODO make sure that all of these calculations are correct. It is confusing to me that the Precision and Recall scores have a positive correlation.
    # Determining which automated label has the highest IoU across each human label
    automated_label_best_fits = np.max(IoU_Matrix,axis=1)
    #human_label_count = automated_label_best_fits.shape[0]
    # Calculating the number of true positives based off of the passed in thresholds.
    tp_count = automated_label_best_fits[automated_label_best_fits >= threshold].shape[0]
    # Calculating the number of false negatives from the number of human labels and true positives
    fn_count = automated_label_best_fits[automated_label_best_fits < threshold].shape[0]

    # Calculating the false positives
    max_val_per_column = np.max(IoU_Matrix,axis=0)
    fp_count = max_val_per_column[max_val_per_column < threshold].shape[0]

    # Calculating the necessary statistics
    try:
        recall = round(tp_count/(tp_count+fn_count),4)
        precision = round(tp_count/(tp_count+fp_count),4)
        f1 = round(2*(recall*precision)/(recall+precision),4)
    except ZeroDivisionError:
        print("Division by zero setting precision, recall, and f1 to zero")
        recall = 0
        precision = 0
        f1 = 0

    entry = {'FOLDER'  : audio_dir,
             'IN FILE'    : filename,
             'TRUE POSITIVE' : tp_count,
             'FALSE NEGATIVE' : fn_count,
             'FALSE POSITIVE': fp_count,
             'PRECISION'  : precision,
             'RECALL' : recall,
             'F1' : f1}

    return pd.DataFrame.from_dict([entry])

def clip_catch(automated_df,manual_df):
    """
    Function that determines the overlap between human and automated labels with respect to the number of samples in the human label.

    Args:
        automated_df (Dataframe) - Dataframe of automated labels for an audio clip.
        manual_df (Dataframe) - Dataframe of human labels for an audio clip.

    Returns:
        Numpy Array of statistics regarding the amount of overlap between the manual and automated labels relative to the number of
        samples.
    """
    # resetting the indices to make this function work
    automated_df.reset_index(inplace = True, drop = True)
    manual_df.reset_index(inplace = True, drop = True)
    # figuring out how many automated labels and human labels exist
    manual_row_count = manual_df.shape[0]
    automated_row_count = automated_df.shape[0]
    # finding the length of the clip as well as the sampling frequency.
    duration = automated_df["CLIP LENGTH"].to_list()[0]
    SAMPLE_RATE = automated_df["SAMPLE RATE"].to_list()[0]
    # initializing the output array, as well as the two arrays used to calculate catch scores
    catch_matrix = np.zeros(manual_row_count)
    bot_arr = np.zeros((int(duration * SAMPLE_RATE)))
    human_arr = np.zeros((int(duration * SAMPLE_RATE)))

    # Determining the automated labelled regions with respect to samples
    # Looping through each human label
    for row in automated_df.index:
        # converting each label into a "pulse" on an array that represents the labels as 0's and 1's on bot array.
        minval = int(round(automated_df["OFFSET"][row]*SAMPLE_RATE,0))
        maxval = int(round((automated_df["OFFSET"][row] + automated_df["DURATION"][row]) *SAMPLE_RATE,0))
        bot_arr[minval:maxval] = 1

    # Looping through each human label and computing catch = (#intersections)/(#samples in label)
    for row in manual_df.index:
        # Determining the beginning of a human label
        minval = int(round(manual_df["OFFSET"][row]*SAMPLE_RATE,0))
        # Determining the end of a human label
        maxval = int(round((manual_df["OFFSET"][row] + manual_df["DURATION"][row]) *SAMPLE_RATE,0))
        # Placing the label relative to the clip
        human_arr[minval:maxval] = 1
        # Determining the length of a label with respect to samples
        samples_in_label = maxval - minval
        # Finding where the human label and all of the annotated labels overlap
        intersection = human_arr * bot_arr
        # Determining how many samples overlap.
        intersection_count = np.count_nonzero(intersection == 1)
        # Intersection/length of label
        catch_matrix[row] = round(intersection_count/samples_in_label,4)
        # resetting the human label
        human_arr[human_arr == 1] = 0

    return catch_matrix



def dataset_IoU(automated_df,manual_df):
    """
    Function that takes in two Pandas dataframes that represent human labels and automated labels.
    It then runs the clip_IoU function across each clip and appends the best fit IoU score to each labels on the manual dataframe as its output.

    Args:
        automated_df (Dataframe) - Dataframe of automated labels for multiple audio clips.
        manual_df (Dataframe) - Dataframe of human labels for multiple audio clips.

    Returns:
        Dataframe of manual labels with the best fit IoU score as a column.
    """
    # Getting a list of clips
    clips = automated_df["IN FILE"].to_list()
    # Removing duplicates
    clips = list(dict.fromkeys(clips))
    # Initializing the ouput dataframe
    manual_df_with_IoU = pd.DataFrame()
    for clip in clips:
        print(clip)
        # Isolating a clip from the human and automated dataframes
        clip_automated_df = automated_df[automated_df["IN FILE"] == clip]
        clip_manual_df = manual_df[manual_df["IN FILE"] == clip]
        # Calculating the IoU scores of each human label.
        IoU_Matrix = clip_IoU(clip_automated_df,clip_manual_df)
        # Finding the best automated IoU score with respect to each label
        automated_label_best_fits = np.max(IoU_Matrix,axis=1)
        clip_manual_df["IoU"] = automated_label_best_fits
        # Appending on the best fit IoU score to each human label
        if manual_df_with_IoU.empty == True:
            manual_df_with_IoU = clip_manual_df
        else:
            manual_df_with_IoU = manual_df_with_IoU.append(clip_manual_df)
    # Adjusting the indices.
    manual_df_with_IoU.reset_index(inplace = True, drop = True)
    return manual_df_with_IoU


def dataset_IoU_Statistics(automated_df,manual_df,threshold = 0.5):
    """
    Wrapper function that takes matrix_IoU_Scores across multiple clips.
    Allows user to modify the threshold that determines whether or not a label is a true positive.

    Args:
        automated_df (Dataframe) - Dataframe of automated labels for multiple audio clips.
        manual_df (Dataframe) - Dataframe of human labels for multiple audio clips.
        threshold (float) - Threshold for determining true positives.

    Returns:
        Dataframe of IoU statistics for multiple audio clips.
    """
    # isolating the names of the clips that have been labelled into an array.
    clips = automated_df["IN FILE"].to_list()
    clips = list(dict.fromkeys(clips))
    # initializing the output Pandas dataframe
    IoU_Statistics = pd.DataFrame()
    # Looping through all of the clips
    for clip in clips:
        print(clip)
        # isolating the clip into its own dataframe with respect to both the passed in human labels and automated labels.
        clip_automated_df = automated_df[automated_df["IN FILE"] == clip]
        clip_manual_df = manual_df[manual_df["IN FILE"] == clip]
        # Computing the IoU Matrix across a specific clip
        IoU_Matrix = clip_IoU(clip_automated_df,clip_manual_df)
        # Calculating the best fit IoU to each label for the clip
        clip_stats_df = matrix_IoU_Scores(IoU_Matrix,clip_manual_df,threshold)
        # adding onto the output array.
        if IoU_Statistics.empty == True:
            IoU_Statistics = clip_stats_df
        else:
            IoU_Statistics = IoU_Statistics.append(clip_stats_df)
    IoU_Statistics.reset_index(inplace = True, drop = True)
    return IoU_Statistics

def global_IoU_Statistics(statistics_df):
    """
    Function that takes the output of dataset_IoU Statistics and outputs a global count of true positives and false positives,
    as well as computing the precision across the dataset.

    Args:
        statistics_df (Dataframe) - Dataframe of matrix IoU scores for multiple clips.

    Returns:
        Dataframe of global IoU statistics.
    """
    # taking the sum of the number of true positives and false positives.
    tp_sum = statistics_df["TRUE POSITIVE"].sum()
    fn_sum = statistics_df["FALSE NEGATIVE"].sum()
    fp_sum = statistics_df["FALSE POSITIVE"].sum()
    # calculating the precision, recall, and f1
    try:
        precision = tp_sum/(tp_sum+fp_sum)
        recall = tp_sum/(tp_sum+fn_sum)
        f1 = 2*(precision*recall)/(precision+recall)
    except ZeroDivisionError:
        print("Error in calculating Precision, Recall, and F1. Likely due to zero division, setting values to zero")
        precision = 0
        recall = 0
        f1 = 0
    # building a dictionary of the above calculations
    entry = {'TRUE POSITIVE' : tp_sum,
        'FALSE NEGATIVE' : fn_sum,
        'FALSE POSITIVE' : fp_sum,
        'PRECISION'  : round(precision,4),
        'RECALL' : round(recall,4),
        'F1' : round(f1,4)}
    # returning the dictionary as a pandas dataframe
    return pd.DataFrame.from_dict([entry])

def dataset_Catch(automated_df,manual_df):
    """
    Function that determines the label-by-label "Catch" across multiple clips.

    Args:
        automated_df (Dataframe) - Dataframe of automated labels for multiple audio clips.
        manual_df (Dataframe) - Dataframe of human labels for multiple audio clips.

    Returns:
        Dataframe of human labels with a column for the catch values of each label.
    """
    # Getting a list of clips
    clips = automated_df["IN FILE"].to_list()
    # Removing duplicates
    clips = list(dict.fromkeys(clips))
    # Initializing the ouput dataframe
    manual_df_with_Catch = pd.DataFrame()
    # Looping through all of the audio clips that have been labelled.
    for clip in clips:
        print(clip)
        # Isolating the clips from both the automated and human dataframes
        clip_automated_df = automated_df[automated_df["IN FILE"] == clip]
        clip_manual_df = manual_df[manual_df["IN FILE"] == clip]
        # Calling the function that calculates the catch over a specific clip
        Catch_Array = clip_catch(clip_automated_df,clip_manual_df)
        # Appending the catch values per label onto the manual dataframe
        clip_manual_df["Catch"] = Catch_Array
        if manual_df_with_Catch.empty == True:
            manual_df_with_Catch = clip_manual_df
        else:
            manual_df_with_Catch = manual_df_with_Catch.append(clip_manual_df)
    # Resetting the indices
    manual_df_with_Catch.reset_index(inplace = True, drop = True)
    return manual_df_with_Catch
