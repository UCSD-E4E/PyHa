 <img src="https://github.com/UCSD-E4E/PyHa/blob/readme/Logos/PyHa.svg" alt="PyHa logo" title="PyHa" align="right" height="300" />

# PyHa
<!-- ## Automated Audio Labeling System -->

A tool designed to convert audio-based "weak" labels to "strong" moment-to-moment labels. Provides a pipeline to compare automated moment-to-moment labels to human labels. Current proof of concept work being fulfilled on Bird Audio clips using Microfaune predictions.

PyHa = Python + Piha (referring to a bird species of our interest known as the screaming-piha)

## Contents

- [Installation and Setup](#installation-and-setup)
- [Functions](#functions)

## Installation and Setup

## Functions

![design](https://user-images.githubusercontent.com/44332326/123478194-f74fda80-d5b3-11eb-81e4-86add2a8c0f0.png)
*This image shows the design of the automated audio labeling system.*

### `isolation_parameters`

Many of the functions take in the `isolation_parameters` argument, and as such it will be defined globally here. 

The `isolation_parameters` dictionary is as follows: 

``` python
isolation_parameters = {
    "technique" : "",
    "threshold_type" : "",
    "threshold_const" : 0.0,
    "threshold_min" : 0.0,
    "window_size" : 0.0,
    "chunk_size" : 0.0,
} 
```
The `technique` parameter can be: Simple, Stack, Steinberg, and Chunk. This input must be a string in all lowercase.  
The `threshold_type` parameter can be: median, mean, average, standard deviation, or pure. This input must be a string in all lowercase.

The remaining parameters are floats representing their respective values. 

<!-- IsoAudio.py file -->

<details>
 <summary>IsoAutio.py files</summary>
 
### [`isolate`](https://github.com/UCSD-E4E/PyHa/blob/main/PyHa/IsoAutio.py)
*Found in [`IsoAutio.py`](https://github.com/UCSD-E4E/PyHa/blob/main/PyHa/IsoAutio.py)*

This function is the wrapper function for all the audio isolation techniques, and will call the respective function based on its parameters. 

| Parameter | Type |  Description |
| --- | --- | --- |
| `local_scores` | list of floats | Local scores of the audio clip as determined by Microfaune Recurrent Neural Network. |
| `SIGNAL` | list of ints | Samples that make up the audio signal. |
| `SAMPLE_RATE` | int | Sampling rate of the audio clip, usually 44100. |
| `audio_dir` | string | Directory of the audio clip. |
| `filename` | string | Name of the audio clip file. |
| `isolation_parameters` | dict | Python Dictionary that controls the various label creation techniques. |

This function returns a dataframe of automated labels for the audio clip based on the passed in isolation technique. 

Usage: 
`isolate(local_scores, SIGNAL, SAMPLE_RATE, audio_dir, filename, isolation_parameters)`

### [`threshold`](https://github.com/UCSD-E4E/PyHa/blob/main/PyHa/IsoAutio.py)
*Found in [`IsoAutio.py`](https://github.com/UCSD-E4E/PyHa/blob/main/PyHa/IsoAutio.py)*

This function takes in the local score array output from a neural network and determines the threshold at which we determine a local score to be a positive ID of a class of interest. Most proof of concept work is dedicated to bird presence. Threshold is determined by "threshold_type" and "threshold_const" from the isolation_parameters dictionary.

| Parameter | Type | Description | 
| --- | --- | --- | 
| `local_scores` | list of floats | Local scores of the audio clip as determined by Microfaune Recurrent Neural Network. | 
| `isolation parameters` | dict | Python Dictionary that controls the various label creation techniques. | 

This function returns a float representing the threshold at which the local scores in the local score array of an audio clip will be viewed as a positive ID.

Usage: `threshold(local_scores, isolation_parameters)`

### [`steinberg_isolate`](https://github.com/UCSD-E4E/PyHa/blob/main/PyHa/IsoAutio.py)
*Found in [`IsoAutio.py`](https://github.com/UCSD-E4E/PyHa/blob/main/PyHa/IsoAutio.py)*

This function uses the technique developed by Gabriel Steinberg that attempts to take the local score array output of a neural network and lump local scores together in a way to produce automated labels based on a class across an audio clip. It is called by the `isolate` function when `isolation_parameters['technique'] == steinberg`. 

| Parameter | Type |  Description |
| --- | --- | --- |
| `local_scores` | list of floats | Local scores of the audio clip as determined by Microfaune Recurrent Neural Network. |
| `SIGNAL` | list of ints | Samples that make up the audio signal. |
| `SAMPLE_RATE` | int | Sampling rate of the audio clip, usually 44100. |
| `audio_dir` | string | Directory of the audio clip. |
| `filename` | string | Name of the audio clip file. |
| `isolation_parameters` | dict | Python Dictionary that controls the various label creation techniques. |
| `manual_id` | string | controls the name of the class written to the pandas dataframe |

This function returns a dataframe of automated labels for the audio clip. 

Usage: `steinberg_isolate(local_scores, SIGNAL, SAMPLE_RATE, audio_dir, filename,isolation_parameters, manual_id)`

### [`simple_isolate`](https://github.com/UCSD-E4E/PyHa/blob/main/PyHa/IsoAutio.py)
*Found in [`IsoAutio.py`](https://github.com/UCSD-E4E/PyHa/blob/main/PyHa/IsoAutio.py)*

This function uses the technique suggested by Irina Tolkova and implemented by Jacob Ayers. Attempts to produce automated annotations of an audio clip based on local score array outputs from a neural network. It is called by the `isolate` function when `isolation_parameters['technique'] == simple`. 

| Parameter | Type |  Description |
| --- | --- | --- |
| `local_scores` | list of floats | Local scores of the audio clip as determined by Microfaune Recurrent Neural Network. |
| `SIGNAL` | list of ints | Samples that make up the audio signal. |
| `SAMPLE_RATE` | int | Sampling rate of the audio clip, usually 44100. |
| `audio_dir` | string | Directory of the audio clip. |
| `filename` | string | Name of the audio clip file. |
| `isolation_parameters` | dict | Python Dictionary that controls the various label creation techniques. |
| `manual_id` | string | controls the name of the class written to the pandas dataframe |

This function returns a dataframe of automated labels for the audio clip. 

Usage: `simple_isolate(local_scores, SIGNAL, SAMPLE_RATE, audio_dir, filename,isolation_parameters, manual_id)`

### [`stack_isolate`](https://github.com/UCSD-E4E/PyHa/blob/main/PyHa/IsoAutio.py)
*Found in [`IsoAutio.py`](https://github.com/UCSD-E4E/PyHa/blob/main/PyHa/IsoAutio.py)*

This function uses a technique created by Jacob Ayers. Attempts to produce automated annotations of an audio clip baseon local score array outputs from a neural network. It is called by the `isolate` function when `isolation_parameters['technique'] == stack`. 

| Parameter | Type |  Description |
| --- | --- | --- |
| `local_scores` | list of floats | Local scores of the audio clip as determined by Microfaune Recurrent Neural Network. |
| `SIGNAL` | list of ints | Samples that make up the audio signal. |
| `SAMPLE_RATE` | int | Sampling rate of the audio clip, usually 44100. |
| `audio_dir` | string | Directory of the audio clip. |
| `filename` | string | Name of the audio clip file. |
| `isolation_parameters` | dict | Python Dictionary that controls the various label creation techniques. |
| `manual_id` | string | controls the name of the class written to the pandas dataframe |

This function returns a dataframe of automated labels for the audio clip. 

Usage: `stack_isolate(local_scores, SIGNAL, SAMPLE_RATE, audio_dir, filename,isolation_parameters, manual_id)`

### [`chunk_isolate`](https://github.com/UCSD-E4E/PyHa/blob/main/PyHa/IsoAutio.py)
*Found in [`IsoAutio.py`](https://github.com/UCSD-E4E/PyHa/blob/main/PyHa/IsoAutio.py)*

This function uses a technique created by Jacob Ayers. Attempts to produce automated annotations of an audio clip baseon local score array outputs from a neural network. It is called by the `isolate` function when `isolation_parameters['technique'] == chunk`. 

| Parameter | Type |  Description |
| --- | --- | --- |
| `local_scores` | list of floats | Local scores of the audio clip as determined by Microfaune Recurrent Neural Network. |
| `SIGNAL` | list of ints | Samples that make up the audio signal. |
| `SAMPLE_RATE` | int | Sampling rate of the audio clip, usually 44100. |
| `audio_dir` | string | Directory of the audio clip. |
| `filename` | string | Name of the audio clip file. |
| `isolation_parameters` | dict | Python Dictionary that controls the various label creation techniques. |
| `manual_id` | string | controls the name of the class written to the pandas dataframe |

This function returns a dataframe of automated labels for the audio clip. 

Usage: `chunk_isolate(local_scores, SIGNAL, SAMPLE_RATE, audio_dir, filename,isolation_parameters, manual_id)`

### [`generate_automated_labels`](https://github.com/UCSD-E4E/PyHa/blob/main/PyHa/IsoAutio.py)
*Found in [`IsoAutio.py`](https://github.com/UCSD-E4E/PyHa/blob/main/PyHa/IsoAutio.py)*

This function applies the isolation technique determined by the `isolation_parameters` dictionary accross a whole folder of audio clips. 

| Parameter | Type |  Description |
| --- | --- | --- |
| `audio_dir` | string | Directory with wav audio files |
| `isolation_parameters` | dict | Python Dictionary that controls the various label creation techniques. |
| `manual_id` | string | controls the name of the class written to the pandas dataframe |
| `weight_path` | string | File path of weights to be used by the RNNDetector for determining presence of bird sounds.
| `Normalized_Sample_Rate` | int | Sampling rate that the audio files should all be normalized to.
| `normalize_local_scores` | boolean | Set whether or not to normalize the local scores. 

This function returns a dataframe of automated labels for the audio clips in audio_dir.

Usage: `generate_automated_labels(audio_dir, isolation_parameters, manual_id, weight_path, Normalized_Sample_Rate, normalize_local_scores)`

### [`kaleidoscope_conversion`](https://github.com/UCSD-E4E/PyHa/blob/main/PyHa/IsoAutio.py)
*Found in [`IsoAutio.py`](https://github.com/UCSD-E4E/PyHa/blob/main/PyHa/IsoAutio.py)*

This function strips away Pandas Dataframe columns necessary for the PyHa package that aren't compatible with the Kaleidoscope software.

| Parameter | Type |  Description |
| --- | --- | --- |
| `df` | Pandas Dataframe | Dataframe compatible with PyHa package whether it be human labels or automated labels. |

This function returns a Pandas Dataframe compatible with Kaleidoscope. 

Usage: `kaleidoscope_conversion(df)`

</details>


<!-- statistics.py file -->
<details>
 <summary>statistics.py file</summary>

</details>


### [`annotation_duration_statistics`](https://github.com/UCSD-E4E/PyHa/blob/main/PyHa/statistics.py)
*Found in [`statistics.py`](https://github.com/UCSD-E4E/PyHa/blob/main/PyHa/statistics.py)*

This function calculates basic statistics related to the duration of annotations of a Pandas Dataframe compatible with PyHa.

| Parameter | Type |  Description |
| --- | --- | --- |
| `df` | Pandas Dataframe | Dataframe of automated labels or manual labels. |

This function returns a Pandas Dataframe containing count, mean, mode, standard deviation, and IQR values based on annotation duration. 

Usage: `annotation_duration_statistics(df)`

### [`bird_label_scores`](https://github.com/UCSD-E4E/PyHa/blob/main/PyHa/statistics.py)
*Found in [`statistics.py`](https://github.com/UCSD-E4E/PyHa/blob/main/PyHa/statistics.py)*

This function to generates a dataframe with statistics relating to the efficiency of the automated label compared to the human label. These statistics include true positive, false positive, false negative, true negative, union, precision, recall, F1, and Global IoU for general clip overlap.

| Parameter | Type |  Description |
| --- | --- | --- |
| `automated_df` | Dataframe | Dataframe of automated labels for one clip |
| `human_df` | Dataframe | Dataframe of human labels for one clip. |

This function returns a dataframe with general clip overlap statistics comparing the automated and human labeling. 

Usage: `bird_label_scores(automated_df, human_df)`

### [`automated_labeling_statistics`](https://github.com/UCSD-E4E/PyHa/blob/main/PyHa/statistics.py)
*Found in [`statistics.py`](https://github.com/UCSD-E4E/PyHa/blob/main/PyHa/statistics.py)*

This function allows users to easily pass in two dataframes of manual labels and automated labels, and returns a dataframe with statistics examining the efficiency of the automated labelling system compared to the human labels for multiple clips. It calls `bird_local_scores` on corresponding audio clips to generate the efficiency statistics for one specific clip which is then all put into one dataframe of statistics for multiple audio clips.

| Parameter | Type |  Description |
| --- | --- | --- |
| `automated_df` | Dataframe | Dataframe of automated labels of multiple clips. |
| `manual_df` | Dataframe |  Dataframe of human labels of multiple clips. |
| `stats_type` | String | String that determines which type of statistics are of interest |
| `threshold` | Float | Defines a threshold for certain types of statistics |

This function returns a dataframe of statistics comparing automated labels and human labels for multiple clips. 

Usage: `automated_labeling_statistics(automated_df, manual_df, stats_type, threshold)`

### [`global_dataset_statistics`](https://github.com/UCSD-E4E/PyHa/blob/main/PyHa/statistics.py)
*Found in [`statistics.py`](https://github.com/UCSD-E4E/PyHa/blob/main/PyHa/statistics.py)*

This function takes in a dataframe of efficiency statistics for multiple clips and outputs their global values.

| Parameter | Type |  Description |
| --- | --- | --- |
| `statistics_df` | Dataframe | Dataframe of statistics value for multiple audio clips as returned by the function automated_labelling_statistics. |

This function returns a dataframe of global statistics for the multiple audio clips' labelling.. 

Usage: `global_dataset_statistics(statistics_df)`

### [`clip_IoU`](https://github.com/UCSD-E4E/PyHa/blob/main/PyHa/statistics.py)
*Found in [`statistics.py`](https://github.com/UCSD-E4E/PyHa/blob/main/PyHa/statistics.py)*

This function takes in the manual and automated labels for a clip and outputs IoU metrics of each human label with respect to each automated label.

| Parameter | Type |  Description |
| --- | --- | --- |
| `automated_df` | Dataframe | Dataframe of automated labels for one clip |
| `human_df` | Dataframe | Dataframe of human labels for one clip. |

This function returns an `IoU_Matrix` (arr) - (human label count) x (automated label count) matrix where each row contains the IoU of each automated annotation with respect to a human label.

Usage: `clip_IoU(automated_df, manual_df)`

### [`matrix_IoU_Scores`](https://github.com/UCSD-E4E/PyHa/blob/main/PyHa/statistics.py)
*Found in [`statistics.py`](https://github.com/UCSD-E4E/PyHa/blob/main/PyHa/statistics.py)*

This function takes in the manual and automated labels for a clip and outputs IoU metrics of each human label with respect to each automated label.

| Parameter | Type |  Description |
| --- | --- | --- |
| `IoU_Matrix`  | arr | (human label count) x (automated label count) matrix where each row contains the IoU of each automated annotation with respect to a human label. |
| manual_df | Dataframe | Dataframe of human labels for an audio clip. |
| threshold | float | IoU threshold for determining true positives, false positives, and false negatives. | 

This function returns a dataframe of clip statistics such as True Positive, False Negative, False Positive, Precision, Recall, and F1 values for an audio clip.

Usage: `matrix_IoU_Scores(IoU_Matrix, manual_df, threshold)`


















<!-- visualizations.py file -->
<details>
 <summary>visualizations.py file</summary>
 
### [`local_line_graph`](https://github.com/UCSD-E4E/PyHa/blob/main/PyHa/visualizations.py)
*Found in [`visualizations.py`](https://github.com/UCSD-E4E/PyHa/blob/main/PyHa/visualizations.py)*

This function produces graphs with the local score plot and spectrogram of an audio clip. It is now integrated with Pandas so you can visualize human and automated annotations.

| Parameter | Type |  Description |
| --- | --- | --- |
| `local_scores` | list of floats | Local scores for the clip determined by the RNN. |
| `clip_name`  | string | Directory of the clip. |
| `sample_rate` | int | Sample rate of the audio clip, usually 44100. |
| `samples` | list of ints | Each of the samples from the audio clip. |
| `automated_df` | Dataframe | Dataframe of automated labelling of the clip. |
| `premade_annotations_df` | Dataframe | Dataframe labels that have been made outside of the scope of this function. |
| `premade_annotations_label` | string | Descriptor of premade_annotations_df |
| `log_scale` | boolean | Whether the axis for local scores should be logarithmically scaled on the plot. |
| `save_fig`  | boolean | Whether the clip should be saved in a directory as a png file. |

This function does not return anything. 

Usage: `local_line_graph(local_scores, clip_name, sample_rate, samples, automated_df, premade_annotations_df, premade_annotations_label, log_scale, save_fig, normalize_local_scores)`

### [`local_score_visualization`](https://github.com/UCSD-E4E/PyHa/blob/main/PyHa/visualizations.py)
*Found in [`visualizations.py`](https://github.com/UCSD-E4E/PyHa/blob/main/PyHa/visualizations.py)*

This is the wrapper function for the local_line_graph function for ease of use. Processes clip for local scores to be used for the local_line_graph function.

| Parameter | Type |  Description |
| --- | --- | --- |
| `clip_path` | string | Path to an audio clip. |
| `weight_path` | string | Weights to be used for RNNDetector. |
| `premade_annotations_df` | Dataframe | Dataframe of annotations to be displayed that have been created outside of the function. |
| `premade_annotations_label` | string | String that serves as the descriptor for the premade_annotations dataframe. |
| `automated_df` | Dataframe | Whether the audio clip should be labelled by the isolate function and subsequently plotted. |
| `log_scale` | boolean | Whether the axis for local scores should be logarithmically scaled on the plot. |
| `save_fig` | boolean | Whether the plots should be saved in a directory as a png file. |

This function does not return anything. 

Usage: `local_score_visualization(clip_path, weight_path, premade_annotations_df, premade_annotations_label,automated_df = False, isolation_parameters, log_scale, save_fig, normalize_local_scores)`

### [`plot_bird_label_scores`](https://github.com/UCSD-E4E/PyHa/blob/main/PyHa/visualizations.py)
*Found in [`visualizations.py`](https://github.com/UCSD-E4E/PyHa/blob/main/PyHa/visualizations.py)*

This function visualizes automated and human annotation scores across an audio clip..

| Parameter | Type |  Description |
| --- | --- | --- |
| `automated_df` | Dataframe | Dataframe of automated labels for one clip. |
| `human_df` | Dataframe | Dataframe of human labels for one clip. |
| `plot_fig` | boolean | Whether or not the efficiency statistics should be displayed. |
| `save_fig` | boolean | Whether or not the plot should be saved within a file. |

This function returns a dataframe with statistics comparing the automated and human labeling. 

Usage: `plot_bird_label_scores(automated_df,human_df,save_fig)`
 
</details>


All files in the `microfaune_package` directory are from the [microfaune repository](https://github.com/microfaune/microfaune), and their associated documentation can be found there.  
