 <img src="https://github.com/UCSD-E4E/PyHa/blob/main/Logos/PyHa.svg" alt="PyHa logo" title="PyHa" align="right" height="300" />

# PyHa
<!-- ## Automated Audio Labeling System -->

A tool designed to convert audio-based "weak" labels to "strong" moment-to-moment labels. Provides a pipeline to compare automated moment-to-moment labels to human labels. Current proof of concept work being fulfilled on Bird Audio clips using Microfaune predictions.

This package is being developed and maintained by the [Engineers for Exploration Acoustic Species Identification Team](http://e4e.ucsd.edu/acoustic-species-identification) in collaboration with the [San Diego Zoo Wildlife Alliance](https://sandiegozoowildlifealliance.org/).

PyHa = Python + Piha (referring to a bird species of our interest known as the screaming-piha)

## Contents
- [Installation and Setup](#installation-and-setup)
- [Functions](#functions)
- [Examples](#examples)

## Installation and Setup
1. Navigate to a desired folder and clone the repository onto your local machine. `git clone https://github.com/UCSD-E4E/PyHa.git`
2. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Miniforge](https://github.com/conda-forge/miniforge).
3. Install the conda environment by running `conda env create --file conda_environments/{filename}`, where `filename` is the name of the yaml containing the environment for your OS.
4. Run `conda activate species-id` to activate the conda environment used to develop the package.
5. Here you can download the Xeno-canto Screaming Piha test set used in our demos: https://drive.google.com/drive/u/0/folders/1lIweB8rF9JZhu6imkuTg_No0i04ClDh1
6. Run `jupyter notebook` while in the proper folder to activate the PyHa_Tutorial.ipynb notebook and make sure PyHa is running properly. Make sure the paths are properly aligned to the TEST folder in the notebook as well as in the ScreamingPiha_Manual_Labels.csv file

## Functions
![design](https://user-images.githubusercontent.com/44332326/126560960-e9816f7e-c31b-40ee-804d-6947053323c2.png)

*This image shows the design of the automated audio labeling system.*

### `isolation_parameters`

Many of the functions take in the `isolation_parameters` argument, and as such it will be defined globally here.

The `isolation_parameters` dictionary definition depends on the model used. The currently supported models are BirdNET-Analyzer, BirdNET-Lite, Microfaune, and TweetyNET.


The BirdNET `isolation_parameters` dictionary is as follows:

``` python
isolation_parameters = {
    "model" : "birdnet",
    "type" : "lite",
    "output_path" : "",
    "lat" : 0.0,
    "lon" : 0.0,
    "week" : 0,
    "overlap" : 0.0,
    "sensitivity" : 0.0,
    "min_conf" : 0.0,
    "custom_list" : "",
    "filetype" : "",
    "num_predictions" : 0,
    "write_to_csv" : False
}
```

The `type` parameter can be: Analyzer or Lite. This input must be a string in all lowercase, and determines which BirdNET model to use.  

<br>


The Microfaune `isolation_parameters` dictionary is as follows:

``` python
isolation_parameters = {
    "model" : "microfaune",
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

<br>

The TweetyNET `isolation_parameters` dictionary is as follows:

```python
isolation_parameters = {
    "model" : "tweetynet",
    "tweety_output": False,
    "technique" : "",
    "threshold_type" : "",
    "threshold_const" : 0.0,
    "threshold_min" : 0.0,
    "window_size" : 0.0,
    "chunk_size" : 0.0,
}
```

The `tweety_output` parameter sets whether to use TweetyNET's original output or isolation techniques. If set to `False`, TweetyNET will use the specified `technique` parameter.

<!-- annotation_post_processing.py file -->

<details>
 <summary>annotation_post_processing.py file</summary>

### [`annotation_chunker`](https://github.com/UCSD-E4E/PyHa/blob/main/PyHa/annotation_post_processing.py)
*Found in [`annotation_post_processing.py`](https://github.com/UCSD-E4E/PyHa/blob/main/PyHa/annotation_post_processing.py)*

This function converts a Kaleidoscope-formatted Dataframe containing annotations to uniform chunks of `chunk_length`. Drops any annotation that less than chunk_length.

| Parameter | Type |  Description |
| --- | --- | --- |
| `kaleidoscope_df` | Dataframe | Dataframe of automated or human labels in Kaleidoscope format |
| `chunk_length` | int | Duration in seconds of each annotation chunk |

This function returns a dataframe with annotations converted to uniform second chunks.

Usage: `annotation_chunker(kaleidoscope_df, chunk_length)`
</details>


<!-- IsoAutio.py file -->

<details>
 <summary>IsoAutio.py file</summary>

### [`isolate`](https://github.com/UCSD-E4E/PyHa/blob/main/PyHa/IsoAutio.py)
*Found in [`IsoAutio.py`](https://github.com/UCSD-E4E/PyHa/blob/main/PyHa/IsoAutio.py)*

This function is the wrapper function for audio isolation techniques, and will call the respective function based on `isolation_parameters` "technique" key.

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

This function uses a technique created by Jacob Ayers. Attempts to produce automated annotations of an audio clip based on local score array outputs from a neural network. It is called by the `isolate` function when `isolation_parameters['technique'] == stack`.

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

This function uses a technique created by Jacob Ayers. Attempts to produce automated annotations of an audio clip based on local score array outputs from a neural network. It is called by the `isolate` function when `isolation_parameters['technique'] == chunk`.

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

This function generates labels across a folder of audio clips determined by the model and other parameters specified in the `isolation_parameters` dictionary.

| Parameter | Type |  Description |
| --- | --- | --- |
| `audio_dir` | string | Directory with wav audio files |
| `isolation_parameters` | dict | Python Dictionary that controls the various label creation techniques. |
| `manual_id` | string | controls the name of the class written to the pandas dataframe |
| `weight_path` | string | File path of weights to be used by the RNNDetector for determining presence of bird sounds.
| `normalized_sample_rate` | int | Sampling rate that the audio files should all be normalized to.
| `normalize_local_scores` | boolean | Set whether or not to normalize the local scores.

This function returns a dataframe of automated labels for the audio clips in audio_dir.

Usage: `generate_automated_labels(audio_dir, isolation_parameters, manual_id, weight_path, normalized_sample_rate, normalize_local_scores)`

### [`generate_automated_labels_birdnet`](https://github.com/UCSD-E4E/PyHa/blob/main/PyHa/IsoAutio.py)
*Found in [`IsoAutio.py`](https://github.com/UCSD-E4E/PyHa/blob/main/PyHa/IsoAutio.py)*

This function is called by `generate_automated_labels` if `isolation_parameters["model"]` is set to `birdnet`. It generates bird labels across a folder of audio clips using BirdNET-Analyzer or BirdNET-Lite given the isolation parameters.

| Parameter | Type |  Description |
| --- | --- | --- |
| `audio_dir` | string | Directory with wav audio files |
| `isolation_parameters` | dict | Python Dictionary that controls the various label creation techniques. |

This function returns a dataframe of automated labels for the audio clips in audio_dir.

Usage: `generate_automated_labels_birdnet(audio_dir, isolation_parameters)`


### [`generate_automated_labels_microfaune`](https://github.com/UCSD-E4E/PyHa/blob/main/PyHa/IsoAutio.py)
*Found in [`IsoAutio.py`](https://github.com/UCSD-E4E/PyHa/blob/main/PyHa/IsoAutio.py)*

This function is called by `generate_automated_labels` if `isolation_parameters["model"]` is set to `microfaune`. It applies the isolation technique determined by the `isolation_parameters` dictionary across a whole folder of audio clips.

| Parameter | Type |  Description |
| --- | --- | --- |
| `audio_dir` | string | Directory with wav audio files |
| `isolation_parameters` | dict | Python Dictionary that controls the various label creation techniques. |
| `manual_id` | string | controls the name of the class written to the pandas dataframe |
| `weight_path` | string | File path of weights to be used by the RNNDetector for determining presence of bird sounds.
| `normalized_sample_rate` | int | Sampling rate that the audio files should all be normalized to.
| `normalize_local_scores` | boolean | Set whether or not to normalize the local scores.

This function returns a dataframe of automated labels for the audio clips in audio_dir.

Usage: `generate_automated_labels_microfaune(audio_dir, isolation_parameters, manual_id, weight_path, normalized_sample_rate, normalize_local_scores)`


### [`generate_automated_labels_tweetynet`](https://github.com/UCSD-E4E/PyHa/blob/main/PyHa/IsoAutio.py)
*Found in [`IsoAutio.py`](https://github.com/UCSD-E4E/PyHa/blob/main/PyHa/IsoAutio.py)*

This function is called by `generate_automated_labels` if `isolation_parameters["model"]` is set to `tweetynet`. It applies the isolation technique determined by the `isolation_parameters` dictionary across a whole folder of audio clips.

| Parameter | Type |  Description |
| --- | --- | --- |
| `audio_dir` | string | Directory with wav audio files |
| `isolation_parameters` | dict | Python Dictionary that controls the various label creation techniques. |
| `manual_id` | string | controls the name of the class written to the pandas dataframe |
| `weight_path` | string | File path of weights to be used by the RNNDetector for determining presence of bird sounds.
| `normalized_sample_rate` | int | Sampling rate that the audio files should all be normalized to.
| `normalize_local_scores` | boolean | Set whether or not to normalize the local scores.

This function returns a dataframe of automated labels for the audio clips in audio_dir.

Usage: `generate_automated_labels_tweetynet(audio_dir, isolation_parameters, manual_id, weight_path, normalized_sample_rate, normalize_local_scores)`



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

### [`annotation_duration_statistics`](https://github.com/UCSD-E4E/PyHa/blob/main/PyHa/statistics.py)
*Found in [`statistics.py`](https://github.com/UCSD-E4E/PyHa/blob/main/PyHa/statistics.py)*

This function calculates basic statistics related to the duration of annotations of a Pandas Dataframe compatible with PyHa.

| Parameter | Type |  Description |
| --- | --- | --- |
| `df` | Pandas Dataframe | Dataframe of automated labels or manual labels. |

This function returns a Pandas Dataframe containing count, mean, mode, standard deviation, and IQR values based on annotation duration.

Usage: `annotation_duration_statistics(df)`

### [`clip_general`](https://github.com/UCSD-E4E/PyHa/blob/main/PyHa/statistics.py)
*Found in [`statistics.py`](https://github.com/UCSD-E4E/PyHa/blob/main/PyHa/statistics.py)*

This function generates a dataframe with statistics relating to the efficiency of the automated label compared to the human label. These statistics include true positive, false positive, false negative, true negative, union, precision, recall, F1, and Global IoU for general clip overlap.

| Parameter | Type |  Description |
| --- | --- | --- |
| `automated_df` | Dataframe | Dataframe of automated labels for one clip |
| `human_df` | Dataframe | Dataframe of human labels for one clip. |

This function returns a dataframe with general clip overlap statistics comparing the automated and human labeling.

Usage: `clip_general(automated_df, human_df)`

### [`automated_labeling_statistics`](https://github.com/UCSD-E4E/PyHa/blob/main/PyHa/statistics.py)
*Found in [`statistics.py`](https://github.com/UCSD-E4E/PyHa/blob/main/PyHa/statistics.py)*

This function allows users to easily pass in two dataframes of manual labels and automated labels, and returns a dataframe with statistics examining the efficiency of the automated labelling system compared to the human labels for multiple clips.

| Parameter | Type |  Description |
| --- | --- | --- |
| `automated_df` | Dataframe | Dataframe of automated labels of multiple clips. |
| `manual_df` | Dataframe |  Dataframe of human labels of multiple clips. |
| `stats_type` | String | String that determines which type of statistics are of interest |
| `threshold` | float | Defines a threshold for certain types of statistics |

This function returns a dataframe of statistics comparing automated labels and human labels for multiple clips.

The `stats_type` parameter can be set as follows:
| Name | Description |
| --- | --- |
|`"IoU"`| Default. Compares the intersection over union of automated annotations with respect to manual annotations for individual clips. |
|`"general"` | Consolidates all automated annotations and compares them to all of the manual annotations that have been consolidated across a clip. |


Usage: `automated_labeling_statistics(automated_df, manual_df, stats_type, threshold)`

### [`global_dataset_statistics`](https://github.com/UCSD-E4E/PyHa/blob/main/PyHa/statistics.py)
*Found in [`statistics.py`](https://github.com/UCSD-E4E/PyHa/blob/main/PyHa/statistics.py)*

This function takes in a dataframe of efficiency statistics for multiple clips and outputs their global values.

| Parameter | Type |  Description |
| --- | --- | --- |
| `statistics_df` | Dataframe | Dataframe of statistics value for multiple audio clips as returned by the function automated_labelling_statistics. |

This function returns a dataframe of global statistics for the multiple audio clips' labelling.

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
| `manual_df `| Dataframe | Dataframe of human labels for an audio clip. |
| `threshold` | float | IoU threshold for determining true positives, false positives, and false negatives. |

This function returns a dataframe of clip statistics such as True Positive, False Negative, False Positive, Precision, Recall, and F1 values for an audio clip.

Usage: `matrix_IoU_Scores(IoU_Matrix, manual_df, threshold)`

### [`clip_catch`](https://github.com/UCSD-E4E/PyHa/blob/main/PyHa/statistics.py)
*Found in [`statistics.py`](https://github.com/UCSD-E4E/PyHa/blob/main/PyHa/statistics.py)*

This function determines whether or not a human label has been found across all of the automated labels.

| Parameter | Type |  Description |
| --- | --- | --- |
| `automated_df` | Dataframe | Dataframe of automated labels for one clip |
| `human_df` | Dataframe | Dataframe of human labels for one clip. |

This function returns a Numpy Array of statistics regarding the amount of overlap between the manual and automated labels relative to the number of samples.

Usage: `clip_catch(automated_df,manual_df)`

### [`global_statistics`](https://github.com/UCSD-E4E/PyHa/blob/main/PyHa/statistics.py)
*Found in [`statistics.py`](https://github.com/UCSD-E4E/PyHa/blob/main/PyHa/statistics.py)*

This function takes the output of dataset_IoU Statistics and outputs a global count of true positives and false positives, as well as computes the precision, recall, and f1 metrics across the dataset.

| Parameter | Type |  Description |
| --- | --- | --- |
| `statistics_df` | Dataframe | Dataframe of matrix IoU scores for multiple clips. |

This function returns a dataframe of global IoU statistics which include the number of true positives, false positives, and false negatives. Contains Precision, Recall, and F1 metrics as well

Usage: `global_statistics(statistics_df)`

### [`dataset_Catch`](https://github.com/UCSD-E4E/PyHa/blob/main/PyHa/statistics.py)
*Found in [`statistics.py`](https://github.com/UCSD-E4E/PyHa/blob/main/PyHa/statistics.py)*

This function determines the overlap of each human label with respect to all of the human labels in a clip across a large number of clips.

| Parameter | Type |  Description |
| --- | --- | --- |
| `automated_df` | Dataframe | Dataframe of automated labels for one clip |
| `human_df` | Dataframe | Dataframe of human labels for one clip. |

This function returns a dataframe of human labels with a column for the catch values of each label.

Usage: `dataset_Catch(automated_df, manual_df)`

### [`clip_statistics`](https://github.com/UCSD-E4E/PyHa/blob/main/PyHa/statistics.py)
*Found in [`statistics.py`](https://github.com/UCSD-E4E/PyHa/blob/main/PyHa/statistics.py)*

| Parameter | Type |  Description |
| --- | --- | --- |
| `automated_df` | Dataframe | Dataframe of automated labels for multiple classes. |
| `human_df` | Dataframe | Dataframe of human labels for multiple classes. |
| `stats_type` | String | String that determines which statistics are of interest. |
| `threshold` | float | Defines a threshold for certain types of statistics. |

This function returns a dataframe with clip overlap statistics comparing automated and human labeling for multiple classes

The `stats_type` parameter can be set as follows:
| Name | Description |
| --- | --- |
|`"IoU"`| Default. Compares the intersection over union of automated annotations with respect to manual annotations for individual clips. |
|`"general"` | Consolidates all automated annotations and compares them to all of the manual annotations that have been consolidated across a clip. |

Usage: `clip_statistics(automated_df, manual_df, stats_type, threshold)`

### [`class_statistics`](https://github.com/UCSD-E4E/PyHa/blob/main/PyHa/statistics.py)
*Found in [`statistics.py`](https://github.com/UCSD-E4E/PyHa/blob/main/PyHa/statistics.py)*

| Parameter | Type |  Description |
| --- | --- | --- |
| `clip_statistics` | Dataframe | Dataframe of multi-class statistics values for audio clips as returned by the function clip_statistics. |

This function returns a dataframe of global efficacy values for multiple classes.

Usage: `class_statistics(clip_statistics)`

</details>


<!-- visualizations.py file -->
<details>
 <summary>visualizations.py file</summary>

### [`spectrogram_graph`](https://github.com/UCSD-E4E/PyHa/blob/main/PyHa/visualizations.py)
*Found in [`visualizations.py`](https://github.com/UCSD-E4E/PyHa/blob/main/PyHa/visualizations.py)*

This function produces graphs with the spectrogram of an audio clip. It is now integrated with Pandas so you can visualize human and automated annotations.

| Parameter | Type |  Description |
| --- | --- | --- |
| `clip_name`  | string | Directory of the clip. |
| `sample_rate` | int | Sample rate of the audio clip, usually 44100. |
| `samples` | list of ints | Each of the samples from the audio clip. |
| `automated_df` | Dataframe | Dataframe of automated labelling of the clip. |
| `premade_annotations_df` | Dataframe | Dataframe labels that have been made outside of the scope of this function. |
| `premade_annotations_label` | string | Descriptor of premade_annotations_df |
| `save_fig`  | boolean | Whether the clip should be saved in a directory as a png file. |

This function does not return anything.

Usage: `spectrogram_graph(clip_name, sample_rate, samples, automated_df, premade_annotations_df, premade_annotations_label, save_fig, normalize_local_scores)`


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

### [`spectrogram_visualization`](https://github.com/UCSD-E4E/PyHa/blob/main/PyHa/visualizations.py)
*Found in [`visualizations.py`](https://github.com/UCSD-E4E/PyHa/blob/main/PyHa/visualizations.py)*

This is the wrapper function for the local_line_graph and spectrogram_graph functions for ease of use. Processes clip for local scores to be used for the local_line_graph function.

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

Usage: `spectrogram_visualization(clip_path, weight_path, premade_annotations_df, premade_annotations_label,automated_df = False, isolation_parameters, log_scale, save_fig, normalize_local_scores)`

### [`binary_visualization`](https://github.com/UCSD-E4E/PyHa/blob/main/PyHa/visualizations.py)
*Found in [`visualizations.py`](https://github.com/UCSD-E4E/PyHa/blob/main/PyHa/visualizations.py)*

This function visualizes automated and human annotation scores across an audio clip.

| Parameter | Type |  Description |
| --- | --- | --- |
| `automated_df` | Dataframe | Dataframe of automated labels for one clip. |
| `human_df` | Dataframe | Dataframe of human labels for one clip. |
| `plot_fig` | boolean | Whether or not the efficiency statistics should be displayed. |
| `save_fig` | boolean | Whether or not the plot should be saved within a file. |

This function returns a dataframe with statistics comparing the automated and human labeling.

Usage: `binary_visualization(automated_df,human_df,save_fig)`

### [`annotation_duration_histogram`](https://github.com/UCSD-E4E/PyHa/blob/main/PyHa/visualizations.py)
*Found in [`visualizations.py`](https://github.com/UCSD-E4E/PyHa/blob/main/PyHa/visualizations.py)*

This function builds a histogram to visualize the length of annotations.

| Parameter | Type |  Description |
| --- | --- | --- |
| `annotation_df` | Dataframe | Dataframe of automated or human labels. |
| `n_bins` | int | Number of histogram bins in the final histogram. |
| `min_length` | int | Minimum length of the audio clip. |
| `max_length` | int | Maximum length of the audio clip. |
| `save_fig` | boolean | Whether or not the plot should be saved within a file. |
| `filename` | String | Name of the file to save the histogram to. |

This function returns a histogram with the length of the annotations.

Usage: `binary_visualization(annotation_df, n_bins, min_length, max_length, save_fig, filename)`
</details>

All files in the `birdnet_analyzer` directory are from the [BirdNET-Analyzer repository](https://github.com/kahst/BirdNET-Analyzer), and their associated documentation can be found there.  

All files in the `birdnet_lite` directory are from a [modified version](https://github.com/UCSD-E4E/BirdNET-Lite) of the [BirdNET Lite repository](https://github.com/kahst/BirdNET-Lite), and their associated documentation can be found there.  

All files in the `microfaune_package` directory are from the [microfaune repository](https://github.com/microfaune/microfaune), and their associated documentation can be found there.    

All files in the `tweetynet` directory are from the [TweetyNet repository](https://github.com/yardencsGitHub/tweetynet), and their associated documentation can be found there.  

## Examples
*These examples were created on an Ubuntu 16.04 machine. Results may vary between different Operating Systems and Tensorflow versions.*

Examples using Microfaune were created using the following dictionary for `isolation_parameters`:

```json
isolation_parameters = {
     "model" : "microfaune",
     "technique" : "steinberg",
     "threshold_type" : "median",
     "threshold_const" : 2.0,
     "threshold_min" : 0.0,
     "window_size" : 2.0,
     "chunk_size" : 5.0
 }
```

### To generate automated labels and get manual labels:
```python
automated_df = generate_automated_labels(path,isolation_parameters,normalize_local_scores=True)
manual_df = pd.read_csv("ScreamingPiha_Manual_Labels.csv")
```

### Function that gathers statistics about the duration of labels
```python
annotation_duration_statistics(automated_df)
```
![image](https://user-images.githubusercontent.com/44332326/127575042-96d46c11-cc3e-470e-a10d-31f1d7ef052a.png)

```python
annotation_duration_statistics(manual_df)
```
![image](https://user-images.githubusercontent.com/44332326/127575181-9ce49439-5396-425d-a1d5-148ef47db373.png)

### Function that converts annotations into 3 second chunks
```python
annotation_chunker(automated_df, 3)
```
![annotation chunker](https://user-images.githubusercontent.com/33042752/176480538-671b731d-89ad-402c-a603-8a0ee35124f6.png)

### Helper function to convert to kaleidoscope-compatible format
```python
kaleidoscope_conversion(manual_df)
```
![image](https://user-images.githubusercontent.com/44332326/127575089-023bc41a-5aaf-43fc-8ea6-3a8b9dd69b66.png)


### Baseline Graph without any annotations
```python
clip_path = "./TEST/ScreamingPiha2.wav"
spectrogram_visualization(clip_path)
```
![image](https://user-images.githubusercontent.com/44332326/126691710-01c4e88c-0c54-4539-a24d-c682cd93aebf.png)


### Baseline Graph with log scale
```python
spectrogram_visualization(clip_path,log_scale = True)
```
![image](https://user-images.githubusercontent.com/44332326/126691745-b1cb8be6-c52f-45cc-b7e6-9973070aacc9.png)


### Baseline graph with normalized local score values between [0,1]
```python
spectrogram_visualization(clip_path, normalize_local_scores = True)
```
![image](https://user-images.githubusercontent.com/44332326/126691803-b01c96e8-31bc-45dd-b936-58f0d9a153b4.png)

### Graph with Automated Labeling
```python
spectrogram_visualization(clip_path,automated_df = True, isolation_parameters = isolation_parameters)
```
![image](https://user-images.githubusercontent.com/44332326/127575291-8e83e9ed-0ca3-4caf-a3fb-a83785123f33.png)


### Graph with Human Labelling
```python
spectrogram_visualization(clip_path, premade_annotations_df = manual_df[manual_df["IN FILE"] == "ScreamingPiha2.wav"],premade_annotations_label = "Piha Human Labels")
```
![image](https://user-images.githubusercontent.com/44332326/127575314-712aeaf8-f88c-44ef-8afa-3c3da86000cb.png)


### Graph with Both Automated and Human Labels
*Legend:*

    - Orange ==> True Positive
    - Red ==> False Negative
    - Yellow ==> False Positive
    - White ==> True Negative

```python
spectrogram_visualization(clip_path,automated_df = True,isolation_parameters=isolation_parameters,premade_annotations_df = manual_df[manual_df["IN FILE"] == "ScreamingPiha2.wav"])
```
![image](https://user-images.githubusercontent.com/44332326/127575359-9dbfd330-f9e1-423c-a063-62b2a9af78dc.png)


### Another Visualization of True Positives, False Positives, False Negatives, and True Negatives
```python
automated_piha_df = automated_df[automated_df["IN FILE"] == "ScreamingPiha2.wav"]
manual_piha_df = manual_df[manual_df["IN FILE"] == "ScreamingPiha2.wav"]
piha_stats = binary_visualization(automated_piha_df,manual_piha_df)
```
![image](https://user-images.githubusercontent.com/44332326/127575392-2c5df40c-27e7-490f-ace5-7d9d253487f7.png)


### Function that generates statistics to gauge efficacy of automated labeling compared to human labels
```python
statistics_df = automated_labeling_statistics(automated_df,manual_df,stats_type = "general")
```
![image](https://user-images.githubusercontent.com/44332326/127575467-cb9a8637-531e-4ed7-a15e-5b5b611ba92c.png)


### Function that takes the statistical output of all of the clips and gets the equivalent global scores
```python
global_dataset_statistics(statistics_df)
```
![image](https://user-images.githubusercontent.com/44332326/127575622-5be17af4-f3a0-40ee-8a54-365825eea03e.png)


### Function that takes in the manual and automated labels for a clip and outputs human label-by-label IoU Scores. Used to derive statistics that measure how well a system is isolating desired segments of audio clips
```python
Intersection_over_Union_Matrix = clip_IoU(automated_piha_df,manual_piha_df)
```
![image](https://user-images.githubusercontent.com/44332326/127575675-71f91fc8-3143-49e6-a10b-0c1781fb498e.png)


### Function that turns the IoU Matrix of a clip into true positive and false positives values, as well as computing the precision, recall, and F1 statistics
```python
matrix_IoU_Scores(Intersection_over_Union_Matrix,manual_piha_df,0.5)
```
![image](https://user-images.githubusercontent.com/44332326/127575732-6c805bcc-a863-4c32-aba6-712ce2bac7bb.png)

### Wrapper function that takes matrix_IoU_Scores across multiple clips. Allows user to modify the threshold that determines whether or not a label is a true positive.
```python
stats_df = automated_labeling_statistics(automated_df,manual_df,stats_type = "IoU",threshold = 0.5)
```
![image](https://user-images.githubusercontent.com/44332326/127575771-9866f288-61cf-47c5-b9de-041b49e583d1.png)

### Function that takes the output of dataset_IoU Statistics and outputs a global count of true positives and false positives, as well as computing common metrics across the dataset
```python
global_stats_df = global_statistics(stats_df)
```
![image](https://user-images.githubusercontent.com/44332326/127575798-f84540ea-5121-4e7a-83c4-4ca5ad02e9d0.png)
