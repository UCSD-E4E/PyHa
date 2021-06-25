# PyHa
<!-- ## Automated Audio Labeling System -->
A tool designed to convert audio-based "weak" labels to "strong" moment-to-moment labels. Provides a pipeline to compare automated moment-to-moment labels to human labels. Current proof of concept work being fulfilled on Bird Audio clips using Microfaune predictions.

PyHa = Python + Piha (referring to a bird species of our interest known as the screaming-piha)

## Installation and Setup

## Functions

### `isolation_parameters`

Many of the functions take in the `isolation_parameters` argument, and as such it will be defined globally here. 

The `isolation_parameters` dictionary is as follows: 

```
isolation_parameters = {
    "technique" : "",
    "threshold_type" : "",
    "threshold_const" : ,
    "threshold_min" : ,
    "bi_directional_jump" : ,
    "chunk_size" : 
} 
```
The `technique` parameter can be: Simple, Stack, Steinberg, and Chunk. 
The `threshold_type` parameter can be: median, mean, average, standard deviation, or pure.
The remaining parameters are floats representing their respective values. 


### `isolate` 

This function is the wrapper function for all the audio isolation techniques, and will call the respective function based on its parameters. 

| Parameter | Type |  Description |
| --- | --- | --- |
| `local_scores` | list of floats | Local scores of the audio clip as determined by Microfaune Recurrent Neural Network. |
| `SIGNAL` | list of ints | Samples that make up the audio signal. |
| `SAMPLE_RATE` | int | Sampling rate of the audio clip, usually 44100. |
| `audio_dir` | string | Directory of the audio clip. |
| `filename` | string | Name of the audio clip file. |
| `isolation_parameters` | dict | Python Dictionary that controls the various label creation techniques. |

### `threshold`

This function takes in the local score array output from a neural network and determines the threshold at which we determine a local score to be a positive ID of a class of interest. Most proof of concept work is dedicated to bird presence. Threshold is determined by "threshold_type" and "threshold_const" from the isolation_parameters dictionary.

| Parameter | Type | Description | 
| --- | --- | --- | 
| `local_scores` | list of floats | Local scores of the audio clip as determined by Microfaune Recurrent Neural Network. | 
| `isolation parameters` | dict | Python Dictionary that controls the various label creation techniques. | 

