import numpy as np
import json
from scipy import interpolate
from scipy.io import wavfile
import random

from microfaune import audio, plot


def read_json_file(json_file_path):
    """ Read json file with labels.

                Parameters
                ----------
                json_file_path : str
                    Path of json file.

                Returns:
                -------
                data_dict : list
                    List of labels, each label is a dictionary item with entries 'id', 'start', 'end', 'annotation'
    """
    with open(json_file_path) as json_data:
        data_dict = json.load(json_data)
    return data_dict


def number_labels(json_file_path):
    """ Count labels in json file.

            Parameters
            ----------
            json_file_path : str
                Path of json file.

            Returns:
            -------
            nb_labels : int
                Number of labels in json file
        """
    data_dict = read_json_file(json_file_path)
    nb_labels = len(data_dict)
    return nb_labels


def prop_labeled(json_file_path, audio_file_path):
    """ Compute proportion of the ratio of labels duration compared to total duration of audio

        Parameters
        ----------
        json_file_path : str
            Path of json file.
        audio_file_path : str
            Path of audio file.

        Returns:
        -------
        ratio : float
            Ratio of duration of labeled segments and total duration of audio.
    """

    fs, data = audio.load_audio(audio_file_path)
    total_duration = len(data) / fs

    data_dict = read_json_file(json_file_path)

    bird_song_duration = 0

    for label in data_dict:
        bird_song_duration += label['end'] - label['start']
    ratio = round(bird_song_duration / total_duration, 2)

    return ratio


def charac_function_audio(json_file_path, audio_file_path):
    """ Derive the characteristic function from the labels in json file
            Parameters
            ----------
            json_file_path : str
                Path of json file.
            audio_file_path : str
                Path of audio file.

            Returns:
            -------
            charac_func : numpy array (nb bites in audio,1)
                Characteristic function derived on audio time scale, equal to 1 on labeled segments, 0 elsewhere
    """
    fs, data = audio.load_audio(audio_file_path)
    charac_func = np.zeros((len(data), 1))

    data_dict = read_json_file(json_file_path)

    for label in data_dict:
        indx_start = int(label['start'] * fs)
        indx_end = int(label['end'] * fs)
        charac_func[indx_start:indx_end + 1, 0] = 1

    return charac_func


def charac_function_spec(audio_file_path, window_length, overlap, charac_func_audio):
    """ Derive the characteristic function from the labels in json file
            Parameters
            ----------
            audio_file_path : str
                Path of audio file.
            window_length : float
                Length of the FFT window in seconds.
            overlap : float
                Overlap of the FFT windows.
            charac_func_audio : numpy array (nb of samples in audio file,1)
                Characteristic function derived in audio time scale, equal to 1 on labeled segments, 0 elsewhere

            Returns:
            -------
            charac_func_spec : numpy array (nb of samples in spectrogram,1)
                Characteristic function derived in spectrogram time scale, equal to 1 on labeled segments, 0 elsewhere
        """

    fs, data = audio.load_audio(audio_file_path)
    duration = len(data) / fs
    size_spec = 2 + int(duration / (window_length * (1 - overlap)))
    size_audio = len(charac_func_audio)
    regroup_factor = int(size_audio / size_spec)

    charac_func_spec = np.zeros((size_spec, 1))

    for i in range(size_spec):
        label_local = np.mean(charac_func_audio[i * regroup_factor: (i + 1) * regroup_factor])
        if label_local > 0.5:
            charac_func_spec[i] = 1

    return charac_func_spec


def charac_function_fs(old_fs, new_fs, charac_func):
    """ Convert the scale of a characteristic function from spec time scale to another time scale with the desired sampling rate
        Parameters
        ----------
            old_fs: int
                Sampling frequency in Hz on which charac_func is derived.
            new_fs: int
                Sampling frequency in Hz to which charac_func needs to be  converted.
            charac_func: numpy array (nb of samples, 1)
                Characteristic function with sampling rate old_fs

        Returns:
        -------
            charac_func_new_fs : numpy array (nb of samples, 1)
                Characteristic function with sampling rate new_fs
        """

    duration = len(charac_func) / old_fs

    t_old = np.linspace(0, duration, num = len(charac_func))
    t_new = np.linspace(0, duration, num = duration*new_fs)
    f = interpolate.interp1d(t_old, charac_func[:,0])
    charac_func_fs = np.zeros((int(duration*new_fs), 1))
    charac_func_fs[:,0] = f(t_new)

    return charac_func_fs


def charac_function_spec_fs(fs, window_length, overlap, charac_func_spec):
    """ Convert the scale of a characteristic function from spec time scale to another time scale with the desired sampling rate
        Parameters
        ----------
            fs: int
                Sampling frequency in Hz.
            window_length : float
                Length of the FFT window in seconds.
            overlap : float
                Overlap of the FFT windows.
            charac_func_spec: numpy array (nb of samples in spectrogram, 1)
                Characteristic function derived in spectrogram time scale, equal to 1 on labeled segments, 0 elsewhere

        Returns:
        -------
            charac_func_fs : numpy array (nb of samples, 1)
                Characteristic function derived with the desired sampling rate
        """

    dt_spec = window_length * (1 - overlap)
    fs_spec = 1./dt_spec
    charac_func_fs = charac_function_fs(fs_spec, fs, charac_func_spec)

    return charac_func_fs


def create_wav_with_label(fs, charac_func_fs, file_path):
    """ Create a false wav file with characteristic function.
        Used in Audacity to plot spectrogram and label at the same time
        Parameters
        ----------
                fs: int
                    Sampling frequency in Hz.
                charac_func_fs : numpy array (fs*duration,1)
                    Characteristic function derived with the desired sampling rate
                file_path : str
                    Path of the wav file path saved
    """
    wavfile.write(file_path, fs, charac_func_fs[:, 0])

    return None


def create_label_json(path, labels=[], start_time=None):
    """ Write a json file from a list of labels
    Parameters
    ----------
        path: str
            Path of the future json file.
        labels: list
            List of labels, each label is a dictionary with keys 'id', 'start', 'end' and 'annotation'
        start_time: float
            Start time of audio extract in seconds
    """
    data_dict = []

    for label in labels:
        data_dict.append({
            'id': 'none',
            'start': label['start'] - start_time,
            'end': label['end'] - start_time,
            'annotation': 'bird'
        })

    with open(path, 'w') as outfile:
        json.dump(data_dict, outfile)

    return None


def extract_labels(json_path, start_time, duration):
    """ Extract the labels of the audio extract from the json creating at labeling
        Parameters
        ----------
            json_path: str
                Path of the json file.
            start_time: float
                Start time of audio extract in seconds
            duration: float
                Duration of audio extract in seconds
        Returns
        -------
            labels: list
                List of labelson the audio extract, each label is a dictionary with keys 'id', 'start', 'end' and 'annotation'
        """
    data_dict = read_json_file(json_path)
    labels = []

    for label in data_dict:

        start_label = label['start']
        end_label = label['end']

        if start_time < start_label:
            if start_time + duration > end_label:
                labels.append({
                    'start': max(start_time, start_label),
                    'end': min(start_time + duration, end_label)
                })
        else:
            if end_label > start_time:
                labels.append({
                    'start': max(start_time, start_label),
                    'end': min(start_time + duration, end_label)
                })

    return labels


def extract_audio(wav_folder_path, json_folder_path, wav_file, json_file, path_database, fs_filter, duration_extract, threshold,
                  nbre_extracts_pos, nbre_extracts_neg, max_counts):
    """ Extract the labels of the audio extract from the json creating at labeling
        Parameters
        ----------
            wav_folder_path: str
                Path of wav audio files
			json_folder_path: str
                Path of wav json files
            wav_file: str
                Name of the wav file
            json_file: str
                Name of the json file
            path_database: str
                Path used to save database
            fs_filter: int
                Sampling rate of the filter in Hz
            duration_extract: float
                Duration of audio to extract in seconds
            threshold: float
                Threshold for the proportion of labelised segments to avoid to have very short bird song in positive audio
            nbre_extracts_pos:
                Number of audio with bird songs to extract
            nbre_extracts_neg:
                Number of audio without bird songs to extract
            max_counts:
                Maximum try for random extraction
    """
    audio_fs, audio_data = wavfile.read(wav_folder_path + wav_file)

    duration_audio = len(audio_data) / audio_fs
    charac_audio = charac_function_audio(json_folder_path + json_file, wav_folder_path + wav_file)
    charac_audio = charac_function_fs(audio_fs, fs_filter, charac_audio)

    positive_extracts = 0
    negative_extracts = 0
    counts = 0

    test_label = np.where(charac_audio > 0)
    prop_labelised = len(test_label[0]) / len(charac_audio)
    # if no bird label in audio no positive audio extracted
    if len(test_label[0]) == 0:
        positive_extracts = nbre_extracts_pos
    # if there are too many labels no negative audio extracted
    elif duration_audio * (1 - prop_labelised) < duration_extract:
        negative_extracts = nbre_extracts_neg

    while (positive_extracts < nbre_extracts_pos or negative_extracts < nbre_extracts_neg) and counts < max_counts:

        start_time = round(random.random() * (duration_audio - duration_extract), 2)

        charac_filter = np.zeros((len(charac_audio), 1))
        indx_start = int(start_time * fs_filter)
        indx_end = int((duration_extract + start_time) * fs_filter) + 1
        charac_filter[indx_start:indx_end] = 1

        apply_filter = charac_filter * charac_audio

        labeled_segment = np.where(apply_filter > 0)
        labeled_prop = len(labeled_segment[0]) / (duration_extract * fs_filter)

        if labeled_prop > threshold and positive_extracts < nbre_extracts_pos:
            path_wav = path_database + 'positive/' + wav_file[:-4] + '_pos_t0_'+ str(start_time) + '.wav'
            indx_start = int(start_time * audio_fs)
            indx_end = int((duration_extract + start_time) * audio_fs) + 1
            wavfile.write(path_wav, audio_fs, audio_data[indx_start:indx_end])

            path_json = path_database + 'json/' + wav_file[:-4] + '_pos_t0_'+ str(start_time) + '.json'
            labels_ext = extract_labels(json_folder_path + json_file, start_time, duration_extract)
            create_label_json(path_json, labels_ext, start_time)

            positive_extracts += 1
            counts += 1

        elif labeled_prop == 0 and negative_extracts < nbre_extracts_neg:
            path_wav = path_database + 'negative/' + wav_file[:-4] + '_neg_t0_'+ str(start_time) + '.wav'
            indx_start = int(start_time * audio_fs)
            indx_end = int((duration_extract + start_time) * audio_fs) + 1
            wavfile.write(path_wav, audio_fs, audio_data[indx_start:indx_end])

            path_json = path_database + 'json/' + wav_file[:-4] + '_neg_t0_'+ str(start_time) + '.json'
            create_label_json(path_json)

            negative_extracts += 1
            counts += 1

        else:
            counts += 1

    return None