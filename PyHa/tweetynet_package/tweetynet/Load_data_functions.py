import os
import sys
import csv
import math
import pickle
from collections import Counter
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
import torch
# from torchsummary import summary

from .TweetyNetAudio import wav2spc, create_spec, load_wav
import random
import librosa
from .CustomAudioDataset import CustomAudioDataset


def get_frames(x, frame_size, hop_length):
    return ((x) / hop_length) + 1 #(x - frame_size)/hop_length + 1

def frames2seconds(x, sr):
    return x/sr

def compute_windows(spc,Y,win_size):
    spc = spc
    Y=Y
    win_size = win_size
    return

def window_data(spcs, ys, uids, time_bins, windowsize):
    windowed_dataset = {"uids": [], "X": [], "Y": []}
    #print("Windowing Spectrogram")
    for i in range(len(uids)):
        spc_split, Y_split, uid_split = window_spectrograms(spcs[i],ys[i], uids[i], time_bins[i], windowsize)
        windowed_dataset["X"].extend(spc_split)
        windowed_dataset["Y"].extend(Y_split)
        windowed_dataset["uids"].extend(uid_split)
    return windowed_dataset

def window_spectrograms(spc, Y, uid, time_bin, windowsize):
    computed = windowsize//time_bin #verify, big assumption. are time bins consistant?
    time_axis = int(computed*(spc.shape[1]//computed))
    freq_axis = int(spc.shape[1]//computed) # 31, 2, 19
    spc_split = np.split(spc[:,:time_axis],freq_axis,axis = 1)
    Y_split = np.split(Y[:time_axis],freq_axis)
    uid_split = [str(i) + "_" + uid for i in range(freq_axis)]
    return spc_split, Y_split, uid_split

def new_calc_Y(sr, spc, annotation, frame_size, hop_length):
    y = [0] * spc.shape[1] # array of zeros
    for i in range(len(annotation)):
        start = get_frames(annotation.loc[i, "OFFSET"] * sr, frame_size, hop_length)
        end = get_frames((annotation.loc[i, "OFFSET"] + annotation.loc[i, "DURATION"]) * sr, frame_size, hop_length)
        for j in range(math.floor(start), math.floor(end)): #CORRECT WAY TO ADD TRUE LABELS?
            y[j] = 1 
    return y

def new_compute_Y(wav, f, spc, df, SR, frame_size, hop_length):
    #df = new_find_tags(data_path, SR, csv_file)
    wav_notes = df[df['IN FILE'] == f ]
    if os.path.isfile(wav):
        #_, sr = librosa.load(wav, sr=SR)
        annotation = wav_notes[['OFFSET','DURATION','MANUAL ID']].reset_index(drop = True)
        y = new_calc_Y(SR, spc, annotation, frame_size, hop_length)
        return np.array(y)
    else:
        print("file does not exist: ", f)
    return [0] * spc.shape[1]

#tags must contain ['OFFSET','DURATION','MANUAL ID'] as headers. may want to abstract this? will take a look at dataset to make this work.
def new_find_tags(csv_path, SR):
    df = pd.read_csv(csv_path, index_col=False, usecols=["IN FILE", "OFFSET", "DURATION", "MANUAL ID","SAMPLE RATE"])
    df = df[df["SAMPLE RATE"] == SR]
    return df

def new_compute_feature(data_path, folder, csv_path, SR, n_mels, frame_size, hop_length):
    print(f"Compute features for dataset {os.path.basename(data_path)}")  
    features = {"uids": [], "X": [], "Y": [], "time_bins": []}
    df = new_find_tags(csv_path, SR) # means we will have data in kaleidoscope format that would work across the whole dataset.
    valid_filenames = set(df["IN FILE"].drop_duplicates().values.tolist())
    file_path = os.path.join(data_path, folder)
    filenames = set(os.listdir(file_path))
    true_wavs = filenames.intersection(valid_filenames)
    for f in true_wavs:
        wav = os.path.join(file_path, f)
        spc,len_audio = wav2spc(wav, fs=SR, n_mels=n_mels)
        time_bins = len_audio/spc.shape[1]
        Y = new_compute_Y(wav,f, spc, df, SR, frame_size, hop_length)
        features["uids"].append(f)
        features["X"].append(spc)
        features["Y"].append(Y)
        features["time_bins"].append(time_bins)
    return features

def new_load_dataset(data_path, folder, csv_path, SR, n_mels, frame_size, hop_length, use_dump=True):
    mel_dump_file = os.path.join(data_path, "downsampled_bin_mel_dataset.pkl")
    print(mel_dump_file)
    print(os.path.exists(mel_dump_file))
    if os.path.exists(mel_dump_file) and use_dump:
        with open(mel_dump_file, "rb") as f:
            dataset = pickle.load(f)
    else:
        dataset = new_compute_feature(data_path, folder, csv_path, SR, n_mels, frame_size, hop_length)
        with open(mel_dump_file, "wb") as f:
            pickle.dump(dataset, f)
    X = dataset['X']
    Y = dataset['Y']
    uids = dataset['uids']
    time_bins = dataset['time_bins']
    return X, Y, uids, time_bins

def new_load_and_window_dataset(data_path, folder, csv_path, SR, n_mels, frame_size, hop_length, windowsize):
    x, y, uids, time_bins = new_load_dataset(data_path, folder, csv_path, SR, n_mels, frame_size, hop_length, use_dump=True)
    dataset = window_data(x, y, uids, time_bins, windowsize)
    X = dataset['X']
    Y = dataset['Y']
    UIDS = dataset['uids']
    return X, Y, UIDS

def create_path2spec(data_path, csv_path, SR, n_mels, frame_size, hop_length):
    print(f"Compute features for {os.path.basename(data_path)}")  
    features = {"uids": [], "X": [], "Y": [], "time_bins": []}
    wav = os.path.join(data_path)
    spc,len_audio = wav2spc(wav, fs=SR, n_mels=n_mels, downsample=True)
    time_bins = len_audio/spc.shape[1]
    df = new_find_tags(csv_path, SR)
    f = os.path.basename(data_path)
    print(f)
    Y = new_compute_Y(wav, f, spc, df, SR, frame_size, hop_length)
    features["uids"].append(data_path)
    features["X"].append(spc)
    features["Y"].append(Y)
    features["time_bins"].append(time_bins)
    return features

def new_load_file(data_path, csv_path, SR, n_mels, frame_size, hop_length):
    dataset = create_path2spec(data_path, csv_path, SR, n_mels, frame_size, hop_length)
    X = dataset['X']
    Y = dataset['Y']
    uids = dataset['uids']
    time_bins = dataset['time_bins']
    return X, Y, uids, time_bins
   
def load_wav_and_annotations(data_path, csv_path, SR=44100, n_mels=86, frame_size=2048, hop_length=1024, windowsize=1):
    x, y, uids, time_bins = new_load_file(data_path, csv_path, SR, n_mels, frame_size, hop_length)
    dataset = window_data(x, y, uids, time_bins, windowsize)
    X = np.array(dataset['X'])
    X = X.reshape(X.shape[0], 1, X.shape[1], X.shape[2])

    Y = np.array([dataset["Y"]])
    Y = Y.reshape(Y.shape[1], Y.shape[2])
    UIDS = np.array([dataset["uids"]])
    UIDS = UIDS.reshape(UIDS.shape[1])
    return X, Y, UIDS

def create_signal2spec(signal, SR, n_mels, frame_size, hop_length):
    features = {"uids": [], "X": [], "Y": [], "time_bins": []}
    spc = create_spec(signal, fs=SR, n_mels=n_mels, n_fft=frame_size, hop_len=hop_length)
    time_bins = (len(signal)/SR)/spc.shape[1]
    Y = np.array([0]*spc.shape[1])
    features["uids"].append("f")
    features["X"].append(spc)
    features["Y"].append(Y)
    features["time_bins"].append(time_bins)
    return features

def load_signal2spec(signal, SR, n_mels, frame_size, hop_length):
    dataset = create_signal2spec(signal, SR, n_mels, frame_size, hop_length)
    X = dataset['X']
    Y = dataset['Y']
    uids = dataset['uids']
    time_bins = dataset['time_bins']
    return X, Y, uids, time_bins

def compute_features(signal, SR=44100, n_mels=86, frame_size=2048, hop_length=1024, windowsize=2):
    x, y, uids, time_bins = load_signal2spec(signal[0], SR, n_mels, frame_size, hop_length)
    dataset = window_data(x, y, uids, time_bins, windowsize)
    X = np.array(dataset['X']).astype(np.float32)/255
    X = X.reshape(X.shape[0], 1, X.shape[1], X.shape[2])
    Y = np.array(dataset["Y"]).astype(np.longlong)
    UIDS = np.array(dataset["uids"])
    tweetynet_features = CustomAudioDataset(X, Y, UIDS)
    return tweetynet_features


# I imagine that this is broken. because the csv is correct

def predictions_to_kaleidoscope(predictions, SIGNAL, audio_dir, audio_file, manual_id, sample_rate):
    #look over this function again
    time_bin_seconds = predictions.iloc[1]["time_bins"]
    zero_sorted_filtered_df = predictions[predictions["pred"] == 0]
    offset = zero_sorted_filtered_df["time_bins"]
    duration = zero_sorted_filtered_df["time_bins"].diff().shift(-1)    
    intermediary_df = pd.DataFrame({"OFFSET": offset, "DURATION": duration})
    #need to fill out df. 
    kaliedoscope_df = []

    if offset.iloc[0] != 0:
        kaliedoscope_df.append(pd.DataFrame({"OFFSET": [0], "DURATION": [offset.iloc[0]]}))
    kaliedoscope_df.append(intermediary_df[intermediary_df["DURATION"] >= 2*time_bin_seconds])
    if offset.iloc[-1] < predictions.iloc[-1]["time_bins"]:
        kaliedoscope_df.append(pd.DataFrame({"OFFSET": [offset.iloc[-1]], "DURATION": [predictions.iloc[-1]["time_bins"] + 
                                predictions.iloc[1]["time_bins"]]}))

    kaliedoscope_df = pd.concat(kaliedoscope_df)
    kaliedoscope_df = kaliedoscope_df.reset_index(drop=True)
    kaliedoscope_df["FOLDER"] = audio_dir
    kaliedoscope_df["IN FILE"] = audio_file
    kaliedoscope_df["CHANNEL"] = 0
    kaliedoscope_df["CLIP LENGTH"] = len(SIGNAL)/sample_rate
    kaliedoscope_df["SAMPLE RATE"] = sample_rate
    kaliedoscope_df["MANUAL ID"] = manual_id

    return kaliedoscope_df
    