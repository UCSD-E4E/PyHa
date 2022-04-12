import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from .network import TweetyNet
#from microfaune.audio import wav2spc, create_spec, load_wav
from .TweetyNetAudio import wav2spc
from .CustomAudioDataset import CustomAudioDataset
from datetime import datetime
from torch.utils.data import DataLoader
import torch

"""
Helper Functions to TweetyNet so it feels more like a Tensorflow Model.
This includes instantiating the model, training the model and testing. 
"""
class TweetyNetModel:
    # Creates a tweetynet instance with training and evaluation functions.
    # input: num_classes = number of classes TweetyNet needs to classify
    #       input_shape = the shape of the spectrograms when fed to the model.
    #       ex: (1, 1025, 88) where (# channels, # of frequency bins/mel bands, # of frames)
    #       device: "cuda" or "cpu" to specify if machine will run on gpu or cpu.
    # output: None
    def __init__(self, num_classes, input_shape, window_size, device, epochs = 1, batchsize = 32, binary=False, criterion=None, optimizer=None):
        self.model = TweetyNet(num_classes=num_classes,
                               input_shape=input_shape,
                               padding='same',
                               conv1_filters=32,
                               conv1_kernel_size=(5, 5),
                               conv2_filters=64,
                               conv2_kernel_size=(5, 5),
                               pool1_size=(8, 1),
                               pool1_stride=(8, 1),
                               pool2_size=(8, 1),
                               pool2_stride=(8, 1),
                               hidden_size=None,
                               rnn_dropout=0.,
                               num_layers=1
                               )
        self.device = device
        self.model.to(device)
        self.binary = binary
        self.window_size = window_size #input_shape[-1] # set for pyrenote
        self.runtime = 0
        self.criterion = criterion if criterion is not None else torch.nn.CrossEntropyLoss().to(device)
        self.optimizer = optimizer if optimizer is not None else torch.optim.Adam(params=self.model.parameters())
        self.epochs = epochs
        self.batchsize = batchsize
        self.n_train_examples = self.batchsize *30 
        self.n_valid_examples = self.batchsize *10 
        #print(self.model)

    def test_load_step(self, test_dataset, hop_length, sr, batch_size=64, model_weights=None, window_size=2):
        if model_weights != None:
            self.model.load_state_dict(torch.load(model_weights))
            
        test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        test_out = self.testing_step(test_data_loader,hop_length,sr, window_size)
        return test_out

    def load_weights(self, model_weights):
        self.model.load_state_dict(torch.load(model_weights))
   
    def test_path(self, wav_path, n_mels):
        test_spectrogram =  wav2spc(wav_path, n_mels=n_mels)
        print(test_spectrogram.shape)
        wav_data = CustomAudioDataset( test_spectrogram, [0]*test_spectrogram.shape[1], wav_path)
        test_data_loader = DataLoader(wav_data, batch_size=1)
        test_out = self.test_a_file(test_data_loader)
        return test_out

    def predict(self, test_dataset, model_weights=None, norm=False, batch_size=1, window_size=1):
        if model_weights != None:
            self.model.load_state_dict(torch.load(model_weights))
        else:
            self.model.load_state_dict(torch.load(r"E:\PyHa\PyHa\tweetynet_package\tweetynet\config\model_weights_test.h5"))
        test_data_loader = DataLoader(test_dataset, batch_size=batch_size)
        predictions = pd.DataFrame()
        self.model.eval()
        local_score = []
        with torch.no_grad():
            for i, data in enumerate(test_data_loader):
                inputs, labels, uids = data
                #inputs = inputs.reshape(inputs.shape[0], 1, inputs.shape[0], inputs.shape[1])
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                output = self.model(inputs, inputs.shape[0], inputs.shape[0])
                local_score.extend([x for x in output[0, 1, :]])
                #add option to normalize
                #be able to create df if interested
                pred = torch.argmax(output, dim=1)
                pred = pred.reshape(pred.shape[1])
                labels = labels.reshape(labels.shape[1])
                #print(uids.shape, pred.shape, labels.shape)
                d = {"uid": uids[0], "pred": pred, "label": labels}
                new_preds = pd.DataFrame(d)
                predictions = predictions.append(new_preds)
        if norm:
            local_score = normalize(local_score, 0, 1)
        local_score = np.array(local_score)
        predictions["local_score"] = local_score
        return predictions, [local_score]

def normalize(arr, t_min, t_max):
    norm_arr = []
    diff = t_max - t_min
    diff_arr = max(arr) - min(arr)
    for i in arr:
        temp = (((i - min(arr))*diff)/diff_arr) + t_min
        norm_arr.append(temp)
    return norm_arr