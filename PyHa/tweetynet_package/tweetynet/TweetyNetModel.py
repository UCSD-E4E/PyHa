import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from .network import TweetyNet
from .TweetyNetAudio import wav2spc
from .CustomAudioDataset import CustomAudioDataset
from datetime import datetime
from torch.utils.data import DataLoader
import torch

"""Make function headers for each function."""

"""
Helper Functions to TweetyNet so it feels more like a Tensorflow Model.
This includes instantiating the model, training the model and testing. 
"""
class TweetyNetModel:
    # Creates a tweetynet instance with predict functions.
    # input: num_classes = number of classes TweetyNet needs to classify
    #       input_shape = the shape of the spectrograms when fed to the model.
    #       ex: (1, 1025, 88) where (# channels, # of frequency bins/mel bands, # of frames)
    #       window_size = the number of time bins in a window for tweetynet to predict on 
    #                     (86 for 2 seconds).
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

    def load_weights(self, model_weights):
        """
        Load the model weights, currently just for CPU.

        Args:
            model_weights: (str)
                - path to the model weights to be used in predicting.

        Returns:
            None
        """
        self.model.load_state_dict(torch.load(model_weights, map_location=torch.device('cpu')))

    def predict(self, test_dataset, model_weights=None, norm=False):
        """
        Predict on a wav file.

        Args:
            test_dataset: (CustomAudioDataset)
                - Contains the spectrogram, labels, and uid of the wav file. 

            model_weights: (str)
                - path to the model weights to be used in predicting.

            norm: (bool)
                - To normalize the local score array or not. 

        Returns:
            Pandas Dataframe of automated labels for the audio clip and the local score array of the clip.
        """
        batch_size=1
        window_size=2

        if model_weights == "retraining":
            pass
        elif model_weights != None:
            self.load_weights(model_weights)
        else:
            self.load_weights(os.path.join("PyHa","tweetynet_package","tweetynet","config","tweetynet_weights.h5"))

        test_data_loader = DataLoader(test_dataset, batch_size=batch_size)
        predictions = pd.DataFrame()
        self.model.eval()
        local_score = []
        dataiter = iter(test_data_loader)
        _, label, uid = dataiter.next()
        time_bin = float(window_size)/label.shape[1]
        st_time = np.array([time_bin*n for n in range(label.shape[1])])

        with torch.no_grad():
            for i, data in enumerate(test_data_loader):
                inputs, labels, uids = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                output = self.model(inputs, inputs.shape[0], inputs.shape[0])
                #print(output)
                #print(output[0,1])
                local_score.extend([x for x in output[0, 1, :]])
                pred = torch.argmax(output, dim=1)
                print(pred)
                pred = pred.reshape(pred.shape[1])
                labels = labels.reshape(labels.shape[1])
                bins = st_time + (int(uids[0].split("_")[0])*window_size)
                d = {"uid": uids[0], "pred": pred, "label": labels, "time_bins": bins}
                new_preds = pd.DataFrame(d)
                predictions = predictions.append(new_preds)

        if norm:
            local_score = self.normalize(local_score, 0, 1)
        local_score = np.array(local_score)
        predictions["local_score"] = local_score
        return predictions, [local_score]

    def normalize(self, arr, t_min, t_max):
        """
            Predict on a wav file.

            Args:
                arr: (Numpy Array)
                    - Local Score array from predict.

                t_min: (int)
                    - minimum value from the local score array.

                t_max: (int)
                    - maximum value from the local score array.

            Returns:
                Numpy array of the normalized local score array.
            """
        norm_arr = []
        diff = t_max - t_min
        arr_min = min(arr)
        diff_arr = max(arr) - arr_min
        for i in arr:
            temp = (((i - arr_min)*diff)/diff_arr) + t_min
            norm_arr.append(temp)
        return norm_arr