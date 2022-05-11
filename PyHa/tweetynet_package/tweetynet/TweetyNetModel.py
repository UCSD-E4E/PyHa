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

    def predict(self, test_dataset, model_weights=None, norm=False, batch_size=1, window_size=2):
        if model_weights != None:
            self.model.load_state_dict(torch.load(model_weights))
        else:
            self.model.load_state_dict(torch.load(os.path.join("PyHa","tweetynet_package","tweetynet","config","tweetynet_weights.h5"), map_location=torch.device('cpu')))
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
                #inputs = inputs.reshape(inputs.shape[0], 1, inputs.shape[0], inputs.shape[1])
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                output = self.model(inputs, inputs.shape[0], inputs.shape[0])
                #Add 0 predictions and 1 predictions
                ####                
                #### Ways to interpret the bidirectional output of Tweetynet. 
                #local_score.extend(np.median(output[0, 0, :], output[0, 1, :]))
                #local_score.extend(np.mean(output[0, 0, :], output[0, 1, :]))
                #local_score.extend(np.subtract(output[0, 0, :], output[0, 1, :]))
                #local_score.extend(np.add(output[0, 0, :], output[0, 1, :]))
                local_score.extend([x for x in output[0, 1, :]])
                #add option to normalize
                #be able to create df if interested
                pred = torch.max(output, dim=1)[1].cpu().detach().numpy()
                #pred = torch.argmax(output, dim=1)
                pred = pred.reshape(pred.shape[1])
                labels = labels.reshape(labels.shape[1])
                #print(uids.shape, pred.shape, labels.shape)
                #print(int(uids[0].split("_")[0])
                bins = st_time + (int(uids[0].split("_")[0])*window_size)
                d = {"uid": uids[0], "pred": pred, "label": labels, "time_bins": bins}
                new_preds = pd.DataFrame(d)
                predictions = predictions.append(new_preds)
        if norm:
            local_score = normalize(local_score, 0, 1)
        local_score = np.array(local_score)
        print(local_score.shape)
        predictions["local_score"] = local_score
        return predictions, [local_score]

def testing_step(self, test_loader, hop_length, sr, window_size):
        
        predictions = pd.DataFrame()
        self.model.eval()

        st_time = []
        dataiter = iter(test_loader)
        label, _, _ = dataiter.next()
        # print(label.shape)
        for i in range(label.shape[-1]): # will change to be more general, does it only for one trainfile?
            st_time.append(get_time(i, hop_length, sr))
        st_time = np.array(st_time)
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                inputs, labels, uids = data
                #inputs = inputs.reshape(inputs.shape[0], 1, inputs.shape[1], inputs.shape[2])
                #print(labels.dtype)
                #labels = labels.long()
                #print(labels.dtype)

                inputs, labels = inputs.to(self.device), labels.to(self.device)

                output = self.model(inputs, inputs.shape[0], labels.shape[0]) # what is this output look like?
                #print(output)

                temp_uids = []
                files = []
                window_file = []
                window_number = []
                frame_number = []
                overall_frame_number = []
                st_batch_times = []
                if self.binary: # weakly labeled
                    labels = torch.from_numpy((np.array([[x] * output.shape[-1] for x in labels])))
                    temp_uids = np.array([[x] * output.shape[-1] for x in uids])
                    files.append(u)
                else:  # in the case of strongly labeled data
                    for u in uids:
                        st_batch_times.extend(st_time + (window_size*int(u.split("_")[0])))
                        for j in range(output.shape[-1]):
                             temp_uids.append(str(j + (output.shape[-1]*int(u.split("_")[0]))) + "_" + u)
                             window_file.append(u)
                             frame_number.append(j)
                             overall_frame_number.append(j+ (output.shape[-1]*int(u.split("_")[0])))
                             window_number.append(int(u.split("_")[0]))
                             files.append("_".join(u.split("_")[1:]))
                    temp_uids = np.array(temp_uids)
                    window_file = np.array(window_file)
                    window_number = np.array(window_number)
                    frame_number = np.array(frame_number)
                    overall_frame_number = np.array(overall_frame_number)
                    st_batch_times = np.array(st_batch_times)
                zero_pred = output[:, 0, :]
                one_pred = output[:, 1, :]

                pred = torch.argmax(output, dim=1) 
                d = {"uid": temp_uids.flatten(), "window file": window_file.flatten(), "file":files, 
                        "overall frame number": overall_frame_number, "frame number": frame_number, "window number": window_number, 
                        "zero_pred": zero_pred.flatten(), "one_pred": one_pred.flatten(), 
                        "pred": pred.flatten(),"label": labels.flatten(), "temporal_frame_start_times": st_batch_times.flatten()}
                new_preds = pd.DataFrame(d)
                predictions = predictions.append(new_preds)

                #tim = {"temporal_frame_start_times": st_time}
                #time_secs = pd.DataFrame(tim)

                #nu_time = pd.concat([time_secs]*425, ignore_index=True)

                #extracted_col = nu_time["temporal_frame_start_times"]
                
                #predictions_timed = predictions.join(extracted_col)
                
        #predictions = prediction_fix(predictions, label.shape[-1])
        predictions = predictions.sort_values(["file", "overall frame number"])
        predictions = predictions.reset_index(drop=True)
        tim = {"temporal_frame_start_times": st_time}
        time_secs = pd.DataFrame(tim)
        print('Finished Testing')
        return predictions, time_secs


def normalize(arr, t_min, t_max):
    norm_arr = []
    diff = t_max - t_min
    diff_arr = max(arr) - min(arr)
    for i in arr:
        temp = (((i - min(arr))*diff)/diff_arr) + t_min
        norm_arr.append(temp)
    return norm_arr