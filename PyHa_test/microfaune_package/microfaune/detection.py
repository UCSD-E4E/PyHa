"""Module containing models for bird song detection."""
import os
import numpy as np
import tensorflow as tf
#from tensorflow.keras import tf.keras.layers
#from tensorflow.math import reduce_max

from .audio import load_wav, create_spec

#RNN_WEIGHTS_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__),"data/model_weights-20190919_220113.h5"))
RNN_WEIGHTS_FILE = os.path.abspath(
os.path.join(os.path.dirname(__file__),
             "data/model_weights-20200528_093824.h5"))
#load chosen weight from data

class RNNDetector:
    """Class wrapping a rnn model

    """
    def __init__(self, weights_file=RNN_WEIGHTS_FILE):
        """Initialization function"""
        self.weights_file = weights_file
        self._model = None

    @property
    def model(self):
        """Tensorflow Keras like model"""
        if self._model is None:
            self._model = self.create_model()
            self._model.load_weights(self.weights_file)
        return self._model

    def create_model(self):
        """Create RNN model."""
        n_filter = 64

        spec = tf.keras.layers.Input(shape=[None, 40, 1], dtype=np.float32)
        x = tf.keras.layers.Conv2D(n_filter, (3, 3), padding="same",
                          activation=None)(spec)
        x = tf.keras.layers.BatchNormalization(momentum=0.95)(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Conv2D(n_filter, (3, 3), padding="same",
                          activation=None)(x)
        x = tf.keras.layers.BatchNormalization(momentum=0.95)(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.MaxPool2D((1, 2))(x)

        x = tf.keras.layers.Conv2D(n_filter, (3, 3), padding="same",
                          activation=None)(x)
        x = tf.keras.layers.BatchNormalization(momentum=0.95)(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Conv2D(n_filter, (3, 3), padding="same",
                          activation=None)(x)
        x = tf.keras.layers.BatchNormalization(momentum=0.95)(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.MaxPool2D((1, 2))(x)

        x = tf.keras.layers.Conv2D(n_filter, (3, 3), padding="same",
                          activation=None)(x)
        x = tf.keras.layers.BatchNormalization(momentum=0.95)(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Conv2D(n_filter, (3, 3), padding="same",
                          activation=None)(x)
        x = tf.keras.layers.BatchNormalization(momentum=0.95)(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.MaxPool2D((1, 2))(x)

        x = tf.math.reduce_max(x, axis=-2)

        x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(64, reset_after=False, return_sequences=True))(x) #reset_after flag
        x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(64, reset_after=False, return_sequences=True))(x) #reset_after

        x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(64, activation="sigmoid"))(x)
        local_pred = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(1, activation="sigmoid"))(x)
        pred = tf.math.reduce_max(local_pred, axis=-2)
        return tf.keras.Model(inputs=spec, outputs=[pred, local_pred])

    def compute_features(self, audio_signals):
        """Compute features on audio signals.

        Parameters
        ----------
        audio_signals: list
            Audio signals of possibly various lengths.

        Returns
        -------
        X: list
            Features for each audio signal
        """
        X = []
        for data in audio_signals:

            x = create_spec(data, fs=44100, n_mels=40, n_fft=2048,
                            hop_len=1024).transpose()
            X.append(x[..., np.newaxis].astype(np.float32)/255)
        return X

    def predict_on_wav(self, wav_file):
        """Detect bird presence in wav file.

        Parameters
        ----------
        wav_file: str
            wav file path.

        Returns
        -------
        score: float
            Prediction score of the classifier on the whole sequence
        local_score: array-like
            Time step prediction score
        """
        fs, data = load_wav(wav_file)
        X = self.compute_features([data])
        scores, local_scores = self.predict(np.array(X))
        return scores[0], local_scores[0]

    def predict(self, X):
        """Predict bird presence on spectograms.

        Parameters
        ----------
        X: array-like
            List of features on which to run the model.

        Returns
        -------
        scores: array-like
            Prediction scores of the classifier on each audio signal
        local_scores: array-like
            Step-by-step  prediction scores for each audio signal
        """
        scores = []
        local_scores = []
        for x in X:
            s, local_s = self.model.predict(x[np.newaxis, ...])
            scores.append(s[0])
            local_scores.append(local_s.flatten())
        scores = np.array(s)
        return scores, local_scores

    def free_mem(self):
        """Release GPU memory."""
        self._model = None
