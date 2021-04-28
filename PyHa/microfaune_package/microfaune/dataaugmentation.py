#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 16:18:32 2019


@author: christian
"""
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from microfaune import plot

class DataAugmentation:
    """Class to generate image data for rnn modeling
    """
    datagenerator_list = None

    def __init__(self, width_shift_range=None, horizontal_flip=True, \
                 brightness_range=None):
        """Initialization data generators

        Parameters
        ----------
        width_shift_range: list
            width of the random horizontal shift
        horizontal_flip: bool
            moving all pixels of the image horizontally,
            while keeping the image dimensions the same.
        brightness_range: list
        The brightness can be augmented by randomly darkening images,
        where has no effect on darkness

        Initiate
        -------
        datagenerator_list: list
            image data generators
        """
        if width_shift_range is None:
            width_shift_range = [-40, 40]
        if brightness_range is None:
            brightness_range = [0.4, 0.9]

        datagen_width_shift = ImageDataGenerator(width_shift_range=width_shift_range)
        datagen_horizontal_flip = ImageDataGenerator(horizontal_flip=horizontal_flip)
        datagen_brightness = ImageDataGenerator(brightness_range=brightness_range)
        self.datagenerator_list = [datagen_width_shift, datagen_horizontal_flip, datagen_brightness]

    def generate_augmentation(self, spec, y_val, my_range=5, to_display=False):
        """ data augmentation of one Spectrogram
        Parameters
        ----------
        spec spectogram
        y_val classification value

        Returns
        -------
        list_s
            list of Spectograms with Y list (duplicate from y input)
            the first Spectograms is the given input S
        list_y
            All y have the value of the given y
        """
        list_s = [spec]
        list_y = [y_val]
        for datagen in self.datagenerator_list:
            data = np.expand_dims(spec, axis=2)

            # expand dimension to one sample
            samples = np.expand_dims(data, 0)
            # prepare iterator
            cursor = datagen.flow(samples, batch_size=1)
            # generate samples and plot
            for _ in range(my_range):
                batch = cursor.next()
                # convert to unsigned integers for viewing
                image = batch[0].astype('uint8')
                image = image[:, :, 0]
                list_s.append(image)
                list_y.append(y_val)
                if to_display:
                    plot.plot_spec(image)
        return list_s, list_y

    def generate_augmentation_list(self, list_s, list_y, my_range=5, to_display=False):
        """ data augmentation of a  list of Spectrograms
        Parameters
        ----------
        list_s  vector of spectograms
        list_y  vector of y

        Returns
        -------
        list_s_augmented
            list of Spectograms augmented
        list_y_augmented
            list of y augmented with duplicate values
        """
        list_s_augmented = []
        list_y_augmented = []
        for spec, y_val in zip(list_s, list_y):
            lstx, lsty = self.generate_augmentation(spec, y_val, my_range, to_display)
            list_s_augmented += lstx
            list_y_augmented += lsty
        return list_s_augmented, list_y_augmented
