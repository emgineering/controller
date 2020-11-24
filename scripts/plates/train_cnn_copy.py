#!/usr/bin/env python


# This file is only for training the spot-number detector using Emily's Nov 24 manually extracted real data. 
# Delete once proper data generation pathways are in place.

import os
import random
import numpy as np
import cv2
import re
from datetime import datetime

from keras import models
from keras import layers
from keras import optimizers

import matplotlib.pyplot as plt

from plate_interpreter import preprocess_plate
from cnn_utils import *

path = os.path.dirname(os.path.realpath(__file__)) + "/"


data_path = '/home/fizzer/plate_mine/'
save_path = path + "../../models/"


VALIDATION_SPLIT = 0.2
LEARNING_RATE = 1e-4


# EARILY STOPPING:
# https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping


# 
def load_data(percentage, train_spot, train_letters, train_numbers):
    

    files = os.listdir(data_path)
    random.shuffle(files)

    files = files[:int(percentage * len(files))]

    spot_images = []
    letter_images = []
    number_images = []

    spot_labels = []
    letter_labels = []
    number_labels = []

    for filename in files:

        # process labels
        labels = os.path.splitext(filename)[0].split('-')
        spot = labels[0][0]
        #letters = labels[1]
        #numbers = labels[2]

        # process image
        img = cv2.imread(data_path + filename)
        spot_img, let_imgs, num_imgs = preprocess_plate(img)

        # append to containers
        if train_spot:
            spot_labels.append(one_hot(8, ord(spot) - ord("1")))
            spot_images.append(spot_img)
        if train_letters:
            for i in range(2):
                letter_labels.append(one_hot(26, ord(letters[i]) - ord("A")))
                letter_images.append(let_imgs[i])
        if train_numbers:
            for i in range(2):
                number_labels.append(one_hot(10, ord(numbers[i]) - ord("0")))
                number_images.append(num_imgs[i])
    
    images = [spot_images, letter_images, number_images]
    labels = [spot_labels, letter_labels, number_labels]

    return images, labels


# DEFINE MODELS
def create_and_train(xdata, ydata, epochs):

    # define model
    in_shape = xdata[0].shape
    print(in_shape)
    out_shape = len(ydata[0])

    img_input = layers.Input(shape=in_shape)
    x = layers.Conv2D(12, (5, 5), activation="relu")(img_input)
    x = layers.MaxPool2D((2,2))(x)
    x = layers.Conv2D(24, (5, 5), activation="relu")(x)
    x = layers.MaxPool2D((2, 2))(x)
    x = layers.Conv2D(32, (5, 5), activation="relu")(x)
    x = layers.MaxPool2D((2, 2))(x)
    x = layers.Conv2D(64, (5, 5), activation="relu")(x)

    x = layers.MaxPool2D((2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(100, activation = 'relu')(x)
    x = layers.Dense(out_shape, activation="softmax")(x)

    model = models.Model(inputs=img_input, outputs=x)
    model.summary()

    model.compile(loss='categorical_crossentropy',
                   optimizer=optimizers.RMSprop(lr=LEARNING_RATE),
                   metrics=['acc'])

    history_conv = model.fit(xdata, ydata,
                              validation_split=VALIDATION_SPLIT,
                              epochs=epochs,
                              batch_size=8)

    return model, history_conv

def create_and_save(xdata, ydata, epochs, name):
    model, _ = create_and_train(xdata, ydata, epochs)

    model.save(save_path + 'archive/' + name + '-{}.h5'.format(datetime.now()))
    model.save(save_path + 'latest/' + name + '.h5')

if __name__ == "__main__":
    yes_regex = "[yYtT]"
    
    train_spot = re.search(yes_regex, raw_input("Train spot: ")) is not None
    train_letters = re.search(yes_regex, raw_input("Train letteres: ")) is not None
    train_numbers = re.search(yes_regex, raw_input("Train numbers: ")) is not None
    

    images, labels = load_data(1,train_spot, train_letters, train_numbers)

    if train_spot:
        create_and_save(np.asarray(images[0]), np.asarray(labels[0]), 10, 'parking_spot')

    if train_letters:
        create_and_save(np.asarray(images[1]), np.asarray(labels[1]), 10, 'letters')

    if train_numbers:
        create_and_save(np.asarray(images[2]), np.asarray(labels[2]), 10, 'numbers')