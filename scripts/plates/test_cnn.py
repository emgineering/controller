#!/usr/bin/env python

# TODO possibly clean up this file?

import os
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
from plate_interpreter import preprocess_plate
from plate_interpreter import show_outputs

# note: this function currently does nothing

from keras import models


# test current models, or best?
best = True


path = os.path.dirname(os.path.realpath(__file__)) + "/"

data_path = path + "../../data/test_images/"
model_path =  path + "../../models/best/" if best else path + "../../models/latest/"


# load test data
tests = os.listdir(data_path)
test_imgs = [cv2.imread(data_path + filename) for filename in tests]


# load latest models
spot_model = models.load_model(model_path + "parking_spot.h5")
letter_model = models.load_model(model_path + "letters.h5")
number_model = models.load_model(model_path + "numbers.h5")


def one_hot(size, index):
    oh = np.zeros(size, dtype=float)
    oh[index] = 1
    return oh

def one_hot_to_number(oh):
    index = np.argmax(oh)
    return chr(ord("0") + index)

def one_hot_to_char(oh):
    index = np.argmax(oh)
    return chr(ord("A") + index)

def one_hot_to_spot_number(oh):
    index = np.argmax(oh)
    return chr(ord("1") + index)



# takes a list of channel-less (e.g. grayscale) images
# and returns list of images with explicitly one channel
def reshape(single_channel):
    if len(single_channel.shape) == 4:
        return single_channel
    return np.reshape(single_channel, single_channel.shape + (1,))

def predict_spot(spot_img):
    spot_img = reshape(np.asarray([spot_img]))
    return spot_model.predict(spot_img)

def predict_letters(let_img):
    let_img = reshape(np.asarray([let_img]))
    return letter_model.predict(let_img)

def predict_numbers(num_img):
    num_img = reshape(np.asarray([num_img]))
    return number_model.predict(num_img)


def predict(full_image):
    spot, lets, nums = preprocess_plate(full_image)
    spot_pred = one_hot_to_spot_number(predict_spot(spot)[0])
    plate_chars = []
    for let in lets:
        plate_chars.append(one_hot_to_char(predict_letters(let)))
    for num in nums:
        plate_chars.append(one_hot_to_number(predict_numbers(num)))
    plate_chars = "".join(plate_chars)

    return spot_pred, plate_chars


for img in test_imgs:
    plt.figure()
    plt.title(predict(img))
    plt.imshow(img)
    plt.show()

