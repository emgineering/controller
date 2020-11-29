import numpy as np
import cv2
import os
import random
import matplotlib.pyplot as plt
from datetime import datetime

import math

from keras import layers
from keras import models
from keras import optimizers
from keras import Model

from keras.utils import plot_model
from keras import backend

path = '/home/fizzer/ros_ws/src/harvest/turn_data/'
#path = '/home/fizzer/ros_ws/src/sandbox/data/harvested/'
files = os.listdir(path)
random.shuffle(files)

#files = files[:1]


epochs = 20

reduction = 0.1
max_linear = 0.1
max_angular = 0.5 if path == 'steering_data_jitterless/' else 0.7
turn_index = 4 if path == 'steering_data_jitterless/' else 3

def parse_file(filename):
    values = os.path.splitext(filename)[0].split(':')
    #linear = float(values[1]) / max_linear
    angular = float(values[1])
    turn = int(values[2])

    img = cv2.imread(path + filename)
    img = cv2.resize(img, None, fx=reduction, fy=reduction) / 255

    # ret,img = cv2.threshold(img,210,255,cv2.THRESH_BINARY)

    return img, turn, [angular]



data = list(zip(*[parse_file(filename) for filename in files]))
images = np.asarray(data[0])
turn_data = np.asarray(data[1])
y_data = np.asarray(data[2])


VALIDATION_SPLIT = 0.2

img_shape = images[0].shape

img_input = layers.Input(shape=img_shape)

#x = layers.convolutional.Conv2D(5, (3, 3), activation="relu")(img_input)
#x = layers.convolutional.MaxPooling2D((2,2))(x)
#x = layers.convolutional.Conv2D(10, (3,3), activation="relu")(x)
#x = layers.convolutional.MaxPooling2D((2,2))(x)
#x = layers.core.Flatten()(x)
#x = layers.core.Dense(50, activation='relu')(x)
#x = layers.core.Dense(8, activation='relu')(x)
#x = layers.core.Dense(1, activation="linear")(x)

x = layers.convolutional.Conv2D(12, (3, 3), activation="relu")(img_input)
x = layers.convolutional.MaxPooling2D((2,2))(x)
x = layers.convolutional.Conv2D(24, (3,3), activation="relu")(x)
x = layers.convolutional.MaxPooling2D((2,2))(x)
x = layers.core.Flatten()(x)
x = layers.core.Dense(50, activation='relu')(x)
x = layers.core.Dense(4, activation='relu')(x)
x = layers.core.Dense(1, activation="linear")(x)

model = Model(inputs=img_input, outputs=x)

model.summary()

LEARNING_RATE = 0.005
model.compile(loss='mean_squared_error',
                   optimizer=optimizers.RMSprop(lr=LEARNING_RATE),
                   )

history_conv = model.fit(images, y_data,
                              validation_split=VALIDATION_SPLIT,
                              epochs=epochs,
                              batch_size=16)

def mean_square_err(predicted, actual):
    predicted = np.asarray(predicted)
    actual = np.asarray(actual)
    return np.sum((predicted - actual)**2) / 2

def predict(index):
    pred = model.predict(np.asarray([images[index]]))
    return pred[0]


for i in range(20):
    print("Index: {}\nPrediction: {}\nGND Truth: {}\nLoss: {}\n".format(i, predict(i), y_data[i], mean_square_err(predict(i), y_data[i])))

model_path = '/home/fizzer/ros_ws/src/controller/models/'

model.save(model_path + 'latest/steer_turn.h5')
model.save(model_path + 'archive/steer_turn-{}.h5'.format(datetime.now()))