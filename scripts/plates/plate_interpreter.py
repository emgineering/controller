#!/usr/bin/env python


import numpy as np
import cv2
import matplotlib.pyplot as plt


nominal = np.asarray([50,140, 250, 350, 450, 550])
tol = 10 # tolerance on horizontal shift of letter boundaries
distance_penalty = 0.05


def segment_plate(plate):

    parking_spot = plate[600:1200, 300:]
    license_plate = plate[1200:1800]

    return parking_spot, license_plate



width = 2 * tol + np.max([nominal[i+1] - nominal[i] for i in range(0, len(nominal)-1)])

def clean(image):
    output = cv2.bitwise_not(image)
    output = cv2.normalize(output,None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    output = np.multiply(output, output[:,:] > 0.5) # threshold
    kernel = np.ones((5,5),np.uint8)
    output = cv2.morphologyEx(output, cv2.MORPH_OPEN, kernel) # opening

    return output


def isolate_letters(image):
    top = 100
    bottom = 400
    height = bottom - top

    red = image[top:bottom]
    height = 300

    # Clarifying image

    red = clean(red[:,:,1])

    # Splitting image

    dividers = np.zeros_like(nominal)

    for i in range(len(nominal)):
        arr = [np.sum(red[:, c]) + distance_penalty * abs(nominal[i] - c) for c in range(nominal[i] - tol,nominal[i] + tol + 1)]
        dividers[i] = nominal[i] + np.argmin(arr) - tol

    characters = []
    for i in range(1,len(dividers)):
        if i == 3:
            continue

        letter_slice = image[top:bottom,dividers[i-1]:dividers[i],:]
        #letter_slice = red[:,dividers[i-1]:dividers[i]]

        # zero-pad sides so all images have same dimensions
        h_margins = width - letter_slice.shape[1]


        left_margin = 255 * np.ones(shape=[height, h_margins//2, 3], dtype=np.uint8)
        right_margin = 255 * np.ones(shape=[height, h_margins - h_margins//2, 3], dtype=np.uint8)

        letter_slice = np.hstack([left_margin, letter_slice, right_margin])
        #letter_slice = np.hstack((np.zeros((height, h_margins//2)), letter_slice, np.zeros((height, h_margins - h_margins//2))))

        characters.append(letter_slice)

    return characters

def preprocess_plate(img):
    spot, plate = segment_plate(img)

    #spot = clean(cv2.cvtColor(spot, cv2.COLOR_BGR2GRAY))
    spot = cv2.resize(spot, None, fx=0.3, fy=0.3)

    chars = isolate_letters(plate)
    letters = chars[:2]
    numbers = chars[2:]

    return spot, letters, numbers

def show_outputs(img):
    spot, lets, nums = preprocess_plate(img)
    chars = np.concatenate((lets, nums))
    _, ax = plt.subplots(1,5)
    ax[0].imshow(spot)
    for i in range(4):
        ax[i+1].imshow(chars[i])
