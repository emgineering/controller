import cv2
import numpy as np
import os
import random
import string
import matplotlib.pyplot as plt

from random import randint
from PIL import Image, ImageFont, ImageDraw

num_images = 676 # should be at least 26^2 to get all combos

path = os.path.dirname(os.path.realpath(__file__)) + "/"
resource_path = path + "../../data/resources/"
save_path = path + '../../data/raw_plates/'

template = cv2.imread(resource_path + 'blank_plate.png')
monospace = ImageFont.truetype(resource_path + "UbuntuMono-R.ttf", 200)

os.system("mkdir -p {}".format(save_path))

for i in range(0, num_images):
    spot = (i % 8) + 1

    letter1 = (i // 26) % 26
    letter2 = i % 26

    num = i % 100

    plate_alpha = "{}{}".format(chr(ord("A") + letter1), chr(ord("A") + letter2))
    plate_num = "{:02d}".format(num)
    s = "P" + str(spot)

    # Write plate to image
    blank_plate = np.copy(template)
    
    # To use monospaced font for the license plate we need to use the PIL
    # package.
    # Convert into a PIL image (this is so we can use the monospaced fonts)
    blank_plate_pil = Image.fromarray(blank_plate)
    # Get a drawing context
    draw = ImageDraw.Draw(blank_plate_pil)
    
    draw.text((48, 105),plate_alpha + " " + plate_num, (255,0,0), font=monospace)
    # Convert back to OpenCV image and save
    blank_plate = np.array(blank_plate_pil)

    # Create parking spot label
    parking_spot = 255 * np.ones(shape=[600, 600, 3], dtype=np.uint8)
    cv2.putText(parking_spot, s, (30, 450), cv2.FONT_HERSHEY_PLAIN, 28,
                (0, 0, 0), 30, cv2.LINE_AA)
    spot_w_plate = np.concatenate((parking_spot, blank_plate), axis=0)

    # Merge images and save
    image = np.concatenate((255 * np.ones(shape=[600, 600, 3],
                                dtype=np.uint8), spot_w_plate), axis=0)
    cv2.imwrite(os.path.join(save_path, "{}:{}:{}.png".format(s,plate_alpha,plate_num)), image)
