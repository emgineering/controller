import image_utils
import os
import cv2
import numpy as np

repetition_count = 5

path = os.path.dirname(os.path.realpath(__file__)) + "/"
dest_dir = path +  "../../data/distorted_plates/"
src_dir = path + "../../data/raw_plates/"

os.system("mkdir -p {}".format(dest_dir))

for name in os.listdir(src_dir):
    prefix = name.split('.')[0]
    image = cv2.imread("{}{}".format(src_dir, name))

    for idx in range(repetition_count):
        new_name = "{}:{}.png".format(prefix, idx)
        distorted = image_utils.distort_image(image)

        if distorted is not None:
            cv2.imwrite("{}{}".format(dest_dir, new_name), distorted)
