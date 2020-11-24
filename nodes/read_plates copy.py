#!/usr/bin/env python

# TODO clean this file!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

import sys
sys.path.insert(1, sys.path[0] + '/../scripts')

import os
import rospy
import cv2

from cv_bridge import CvBridge
import time
import matplotlib.pyplot as plt

import numpy as np
from tensorflow.compat.v1 import Session, get_default_graph
from tensorflow.keras import models
from tensorflow.keras.backend import set_session

from std_msgs.msg import String
from sensor_msgs.msg import Image


# how to import package from another path?
from plates.plate_interpreter import preprocess_plate
from plates.plate_locator import find_plate
from plates.cnn_utils import *


def confidence(oh):
    return np.max(oh) / np.sum(oh)


class Reader:
    def __init__(self):

        # Ideas:
        # Track current plate index
        # - when we stop we're gonna get over-reporting
        # be sure the neural networks aren't lying
        # - If it's wrong often enough the sum will lie
        #   + fix this by multiplying by velocity
        #   + multiplying by area is a decent idea
        self.path = os.path.dirname(os.path.realpath(__file__)) + "/"

        self.expected_positions = [2,3,4,5,6,1]
        self.expected_position_index = 0
        self.mode_interval = 10

        # number of frames to wait without input before being willing to move to the next expected position
        self.buffer_frames = 50
        self.frame_count = self.buffer_frames + 1
        self.skipped = 0

        # message credentials
        self.teamID = 'best4eva'
        self.teamPassword = 'password'

        # set up prediction storage system
        self.predictions = []
        for i in range(8):
            self.predictions.append([
                    [np.zeros(26, dtype=float), np.zeros(26, dtype=float)],
                    [np.zeros(10, dtype=float), np.zeros(10, dtype=float)]
                ])

        self.spot_estimate = []
        self.letter_estimates = [[], []]
        self.digit_estimates = [[], []]

        # load models
        self.session = Session()
        self.graph = get_default_graph()
        set_session(self.session)

        model_path = self.path + "../models/best/"

        self.spot_model = models.load_model(model_path + "parking_spot.h5")
        self.letter_model = models.load_model(model_path + "letters.h5")
        self.number_model = models.load_model(model_path + "numbers.h5")

        # set camera feed subscriber
        self.sub = rospy.Subscriber("/R1/pi_camera/image_raw", Image, self.read_camera_feed)
        self.bridge = CvBridge()

        # subscribe to message that may prematurely end run if goal is reached
        #self.end_signal = rospy.Subscriber("djkla", Type, self.end_run)......................................................

        # set license plate publisher
        self.reporter = rospy.Publisher("/license_plate", String, queue_size=10)
        time.sleep(1)


        # set timer to signal end of run before 4 minutes elapse
        self.timer = rospy.Timer(rospy.Duration(239), self.end_run, oneshot=True)
        self.start_run()


    def start_run(self):
        rospy.logfatal("timer started")
        self.send_message(0, "AA11")

    def end_run(self, signal):
        rospy.logfatal("run ended")
        self.send_message(-1, "AA11")

    def send_message(self, position, plate):
        msg = ",".join([self.teamID, self.teamPassword, str(position), plate])
        self.reporter.publish(msg)


    def read_camera_feed(self, frame):

        cv_image = self.bridge.imgmsg_to_cv2(frame, desired_encoding="passthrough")
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

        # transform raw feed into plate only, if it is recognizable
        plate, h_distance = find_plate(cv_image)

        if plate is not None:
            # reset counter
            self.frame_count = 0

            self.predict_plate(plate, h_distance)
            
            spot, plate = self.decode_estimates()

            print("Spot {}: {}".format(spot, plate))
        else:
            # count the frames we have gone without seeing plate. When it reaches
            # a sufficient threshold it submits the current plate guess.
            self.frame_count += 1
            if self.frame_count == self.buffer_frames:
                spot, plate = self.decode_estimates()
                print("Submitting guess: {}: {}".format(spot, plate))
                self.submit()

                
                
    

    def predict_spot(self, spot_img):
        return self.spot_model.predict(np.asarray([spot_img]))

    def predict_letters(self, let_img):
        return self.letter_model.predict(np.asarray([let_img]))

    def predict_numbers(self, num_img):
        return self.number_model.predict(np.asarray([num_img]))

    def decode_estimates(self):
        spot_sum = np.sum(np.array(self.spot_estimate), axis=0)
        spot = one_hot_to_spot_number(spot_sum)

        letters = []
        for i in range(len(self.letter_estimates)):
            letter_sum = np.sum(np.array(self.letter_estimates[i]), axis=0)
            letters.append(one_hot_to_char(letter_sum))
        
        digits = []
        for i in range(len(self.digit_estimates)):
            digit_sum = np.sum(np.array(self.digit_estimates[i]), axis=0)
            digits.append(one_hot_to_number(digit_sum))

        plate = "{}{}".format("".join(letters), "".join(digits))

        return spot, plate

    def submit(self):
        
        spot, plate = self.decode_estimates()
        self.send_message(spot, plate)

        # saving current estimates for future reference, if needed; not currently used
        self.predictions[int(spot) - 1][:2] = self.letter_estimates
        self.predictions[int(spot) - 1][2:] = self.digit_estimates

        # clear estimates
        self.spot_estimate = []
        self.letter_estimates = [[], []]
        self.digit_estimates = [[], []]


    def predict_plate(self, plate, h_distance):
        with self.graph.as_default():

            set_session(self.session)
            spot, lets, nums = preprocess_plate(plate)

            spot_pred = self.predict_spot(spot)

            self.spot_estimate.append(spot_pred)
            if len(self.spot_estimate) > self.mode_interval:
                self.spot_estimate.pop(0)


            if np.max(spot_pred) > 0.9:
                plate_guess = []
                #cumulative_guess = []
                for i in range(2):
                    let_pred = self.predict_letters(lets[i])
                    #self.predictions[spot_num - 1][i] = np.add(self.predictions[spot_num - 1][i], let_pred)

                    self.letter_estimates[i].append(let_pred)
                    if len(self.letter_estimates[i]) > self.mode_interval:
                        self.letter_estimates[i].pop(0)

                    #plate_guess.append(one_hot_to_char(let_pred))
                    #cumulative_guess.append(one_hot_to_char(self.predictions[spot_num-1][i]))
                for i in range(2):
                    num_pred = self.predict_numbers(nums[i])
                    #self.predictions[spot_num - 1][2 + i] = np.add(self.predictions[spot_num - 1][2 + i], num_pred)
                    
                    self.digit_estimates[i].append(num_pred)
                    if len(self.digit_estimates[i]) > self.mode_interval:
                        self.digit_estimates[i].pop(0)
                    
                    plate_guess.append(one_hot_to_number(num_pred))
                    #cumulative_guess.append(one_hot_to_number(self.predictions[spot_num-1][2 + i]))

                #rospy.logfatal('''\n
                #    Spot number {} ({}% confidence)\n
                #    \tCurrent guess: {}
                #    \tCumulative guess: {}
                #'''.format(spot_num, int(np.max(spot_pred)*100), "".join(plate_guess), "".join(cumulative_guess)))
                #self.send_message(spot_num, "".join(cumulative_guess))


            # make predictions.................................................................................................................



if __name__ == '__main__':

    rospy.init_node('read', log_level=rospy.DEBUG)
    reader = Reader()
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
