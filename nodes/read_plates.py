#!/usr/bin/env python

import sys
import os
import time
from collections import deque

import numpy as np
import rospy
import cv2

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import models

from std_msgs.msg import String, Int32
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

sys.path.insert(1, sys.path[0] + '/../scripts')
from plates.plate_interpreter import preprocess_plate
from plates.plate_locator import find_plate
import plates.cnn_utils as cnn_utils

class Reader:
    def __init__(self):
        # message credentials
        self.teamID = 'best4eva'
        self.teamPassword = 'password'

        # The number of license plate readings that get stored in the rolling buffer - the mode of this
        # buffer is used to determine the most likely match.
        averaging_interval = 10

        self.last_plate = 1

        # Rolling buffers that store the last `averaging_interval` predictions.
        self.clarity_values = deque([], averaging_interval)
        self.spot_estimate = deque([], averaging_interval)
        self.letter_estimates = [deque([], averaging_interval), deque([], averaging_interval)]
        self.digit_estimates = [deque([], averaging_interval), deque([], averaging_interval)]

        # number of frames between detecting plates that will trigger a submission/broadcast
        self.submission_buffer_frames = 30
        self.broadcast_buffer_frames = 10
        self.frame_count = self.submission_buffer_frames + 1

        self.clarity_threshold = 0.8

        # load models
        self.session = tf.Session()
        self.graph = tf.get_default_graph()
        keras.backend.set_session(self.session)

        path = os.path.dirname(os.path.realpath(__file__)) + "/"
        model_path = path + "../models/best/"
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
        self.plate_image_feed = rospy.Publisher("/processed_plates", Image, queue_size = 2)
        self.postion_reporter = rospy.Publisher("/current_spot_prediction", Int32, queue_size=1)
        
        # allow subscriber in score tracker time to set up
        time.sleep(1)

        # set timer to signal end of run before 4 minutes elapse
        self.timer = rospy.Timer(rospy.Duration(239), self.end_run, oneshot=True)
        self.start_run()


    def start_run(self):
        rospy.logfatal("timer started")
        self.send_message(0, "AA11")

    def end_run(self, signal):
        time.sleep(3)
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

        if plate is None:
            # count the frames we have gone without seeing plate. 
            self.frame_count += 1

            # When the count reaches a sufficient threshold it submits the current plate guess.
            if self.frame_count == self.submission_buffer_frames:   
                print("buffer has {} guesses".format(len(self.spot_estimate))) 
                self.submit()            
                
                # submit only if the queues are full, to prevent picking up random single-frame errors
                #if len(self.spot_estimate) == self.spot_estimate.maxlen:
                #    self.submit()
                self.clear_estimates()

            return

        clarity = self.get_clarity(plate)
        if clarity > self.clarity_threshold:
            # reset counter
            self.plate_image_feed.publish(self.bridge.cv2_to_imgmsg(plate, encoding="rgb8"))
            
            self.frame_count = 0
            self.predict_plate(plate, clarity)

    def get_clarity(self, plate):
        clarity = cv2.Laplacian(plate, cv2.CV_64F).var()
        return clarity


    # lets the driving modules know what we expect the current position to be
    def broadcast_current_position(self):
        spot_sum = np.sum(np.array(self.spot_estimate), axis=0)
        spot = np.argmax(spot_sum) + 1
        self.postion_reporter.publish(spot)

    # use a weighted average to pick the best guesses
    def decode_estimates(self):

        clarity_vector = np.transpose(self.clarity_values)

        spot_sum = np.dot(clarity_vector, self.spot_estimate)
        spot = cnn_utils.one_hot_to_spot_number(spot_sum)

        # Find the average letter sequence
        letters = []
        for i in range(len(self.letter_estimates)):
            letter_sum = np.dot(clarity_vector, self.letter_estimates[i])
            letters.append(cnn_utils.one_hot_to_char(letter_sum))
        
        # Find the average digit sequence
        digits = []
        for i in range(len(self.digit_estimates)):
            digit_sum = np.dot(clarity_vector, self.digit_estimates[i])
            digits.append(cnn_utils.one_hot_to_number(digit_sum))

        plate = "{}{}".format("".join(letters), "".join(digits))
        return spot, plate


    # finds the highest-quality guess in the buffer and makes a prediction accordingly
    def decode_estimates_clearest(self):
        best_index = np.argmax(self.clarity_values)
        print(best_index, self.clarity_values[best_index])

        # Find the average spot number
        spot = cnn_utils.one_hot_to_spot_number(self.spot_estimate[best_index])

        # Find the best letter sequence
        letters = []
        for i in range(len(self.letter_estimates)):
            letter_sum = self.letter_estimates[i][best_index]
            letters.append(cnn_utils.one_hot_to_char(letter_sum))
        
        # Find the best digit sequence
        digits = []
        for i in range(len(self.digit_estimates)):
            digit_sum = self.digit_estimates[i][best_index]
            digits.append(cnn_utils.one_hot_to_number(digit_sum))

        plate = "{}{}".format("".join(letters), "".join(digits))

        return spot, plate

    # Computes the average of the spot buffer and the plate buffers
    def decode_estimates_average(self):
        # Find the average spot number
        spot_sum = np.sum(np.array(self.spot_estimate), axis=0)
        spot = cnn_utils.one_hot_to_spot_number(spot_sum)

        # Find the average letter sequence
        letters = []
        for i in range(len(self.letter_estimates)):
            letter_sum = np.sum(np.array(self.letter_estimates[i]), axis=0)
            letters.append(cnn_utils.one_hot_to_char(letter_sum))
        
        # Find the average digit sequence
        digits = []
        for i in range(len(self.digit_estimates)):
            digit_sum = np.sum(np.array(self.digit_estimates[i]), axis=0)
            digits.append(cnn_utils.one_hot_to_number(digit_sum))

        plate = "{}{}".format("".join(letters), "".join(digits))
        return spot, plate
    
    # clears the prediction buffers for parking spot number and plate readings
    def clear_estimates(self):
        # clear estimates
        self.clarity_values.clear()
        self.spot_estimate.clear()
        self.letter_estimates[0].clear()
        self.letter_estimates[1].clear()
        self.digit_estimates[0].clear()
        self.digit_estimates[1].clear()


    # Submits the average estimate of the plate reading to the scoring application
    def submit(self):
        spot, plate = self.decode_estimates()
        print("Submitting guess: {}: {}".format(spot, plate))
        self.send_message(spot, plate)

        if int(spot) == self.last_plate:
            self.end_run(0)

    # Given a snapshot of the car's view, updates rolling buffers to store new plate predictions
    # (as long as they are of a certain quality)
    def predict_plate(self, plate, clarity):
        # This method is often executed in a different thread than __init__, so we need
        # to recall the default graph.
        with self.graph.as_default():
            keras.backend.set_session(self.session)
            spot, lets, nums = preprocess_plate(plate)

            spot_pred = self.spot_model.predict(np.asarray([spot]))[0]
            if np.max(spot_pred) > 0.9:
                self.spot_estimate.append(spot_pred)
                self.clarity_values.append(clarity)

                for i in range(2):
                    let_pred = self.letter_model.predict(np.asarray([lets[i]]))[0]
                    num_pred = self.number_model.predict(np.asarray([nums[i]]))[0]
                    
                    self.letter_estimates[i].append(let_pred)
                    self.digit_estimates[i].append(num_pred)

            else:
                print("insufficient conficence")




if __name__ == '__main__':
    rospy.init_node('read', log_level=rospy.DEBUG)

    reader = Reader()

    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
