#!/usr/bin/env python

import rospy
import sys

from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from std_msgs.msg import String, Bool, Int32
import message_filters

import numpy as np
import cv2
from cv_bridge import CvBridge

from tensorflow.compat.v1.keras.backend import set_session
from tensorflow.compat.v1.keras.models import load_model
from tensorflow.compat.v1 import Session, get_default_graph


class Steer:

    def __init__(self):

        # set these to adjust robot speed. 
        # (Note: may need to retrain model for large changes)
        self.speed_ratio = 0.1
        self.max_angular_vel = 5  #1
        self.max_linear_vel = self.speed_ratio * self.max_angular_vel  #0.15

        # When true, prevents steering until a message to release is received
        self.lock = rospy.get_param('~lock_on_start')

        # set variables for the tensorflow session so that the
        # model can be accessed by callbacks in other threads
        self.session = Session()
        self.graph = get_default_graph()
        set_session(self.session)

        # load steering model
        use_latest = rospy.get_param('~use_latest_model')
        model_path = '/home/fizzer/ros_ws/src/controller/models/' + ('latest/' if use_latest else 'best/')
        self.model = load_model(model_path + 'steer.h5')
        self.turn_model = load_model(model_path + 'steer_turn.h5')
        self.active_model = self.model
        self.turning = False

        # prepare image processing variables
        h, w = self.model.layers[0].input_shape[0][1:3]
        self.target_image_shape = (w, h)
        self.bridge = CvBridge()


        if self.lock:
            # Ensure the license plate reader module has started up before continuing
            rospy.wait_for_message("/license_plate", String)
            self.release_lock()

        # set up topic subscribers and publishers
        self.vel_pub = rospy.Publisher('/R1/cmd_vel', Twist, queue_size=1, latch=False)
        self.image_sub = message_filters.Subscriber('/R1/pi_camera/image_raw', Image)
        self.image_sub.registerCallback(self.process_image)

        self.detector_sub = rospy.Subscriber('/R1/brake', Bool, self.receive_brake_update)

        self.spot_sub = rospy.Subscriber("/prelim_spot_pred", Int32, self.receive_spot_update)
        self.turn_signal_receiver = rospy.Subscriber('/turn_signal', Bool, self.receive_turn_signal)
        self.latest_parking_spot = 0


    def receive_turn_signal(self, msg):
        self.active_model = self.turn_model if msg.data else self.model
        self.turning = msg.data

        if msg.data == True:
            # on pulling out of the turn, slow down to keep pace with the truck
            self.max_linear_vel = 0.15
            self.max_angular_vel = self.max_linear_vel / self.speed_ratio

    def receive_brake_update(self, msg):
        if msg.data:
            self.enable_lock()
        else:
            self.release_lock()

    def receive_spot_update(self, msg):
        self.latest_parking_spot = msg.data

    def process_image(self, frame):

        if self.lock:
            self.hard_stop()
            return

        # reduce image size to match model input
        cv_image = self.bridge.imgmsg_to_cv2(frame, desired_encoding="passthrough")
        cv_image = cv2.resize(cv_image, self.target_image_shape)

        # decide velocity based on model prediction
        with self.graph.as_default():
            set_session(self.session)

            if self.turning:
                result = self.turn_model.predict(np.asarray([cv_image / 255]))
            else:
                result = self.model.predict(np.asarray([cv_image / 255]))

            cmd_vel = self.transform_data(result[0])

            self.vel_pub.publish(cmd_vel)


    def hard_stop(self):
        # create a new twist with all velocities 0
        cmd_vel = Twist()
        self.vel_pub.publish(cmd_vel)
    
    def enable_lock(self):
        self.hard_stop()
        self.lock = True

    def release_lock(self):
        self.lock = False

    def transform_data(self, angular):
        cmd_vel = Twist()
        cmd_vel.angular.z = self.max_angular_vel * angular
        cmd_vel.linear.x = self.max_linear_vel * (1.0 - abs(angular))
        return cmd_vel

 

if __name__ == '__main__':

    rospy.init_node('steer', log_level=rospy.DEBUG)
    drive = Steer()
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
