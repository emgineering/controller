#!/usr/bin/env python

import rospy
import sys

from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from std_msgs.msg import String, Bool, Int32
import message_filters

from enum import Enum

import numpy as np
import cv2
from cv_bridge import CvBridge

from tensorflow.compat.v1.keras.backend import set_session
from tensorflow.compat.v1.keras.models import load_model
from tensorflow.compat.v1 import Session, get_default_graph


class Loop(Enum):
    OUTER = 1
    INNER = 0

class Steer:

    def __init__(self):

        # set these to adjust robot speed. 
        # (Note: may need to retrain model for large changes)

        # increase speed_ratio if the robot is tending
        self.speed_ratio = 13
        self.speed_percentage = 1
        self.update_base_speed(0.35)

        self.current_loop = Loop.OUTER
        

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

        self.p_detect = PedestrianDetector()
        self.v_detect = VehicleDetector()

        self.approaching_plate_sub = rospy.Subscriber('/approaching_plate', Bool, self.receive_plate_signal)

        self.turn_signal_receiver = rospy.Subscriber('/turn_signal', Bool, self.receive_turn_signal)
        self.latest_parking_spot = 0

    def receive_plate_signal(self, msg):
        if self.current_loop == Loop.OUTER:
            # outer loop, viewing plate
            if msg.data:
                self.update_base_speed(0.3)
            # outer loop, driving
            else:
                self.update_base_speed(0.4)
        else:
            # inner loop, viewing plate
            if msg.data:
                self.update_base_speed(0.25)
            # inner loop, driving
            else:
                self.update_base_speed(0.3)

    def update_base_speed(self, new_base_linear_vel):
        self.base_speed = new_base_linear_vel
        self.max_linear_vel = self.base_speed * self.speed_percentage
        self.max_angular_vel = self.max_linear_vel * self.speed_ratio

    def update_speed(self, percentage):
        self.speed_percentage = percentage
        self.max_linear_vel = self.base_speed * self.speed_percentage
        self.max_angular_vel = self.max_linear_vel * self.speed_ratio

    def receive_turn_signal(self, msg):
        self.active_model = self.turn_model if msg.data else self.model
        self.turning = msg.data

        if msg.data == True:
            # on pulling out of the turn, slow down to keep pace with the truck
            self.current_loop = Loop.INNER

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

        self.update_speed(
            min(self.p_detect.get_recommended_speed(cv_image), self.v_detect.get_recommended_speed(cv_image))
            )

        self.predict_velocity(cv_image)
        

    def predict_velocity(self, image):
        cv_image = cv2.resize(image, self.target_image_shape)

        # decide velocity based on model prediction
        with self.graph.as_default():
            set_session(self.session)

            result = self.active_model.predict(np.asarray([cv_image / 255]))
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



class DriveOverride:

    def __init__(self):

        self.is_braked = False
        self.active = True

        # frames to wait before checking for brake/slowdown conditions again
        self.cooldown = 0
        self.frame_count = 0
    
    def get_recommended_speed(self, image):
        if not self.active:
            return True

        if self.is_braked:

            self.process_stopped(image)

            if self.check_resume_condition():
                self.is_braked = False
                self.frame_count = 0
                return self.recommend_speed()
            else:
                return 0
        
        # if a cooldown has been specified, ignore brake/slowdown conditions until it expires
        if self.frame_count < self.cooldown:
            self.frame_count += 1
            return 1

        self.process_running(image)

        # if the cooldown period has passed, check if stop condition has been met
        if self.check_stop_condition():
            self.is_braked = True
            return 0

        return self.recommend_speed()

    def recommend_speed(self):
        return 1

    def check_stop_condition(self):
        return False

    def check_resume_condition(self):
        return True

    def process_stopped(self, image):
        return

    def process_running(self, image):
        return

    def log(self, msg):
        bg_color = 40
        style = 1
        color = 33
        print("\033[{};{};{}m{}\033[0;37;40m".format(color, style, bg_color, msg))




class VehicleDetector(DriveOverride):

    def __init__(self):
        DriveOverride.__init__(self)
        self.gray_stop_threshold = 5000
        self.gray_slow_threshold = 2000
        self.gray_start_threshold = 3500

        self.cooldown = 5

        self.gray_pixels = 0

    def get_gray_in_range(self, image):
        low_thresh = 47
        high_thresh = 77
        low = image[:,:,0] > low_thresh
        high = image[:,:,0] < high_thresh

        bg = image[:,:,0] == image[:,:,1] # B == G
        gr = image[:,:,1] == image[:,:,2] # G == R

        mask = np.bitwise_and(np.bitwise_and(bg, gr), np.bitwise_and(low,high))
        return mask.sum()

    def process_running(self, image):
        self.gray_pixels = self.get_gray_in_range(image)
        
    def process_stopped(self, image):
        self.gray_pixels = self.get_gray_in_range(image)

    def check_stop_condition(self):
        if self.gray_pixels > self.gray_stop_threshold:
            self.log("STOP: close to vehicle")
            return True
        return False

    def check_resume_condition(self):
        if self.gray_pixels < self.gray_start_threshold:
            self.log("vehicle is far enough; moving on")
            return True
        return False

    def recommend_speed(self):
        gray = np.clip(self.gray_pixels, self.gray_slow_threshold, self.gray_stop_threshold)
        return 1 - (gray - self.gray_slow_threshold) / (self.gray_stop_threshold - self.gray_slow_threshold)


class PedestrianDetector(DriveOverride):

    def __init__(self):
        DriveOverride.__init__(self)

        self.cooldown = 100

        self.previous_frame = None
        self.red_lower_thresh = 1000
        self.red_threshold = 13000
        self.red_pixels = 0

        self.noise_threshold = 120
        self.movement_threshold = 100
        self.pedestrian_has_crossed = False
        self.different_pixels = 0

    def recommend_speed(self):
        red = np.clip(self.red_pixels, 1000, 10000)
        return 1 - np.log(red / self.red_lower_thresh) / np.log(self.red_threshold / self.red_lower_thresh)

    def process_stopped(self, image):
        if self.previous_frame is None:
            self.previous_frame = image
            return False

        prev = cv2.cvtColor(self.previous_frame, cv2.COLOR_BGR2GRAY).astype('float32')
        curr = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype('float32')

        diff = np.absolute(prev - curr)
        self.different_pixels = (diff > self.noise_threshold).sum()

        self.previous_frame = image
        

    def process_running(self, image):
        self.red_pixels = np.bitwise_and(image[:,:,0] > image[:,:,2], image[:,:,0] > 250).sum()

    def check_stop_condition(self):

        if self.red_pixels > self.red_lower_thresh:
            print(self.red_pixels)

        if self.red_pixels > self.red_threshold:
            self.log("STOP: crosswalk detected")
            return True
        return False

    def check_resume_condition(self):

        if self.pedestrian_has_crossed and self.different_pixels == 0:
            self.log("no motion detected; moving on")

            # reset flag for next time
            self.pedestrian_has_crossed = False
            self.previous_frame = None
            return True

        if not self.pedestrian_has_crossed and self.different_pixels > 200 and self.different_pixels < 300:
            self.log("saw pedestrian cross street")
            self.pedestrian_has_crossed = True

        #self.pedestrian_has_crossed = self.different_pixels == 0

        return False




 

if __name__ == '__main__':

    rospy.init_node('steer', log_level=rospy.DEBUG)
    drive = Steer()
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
