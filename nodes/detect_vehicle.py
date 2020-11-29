#!/usr/bin/env python

import rospy
import numpy as np
import cv2

from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from std_msgs.msg import String, Bool
from cv_bridge import CvBridge

class DriveOverride:

    def __init__(self):

        self.is_braked = False

        # frames to wait before checking for brake conditions again
        self.cooldown = 150
        self.frame_count = 0

        self.camera_sub = rospy.Subscriber('/R1/pi_camera/image_raw', Image, self.read_camera_feed)
        self.brake_pub = rospy.Publisher('/R1/brake', Bool, queue_size=3)


        self.bridge = CvBridge()

    def log(self, msg):
        bg_color = 40
        style = 1
        color = 33
        print("\033[{};{};{}m{}\033[0;37;40m".format(color, style, bg_color, msg))

    def check_stop_condition(self, image):
        return False

    def check_resume_condition(self, image):
        return True

    def read_camera_feed(self, image):


        cv_image = self.bridge.imgmsg_to_cv2(image, desired_encoding="passthrough")

        if self.is_braked:
            if self.check_resume_condition(cv_image):
                self.resume_moving()
            return

        if self.frame_count < self.cooldown:
            self.frame_count += 1
            return

        # if the cooldown period has passed, check if stop condition has been met
        if self.check_stop_condition(cv_image):
            self.stop_moving()

    def stop_moving(self):

        self.brake_pub.publish(True)
        self.is_braked = True
        return

        # wait until getting confirmation that the robot is no longer moving
        '''while True:
            cmd_vel = rospy.wait_for_message('/R1/cmd_vel', Twist)

            if cmd_vel.linear.x == 0 and cmd_vel.angular.z == 0:
                self.is_braked = True
                self.log("robot has stopped")
                return

        '''

    def resume_moving(self):
        self.frame_count = 0
        self.is_braked = False
        self.brake_pub.publish(False)



class VehicleDetector(DriveOverride):

    def __init__(self):
        DriveOverride.__init__(self)
        self.gray_stop_threshold = 5000
        self.gray_start_threshold = 2000

        

    def get_gray_in_range(self, image):
        low_thresh = 47
        high_thresh = 77
        low = image[:,:,0] > low_thresh
        high = image[:,:,0] < high_thresh

        # road: 81-85
        # trees: 39-44
        # stripes: 


        bg = image[:,:,0] == image[:,:,1] # B == G
        gr = image[:,:,1] == image[:,:,2] # G == R

        mask = np.bitwise_and(np.bitwise_and(bg, gr), np.bitwise_and(low,high))
        return mask.sum()


    def check_stop_condition(self, image):

        gray_pixels = self.get_gray_in_range(image)

        if gray_pixels > self.gray_stop_threshold:
            self.log("STOP: close to vehicle")
            return True
        return False

    def check_resume_condition(self, image):

        gray_pixels = self.get_gray_in_range(image)
        
        if gray_pixels < self.gray_start_threshold:
            self.log("vehicle is far enough; moving on")
            return True

        return False


if __name__ == '__main__':
    rospy.init_node('vehicle_detector', log_level=rospy.DEBUG)

    detect = VehicleDetector()

    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
