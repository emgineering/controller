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
                print("robot has stopped")
                return

        '''

    def resume_moving(self):
        self.frame_count = 0
        self.is_braked = False
        self.brake_pub.publish(False)



class PedestrianDetector(DriveOverride):

    def __init__(self):
        DriveOverride.__init__(self)
        self.previous_frame = None
        self.red_threshold = 10000
        self.noise_threshold = 120
        self.movement_threshold = 100
        self.pedestrian_has_crossed = False
        

    def check_stop_condition(self, image):

        red = np.bitwise_and(image[:,:,0] > image[:,:,2], image[:,:,0] > 250)
        if red.sum() > self.red_threshold:
            print("STOP: crosswalk detected")
        return red.sum() > self.red_threshold

    def check_resume_condition(self, image):

        if self.previous_frame is None:
            self.previous_frame = image
            return False

        prev = cv2.cvtColor(self.previous_frame, cv2.COLOR_BGR2GRAY).astype('float32')
        curr = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype('float32')

        diff = np.absolute(prev - curr)
        different_pixels = (diff > self.noise_threshold).sum()

        self.previous_frame = image

        if self.pedestrian_has_crossed and different_pixels == 0:
            print("no motion detected; moving on")

            # reset flag for next time
            self.pedestrian_has_crossed = False
            self.previous_frame = None
            return True

        if not self.pedestrian_has_crossed and different_pixels > 200 and different_pixels < 300:
            print("saw pedestrian cross street")
            self.pedestrian_has_crossed = True

        return False


if __name__ == '__main__':
    rospy.init_node('pedestrian_detector', log_level=rospy.DEBUG)

    detect = PedestrianDetector()

    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
