#!/usr/bin/env python

"""
Turn towards the closest person found by yolo
"""

import rospy 
from std_msgs.msg import String, Bool
import time
import math
from hsrb_interface import Robot
from tmc_yolo2_ros.msg import Detections
import numpy as np

# Camera projection matrix from calibration
# xtion
#projection = np.array([537.4419394922311, 0.0, 320.7446731239431, 0.0, 0.0, 537.3267489282307, 236.6190392583983, 0.0, 0.0, 0.0, 1.0, 0.0]).reshape((3, 4))
# head center
projection = np.array([233.0644925835, 0.0, 320.0, 0.0, 0.0, 233.0644925835, 240.0, 0.0, 0.0, 0.0, 1.0, 0.0]).reshape((3, 4))
projection_inv = np.linalg.pinv(projection)


class LocatePerson:
    def __init__(self, fastmove):
        rospy.Subscriber("/yolo2_node/detections",Detections, self.detections_callback)
        self.found_person = False
        self.current_angle = 0
        self.current_pos = 0
        self.x = 0
        self.y = 0
        self.w = 0
        self.h = 0
        self.fastmove = fastmove

    def detections_callback(self,data):
        global projection_inv
        detected_people = []
        self.found_person = False
        closest_angle = 360
        pos_count = 0
        x = 0
        y = 0
        w = 0
        h = 0
        for detected_object in data.detections:
            if detected_object.class_name == 'person':
                self.found_person = True
                two_d = np.array([detected_object.x, detected_object.y, 1.0])
                three_d = np.dot(projection_inv, two_d)
                angle = -math.atan(three_d[0])
                if(angle < closest_angle):
                    closest_angle = angle
                    self.current_pos = pos_count
                    x = detected_object.x
                    y = detected_object.y
                    w = detected_object.width
                    h = detected_object.height
                pos_count = pos_count + 1
        self.current_angle = closest_angle
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def turn_to_person(self):
        if(self.found_person):
            print("I found a person at angle:",self.current_angle)
            self.fastmove.go(0, 0, self.current_angle, 100, relative=True, angular_thresh=5)

    def main(self):
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('person_angle_node')
    turn = Turn_to_person()
    rospy.spin()