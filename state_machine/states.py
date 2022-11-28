import random

import numpy as np
import actionlib
import rospy
import smach
import villa_top_level.common.states as common_states
from activity_rnn.msg import DetectedActivity
from hsrb_interface import geometry, exceptions
from std_srvs.srv import Empty
from villa_helpers.msg import FreeSpaceAction, FreeSpaceGoal
from villa_top_level.common import speech
import math
from std_srvs.srv import Trigger

# import the code to find the referee
import personangle

# bounding boxes
from villa_object_cloud.srv import GetObjects, GetObjectsRequest, GetSurfaces, GetSurfacesRequest, GetSurfaceOccupancy, \
    GetSurfaceOccupancyRequest, GetBoundingBoxes, \
    GetBoundingBoxesRequest

HANDOVER_POSE = {"arm_lift_joint": 0.376,
                 "arm_flex_joint": -0.9,
                 "arm_roll_joint": 0.,
                 "wrist_roll_joint": 1.6,
                 "wrist_flex_joint": -0.63}

NEUTRAL_POSE = {"arm_lift_joint": 0.,
                "arm_flex_joint": 0.,
                "arm_roll_joint": 0.,
                "wrist_roll_joint": 1.6,
                "wrist_flex_joint": -1.5}

# Returns the angle between two points with (x1, y1) as the origin
# and (x2, y2) as the endpoint
def getAngle(x1, y1, x2, y2):
    # horizontal plane
    angle = math.atan((y2 - y1)/(x2 - x1))

    # adjust arctan so it returns an angle 0-2pi
    if(x2 < x1):
        angle = angle + math.pi
    if(angle < 0):
        angle = angle + 2 * math.pi
    return angle

# Returns the absolute value of the difference between two angles (both 0-2pi)
def getAngleDiff(a1, a2):
    return min((2 * math.pi) - abs(a1 - a2), abs(a1 - a2))

# from groceries
def BoundingBoxesRequest(surface_height=0):
    zscale = 1.5
    xscale = 3
    yscale = 4
    objects_req = GetBoundingBoxesRequest()
    objects_req.search_box.header.frame_id = 'base_link'
    objects_req.search_box.origin.z = surface_height + zscale / 2
    objects_req.search_box.origin.x = xscale / 2
    objects_req.search_box.scale.x = xscale
    objects_req.search_box.scale.y = yscale
    objects_req.search_box.scale.z = zscale
    return objects_req

# Wait until the referee shakes the robot's wrist to start
class WaitForStart(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['signalled', 'not_signalled'])
        self.trigger_service_client = rospy.ServiceProxy('wrist_trigger', Trigger)

    def execute(self, userdata):
        
        try:
            if smach.State.simulation:
                return 'signalled'
        except AttributeError:
            pass

        try:
            rospy.wait_for_service('wrist_trigger', timeout=10)
        except rospy.ROSException:
            rospy.logerr("Couldn't obtain wrist_trigger service handle")
            return 'not_signalled'
        rospy.loginfo('Waiting for start signal...')
        try:
            response = self.trigger_service_client()
            rospy.loginfo(response.message)
            if response.success:
                return 'signalled'
            else:
                return 'not_signalled'
        except rospy.ServiceException():
            return 'not_signalled'


# Rotate the head until the body of the referee is found
class FindBodyReferee(smach.State):
    def __init__(self, omni_base):
        smach.State.__init__(self, outcomes=['succeeded'])
        self.omni_base = omni_base

    # using LocatePerson in personangle.py
    def execute(self, userdata):
        #omni_base = FastMove(robot.get('omni_base'))
        detect_person = personangle.LocatePerson(self.omni_base)
        detect_person.turn_to_person()
        return 'succeeded'

#TODO class
# Find the hand of the referee and return the x, y, z coordinates of the
# finger base and finger tip
class FindHand(smach.State):
    def __init__(self, speech):
        smach.State.__init__(self, outcomes=['succeeded'], 
                output_keys=['bx', 'by', 'bz', 'tx', 'ty', 'tz'])
        self.speech = speech
    def execute(self, userdata):
        self.speech.say("What do you need?")
        # run program point_tracker/src/point_tracker_node.cpp
        userdata.bx = 
        userdata.by = 
        userdata.bz = 
        userdata.tx = 
        userdata.ty = 
        userdata.tz = 
        return 'succeeded'

#TODO comment
# Returns the angle (0-2pi) between two points with (x1, y1) as the origin
# and (x2, y2) as the endpoint
class CreateVector(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['succeeded'], 
                input_keys=['bx', 'by', 'bz', 'tx', 'ty', 'tz'], output_keys=['point_angle'])

    def execute(self, userdata):
        # frontal plane
        # xyangle = math.atan((tipY - baseY)/(tipX - baseX))

        # median plane
        # yzangle = math.atan((tipY - baseY)/(tipZ - baseZ))

        # horizontal plane
        # xzangle = math.atan((tipZ - baseZ)/(tipX - baseX))

        # xzangle is the most important

        userdata.point_angle = getAngle(userdata.bx, userdata.bz, userdata.tx, userdata.tz)
        return 'succeeded'

# Find bounding boxes of 
class FindBoundingBoxes(smach.State):
    def __init__(self, speech):
        smach.State.__init__(self, outcomes=['succeeded', 'aborted', 'finished'], 
                input_keys=['point_angle', 'bx', 'bz', 'times_in']
                output_keys=['times_out'])
        self.speech = speech

        rospy.wait_for_service('/object_cloud/clear_octree')
        rospy.wait_for_service('/object_cloud/get_bounding_boxes')
        rospy.wait_for_service('/object_cloud/get_surfaces')
        rospy.wait_for_service('/viewpoint_controller/stop')
        self.clear_octree = rospy.ServiceProxy('/object_cloud/clear_octree', Empty)
        self.get_bounding_boxes = rospy.ServiceProxy('/object_cloud/get_bounding_boxes', GetBoundingBoxes)
        self.get_surfaces = rospy.ServiceProxy("/object_cloud/get_surfaces", GetSurfaces)
        self.stop_head = rospy.ServiceProxy('/viewpoint_controller/stop', Empty)
        self.start_head = rospy.ServiceProxy('/viewpoint_controller/start', Empty)
        self.clear_octomap = rospy.ServiceProxy('/parallel_planner/clear_octomap', Empty)

    def execute(self, userdata):
        self.clear_octomap()

        # try to get the bounding boxes 3 times
        count = 0
        while count < 3:
            self.clear_octree()
            rospy.sleep(1)

            # Get the bounding boxes!
            objects_req = BoundingBoxesRequest()
            bboxes = self.get_bounding_boxes(objects_req)
            objects = bboxes.bounding_boxes.markers[:]

            if len(objects) > 0:
                break
            print ("Waiting for bounding boxes")
            count += 1

        if (len(objects)) == 0:
            return "aborted"

        print("number of objects found: " + len(bboxes.bounding_boxes.markers))
        if len(bboxes.bounding_boxes.markers) <= 0:
            self.speech.say("Can't see any objects.")
            return 'aborted'

        objects = bboxes.bounding_boxes.markers[:]

        # find the object with the smallest angular difference between 
        # the pointing vector and a vector from the finger base to the object
        smallestAngDiff = 2 * math.pi
        bestFitObj = None
        for obj in objects:
            objX = obj.pose.position.x
            objY = obj.pose.position.y
            objZ = obj.pose.position.z
            obj_xzangle = getAngle(userdata.bx, userdata.bz, objX, objZ);
            diff = getAngleDiff(userdata.point_angle, obj_xzangle)
            if(diff < smallestAngDiff):
                smallestAngDiff = diff
                bestFitObj = obj
        
        self.speech.say("You are pointing to the %s. " % bestFitObj.text)
        userdata.times_out = userdata.times_in + 1
        if(userdata.times_out >= 5):
            self.speech.say("Task done.")
            return 'finished'
        else:
            self.speech.say("I can help you identify another object.")
            return 'succeeded'
        

        
#TODO class
class IdentifyObject(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=[])        

    
#TODO class
class EndState(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=[])
    def execute():



# ACTUAL CODE WE WROTE GOES HERE (END)


# states needed
# waitforstart, findbodyreferee, findhand, createvector, identifyobject, endstate