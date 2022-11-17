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


# ACTUAL CODE WE WROTE GOES HERE (START)

# Wait until the referee shakes the robot's wrist to start
# TODO change parameter to (smach.State)?
class WaitForStart(State):
    def __init__(self):
        State.__init__(self, outcomes=['signalled', 'not_signalled'])
        self.trigger_service_client = rospy.ServiceProxy('wrist_trigger', Trigger)

    def execute(self, userdata):
        # Assume we've been tapped in simulation
        
        try:
            if State.simulation:
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
    def __init__(self):
        smach.State.__init__(self, outcomes=['succeeded'])

    # using LocatePerson in personangle.py
    def execute(robot):
        omni_base = FastMove(robot.get('omni_base'))
        detect_person = LocatePerson(omni_base)
        detect_person.turn_to_person()
        return 'succeeded'

        # try:
        #     self.whole_body.gaze_point(
        #         point=geometry.Vector3(x=userdata["position"].x, y=userdata["position"].y, z=1),
        #         ref_frame_id='map')
        # except exceptions.FollowTrajectoryError:
        #     pass

#TODO class
class FindHand(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=[])
    def execute(robot):
        # run program point_tracker/src/point_tracker_node.cpp

#TODO comment
class CreateVector(smach.State):
    def execute(baseX, baseY, baseZ, tipX, tipY, tipZ):
        # frontal plane
        xyangle = math.atan((tipY - baseY)/(tipX - baseX))

        # median plane
        yzangle = math.atan((tipY - baseY)/(tipZ - baseZ))

        # horizontal plane
        xzangle = math.atan((tipZ - baseZ)/(tipX - baseX))

    def

#TODO class
class IdentifyObject(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=[])
    
    
# from groceries, convert this into getting the bounding boxes
class FindBoundingBoxes(smach.State):
    def __init__(self, ltmc, groovy_motion, fast_move, speech):
        smach.State.__init__(self, outcomes=['succeeded', 'aborted'], input_keys=['seetable_pose'],
                             output_keys=['picked_up_object', 'tweak_pose'])
        self.poses_pub = rospy.Publisher('grasp_poses', PoseArray, latch=True, queue_size=100)
        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)
        self.ltmc = ltmc
        self.groovy_motion = groovy_motion
        self.fast_move = fast_move
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

        self.pick_up = actionlib.SimpleActionClient('/villa/pick_up', PickUpAction)
        print("waiting for pick_pick action server")
        self.pick_up.wait_for_server(rospy.Duration(10))

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

    # currently just prints the positions of the bounding boxes 
    def execute(self, userdata):
        self.clear_octomap()

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
        # if len(bboxes.bounding_boxes.markers) <= 0:
        #     self.speech.say("Can't see any objects. Let me shimmy.")
        #     userdata.tweak_pose = utils.shimmy()
        #     return 'aborted'

        # Select random object to pick up
        objects = bboxes.bounding_boxes.markers[:]
        for obj in objects:
            print (obj.pose.position)

    
#TODO class
class EndState(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=[])
    def execute():



# ACTUAL CODE WE WROTE GOES HERE (END)


# states needed
# waitforstart, findbodyreferee, findhand, createvector, identifyobject, endstate




#______________________________________________________________________________
# POTENTIAL RESOURCES COPIED FROM OTHER FILES

# from common

import rospy
from plan_execution.helpers import *
from plan_execution.msg import ExecutePlanAction
from smach import State
from smach_ros import SimpleActionState
from std_msgs.msg import String
from std_srvs.srv import Trigger

topics = {'plan_execution': "/plan_executor/execute_plan",
          'request_transcript': "/google_speech/request_sound_transcript"}


class Wait(State):
    def __init__(self, amount):
        State.__init__(self, outcomes=["succeeded"])
        self.amount = amount

    def execute(self, ud):
        rospy.sleep(self.amount)
        return "succeeded"


class NoOp(Wait):
    def __init__(self):
        Wait.__init__(self, 0)


class WaitForStart(State):
    def __init__(self):
        State.__init__(self, outcomes=['signalled', 'not_signalled'])
        self.trigger_service_client = rospy.ServiceProxy('wrist_trigger', Trigger)

    def execute(self, userdata):
        # Assume we've been tapped in simulation
        
        try:
            if State.simulation:
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

class WaitForKeyPress(State):
    def __init__(self):
        State.__init__(self, outcomes=['signalled', 'not_signalled', 'aborted'])

    def execute(self, userdata):
        try:
            user_input = input("Press any key to start\n")
            return 'signalled'
        except KeyboardInterrupt:
            return 'aborted'



class GoTo(SimpleActionState):
    def __init__(self):
        smach.State.__init__(self, outcomes=['succeeded', 'aborted'])

    def navigate_to_cb(self, userdata, goal):
        goal_rule = to_asp_rule_body("not is_near", [1, userdata.location])
        goal.aspGoal = [goal_rule]


class GoToConcept(State):
    def __init__(self, fast_move, ltmc, destination):
        State.__init__(self, outcomes=['succeeded', 'aborted'])
        self.fast_move = fast_move
        self.ltmc = ltmc
        self.destination = destination

    def execute(self, userdata):
        location = self.ltmc.get_map('map').get_pose(self.destination)
        result = self.fast_move.go(location.x, location.y, yaw=location.theta, angular_thresh=10, relative=False)
        if result:
            return 'succeeded'
        else:
            return 'aborted'


class ReadQRCode(State):
    def __init__(self, timeout=30):
        State.__init__(self, outcomes=['succeeded', 'aborted'], output_keys=['text'])
        self.received_data = False
        self.question = ""
        self.timeout = timeout

    def callback(self, data):
        self.received_data = True
        self.question = data.data

    def execute(self, userdata):
        self.question = ""
        self.received_data = False
        rospy.Subscriber("/barcode", String, self.callback)

        r = rospy.Rate(1)  # 1hz
        time_cnt = 0
        while time_cnt < self.timeout and not rospy.is_shutdown():
            time_cnt += 1
            if self.received_data:
                userdata.text = self.question
                print("QR code parse: {}".format(self.question))
                return "succeeded"
            r.sleep()

        return 'aborted'


class ExecuteGoal(SimpleActionState):
    def __init__(self):
        SimpleActionState.__init__(self, topics["plan_execution"],
                                   ExecutePlanAction,
                                   goal_cb=self.goal_cb,
                                   result_cb=self.result_cb,
                                   input_keys=['goal'],
                                   output_keys=['result'])

    def goal_cb(self, userdata, goal):
        goal.aspGoal = userdata.goal.aspGoal

    def result_cb(self, userdata, state, result):
        print (state, result)
        userdata['result'] = result

    """
    def _goal_feedback_cb(self, feedback):
        rospy.logerr("GOT FEEDBACK :" + str(feedback))
        ACTION_START_TYPE = 2
        speech = None
        try:
            speech = villa_audio.tts.TextToSpeech(timeout=5.0)
        except RuntimeError:
            return
        if feedback.event_type == ACTION_START_TYPE:
            if feedback.plan[0].name == "navigate_to":
                speech.say("I'm starting to move.", wait=False)
            if feedback.plan[0].name == "perceive_surface":
                speech.say("I'm starting to scan.", wait=False)
    """


# from receptionist, change to find referee?
class FindHost(smach.State):
    def __init__(self, ltmc, groovy_motion, fast_move):
        smach.State.__init__(self, outcomes=['succeeded', 'aborted'])
        self.ltmc = ltmc
        self.groovy_motion = groovy_motion
        self.fast_move = fast_move
        self.lac = actionlib.SimpleActionClient('execute_logical_action', LogicalNavAction)
        self.lac.wait_for_server() 

    def execute(self, userdata):
        # logic_goal = LogicalNavGoal()
        # logic_goal.command.name = 'navigate_to'
        # logic_goal.command.value.append('seat_view')

        # res = self.lac.send_goal_and_wait(logic_goal)

        location = self.ltmc.get_map('map').get_pose('seat_view')
        result = self.fast_move.go(location.x, location.y, yaw=location.theta, angular_thresh=10, relative=False)
        if not result:
            return 'aborted'
        userdata.introduceTo = 'host'

        return 'succeeded'

# from receptionist, change to find referee?
class FindGuest(smach.State):
    def __init__(self, ltmc, groovy_motion, fast_move):
        smach.State.__init__(self, outcomes=['succeeded', 'aborted'])
        self.ltmc = ltmc
        self.groovy = groovy_motion
        self.fast_move = fast_move

    def execute(self, userdata):
        greet_pose = self.ltmc.greet_person(userdata.guest1_id)
        if greet_pose == []:
            return 'aborted'
        self.fast_move.go(greet_pose.x, greet_pose.y, yaw=greet_pose.theta, angular_thresh=5, relative=False)
        userdata.introduceTo = 'guest'

        return 'succeeded'

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


# from go and get it
class GoToPerson(smach.State):
    def __init__(self, ltmc, groovy_motion, fast_move, gripper):
        smach.State.__init__(self, outcomes=['succeeded', 'aborted', 'preempted'],
                             input_keys=['deposit'],
                             output_keys=['pre_force'])
        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)
        self.ltmc = ltmc
        self.groovy_motion = groovy_motion
        self.fast_move = fast_move
        self.gripper = gripper
        self.message = ""

        rospy.Subscriber("message", String, self.message_cb)


    def execute(self, userdata):
        
        print ("Going to deposit at ", userdata.deposit)

        deposit = self.ltmc.get_map('map').get_pose(userdata.deposit)

        approach_x = deposit.x
        approach_y = deposit.y
        approach_yaw = -deposit.theta

        # self.fast_move.go(2.37, 0.3, yaw=-math.pi/2, angular_thresh=5, relative=False)
        self.fast_move.go(approach_x, approach_y, yaw=approach_yaw, angular_thresh=5, relative=False)

        rospy.sleep(10)

        if self.message == "done":
            print ("done")
            return "succeeded"

        # open gripper
        self.gripper.command(1.3)

        return 'aborted'

    def message_cb(self, msg):
        self.message = msg.data

class NoticeCall(smach.State):
    def __init__(self, groovy, whole_body, referee=False, retry_phrases=[]):
        smach.State.__init__(self, outcomes=['succeeded', 'aborted'],
                             output_keys=["position"])
        self.groovy = groovy
        self.whole_body = whole_body
        self.retry_phrases = retry_phrases
        if referee:
            self.activity_type = ["BOTH_HANDS_UP", "RIGHT_HAND_UP", "LEFT_HAND_UP"]
        else:
            self.activity_type = ["BOTH_HANDS_UP", "RIGHT_HAND_UP", "LEFT_HAND_UP"]

        rospy.Subscriber('/openpose/activity', DetectedActivity, self.__activity_callback)
        self.activity = None

    def __activity_callback(self, msg):
        self.activity = msg

    def execute(self, userdata):
        # TODO: Work in simulation
        # try:
        #     if smach.State.simulation:
        #         userdata["position"] = ...
        #         return 'succeeded'
        # except AttributeError:
        #     pass

        pan_back_left = 1.7
        pan_back_right = -2.36
        pan_mid = (pan_back_left + pan_back_right) / 2.0

        sweep_time = 30
        trajectory = [
            {"head_pan_joint": pan_back_left, "time": sweep_time},
            {"head_tilt_joint": 0., "head_pan_joint": pan_back_right, "time": sweep_time},
            {"head_pan_joint": pan_mid, "time": sweep_time}
        ]
        motion = self.groovy.execute_trajectory(trajectory)

        if self.retry_phrases:
            speech.Say(self.retry_phrases[0], wait=False).execute(None)

        start_time = rospy.Time.now()
        go = None
        while (1==1):
            # Milliseconds
            if self.activity is not None:
                timediff = (rospy.Time.now() - self.activity.header.stamp).to_nsec() / 1e6

            if (self.activity is not None
                    and self.activity.activity in self.activity_type and timediff < 500):
                go = self.activity.location
                print(go.x, go.y, go.z)
                self.whole_body.move_to_joint_positions({"head_pan_joint": pan_mid})
                userdata["position"] = go
                break

            # Ask the referee again
            if self.retry_phrases and random.random() < 0.000001:
                if len(self.retry_phrases) > 0 and (rospy.Time.now() - start_time) > rospy.Duration(10):
                    speech.Say(random.choice(self.retry_phrases), wait=False).execute(None)
            if (rospy.Time.now() - start_time > rospy.Duration.from_sec(60)):
                motion = self.groovy.execute_trajectory(trajectory)
                start_time = rospy.Time.now()        

        if go is None:
            self.whole_body.move_to_joint_positions({"head_pan_joint": pan_mid})
            return 'aborted'

        return 'succeeded'



class RequestOrder(smach.State):
    def __init__(self, phrases):
        smach.State.__init__(self, outcomes=['succeeded'])
        self.phrases = phrases

    def execute(self, userdata):
        phrase = sorted(self.phrases, key=lambda k: random.random())[0]
        for part in phrase:
            speech.Say(part).execute(None)

        speech.Say("What food would you like to order?").execute(None)

        return 'succeeded'


class GotoPosition(smach.State):
    def __init__(self, fastmove, whole_body, plan_timeout=30):
        smach.State.__init__(self, outcomes=['succeeded', 'aborted'],
                             input_keys=["position"])
        self.whole_body = whole_body
        self.fastmove = fastmove

        rospy.wait_for_service('/viewpoint_controller/start')
        self.start_head = rospy.ServiceProxy('/viewpoint_controller/start', Empty)

        self.free_space = actionlib.SimpleActionClient('free_space/move', FreeSpaceAction)
        self.free_space.wait_for_server()

    def execute(self, userdata):
        self.start_head()

        goal = FreeSpaceGoal()
        goal.goal.header.frame_id = "map"
        goal.goal.pose.position.x = userdata["position"].x
        goal.goal.pose.position.y = userdata["position"].y
        goal.goal.pose.orientation.w = 1
        print("sending goal")
        res = self.free_space.send_goal_and_wait(goal)
	
        if res != actionlib.GoalStatus.SUCCEEDED:
            i = 0
            while(i < 100):
                if i % 5 == 0:
                    speech.Say("Calculating Route").execute(None)
                res = self.free_space.send_goal_and_wait(goal)
                if(res == actionlib.GoalStatus.SUCCEEDED):
                    break
                i += 1
            rospy.loginfo("free_space: Move Failed")

            dist = np.linalg.norm([self.fastmove.global_pose.pose.position.x - goal.goal.pose.position.x,
                                   self.fastmove.global_pose.pose.position.y - goal.goal.pose.position.y])
            if dist > 4:
                # Say make room for me?
                speech.Say("Clear Space!").execute(None)
                return 'aborted'

        # Move robot's head
        try:
            self.whole_body.gaze_point(
                point=geometry.Vector3(x=userdata["position"].x, y=userdata["position"].y, z=1),
                ref_frame_id='map')
        except exceptions.FollowTrajectoryError:
            pass

        return 'succeeded'


class RepeatOrder(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['succeeded', 'aborted'], input_keys=["transcript"])

    def execute(self, userdata):
        speech.Say("Hey barman").execute(None)
        speech.Say("Please prepare").execute(None)
        common_states.Wait(1).execute(None)
        speech.Say("Food!").execute(None)
        common_states.Wait(1.5).execute(None)
        speech.Say("Tap my wrist when you have prepared the order").execute(None)
        return 'succeeded'


class HandoverPose(smach.State):
    def __init__(self, whole_body, gripper):
        smach.State.__init__(self, outcomes=['succeeded', 'aborted'])
        self.whole_body = whole_body
        self.gripper = gripper

    def execute(self, userdata):
        speech.Say("I'll take that plate").execute(None)
        self.whole_body.move_to_joint_positions(HANDOVER_POSE)

        # Open Gripper
        self.gripper.command(1.0)

        speech.Say("Can you please get the plate and place it in my hand in.").execute(None)
        speech.Say("10").execute(None)
        speech.Say("9").execute(None)
        speech.Say("8").execute(None)
        speech.Say("7").execute(None)
        speech.Say("6").execute(None)
        speech.Say("5").execute(None)
        speech.Say("4").execute(None)
        speech.Say("3").execute(None)
        speech.Say("2").execute(None)
        speech.Say("1").execute(None)

        try:
            self.gripper.grasp(effort=-1)
        except Exception as e:
            pass

        self.whole_body.move_to_joint_positions(NEUTRAL_POSE)

        return 'succeeded'


class HandoverCustomer(smach.State):
    def __init__(self, groovy, gripper, whole_body):
        smach.State.__init__(self, outcomes=['succeeded', 'aborted'])
        self.groovy = groovy
        self.gripper = gripper
        self.whole_body = whole_body

    def execute(self, userdata):
        speech.Say("Here is your food.").execute(None)
        speech.Say("I will hand it to you my dear patron.").execute(None)

        speech.Say("Please take the plate from my hand in.").execute(None)
        speech.Say("10").execute(None)
        speech.Say("9").execute(None)
        speech.Say("8").execute(None)
        speech.Say("7").execute(None)
        speech.Say("6").execute(None)
        speech.Say("5").execute(None)
        speech.Say("4").execute(None)
        speech.Say("3").execute(None)
        speech.Say("2").execute(None)
        speech.Say("1").execute(None)

        try:
            self.gripper.command(1.0)
        except Exception as e:
            return 'aborted'

        speech.Say("Enjoy your meal!").execute(None)
        self.groovy.unsafe_move_to_neutral()(5)

        return 'succeeded'