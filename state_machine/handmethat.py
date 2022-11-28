#!/usr/bin/env python
import math

import rospy
import smach
from threading import Event
from hsrb_interface import Robot, exceptions
from knowledge_representation import xml_kbase
from smach import StateMachine
from villa_top_level.spr.spr_hint_phrases import spr_hint_phrases

from spr_qa.questionanswerer import get_default_spr_answerer
from villa_helpers.fast_move import FastMove

from villa_top_level import common
from villa_top_level.common import control_flow, motion, speech, knowledge, util
#from villa_top_level.spr import states, machines

import states



def main():
    # rospy.init_node('spr')

    # access robot's parts
    fastmove = None
    answer_base = get_default_spr_answerer()
    kbase = xml_kbase.get_default_xml_kbase()
    util.register_transcription_hints(kbase.question_parser.get_all_questions())
    while not rospy.is_shutdown():
        try:
            robot = Robot()
            whole_body = robot.get('whole_body')
            omni_base = robot.get('omni_base')
            fastmove = FastMove(omni_base)
            break
        except (exceptions.ResourceNotFoundError, exceptions.RobotConnectionError) as e:
            rospy.logerr('Failed to obtain resources: {}\nRetrying...'.format(e))

    whole_body.move_to_neutral()

    sm = StateMachine(["succeeded", "aborted"])
    sm.userdata.si1 = 0
    sm.userdata.si2 = 0
    sm.userdata.si3 = 0
    sm.userdata.si4 = 0
    sm.userdata.si5 = 0
    sm.userdata.si6 = 0
    sm.userdata.sm_angle = 0
    sm.userdata.times_done = 0
    
    with sm:
        StateMachine.add("WAIT_START", states.WaitForStart(),
                transitions={"signalled": "FIND_REF", "not_signalled": "WAIT_START"})
        StateMachine.add("FIND_REF", states.FindBodyReferee(omni_base),
                transitions={"succeeded": "FIND_HAND"})
        StateMachine.add("FIND_HAND", states.FindHand(speech), 
                transitions={"succeeded": "CREATE_VECTOR"},
                remapping={"bx": "si1", "by": "si2", "bz": "si3", 
                        "tx": "si4", "ty": "si5", "tz": "si6"})
        StateMachine.add("CREATE_VECTOR", states.CreateVector(), 
                transitions={"succeeded": "BOUND_BOXES"},
                remapping={"bx": "si1", "by": "si2", "bz": "si3", 
                        "tx": "si4", "ty": "si5", "tz": "si6",
                        "point_angle": "sm_angle"})
        StateMachine.add("BOUND_BOXES", states.GetBoundingBoxes(speech), 
                transitions={"succeeded": "FIND_HAND",
                            "aborted": "BOUND_BOXES",
                            "finished": "succeeded"},
                remapping={"bx": "si1", "bz": "si3", 
                        "point_angle": "sm_angle",
                        "times_in": "times_done",
                        "times_out": "times_done"})

    # Execute SMACH plan
    outcome = sm.execute()


if __name__ == '__main__':
    main()