#!/usr/bin/python
# -*- coding: utf-8 -*-

import cv2
import mediapipe as mp


def initialize():
    global mp_drawing
    global mp_drawing_styles
    global mp_hands
    global hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, model_complexity=1,
                        min_detection_confidence=0.5,
                        min_tracking_confidence =0.25)
    return True


def process(cv2_img):
    returnList = []
    results = hands.process(cv2.cvtColor(cv2_img,
                            cv2.COLOR_BGR2RGB))

    # Draw the hand annotations on the image.

    cv2_img.flags.writeable = True
    cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(cv2_img, hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

        # print the coordinates

            (image_height, image_width, _) = (480, 640, None)
            tipX = \
                hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width
            tipY = \
                hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height
            tipZ = \
                hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].z

            baseX = \
                hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x * image_width
            baseY = \
                hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y * image_height
            baseZ = \
                hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].z

            wristX = \
                hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * image_width
            wristY = \
                hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * image_height

        # print("Finger tip: x = " + str(tipX)
        #     + ", y = " + str(tipY)
        #     + ", z = " + str(tipZ))

        # draw fingertip dot

            cv2.circle(cv2_img, (int(tipX), int(tipY)), 10, (0, 0, 255), -1)

        # draw finger base dot

            cv2.circle(cv2_img, (int(baseX),
                        int(baseY)), 10, (255, 0, 0),
                        -1)

        # draw connecting line

            cv2.line(cv2_img, (int(baseX), int(baseY)), (int(tipX),
                        int(tipY)), (255, 0, 0), 2)

        # Flip the image horizontally for a selfie-view display.

            returnList = [tipX, tipY, tipZ, baseX, baseY, baseZ, wristX, wristY]

    return returnList
