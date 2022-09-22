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
  with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    return True
  return False

def process(image):
  # For webcam input:
  if not image:
    print("Ignoring empty camera frame.")
    # If loading a video, use 'break' instead of 'continue'.
    return False  # error

  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  results = hands.process(image)

  # Draw the hand annotations on the image.
  image.flags.writeable = True
  image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
  if results.multi_hand_landmarks:
    for hand_landmarks in results.multi_hand_landmarks:
      mp_drawing.draw_landmarks(
          image,
          hand_landmarks,
          mp_hands.HAND_CONNECTIONS,
          mp_drawing_styles.get_default_hand_landmarks_style(),
          mp_drawing_styles.get_default_hand_connections_style())
      
      # print the coordinates
      image_height, image_width, _ = image.shape
      tipX = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x
      tipY = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
      tipZ = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].z

      baseX = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x
      baseY = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y
      baseZ = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].z

      # print("Finger tip: x = " + str(tipX)
      #     + ", y = " + str(tipY)
      #     + ", z = " + str(tipZ))

      # draw fingertip dot
      cv2.circle(image,(int(tipX * image_width), int(tipY * image_height)), 10, (0,0,255), -1)  
      # draw finger base dot
      cv2.circle(image,(int(baseX * image_width), int(baseY * image_height)), 10, (255,0,0), -1)  
      # draw connecting line
      cv2.line(image, (int(baseX * image_width), int(baseY * image_height)), 
          (int(tipX * image_width), int(tipY * image_height)), (255,0,0), 2) 

      # Flip the image horizontally for a selfie-view display.
      cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))

      resultList = [tipX, tipY, tipZ, baseX, baseY, baseZ]
      return resultList
