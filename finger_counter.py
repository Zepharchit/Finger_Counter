import numpy as np
import cv2
import mediapipe as mp
import hand_tracking_template as htt
import time

w,h = 640,640
frame = cv2.VideoCapture(0)
frame.set(3,w)
frame.set(4,h)
curr_time =0
prev_time =0
detector = htt.Hands_detection(detection_confidence=0.75)
# as per mediapipe for finger tips
tip_id = [4,8,12,16,20]
while True:

    _ , img = frame.read()
    imk =  detector.hands_detect(img)
    landmarks = detector.find_locations(imk,draw=False)
    #print(landmarks)

    if len(landmarks)!=0:
        fingers = []
        #for thumb checking closing with respect ot the x axis
        # currently for left hand
        if landmarks[tip_id[0]][1] < landmarks[tip_id[0]-1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        for id in range(1,5):

            if landmarks[tip_id[id]][2] < landmarks[tip_id[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        #print(fingers)
        totalFinger = fingers.count(1)
        #print(totalFinger)
        cv2.putText(imk, str(int(totalFinger)), (320, 80), cv2.FONT_HERSHEY_PLAIN, 6, (255, 0, 0), 5)

    curr_time = time.time()
    fps = 1 / (curr_time-prev_time)
    prev_time = curr_time

    cv2.putText(imk, str(int(fps)), (20, 80), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 2)

    cv2.imshow("Image", img)
    cv2.waitKey(1)