#imports
import cv2
import mediapipe as mp
import time

class Hands_detection():
    def __init__(self,mode=False,maxhands=2,detection_confidence=0.5,tracking_confidence=0.5):
        self.mode = mode
        self.maxhands= maxhands
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence

        self.mphands = mp.solutions.hands
        self.hands = self.mphands.Hands(self.mode,self.maxhands,self.detection_confidence,self.tracking_confidence)
        self.mpDraw = mp.solutions.drawing_utils

    def hands_detect(self,img,draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.result = self.hands.process(img_rgb)
        # print(result.multi_hand_landmarks)

        if self.result.multi_hand_landmarks:
            for land in self.result.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, land, self.mphands.HAND_CONNECTIONS)

            return img

    def find_locations(self,img,handNo=0,draw=True):
        landmark_loc = []
        if self.result.multi_hand_landmarks:
            my_hand = self.result.multi_hand_landmarks[handNo]
            for id, loc in enumerate(my_hand.landmark):
                h, w = img.shape[:2]
                px, py = int(loc.x * w), int(loc.y * h)
                landmark_loc.append([id,px,py])

                if draw:
                    cv2.circle(img,(px,py),12,(255,0,255),cv2.FILLED)
            return  landmark_loc


def main():
    frame = cv2.VideoCapture(0)
    previous_time = 0
    current_time = 0
    detector = Hands_detection()
    while True:
        _, img = frame.read()

        imk = detector.hands_detect(img)
        locations = detector.find_locations(imk)
        if len(locations) != 0:
            print(locations[4])


        current_time = time.time()
        fps = 1 / (current_time - previous_time)
        previous_time = current_time

        cv2.putText(imk, str(int(fps)), (20, 80), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 2)

        cv2.imshow("Image",img)
        cv2.waitKey(1)


if __name__=='__main__':
    main()

































