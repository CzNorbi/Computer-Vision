import cv2
import HandTrackingModule as htm
import numpy as np


def landmarkDifference(lm1, lm2):
    return np.sqrt((lm2[1] - lm1[1])**2 + (lm2[2] - lm1[2])**2)


def main():
    cap = cv2.VideoCapture(0)
    detector = htm.HandDetector(detectionCon=0.7)
    i = 0

    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img)  
        fingers = []

        if len(lmList) != 0:
            if landmarkDifference(lmList[0], lmList[4]) > landmarkDifference(lmList[0], lmList[1]):
                fingers.append(1)
            else:
                fingers.append(0)

            if landmarkDifference(lmList[0], lmList[8]) > landmarkDifference(lmList[0], lmList[5]):
                fingers.append(1)
            else:
                fingers.append(0)

            if landmarkDifference(lmList[0], lmList[12]) > landmarkDifference(lmList[0], lmList[9]):
                fingers.append(1)
            else:
                fingers.append(0)

            if landmarkDifference(lmList[0], lmList[16]) > landmarkDifference(lmList[0], lmList[13]):
                fingers.append(1)
            else:
                fingers.append(0)
                
            if landmarkDifference(lmList[0], lmList[20]) > landmarkDifference(lmList[0], lmList[19]):
                fingers.append(1)
            else:
                fingers.append(0)

        print(fingers)

        cv2.imshow('Image', cv2.flip(img, 1))

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()