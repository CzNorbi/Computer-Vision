import cv2
import mediapipe as mp
import time

class poseDetector():

    def __init__(self, mode=False, complex=1, smooth_landmarks=True, segmentation=True, smooth_segmentation=True,
                 detectionCon=0.5, trackCon=0.5):

        self.mode = mode
        self.complex = complex
        self.smooth_landmarks = smooth_landmarks
        self.segmentation = segmentation
        self.smooth_segmentation = smooth_segmentation
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpDrawStyle = mp.solutions.drawing_styles
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.complex, self.smooth_landmarks, self.segmentation,
                                     self.smooth_segmentation, self.detectionCon, self.trackCon)

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        drawSpecCircle = self.mpDraw.DrawingSpec(circle_radius=2, color=(0, 0, 255))
        drawSpecLine = self.mpDraw.DrawingSpec(thickness=3, color=(0, 255, 0))
        self.results = self.pose.process(imgRGB)

        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,
                                           self.mpPose.POSE_CONNECTIONS, drawSpecCircle, drawSpecLine)
        return img

    def getPosition(self, img, draw=True):
        lmlist = []

        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                lmlist.append([id, cx, cy])

                if draw:
                    cv2.circle(img, (cx, cy), 5 , (255, 0, 255), cv2.FILLED)

        return lmlist

def main():
    cap = cv2.VideoCapture(0)
    pTime = 0
    cTime = 0
    detector = poseDetector()
    while True:
        success, img = cap.read()
        img = detector.findPose(img)
        lmList = detector.getPosition(img)

        if len(lmList) != 0:
            print(lmList)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.imshow("Image", cv2.flip(img, 1))
        if cv2.waitKey(1) == ord('q'):
            break


if __name__ == "__main__":
    main()