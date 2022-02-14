import cv2
import mediapipe as mp
import time


class FaceDetector():
    def __init__(self, minDetectionCon=0.5):
        self.minDetectionCon = minDetectionCon

        self.mpFaceDetection = mp.solutions.face_detection
        self.faceDetection = self.mpFaceDetection.FaceDetection()
        self.mpDraw = mp.solutions.drawing_utils
    

    def findFace(self, img, draw=True, drawkeypoints=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)
        self.facialKeypoints = mp.solutions.face_detection.FaceKeyPoint
        bboxs = []
        keypoints = {}
        
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                boundingBox = detection.location_data.relative_bounding_box

                h, w, c = img.shape
                bbox = (int(boundingBox.xmin * w), int(boundingBox.ymin * h),
                        int(boundingBox.width * w), int(boundingBox.height * h))
                bboxs.append([id, bbox, detection.score])
                for kp in self.facialKeypoints: 
                    keypoint = mp.solutions.face_detection.get_key_point(detection, kp)
                    keypoints[kp.name] = {"x" : int(keypoint.x * w), "y" : int(keypoint.y * h)}
                    if drawkeypoints:
                        cv2.circle(img, (keypoints[kp.name]["x"], keypoints[kp.name]["y"]), 3, (0, 0, 255), 2)

                if draw:    
                    #cv2.rectangle(img, bbox, (0, 255, 0), 2)
                    img = self.fancyDraw(img, bbox)
                    cv2.putText(img, f'{int(detection.score[0] * 100)}%', (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 0), 2)
        
        return img, bboxs, keypoints
    
    
    def fancyDraw(self, img, bbox, length=30, thickness=7):
        x, y, w, h = bbox
        cv2.rectangle(img, bbox, (0, 255, 0), 2)
        cv2.line(img, (x, y), (x + length, y), (0, 255, 0), thickness)
        cv2.line(img, (x, y), (x, y + length), (0, 255, 0), thickness)

        cv2.line(img, (x + w, y), (x + w - length, y), (0, 255, 0), thickness)
        cv2.line(img, (x + w, y), (x + w, y + length), (0, 255, 0), thickness)

        cv2.line(img, (x, y + h), (x + length, y +h), (0, 255, 0), thickness)
        cv2.line(img, (x, y + h), (x, y +h - length), (0, 255, 0), thickness)

        cv2.line(img, (x + w, y + h), (x + w - length, y + h), (0, 255, 0), thickness)
        cv2.line(img, (x + w, y + h), (x + w, y + h - length), (0, 255, 0), thickness)
        
        return img


def main():
    cap = cv2.VideoCapture(0)
    detector = FaceDetector()
    pTime = 0
    cTime = 0   

    while True:
        success, img = cap.read()
        img, bboxs, keypoints = detector.findFace(img, drawkeypoints=False)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)


        cv2.imshow('Image', img)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
