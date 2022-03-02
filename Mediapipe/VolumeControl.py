import cv2
import HandTrackingModule as htm
import numpy as np
import time
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume


def main():
    cap = cv2.VideoCapture(0)
    detector = htm.HandDetector(detectionCon=0.7)
    pTime = 0
    cTime = 0

    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))

    print(volume.GetVolumeRange())
    volRange = volume.GetVolumeRange()
    minVol = volRange[0]
    maxVol = volRange[1]
    vol = minVol
    volBar = 400

    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img, draw=False)
        
        if len(lmList) != 0:
            x1, y1 = lmList[4][1], lmList[4][2]
            x2, y2 = lmList[8][1], lmList[8][2]
            mx, my = (x1 + x2) // 2, (y1 + y2) // 2

            cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 10, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            vol = np.interp(length, [30, 250], [minVol, maxVol])
            volBar = np.interp(length, [30, 250], [400, 150])
            volume.SetMasterVolumeLevel(vol, None)
            # print(length, vol)

            if length < 30:
                cv2.circle(img, (mx, my), 10, (0, 0, 255), cv2.FILLED)
            else:
                cv2.circle(img, (mx, my), 10, (0, 255, 0), cv2.FILLED)
        
        cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
        cv2.rectangle(img, (48, int(volBar)), (85, 400), (0, 255, 0), cv2.FILLED)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        img = cv2.flip(img, 1)
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.imshow('Image', img)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()