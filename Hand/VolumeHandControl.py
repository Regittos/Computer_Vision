import cv2
import time
import numpy as np
import HandTrackingModule as htm
import math

from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume


cap = cv2.VideoCapture(0)
wCam, hCam = 640, 480
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0

detector = htm.handDetector(detectionCon=0.7)



devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
#volume.GetMute()
volume.GetMasterVolumeLevel()
print(volume.GetVolumeRange())

volRange = volume.GetVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]

vol = 0
while True:
    success, img = cap.read()
    img=detector.findHands(img)
    lmList = detector.findPosition(img, draw = False)
    if len(lmList) !=0:
        #print(lmList[4], lmList[8])

        x1,y1 = lmList[4][1], lmList[4][2]
        x2,y2 = lmList[8][1], lmList[8][2]
        cx,cy = (x1+x2)//2, (y1+y2)//2

        cv2.circle(img, (x1,y1), 13, (255,0,0),cv2.FILLED)
        cv2.circle(img, (x2,y2), 13, (255,0,0),cv2.FILLED)
        cv2.line(img,(x1,y1),(x2,y2),(255,0,0),3)
        cv2.circle(img, (cx, cy), 13, (255, 0, 0), cv2.FILLED)
        length = math.hypot(x2-x1,y2-y1)
        #print(length)
        #Hand range 50 - 300
        #Volume range 65.25 - 0
        vol = np.interp(length, [20,150],[minVol, maxVol])
       # print(vol)

        volume.SetMasterVolumeLevel(vol, None)

    cv2.rectangle(img, (50, 180), (85, 400),(255,0,0),3)
    bar = np.interp(vol, [minVol, maxVol],[400,180])
    #bar = 400 - int(2*vol)
    cv2.rectangle(img, (50, int(bar)), (85, 400),(0,255,0),cv2.FILLED)
    vo = np.interp(vol, [minVol, maxVol], [0, 100])
    cv2.putText(img, f'Volume: {str(int(vo))}', (50, 140), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)


    cTime = time.time()
    fps=1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {str(int(fps))}', (40, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    cv2.imshow("Img", img)
    cv2.waitKey(1)