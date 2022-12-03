import cv2
import time
import os
import HandTrackingModule as htm
import pyautogui as pg
import math

wCam, hCam = 640, 480
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

pTime = 0
detector = htm.handDetector()
x, y = pg.size()
while True:

    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img)
    if len(lmList) != 0:
        cx, cy = int(lmList[8][1] * x / wCam), int(lmList[8][2] * y / hCam)
        pg.moveTo(x-cx, cy)
        lenght = int(math.hypot(lmList[8][1] - lmList[4][1], lmList[8][2] - lmList[4][2]))
        #lenght =  lmList[4][1] - lmList[8][1]
        if lenght < 15:
            pg.click()
        print(lenght)


    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS:{str(int(fps))}',(50,70),cv2.FONT_HERSHEY_PLAIN, 3 , (255,0,0),3)
    cv2.imshow("Image",img)
    cv2.waitKey(1)




