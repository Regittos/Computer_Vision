import cv2
import mediapipe as mp
import time

class FaceDetector():
    def __init__(self,detecionCon = 0.75, model_selection = 0):
        self.detectionCon = detecionCon
        self.model_selection = model_selection
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(detecionCon, model_selection)

    def findFaces(self,img, draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)
        bboxs = []
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                bboxC = detection.location_data.relative_bounding_box
                ih,iw,ic = img.shape
                bbox = int(bboxC.xmin * iw),int(bboxC.ymin * ih), \
                       int(bboxC.width * iw), int(bboxC.height * ih)
                bboxs.append([id,bbox,detection.score])
                if draw:
                    self.fancyDraw(img, bbox)

                    cv2.putText(img, f'{str(int(detection.score[0]*100))}%',
                            (bbox[0], bbox[1]), cv2.FONT_HERSHEY_PLAIN, 3,
                            (0, 255, 0), 2)
        return img,bboxs
    def fancyDraw(self,img, bbox,l=30, t = 10):
        x,y,w,h=bbox
        x1, y1 = x + w,h + y
        cv2.rectangle(img, bbox, (255, 0, 255), 2)
        cv2.line(img, (x, y), (x + l, y), (255, 0, 255), t)
        cv2.line(img, (x, y), (x, y + l), (255, 0, 255), t)
        cv2.line(img, (x1, y), (x1 - l, y), (255, 0, 255), t)
        cv2.line(img, (x1, y), (x1, y + l), (255, 0, 255), t)
        cv2.line(img, (x, y1), (x + l, y1), (255, 0, 255), t)
        cv2.line(img, (x, y1), (x, y1 - l), (255, 0, 255), t)
        cv2.line(img, (x1, y1), (x1, y1 - l), (255, 0, 255), t)
        cv2.line(img, (x1, y1), (x1 - l, y1), (255, 0, 255), t)
        return img
def main():
    cap = cv2.VideoCapture(0)
    #cap = cv2.VideoCapture('Repo/1.mp4')
    pTime = 0
    detector = FaceDetector(0.5)
    while True:
        success, img = cap.read()
        img,bboxs=detector.findFaces(img)
        print(bboxs)
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        cv2.putText(img,str(int(fps)),(70,50),cv2.FONT_HERSHEY_PLAIN,3,(0,0,255),2)
        cv2.imshow("Image", img)
        cv2.waitKey(10)

if __name__ == "__main__":
    main()
