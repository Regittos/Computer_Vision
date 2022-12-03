import time
import cv2
import mediapipe as mp

class FaceMeshDetector():
    def __init__(self,static_image = False, num_faces = 2,
                 refine = False, detection_con = 0.5, tracking_con = 0.5):
        self.static_image = static_image
        self.num_faces = num_faces
        self.refine = refine
        self.detection_con = detection_con
        self.tracking_con = tracking_con

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.static_image,self.num_faces,self.refine,
                                                 self.detection_con, self.tracking_con)
        self.drawSpec = self.mpDraw.DrawingSpec(color=(0,255,0),thickness = 1, circle_radius=1 )


    def findFaceMesh(self, img, draw=True):
        faces=[]
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRGB)
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms,self.mpFaceMesh.FACEMESH_CONTOURS,self.drawSpec)
                face = []
            for id,lm in enumerate(faceLms.landmark):
                #print(lm)
                ih, iw, ic= img.shape
                x,y, = int(lm.x*iw), int(lm.y*ih)
#                cv2.putText(img, str(id), (x, y), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
                #print(id,x,y)
                face.append([x, y])
            faces.append(face)
        return img,faces



def main():
    #cap = cv2.VideoCapture('Repo/1.mp4 ')
    cap = cv2.VideoCapture(0)
    pTime = 0
    detector = FaceMeshDetector()
    while True:
        success, img = cap.read()
        img,faces=detector.findFaceMesh(img)
        if len(faces)!=0:
            print(faces[0])
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()