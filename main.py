import mediapipe as mp 
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

cap = cv2.VideoCapture(0)

# Initiate holistic model

def getPointXY(pt1, index, width, height):
    x = int(pt1[index].x * width)
    y = int(pt1[index].y * height)
    return (x,y)

def pointDistancexXY2(XY1, XY2):
    x1,y1 = XY1
    x2,y2 = XY2
    X = x1 - x2
    Y = y1 - y2
    distance = math.sqrt((X*X) + (Y*Y))
    return distance

while cap.isOpened():
    ret, frame = cap.read()
    height, width, _ = frame.shape
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = face_mesh.process(rgb_image)

    pt1 = []
    if not result.multi_face_landmarks == None:
        for facial_landmarks in result.multi_face_landmarks:
            for i in range(0, 468):
                pt1.append(facial_landmarks.landmark[i])

    if not result.multi_face_landmarks == None:

        pointDistance = {}

        # mouth distance
        XY1 = getPointXY(pt1, 0, width, height)
        cv2.circle(frame, XY1, 1, (100,100,0), 1)
        XY2 = getPointXY(pt1, 17, width, height)
        cv2.circle(frame, XY2, 1, (100,100,0), 1)
        distance = pointDistancexXY2(XY1,XY2)
        pointDistance.update({'mouthDistance': distance})

        # right eye
        XY1 = getPointXY(pt1, 159, width, height)
        cv2.circle(frame, XY1, 1, (100,100,0), 1)
        XY2 = getPointXY(pt1, 145, width, height)
        cv2.circle(frame, XY2, 1, (100,100,0), 1)
        distance = pointDistancexXY2(XY1,XY2)
        pointDistance.update({'rightEye': distance})

        # left eye
        XY1 = getPointXY(pt1, 386, width, height)
        cv2.circle(frame, XY1, 1, (100,100,0), 1)
        XY2 = getPointXY(pt1, 374, width, height)
        cv2.circle(frame, XY2, 1, (100,100,0), 1)
        distance = pointDistancexXY2(XY1,XY2)
        pointDistance.update({'leftEye': distance})



        print(str(pointDistance['rightEye']) + " " + str(pointDistance['leftEye']))
        
    cv2.imshow('Raw Webcam Feed', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


