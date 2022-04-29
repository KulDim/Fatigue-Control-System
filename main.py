
import mediapipe as mp 
import cv2
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()


cap = cv2.VideoCapture(0)

# Initiate holistic model
    
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
        x = int(pt1[0].x * width)
        y = int(pt1[0].y * height)
        cv2.circle(frame, (x,y), 1, (100,100,0), 0)


    cv2.imshow('Raw Webcam Feed', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()