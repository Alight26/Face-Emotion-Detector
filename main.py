import cv2 as cv 
from deepface import DeepFace 

emotion = cv.CascadeClassifier('haarcascade_frontalFace_default.xml')

cap = cv.VideoCapture(1)

while True:
    ret, img = cap.read()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = emotion.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv.rectangle(img, (x,y), (x+w, y+h), (0, 0, 225), 2)
        roi_gray = gray[x:x+w, y:y+h]
        roi_color = img[x:x+w, y:y+h]


    cv.imshow('img', img)
    k = cv.waitKey(30) & 0xff
    if k == ord('q'):
        break



