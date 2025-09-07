import cv2 as cv 
from deepface import DeepFace 

detector = cv.CascadeClassifier('haarcascade_frontalFace_default.xml')

cap = cv.VideoCapture(1)

while True:
    ret, img = cap.read()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 3)

    for (x, y, w, h) in faces:
        
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

        # Does the Facial Analysis 
        detection = DeepFace.analyze(roi_color, actions=['emotion'], enforce_detection=False)

        # detects the most dominant emotion 
        emotion = detection[0]['dominant_emotion']
        
        cv.rectangle(img, (x,y), (x+w, y+h), (0, 0, 225), 2)
        cv.putText(img, emotion, (50, 50), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1, cv.LINE_AA)





    cv.imshow('img', img)
    k = cv.waitKey(30) & 0xff
    if k == ord('q'):
        break



