# Facial Expression Recognition Project

# import libraries
from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np

face_classifier = cv2.CascadeClassifier(r'C:\Users\Cash Crusaders\anaconda3\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml')
classifier = load_model(r'C:\Users\Cash Crusaders\Desktop\My Portfolio\Projects\Data Science Projects\Computer Vision Project 2 - Facial Expression Recognition ( using Keras)\Emotion_little_vgg.h5')

class_labels = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']

cap = cv2.VideoCapture(0) # if using external camera then place 1 not 0

# grab a single frame of video
while True:
    ret, frame = cap.read()
    labels = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # convert input img into gray
    faces = face_classifier.detectMultiScale(gray, 1.3, 5) # scale down input img data

# we want to draw a rectangle in the face
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)


        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

        # make prediction on the ROI(reason of interest) and look up in the folders for the class it represents
            preds = classifier.predict(roi)[0]
            label = class_labels[preds.argmax()]
            label_pos = (x, y)
            cv2.putText(frame, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        else:
            cv2.putText(frame, 'No face found', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
    cv2.imshow('Emotion Detector ', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


