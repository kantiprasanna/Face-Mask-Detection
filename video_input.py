import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img, ImageDataGenerator
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from keras.models import Sequential
from glob import glob
import matplotlib.pyplot as plt
import  PIL
from PIL import Image
#%matplotlib inline
import argparse
from imutils import paths
import matplotlib.pyplot as plt

cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)
model =  load_model("C:/Users/keert/miniproject/finalVersion")  
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


while(True):
    (ret,frame) = cap.read() 

    faces = face_cascade.detectMultiScale(frame)

    for (x, y, w, h) in faces:       
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face = frame[y : y + w, x : x + h]
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (224, 224))
        face = img_to_array(face)
        face = preprocess_input(face)
        face = np.expand_dims(face, axis=0)
        (mask, withoutMask) = model.predict(face)[0]
        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
        cv2.putText(frame, label, (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (x, y), (x + h, y + w), color, 2)

    cv2.imshow('frame',frame)
    c = cv2.waitKey(1)
    if c & 0xFF == ord('q'):
        break