#import statements

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

cap = cv2.VideoCapture(0)    # starts the camera 
cap.set(3,640)         # sets the width of the frame             
cap.set(4,480)         # sets the height of the frame
model =  load_model("finalVersion")   # loads the trained model
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')  # loads the classifier used to detect frontal-face


while(True):
    (ret,frame) = cap.read() # reads a bool if the frame is read correctly

    faces = face_cascade.detectMultiScale(frame)     # detects multiscale or faces

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)   # dimensions of the rectangle to be extracted
        face = frame[y : y + w, x : x + h]    # saves just the rectangular faces 
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB) # converts image from blue-green-red to red-green-blue
        face = cv2.resize(face, (224, 224))  # resizes the image 
        face = img_to_array(face)  # converts the image into numpy_array
        face = preprocess_input(face)  # adequates the image to the format which the model requires
        face = np.expand_dims(face, axis=0)   # expands the shape of the array
        (mask, withoutMask) = model.predict(face)[0]   # predicts the output and the values are stored in mask and withoutMask
        label = "Mask" if mask > withoutMask else "No Mask"   # checks whether the input image is with-mask or without-mask
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)    
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)   
        cv2.putText(frame, label, (x, y - 10),      # places the text onto the image
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)  # sets the font style for the output text
        cv2.rectangle(frame, (x, y), (x + h, y + w), color, 2) # draws the rectagle onto the output image


    cv2.imshow('frame',frame)   # shows the frames as the output
    c = cv2.waitKey(1)          # waits for specified milliseconds
    if c & 0xFF == ord('q'):    # when we press 'q' video stream stops ...breaking point
        break
