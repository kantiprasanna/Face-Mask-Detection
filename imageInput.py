#import statements
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img, ImageDataGenerator
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from keras.models import Sequential
import numpy as np
import matplotlib.pyplot as plt
import  PIL
from PIL import Image
import argparse
from imutils import paths
import matplotlib.pyplot as plt
import cv2


cap = cv2.VideoCapture(0)  # starts the camera 


while True:
    ret, frame = cap.read()   # reads a bool if the frame is read correctly
    cv2.imshow('frame',frame)     # to show the frame
    if cv2.waitKey(1) & 0xFF == ord('q'):  # when we press 'q' the image will be captured
        break

cap.release() # closes already opened file or camera
cv2.destroyAllWindows() # closes all the opened windows

img = Image.fromarray(frame) # converts numpy-array to image 
img.save("test.png")  # saves the image to local memory

model =  load_model("finalVersion")    # loads the trained model
img_path = "test.png"     # path of the image which we saved previously

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')  # loads the classifier used to detect frontal-face
img = cv2.imread(img_path)  # reads the image from the specified path
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # converts the image from blue-green-red to gray
faces = face_cascade.detectMultiScale(img, 1.1, 4)   # detects multiscale or faces

for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)   # dimensions of the rectangle to be extracted
    face = img[y : y + w, x : x + h]    # saves just the rectangular faces 
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB) # converts image from blue-green-red to red-green-blue
    face = cv2.resize(face, (224, 224))  # resizes the image 
    face = img_to_array(face)  # converts the image into numpy_array
    face = preprocess_input(face)  # adequates the image to the format which the model requires
    face = np.expand_dims(face, axis=0)   # expands the shape of the array
    (mask, withoutMask) = model.predict(face)[0]   # predicts the output and the values are stored in mask and withoutMask
    label = "Mask" if mask > withoutMask else "No Mask"   # checks whether the input image is with-mask or without-mask
    color = (0, 255, 0) if label == "Mask" else (0, 0, 255)    
    label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)   
    cv2.putText(img, label, (x, y - 10),      # places the text onto the image
    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)  # sets the font style for the output text
    cv2.rectangle(img, (x, y), (x + h, y + w), color, 2) # draws the rectagle onto the output image

cv2.waitKey(0)    # waits for specified milliseconds
   
cv2.destroyAllWindows()  # closes all opened windows 

imag = Image.fromarray(img,'RGB')   # converts numpy-array to image in red-green-blue
imag.save('final.png')  # saves the image into local disk
imag.show()  # shows the image in specific frame
