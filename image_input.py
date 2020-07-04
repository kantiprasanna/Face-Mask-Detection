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


cap = cv2.VideoCapture(0)  #to start camera which is connected to pc


while True:
    ret, frame = cap.read()   #to read image frames 
    cv2.imshow('frame',frame)     #to show the image 
    if cv2.waitKey(1) & 0xFF == ord('q'):  #to quit the frame
        break

cap.release() 
cv2.destroyAllWindows() #closes all opened windows

img = Image.fromarray(frame) #to convert numpy-array to image
img.save("test.png")  #to save image into local memory

model =  load_model("C:/Users/keert/miniproject/finalVersion")    #to load the trained model
img_path = "C:/Users/keert/miniproject/test.png"     #path of the image

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')  #to load the xml file
img = cv2.imread(img_path)  #to read the image from the specified path
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  #to convert the image into BGR2BGRAY
faces = face_cascade.detectMultiScale(img, 1.1, 4)   #detect multiscale or faces

for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)   #
    face = img[y : y + w, x : x + h]    #save just the rectangle faces into face
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB) # to change the color
    face = cv2.resize(face, (224, 224))  # resize the image 
    face = img_to_array(face)  # to convert into numpy-array
    face = preprocess_input(face)  #to process the input 
    face = np.expand_dims(face, axis=0)  
    (mask, withoutMask) = model.predict(face)[0]   #predict the output form the model 
    label = "Mask" if mask > withoutMask else "No Mask"   #check whether the input image is with-mask or without-mask
    color = (0, 255, 0) if label == "Mask" else (0, 0, 255)   
    label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)   
    cv2.putText(img, label, (x, y - 10),      #to put text onto the image
    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
    cv2.rectangle(img, (x, y), (x + h, y + w), color, 2)   #to draw rectangle on the output

cv2.waitKey(0)   
  
#closing all open windows  
cv2.destroyAllWindows()  

imag = Image.fromarray(img,'RGB')   #to convert numpy-array to image
imag.save('final.png')  #to save image
imag.show()  #to show the output