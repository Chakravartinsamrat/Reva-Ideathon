import os


import cv2
import mediapipe as mp
import time

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' #some random warning from tensorflow

cap = cv2.VideoCapture(0) #cam 0

mpHands= mp.solutions.hands
hands= mpHands.Hands()  #remeber to put "n" number of hands later 

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)



    cv2.imshow("Image",img)
    cv2.waitKey(1)