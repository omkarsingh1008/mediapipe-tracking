import requests
import cv2
import numpy as np
import imutils
  
# Replace the below URL with your own. Make sure to add "/shot.jpg" at last.
cap = cv2.VideoCapture("http://192.168.0.104:8080/video")
  
# While loop to continuously fetching data from the Url
while True:
    ret,frame = cap.read()
    cv2.imshow("Android_cam", frame)
  
    # Press Esc key to exit
    if cv2.waitKey(1) == 27:
        break