import cv2
import logging
from process import process

cap = cv2.VideoCapture("videos/0001-5400.mp4")

if not cap.isOpened():
    logging.error("cant open file %s","videos/0001-5400.mp4")
    exit(1)

while cap.isOpened():
    ret, frame = cap.read()
    if ret == True:
        
        frame = process(frame)

        # Display the resulting frame
        cv2.imshow('Frame',frame)
    
        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    
        # Break the loop
    else: 
        break