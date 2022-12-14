import cv2
import logging
from process import process
import os

vid_files = os.listdir("videos")

for f_vid in vid_files:
    f_vid = "videos/"+f_vid
    # open video
    cap = cv2.VideoCapture(f_vid)


    if not cap.isOpened():
        logging.error("cant open file %s",f_vid)
        exit(1)
    else:
        logging.info("working on %s", f_vid)

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