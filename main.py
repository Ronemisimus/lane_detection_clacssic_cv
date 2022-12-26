import cv2
import logging
from process import process
import os
from color_logs import CustomFormatter
import sys

# create logger with 'spam_application'
logger = logging.getLogger("lane detector")
logger.setLevel(logging.DEBUG)

save = True
out_fname = "out_videos/first_algo_lane_detection.mp4"

# create console handler with a higher log level
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)

ch.setFormatter(CustomFormatter())

logger.addHandler(ch)

vid_files = os.listdir("videos")

for f_vid in vid_files:
    f_vid = "videos/"+f_vid
    # open video
    cap = cv2.VideoCapture(f_vid)

    if save:
        out = cv2.VideoWriter(out_fname, cv2.VideoWriter_fourcc(*'X264'),10, (1920,1080))

    if not cap.isOpened() or not out.isOpened():
        logger.error("cant open file %s",f_vid)
        exit(1)
    else:
        logger.info("working on %s", f_vid)
    prev_lines=[]
    counter=60
    message =''
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            
            frame, prev_lines, message, counter = process(frame,prev_lines,message,counter)

            # Display the resulting frame
            cv2.imshow('Frame',frame)
        
            if save:
                out.write(frame)
            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        
            # Break the loop
        else: 
            break
    if save and out.isOpened():
        out.release()