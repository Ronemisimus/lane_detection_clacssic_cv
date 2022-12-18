import cv2
import numpy as np

def process(frame:np.ndarray):
    height, width, depth = frame.shape

    checked_area = cut_img_center(frame,width,height)    

    checked_area = cv2.cvtColor(checked_area, cv2.COLOR_BGR2GRAY)

    # resize image to get constant resolution? - might make the process better on diffrent videos in deffrent resolutions

    # equilize histogram
    
    # bilateralFilter

    # dilate and erode pre canny

    # canny

    # make double lines into single lines with a dilate erode oporation

    # filate erode post canny
    
    # tresh hold instead of canny?

    # hough transform

    # choose best lines
    
    return checked_area

def cut_img_center(img,width,height):
    min_y = 4*height//7
    min_x = 2*width//7
    max_x = 5*width//7

    return img[min_y:,min_x:max_x]


def histEquallize(img):
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8,8))

    return clahe.apply(img)

