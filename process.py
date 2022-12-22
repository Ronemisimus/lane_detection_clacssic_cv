import cv2
import numpy as np

def process(frame:np.ndarray,prev_lines):
    height, width, depth = frame.shape    
    checked_area = frame.copy()

    checked_area = equalize_per_channel(checked_area)

    checked_area = to_gray(checked_area)

    checked_area = blur(checked_area,(5,5))

    high = 360
    low=high//3

    edges = canny(checked_area,low,high)

    checked_area = frame.copy()

    checked_area = equalize_per_channel(checked_area)

    checked_area = close(checked_area,left_lane_kernel(3,5))
    checked_area = close(checked_area,right_lane_kernel(3,5))

    dual_sobel = vertical_sobel(checked_area)

    dual_sobel = equalize_per_channel(dual_sobel)

    high = 850
    low=high//2
    dual_sobel = canny(dual_sobel,low,high)

    checked_area = np.bitwise_and(edges,dual_sobel)

    # extract triangle from image
    checked_area = cut_img_center(checked_area,width,height)

    lines = get_lines(checked_area)
    left_lane,right_lane = separate_lines(lines,0.3)

    lines = choose_best_lines(frame,left_lane, right_lane)

    lines, prev_lines = accumalative_avg(lines,prev_lines)
    
    if lines is not None and np.sum(np.isnan(lines))==0:
        lines = np.int32(lines)
        x1,y1,x2,y2 = lines[0]
        cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 10)
        x1,y1,x2,y2 = lines[1]
        cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 10)  

    return frame, prev_lines

def cut_img_center(img,width,height):
    center = [width//2,height//2]
    right = [width,height-height//10]
    left = [0,height-height//10]
    left_corner=[0,height]
    right_corner=[width,height]

    WHITE = (1,1,1)
    pts = [np.int32([
        left,
        center,  
        right])]

    mask = np.zeros_like(img, dtype=np.uint8)
    cv2.fillPoly(mask, pts, color=WHITE)

    pts = [np.int32([
        left,right,right_corner,left_corner
    ])]

    cv2.fillPoly(mask,pts,color=WHITE)

    img = np.where(mask ==1,img,0)

    return img

def equalize_per_channel(img):
    b,g,r  = cv2.split(img)
    b = histEquallize(b)
    g = histEquallize(g)
    r = histEquallize(r)
    return cv2.merge((b,g,r))

def histEquallize(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    return clahe.apply(img)

def right_lane_kernel(k,m):
    return np.array([
        [-1]*i+[1]*k+[-1]*(m-k-i)
        for i in range(0,m-k+1)
    ])

def left_lane_kernel(k,m):
    return np.array([
        [-1]*(m-k-i)+[1]*k+[-1]*i
        for i in range(0,m-k+1)
    ])

def hsl_transform(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2HLS_FULL)

    low_yellow = (73,161,120)
    high_yellow = (86,254,255)

    yellow_mask = cv2.inRange(img,low_yellow,high_yellow)

    white_mask = cv2.inRange(img,200,255)

    return np.bitwise_or(yellow_mask,white_mask)

def vertical_sobel(img):
    left = cv2.Sobel(img,cv2.CV_8U,1,0,ksize=3)
    kernel = np.array([[1,0,-1],[2,0,-2],[1,0,-1]]) 
    right = cv2.filter2D(img,-1,kernel=kernel)
    return np.bitwise_or(left,right)

def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def blur(img,ksize):
    return cv2.GaussianBlur(img,ksize=ksize,sigmaX=0)

def canny(img,low,high):
    return cv2.Canny(img,low,high)

def open(img,kernel):
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

def close(img,kernel):
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

def get_lines(edges):
    return cv2.HoughLinesP(edges,1,np.pi/180,10,minLineLength=30,maxLineGap=10)

def separate_lines(lines, tol):
    right_lane = []
    left_lane = []
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line[0]
            p = np.polyfit([x1,x2],[y1,y2],1)
            if p[0]>=tol:
                right_lane.append(p)
            elif p[0]<=-tol:
                left_lane.append(p)
    return left_lane, right_lane

def choose_best_lines(img,left_lane,right_lane):
    # choose best lines
    if len(left_lane)>0 and len(right_lane)>0:
        right_avg = np.average(right_lane, axis=0)
        left_avg = np.average(left_lane, axis=0)
        left_line = make_points(img, left_avg)
        right_line = make_points(img, right_avg)
        return np.array([left_line, right_line])
    return None

def make_points(image, average): 
    slope, y_int = average 
    y1 = image.shape[0]
    y2 = int(y1*3//5)
    x1 = int((y1-y_int)//slope)
    x2 = int((y2-y_int)//slope)
    return np.array([x1, y1, x2, y2])

def accumalative_avg(lines,prev_lines):
    if lines is not None:
        if len(lines)<3:
            prev_lines.append(lines)
        else:
            prev_lines = prev_lines[1:]+[lines]
        return lines, prev_lines
    elif len(prev_lines)!=0:
        total:np.ndarray = np.array(prev_lines)
        return np.average(total,axis=0), prev_lines
    else:
        return np.nan, prev_lines




