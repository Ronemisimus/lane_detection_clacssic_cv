import cv2
import numpy as np

def process(frame:np.ndarray,prev_lines):
    height, width, depth = frame.shape    

    work_frame = equalize_per_channel(frame)

    work_frame = blur(work_frame,(3,3))

    work_frame = close(work_frame,left_lane_kernel(3,20))

    work_frame = close(work_frame,right_lane_kernel(3,20))

    high = 160
    low = high*2//3
    work_frame = canny(to_gray(work_frame),low,high)

    work_frame = cut_img_center(work_frame,width,height)

    lines = get_lines(work_frame)    

    left_lane, right_lane = separate_lines(lines,15*np.pi/180,width,height)

    lines = choose_best_lines(frame,left_lane,right_lane)

    #lines, prev_lines = accumalative_avg(lines,prev_lines)

    draw_lines(frame,lines,(0,0,255),True)

    frame = draw_rect(frame, lines, (255,255,255),True)

    return frame, prev_lines

def cut_img_center(img,width,height):
    center_left = [width//2-width//20,height//2]
    center_right = [width//2+width//20,height//2]
    right = [width,height-height//4]
    left = [0,height-height//4]
    left_corner=[0,height]
    right_corner=[width,height]

    WHITE = (1,1,1)
    pts = [np.int32([
        left,
        center_left,
        center_right,  
        right])]

    mask = np.zeros_like(img, dtype=np.uint8)
    cv2.fillPoly(mask, pts, color=WHITE)

    pts = [np.int32([
        left,right,right_corner,left_corner
    ])]

    cv2.fillPoly(mask,pts,color=WHITE)

    img = np.where(mask==1,img,0)

    return img

def equalize_per_channel(img):
    b,g,r  = cv2.split(img)
    b = histEquallize(b)
    g = histEquallize(g)
    r = histEquallize(r)
    return cv2.merge((b,g,r))

def histEquallize(img):
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(16,16))

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
    return cv2.HoughLinesP(edges,1,np.pi/180,40,minLineLength=20,maxLineGap=10)

def separate_lines(lines, tol,width,height):
    right_lane = []
    left_lane = []
    x_offset_tol_out = width*2/3
    x_offset_tol_in = width/4
    refrence_vector = np.array([0,1])
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line[0]
            max_y = max(y1,y2)
            if max_y == y1:
                x1,x2 = x2,x1
                y1,y2 = y2,y1 
            line_base = np.array([y2-y1,x2-x1])
            line_base_norm = line_base/np.linalg.norm(line_base)
            theta = np.arccos(refrence_vector@line_base_norm)
            base_mult = (height-y2)/line_base[0]
            base_point = np.array([y2,x2])+base_mult*line_base
            x_base = base_point[1]
            if theta > tol and theta< np.pi-tol and x_base != np.inf and x_base!=-np.inf:
                if theta>=np.pi/2 and x_base < width/2 - x_offset_tol_in and x_base>-x_offset_tol_out:
                    left_lane.append((theta,x_base))
                elif theta <=np.pi/2 and x_base > width/2 + x_offset_tol_in and x_base<width+x_offset_tol_out:
                    right_lane.append((theta,x_base))
    return left_lane, right_lane

def group_lines(lines,steps,field):
    line_groups = []
    line_group_centers=[]
    for line in lines:
        index, = np.nonzero(np.abs(np.array(line_group_centers)-line[field])<steps)
        if len(index)==0:
            line_groups.append([line])
            line_group_centers.append(line[field])
        else:
            index = np.argmin(np.array(line_group_centers)-line[field])
            line_groups[index].append(line)
            line_group_centers[index] = np.average(line_groups[index],axis=0)[field]
    return line_groups, line_group_centers



def choose_best_lines(img,left_lane,right_lane):
    width = img.shape[1]
    # choose best lines
    if len(left_lane)>0 and len(right_lane)>0:
        left_lane_groups, left_lane_groups_centers = group_lines(left_lane,100,1)
        right_lane_groups, right_lane_groups_centers = group_lines(right_lane,10,1)
        close_group_left = np.argmin(abs(np.array(left_lane_groups_centers)-width/2))
        close_group_right = np.argmin(abs(np.array(right_lane_groups_centers)-width/2))
        left_lane_groups, left_lane_groups_centers = group_lines(left_lane_groups[close_group_left],5*np.pi/180,0)
        right_lane_groups, right_lane_groups_centers = group_lines(right_lane_groups[close_group_right],5*np.pi/180,0)
        max_group_left = np.argmax([len(group) for group in left_lane_groups])
        max_group_right = np.argmax([len(group) for group in right_lane_groups])
        right_avg = np.average(right_lane_groups[max_group_right], axis=0)
        left_avg = np.average(left_lane_groups[max_group_left], axis=0)
        return [left_avg, right_avg]
    return None

def make_points(image, average): 
    theta, x_base = average
    height,width,depth = image.shape
    x = np.cos(theta)
    base = np.array([np.sqrt(1-np.square(x)),x])
    base_point = np.array([height,x_base])
    y1 = image.shape[0]*6//7
    pt1_mult = (height-y1)/base[0]
    x1 = base_point - pt1_mult*base
    x1 = int(x1[1])
    y2 = int(y1*2//3)
    pt2_mult = (height-y2)/base[0]
    x2 = base_point - pt2_mult*base
    x2 = int(x2[1])
    return np.array([x1, y1, x2, y2])

def accumalative_avg(lines,prev_lines):
    if lines is not None:
        if len(prev_lines)<2:
            prev_lines.append(lines)
        else:
            prev_lines = prev_lines[1:]+[lines]
        temp_prev_lines = np.array(prev_lines)
        w = np.ones((len(temp_prev_lines)))
        w[-1] = 2
        lines:np.ndarray = np.average(temp_prev_lines,axis=0,weights=w)
        lines = [line for line in lines]
        return lines, prev_lines
    elif len(prev_lines)!=0:
        total:np.ndarray = np.array(prev_lines)
        lines = np.average(total,axis=0)
        lines = [line for line in lines]
        return lines, prev_lines
    else:
        return np.nan, prev_lines

def draw_lines(img, lines,color,run_make_points):
    if lines is not None and np.sum(np.isnan(lines))==0:
        for line in lines:
            if run_make_points:
                x1,y1,x2,y2 = make_points(img,line)
            else:
                x1,y1,x2,y2 = line
            cv2.line(img, (x1,y1),(x2,y2),color,2)


def draw_rect(frame,lines,color,run_make_points):
    line_img = np.zeros_like(frame)
    if lines is not None and np.sum(np.isnan(lines))==0:
        if run_make_points:
            x1,y1,x2,y2 = make_points(frame,lines[0])
            x3,y3,x4,y4 = make_points(frame,lines[1])
        else:
            x1,y1,x2,y2 = lines[0]
            x3,y3,x4,y4 = lines[1]
        pts = [np.array([[x1,y1],[x2,y2],[x4,y4],[x3,y3]])]
        
        cv2.fillPoly(line_img,pts,color=color)
    frame = cv2.addWeighted(frame,0.8,line_img,0.2,10)
    return frame

