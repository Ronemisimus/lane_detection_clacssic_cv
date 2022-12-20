import cv2
import numpy as np

def process(frame:np.ndarray):
    height, width, depth = frame.shape    
    checked_area = frame.copy()

    # resize to treat all videos equally?

    # equilize histogram per channel
    b,g,r  = cv2.split(checked_area)
    b = histEquallize(b)
    g = histEquallize(g)
    r = histEquallize(r)
    checked_area = cv2.merge((b,g,r))

    # convert to gray
    checked_area = cv2.cvtColor(checked_area, cv2.COLOR_BGR2GRAY)

    # Filter
    checked_area = cv2.GaussianBlur(checked_area,ksize=(5,5),sigmaX=0)

    # dilate and erode to remove noise
    kernel_params = (3,7)
    checked_area = cv2.erode(checked_area,left_lane_kernel(*kernel_params),iterations=1)
    checked_area = cv2.dilate(checked_area,left_lane_kernel(*kernel_params),iterations=1)
    
    checked_area = cv2.erode(checked_area,right_lane_kernel(*kernel_params),iterations=1)
    checked_area = cv2.dilate(checked_area,right_lane_kernel(*kernel_params),iterations=1)

    # canny
    checked_area = cv2.Canny(checked_area,420,500)

    # dilate and erode to remove noise
    kernel_params = (3,30)
    checked_area = cv2.dilate(checked_area,left_lane_kernel(*kernel_params),iterations=2)
    checked_area = cv2.erode(checked_area,left_lane_kernel(*kernel_params),iterations=2)
    
    checked_area = cv2.dilate(checked_area,right_lane_kernel(*kernel_params),iterations=2)
    checked_area = cv2.erode(checked_area,right_lane_kernel(*kernel_params),iterations=2)

    # dilate and erode to connect striped lines
    #checked_area = cv2.dilate(checked_area,np.ones((30,1)),iterations=1)
    #checked_area = cv2.erode(checked_area,np.ones(30,1),iterations=1)
    

    """ # HSL
    checked_area = cv2.cvtColor(checked_area,cv2.COLOR_BGR2HLS_FULL)
    avg_l = np.average(checked_area[:,:,1][checked_area[:,:,1]>10])

    # HSL threshold
    h, l, s = cv2.split(checked_area)

    l = cv2.erode(l,right_lane_kernel(),iterations=1)
    l = cv2.dilate(l,right_lane_kernel(),iterations=1)
    
    l = cv2.erode(l,left_lane_kernel(),iterations=1)
    l = cv2.dilate(l,left_lane_kernel(),iterations=1)

    relation = 1.2
    change_map = l>relation * avg_l   
    l[change_map]=relation*l[change_map]
    checked_area = cv2.merge((h,l,s))

    # back to BGR
    checked_area = cv2.cvtColor(checked_area,cv2.COLOR_HLS2BGR_FULL)
    #checked_area = cv2.cvtColor(checked_area,cv2.COLOR_BGR2GRAY) """

    # sobel?
    """ left = cv2.Sobel(checked_area,cv2.CV_8U,1,0,ksize=3)
    kernel = np.array([[1,0,-1],[2,0,-2],[1,0,-1]]) 
    right = cv2.filter2D(checked_area,-1,kernel=kernel)
    verticals = left+right
    verticals = remove_triangle_edge(verticals,width,height)
    checked_area = verticals """


    # connect close lines (both sides of the thick lane line)


    # threshold
    #max_blue, max_green, max_red = np.max(checked_area,axis=(0,1))
    #ratio =0.3
    #lowb = (int(max_blue*ratio),int(max_green*ratio),int(max_red*ratio))
    #highb = (int(max_blue),int(max_green),int(max_red))
    #checked_area = cv2.inRange(checked_area, lowb, highb)
    
    # make double lines into single lines with a dilate erode oporation
    #checked_area = cv2.filter2D(checked_area,-1,kernel=left_lane_kernel(2,7)*-1)
    #checked_area = cv2.filter2D(checked_area,-1,kernel=right_lane_kernel(2,7)*-1)

    #checked_area = cv2.erode(checked_area,left_lane_kernel(1,5),iterations=2)
    
    #checked_area = cv2.erode(checked_area,right_lane_kernel(1,5),iterations=2)



    # extract triangle from image
    checked_area = cut_img_center(checked_area,width,height)
    

    # hough transform
    tol = 0.5
    lines = cv2.HoughLinesP(checked_area,1,np.pi/180,50,minLineLength=50,maxLineGap=10)
    if lines is not None:
        right_lane = []
        left_lane = []
        for line in lines:
            x1,y1,x2,y2 = line[0]
            p = np.polyfit([x1,x2],[y1,y2],1)
            if p[0]>=tol:
                right_lane.append((x1,y1))
                right_lane.append((x2,y2))
            elif p[0]<=-tol:
                left_lane.append((x1,y1))
                left_lane.append((x2,y2))
            cv2.line(frame,(x1,y1),(x2,y2),(0,0,255),2)
        
        
        # choose best lines
        if len(left_lane)>0:
            left_lane = np.array(left_lane)
            left_lane = np.polyfit(left_lane[:,0],left_lane[:,1],1)
            x_left =  np.array([[2*width//5,1],[5*width//11,1]])
            y_left = (x_left@left_lane).astype(int)
            cv2.line(frame,(x_left[0,0],y_left[0]),(x_left[1,0],y_left[1]),(0,255,0),2)
        else:
            x_left=None
        
        if len(right_lane)>0:
            right_lane = np.array(right_lane)
            right_lane = np.polyfit(right_lane[:,0],right_lane[:,1],1)
        
            x_right = np.array([[3*width//5,1],[6*width//11,1]])
            y_right = (x_right@right_lane).astype(int)

            cv2.line(frame,(x_right[0,0],y_right[0]),(x_right[1,0],y_right[1]),(0,255,0),2)
        else:
            x_right = None

        if x_left is not None and x_right is not None:
            pts = [np.array([(x_left[0,0],y_left[0]),(x_right[0,0],y_right[0]),(x_right[1,0],y_right[1]),(x_left[1,0],y_left[1])], dtype=np.int32)]

            cv2.fillPoly(frame,pts,color=(0,170,0))

    return frame

def cut_img_center(img,width,height):
    center = [width//2,height//2]
    right = [width,height-height//10]
    left = [0,height-height//10]

    WHITE = (1,1,1)
    pts = [np.int32([
        left,
        center,  
        right])]

    mask = np.zeros_like(img, dtype=np.uint8)
    cv2.fillPoly(mask, pts, color=WHITE)

    img = np.where(mask ==1,img,0)

    return img


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

