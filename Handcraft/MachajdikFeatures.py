import cv2
from collections import defaultdict
import numpy as np

def dynamics(img):
    static_dict = defaultdict(list)
    slant_dict = defaultdict(list)

    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(imgray,100,200)
    lines = cv2.HoughLines(edges, 1, np.pi/180,130)
    
    if lines is None:
        return [0,0,0,0,0,0]
    
    for line in lines:
        length, thetha = line[0]
        degree = thetha * 57.2958
        if (degree > -15 and degree < 15) or (degree > 75 and degree < 105):
            static_dict['length'].append(length)
            static_dict['degree'].append(degree)
        else:
            slant_dict['length'].append(length)
            slant_dict['degree'].append(degree)

    if len(static_dict) == 0:
        len_statics = 0
        degree_statics = 0
        abs_degree_statics = 0
    else:
        len_statics = np.mean(static_dict['length'])
        degree_statics = np.mean(static_dict['degree'])
        abs_degree_statics = np.mean(np.abs(static_dict['degree']))

    if len(slant_dict) == 0:
        len_dynamics = 0
        degree_dynamics = 0
        abs_degree_dynamics = 0
    else:
        len_dynamics = np.mean(slant_dict['length'])
        degree_dynamics = np.mean(slant_dict['degree'])
        abs_degree_dynamics = np.mean(np.abs(slant_dict['degree']))

    return [len_statics, degree_statics, abs_degree_statics, len_dynamics, degree_dynamics, abs_degree_dynamics]

def LevelOfDetail(img):
   # binaray image로 변환
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    #Morphology의 opening, closing을 통해서 노이즈나 Hole제거
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel,iterations=2)

    # dilate를 통해서 확실한 Backgroud
    sure_bg = cv2.dilate(opening,kernel,iterations=3)

    #distance transform을 적용하면 중심으로 부터 Skeleton Image를 얻을 수 있음.
    # 즉, 중심으로 부터 점점 옅어져 가는 영상.
    # 그 결과에 thresh를 이용하여 확실한 FG를 파악
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    ret, sure_fg = cv2.threshold(dist_transform,0.5*dist_transform.max(),255,0)
    sure_fg = np.uint8(sure_fg)

    # Background에서 Foregrand를 제외한 영역을 Unknow영역으로 파악
    unknown = cv2.subtract(sure_bg, sure_fg)

    ret, markers = cv2.connectedComponents(sure_fg)

    return ret