import cv2
from collections import defaultdict
import numpy as np

def dynamics(img):
    static_dict = defaultdict(list)
    slant_dict = defaultdict(list)

    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(imgray,100,200)
    lines = cv2.HoughLines(edges, 1, np.pi/180,130)

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