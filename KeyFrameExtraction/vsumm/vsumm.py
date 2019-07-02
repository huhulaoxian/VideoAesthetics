import numpy as np
import matplotlib
matplotlib.use('TkAGG')
from matplotlib import pyplot as plt
import cv2
import os
import math
import video_segmentation as vs
import feature_extraction as feat
import clusterization as cl
import shutil
import time
import argparse

def vsumm_frames_in_memory(video):
    segmentation = vs.VideoSegmentation(video)
    frames = segmentation.read_and_keep_frames()

    if len(frames) == 0:
        return False

    features = feat.extract_features(frames)
    keyframes = cl.find_clusters(features)

    # summary_folder = 'summaryM-'+video[7:-4]
    # if not os.path.isdir(summary_folder):
    #     os.mkdir(summary_folder)
    #
    # for k in keyframes:
    #     frame = frames[k.frame_id-1]
    #     frame_name = summary_folder+'/frame-'+str(k.frame_id).zfill(6)+'.jpg'
    #     cv2.imwrite(frame_name,frame)

    return True

def vsumm_frames_in_disk(video):
#     frames_folder = 'frames-'+video[7:-4]
#     if not os.path.isdir(frames_folder):
#         os.mkdir(frames_folder)

    segmentation = vs.VideoSegmentation(video)
    frames = segmentation.read_and_keep_frames()
    features = feat.extract_features(frames)
    keyframes = cl.find_clusters(features)

    # segmentation = vs.VideoSegmentation(video)
    # segmentation.read_and_save_frames(frames_folder)
    # frames_list = os.listdir(frames_folder)
    # features = feat.read_frames_extract_features(frames_folder,frames_list)
    # keyframes = cl.find_clusters(features)

#     summary_folder = 'summaryD-'+video[7:-4]
#     if not os.path.isdir(summary_folder):
#         os.mkdir(summary_folder)
    keyframes_value = [frames[k.frame_id] for k in keyframes]
    for num,keyframe in enumerate(keyframes_value):
        frame_name = 'opencv'+str(num)+'.jpg'
        cv2.imwrite(frame_name, keyframe)
#     for k in keyframes:
#         kframe = frames_folder+'/frame-'+str(k.frame_id).zfill(6)+'.jpg'
#         shutil.copy(kframe,summary_folder)


def main():
    
    parser = argparse.ArgumentParser(description='Extraction of keyframes using VSUMM')
    parser.add_argument('video_file',type=str, help='input video_file_path')
    parser.add_argument('csv_path',type=str, help='input csv_file_path')
    
    args = parser.parse_args()
    
    videofile = args.video_file
    csv_path = args.csv_path
    start = time.time()  
    if not vsumm_frames_in_memory(videofile):
        print('cannot keep all frames in memory')

    vsumm_frames_in_disk(videofile)
    end = time.time()
    elapsed_time = end-start
    print ('elapsed time vsumm with frames in disk:', elapsed_time)
if __name__ == '__main__':
    main()