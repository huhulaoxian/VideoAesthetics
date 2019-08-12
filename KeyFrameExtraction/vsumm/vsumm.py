import numpy as np
import matplotlib
matplotlib.use('TkAGG')
from matplotlib import pyplot as plt
import cv2
import os
import math
import video_segmentation as vs
import yg_feature_extraction as feat
import yg_clusterization as cl
import shutil
import time
import argparse
import csv

def bgr2hsv(x):
    hsv_frame = cv2.cvtColor(x, cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(hsv_frame)
    return h

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

def vsumm_frames_in_disk(video,csv_path):
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

    video_file = video.split('/')[1]
    video_name = video_file.split('.')[0]

    summary_folder = 'summary/D-' + video_name

    if not os.path.isdir(summary_folder):
        os.mkdir(summary_folder)

    keyframes_value = [frames[k.frame_id] for k in keyframes if np.std(bgr2hsv(frames[k.frame_id])) > 0.1]


    for num,keyframe in enumerate(keyframes_value):
        frame_name = summary_folder+'/' + video_name + '_' + str(num)+'.jpg'
        cv2.imwrite(frame_name, keyframe)

        with open(csv_path, 'a', newline='') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',')
            spamwriter.writerow([video_file, frame_name])

def main():
    
    parser = argparse.ArgumentParser(description='Extraction of keyframes using VSUMM')
    parser.add_argument('video_file',type=str, help='input video_file_path')
    parser.add_argument('csv_path',type=str, help='input csv_file_path')
    
    args = parser.parse_args()

    video_file = args.video_file
    csv_path = args.csv_path

    file_name = csv_path
    if not os.path.exists(file_name):
        with open(file_name, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(['video_name', 'frame_name'])

    error_name = 'error.csv'
    if not os.path.exists(file_name):
        with open(error_name, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(['video_name'])

    # if not vsumm_frames_in_memory(videofile):
    #     print('cannot keep all frames in memory')


    video_list = os.listdir(video_file)

    for video in video_list:
        try:
            start = time.time()
            video_path = os.path.join(video_file,video)
            vsumm_frames_in_disk(video_path,csv_path)
            end = time.time()
            elapsed_time = end - start
            print ('elapsed time vsumm with frames in disk:', elapsed_time)
        except:
            with open(error_name, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter=',')
                writer.writerow([video])

if __name__ == '__main__':
    main()