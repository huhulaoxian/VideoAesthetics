import cv2
import matplotlib

matplotlib.use('TkAGG')
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import argrelextrema
import argparse


class Frame:
    def __init__(self, id, frame, value):
        self.id = id
        self.frame = frame
        self.value = value


def CalHist(img):
    b, g, r = cv2.split(img)
    bHist = cv2.calcHist([b], [0], None, [256], [0, 256])
    gHist = cv2.calcHist([g], [0], None, [256], [0, 256])
    rHist = cv2.calcHist([r], [0], None, [256], [0, 256])

    return bHist, gHist, rHist


def main():
    parser = argparse.ArgumentParser(description="key frame extraction using RGB color histogram")

    parser.add_argument('video_path', type=str, help='video path')
    parser.add_argument('saved_path', type=str, help='file path for extracted frame')
    parser.add_argument('numOfFrame', type=int, help='Paremeter to frame you want from video')

    args = parser.parse_args()

    video_path = args.video_path
    saved_path = args.saved_path
    numOfFrame = args.numOfFrame

    cap = cv2.VideoCapture(video_path)

    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = round(cap.get(cv2.CAP_PROP_FPS))

    print("video frame count : ", length, "video width : ", width, "video height : ", height, "fps : ", fps)

    cap = cv2.VideoCapture(video_path)

    ret, fram = cap.read()
    prev_values = fram
    diff_list = []
    frames = []

    for i in range(length):
        ret, fram = cap.read()

        if i % fps == 0:
            next_values = fram
            diff = cv2.absdiff(prev_values,next_values)
            total_diff = np.sum(diff)

            diff_list.append(total_diff)
            prev_values = next_values
            frame = Frame(i, fram, total_diff)
            frames.append(frame)

    # identify local maxima
    diff_array = np.array(diff_list)
    frame_indexes = np.asarray(argrelextrema(diff_array, np.greater))[0]

    diff_dict = {}
    for i in frame_indexes:
        diff_dict[i] = diff_array[i]

    sorted_dict = sorted(diff_dict.items(), key=lambda x: x[1], reverse=True)
    sorted_indexes = [j[0] for i, j in enumerate(sorted_dict) if i < numOfFrame]

    for i in sorted_indexes:
        name = "/frame_" + str(frames[i - 1].id) + ".jpg"
        cv2.imwrite(saved_path + name, frames[i - 1].frame)

    plt.figure(figsize=(40, 20))
    plt.locator_params(numticks=100)
    plt.stem(diff_array)
    plt.savefig(saved_path + '/plot.png')


if __name__ == "__main__":
    main()