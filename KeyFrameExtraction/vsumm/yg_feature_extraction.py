from sklearn.preprocessing import MinMaxScaler
import cv2
import numpy as np


class Histogram:
    def __init__(self, hist, histF, frame,id):
        self.hist = hist
        self.histF = histF
        self.frame = frame
        self.frame_id = id

def bgr2hsv(x):
    hsv_frame = cv2.cvtColor(x, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_frame)
    return h


def compute_histogram(frame, bins=16):
    hue_hist = []
    hist = cv2.calcHist([frame], None, None, [bins], [0, 180])
    hue_hist += list(np.array(hist).flatten())
    return hue_hist


def scaler(x):
    scaler = MinMaxScaler()
    X = np.array(x).reshape(-1, 1)
    scaler.fit(X)
    X_scaled = scaler.transform(X)
    return X_scaled


def extract_features(frames):
    hue_frames = list(map(bgr2hsv, frames))
    hue_hist = []
    frame_list = []
    for hue_frame in hue_frames:
        if np.std(hue_frame) > 0.5:
            hue_hist.append(compute_histogram(hue_frame))
            frame_list.append(hue_frame)

    hists = list(map(scaler, hue_hist))

    features = []

    for i, hist in enumerate(hists):
        features.append(Histogram(hist.flatten().tolist(), hist.flatten().tolist(), frame_list[i], i))

    return features