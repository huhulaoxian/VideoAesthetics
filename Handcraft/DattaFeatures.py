from __future__ import division
import csv
import numpy as np
import pywt
import cv2
import argparse


# Exposure of Light
def f1(IV):
    return np.sum(IV) / (IV.shape[0] * IV.shape[1])


# Average Saturation / Saturation Indicator
def f3(IS):
    return np.sum(IS) / (IS.shape[0] * IS.shape[1])


# Average Hue / Hue Indicator
def f4(IH):
    return np.sum(IH) / (IH.shape[0] * IH.shape[1])


# Average hue in inner rectangle for rule of thirds inference
def f5(IH):
    X = IH.shape[0]
    Y = IH.shape[1]
    return sum(sum(IH[int(X / 3): int(2 * X / 3), int(Y / 3): int(2 * Y / 3)])) * 9 / (X * Y)


# Average saturation in inner rectangle for rule of thirds inference
def f6(IS):
    X = IS.shape[0]
    Y = IS.shape[1]
    return sum(sum(IS[int(X / 3): int(2 * X / 3), int(Y / 3): int(2 * Y / 3)])) * (9 / (X * Y))


# Average V in inner rectangle for rule of thirds inference
def f7(IV):
    X = IV.shape[0]
    Y = IV.shape[1]
    return sum(sum(IV[int(X / 3): int(2 * X / 3), int(Y / 3): int(2 * Y / 3)])) * (9 / (X * Y))


# Wavelet function
# Datta, R., Joshi, D., Li, J., & Wang, J. Z. (2006, May).
def f10_21(channels):
    epsilon = 50
    feature_values = []

    # Spacial Smoothness of (first~third) level of H,S,V property
    for channel in channels:
        coeffs = pywt.wavedecn(channel, wavelet='db1', level=3)
        levels = [1, 2, 3]
        for level in levels:
            ad, da, dd = coeffs[level]['ad'], coeffs[level]['da'], coeffs[level]['dd']
            numerator = np.sum(ad) + np.sum(da) + np.sum(dd)
            if numerator == 0:
                numerator = epsilon

            denominator = np.sum(abs(ad)) + np.sum(abs(da)) + np.sum(abs(dd))
            feature_values.append(numerator / denominator)

    # Sum of the average wavelet coefficients over all three frequency levels of H,S,V property
    feature_values.append(np.sum(feature_values[0:3]))
    feature_values.append(np.sum(feature_values[3:6]))
    feature_values.append(np.sum(feature_values[6:9]))

    return feature_values


# Image Size feature
def f22(img):
    return img.shape[0] + img.shape[1]


# Aspect Ratio Feature
def f23(img):
    return img.shape[0] / img.shape[1]

# Low Depth of Field Indicators 
def f53_55(channels):
    DOF_features = []
    for channel in channels:
        v1 = v2 = v3 = 0
        coeffs = pywt.wavedecn(channel, wavelet='db1', level=3)
        ad,da,dd = coeffs[3]['ad'], coeffs[3]['da'], coeffs[3]['dd']

        sumv1 = np.sum(ad[4:12,4:12])
        if sumv1 > 0:
            v1 = np.sum(np.abs(coeffs[3]['ad'][4:12,4:12])) / sumv1

        sumv2 = np.sum(da[4:12,4:12])
        if sumv2 > 0:
            v2 = np.sum(np.abs(coeffs[3]['da'][4:12,4:12])) / sumv2

        sumv3 = np.sum(dd[4:12,4:12])
        if sumv3 > 0:
            v3 = np.sum(np.abs(coeffs[3]['dd'][4:12,4:12])) / sumv3

        if sumv1 == 0:
            v1 = (v2 + v3)/2
        if sumv2 == 0:
            v2 = (v1 + v3)/2
        if sumv3 == 0:
            v3 = (v1 + v2)/2
            
        DOF_features.append(v1 + v2 + v3)
        
    return DOF_features
    
def main():
    global img
    parser = argparse.ArgumentParser(description='Datta features')
    parser.add_argument('img_path', type=str, help='input your img path')
    parser.add_argument('csv_path', type=str, help='input your save path')

    args = parser.parse_args()

    img_path = args.img_path
    csv_path = args.csv_path

    img = cv2.imread(img_path)

    resized_img = cv2.resize(img, dsize=(128, 128), interpolation=cv2.INTER_AREA)
    HSV_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2HSV)
    #LUV_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2LUV)

    IH, IS, IV = cv2.split(HSV_img)
    channels = [IH, IS, IV]

    feature_vec = []
    feature_vec.append(f1(IV))
    feature_vec.append(f3(IS))
    feature_vec.append(f4(IH))
    feature_vec.append(f5(IH))
    feature_vec.append(f6(IS))
    feature_vec.append(f7(IV))
    wavelet_features = f10_21(channels)

    for i in wavelet_features:
        feature_vec.append(i)

    feature_vec.append(f22())
    feature_vec.append(f23())

    DOF_features = f53_55(channels)
    
    for i in DOF_features:
        feature_vec.append(i)
        
    with open(csv_path, 'a') as f:
        wr = csv.writer(f)
        wr.writerow(feature_vec)
    print(len(feature_vec))

    return feature_vec

if __name__ == "__main__":
    main()