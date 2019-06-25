import DattaFeatures
import WuFeatures
import cv2
import argparse
import csv

def main():
    parser = argparse.ArgumentParser(description='Datta features')
    parser.add_argument('img_path', type=str, help='input your img path')
    parser.add_argument('csv_path', type=str, help='input your save path')

    args = parser.parse_args()

    img_path = args.img_path
    csv_path = args.csv_path

    img = cv2.imread(img_path)
    resized_img = cv2.resize(img, dsize=(128, 128), interpolation=cv2.INTER_AREA)
    HSV_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2HSV)

    IH, IS, IV = cv2.split(HSV_img)
    channels = [IH, IS, IV]

    feature_vector = []

    # Exposure of Light
    feature_vector.append(DattaFeatures.f1(IV))

    # Saturation and Hue
    feature_vector.append(DattaFeatures.f3(IS))
    feature_vector.append(DattaFeatures.f4(IH))

    # The Rule of Thirds
    feature_vector.append(DattaFeatures.f5(IH))
    feature_vector.append(DattaFeatures.f6(IS))
    feature_vector.append(DattaFeatures.f7(IV))

    # Size and Aspect Ratio
    feature_vector.append(DattaFeatures.f22(img))
    feature_vector.append(DattaFeatures.f23(img))

    color = WuFeatures.Color()
    texture = WuFeatures.Texture()

    # The effects of saturation and brightness on emotion (e.g., pleasure, arousal and dominance)
    wu_f1_3 = color.f1_3(img)
    for i in wu_f1_3:
        feature_vector.append(i)

    # Colorfulness
    feature_vector.append(color.f4(img))

    # W3C colors
    wu_f10_25 = color.f10_25(img)
    for i in wu_f10_25:
        feature_vector.append(i)

    # Tamura features
    wu_f26_28 = texture.f26_28(img_path)
    for i in wu_f26_28:
        feature_vector.append(i)

    # Wavelet-based features
    wu_f29_40 = texture.f29_40(img)
    for i in wu_f29_40:
        feature_vector.append(i)

    # Gray-Level Co-occurance Matrix
    wu_f41_44 = texture.f41_44(img)
    for i in wu_f41_44:
        feature_vector.append(i)

    with open(csv_path, 'a') as f:
        wr = csv.writer(f)
        wr.writerow(feature_vector)
    print(len(feature_vector))

    return feature_vector

if __name__=="__main__":
    main()