import DattaFeatures
import WuFeatures
import cv2
import argparse
import csv
import MachajdikFeatures
import os

def main():
    parser = argparse.ArgumentParser(description='Image Aesthetics features')
    parser.add_argument('img_csv', type=str, help='input your img csv path')
    parser.add_argument('csv_path', type=str, help='input your save path')

    args = parser.parse_args()

    img_csv = args.img_csv
    csv_path = args.csv_path

    feature_attribute = [
     'Exposure_of_light','saturation','hue','RoT1','RoT2','RoT3','size','aspect',
     'DOF1','DOF2','DOF3','pleasure','arousal','dominance','colofulness','black','silver',
     'gray','white','maroon','red','purple','fuchsia','green','lime','olive','yellow','navy',
     'blue','teal','aqua','coarseness','t_contrast','directionality','wavelet1','wavelet2',
     'wavelet3','wavelet4','wavelet5','wavelet6','wavelet7','wavelet8','wavelet9','wavelet10',
     'wavelet11','wavelet12','g_contrast','correlation','energy','homogeneity','len_statics',
     'degree_statics','abs_degree_statics','len_dynamics','degree_dynamics','abs_degree_dynamics'
    ]
    
    if not os.path.exists(csv_path):
        with open(csv_path, 'a') as f:
            wr = csv.writer(f)
            wr.writerow(feature_attribute)
    
    with open(img_csv, 'r', encoding='utf-8') as csv_file:
        rdr = csv.reader(csv_file)

        for img_path in rdr:
            img_path = img_path[0]
            print(img_path)
            img = cv2.imread(img_path)
    
            # Size and Aspect Ratio
            f22 = DattaFeatures.f22(img)
            f23 = DattaFeatures.f23(img)

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
            feature_vector.append(f22)
            feature_vector.append(f23)

            # DOF features
            DOF_features = DattaFeatures.f53_55(channels)
            for i in DOF_features:
                feature_vector.append(i) 

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

            # Dynamic features
            dynamics = MachajdikFeatures.dynamics(img)
            for i in dynamics:
                feature_vector.append(i)

            with open(csv_path, 'a') as f:
                wr = csv.writer(f)
                wr.writerow(feature_vector)

if __name__=="__main__":
    main()