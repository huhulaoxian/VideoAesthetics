import cv2
from skimage.feature import greycomatrix, greycoprops
import pywt
import tamura_tf
import tensorflow as tf
import numpy as np
import argparse


# emotional impact(pleasure arousal and dominance)


# Hue : Hue is the color portion of the color model, expressed as a number from 0 to 360 degrees:

# Saturation : Saturation is the amount of gray in the color, from 0 to 100 percent. Reducing the saturation
# toward zero to introduce more gray produces a faded effect. Sometimes, saturation is expressed in a range from just 0-1, where 0 is gray and 1 is a primary color.

# Value (Brightness) : Value works in conjunction with saturation and describes the brightness or intensity of the color, from 0-100 percent, where 0 is completely black,
# and 100 is the brightest and reveals the most color.

# calculate the mean and the standard deviation for both the brightness and the saturation
class Color:
    # Valdez, P., & Mehrabian, A. (1994)
    # emotional impact(pleasure arousal and dominance)
    def f1_3(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        h, s, v = cv2.split(hsv)

        # mean & standard deviation
        mean_s, std_s = np.mean(h), np.std(h)
        mean_v, std_v = np.mean(v), np.std(v)

        # Pleasure = 0.69y + 0.22s
        # Arousal : -0.31y + 0.60s
        # Dominance : -0.76y + 0.32s
        # y : the average brightness of an image
        # s : its average saturation

        Pleasure = (0.69 * mean_v) + (0.22 * mean_s)
        Arousal = (-0.31 * mean_v) + (0.60 * mean_s)
        Dominance = (-0.76 * mean_v) + (0.32 * mean_s)

        return Pleasure, Arousal, Dominance

    # Hasler, D., & Suesstrunk, S. E. (2003, June)
    def f4(self, img):
        # split the image into its respective RGB components
        (B, G, R) = cv2.split(img.astype("float"))

        # compute rg = R - G
        rg = np.absolute(R - G)

        # compute yb = 0.5 * (R + G) - B
        yb = np.absolute(0.5 * (R + G) - B)

        # compute the mean and standard deviation of both `rg` and `yb`
        (rbMean, rbStd) = (np.mean(rg), np.std(rg))
        (ybMean, ybStd) = (np.mean(yb), np.std(yb))

        # combine the mean and standard deviations
        stdRoot = np.sqrt((rbStd ** 2) + (ybStd ** 2))
        meanRoot = np.sqrt((rbMean ** 2) + (ybMean ** 2))

        # derive the "colorfulness" metric and return it
        Colorfulness = stdRoot + (0.3 * meanRoot)

        return Colorfulness

    # Heckbert, P. (1982)
    #     def f5-9(self,img):
    #         NUM_CLUSTERS = 5
    #         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #         img = cv2.resize(img,(150,150))     # optional, to reduce time
    #         ar = np.asarray(img)
    #         shape = ar.shape
    #         ar = ar.reshape(scipy.product(shape[:2]), shape[2]).astype(float)

    #         codes, dist = scipy.cluster.vq.kmeans(ar, NUM_CLUSTERS)

    #         vecs, dist = scipy.cluster.vq.vq(ar, codes)         # assign codes
    #         counts, bins = scipy.histogram(vecs, len(codes))    # count occurrences

    #         value_list = []
    #         for i in range(NUM_CLUSTERS):
    #             peak = codes[i]
    #             colour = binascii.hexlify(bytearray(int(c) for c in peak)).decode('ascii')

    #             # uniform color quantification
    #             quantized_color = np.round(peak*(3/255))*(255/3)
    #             value_list.append(quantized_color)

    #         return np.array(value_list)

    # Machajdik, J., & Hanbury, A. (2010, October).
    def f10_25(self, img):
        # Color Distribution W3C
        # White, Silver, Gray, Black, Marron, Yellow, Olive, Lime, Green, Aqua, Teal, Blue, Navy, Fuchsia, Purple
        colors = {
            'black': np.uint8([[[0, 0, 0]]]),
            'silver': np.uint8([[[192, 192, 192]]]),
            'gray': np.uint8([[[182, 182, 182]]]),
            'white': np.uint8([[[255, 255, 255]]]),
            'maroon': np.uint8([[[0, 0, 128]]]),
            'red': np.uint8([[[0, 0, 255]]]),
            'purple': np.uint8([[[128, 0, 128]]]),
            'fuchsia': np.uint8([[[255, 0, 255]]]),
            'green': np.uint8([[[0, 128, 0]]]),
            'lime': np.uint8([[[0, 255, 0]]]),
            'olive': np.uint8([[[0, 128, 128]]]),
            'yellow': np.uint8([[[0, 255, 255]]]),
            'navy': np.uint8([[[128, 0, 0]]]),
            'blue': np.uint8([[[255, 0, 0]]]),
            'teal': np.uint8([[[128, 128, 0]]]),
            'aqua': np.uint8([[[255, 255, 0]]])
        }
        color_list = []

        for color in colors.keys():
            # sensitivity is a int, typically set to 15 - 20
            sensitivity = 15
            color_hsv = cv2.cvtColor(colors[color], cv2.COLOR_BGR2HSV)

            # HSV 이미지에서 특정 값만 추출하기 위한 임계값
            lower = np.array([color_hsv[0][0][0] - sensitivity, 0, color_hsv[0][0][2] - 127])
            upper = np.array([color_hsv[0][0][0] + sensitivity, 255, color_hsv[0][0][2] + 127])
            mask = cv2.inRange(img, lower, upper)

            # bitwise_and 연산자는 둘다 0이 아닌 경우만 값을 통과 시킴.
            output = cv2.bitwise_and(img, img, mask=mask)

            ratio_color = cv2.countNonZero(mask) / (img.size / 3)
            color_percentage = np.round(ratio_color * 100, 2)
            # print('{0} pixel percentage: {1}'.format(color,color_percentage))
            color_list.append(color_percentage)
        return color_list


class Texture:
    # Tamura, H., Mori, S., & Yamawaki, T. (1978).
    def f26_28(self, img_path):
        img = np.array(cv2.imread(img_path))[np.newaxis,]
        with tf.Session() as sess:
            style_image = tf.placeholder(tf.float32, shape=img.shape, name='style_image')

            frcs = tamura_tf.coarseness(style_image)
            fcon = tamura_tf.contrast(style_image)
            fdir = tamura_tf.directionality(style_image)
            sess.run(tf.global_variables_initializer())

            coarseness = sess.run(frcs, feed_dict={style_image: img})
            contrast = sess.run(fcon, feed_dict={style_image: img})
            directionality = sess.run(fdir, feed_dict={style_image: img})

            Tamura_features = [coarseness, contrast, directionality]
        return Tamura_features

    def f29_40(self, img):
        epsilon = 0.0001
        feature_values = []

        HSV_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        IH, IS, IV = cv2.split(HSV_img)
        channels = [IH, IS, IV]

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
                feature_values.append(numerator+epsilon / denominator+epsilon)

        # Sum of the average wavelet coefficients over all three frequency levels of H,S,V property
        feature_values.append(np.sum(feature_values[0:3]))
        feature_values.append(np.sum(feature_values[3:6]))
        feature_values.append(np.sum(feature_values[6:9]))

        return feature_values

        # Machajdik, J., & Hanbury, A. (2010, October).

    # Haralick, R. M., & Shapiro, L. G. (1992).
    def f41_44(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        channels = [h, s, v]

        for channel in channels:
            glcm = greycomatrix(channel, [5], [0], 256, symmetric=True, normed=True)

            contrast = greycoprops(glcm, 'contrast')[0, 0]
            correlation = greycoprops(glcm, 'correlation')[0, 0]
            energy = greycoprops(glcm, 'energy')[0, 0]
            homogeneity = greycoprops(glcm, 'homogeneity')[0, 0]

            GLCM_features = [contrast, correlation, energy, homogeneity]

        return GLCM_features


def main():
    parser = argparse.ArgumentParser(description='Wu features')
    parser.add_argument('img_path', type=str, help='input your img path')
    parser.add_argument('csv_path', type=str, help='input your save path')

    args = parser.parse_args()

    img_path = args.img_path
    csv_path = args.csv_path

    img = cv2.imread(img_path)

    feature_vec = []

    # color
    color = Color()
    f1_3 = color.f1_3(img)
    for i in f1_3:
        feature_vec.append(i)

    # Tamura features
    f10_25 = color.f10_25(img)
    for i in f10_25:
        feature_vec.append(i)

    # texture
    texture = Texture()

    # Tamura features
    f26_28 = texture.f26_28(img_path)
    for i in f26_28:
        feature_vec.append(i)

    # Wavelet features
    f29_40 = texture.f29_40(img)
    for i in f29_40:
        feature_vec.append(i)

        # GLCM features
    f41_44 = texture.f41_44(img)
    for i in f41_44:
        feature_vec.append(i)

    return feature_vec

    with open(csv_path, 'a') as f:
        wr = csv.writer(f)
        wr.writerow(feature_vec)
    print(len(feature_vec))
if __name__ == "__main__":
    main()