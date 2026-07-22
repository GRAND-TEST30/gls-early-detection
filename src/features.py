import cv
import numpy as np

from skimage.feature import graycomatrix
from skimage.feature import graycoprops


class FeatureExtractor:

    def __init__(self):
        pass

    ####################################################
    # COLOUR FEATURES
    ####################################################

    def colour_features(self, image):

        img = np.array(image)

        red = img[:, :, 0]
        green = img[:, :, 1]
        blue = img[:, :, 2]

        hsv = cv2.cvtColor(
            img,
            cv2.COLOR_RGB2HSV
        )

        h = hsv[:, :, 0]
        s = hsv[:, :, 1]
        v = hsv[:, :, 2]

        features = {

            "mean_red": float(np.mean(red)),

            "mean_green": float(np.mean(green)),

            "mean_blue": float(np.mean(blue)),

            "mean_hue": float(np.mean(h)),

            "mean_saturation": float(np.mean(s)),

            "mean_value": float(np.mean(v))

        }

        return features

    ####################################################
    # TEXTURE FEATURES
    ####################################################

    def texture_features(self, image):

        gray = cv2.cvtColor(
            np.array(image),
            cv2.COLOR_RGB2GRAY
        )

        glcm = graycomatrix(
            gray,
            distances=[1],
            angles=[0],
            levels=256,
            symmetric=True,
            normed=True
        )

        features = {

            "contrast":
                float(graycoprops(glcm, 'contrast')[0, 0]),

            "homogeneity":
                float(graycoprops(glcm, 'homogeneity')[0, 0]),

            "energy":
                float(graycoprops(glcm, 'energy')[0, 0]),

            "correlation":
                float(graycoprops(glcm, 'correlation')[0, 0]),

            "ASM":
                float(graycoprops(glcm, 'ASM')[0, 0])

        }

        return features

    ####################################################
    # LESION FEATURES
    ####################################################

    def lesion_features(self, image):

        gray = cv2.cvtColor(
            np.array(image),
            cv2.COLOR_RGB2GRAY
        )

        blur = cv2.GaussianBlur(
            gray,
            (5, 5),
            0
        )

        _, thresh = cv2.threshold(
            blur,
            0,
            255,
            cv2.THRESH_BINARY_INV +
            cv2.THRESH_OTSU
        )

        contours, _ = cv2.findContours(
            thresh,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        lesion_count = 0

        lesion_area = 0

        largest = 0

        for cnt in contours:

            area = cv2.contourArea(cnt)

            if area < 25:
                continue

            lesion_count += 1

            lesion_area += area

            largest = max(
                largest,
                area
            )

        return {

            "lesion_count": lesion_count,

            "largest_lesion": round(
                largest,
                2
            ),

            "total_lesion_area": round(
                lesion_area,
                2
            )

        }

    ####################################################
    # COMPLETE FEATURE SET
    ####################################################

    def extract_all(self, image):

        features = {}

        features.update(
            self.colour_features(image)
        )

        features.update(
            self.texture_features(image)
        )

        features.update(
            self.lesion_features(image)
        )

        return features
