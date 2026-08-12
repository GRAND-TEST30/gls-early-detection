import cv2
import numpy as np

from skimage.feature import graycomatrix
from skimage.feature import graycoprops


class FeatureExtractor:

    def __init__(self):
        pass

    ####################################################
    # COLOUR FEATURES
    ####################################################

    def colour_features(self, image, leaf_mask=None):

        img = np.array(image.convert("RGB"))

        # If no mask is supplied, analyse the whole image.
        if leaf_mask is None:
            valid_mask = np.ones(
                img.shape[:2],
                dtype=bool
            )
        else:
            valid_mask = leaf_mask > 0

        # Prevent empty-mask errors
        if not np.any(valid_mask):
            return {
                "mean_red": 0.0,
                "mean_green": 0.0,
                "mean_blue": 0.0,
                "mean_hue": 0.0,
                "mean_saturation": 0.0,
                "mean_value": 0.0
            }

        red = img[:, :, 0][valid_mask]
        green = img[:, :, 1][valid_mask]
        blue = img[:, :, 2][valid_mask]

        hsv = cv2.cvtColor(
            img,
            cv2.COLOR_RGB2HSV
        )

        h = hsv[:, :, 0][valid_mask]
        s = hsv[:, :, 1][valid_mask]
        v = hsv[:, :, 2][valid_mask]

        return {

            "mean_red":
                float(np.mean(red)),

            "mean_green":
                float(np.mean(green)),

            "mean_blue":
                float(np.mean(blue)),

            "mean_hue":
                float(np.mean(h)),

            "mean_saturation":
                float(np.mean(s)),

            "mean_value":
                float(np.mean(v))

        }

    ####################################################
    # TEXTURE FEATURES
    ####################################################

    def texture_features(self, image, leaf_mask=None):

        img = np.array(
            image.convert("RGB")
        )

        gray = cv2.cvtColor(
            img,
            cv2.COLOR_RGB2GRAY
        )

        # Restrict analysis to the leaf
        if leaf_mask is not None:

            mask = leaf_mask > 0

            if not np.any(mask):
                return {
                    "contrast": 0.0,
                    "homogeneity": 0.0,
                    "energy": 0.0,
                    "correlation": 0.0,
                    "ASM": 0.0
                }

            # Replace background with the mean
            # intensity of the leaf.
            leaf_values = gray[mask]

            mean_value = int(
                np.mean(leaf_values)
            )

            gray_for_texture = gray.copy()

            gray_for_texture[~mask] = mean_value

        else:

            gray_for_texture = gray

        # GLCM works on uint8 image
        gray_for_texture = np.clip(
            gray_for_texture,
            0,
            255
        ).astype(np.uint8)

        glcm = graycomatrix(
            gray_for_texture,
            distances=[1],
            angles=[0],
            levels=256,
            symmetric=True,
            normed=True
        )

        return {

            "contrast":
                float(
                    graycoprops(
                        glcm,
                        "contrast"
                    )[0, 0]
                ),

            "homogeneity":
                float(
                    graycoprops(
                        glcm,
                        "homogeneity"
                    )[0, 0]
                ),

            "energy":
                float(
                    graycoprops(
                        glcm,
                        "energy"
                    )[0, 0]
                ),

            "correlation":
                float(
                    graycoprops(
                        glcm,
                        "correlation"
                    )[0, 0]
                ),

            "ASM":
                float(
                    graycoprops(
                        glcm,
                        "ASM"
                    )[0, 0]
                )

        }

    ####################################################
    # LESION FEATURES
    ####################################################

    def lesion_features(self, image, leaf_mask=None):

        img = np.array(
            image.convert("RGB")
        )

        gray = cv2.cvtColor(
            img,
            cv2.COLOR_RGB2GRAY
        )

        # ------------------------------------------------
        # CREATE LEAF MASK
        # ------------------------------------------------

        if leaf_mask is None:

            leaf_mask = np.ones(
                gray.shape,
                dtype=np.uint8
            ) * 255

        else:

            leaf_mask = np.where(
                leaf_mask > 0,
                255,
                0
            ).astype(np.uint8)

        leaf_area = int(
            np.sum(leaf_mask > 0)
        )

        if leaf_area == 0:

            return {
                "lesion_count": 0,
                "largest_lesion": 0.0,
                "total_lesion_area": 0.0,
                "lesion_ratio": 0.0,
                "disease_coverage": 0.0
            }

        # ------------------------------------------------
        # BLUR
        # ------------------------------------------------

        blur = cv2.GaussianBlur(
            gray,
            (5, 5),
            0
        )

        # ------------------------------------------------
        # OTSU DARK-REGION DETECTION
        # ------------------------------------------------

        _, dark_mask = cv2.threshold(
            blur,
            0,
            255,
            cv2.THRESH_BINARY_INV +
            cv2.THRESH_OTSU
        )

        # ------------------------------------------------
        # IMPORTANT:
        # ONLY DETECT LESIONS INSIDE THE LEAF
        # ------------------------------------------------

        lesion_mask = cv2.bitwise_and(
            dark_mask,
            leaf_mask
        )

        # ------------------------------------------------
        # REMOVE SMALL NOISE
        # ------------------------------------------------

        kernel = np.ones(
            (3, 3),
            np.uint8
        )

        lesion_mask = cv2.morphologyEx(
            lesion_mask,
            cv2.MORPH_OPEN,
            kernel
        )

        lesion_mask = cv2.morphologyEx(
            lesion_mask,
            cv2.MORPH_CLOSE,
            kernel
        )

        # ------------------------------------------------
        # FIND LESION CONTOURS
        # ------------------------------------------------

        contours, _ = cv2.findContours(
            lesion_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        lesion_count = 0
        total_lesion_area = 0.0
        largest_lesion = 0.0

        valid_areas = []

        for contour in contours:

            area = cv2.contourArea(
                contour
            )

            # Ignore tiny noise
            if area < 25:
                continue

            lesion_count += 1

            total_lesion_area += area

            largest_lesion = max(
                largest_lesion,
                area
            )

            valid_areas.append(area)

        # ------------------------------------------------
        # LESION RATIO
        # ------------------------------------------------

        lesion_ratio = (
            total_lesion_area /
            leaf_area
        ) * 100

        # ------------------------------------------------
        # SAFETY LIMIT
        # ------------------------------------------------

        lesion_ratio = max(
            0.0,
            min(
                lesion_ratio,
                100.0
            )
        )

        return {

            "lesion_count":
                int(lesion_count),

            "largest_lesion":
                round(
                    largest_lesion,
                    2
                ),

            "total_lesion_area":
                round(
                    total_lesion_area,
                    2
                ),

            "lesion_ratio":
                round(
                    lesion_ratio,
                    2
                ),

            "disease_coverage":
                round(
                    lesion_ratio,
                    2
                )

        }

    ####################################################
    # COMPLETE FEATURE SET
    ####################################################

    def extract_all(
        self,
        image,
        leaf_mask=None
    ):

        features = {}

        features["colour_features"] = (
            self.colour_features(
                image,
                leaf_mask
            )
        )

        features["texture_features"] = (
            self.texture_features(
                image,
                leaf_mask
            )
        )

        features["lesion_features"] = (
            self.lesion_features(
                image,
                leaf_mask
            )
        )

        return features
