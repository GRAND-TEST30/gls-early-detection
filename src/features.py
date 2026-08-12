import cv2
import numpy as np

from skimage.feature import graycomatrix
from skimage.feature import graycoprops


class FeatureExtractor:

    def __init__(self):
        pass

    # =========================================================
    # PREPARE MASK
    # =========================================================

    def _prepare_mask(self, image, mask=None):
        """
        Prepare a valid binary mask.

        If no mask is provided, use non-black pixels.
        """

        img = np.array(
            image.convert("RGB")
        )

        if mask is not None:

            valid_mask = mask > 0

        else:

            valid_mask = np.any(
                img > 0,
                axis=2
            )

        return valid_mask

    # =========================================================
    # COLOUR FEATURES
    # =========================================================

    def colour_features(
        self,
        image,
        mask=None
    ):
        """
        Extract colour characteristics only from
        pixels belonging to the leaf.
        """

        img = np.array(
            image.convert("RGB")
        )

        valid_mask = self._prepare_mask(
            image,
            mask
        )

        if not np.any(valid_mask):

            return {
                "mean_red": 0.0,
                "mean_green": 0.0,
                "mean_blue": 0.0,
                "mean_hue": 0.0,
                "mean_saturation": 0.0,
                "mean_value": 0.0,
                "green_red_ratio": 0.0,
                "green_percentage": 0.0
            }

        # RGB channels
        red = img[:, :, 0][valid_mask]
        green = img[:, :, 1][valid_mask]
        blue = img[:, :, 2][valid_mask]

        # HSV
        hsv = cv2.cvtColor(
            img,
            cv2.COLOR_RGB2HSV
        )

        h = hsv[:, :, 0][valid_mask]
        s = hsv[:, :, 1][valid_mask]
        v = hsv[:, :, 2][valid_mask]

        # Green dominance
        green_red_ratio = (
            np.mean(green) /
            (np.mean(red) + 1e-6)
        )

        # Percentage of pixels where green dominates
        green_pixels = (
            (green > red) &
            (green > blue)
        )

        green_percentage = (
            np.sum(green_pixels) /
            len(green_pixels)
        ) * 100

        return {

            "mean_red": round(
                float(np.mean(red)),
                2
            ),

            "mean_green": round(
                float(np.mean(green)),
                2
            ),

            "mean_blue": round(
                float(np.mean(blue)),
                2
            ),

            "mean_hue": round(
                float(np.mean(h)),
                2
            ),

            "mean_saturation": round(
                float(np.mean(s)),
                2
            ),

            "mean_value": round(
                float(np.mean(v)),
                2
            ),

            "green_red_ratio": round(
                float(green_red_ratio),
                3
            ),

            "green_percentage": round(
                float(green_percentage),
                2
            )
        }

    # =========================================================
    # TEXTURE FEATURES
    # =========================================================

    def texture_features(
        self,
        image,
        mask=None
    ):
        """
        Extract GLCM texture features from the leaf region.
        """

        img = np.array(
            image.convert("RGB")
        )

        gray = cv2.cvtColor(
            img,
            cv2.COLOR_RGB2GRAY
        )

        valid_mask = self._prepare_mask(
            image,
            mask
        )

        if not np.any(valid_mask):

            return {
                "contrast": 0.0,
                "homogeneity": 0.0,
                "energy": 0.0,
                "correlation": 0.0,
                "ASM": 0.0
            }

        # -----------------------------------------------------
        # Replace background with mean leaf intensity
        # -----------------------------------------------------

        leaf_values = gray[valid_mask]

        mean_gray = int(
            np.mean(leaf_values)
        )

        gray_masked = gray.copy()

        gray_masked[
            ~valid_mask
        ] = mean_gray

        # -----------------------------------------------------
        # Reduce image size for GLCM efficiency
        # -----------------------------------------------------

        max_dimension = 512

        height, width = gray_masked.shape

        if max(height, width) > max_dimension:

            scale = (
                max_dimension /
                max(height, width)
            )

            new_width = max(
                1,
                int(width * scale)
            )

            new_height = max(
                1,
                int(height * scale)
            )

            gray_masked = cv2.resize(
                gray_masked,
                (new_width, new_height),
                interpolation=cv2.INTER_AREA
            )

        # -----------------------------------------------------
        # Quantize grayscale image
        # -----------------------------------------------------

        levels = 64

        gray_quantized = (
            gray_masked.astype(
                np.float32
            ) *
            (levels - 1) /
            255
        ).astype(
            np.uint8
        )

        # -----------------------------------------------------
        # GLCM
        # -----------------------------------------------------

        glcm = graycomatrix(
            gray_quantized,
            distances=[1],
            angles=[
                0,
                np.pi / 4,
                np.pi / 2,
                3 * np.pi / 4
            ],
            levels=levels,
            symmetric=True,
            normed=True
        )

        contrast = np.mean(
            graycoprops(
                glcm,
                "contrast"
            )
        )

        homogeneity = np.mean(
            graycoprops(
                glcm,
                "homogeneity"
            )
        )

        energy = np.mean(
            graycoprops(
                glcm,
                "energy"
            )
        )

        correlation = np.mean(
            graycoprops(
                glcm,
                "correlation"
            )
        )

        asm = np.mean(
            graycoprops(
                glcm,
                "ASM"
            )
        )

        return {

            "contrast": round(
                float(contrast),
                4
            ),

            "homogeneity": round(
                float(homogeneity),
                4
            ),

            "energy": round(
                float(energy),
                4
            ),

            "correlation": round(
                float(correlation),
                4
            ),

            "ASM": round(
                float(asm),
                4
            )
        }

    # =========================================================
    # LESION DETECTION
    # =========================================================

    def lesion_features(
        self,
        image,
        mask=None
    ):
        """
        Detect dark/brown lesion-like regions inside
        the segmented maize leaf.

        IMPORTANT:
        Background pixels are excluded using the leaf mask.
        """

        img = np.array(
            image.convert("RGB")
        )

        valid_mask = self._prepare_mask(
            image,
            mask
        )

        if not np.any(valid_mask):

            return {
                "lesion_count": 0,
                "largest_lesion": 0.0,
                "total_lesion_area": 0.0,
                "lesion_ratio": 0.0,
                "disease_coverage": 0.0
            }

        # -----------------------------------------------------
        # HSV representation
        # -----------------------------------------------------

        hsv = cv2.cvtColor(
            img,
            cv2.COLOR_RGB2HSV
        )

        h = hsv[:, :, 0]
        s = hsv[:, :, 1]
        v = hsv[:, :, 2]

        # -----------------------------------------------------
        # Detect lesion-like pixels
        # -----------------------------------------------------
        #
        # Gray Leaf Spot lesions often appear darker,
        # brownish or grayish compared with healthy green
        # tissue.
        #
        # This is a rule-based detector, not a trained model.
        # -----------------------------------------------------

        lesion_mask = (

            valid_mask &

            (
                (
                    (v < 150) &
                    (s > 25)
                )

                |

                (
                    (v < 125) &
                    (s < 100)
                )

                |

                (
                    (h >= 5) &
                    (h <= 30) &
                    (s > 35) &
                    (v < 180)
                )
            )
        )

        lesion_binary = (
            lesion_mask.astype(
                np.uint8
            ) * 255
        )

        # -----------------------------------------------------
        # Remove tiny noise
        # -----------------------------------------------------

        kernel = np.ones(
            (3, 3),
            np.uint8
        )

        lesion_binary = cv2.morphologyEx(
            lesion_binary,
            cv2.MORPH_OPEN,
            kernel
        )

        lesion_binary = cv2.morphologyEx(
            lesion_binary,
            cv2.MORPH_CLOSE,
            kernel
        )

        # -----------------------------------------------------
        # Find lesions
        # -----------------------------------------------------

        contours, _ = cv2.findContours(
            lesion_binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        lesion_count = 0
        total_lesion_area = 0.0
        largest_lesion = 0.0

        # Minimum lesion size
        minimum_area = 25

        for contour in contours:

            area = cv2.contourArea(
                contour
            )

            if area < minimum_area:
                continue

            lesion_count += 1

            total_lesion_area += area

            largest_lesion = max(
                largest_lesion,
                area
            )

        # -----------------------------------------------------
        # Calculate lesion ratio
        # -----------------------------------------------------

        leaf_area = np.sum(
            valid_mask
        )

        if leaf_area > 0:

            lesion_ratio = (
                total_lesion_area /
                leaf_area
            ) * 100

        else:

            lesion_ratio = 0.0

        # Safety boundary
        lesion_ratio = max(
            0.0,
            min(
                100.0,
                lesion_ratio
            )
        )

        return {

            "lesion_count": int(
                lesion_count
            ),

            "largest_lesion": round(
                float(largest_lesion),
                2
            ),

            "total_lesion_area": round(
                float(total_lesion_area),
                2
            ),

            "lesion_ratio": round(
                float(lesion_ratio),
                2
            ),

            "disease_coverage": round(
                float(lesion_ratio),
                2
            )
        }

    # =========================================================
    # COMPLETE FEATURE EXTRACTION
    # =========================================================

    def extract_all(
        self,
        image,
        mask=None
    ):
        """
        Extract colour, texture and lesion features.
        """

        features = {}

        features["colour_features"] = (
            self.colour_features(
                image,
                mask
            )
        )

        features["texture_features"] = (
            self.texture_features(
                image,
                mask
            )
        )

        features["lesion_features"] = (
            self.lesion_features(
                image,
                mask
            )
        )

        return features
