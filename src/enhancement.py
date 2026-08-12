import cv2
import numpy as np
from PIL import Image


class LeafSegmenter:

    def __init__(self):
        pass

    # =========================================================
    # SEGMENT LEAF
    # =========================================================

    def segment_leaf(self, image):
        """
        Segment the maize leaf from the surrounding background.

        Returns:
            segmented_image: PIL Image
            mask: binary NumPy array
        """

        # Convert PIL image to RGB NumPy array
        img = np.array(image.convert("RGB"))

        # Convert RGB to HSV
        hsv = cv2.cvtColor(
            img,
            cv2.COLOR_RGB2HSV
        )

        # -----------------------------------------------------
        # GREEN MASK
        # -----------------------------------------------------

        lower_green = np.array(
            [20, 25, 20],
            dtype=np.uint8
        )

        upper_green = np.array(
            [95, 255, 255],
            dtype=np.uint8
        )

        green_mask = cv2.inRange(
            hsv,
            lower_green,
            upper_green
        )

        # -----------------------------------------------------
        # ADDITIONAL VEGETATION MASK
        # -----------------------------------------------------

        # Exclude extremely dark pixels
        brightness_mask = hsv[:, :, 2] > 25

        # Combine masks
        mask = green_mask.copy()

        mask[~brightness_mask] = 0

        # -----------------------------------------------------
        # MORPHOLOGICAL CLEANING
        # -----------------------------------------------------

        kernel = np.ones(
            (5, 5),
            np.uint8
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel
        )

        # -----------------------------------------------------
        # KEEP MAJOR LEAF COMPONENTS
        # -----------------------------------------------------

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask,
            connectivity=8
        )

        if num_labels > 1:

            # Ignore background label 0
            component_areas = stats[1:, cv2.CC_STAT_AREA]

            largest_index = np.argmax(
                component_areas
            ) + 1

            # Create clean mask
            clean_mask = np.zeros_like(mask)

            clean_mask[
                labels == largest_index
            ] = 255

            mask = clean_mask

        # -----------------------------------------------------
        # FINAL MORPHOLOGICAL CLEANING
        # -----------------------------------------------------

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel
        )

        # -----------------------------------------------------
        # CREATE SEGMENTED IMAGE
        # -----------------------------------------------------

        segmented = cv2.bitwise_and(
            img,
            img,
            mask=mask
        )

        segmented_image = Image.fromarray(
            segmented
        )

        return segmented_image, mask

    # =========================================================
    # LEAF AREA
    # =========================================================

    def calculate_leaf_area(self, mask):
        """
        Calculate number of pixels belonging to the leaf.
        """

        if mask is None:
            return 0

        leaf_pixels = np.sum(
            mask > 0
        )

        return int(leaf_pixels)

    # =========================================================
    # LEAF COVERAGE
    # =========================================================

    def calculate_leaf_coverage(self, mask):
        """
        Calculate percentage of the image occupied by the leaf.
        """

        if mask is None:
            return 0.0

        total_pixels = mask.size

        if total_pixels == 0:
            return 0.0

        leaf_pixels = np.sum(
            mask > 0
        )

        coverage = (
            leaf_pixels /
            total_pixels
        ) * 100

        return round(
            float(coverage),
            2
        )

    # =========================================================
    # APPLY MASK
    # =========================================================

    def apply_mask(self, image, mask):
        """
        Apply a leaf mask to an image.
        """

        img = np.array(
            image.convert("RGB")
        )

        masked = cv2.bitwise_and(
            img,
            img,
            mask=mask
        )

        return Image.fromarray(
            masked
        )
