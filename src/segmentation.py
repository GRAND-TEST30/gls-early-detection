import cv2
import numpy as np
from PIL import Image


class LeafSegmenter:

    def __init__(self):
        pass

    def segment_leaf(self, image):
        """
        Segment maize leaf from background.
        """

        img = np.array(image)

        hsv = cv2.cvtColor(
            img,
            cv2.COLOR_RGB2HSV
        )

        lower_green = np.array([20, 25, 20])

        upper_green = np.array([95, 255, 255])

        mask = cv2.inRange(
            hsv,
            lower_green,
            upper_green
        )

        kernel = np.ones((5, 5), np.uint8)

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

        segmented = cv2.bitwise_and(
            img,
            img,
            mask=mask
        )

        return Image.fromarray(segmented), mask

    def calculate_leaf_area(self, mask):

        leaf_pixels = np.sum(mask > 0)

        return int(leaf_pixels)
