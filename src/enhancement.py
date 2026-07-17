import cv2
import numpy as np
from PIL import Image


class ImageEnhancer:

    def __init__(self):
        pass

    def enhance(self, image):
        """
        Perform image enhancement using:
        - CLAHE
        - Noise Reduction
        - Sharpening
        """

        img = np.array(image)

        # Convert RGB -> LAB
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

        l, a, b = cv2.split(lab)

        # CLAHE
        clahe = cv2.createCLAHE(
            clipLimit=2.5,
            tileGridSize=(8, 8)
        )

        l = clahe.apply(l)

        lab = cv2.merge((l, a, b))

        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        # Noise Reduction
        enhanced = cv2.fastNlMeansDenoisingColored(
            enhanced,
            None,
            10,
            10,
            7,
            21
        )

        # Sharpen
        kernel = np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ])

        enhanced = cv2.filter2D(
            enhanced,
            -1,
            kernel
        )

        return Image.fromarray(enhanced)
