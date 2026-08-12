import cv2
import numpy as np
from PIL import Image


class ImageEnhancer:

    def __init__(self):
        """
        Initialize the image enhancement module.

        Enhancement pipeline:
        1. LAB colour-space conversion
        2. CLAHE contrast enhancement
        3. Non-local means noise reduction
        4. Edge-preserving sharpening
        """

        # CLAHE configuration
        self.clahe = cv2.createCLAHE(
            clipLimit=2.5,
            tileGridSize=(8, 8)
        )

        # Sharpening kernel
        self.sharpen_kernel = np.array(
            [
                [0, -1, 0],
                [-1, 5, -1],
                [0, -1, 0]
            ],
            dtype=np.float32
        )

    # =========================================================
    # IMAGE VALIDATION
    # =========================================================

    def _prepare_image(self, image):
        """
        Convert input image into a valid RGB uint8 NumPy array.
        """

        if isinstance(image, Image.Image):

            image = image.convert("RGB")

            img = np.array(image)

        else:

            img = np.asarray(image)

            if img.ndim == 2:

                img = cv2.cvtColor(
                    img,
                    cv2.COLOR_GRAY2RGB
                )

            elif img.ndim == 3 and img.shape[2] == 4:

                img = cv2.cvtColor(
                    img,
                    cv2.COLOR_RGBA2RGB
                )

        # Ensure uint8 representation
        if img.dtype != np.uint8:

            img = np.clip(
                img,
                0,
                255
            ).astype(np.uint8)

        return img

    # =========================================================
    # CLAHE ENHANCEMENT
    # =========================================================

    def _apply_clahe(self, img):
        """
        Enhance local contrast using CLAHE.

        CLAHE operates on the L channel of LAB,
        preserving the original colour information
        better than applying contrast directly to RGB.
        """

        lab = cv2.cvtColor(
            img,
            cv2.COLOR_RGB2LAB
        )

        l_channel, a_channel, b_channel = cv2.split(
            lab
        )

        enhanced_l = self.clahe.apply(
            l_channel
        )

        enhanced_lab = cv2.merge(
            (
                enhanced_l,
                a_channel,
                b_channel
            )
        )

        enhanced = cv2.cvtColor(
            enhanced_lab,
            cv2.COLOR_LAB2RGB
        )

        return enhanced

    # =========================================================
    # NOISE REDUCTION
    # =========================================================

    def _reduce_noise(self, img):
        """
        Reduce image noise using Non-Local Means denoising.

        This is useful for reducing small image artifacts
        before lesion detection.
        """

        denoised = cv2.fastNlMeansDenoisingColored(
            img,
            None,
            10,
            10,
            7,
            21
        )

        return denoised

    # =========================================================
    # SHARPENING
    # =========================================================

    def _sharpen(self, img):
        """
        Enhance edges and fine structures that may
        correspond to disease lesions.
        """

        sharpened = cv2.filter2D(
            img,
            -1,
            self.sharpen_kernel
        )

        return sharpened

    # =========================================================
    # COMPLETE ENHANCEMENT PIPELINE
    # =========================================================

    def enhance(self, image):
        """
        Perform complete image enhancement.

        Pipeline:

            Input
              ↓
            RGB validation
              ↓
            LAB conversion
              ↓
            CLAHE
              ↓
            Noise reduction
              ↓
            Sharpening
              ↓
            Enhanced PIL Image
        """

        # -----------------------------------------------------
        # Prepare image
        # -----------------------------------------------------

        img = self._prepare_image(
            image
        )

        # -----------------------------------------------------
        # CLAHE
        # -----------------------------------------------------

        enhanced = self._apply_clahe(
            img
        )

        # -----------------------------------------------------
        # Noise reduction
        # -----------------------------------------------------

        enhanced = self._reduce_noise(
            enhanced
        )

        # -----------------------------------------------------
        # Sharpening
        # -----------------------------------------------------

        enhanced = self._sharpen(
            enhanced
        )

        # -----------------------------------------------------
        # Convert back to PIL
        # -----------------------------------------------------

        return Image.fromarray(
            enhanced
        )

    # =========================================================
    # OPTIONAL NUMPY OUTPUT
    # =========================================================

    def enhance_array(self, image):
        """
        Return the enhanced image as a NumPy array.

        Useful for OpenCV-based downstream processing.
        """

        enhanced = self.enhance(
            image
        )

        return np.array(
            enhanced
        )
