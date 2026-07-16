"""
Advanced Preprocessing and Analysis Module for Early Gray Leaf Spot Detection
This module performs per-image analysis without any pre-trained dataset.
It uses classical computer vision techniques for lesion detection and severity estimation.
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GLSEarlyDetector:
    def __init__(self):
        self.img_size = (224, 224)
        logger.info("GLSEarlyDetector initialized - Per-image analysis mode activated")

    def load_and_validate_image(self, image_input):
        """Load image and perform basic validation"""
        logger.info("Loading and validating image")
        try:
            if isinstance(image_input, str):
                image = Image.open(image_input)
            else:
                image = Image.open(image_input)
            image = image.convert('RGB')
            if image.size[0] < 150 or image.size[1] < 150:
                logger.warning("Low resolution image detected")
            logger.info(f"Image loaded successfully: {image.size}")
            return image
        except Exception as e:
            logger.error(f"Failed to load image: {e}")
            raise

    def multi_stage_enhancement(self, image):
        """Detailed multi-stage enhancement for better early lesion visibility"""
        logger.info("Starting multi-stage image enhancement")
        img = np.array(image)
        
        # Stage 1: LAB + CLAHE for contrast
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.8, tileGridSize=(8,8))
        cl = clahe.apply(l)
        enhanced = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2RGB)
        
        # Stage 2: Sharpening
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        
        # Stage 3: Mild denoising
        denoised = cv2.fastNlMeansDenoisingColored(sharpened, None, 10, 10, 7, 21)
        
        # Stage 4: Contrast and brightness adjustment
        pil_img = Image.fromarray(denoised)
        enhancer_contrast = ImageEnhance.Contrast(pil_img)
        final = enhancer_contrast.enhance(1.75)
        
        logger.info("Image enhancement completed")
        return final

    def extract_lesion_features(self, image):
        """Extract detailed lesion features"""
        img = np.array(image)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Multiple thresholds
        _, thresh = cv2.threshold(gray, 165, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        features = {
            "total_lesions": len(contours),
            "small_lesions": len([c for c in contours if cv2.contourArea(c) < 500]),
            "medium_lesions": len([c for c in contours if 500 <= cv2.contourArea(c) < 1500]),
            "large_lesions": len([c for c in contours if cv2.contourArea(c) >= 1500]),
            "total_area": int(sum(cv2.contourArea(c) for c in contours)),
            "avg_lesion_size": round(sum(cv2.contourArea(c) for c in contours) / len(contours), 2) if contours else 0,
            "lesion_density": round(len(contours) / (img.shape[0] * img.shape[1]) * 10000, 4)
        }
        return features

    def predict_stage_and_lifespan(self, features):
        """Rule-based prediction with detailed scoring"""
        score = (features["small_lesions"] * 1.5) + (features["medium_lesions"] * 4) + (features["large_lesions"] * 8) + (features["total_area"] / 700)
        
        if score < 15:
            stage = "Healthy"
            confidence = 93
            days = 88
        elif score < 40:
            stage = "Early Stage"
            confidence = 82
            days = 50
        elif score < 75:
            stage = "Moderate Stage"
            confidence = 78
            days = 24
        else:
            stage = "Severe Stage"
            confidence = 85
            days = 9
        
        return stage, confidence, days

    def get_recommendation(self, stage):
        """Detailed recommendation based on stage"""
        if stage == "Healthy":
            return "The plant appears healthy. Continue good agricultural practices and regular monitoring."
        elif stage == "Early Stage":
            return "Early Gray Leaf Spot detected. Apply recommended fungicide immediately and remove affected lower leaves."
        elif stage == "Moderate Stage":
            return "Moderate infection. Urgent fungicide application and improved air circulation recommended."
        else:
            return "Severe infection. High risk of significant yield loss. Consider isolating or removing heavily affected plants."

    def full_analysis(self, image_input):
        """Complete analysis pipeline"""
        logger.info("Starting full analysis pipeline")
        image = self.load_and_validate_image(image_input)
        enhanced = self.multi_stage_enhancement(image)
        features = self.extract_lesion_features(enhanced)
        stage, confidence, days = self.predict_stage_and_lifespan(features)
        recommendation = self.get_recommendation(stage)
        
        result = {
            "stage": stage,
            "confidence": confidence,
            "remaining_days": days,
            "features": features,
            "recommendation": recommendation
        }
        logger.info(f"Analysis completed: {stage}")
        return result
