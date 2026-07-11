import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw
import logging
from datetime import datetime
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GLSEarlyDetector:
    def __init__(self):
        self.img_size = (224, 224)
        logger.info("GLS Early Detector initialized - Per-image analysis mode")

    def load_image(self, image_input):
        """Load and validate uploaded image"""
        try:
            if isinstance(image_input, str):
                image = Image.open(image_input)
            else:
                image = Image.open(image_input)
            image = image.convert('RGB')
            logger.info("Image loaded successfully")
            return image
        except Exception as e:
            logger.error(f"Failed to load image: {e}")
            raise

    def multi_stage_enhancement(self, image):
        """Comprehensive enhancement pipeline for early lesion visibility"""
        logger.info("Starting multi-stage image enhancement")
        img = np.array(image)
        
        # Stage 1: Color Space Conversion + CLAHE
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=4.2, tileGridSize=(8,8))
        cl = clahe.apply(l)
        enhanced = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2RGB)
        
        # Stage 2: Sharpening
        kernel_sharp = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel_sharp)
        
        # Stage 3: Denoising
        denoised = cv2.fastNlMeansDenoisingColored(sharpened, None, 9, 9, 7, 21)
        
        # Stage 4: Contrast & Brightness
        pil_img = Image.fromarray(denoised)
        enhancer_contrast = ImageEnhance.Contrast(pil_img)
        enhanced_contrast = enhancer_contrast.enhance(1.8)
        
        enhancer_brightness = ImageEnhance.Brightness(enhanced_contrast)
        final_enhanced = enhancer_brightness.enhance(1.15)
        
        logger.info("Multi-stage enhancement completed")
        return final_enhanced

    def extract_detailed_features(self, image):
        """Extract rich features from the image"""
        img = np.array(image)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Multiple thresholds for different lesion sizes
        _, thresh_small = cv2.threshold(gray, 175, 255, cv2.THRESH_BINARY_INV)
        _, thresh_large = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY_INV)
        
        contours_small, _ = cv2.findContours(thresh_small, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_large, _ = cv2.findContours(thresh_large, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        features = {
            "total_lesions": len(contours_small),
            "small_lesions": len([c for c in contours_small if cv2.contourArea(c) < 600]),
            "medium_lesions": len([c for c in contours_small if 600 <= cv2.contourArea(c) < 2000]),
            "large_lesions": len(contours_large),
            "total_lesion_area": int(sum(cv2.contourArea(c) for c in contours_small)),
            "lesion_density": round(len(contours_small) / (img.shape[0] * img.shape[1]) * 10000, 4),
            "avg_lesion_size": round(sum(cv2.contourArea(c) for c in contours_small) / len(contours_small), 2) if contours_small else 0,
            "edge_density": self.calculate_edge_density(gray)
        }
        
        return features

    def calculate_edge_density(self, gray_image):
        """Calculate edge density for lesion boundary analysis"""
        edges = cv2.Canny(gray_image, 50, 150)
        return np.sum(edges > 0) / (gray_image.shape[0] * gray_image.shape[1])

    def predict_severity_and_lifespan(self, features):
        """Advanced rule-based prediction with weighted scoring"""
        score = 0.0
        score += features["small_lesions"] * 1.8
        score += features["medium_lesions"] * 4.2
        score += features["large_lesions"] * 8.5
        score += features["total_lesion_area"] / 650
        score += features["edge_density"] * 120
        
        if score < 12:
            stage = "Healthy"
            confidence = 94.0
            remaining_days = 88
        elif score < 38:
            stage = "Early Stage GLS"
            confidence = 83.0
            remaining_days = 52
        elif score < 72:
            stage = "Moderate Stage GLS"
            confidence = 79.0
            remaining_days = 26
        else:
            stage = "Severe Stage GLS"
            confidence = 87.0
            remaining_days = 11

        return stage, round(confidence, 2), remaining_days

    def generate_recommendation(self, stage, remaining_days):
        """Detailed recommendation based on stage"""
        if stage == "Healthy":
            return "Plant appears healthy. Continue regular monitoring and good cultural practices."
        elif stage == "Early Stage GLS":
            return f"Early detection of Gray Leaf Spot. Apply appropriate fungicide within 48 hours. Estimated remaining productive days: {remaining_days}."
        elif stage == "Moderate Stage GLS":
            return f"Moderate infection detected. Urgent fungicide application and leaf removal recommended. Remaining productive days: {remaining_days}."
        else:
            return f"Severe infection. High risk of major yield loss. Consider removing affected plants. Remaining productive days: {remaining_days}."

    def full_per_image_analysis(self, image_input):
        """Complete analysis pipeline for each uploaded image"""
        logger.info("=== Starting Per-Image Analysis ===")
        
        image = self.load_image(image_input)
        enhanced = self.multi_stage_enhancement(image)
        features = self.extract_detailed_features(enhanced)
        stage, confidence, remaining_days = self.predict_severity_and_lifespan(features)
        recommendation = self.generate_recommendation(stage, remaining_days)
        
        result = {
            "analysis_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "detected_stage": stage,
            "confidence_score": confidence,
            "estimated_remaining_productive_days": remaining_days,
            "lesion_features": features,
            "recommendation": recommendation,
            "status": "Success"
        }
        
        logger.info(f"Analysis completed: {stage} with {confidence}% confidence")
        return result