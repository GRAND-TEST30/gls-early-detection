from PIL import Image, ImageEnhance
import numpy as np

class GLSEarlyDetector:
    def __init__(self):
        pass

    def load_image(self, image_input):
        if isinstance(image_input, str):
            image = Image.open(image_input)
        else:
            image = Image.open(image_input)
        return image.convert('RGB')

    def enhance_image(self, image):
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(1.8)

    def extract_features(self, image):
        img_array = np.array(image.convert('L'))
        threshold = 170
        lesions = np.sum(img_array < threshold)
        total_pixels = img_array.size
        lesion_ratio = lesions / total_pixels * 100
        return {
            "lesion_ratio": round(lesion_ratio, 2),
            "total_pixels": total_pixels
        }

    def predict_stage(self, features):
        ratio = features["lesion_ratio"]
        if ratio < 2:
            return "Healthy", 90, 85
        elif ratio < 8:
            return "Early Stage", 78, 48
        elif ratio < 18:
            return "Moderate Stage", 75, 25
        else:
            return "Severe Stage", 82, 10

    def full_analysis(self, image_input):
        image = self.load_image(image_input)
        enhanced = self.enhance_image(image)
        features = self.extract_features(enhanced)
        stage, confidence, days = self.predict_stage(features)
        
        return {
            "stage": stage,
            "confidence": confidence,
            "remaining_days": days,
            "recommendation": self.get_recommendation(stage)
        }

    def get_recommendation(self, stage):
        if stage == "Healthy":
            return "Plant is healthy. Good job!"
        elif stage == "Early Stage":
            return "Early Gray Leaf Spot detected. Apply fungicide soon."
        elif stage == "Moderate Stage":
            return "Moderate infection. Urgent treatment needed."
        else:
            return "Severe infection. Take immediate action."
