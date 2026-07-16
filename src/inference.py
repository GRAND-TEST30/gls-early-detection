from src.preprocess import GLSEarlyDetector
from PIL import Image
from datetime import datetime

class GLSInferenceEngine:
    def __init__(self):
        self.detector = GLSEarlyDetector()

    def run_full_analysis(self, image_input):
        """Simple version - no validation method"""
        try:
            # Handle both file and PIL Image
            if isinstance(image_input, Image.Image):
                image = image_input
            else:
                image = Image.open(image_input)
            
            result = self.detector.full_analysis(image)
            
            result["processing_time_seconds"] = "N/A"
            result["image_dimensions"] = image.size
            
            return result
        except Exception as e:
            return {
                "status": "Failed",
                "error_message": str(e)
            }
