import logging
from PIL import Image
from datetime import datetime
from src.preprocess import GLSEarlyDetector

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GLSInferenceEngine:
    def __init__(self):
        self.detector = GLSEarlyDetector()
        logger.info("GLS Inference Engine initialized - Ready for per-image analysis")

    def validate_image(self, image_input):
        """Validate and return PIL Image"""
        try:
            if isinstance(image_input, Image.Image):
                image = image_input
            else:
                image = Image.open(image_input)
            
            if image.size[0] < 100 or image.size[1] < 100:
                logger.warning("Image resolution is too low")
                raise ValueError("Image resolution too low for reliable analysis")
            
            logger.info(f"Image validated: {image.size[0]}x{image.size[1]} pixels")
            return image
        except Exception as e:
            logger.error(f"Image validation failed: {e}")
            raise ValueError(f"Invalid image: {str(e)}")

    def run_full_analysis(self, image_input):
        """Run complete analysis pipeline"""
        start_time = datetime.now()
        logger.info("=== Starting Full Inference Pipeline ===")
        
        try:
            image = self.validate_image(image_input)
            result = self.detector.full_analysis(image)
            
            result["processing_time_seconds"] = round((datetime.now() - start_time).total_seconds(), 2)
            result["image_dimensions"] = image.size
            
            logger.info(f"Inference completed successfully")
            return result
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return {
                "status": "Failed",
                "error_message": str(e),
                "analysis_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
