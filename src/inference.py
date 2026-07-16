import logging
from PIL import Image
from datetime import datetime
from src.preprocess import GLSEarlyDetector

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GLSInferenceEngine:
    def __init__(self):
        self.detector = GLSEarlyDetector()
        logger.info("GLS Inference Engine initialized")

    def run_full_analysis(self, image_input):
        """Fixed version - handles PIL Image directly"""
        start_time = datetime.now()
        logger.info("Starting analysis")

        try:
            # If it's already a PIL Image (from Image.open), use it directly
            if isinstance(image_input, Image.Image):
                image = image_input
            else:
                # If it's a file uploader object, open it
                image = Image.open(image_input)
            
            result = self.detector.full_analysis(image)
            
            result["processing_time_seconds"] = round((datetime.now() - start_time).total_seconds(), 2)
            result["image_dimensions"] = image.size
            
            logger.info("Analysis completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return {
                "status": "Failed",
                "error_message": str(e),
                "analysis_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
