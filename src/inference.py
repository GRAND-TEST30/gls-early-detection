import logging
from PIL import Image
from src.preprocess import GLSEarlyDetector
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GLSInferenceEngine:
    def __init__(self):
        self.detector = GLSEarlyDetector()
        logger.info("GLS Inference Engine initialized - Ready for per-image analysis")

    def validate_image(self, image_input):
        """Validate uploaded image format and size"""
        try:
            if isinstance(image_input, str):
                image = Image.open(image_input)
            else:
                image = Image.open(image_input)
            
            # Basic validation
            if image.size[0] < 100 or image.size[1] < 100:
                logger.warning("Image resolution is too low")
                raise ValueError("Image resolution too low for reliable analysis")
            
            logger.info(f"Image validated: {image.size[0]}x{image.size[1]} pixels")
            return image
        except Exception as e:
            logger.error(f"Image validation failed: {e}")
            raise ValueError(f"Invalid image: {str(e)}")

    def run_full_analysis(self, image_input):
        """Run complete analysis pipeline on a single uploaded image"""
        start_time = datetime.now()
        logger.info("=== Starting Full Inference Pipeline ===")
        
        try:
            # Step 1: Validation
            image = self.validate_image(image_input)
            
            # Step 2: Full Analysis using Preprocessor
            result = self.detector.full_per_image_analysis(image)
            
            # Step 3: Add metadata
            result["processing_time_seconds"] = round((datetime.now() - start_time).total_seconds(), 2)
            result["image_dimensions"] = image.size
            
            logger.info(f"Inference completed successfully in {result['processing_time_seconds']} seconds")
            return result
            
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return {
                "status": "Failed",
                "error_message": str(e),
                "analysis_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

    def generate_detailed_report(self, result):
        """Generate a detailed human-readable report"""
        report = f"""
        GLS EARLY DETECTION REPORT
        =========================
        Analysis Time     : {result.get('analysis_time', 'N/A')}
        Detected Stage    : {result.get('detected_stage', 'N/A')}
        Confidence        : {result.get('confidence_score', 0)}%
        Estimated Lifespan: {result.get('estimated_remaining_productive_days', 0)} days
        
        Lesion Statistics:
        - Total Lesions     : {result.get('lesion_features', {}).get('total_lesions', 0)}
        - Small Lesions     : {result.get('lesion_features', {}).get('small_lesions', 0)}
        - Lesion Density    : {result.get('lesion_features', {}).get('lesion_density', 0)} per 10k pixels
        
        Recommendation:
        {result.get('recommendation', 'No recommendation available')}
        """
        return report.strip()

    def save_analysis_log(self, result, filename="analysis_log.json"):
        """Save analysis result to log file"""
        try:
            with open(filename, 'a') as f:
                json.dump(result, f)
                f.write("\n")
            logger.info(f"Analysis log saved to {filename}")
        except Exception as e:
            logger.warning(f"Failed to save log: {e}")