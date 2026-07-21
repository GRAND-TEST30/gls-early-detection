from datetime import datetime
from PIL import Image

from src.preprocess import GLSEarlyDetector


class GLSInferenceEngine:
    """
    Runs the complete Gray Leaf Spot analysis pipeline.

    Workflow
    --------
    Image
        ↓
    Preprocessing
        ↓
    Enhancement
        ↓
    Segmentation
        ↓
    Feature Extraction
        ↓
    Disease Analysis
        ↓
    Results Dictionary
    """

    def __init__(self):

        self.detector = GLSEarlyDetector()

    ###############################################################
    # FULL ANALYSIS
    ###############################################################

    def run_full_analysis(self, image_input):

        start_time = datetime.now()

        try:

            #########################################################
            # Load Image
            #########################################################

            if isinstance(image_input, Image.Image):

                image = image_input.convert("RGB")

            else:

                image = Image.open(image_input).convert("RGB")

            #########################################################
            # Run Detector
            #########################################################

            result = self.detector.full_analysis(image)

            #########################################################
            # Extra Information
            #########################################################

            end_time = datetime.now()

            processing_time = (
                end_time - start_time
            ).total_seconds()

            result["status"] = "Success"

            result["processing_time_seconds"] = round(
                processing_time,
                3
            )

            result["analysis_timestamp"] = datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )

            result["image_dimensions"] = {

                "width": image.width,

                "height": image.height

            }

            result["image_mode"] = image.mode

            return result

        #############################################################
        # ERROR HANDLING
        #############################################################

        except Exception as e:

            return {

                "status": "Failed",

                "error_message": str(e)

            }

    ###############################################################
    # QUICK SUMMARY
    ###############################################################

    def get_summary(self, result):

        if result.get("status") != "Success":

            return "Analysis failed."

        summary = f"""
Gray Leaf Spot Analysis Summary

Disease Stage      : {result.get('stage')}

Confidence         : {result.get('confidence')}%

Disease Coverage   : {result.get('disease_coverage')}%

Health Score       : {result.get('health_score')}/100

Remaining Days     : {result.get('remaining_days')} days

Leaf Area          : {result.get('leaf_area')} pixels

Recommendation

{result.get('recommendation')}
"""

        return summary
