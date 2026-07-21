from PIL import Image
import numpy as np

from src.enhancement import ImageEnhancer
from src.segmentation import LeafSegmenter
from src.features import FeatureExtractor


class GLSEarlyDetector:

    def __init__(self):

        self.enhancer = ImageEnhancer()
        self.segmenter = LeafSegmenter()
        self.extractor = FeatureExtractor()

    ###########################################################
    # LOAD IMAGE
    ###########################################################

    def load_image(self, image_input):

        if isinstance(image_input, Image.Image):
            return image_input.convert("RGB")

        return Image.open(image_input).convert("RGB")

    ###########################################################
    # HEALTH SCORE
    ###########################################################

    def calculate_health_score(
        self,
        disease_coverage,
        lesion_count,
        mean_green
    ):
        """
        Calculate overall leaf health score (0–100)
        """

        score = 100

        # Disease coverage penalty
        score -= disease_coverage * 2

        # Lesion count penalty
        score -= lesion_count * 1.2

        # Green intensity bonus
        if mean_green > 140:
            score += 5
        elif mean_green > 120:
            score += 2

        score = max(0, min(100, score))

        return round(score, 1)

    ###########################################################
    # STAGE PREDICTION
    ###########################################################

    def predict_stage(
        self,
        disease_coverage,
        lesion_count
    ):

        if disease_coverage < 2 and lesion_count < 5:

            return "Healthy", 96, 90

        elif disease_coverage < 8:

            return "Early Stage", 91, 55

        elif disease_coverage < 18:

            return "Moderate Stage", 84, 28

        else:

            return "Severe Stage", 76, 10

    ###########################################################
    # RECOMMENDATION
    ###########################################################

    def get_recommendation(self, stage):

        if stage == "Healthy":

            return (
                "Leaf appears healthy. Continue regular monitoring "
                "and maintain good agronomic practices."
            )

        elif stage == "Early Stage":

            return (
                "Early Gray Leaf Spot detected.\n\n"
                "• Begin monitoring affected plants.\n"
                "• Consider preventive fungicide application.\n"
                "• Improve field ventilation.\n"
                "• Avoid prolonged leaf wetness."
            )

        elif stage == "Moderate Stage":

            return (
                "Moderate infection detected.\n\n"
                "• Apply recommended fungicide.\n"
                "• Remove heavily infected leaves.\n"
                "• Monitor disease progression every 2–3 days."
            )

        else:

            return (
                "Severe Gray Leaf Spot detected.\n\n"
                "• Immediate intervention required.\n"
                "• Apply fungicide immediately.\n"
                "• Remove severely infected leaves.\n"
                "• Assess possible yield loss."
            )

    ###########################################################
    # COMPLETE ANALYSIS
    ###########################################################

    def full_analysis(self, image_input):

        #######################################################
        # Load
        #######################################################

        image = self.load_image(image_input)

        #######################################################
        # Enhancement
        #######################################################

        enhanced = self.enhancer.enhance(image)

        #######################################################
        # Segmentation
        #######################################################

        segmented, mask = self.segmenter.segment_leaf(enhanced)

        leaf_area = self.segmenter.calculate_leaf_area(mask)

        #######################################################
        # Feature Extraction
        #######################################################

        all_features = self.extractor.extract_all(segmented)

        #######################################################
        # Split Features
        #######################################################

        colour_features = {

            "mean_red": all_features["mean_red"],
            "mean_green": all_features["mean_green"],
            "mean_blue": all_features["mean_blue"],
            "mean_hue": all_features["mean_hue"],
            "mean_saturation": all_features["mean_saturation"],
            "mean_value": all_features["mean_value"]

        }

        texture_features = {

            "contrast": all_features["contrast"],
            "homogeneity": all_features["homogeneity"],
            "energy": all_features["energy"],
            "correlation": all_features["correlation"],
            "ASM": all_features["ASM"]

        }

        lesion_features = {

            "lesion_count": all_features["lesion_count"],
            "largest_lesion": all_features["largest_lesion"],
            "total_lesion_area": all_features["total_lesion_area"]

        }

        #######################################################
        # Disease Coverage
        #######################################################

        lesion_area = lesion_features["total_lesion_area"]

        if leaf_area == 0:

            disease_coverage = 0

        else:

            disease_coverage = (
                lesion_area / leaf_area
            ) * 100

        #######################################################
        # Health Score
        #######################################################

        health_score = self.calculate_health_score(

            disease_coverage,

            lesion_features["lesion_count"],

            colour_features["mean_green"]

        )

        #######################################################
        # Stage Prediction
        #######################################################

        stage, confidence, remaining_days = self.predict_stage(

            disease_coverage,

            lesion_features["lesion_count"]

        )

        #######################################################
        # Recommendation
        #######################################################

        recommendation = self.get_recommendation(stage)

        #######################################################
        # Return Everything
        #######################################################

        return {

            ###################################################
            # Prediction
            ###################################################

            "stage": stage,

            "confidence": confidence,

            "remaining_days": remaining_days,

            "recommendation": recommendation,

            ###################################################
            # Statistics
            ###################################################

            "leaf_area": leaf_area,

            "disease_coverage": round(
                disease_coverage,
                2
            ),

            "health_score": health_score,

            ###################################################
            # Feature Groups
            ###################################################

            "colour_features": colour_features,

            "texture_features": texture_features,

            "lesion_features": lesion_features,

            ###################################################
            # Images
            ###################################################

            "enhanced_image": enhanced,

            "segmented_image": segmented

        }
