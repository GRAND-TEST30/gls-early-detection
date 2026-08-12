from PIL import Image

from src.enhancement import ImageEnhancer
from src.segmentation import LeafSegmenter
from src.features import FeatureExtractor


class GLSEarlyDetector:

    def __init__(self):

        self.enhancer = ImageEnhancer()
        self.segmenter = LeafSegmenter()
        self.extractor = FeatureExtractor()

    # =========================================================
    # LOAD IMAGE
    # =========================================================

    def load_image(self, image_input):
        """
        Load image from a PIL Image or file path.
        Always return RGB PIL Image.
        """

        if isinstance(image_input, Image.Image):
            return image_input.convert("RGB")

        return Image.open(image_input).convert("RGB")

    # =========================================================
    # HEALTH SCORE
    # =========================================================

    def calculate_health_score(
        self,
        disease_coverage,
        lesion_count,
        mean_green
    ):
        """
        Calculate an overall leaf health score from 0–100.

        Higher disease coverage and lesion count reduce
        the health score.

        Higher green intensity can slightly improve the score.
        """

        score = 100.0

        # -----------------------------------------------------
        # Disease coverage penalty
        # -----------------------------------------------------

        score -= disease_coverage * 2.0

        # -----------------------------------------------------
        # Lesion count penalty
        # -----------------------------------------------------

        score -= lesion_count * 1.2

        # -----------------------------------------------------
        # Green intensity adjustment
        # -----------------------------------------------------

        if mean_green > 140:

            score += 5

        elif mean_green > 120:

            score += 2

        # -----------------------------------------------------
        # Keep score between 0 and 100
        # -----------------------------------------------------

        score = max(
            0,
            min(
                100,
                score
            )
        )

        return round(
            score,
            1
        )

    # =========================================================
    # STAGE PREDICTION
    # =========================================================

    def predict_stage(
        self,
        disease_coverage,
        lesion_count
    ):
        """
        Estimate disease stage using the extracted
        lesion characteristics.

        NOTE:
        This is currently rule-based and not a trained
        machine-learning classifier.
        """

        # -----------------------------------------------------
        # Healthy
        # -----------------------------------------------------

        if (
            disease_coverage < 2
            and lesion_count < 5
        ):

            return (
                "Healthy",
                96,
                90
            )

        # -----------------------------------------------------
        # Early Stage
        # -----------------------------------------------------

        elif disease_coverage < 8:

            return (
                "Early Stage",
                91,
                55
            )

        # -----------------------------------------------------
        # Moderate Stage
        # -----------------------------------------------------

        elif disease_coverage < 18:

            return (
                "Moderate Stage",
                84,
                28
            )

        # -----------------------------------------------------
        # Severe Stage
        # -----------------------------------------------------

        else:

            return (
                "Severe Stage",
                76,
                10
            )

    # =========================================================
    # RECOMMENDATION
    # =========================================================

    def get_recommendation(
        self,
        stage
    ):
        """
        Generate a recommendation according to
        the estimated disease stage.
        """

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

    # =========================================================
    # COMPLETE ANALYSIS
    # =========================================================

    def full_analysis(
        self,
        image_input
    ):
        """
        Perform complete per-image Gray Leaf Spot analysis.

        Pipeline:

        Image
          ↓
        Enhancement
          ↓
        Leaf Segmentation
          ↓
        Leaf Mask
          ↓
        Colour Analysis
        Texture Analysis
        Lesion Detection
          ↓
        Disease Coverage
          ↓
        Health Score
          ↓
        Stage Prediction
          ↓
        Recommendation
        """

        # =====================================================
        # 1. LOAD IMAGE
        # =====================================================

        image = self.load_image(
            image_input
        )

        # =====================================================
        # 2. IMAGE INFORMATION
        # =====================================================

        width, height = image.size

        image_pixels = (
            width * height
        )

        image_dimensions = {
            "width": int(width),
            "height": int(height)
        }

        # =====================================================
        # 3. IMAGE ENHANCEMENT
        # =====================================================

        enhanced = self.enhancer.enhance(
            image
        )

        # =====================================================
        # 4. LEAF SEGMENTATION
        # =====================================================

        segmented, mask = (
            self.segmenter.segment_leaf(
                enhanced
            )
        )

        # =====================================================
        # 5. LEAF AREA
        # =====================================================

        leaf_area = (
            self.segmenter.calculate_leaf_area(
                mask
            )
        )

        # =====================================================
        # 6. LEAF COVERAGE
        # =====================================================

        leaf_coverage = (
            self.segmenter.calculate_leaf_coverage(
                mask
            )
        )

        # =====================================================
        # 7. FEATURE EXTRACTION
        # =====================================================
        #
        # IMPORTANT:
        # We pass the ENHANCED image together with the
        # LEAF MASK.
        #
        # We do NOT pass the black-background segmented
        # image because that can cause the background to
        # be interpreted as lesions.
        # =====================================================

        all_features = (
            self.extractor.extract_all(
                enhanced,
                mask
            )
        )

        # =====================================================
        # 8. FEATURE GROUPS
        # =====================================================

        colour_features = (
            all_features.get(
                "colour_features",
                {}
            )
        )

        texture_features = (
            all_features.get(
                "texture_features",
                {}
            )
        )

        lesion_features = (
            all_features.get(
                "lesion_features",
                {}
            )
        )

        # =====================================================
        # 9. LESION INFORMATION
        # =====================================================

        lesion_count = int(
            lesion_features.get(
                "lesion_count",
                0
            )
        )

        largest_lesion = float(
            lesion_features.get(
                "largest_lesion",
                0
            )
        )

        total_lesion_area = float(
            lesion_features.get(
                "total_lesion_area",
                0
            )
        )

        # =====================================================
        # 10. DISEASE COVERAGE
        # =====================================================
        #
        # Disease coverage is based on lesion area relative
        # to the detected leaf area.
        #
        # It is also bounded between 0 and 100%.
        # =====================================================

        if leaf_area > 0:

            disease_coverage = (
                total_lesion_area /
                leaf_area
            ) * 100

        else:

            disease_coverage = 0.0

        # -----------------------------------------------------
        # Safety protection
        # -----------------------------------------------------

        disease_coverage = max(
            0.0,
            min(
                100.0,
                disease_coverage
            )
        )

        disease_coverage = round(
            disease_coverage,
            2
        )

        # =====================================================
        # 11. LESION RATIO
        # =====================================================

        lesion_ratio = disease_coverage

        # Make sure the nested lesion dictionary contains
        # the final calculated values.

        lesion_features[
            "lesion_ratio"
        ] = lesion_ratio

        lesion_features[
            "disease_coverage"
        ] = disease_coverage

        # =====================================================
        # 12. GREEN VALUE
        # =====================================================

        mean_green = float(
            colour_features.get(
                "mean_green",
                0
            )
        )

        # =====================================================
        # 13. HEALTH SCORE
        # =====================================================

        health_score = (
            self.calculate_health_score(
                disease_coverage,
                lesion_count,
                mean_green
            )
        )

        # =====================================================
        # 14. STAGE PREDICTION
        # =====================================================

        stage, confidence, remaining_days = (
            self.predict_stage(
                disease_coverage,
                lesion_count
            )
        )

        # =====================================================
        # 15. RECOMMENDATION
        # =====================================================

        recommendation = (
            self.get_recommendation(
                stage
            )
        )

        # =====================================================
        # 16. RETURN COMPLETE ANALYSIS
        # =====================================================

        return {

            # -------------------------------------------------
            # DISEASE ASSESSMENT
            # -------------------------------------------------

            "stage": stage,

            "confidence": confidence,

            "remaining_days": remaining_days,

            "health_score": health_score,

            "recommendation": recommendation,

            # -------------------------------------------------
            # IMAGE INFORMATION
            # -------------------------------------------------

            "image_dimensions": image_dimensions,

            "image_pixels": image_pixels,

            # -------------------------------------------------
            # LEAF INFORMATION
            # -------------------------------------------------

            "leaf_area": leaf_area,

            "leaf_coverage": leaf_coverage,

            "disease_coverage": disease_coverage,

            # -------------------------------------------------
            # COLOUR FEATURES
            # -------------------------------------------------

            "colour_features": colour_features,

            # -------------------------------------------------
            # TEXTURE FEATURES
            # -------------------------------------------------

            "texture_features": texture_features,

            # -------------------------------------------------
            # LESION FEATURES
            # -------------------------------------------------

            "lesion_features": lesion_features,

            # -------------------------------------------------
            # PROCESSED IMAGES
            # -------------------------------------------------

            "enhanced_image": enhanced,

            "segmented_image": segmented
        }
