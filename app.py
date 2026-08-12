import streamlit as st
from PIL import Image
from src.inference import GLSInferenceEngine
from datetime import datetime


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="GLS Early Detector",
    page_icon="🌽",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-header {
        font-size: 2.5rem;
        color: #2E8B57;
        font-weight: 700;
    }

    .report-box {
        background-color: #f0f8f0;
        padding: 20px;
        border-radius: 10px;
    }

    .section-title {
        color: #2E8B57;
        font-weight: 700;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# APPLICATION TITLE
# ============================================================

st.title("🌽 Early Detection of Gray Leaf Spot")

st.markdown(
    "**Cercospora zeae-maydis** in Maize - "
    "Per-Image Analysis (No Pre-trained Dataset)"
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("About the System")

    st.info(
        """
        This tool performs **per-image analysis** without
        any pre-training dataset.

        It uses computer vision and rule-based analysis to
        examine maize leaf images for visual indicators
        associated with Gray Leaf Spot.
        """
    )

    st.markdown("### Features")

    st.markdown(
        """
        - 🌿 Leaf segmentation
        - 🦠 Lesion detection
        - 🎨 Colour analysis
        - 🧩 Texture analysis
        - 📊 Disease coverage
        - 📈 Severity classification
        - ❤️ Health score
        - ⏳ Remaining-day estimation
        - 📋 Detailed recommendations
        """
    )


# ============================================================
# MAIN PAGE LAYOUT
# ============================================================

col1, col2 = st.columns([2, 1])


# ============================================================
# LEFT COLUMN
# ============================================================

with col1:

    st.subheader("Upload Maize Leaf Image")

    uploaded_file = st.file_uploader(
        "Choose a clear photo of maize leaf",
        type=["jpg", "jpeg", "png"],
        help=(
            "For best results, use a clear maize leaf image "
            "with good lighting and minimal background obstruction."
        )
    )

    # --------------------------------------------------------
    # IMAGE UPLOAD
    # --------------------------------------------------------

    if uploaded_file:

        try:

            image = Image.open(uploaded_file).convert("RGB")

            # IMPORTANT:
            # use_container_width replaces the deprecated
            # use_column_width parameter.
            st.image(
                image,
                caption="Uploaded Maize Leaf",
                use_container_width=True
            )

        except Exception as e:

            st.error(
                f"Unable to read the uploaded image: {str(e)}"
            )

            st.stop()


        # ----------------------------------------------------
        # ANALYSIS BUTTON
        # ----------------------------------------------------

        analyze_button = st.button(
            "🔍 Perform Early Detection Analysis",
            type="primary",
            use_container_width=True
        )


        # ====================================================
        # RUN ANALYSIS
        # ====================================================

        if analyze_button:

            with st.spinner(
                "Analyzing image for Gray Leaf Spot... "
                "This may take a few seconds."
            ):

                try:

                    engine = GLSInferenceEngine()

                    result = engine.run_full_analysis(image)

                except Exception as e:

                    st.error(
                        "An error occurred while analyzing "
                        "the image."
                    )

                    st.exception(e)

                    st.stop()


            # =================================================
            # CHECK ANALYSIS STATUS
            # =================================================

            if result.get("status") == "Failed":

                st.error(
                    result.get(
                        "error_message",
                        "Unknown analysis error."
                    )
                )

                st.stop()


            # =================================================
            # SUCCESS
            # =================================================

            st.success(
                "✅ Analysis Completed Successfully!"
            )


            # =================================================
            # DISEASE SUMMARY
            # =================================================

            st.subheader("🌽 Disease Summary")

            c1, c2, c3, c4 = st.columns(4)


            with c1:

                st.metric(
                    "Disease Stage",
                    result.get(
                        "stage",
                        "N/A"
                    )
                )


            with c2:

                st.metric(
                    "Confidence",
                    f"{result.get('confidence', 0)}%"
                )


            with c3:

                st.metric(
                    "Remaining Days",
                    f"{result.get('remaining_days', 0)}"
                )


            with c4:

                st.metric(
                    "Health Score",
                    f"{result.get('health_score', 0)}/100"
                )


            st.divider()


            # =================================================
            # IMAGE PROCESSING
            # =================================================

            st.subheader("🖼 Image Processing")

            img1, img2 = st.columns(2)


            with img1:

                enhanced_image = result.get(
                    "enhanced_image"
                )

                if enhanced_image is not None:

                    st.image(
                        enhanced_image,
                        caption="Enhanced Image",
                        use_container_width=True
                    )

                else:

                    st.warning(
                        "Enhanced image is not available."
                    )


            with img2:

                segmented_image = result.get(
                    "segmented_image"
                )

                if segmented_image is not None:

                    st.image(
                        segmented_image,
                        caption="Segmented Leaf",
                        use_container_width=True
                    )

                else:

                    st.warning(
                        "Segmented image is not available."
                    )


            st.divider()


            # =================================================
            # LEAF STATISTICS
            # =================================================

            st.subheader("🌿 Leaf Statistics")

            l1, l2, l3 = st.columns(3)


            with l1:

                leaf_area = result.get(
                    "leaf_area",
                    0
                )

                try:
                    leaf_area_display = f"{float(leaf_area):,.0f}"
                except:
                    leaf_area_display = str(leaf_area)

                st.metric(
                    "Leaf Area",
                    f"{leaf_area_display} pixels"
                )


            with l2:

                disease_coverage = result.get(
                    "disease_coverage",
                    0
                )

                try:
                    disease_coverage_display = (
                        f"{float(disease_coverage):.2f}%"
                    )
                except:
                    disease_coverage_display = str(
                        disease_coverage
                    )

                st.metric(
                    "Disease Coverage",
                    disease_coverage_display
                )


            with l3:

                total_pixels = result.get(
                    "total_pixels",
                    0
                )

                try:
                    total_pixels_display = (
                        f"{float(total_pixels):,.0f}"
                    )
                except:
                    total_pixels_display = str(
                        total_pixels
                    )

                st.metric(
                    "Image Pixels",
                    total_pixels_display
                )


            st.divider()


            # =================================================
            # LESION STATISTICS
            # =================================================

            lesion = result.get(
                "lesion_features",
                {}
            )

            if not isinstance(lesion, dict):

                lesion = {}


            st.subheader("🦠 Lesion Statistics")

            a, b, c, d = st.columns(4)


            with a:

                st.metric(
                    "Detected Lesions",
                    lesion.get(
                        "lesion_count",
                        0
                    )
                )


            with b:

                largest_lesion = lesion.get(
                    "largest_lesion",
                    0
                )

                try:
                    largest_lesion = (
                        f"{float(largest_lesion):.2f}"
                    )
                except:
                    largest_lesion = str(
                        largest_lesion
                    )

                st.metric(
                    "Largest Lesion",
                    largest_lesion
                )


            with c:

                total_lesion_area = lesion.get(
                    "total_lesion_area",
                    0
                )

                try:
                    total_lesion_area = (
                        f"{float(total_lesion_area):.2f}"
                    )
                except:
                    total_lesion_area = str(
                        total_lesion_area
                    )

                st.metric(
                    "Total Lesion Area",
                    total_lesion_area
                )


            with d:

                lesion_ratio = lesion.get(
                    "lesion_ratio",
                    0
                )

                try:
                    lesion_ratio = (
                        f"{float(lesion_ratio):.2f}%"
                    )
                except:
                    lesion_ratio = str(
                        lesion_ratio
                    )

                st.metric(
                    "Lesion Ratio",
                    lesion_ratio
                )


            with st.expander(
                "🔬 View Complete Lesion Features"
            ):

                st.json(lesion)


            st.divider()


            # =================================================
            # COLOUR ANALYSIS
            # =================================================

            colour = result.get(
                "colour_features",
                {}
            )

            if not isinstance(colour, dict):

                colour = {}


            st.subheader("🎨 Colour Analysis")


            r1, r2, r3 = st.columns(3)


            with r1:

                st.metric(
                    "Mean Red",
                    f"{float(colour.get('mean_red', 0)):.2f}"
                )


            with r2:

                st.metric(
                    "Mean Green",
                    f"{float(colour.get('mean_green', 0)):.2f}"
                )


            with r3:

                st.metric(
                    "Mean Blue",
                    f"{float(colour.get('mean_blue', 0)):.2f}"
                )


            r4, r5, r6 = st.columns(3)


            with r4:

                st.metric(
                    "Mean Hue",
                    f"{float(colour.get('mean_hue', 0)):.2f}"
                )


            with r5:

                st.metric(
                    "Mean Saturation",
                    f"{float(colour.get('mean_saturation', 0)):.2f}"
                )


            with r6:

                st.metric(
                    "Mean Brightness",
                    f"{float(colour.get('mean_value', 0)):.2f}"
                )


            with st.expander(
                "🎨 View Complete Colour Features"
            ):

                st.json(colour)


            st.divider()


            # =================================================
            # TEXTURE ANALYSIS
            # =================================================

            texture = result.get(
                "texture_features",
                {}
            )

            if not isinstance(texture, dict):

                texture = {}


            st.subheader("🧩 Texture Analysis")


            t1, t2 = st.columns(2)


            with t1:

                st.write(
                    f"**Contrast:** "
                    f"{float(texture.get('contrast', 0)):.4f}"
                )

                st.write(
                    f"**Homogeneity:** "
                    f"{float(texture.get('homogeneity', 0)):.4f}"
                )

                st.write(
                    f"**Energy:** "
                    f"{float(texture.get('energy', 0)):.4f}"
                )


            with t2:

                st.write(
                    f"**Correlation:** "
                    f"{float(texture.get('correlation', 0)):.4f}"
                )

                st.write(
                    f"**ASM:** "
                    f"{float(texture.get('ASM', 0)):.4f}"
                )


            with st.expander(
                "🧩 View Complete Texture Features"
            ):

                st.json(texture)


            st.divider()


            # =================================================
            # RECOMMENDATION
            # =================================================

            st.subheader("📋 Recommendation")


            recommendation = result.get(
                "recommendation",
                "No recommendation available."
            )


            st.success(
                recommendation
            )


            st.divider()


            # =================================================
            # ANALYSIS INFORMATION
            # =================================================

            st.subheader("📈 Analysis Information")


            i1, i2, i3 = st.columns(3)


            with i1:

                processing_time = result.get(
                    "processing_time_seconds",
                    "N/A"
                )

                st.metric(
                    "Processing Time",
                    f"{processing_time} sec"
                )


            # -------------------------------------------------
            # IMAGE DIMENSIONS
            # -------------------------------------------------

            dimensions = result.get(
                "image_dimensions",
                None
            )


            width = 0
            height = 0


            # Handle dictionary format
            if isinstance(dimensions, dict):

                width = dimensions.get(
                    "width",
                    0
                )

                height = dimensions.get(
                    "height",
                    0
                )


            # Handle tuple/list format
            elif isinstance(
                dimensions,
                (tuple, list)
            ) and len(dimensions) >= 2:

                width = dimensions[0]
                height = dimensions[1]


            with i2:

                st.metric(
                    "Image Width",
                    width
                )


            with i3:

                st.metric(
                    "Image Height",
                    height
                )


            st.divider()


            # =================================================
            # COMPLETE ANALYSIS
            # =================================================

            with st.expander(
                "📊 Complete Analysis Dictionary"
            ):

                # Remove image objects before displaying
                # JSON to prevent serialization problems.
                display_result = dict(result)

                display_result.pop(
                    "enhanced_image",
                    None
                )

                display_result.pop(
                    "segmented_image",
                    None
                )

                st.json(display_result)


            # =================================================
            # DOWNLOAD REPORT
            # =================================================

            st.subheader("📥 Download Analysis Report")


            report_lines = []


            report_lines.append(
                "EARLY DETECTION OF GRAY LEAF SPOT"
            )

            report_lines.append(
                "Cercospora zeae-maydis in Maize"
            )

            report_lines.append(
                "=" * 60
            )

            report_lines.append("")


            report_lines.append(
                f"Disease Stage: "
                f"{result.get('stage', 'N/A')}"
            )

            report_lines.append(
                f"Confidence: "
                f"{result.get('confidence', 0)}%"
            )

            report_lines.append(
                f"Remaining Days: "
                f"{result.get('remaining_days', 0)}"
            )

            report_lines.append(
                f"Health Score: "
                f"{result.get('health_score', 0)}/100"
            )

            report_lines.append("")


            report_lines.append(
                f"Leaf Area: "
                f"{result.get('leaf_area', 0)} pixels"
            )

            report_lines.append(
                f"Disease Coverage: "
                f"{result.get('disease_coverage', 0)}%"
            )

            report_lines.append("")


            report_lines.append(
                "LESION FEATURES"
            )

            report_lines.append(
                "-" * 40
            )


            for key, value in lesion.items():

                report_lines.append(
                    f"{key}: {value}"
                )


            report_lines.append("")


            report_lines.append(
                "COLOUR FEATURES"
            )

            report_lines.append(
                "-" * 40
            )


            for key, value in colour.items():

                report_lines.append(
                    f"{key}: {value}"
                )


            report_lines.append("")


            report_lines.append(
                "TEXTURE FEATURES"
            )

            report_lines.append(
                "-" * 40
            )


            for key, value in texture.items():

                report_lines.append(
                    f"{key}: {value}"
                )


            report_lines.append("")


            report_lines.append(
                "RECOMMENDATION"
            )

            report_lines.append(
                "-" * 40
            )

            report_lines.append(
                str(recommendation)
            )


            report_lines.append("")


            report_lines.append(
                f"Generated: "
                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )


            report_text = "\n".join(
                report_lines
            )


            st.download_button(
                label="📥 Download Full Report",
                data=report_text,
                file_name=(
                    "gls_analysis_"
                    f"{datetime.now().strftime('%Y%m%d_%H%M')}.txt"
                ),
                mime="text/plain",
                use_container_width=True
            )


# ============================================================
# RIGHT COLUMN
# ============================================================

with col2:

    st.subheader("How it Works")

    st.markdown(
        """
        ### 🔬 Analysis Pipeline

        **1. Image Upload**

        The system receives a maize leaf image.

        **2. Image Enhancement**

        Image quality and contrast are enhanced.

        **3. Leaf Segmentation**

        The system separates the maize leaf from
        the surrounding background.

        **4. Lesion Detection**

        Potential disease lesions are identified
        from the leaf region.

        **5. Colour Analysis**

        RGB, HSV and colour characteristics are
        extracted from the image.

        **6. Texture Analysis**

        Texture characteristics are calculated
        to identify visual changes associated with
        disease.

        **7. Disease Assessment**

        The extracted visual characteristics are
        evaluated using the current rule-based
        analysis system.

        **8. Stage Classification**

        The system estimates the disease stage.

        **9. Recommendation**

        A recommendation is generated according
        to the estimated disease stage.
        """
    )


    st.divider()


    st.subheader("📌 Important Note")

    st.info(
        """
        This system performs **per-image computer
        vision analysis** and does not use a
        pre-trained disease dataset.

        Therefore, the displayed confidence and
        disease-stage estimates should be interpreted
        as analytical estimates rather than clinical
        or laboratory confirmation.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Final Year Project | Early Detection of Gray Leaf Spot "
    "using Computer Vision"
)
