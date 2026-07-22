import streamlit as st
from PIL import Image
from src.inference import GLSInferenceEngine
from datetime import datetime

# Page Configuration
st.set_page_config(
    page_title="GLS Early Detector",
    page_icon="🌽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; color: #2E8B57; }
    .report-box { background-color: #f0f8f0; padding: 20px; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

st.title("🌽 Early Detection of Gray Leaf Spot")
st.markdown("**Cercospora zeae-maydis** in Maize - Per-Image Analysis (No Pre-trained Dataset)")

# Sidebar
with st.sidebar:
    st.header("About the System")
    st.info("""
    This tool performs **per-image analysis** without any pre-training dataset.
    It uses advanced computer vision and rule-based logic to detect early Gray Leaf Spot.
    """)
    st.markdown("### Features")
    st.markdown("- Early lesion detection\n- Severity classification\n- Lifespan estimation\n- Detailed recommendations")

# Main Interface
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Upload Maize Leaf Image")
    uploaded_file = st.file_uploader(
        "Choose a clear photo of maize leaf", 
        type=["jpg", "jpeg", "png"],
        help="For best results, use good lighting and focus on the leaf surface"
    )

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        analyze_button = st.button("🔍 Perform Early Detection Analysis", type="primary")

        if analyze_button:
            with st.spinner("Analyzing image for Gray Leaf Spot... This may take a few seconds."):
                engine = GLSInferenceEngine()
                result = engine.run_full_analysis(image)  # Pass PIL Image
                
                if result.get("status") == "Failed":
                    st.error(result.get("error_message", "Unknown error"))
           else
    st.success("✅ Analysis Completed Successfully!")

    #######################################################
    # DISEASE SUMMARY
    #######################################################

    st.subheader("🌽 Disease Summary")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Disease Stage",
            result.get("stage", "N/A")
        )

    with c2:
        st.metric(
            "Confidence",
            f"{result.get('confidence',0)}%"
        )

    with c3:
        st.metric(
            "Remaining Days",
            f"{result.get('remaining_days',0)}"
        )

    with c4:
        st.metric(
            "Health Score",
            f"{result.get('health_score',0)}/100"
        )

    st.divider()

    #######################################################
    # IMAGES
    #######################################################

    st.subheader("🖼 Image Processing")

    img1, img2 = st.columns(2)

    with img1:

        st.image(
            result.get("enhanced_image"),
            caption="Enhanced Image",
            use_container_width=True
        )

    with img2:

        st.image(
            result.get("segmented_image"),
            caption="Segmented Leaf",
            use_container_width=True
        )

    st.divider()

    #######################################################
    # LEAF INFORMATION
    #######################################################

    st.subheader("🌿 Leaf Statistics")

    l1, l2 = st.columns(2)

    with l1:

        st.metric(
            "Leaf Area",
            f"{result.get('leaf_area',0):,} pixels"
        )

    with l2:

        st.metric(
            "Disease Coverage",
            f"{result.get('disease_coverage',0):.2f}%"
        )

    st.divider()

    #######################################################
    # LESION STATISTICS
    #######################################################

    lesion = result.get("lesion_features", {})

    st.subheader("🦠 Lesion Statistics")

    a, b, c = st.columns(3)

    with a:

        st.metric(
            "Detected Lesions",
            lesion.get("lesion_count",0)
        )

    with b:

        st.metric(
            "Largest Lesion",
            f"{lesion.get('largest_lesion',0):.2f}"
        )

    with c:

        st.metric(
            "Lesion Area",
            f"{lesion.get('total_lesion_area',0):.2f}"
        )

    st.divider()

    #######################################################
    # COLOUR ANALYSIS
    #######################################################

    colour = result.get("colour_features", {})

    st.subheader("🎨 Colour Analysis")

    r1, r2, r3 = st.columns(3)

    with r1:

        st.metric(
            "Mean Red",
            round(colour.get("mean_red",0),2)
        )

    with r2:

        st.metric(
            "Mean Green",
            round(colour.get("mean_green",0),2)
        )

    with r3:

        st.metric(
            "Mean Blue",
            round(colour.get("mean_blue",0),2)
        )

    st.write(
        f"**Mean Hue:** {colour.get('mean_hue',0):.2f}"
    )

    st.write(
        f"**Mean Saturation:** {colour.get('mean_saturation',0):.2f}"
    )

    st.write(
        f"**Mean Brightness:** {colour.get('mean_value',0):.2f}"
    )

    st.divider()

    #######################################################
    # TEXTURE ANALYSIS
    #######################################################

    texture = result.get("texture_features", {})

    st.subheader("🧩 Texture Analysis")

    t1, t2 = st.columns(2)

    with t1:

        st.write(
            f"**Contrast:** {texture.get('contrast',0):.4f}"
        )

        st.write(
            f"**Homogeneity:** {texture.get('homogeneity',0):.4f}"
        )

        st.write(
            f"**Energy:** {texture.get('energy',0):.4f}"
        )

    with t2:

        st.write(
            f"**Correlation:** {texture.get('correlation',0):.4f}"
        )

        st.write(
            f"**ASM:** {texture.get('ASM',0):.4f}"
        )

    st.divider()

    #######################################################
    # RECOMMENDATION
    #######################################################

    st.subheader("📋 Recommendation")

    st.success(
        result.get(
            "recommendation",
            "No recommendation available."
        )
    )

    st.divider()

    #######################################################
    # ANALYSIS INFORMATION
    #######################################################

    st.subheader("📈 Analysis Information")

    i1, i2, i3 = st.columns(3)

    with i1:

        st.metric(
            "Processing Time",
            f"{result.get('processing_time_seconds',0)} sec"
        )

    with i2:

        dimensions = result.get("image_dimensions",{})

        st.metric(
            "Image Width",
            dimensions.get("width",0)
        )

    with i3:

        st.metric(
            "Image Height",
            dimensions.get("height",0)
        )

    with st.expander("📊 Complete Analysis Dictionary"):

        st.json(result)

with col2:
    st.subheader("How it Works")
    st.markdown("""
    1. Image uploaded
    2. Multi-stage enhancement
    3. Lesion feature extraction
    4. Severity classification
    5. Lifespan prediction
    """)
   
    if uploaded_file and 'result' in locals():
        report_text = str(result)
        st.download_button(
            label="Download Full Report",
            data=report_text,
            file_name=f"gls_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
            mime="text/plain"
        )

st.markdown("---")
st.caption("Final Year Project | Early Detection of Gray Leaf Spot using Computer Vision")
