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
                else:
                    st.success("Analysis Completed Successfully!")
                   
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Detected Stage", result.get("stage", "N/A"))
                    with col_b:
                        st.metric("Confidence", f"{result.get('confidence', 0)}%")
                    with col_c:
                        st.metric("Est. Remaining Days", f"{result.get('remaining_days', 0)} days")
                   
                    st.subheader("Detailed Recommendation")
                    st.info(result.get("recommendation", "No recommendation"))
                   
                    with st.expander("View Detailed Lesion Statistics"):
                        st.json(result.get("lesion_features", {}))

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
