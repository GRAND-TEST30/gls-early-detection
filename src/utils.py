import streamlit as st
import matplotlib.pyplot as plt

def display_prediction(pred_class, confidence, severity):
    if pred_class == "Gray Leaf Spot":
        st.success(f"**{pred_class}** detected")
        st.metric("Confidence", f"{confidence:.2f}%")
        st.warning(f"**Severity:** {severity} - Immediate action recommended!")
    elif pred_class == "Healthy":
        st.success("Confirm **Healthy Leaf**")
        st.metric("Confidence", f"{confidence:.2f}%")
    else:
        st.info(f"**{pred_class}** detected")
        st.metric("Confidence", f"{confidence:.2f}%")