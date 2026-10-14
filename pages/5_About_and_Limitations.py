"""PNEUMO.AI — About & Limitations page."""

import streamlit as st

from utils.config import APP_TAGLINE, RESEARCH_DISCLAIMER
from utils.theme import disclaimer_box, inject_global_style, page_header

st.set_page_config(page_title="About | PNEUMO.AI", page_icon=":books:", layout="wide")
inject_global_style()
page_header("About", "About PNEUMO.AI", APP_TAGLINE)

st.markdown(
    """
PNEUMO.AI is a final-year deep learning project that studies a focused
research question:

> **How anatomically consistent are DenseNet121's Grad-CAM explanations with
> approximate lung regions, and how does this consistency relate to
> prediction behavior in pediatric chest X-rays?**

It combines a fine-tuned image classifier with two explainability tools —
Grad-CAM and an approximate anatomical lung reference — to examine that
question empirically on a held-out test set, rather than to build a
deployable diagnostic product.
"""
)

st.markdown("---")
st.markdown("### What this system is")
st.markdown(
    """
- A binary chest X-ray classifier (NORMAL vs. PNEUMONIA) built by transfer
  learning on DenseNet121
- An explainability study using Grad-CAM, cross-referenced against an
  approximate lung region from a pretrained anatomical segmentation model
- A demonstration of how model attention can be examined for anatomical
  plausibility, not just accuracy
"""
)

st.markdown("### What this system is not")
st.markdown(
    """
This application does **not** claim that:

- Grad-CAM proves where pneumonia is physically located in the image
- Grad-CAM provides a causal explanation of the model's decision
- the PSPNet lung mask is a ground-truth anatomical segmentation
- low Grad-CAM/lung overlap proves the model looked outside the lungs
- lung overlap directly measures the model's reliability
- lung masking improved classification performance
- the model is clinically validated
- the model can replace a radiologist
- the model can diagnose an individual patient in a clinical setting
"""
)

st.markdown("---")
disclaimer_box(RESEARCH_DISCLAIMER)

st.markdown("<br>", unsafe_allow_html=True)
st.caption(
    "Built with Streamlit, TensorFlow/Keras (DenseNet121), TorchXRayVision (PSPNet), "
    "and OpenCV. Model experimentation was performed in Google Colab; this application "
    "loads the final trained model and reproduces its published methodology and results."
)
