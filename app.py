"""
PNEUMO.AI — Home page.

Run with:
    streamlit run app.py
"""

import streamlit as st

from utils.config import APP_TAGLINE, APP_TITLE, RESEARCH_DISCLAIMER
from utils.theme import disclaimer_box, inject_global_style

st.set_page_config(
    page_title="PNEUMO.AI",
    page_icon=":lungs:",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_global_style()

# ----------------------------------------------------------------------
# Hero
# ----------------------------------------------------------------------
st.markdown(
    '<div class="pneumo-eyebrow">Final-Year Deep Learning Project</div>',
    unsafe_allow_html=True,
)
st.markdown(f"# {APP_TITLE}")
st.markdown(f"#### {APP_TAGLINE}")

st.markdown(
    """
A research prototype that pairs a fine-tuned **DenseNet121** classifier with
**Grad-CAM** explainability and an approximate **anatomical lung reference**,
to study how consistent the model's visual attention is with lung anatomy —
in pediatric chest X-rays screened for pneumonia.
"""
)

col1, col2 = st.columns([1, 1])
with col1:
    if st.button("Analyze a Chest X-Ray", type="primary", use_container_width=True):
        st.switch_page("pages/1_Pneumonia_Detection.py")
with col2:
    if st.button("View Model Performance", use_container_width=True):
        st.switch_page("pages/3_Model_Performance.py")

st.markdown("---")

# ----------------------------------------------------------------------
# Pipeline overview
# ----------------------------------------------------------------------
st.markdown("### How it works")

steps = [
    "Chest X-Ray",
    "DenseNet121",
    "Pneumonia\nProbability",
    "Grad-CAM",
    "Approx. Lung\nReference",
    "Explainability\nAnalysis",
]
cols = st.columns(len(steps))
for c, step in zip(cols, steps):
    with c:
        st.markdown(
            f'<div class="pneumo-pipeline-step">{step.replace(chr(10), "<br>")}</div>',
            unsafe_allow_html=True,
        )

st.markdown("<br>", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# Quick project facts
# ----------------------------------------------------------------------
st.markdown("### At a glance")
fact_cols = st.columns(4)
facts = [
    ("Backbone", "DenseNet121"),
    ("Test ROC-AUC", "0.963"),
    ("Test images", "640"),
    ("Explainability tool", "Grad-CAM"),
]
for c, (label, value) in zip(fact_cols, facts):
    with c:
        st.markdown(
            f'<div class="pneumo-metric-card">'
            f'<div class="pneumo-metric-value">{value}</div>'
            f'<div class="pneumo-metric-label">{label}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("### Explore")
st.markdown(
    """
- **Pneumonia Detection** — upload a chest X-ray and get a live prediction with Grad-CAM
- **Explainability** — what Grad-CAM and the lung reference show, and what the analysis found
- **Model Performance** — DenseNet121 vs. CNN baseline on the held-out test set
- **Dataset & Methodology** — how the data was split and the full experimental pipeline
- **About & Limitations** — what this system is, and is not
"""
)

st.markdown("<br>", unsafe_allow_html=True)
disclaimer_box(RESEARCH_DISCLAIMER)
