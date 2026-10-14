"""PNEUMO.AI — Explainability page."""

import streamlit as st

from utils.config import (
    ATTENTION_MEDIANS,
    REPRESENTATIVE_CASES_IMAGE,
    RESEARCH_DISCLAIMER,
    SPEARMAN_CORRELATIONS,
)
from utils.metrics import load_overlap_analysis
from utils.theme import disclaimer_box, inject_global_style, metric_card, page_header

st.set_page_config(page_title="Explainability | PNEUMO.AI", page_icon=":mag:", layout="wide")
inject_global_style()
page_header(
    "Explainability",
    "Anatomical-Consistency Analysis",
    "How Grad-CAM attention relates to an approximate lung reference, across the full 640-image test set.",
)

# ------------------------------------------------------------------
# Concepts
# ------------------------------------------------------------------
st.markdown("### The building blocks")

concept_cols = st.columns(2)
with concept_cols[0]:
    st.markdown(
        """
**What is Grad-CAM?**
Grad-CAM highlights which regions of an X-ray most influenced the DenseNet121
model's pneumonia prediction, by tracing gradients back to the last
convolutional layer (`conv5_block16_2_conv`). Warmer colors mean the region
contributed more to the predicted probability.

**What is the approximate lung reference?**
A pretrained TorchXRayVision PSPNet model (not trained on this dataset, and
not a pneumonia detector) is used to roughly locate the left and right lung
fields. It is a general anatomical reference, not a precise or ground-truth
segmentation.
"""
    )
with concept_cols[1]:
    st.markdown(
        """
**Attention inside lung** — the share of total Grad-CAM activation that
falls inside the approximate lung region.

**High-attention overlap** — of the *most* activated Grad-CAM pixels (top
20%), what share falls inside the approximate lung region.

**Lung coverage** — how much of the image the approximate reference itself
classifies as lung tissue.
"""
    )

st.markdown("---")

# ------------------------------------------------------------------
# Representative cases
# ------------------------------------------------------------------
st.markdown("### Representative cases")
if REPRESENTATIVE_CASES_IMAGE.exists():
    st.image(str(REPRESENTATIVE_CASES_IMAGE), use_container_width=True)
else:
    st.warning("representative_gradcam_cases.png not found in visuals/.")

st.markdown("---")

# ------------------------------------------------------------------
# Findings from the full analysis
# ------------------------------------------------------------------
st.markdown("### Key findings (640-image held-out test set)")

finding_cols = st.columns(2)
with finding_cols[0]:
    st.markdown(
        metric_card("NORMAL — median attention inside lung", f"{ATTENTION_MEDIANS['NORMAL']:.1f}%"),
        unsafe_allow_html=True,
    )
with finding_cols[1]:
    st.markdown(
        metric_card("PNEUMONIA — median attention inside lung", f"{ATTENTION_MEDIANS['PNEUMONIA']:.1f}%"),
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

corr_cols = st.columns(3)
with corr_cols[0]:
    st.markdown(
        metric_card("Spearman ρ — overall", f"{SPEARMAN_CORRELATIONS['overall']:.3f}"),
        unsafe_allow_html=True,
    )
with corr_cols[1]:
    st.markdown(
        metric_card("Spearman ρ — pneumonia cases", f"{SPEARMAN_CORRELATIONS['pneumonia']:.3f}"),
        unsafe_allow_html=True,
    )
with corr_cols[2]:
    st.markdown(
        metric_card("Spearman ρ — normal cases", f"{SPEARMAN_CORRELATIONS['normal']:.3f}"),
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)
st.markdown(
    """
Grad-CAM attention-inside-lung overlap was **strongly associated with the
estimated lung coverage** produced by the approximate reference — especially
for pneumonia cases (ρ ≈ 0.86), and more moderately for normal cases
(ρ ≈ 0.47). In other words, when the approximate lung reference marks more
of the image as lung, Grad-CAM's attention tends to overlap it more too.

**This means overlap should be read as an anatomical-consistency /
explainability signal, not as a direct reliability score.** A lower overlap
does not, by itself, prove the model looked "outside the lungs" in a
clinically meaningful sense — it can also reflect how much lung area the
approximate reference itself detected for that image.
"""
)

st.markdown("---")

# ------------------------------------------------------------------
# Full results table
# ------------------------------------------------------------------
st.markdown("### Full test-set results")
try:
    df = load_overlap_analysis()
    st.dataframe(df, use_container_width=True, height=360)
    st.caption(f"{len(df)} test images — attention/overlap metrics computed once during experimentation.")
except FileNotFoundError as exc:
    st.warning(str(exc))

disclaimer_box(RESEARCH_DISCLAIMER)
