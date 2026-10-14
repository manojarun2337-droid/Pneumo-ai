"""PNEUMO.AI — Chest X-Ray analysis page."""

import tempfile
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

from utils.config import CLASSIFICATION_THRESHOLD, RESEARCH_DISCLAIMER
from utils.gradcam import make_gradcam_heatmap, overlay_heatmap_on_image, resize_heatmap
from utils.metrics import calculate_overlap_metrics
from utils.prediction import classify, load_model, predict_pneumonia_probability, preprocess_image
from utils.segmentation import get_approximate_lung_mask, load_segmentation_model, resize_mask_to_224
from utils.theme import disclaimer_box, inject_global_style, metric_card, page_header, prediction_badge

st.set_page_config(page_title="Pneumonia Detection | PNEUMO.AI", page_icon=":lungs:", layout="wide")
inject_global_style()
page_header(
    "Analyze",
    "Chest X-Ray Analysis",
    "Upload a frontal chest X-ray to run the DenseNet121 classifier and view its Grad-CAM explanation.",
)

uploaded_file = st.file_uploader(
    "Upload a chest X-ray (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"]
)

if uploaded_file is None:
    st.info("Upload an image above to run an analysis.")
    st.stop()

pil_image = Image.open(uploaded_file)

# TorchXRayVision's loader reads from a file path, so persist the upload
# to a temporary file for the segmentation step.
suffix = Path(uploaded_file.name).suffix or ".png"
with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
    pil_image.convert("RGB").save(tmp.name)
    tmp_path = tmp.name

col_img, col_pred = st.columns([1, 1])

with col_img:
    st.markdown("#### Original X-ray")
    st.image(pil_image, use_container_width=True)

with st.spinner("Running DenseNet121 inference..."):
    model = load_model()
    display_array, model_input = preprocess_image(pil_image)
    probability = predict_pneumonia_probability(model, model_input)
    prediction = classify(probability)

with col_pred:
    st.markdown("#### Prediction")
    st.markdown(
        f'<div class="pneumo-card">'
        f'<div style="font-size:2.4rem; font-weight:700;">{probability * 100:.1f}%</div>'
        f'<div style="color:#93A3BE; margin-bottom:0.9rem;">Pneumonia probability</div>'
        f"{prediction_badge(prediction)}"
        f'<div style="color:#93A3BE; margin-top:0.9rem; font-size:0.85rem;">'
        f"Classification threshold: {CLASSIFICATION_THRESHOLD:.2f} (selected on validation data, not 0.5)"
        f"</div></div>",
        unsafe_allow_html=True,
    )

st.markdown("---")
st.markdown("#### Visual Explanation")

with st.spinner("Generating Grad-CAM and approximate lung reference..."):
    heatmap = make_gradcam_heatmap(model_input, model)
    heatmap_224 = resize_heatmap(heatmap)
    gradcam_overlay = overlay_heatmap_on_image(display_array, heatmap_224)

    seg_model = load_segmentation_model()
    lung_mask_512 = get_approximate_lung_mask(tmp_path, seg_model)
    lung_mask_224 = resize_mask_to_224(lung_mask_512)

    # Grad-CAM + lung boundary combined view
    lung_overlay = overlay_heatmap_on_image(display_array, heatmap_224, alpha=0.35)
    lung_tint = display_array.copy()
    lung_tint[lung_mask_224 > 0] = (
        0.6 * lung_tint[lung_mask_224 > 0] + 0.4 * np.array([34, 197, 94])
    ).astype(np.uint8)

viz_cols = st.columns(4)
with viz_cols[0]:
    st.image(display_array, caption="Original X-ray", use_container_width=True)
with viz_cols[1]:
    st.image(gradcam_overlay, caption="Grad-CAM", use_container_width=True)
with viz_cols[2]:
    st.image(lung_tint, caption="Approximate Lung Reference", use_container_width=True)
with viz_cols[3]:
    st.image(lung_overlay, caption="Grad-CAM + Lung Reference", use_container_width=True)

st.markdown("---")
st.markdown("#### Explainability Metrics")

overlap = calculate_overlap_metrics(heatmap_224, lung_mask_224)

metric_cols = st.columns(3)
with metric_cols[0]:
    st.markdown(
        metric_card("Attention inside approx. lung", f"{overlap['attention_inside_lung_pct']:.1f}%"),
        unsafe_allow_html=True,
    )
with metric_cols[1]:
    st.markdown(
        metric_card("High-attention overlap", f"{overlap['high_attention_inside_lung_pct']:.1f}%"),
        unsafe_allow_html=True,
    )
with metric_cols[2]:
    st.markdown(
        metric_card("Approx. lung coverage", f"{overlap['lung_coverage_pct']:.1f}%"),
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)
st.markdown(
    """
**How to read this:** these numbers describe how much of the model's Grad-CAM
attention falls inside an *approximate* lung region — they are an
anatomical-consistency signal, not a measure of diagnostic correctness. See
the **Explainability** page for what the full 640-image test-set analysis
found about this relationship.
"""
)

disclaimer_box(RESEARCH_DISCLAIMER)
