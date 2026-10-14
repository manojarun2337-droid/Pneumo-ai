"""PNEUMO.AI — Dataset & Methodology page."""

import streamlit as st

from utils.config import DATASET_SPLITS, RESEARCH_DISCLAIMER, VALIDATION_SPLIT_NOTE
from utils.theme import disclaimer_box, inject_global_style, metric_card, page_header

st.set_page_config(page_title="Dataset & Methodology | PNEUMO.AI", page_icon=":test_tube:", layout="wide")
inject_global_style()
page_header(
    "Methodology",
    "Dataset & Methodology",
    "Where the data came from, how it was split, and the full experimental pipeline.",
)

# ------------------------------------------------------------------
# Dataset composition
# ------------------------------------------------------------------
st.markdown("### Dataset composition")

split_cols = st.columns(3)
with split_cols[0]:
    st.markdown(metric_card("Training images", f"{DATASET_SPLITS['train']['total']:,}"), unsafe_allow_html=True)
    st.caption(
        f"NORMAL: {DATASET_SPLITS['train']['NORMAL']:,} · "
        f"PNEUMONIA: {DATASET_SPLITS['train']['PNEUMONIA']:,}"
    )
with split_cols[1]:
    st.markdown(metric_card("Validation images", f"{DATASET_SPLITS['validation']['total']:,}"), unsafe_allow_html=True)
    st.caption("15% split carved out of the training set")
with split_cols[2]:
    st.markdown(metric_card("Test images", f"{DATASET_SPLITS['test']['total']:,}"), unsafe_allow_html=True)
    st.caption(
        f"NORMAL: {DATASET_SPLITS['test']['NORMAL']:,} · "
        f"PNEUMONIA: {DATASET_SPLITS['test']['PNEUMONIA']:,}"
    )

st.info(VALIDATION_SPLIT_NOTE)

st.markdown("---")

# ------------------------------------------------------------------
# Methodology pipeline
# ------------------------------------------------------------------
st.markdown("### Experimental pipeline")

pipeline_steps = [
    "Dataset",
    "Train / Validation / Test split",
    "CNN baseline",
    "DenseNet121 transfer learning",
    "Fine-tuning (last 50 layers unfrozen)",
    "Validation threshold selection",
    "Held-out test evaluation",
    "Grad-CAM",
    "PSPNet approximate lung reference",
    "Overlap analysis",
    "Statistical analysis",
]

for i, step in enumerate(pipeline_steps):
    arrow = "&nbsp;&nbsp;&nbsp;&nbsp;↓" if i > 0 else ""
    if arrow:
        st.markdown(f'<div style="color:#93A3BE; margin: 0.1rem 0;">{arrow}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="pneumo-card" style="padding:0.7rem 1.2rem;">{step}</div>', unsafe_allow_html=True)

st.markdown("---")

# ------------------------------------------------------------------
# Model details
# ------------------------------------------------------------------
st.markdown("### Model architecture")
st.markdown(
    """
```
DenseNet121 (ImageNet weights)
  → GlobalAveragePooling2D
  → Dense(128, activation="relu")
  → Dropout(0.5)
  → Dense(1, activation="sigmoid")
```

- **Input:** 224 × 224 × 3, preprocessed with `tf.keras.applications.densenet.preprocess_input`
- **Stage 1:** backbone frozen, head trained from scratch
- **Stage 2 (fine-tuning):** last 50 of 427 DenseNet121 layers unfrozen, trained at a
  reduced learning rate (1e-5)
- **Threshold:** 0.10, selected by maximizing F1-score on the validation set
- **Grad-CAM target layer:** `conv5_block16_2_conv` (last convolutional layer of the backbone)
- **Approximate lung reference:** TorchXRayVision's pretrained chest-X-ray PSPNet
  (left + right lung channels combined), not fine-tuned on this dataset
"""
)

disclaimer_box(RESEARCH_DISCLAIMER)
