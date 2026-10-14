"""
Explainability / anatomical-consistency metrics.

Definitions (unchanged from the notebook's calculate_attention_overlap):

  attention_inside_lung_pct:
      sum(Grad-CAM * lung_mask) / sum(Grad-CAM) * 100

  high_attention_inside_lung_pct:
      percentage of the top-20% Grad-CAM pixels that fall inside the
      approximate lung region

  lung_coverage_pct:
      percentage of image pixels the PSPNet reference classifies as lung

These are explainability / anatomical-consistency signals, not clinical
diagnostic metrics and not a measure of model reliability.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from utils.config import OVERLAP_CSV_PATH


def calculate_overlap_metrics(heatmap_224: np.ndarray, lung_mask_224: np.ndarray) -> dict:
    """Compute the three explainability metrics for one Grad-CAM heatmap
    and its matching approximate lung mask (both 224x224)."""
    cam = heatmap_224.astype(np.float32)
    mask = lung_mask_224.astype(np.float32)

    total_attention = cam.sum()
    lung_attention = (cam * mask).sum()
    attention_inside_lung = (
        float(lung_attention / total_attention * 100) if total_attention > 0 else 0.0
    )

    cam_threshold = np.percentile(cam, 80)
    high_attention = cam >= cam_threshold
    total_high_attention = high_attention.sum()
    high_attention_inside = (high_attention & (mask > 0)).sum()
    high_attention_lung_pct = (
        float(high_attention_inside / total_high_attention * 100)
        if total_high_attention > 0
        else 0.0
    )

    lung_coverage_pct = float(mask.mean() * 100)

    return {
        "attention_inside_lung_pct": attention_inside_lung,
        "high_attention_inside_lung_pct": high_attention_lung_pct,
        "lung_coverage_pct": lung_coverage_pct,
    }


@st.cache_data(show_spinner=False)
def load_overlap_analysis() -> pd.DataFrame:
    """Load the pre-computed 640-image test-set explainability analysis.

    This is the published result set from the notebook — it is displayed
    as-is and never recalculated by the app.
    """
    if not OVERLAP_CSV_PATH.exists():
        raise FileNotFoundError(
            f"Explainability results file not found at '{OVERLAP_CSV_PATH}'."
        )
    return pd.read_csv(OVERLAP_CSV_PATH)
