"""PNEUMO.AI — Model Performance page."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.config import (
    CLASSIFICATION_THRESHOLD,
    CNN_BASELINE_TEST_METRICS,
    DENSENET_CONFUSION_MATRIX,
    DENSENET_TEST_METRICS,
    RESEARCH_DISCLAIMER,
)
from utils.theme import disclaimer_box, inject_global_style, metric_card, page_header

st.set_page_config(page_title="Model Performance | PNEUMO.AI", page_icon=":bar_chart:", layout="wide")
inject_global_style()
page_header(
    "Performance",
    "Model Performance",
    "DenseNet121 held-out test-set results, threshold selected on validation data.",
)

st.caption(
    f"All figures below are **held-out test-set results** (640 images), "
    f"using the locked classification threshold of {CLASSIFICATION_THRESHOLD:.2f}. "
    "They describe performance on this specific test set and should not be read as clinical performance."
)

# ------------------------------------------------------------------
# Headline metrics
# ------------------------------------------------------------------
labels = ["Accuracy", "Precision", "Sensitivity", "Specificity", "F1-score", "ROC-AUC"]
keys = ["accuracy", "precision", "sensitivity", "specificity", "f1_score", "roc_auc"]

cols = st.columns(6)
for c, label, key in zip(cols, labels, keys):
    value = DENSENET_TEST_METRICS[key]
    display_value = f"{value:.4f}" if key == "roc_auc" else f"{value * 100:.2f}%"
    with c:
        st.markdown(metric_card(label, display_value), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("---")

# ------------------------------------------------------------------
# Confusion matrix + CNN comparison
# ------------------------------------------------------------------
col_cm, col_cmp = st.columns([1, 1.3])

with col_cm:
    st.markdown("#### Confusion Matrix")
    cm = DENSENET_CONFUSION_MATRIX
    z = [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]]
    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=["Predicted NORMAL", "Predicted PNEUMONIA"],
            y=["Actual NORMAL", "Actual PNEUMONIA"],
            text=z,
            texttemplate="%{text}",
            textfont={"size": 20, "color": "#E6EDF7"},
            colorscale=[[0, "#121B2E"], [1, "#0EA5C4"]],
            showscale=False,
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#E6EDF7",
        margin=dict(l=10, r=10, t=10, b=10),
        height=340,
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig, use_container_width=True)

with col_cmp:
    st.markdown("#### DenseNet121 vs. CNN Baseline")
    comparison_df = pd.DataFrame(
        {
            "Metric": labels,
            "CNN Baseline": [
                f"{CNN_BASELINE_TEST_METRICS['accuracy'] * 100:.2f}%",
                f"{CNN_BASELINE_TEST_METRICS['precision'] * 100:.2f}%",
                f"{CNN_BASELINE_TEST_METRICS['sensitivity'] * 100:.2f}%",
                f"{CNN_BASELINE_TEST_METRICS['specificity'] * 100:.2f}%",
                f"{CNN_BASELINE_TEST_METRICS['f1_score'] * 100:.2f}%",
                f"{CNN_BASELINE_TEST_METRICS['roc_auc']:.4f}",
            ],
            "DenseNet121": [
                f"{DENSENET_TEST_METRICS['accuracy'] * 100:.2f}%",
                f"{DENSENET_TEST_METRICS['precision'] * 100:.2f}%",
                f"{DENSENET_TEST_METRICS['sensitivity'] * 100:.2f}%",
                f"{DENSENET_TEST_METRICS['specificity'] * 100:.2f}%",
                f"{DENSENET_TEST_METRICS['f1_score'] * 100:.2f}%",
                f"{DENSENET_TEST_METRICS['roc_auc']:.4f}",
            ],
        }
    )
    st.dataframe(comparison_df, use_container_width=True, hide_index=True, height=250)

    fig_bar = go.Figure()
    bar_metrics = ["Accuracy", "Precision", "Sensitivity", "Specificity", "F1-score"]
    bar_keys = ["accuracy", "precision", "sensitivity", "specificity", "f1_score"]
    fig_bar.add_trace(
        go.Bar(
            name="CNN Baseline",
            x=bar_metrics,
            y=[CNN_BASELINE_TEST_METRICS[k] * 100 for k in bar_keys],
            marker_color="#4B5B7A",
        )
    )
    fig_bar.add_trace(
        go.Bar(
            name="DenseNet121",
            x=bar_metrics,
            y=[DENSENET_TEST_METRICS[k] * 100 for k in bar_keys],
            marker_color="#0EA5C4",
        )
    )
    fig_bar.update_layout(
        barmode="group",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#E6EDF7",
        margin=dict(l=10, r=10, t=20, b=10),
        height=280,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        yaxis_title="%",
    )
    st.plotly_chart(fig_bar, use_container_width=True)

st.markdown(
    """
DenseNet121 improves on the CNN baseline across every metric, most notably
ROC-AUC (0.963 vs 0.931) and specificity. The 0.10 threshold was selected by
maximizing F1-score on the validation set, not chosen for any clinical
screening objective. At this operating point, the held-out test set shows
high sensitivity and correspondingly lower specificity (more false
positives). This is a reported property of the F1-maximizing threshold on
this dataset, not the outcome of a clinical threshold-selection study.
"""
)

disclaimer_box(RESEARCH_DISCLAIMER)
