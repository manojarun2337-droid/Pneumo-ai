"""Shared visual theme and small reusable UI helpers.

Kept deliberately simple (plain CSS injected once per page) so the whole
app stays easy to maintain without a frontend build step.
"""

from __future__ import annotations

import streamlit as st

PRIMARY = "#0EA5C4"      # teal-blue accent
PRIMARY_DARK = "#0B7A97"
ACCENT = "#38BDF8"       # light cyan accent
BG = "#0B1220"           # deep navy background
PANEL = "#121B2E"        # card background
PANEL_BORDER = "#1E2A44"
TEXT = "#E6EDF7"
TEXT_MUTED = "#93A3BE"
GOOD = "#22C55E"
WARN = "#F59E0B"
BAD = "#EF4444"

_CSS = f"""
<style>
    .stApp {{
        background: radial-gradient(circle at 15% 0%, #0F1B33 0%, {BG} 45%);
        color: {TEXT};
    }}
    section[data-testid="stSidebar"] {{
        background-color: #0A1120;
        border-right: 1px solid {PANEL_BORDER};
    }}
    h1, h2, h3, h4 {{
        color: {TEXT};
        font-family: "Helvetica Neue", Arial, sans-serif;
        letter-spacing: 0.2px;
    }}
    p, li, span, label {{
        color: {TEXT};
    }}
    .pneumo-eyebrow {{
        color: {ACCENT};
        text-transform: uppercase;
        letter-spacing: 2.5px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-bottom: 0.35rem;
    }}
    .pneumo-card {{
        background: linear-gradient(180deg, {PANEL} 0%, #0F1729 100%);
        border: 1px solid {PANEL_BORDER};
        border-radius: 14px;
        padding: 1.3rem 1.5rem;
        margin-bottom: 1rem;
    }}
    .pneumo-metric-card {{
        background: linear-gradient(180deg, {PANEL} 0%, #0F1729 100%);
        border: 1px solid {PANEL_BORDER};
        border-radius: 14px;
        padding: 1.1rem 1.2rem;
        text-align: center;
    }}
    .pneumo-metric-value {{
        font-size: 1.9rem;
        font-weight: 700;
        color: {ACCENT};
        line-height: 1.1;
    }}
    .pneumo-metric-label {{
        font-size: 0.82rem;
        color: {TEXT_MUTED};
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 0.25rem;
    }}
    .pneumo-badge {{
        display: inline-block;
        padding: 0.35rem 0.9rem;
        border-radius: 999px;
        font-weight: 700;
        font-size: 1.05rem;
        letter-spacing: 0.5px;
    }}
    .pneumo-badge-pneumonia {{
        background: rgba(239, 68, 68, 0.15);
        color: {BAD};
        border: 1px solid rgba(239, 68, 68, 0.4);
    }}
    .pneumo-badge-normal {{
        background: rgba(34, 197, 94, 0.15);
        color: {GOOD};
        border: 1px solid rgba(34, 197, 94, 0.4);
    }}
    .pneumo-disclaimer {{
        background: rgba(245, 158, 11, 0.08);
        border: 1px solid rgba(245, 158, 11, 0.35);
        border-radius: 10px;
        padding: 0.8rem 1.1rem;
        color: {TEXT_MUTED};
        font-size: 0.88rem;
    }}
    .pneumo-pipeline-step {{
        background: {PANEL};
        border: 1px solid {PANEL_BORDER};
        border-radius: 12px;
        padding: 0.8rem 0.6rem;
        text-align: center;
        font-size: 0.85rem;
        color: {TEXT};
        height: 100%;
    }}
    div[data-testid="stMetricValue"] {{
        color: {ACCENT};
    }}
    hr {{
        border-color: {PANEL_BORDER};
    }}
</style>
"""


def inject_global_style() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


def page_header(eyebrow: str, title: str, subtitle: str = "") -> None:
    st.markdown(f'<div class="pneumo-eyebrow">{eyebrow}</div>', unsafe_allow_html=True)
    st.markdown(f"## {title}")
    if subtitle:
        st.markdown(f'<p style="color:{TEXT_MUTED}; margin-top:-0.4rem;">{subtitle}</p>', unsafe_allow_html=True)
    st.markdown("---")


def metric_card(label: str, value: str) -> str:
    return (
        f'<div class="pneumo-metric-card">'
        f'<div class="pneumo-metric-value">{value}</div>'
        f'<div class="pneumo-metric-label">{label}</div>'
        f"</div>"
    )


def prediction_badge(prediction: str) -> str:
    css_class = "pneumo-badge-pneumonia" if prediction == "PNEUMONIA" else "pneumo-badge-normal"
    return f'<span class="pneumo-badge {css_class}">{prediction}</span>'


def disclaimer_box(text: str) -> None:
    st.markdown(f'<div class="pneumo-disclaimer">{text}</div>', unsafe_allow_html=True)
