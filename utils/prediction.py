"""
DenseNet121 model loading and inference.

This mirrors the trained model and preprocessing from
notebooks/pneumonia_detection.ipynb exactly:
  - tf.keras.applications.densenet.preprocess_input
  - input size 224x224x3
  - locked classification threshold of 0.10
"""

from __future__ import annotations

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from keras.layers import Dense

from utils.config import CLASSIFICATION_THRESHOLD, IMAGE_SIZE, MODEL_PATH


@st.cache_resource(show_spinner="Loading DenseNet121 model...")
def load_model() -> tf.keras.Model:
    """Load the trained DenseNet121 classifier."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found at '{MODEL_PATH}'. Make sure "
            "'densenet121_stage2_best.keras' is inside the models/ folder."
        )

    # Compatibility fix for the saved Keras model.
    # The saved configuration contains quantization_config=None,
    # which Keras 3.12 does not accept as a Dense argument.
    original_dense_from_config = Dense.from_config

    Dense.from_config = classmethod(
        lambda cls, config: original_dense_from_config(
            {
                key: value
                for key, value in config.items()
                if key != "quantization_config"
            }
        )
    )

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    return model


def preprocess_image(
    pil_image: Image.Image,
) -> tuple[np.ndarray, np.ndarray]:
    """Resize and apply the exact DenseNet121 preprocessing used in training."""

    rgb_image = pil_image.convert("RGB").resize(
        IMAGE_SIZE,
        resample=Image.NEAREST,
    )

    display_array = np.array(rgb_image).astype(np.uint8)

    model_input = np.expand_dims(
        display_array.astype(np.float32),
        axis=0,
    )

    model_input = tf.keras.applications.densenet.preprocess_input(
        model_input
    )

    return display_array, model_input


def predict_pneumonia_probability(
    model: tf.keras.Model,
    model_input: np.ndarray,
) -> float:
    """Run the DenseNet121 model and return pneumonia probability."""

    probability = float(
        model.predict(model_input, verbose=0)[0][0]
    )

    return probability


def classify(probability: float) -> str:
    """Apply the locked 0.10 validation-set threshold."""

    return (
        "PNEUMONIA"
        if probability >= CLASSIFICATION_THRESHOLD
        else "NORMAL"
    )