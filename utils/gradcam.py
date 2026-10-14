"""
Grad-CAM for the DenseNet121 pneumonia classifier.

This is a direct adaptation of the working implementation validated in
notebooks/pneumonia_detection.ipynb (make_gradcam_heatmap). The target
convolutional layer is conv5_block16_2_conv, the last Conv2D layer of the
DenseNet121 backbone. Nothing about the methodology has been changed.
"""

from __future__ import annotations

import cv2
import numpy as np
import tensorflow as tf

from utils.config import IMAGE_SIZE, LAST_CONV_LAYER_NAME


def make_gradcam_heatmap(
    img_array: np.ndarray,
    model: tf.keras.Model,
    last_conv_layer_name: str = LAST_CONV_LAYER_NAME,
) -> np.ndarray:
    """Compute the raw (small, e.g. 7x7) Grad-CAM heatmap for the
    pneumonia class, exactly as implemented in the notebook.

    Args:
        img_array: preprocessed model input, shape (1, 224, 224, 3).
        model: the loaded Sequential([DenseNet121, GAP, Dense(128),
               Dropout, Dense(1, sigmoid)]) model.
        last_conv_layer_name: name of the target Conv2D layer inside the
               DenseNet121 backbone.

    Returns:
        A 2D numpy array (the heatmap at the backbone's spatial
        resolution), normalized to the 0-1 range.
    """
    base_model = model.layers[0]

    # Make sure the backbone is built before we try to fetch a layer from it
    _ = base_model(img_array, training=False)

    last_conv_layer = base_model.get_layer(last_conv_layer_name)

    grad_model = tf.keras.models.Model(
        inputs=base_model.input,
        outputs=[last_conv_layer.output, base_model.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, backbone_output = grad_model(img_array, training=False)

        # Pass the backbone features through the trained classification head
        x = model.layers[1](backbone_output, training=False)
        x = model.layers[2](x, training=False)
        x = model.layers[3](x, training=False)
        predictions = model.layers[4](x, training=False)

        pneumonia_score = predictions[:, 0]

    grads = tape.gradient(pneumonia_score, conv_outputs)

    if grads is None:
        raise RuntimeError(
            "Gradients are None. The prediction is not connected to the "
            "selected convolutional layer."
        )

    pooled_grads = tf.reduce_mean(grads, axis=(1, 2))

    conv_outputs = conv_outputs[0]
    pooled_grads = pooled_grads[0]

    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)

    max_value = tf.reduce_max(heatmap)
    heatmap = tf.where(max_value > 0, heatmap / max_value, heatmap)

    return heatmap.numpy()


def resize_heatmap(heatmap: np.ndarray, size: tuple[int, int] = IMAGE_SIZE) -> np.ndarray:
    """Upsample the raw Grad-CAM heatmap to the image resolution (224x224)."""
    return cv2.resize(heatmap, size, interpolation=cv2.INTER_LINEAR)


def overlay_heatmap_on_image(
    display_array: np.ndarray,
    heatmap_224: np.ndarray,
    alpha: float = 0.4,
) -> np.ndarray:
    """Blend the resized Grad-CAM heatmap (jet colormap) onto the original
    RGB image, matching the notebook's cv2.addWeighted overlay."""
    heatmap_uint8 = np.uint8(255 * heatmap_224)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(display_array, 1 - alpha, heatmap_color, alpha, 0)
    return overlay
