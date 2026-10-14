"""
Approximate lung-region reference using TorchXRayVision's pretrained PSPNet.

IMPORTANT — read before touching this file:
This is NOT a pneumonia segmentation model, and it was NOT trained on this
project's dataset. It is a general anatomical chest X-ray segmentation
model used only to obtain a rough "where are the lungs" reference so that
Grad-CAM attention can be checked for anatomical plausibility. The output
must always be labeled "Approximate Lung Reference" in the UI, never
"ground truth" or "segmentation mask".

Adapted directly from the working implementation in
notebooks/pneumonia_detection.ipynb (calculate_attention_overlap /
cells using xrv.baseline_models.chestx_det.PSPNet()).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch
import torchvision

# Left Lung = channel 4, Right Lung = channel 5 in PSPNet's target list
LEFT_LUNG_CHANNEL = 4
RIGHT_LUNG_CHANNEL = 5
PSPNET_INPUT_SIZE = 512


@st.cache_resource(show_spinner="Loading anatomical reference model...")
def load_segmentation_model():
    """Load TorchXRayVision's pretrained chest-X-ray PSPNet once."""
    import torchxrayvision as xrv  # imported lazily: heavy, torch-based

    seg_model = xrv.baseline_models.chestx_det.PSPNet()
    seg_model.eval()
    return seg_model


def _load_and_preprocess_for_pspnet(image_path: str) -> np.ndarray:
    """Reproduce the notebook's TorchXRayVision preprocessing pipeline:
    load_image -> XRayCenterCrop -> XRayResizer(512)."""
    import torchxrayvision as xrv

    img = xrv.utils.load_image(image_path)

    transform = torchvision.transforms.Compose(
        [xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(PSPNET_INPUT_SIZE)]
    )
    return transform(img)


def get_approximate_lung_mask(image_path: str, seg_model) -> np.ndarray:
    """Run PSPNet and return an approximate binary lung mask at 512x512
    (left + right lung channels combined, sigmoid > 0.5).

    `image_path` must be a path on disk (TorchXRayVision's loader reads
    from a file path, exactly as in the notebook), so uploaded files are
    written to a temporary path before calling this function.
    """
    processed = _load_and_preprocess_for_pspnet(image_path)
    tensor = torch.from_numpy(processed).unsqueeze(0)

    try:
        device = next(seg_model.parameters()).device
        tensor = tensor.to(device)
    except StopIteration:
        pass

    with torch.no_grad():
        output = seg_model(tensor)

    left_logits = output[0, LEFT_LUNG_CHANNEL].detach().cpu().numpy()
    right_logits = output[0, RIGHT_LUNG_CHANNEL].detach().cpu().numpy()

    left_prob = 1 / (1 + np.exp(-left_logits))
    right_prob = 1 / (1 + np.exp(-right_logits))

    lung_prob = np.maximum(left_prob, right_prob)
    lung_mask = (lung_prob > 0.5).astype(np.float32)

    return lung_mask


def resize_mask_to_224(lung_mask_512: np.ndarray, size: tuple[int, int] = (224, 224)) -> np.ndarray:
    """Resize the 512x512 PSPNet mask down to the 224x224 DenseNet/Grad-CAM
    resolution, using nearest-neighbor to keep it binary."""
    return cv2.resize(lung_mask_512, size, interpolation=cv2.INTER_NEAREST)
