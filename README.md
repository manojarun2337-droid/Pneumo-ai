# PNEUMO.AI
**An Explainable Deep Learning System for Pneumonia Detection from Chest X-Rays**

A local Streamlit application wrapping an already-trained DenseNet121 pneumonia
classifier with Grad-CAM explainability and an approximate anatomical lung
reference, for anatomical-consistency analysis of the model's attention.

> All model experimentation (training, threshold selection, explainability
> analysis) was already completed in Google Colab — see
> `notebooks/pneumonia_detection.ipynb`. This application only loads the final
> model and reproduces the tested methodology; it does not retrain, fine-tune,
> or re-derive any of the published results.

---

## 1. Project structure

```
AI_Pneumonia_App/
├── app.py                          # Home page (run this with `streamlit run`)
├── pages/
│   ├── 1_Pneumonia_Detection.py    # Upload + predict + Grad-CAM + lung reference
│   ├── 2_Explainability.py         # Grad-CAM/lung-overlap concepts + findings
│   ├── 3_Model_Performance.py      # Test-set metrics, confusion matrix, CNN comparison
│   ├── 4_Dataset_and_Methodology.py
│   └── 5_About_and_Limitations.py
├── utils/
│   ├── config.py                   # All locked constants/results — single source of truth
│   ├── prediction.py                # Model loading + DenseNet121 inference
│   ├── gradcam.py                   # Grad-CAM heatmap + overlay
│   ├── segmentation.py              # TorchXRayVision PSPNet approximate lung reference
│   ├── metrics.py                   # Explainability overlap metric calculations
│   └── theme.py                     # Shared CSS + small UI helpers
├── models/
│   └── densenet121_stage2_best.keras   # Final trained model (already provided)
├── data/
│   └── densenet_gradcam_lung_overlap_full.csv   # Full 640-image analysis (already provided)
├── visuals/
│   └── representative_gradcam_cases.png
├── notebooks/
│   └── pneumonia_detection.ipynb   # Source of truth for all methodology
├── sample_images/                  # (optional) put a few test X-rays here for quick demos
├── .streamlit/config.toml          # Dark theme
├── requirements.txt
└── README.md
```

## 2. Installation (Windows, CPU-only)

```bat
cd AI_Pneumonia_App
python -m venv venv
venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Windows-specific notes / likely issues

- **PyTorch**: the default `pip install torch` on Windows should already pull a
  CPU-only wheel, but if it tries to pull a CUDA build (very large download) or
  fails, install explicitly from the CPU index:
  ```bat
  pip install torch --index-url https://download.pytorch.org/whl/cpu
  ```
- **TensorFlow**: `tensorflow-cpu` is not published for all Windows/Python
  combinations. If `pip install tensorflow-cpu` fails, use the regular
  `tensorflow` package instead (it runs fine on CPU when no GPU is present,
  just with a larger install size):
  ```bat
  pip install tensorflow
  ```
- **torchxrayvision**: pulls in `scikit-image` and `pydicom`; if it fails to
  build on Windows, upgrading `pip` and `setuptools` first usually resolves it.
- **opencv-python-headless**: used instead of `opencv-python` to avoid a GUI
  dependency conflict; don't install both in the same environment.
- **NumPy**: kept below 2.0 in `requirements.txt` — some TensorFlow/Torch
  builds you may already have installed are not yet compatible with NumPy 2.x.
- First run will download PSPNet's pretrained weights (TorchXRayVision) if not
  already cached — this requires an internet connection the first time only.

## 3. Running the app

```bat
streamlit run app.py
```

Then open the URL Streamlit prints (typically `http://localhost:8501`).

## 4. What was verified

The following were checked directly against `notebooks/pneumonia_detection.ipynb`
and/or against the actual provided model file before this app was written:

- Model architecture, input size, and preprocessing function
- The locked classification threshold (0.10) and the exact final test-set
  metrics and confusion matrix
- The Grad-CAM implementation and target layer (`conv5_block16_2_conv`) —
  re-run end-to-end against the real model file in this environment
- The PSPNet lung-channel indices (left=4, right=5) and mask logic
- The CSV column schema for the 640-image explainability analysis
- The reported Spearman correlations and per-class median attention values

The one part that could not be executed in the build environment (no GPU/no
torch install available there) is a live run of the PSPNet segmentation
step — its code is a line-for-line adaptation of the notebook's tested
implementation, but please do a manual smoke test the first time you run the
app locally with a real X-ray.

## 5. Remaining things to check on your machine

1. Run `streamlit run app.py`, go to **Pneumonia Detection**, and upload a real
   chest X-ray to confirm inference + Grad-CAM + the approximate lung
   reference all render correctly end-to-end.
2. Confirm the first-run PSPNet weight download succeeds (needs internet once).
3. If you add your own images to `sample_images/`, they are not wired into
   the UI automatically — the app only analyzes uploads. Wire in a "try a
   sample image" selector there if you'd like that for a live demo.
4. Streamlit's `use_container_width` parameter (used throughout the app) is
   being phased out in favor of `width=`. It still works as of Streamlit
   1.4x, but if you upgrade Streamlit and see deprecation warnings, that's
   expected and cosmetic only.
