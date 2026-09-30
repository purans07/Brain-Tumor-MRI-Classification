"""Streamlit interface for the trained brain tumor classifier."""

import sys
from pathlib import Path

# Make the project package importable when Streamlit executes this file from
# the ``app`` directory rather than from the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from PIL import Image

from src.config import CLASS_NAMES, MODEL_ROOT
from src.predict import load_model, predict_image

DEFAULT_MODEL = MODEL_ROOT / "efficientnet_b0_finetuned.h5"


@st.cache_resource
def get_model(model_path: str):
    return load_model(model_path)


st.set_page_config(page_title="Brain Tumor MRI Classifier", page_icon="🧠", layout="centered")
st.title("Brain Tumor MRI Classifier")
st.caption("Educational decision-support prototype. This tool is not a medical diagnosis and must not replace a qualified clinician.")

model_path = st.sidebar.text_input("Model path", str(DEFAULT_MODEL))
uploaded = st.file_uploader("Upload a brain MRI image", type=["jpg", "jpeg", "png"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)
    if not Path(model_path).exists():
        st.warning(f"No trained model found at {model_path}. Run the training script first.")
    else:
        result = predict_image(get_model(model_path), image, CLASS_NAMES)
        st.subheader(f"Prediction: {result['class'].replace('_', ' ').title()}")
        st.metric("Confidence", f"{result['confidence']:.1%}")
        st.bar_chart(result["probabilities"])
        st.info("Please have a radiologist or other qualified medical professional review the image and prediction.")
