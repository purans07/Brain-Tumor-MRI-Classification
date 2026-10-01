"""Streamlit interface for the brain tumor MRI classification prototype."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# Streamlit executes files from the app directory. Add the project root so
# the reusable src package works from both the CLI and the Streamlit runner.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError

from src.config import CLASS_NAMES, MODEL_ROOT, OUTPUT_ROOT
from src.explainability import make_gradcam_heatmap, overlay_gradcam
from src.predict import load_model, predict_image

DEFAULT_MODEL = MODEL_ROOT / "efficientnet_b0_finetuned.h5"
MODEL_LABEL = "EfficientNetB0 fine-tuned"
MODEL_VERSION = "1.0.0"
MODEL_INPUT = "224 × 224 RGB"
TEST_ACCURACY = 0.6911
TEST_MACRO_F1 = 0.6770


def load_benchmark_metrics() -> tuple[float, float]:
    """Use freshly generated evaluation metrics when they are available."""

    metrics_path = OUTPUT_ROOT / "model_results" / "efficientnet_b0_finetuned" / "summary_metrics.csv"
    if metrics_path.exists():
        try:
            metrics = pd.read_csv(metrics_path).iloc[0]
            return float(metrics["accuracy"]), float(metrics["macro_f1"])
        except (KeyError, IndexError, ValueError, OSError):
            pass
    return TEST_ACCURACY, TEST_MACRO_F1


BENCHMARK_ACCURACY, BENCHMARK_MACRO_F1 = load_benchmark_metrics()


st.set_page_config(
    page_title="NeuroScan MRI Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner="Loading the trained model…")
def get_model(model_path: str):
    """Load the model once per Streamlit session and path."""

    return load_model(model_path)


def display_name(value: str) -> str:
    return value.replace("_", " ").title()


def confidence_label(confidence: float) -> str:
    if confidence >= 0.80:
        return "High confidence"
    if confidence >= 0.60:
        return "Moderate confidence"
    return "Low confidence"


def image_quality_report(image: Image.Image) -> dict:
    """Return lightweight quality indicators before a scan is interpreted."""

    rgb = image.convert("RGB")
    grayscale = np.asarray(rgb.convert("L"), dtype=np.float32) / 255.0
    mean_brightness = float(grayscale.mean())
    contrast = float(grayscale.std())
    warnings = []
    width, height = rgb.size
    if min(width, height) < 224:
        warnings.append("The image is smaller than the model input and will be resized.")
    if mean_brightness < 0.08 or mean_brightness > 0.92:
        warnings.append("Brightness is unusually extreme; verify that the scan is visible and correctly exported.")
    if contrast < 0.045:
        warnings.append("Contrast is very low; the scan may be blank, compressed, or poorly exposed.")
    return {
        "width": width,
        "height": height,
        "brightness": mean_brightness,
        "contrast": contrast,
        "warnings": warnings,
    }


def render_probability_bars(probabilities: dict[str, float]) -> None:
    """Render compact, readable probability bars without a charting dependency."""

    rows = []
    for label, value in sorted(probabilities.items(), key=lambda item: item[1], reverse=True):
        rows.append(
            f"""
            <div class="prob-row">
              <div class="prob-label"><span>{display_name(label)}</span><strong>{value:.1%}</strong></div>
              <div class="prob-track"><div class="prob-fill" style="width:{value * 100:.2f}%"></div></div>
            </div>
            """
        )
    st.markdown("".join(rows), unsafe_allow_html=True)


def prediction_record(filename: str, result: dict) -> dict:
    return {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "file": filename,
        "prediction": display_name(result["class"]),
        "confidence": result["confidence"],
        "confidence_band": confidence_label(result["confidence"]),
    }


def ensure_history() -> None:
    if "scan_history" not in st.session_state:
        st.session_state.scan_history = []


def model_is_ready(model_path: str) -> bool:
    if Path(model_path).exists():
        return True
    st.error("The selected model file was not found. Train a model or choose a valid checkpoint in the sidebar.")
    return False


def analyze_image(uploaded_file, model):
    try:
        uploaded_file.seek(0)
        image = Image.open(uploaded_file).convert("RGB")
    except (UnidentifiedImageError, OSError):
        st.error("This file could not be read as an image. Please upload a valid JPG, JPEG, or PNG file.")
        return None, None
    return image, predict_image(model, image, CLASS_NAMES)


def render_result(result: dict, filename: str, image: Image.Image, model) -> None:
    predicted = display_name(result["class"])
    confidence = result["confidence"]
    band = confidence_label(confidence)

    st.markdown(
        f"""
        <div class="result-card">
          <div class="result-eyebrow">MODEL OUTPUT</div>
          <div class="result-title">{predicted}</div>
          <div class="result-meta">{band} · {filename}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_col, meter_col = st.columns([1, 2])
    with metric_col:
        st.metric("Top probability", f"{confidence:.1%}")
    with meter_col:
        st.caption("Confidence is a model probability, not a clinical risk score.")
        st.progress(confidence)

    st.markdown("#### Class probabilities")
    render_probability_bars(result["probabilities"])

    quality = image_quality_report(image)
    with st.expander("Image quality and scan details"):
        q1, q2, q3 = st.columns(3)
        q1.metric("Resolution", f"{quality['width']} × {quality['height']} px")
        q2.metric("Brightness", f"{quality['brightness']:.1%}")
        q3.metric("Contrast", f"{quality['contrast']:.1%}")
        if quality["warnings"]:
            for warning in quality["warnings"]:
                st.warning(warning)
        else:
            st.success("Basic image-quality checks passed. This does not confirm clinical suitability.")

    show_attention = st.checkbox("Show model attention map", key="show_gradcam", help="Grad-CAM highlights image regions that influenced the model. It is not a lesion boundary or a diagnosis.")
    if show_attention:
        try:
            with st.spinner("Generating attention map…"):
                heatmap, _, _ = make_gradcam_heatmap(
                    np.asarray(image.resize((224, 224)), dtype=np.float32)[None, ...],
                    model,
                )
                attention = overlay_gradcam(image, heatmap)
            st.image(attention, caption="Grad-CAM attention map — explanatory only", use_container_width=True)
        except Exception as exc:
            st.info(f"The attention map is unavailable for this checkpoint: {exc}")

    if confidence < 0.60:
        st.warning("The model is uncertain. Treat this output as a screening aid only and request qualified clinical review.")
    else:
        st.info("Review the original MRI and the full clinical context with a qualified radiologist. This prototype does not make a diagnosis.")

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "file": filename,
        "model": MODEL_LABEL,
        "prediction": result["class"],
        "confidence": confidence,
        "probabilities": result["probabilities"],
        "image": {
            "width": quality["width"],
            "height": quality["height"],
            "brightness": quality["brightness"],
            "contrast": quality["contrast"],
            "quality_warnings": quality["warnings"],
        },
        "medical_disclaimer": "Educational prototype only. Not a medical diagnosis or clinical decision system.",
    }
    st.download_button(
        "Download scan report",
        data=json.dumps(report, indent=2),
        file_name=f"{Path(filename).stem}_scan_report.json",
        mime="application/json",
        use_container_width=True,
    )


def render_sidebar() -> str:
    with st.sidebar:
        st.markdown("<div class='sidebar-brand'><span class='brand-mark'>✦</span><span>NeuroScan</span></div>", unsafe_allow_html=True)
        st.caption("MRI classification workspace")
        st.divider()
        model_path = str(DEFAULT_MODEL)
        with st.expander("Model information"):
            st.caption(f"{MODEL_LABEL} · version {MODEL_VERSION}")
            st.caption(f"Input: {MODEL_INPUT}")
            st.caption("The production checkpoint is managed by the application.")

        st.markdown("**Workspace status**")
        if Path(model_path).exists():
            st.success("Model ready", icon="✅")
        else:
            st.warning("Model path unavailable", icon="⚠️")
        st.caption("Session-only analysis · uploads are not written to disk by this app")

        st.divider()
        if st.button("Clear session history", use_container_width=True):
            st.session_state.scan_history = []
            st.session_state.pop("last_result", None)
            st.rerun()
        st.caption("NeuroScan v1.0 · Educational prototype")
    return model_path


def main() -> None:
    ensure_history()
    model_path = render_sidebar()

    st.markdown(
        """
        <div class="hero">
          <div class="hero-kicker">NEUROIMAGING WORKSPACE</div>
          <h1>Brain MRI classifier</h1>
          <p>Organize an image review, inspect model probabilities, and export a lightweight analysis record.</p>
          <div class="hero-tags"><span>4 classes</span><span>224 × 224 input</span><span>EfficientNetB0</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Model accuracy", f"{BENCHMARK_ACCURACY:.2%}", help="Held-out test accuracy from the project evaluation.")
    kpi2.metric("Macro F1", f"{BENCHMARK_MACRO_F1:.3f}", help="Macro F1 across the four folder-defined classes.")
    kpi3.metric("Classes", "4")
    kpi4.metric("Input format", "JPG · PNG")

    st.markdown("<div class='notice'><strong>Clinical safety notice</strong><br>NeuroScan is an educational decision-support prototype. It is not a medical device, does not estimate patient risk, and must not replace radiologist review.</div>", unsafe_allow_html=True)

    single_tab, batch_tab, history_tab, about_tab = st.tabs(["Single scan", "Batch scan", "History", "Model & safety"])

    with single_tab:
        left, right = st.columns([1.05, 1.35], gap="large")
        with left:
            st.markdown("### Start a scan")
            st.caption("Upload one brain MRI image for a focused prediction report.")
            uploaded = st.file_uploader("MRI image", type=["jpg", "jpeg", "png"], key="single_upload", label_visibility="collapsed")
            analyze = st.button("Analyze image", type="primary", use_container_width=True, disabled=uploaded is None)
            st.caption("Maximum file size follows the Streamlit server configuration.")
            if uploaded is not None:
                try:
                    uploaded.seek(0)
                    preview = Image.open(uploaded).convert("RGB")
                    st.image(preview, caption=uploaded.name, use_container_width=True)
                    quality = image_quality_report(preview)
                    st.caption(f"{quality['width']} × {quality['height']} px · {uploaded.size / 1024:.1f} KB")
                except (UnidentifiedImageError, OSError):
                    st.error("Preview unavailable for this file.")

        with right:
            st.markdown("### Review result")
            if analyze and uploaded is not None and model_is_ready(model_path):
                _, result = analyze_image(uploaded, get_model(model_path))
                if result is not None:
                    st.session_state.last_result = result
                    st.session_state.last_filename = uploaded.name
                    uploaded.seek(0)
                    st.session_state.last_image = Image.open(uploaded).convert("RGB")
                    st.session_state.scan_history.insert(0, prediction_record(uploaded.name, result))
            if st.session_state.get("last_result"):
                render_result(
                    st.session_state.last_result,
                    st.session_state.get("last_filename", "uploaded image"),
                    st.session_state.get("last_image", Image.new("RGB", (224, 224))),
                    get_model(model_path),
                )
            else:
                st.markdown("<div class='empty-state'><div class='empty-icon'>◌</div><strong>Your result will appear here</strong><span>Upload an image and choose Analyze image to begin.</span></div>", unsafe_allow_html=True)

    with batch_tab:
        st.markdown("### Batch review")
        st.caption("Run the same model across multiple images and download a CSV summary.")
        batch_files = st.file_uploader("MRI images", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="batch_upload")
        run_batch = st.button("Run batch analysis", type="primary", disabled=not batch_files)
        if run_batch and batch_files and model_is_ready(model_path):
            model = get_model(model_path)
            rows = []
            progress = st.progress(0, text="Analyzing images…")
            for index, uploaded_file in enumerate(batch_files, start=1):
                _, result = analyze_image(uploaded_file, model)
                if result is not None:
                    rows.append(prediction_record(uploaded_file.name, result))
                    st.session_state.scan_history.insert(0, rows[-1])
                progress.progress(index / len(batch_files), text=f"Analyzed {index} of {len(batch_files)}")
            progress.empty()
            if rows:
                st.session_state.batch_results = pd.DataFrame(rows)

        if "batch_results" in st.session_state:
            batch_df = st.session_state.batch_results.copy()
            batch_df["confidence"] = batch_df["confidence"].map(lambda value: f"{value:.1%}")
            st.dataframe(batch_df, use_container_width=True, hide_index=True)
            csv = st.session_state.batch_results.to_csv(index=False).encode("utf-8")
            st.download_button("Download batch CSV", csv, "neuroscan_batch_results.csv", "text/csv", use_container_width=True)

    with history_tab:
        st.markdown("### Session history")
        st.caption("History is kept only in the current browser session and is cleared when you reset it.")
        if st.session_state.scan_history:
            history_df = pd.DataFrame(st.session_state.scan_history)
            history_df["confidence"] = history_df["confidence"].map(lambda value: f"{value:.1%}")
            st.dataframe(history_df, use_container_width=True, hide_index=True)
        else:
            st.markdown("<div class='empty-state compact'><div class='empty-icon'>⌁</div><strong>No scans yet</strong><span>Completed single or batch scans will appear here.</span></div>", unsafe_allow_html=True)

    with about_tab:
        info_left, info_right = st.columns(2, gap="large")
        with info_left:
            st.markdown("### Model card")
            st.markdown(
                f"""
                **Architecture**<br>
                {MODEL_LABEL} · version {MODEL_VERSION}<br><br>
                **Input**<br>
                {MODEL_INPUT}<br><br>
                **Classes**<br>
                {', '.join(display_name(name) for name in CLASS_NAMES)}<br><br>
                **Held-out test metrics**<br>
                Accuracy: **{BENCHMARK_ACCURACY:.2%}** · Macro F1: **{BENCHMARK_MACRO_F1:.3f}**
                """,
                unsafe_allow_html=True,
            )
        with info_right:
            st.markdown("### Safe use checklist")
            st.markdown(
                """
                1. Use the output as a research demonstration only.
                2. Review the original scan and patient context independently.
                3. Ask a qualified radiologist to interpret any clinical image.
                4. Validate externally before considering any operational use.
                """
            )
            st.warning("Model probabilities can be wrong or overconfident, especially on images from another scanner, population, or acquisition protocol.")


st.markdown(
    """
    <style>
    :root { --ink: #172033; --muted: #64748b; --line: #e2e8f0; --blue: #2563eb; --teal: #0f766e; }
    .block-container { padding-top: 2rem; padding-bottom: 4rem; max-width: 1400px; }
    [data-testid="stSidebar"] { background: #f8fafc; border-right: 1px solid #e2e8f0; }
    [data-testid="stSidebar"] .block-container { padding-top: 2rem; }
    .sidebar-brand { display: flex; align-items: center; gap: .55rem; color: #172033; font-size: 1.35rem; font-weight: 800; letter-spacing: -.03em; }
    .brand-mark { display: inline-flex; width: 2rem; height: 2rem; align-items: center; justify-content: center; border-radius: .7rem; color: white; background: linear-gradient(135deg, #2563eb, #14b8a6); }
    .hero { padding: 2.1rem 2.3rem 2rem; margin-bottom: 1.2rem; border: 1px solid #dbeafe; border-radius: 1.25rem; background: radial-gradient(circle at 90% 0%, #dbeafe 0%, #eff6ff 36%, #ffffff 78%); }
    .hero-kicker { color: #2563eb; font-size: .74rem; font-weight: 800; letter-spacing: .14em; }
    .hero h1 { margin: .4rem 0 .35rem; color: #172033; font-size: clamp(2rem, 4vw, 3.25rem); letter-spacing: -.055em; line-height: 1; }
    .hero p { max-width: 700px; margin: 0; color: #475569; font-size: 1.04rem; }
    .hero-tags { display: flex; flex-wrap: wrap; gap: .5rem; margin-top: 1.3rem; }
    .hero-tags span { padding: .35rem .65rem; border: 1px solid #bfdbfe; border-radius: 99px; color: #1d4ed8; background: rgba(255,255,255,.72); font-size: .78rem; font-weight: 700; }
    .notice { margin: 1.1rem 0 1.5rem; padding: .9rem 1.1rem; border-left: 4px solid #f59e0b; border-radius: .6rem; color: #713f12; background: #fffbeb; font-size: .9rem; }
    .result-card { padding: 1.2rem 1.3rem; margin: .45rem 0 1rem; border: 1px solid #bbf7d0; border-radius: 1rem; background: linear-gradient(135deg, #f0fdf4, #ffffff); }
    .result-eyebrow { color: #15803d; font-size: .7rem; font-weight: 800; letter-spacing: .14em; }
    .result-title { margin-top: .25rem; color: #14532d; font-size: 2rem; font-weight: 800; letter-spacing: -.04em; }
    .result-meta { color: #64748b; font-size: .82rem; }
    .prob-row { margin: .75rem 0; }
    .prob-label { display: flex; justify-content: space-between; margin-bottom: .28rem; color: #334155; font-size: .88rem; }
    .prob-label strong { color: #172033; }
    .prob-track { height: .55rem; overflow: hidden; border-radius: 99px; background: #e2e8f0; }
    .prob-fill { height: 100%; border-radius: 99px; background: linear-gradient(90deg, #2563eb, #14b8a6); }
    .empty-state { display: flex; min-height: 22rem; flex-direction: column; align-items: center; justify-content: center; gap: .45rem; border: 1px dashed #cbd5e1; border-radius: 1rem; color: #64748b; text-align: center; }
    .empty-state strong { color: #334155; font-size: 1.05rem; }
    .empty-state.compact { min-height: 12rem; }
    .empty-icon { color: #93c5fd; font-size: 3rem; line-height: 1; }
    /* Fixed light application theme; it is independent of browser/system mode. */
    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stHeader"] { background: #f8fafc !important; }
    [data-testid="stAppViewContainer"] .block-container { color: #172033; }
    [data-testid="stAppViewContainer"] h2,
    [data-testid="stAppViewContainer"] h3,
    [data-testid="stAppViewContainer"] h4 { color: #172033 !important; }
    [data-testid="stAppViewContainer"] [data-testid="stCaptionContainer"],
    [data-testid="stAppViewContainer"] [data-testid="stCaptionContainer"] * { color: #64748b !important; }
    [data-testid="stAppViewContainer"] .stMarkdown p,
    [data-testid="stAppViewContainer"] .stMarkdown li { color: #475569; }
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] .stMarkdown li,
    [data-testid="stSidebar"] .stMarkdown strong { color: #334155 !important; }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4 { color: #172033 !important; }
    .hero p { color: #475569 !important; }
    .hero h1 { color: #172033 !important; }

    /* KPI cards must define both their surface and their foreground colors. */
    div[data-testid="stMetric"] {
      padding: .85rem 1rem;
      border: 1px solid #cbd5e1;
      border-radius: .8rem;
      background: #ffffff;
      color: #172033;
    }
    div[data-testid="stMetric"] [data-testid="stMetricLabel"],
    div[data-testid="stMetric"] [data-testid="stMetricLabel"] *,
    div[data-testid="stMetric"] label {
      color: #475569 !important;
      opacity: 1 !important;
      white-space: normal !important;
      overflow: visible !important;
      text-overflow: clip !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"],
    div[data-testid="stMetric"] [data-testid="stMetricValue"] * {
      color: #172033 !important;
      opacity: 1 !important;
      white-space: normal !important;
      overflow: visible !important;
      text-overflow: clip !important;
      word-break: normal !important;
      font-size: clamp(1.2rem, 2.2vw, 2rem) !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] { overflow: visible !important; }
    div[data-testid="stMetric"] [data-testid="stMetricDelta"],
    div[data-testid="stMetric"] svg { color: #64748b !important; fill: #64748b !important; }

    /* Keep native Streamlit controls aligned with the fixed light theme. */
    [data-testid="stFileUploader"] section,
    [data-testid="stFileUploader"] section > div { border-color: #cbd5e1; background: #f8fafc; }
    [data-testid="stFileUploader"] label,
    [data-testid="stFileUploader"] small,
    [data-testid="stFileUploader"] span { color: #475569 !important; }
    [data-testid="stTabs"] button { color: #475569 !important; }
    [data-testid="stTabs"] button[aria-selected="true"] { color: #ff6b6b !important; }
    [data-testid="stProgress"] p { color: #475569 !important; }
    [data-testid="stAlert"] { color: #172033; }
    [data-testid="stAlert"] p { color: inherit !important; }
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"],
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] * { color: #475569 !important; }
    [data-testid="stSidebar"] [data-testid="stAlert"] { color: #166534; }
    [data-testid="stSidebar"] button[kind="secondary"],
    [data-testid="stSidebar"] button:not([kind="primary"]) {
      color: #172033 !important;
      border: 1px solid #cbd5e1 !important;
      background: #ffffff !important;
    }
    button[kind="primary"] {
      border-radius: .65rem;
      color: #ffffff !important;
      background: #ef4444 !important;
      border-color: #ef4444 !important;
    }
    button[kind="primary"]:hover { background: #dc2626 !important; border-color: #dc2626 !important; }
    @media (max-width: 1200px) {
      div[data-testid="stMetric"] { padding: .7rem .75rem; }
      div[data-testid="stMetric"] [data-testid="stMetricValue"],
      div[data-testid="stMetric"] [data-testid="stMetricValue"] * { font-size: 1.25rem !important; }
    }
    @media (max-width: 800px) {
      .block-container { padding-left: 1rem; padding-right: 1rem; }
      .hero { padding: 1.4rem; }
      .hero h1 { font-size: 2.15rem; }
      .hero p { font-size: .95rem; }
      .notice { font-size: .84rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


if __name__ == "__main__":
    main()
