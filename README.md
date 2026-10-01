# Brain Tumor MRI Image Classification

This repository contains an end-to-end TensorFlow/Keras image-classification project for four folder-defined classes: `glioma`, `meningioma`, `no_tumor`, and `pituitary`. It includes real-data profiling, exploratory visualizations, conservative training augmentation, a custom CNN, an EfficientNetB0 transfer-learning model, evaluation utilities, Grad-CAM explainability, a notebook, and a Streamlit demo.

## Dataset facts

The supplied dataset contains 2,443 JPG images in the provided `train`, `valid`, and `test` folders. Every readable image is 640×640 RGB, with no corrupted images or exact duplicate files detected by the included scan. The observed class counts are:

| Class | Images | Share |
|---|---:|---:|
| glioma | 805 | 32.95% |
| meningioma | 545 | 22.31% |
| no_tumor | 483 | 19.77% |
| pituitary | 610 | 24.97% |

The supplied README says “three tumor types” but also lists `No Tumor`; the four actual class folders and CSV labels are used as the source of truth. The dataset README identifies the source as a Roboflow dataset under CC BY 4.0. Confirm licensing and attribution before public redistribution.

## Run the project

Create an environment with Python 3.10–3.12 and install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Generate metadata and the 15 required EDA figures:

```bash
python -m src.analysis
```

Train both models. The EfficientNet builder attempts ImageNet weights and falls back to random initialization only when the weights are unavailable, with a warning. The training path uses conservative MRI augmentation and class-balanced weights derived only from the training split.

```bash
python -m src.train --epochs 10
```

For a stronger EfficientNet run, train the classifier head, fine-tune the top backbone layers, and evaluate only after the final checkpoint is selected:

```bash
python -m src.train --models efficientnet_b0 --epochs 20
python -m src.fine_tune --checkpoint models/efficientnet_b0.best.keras --epochs 8 --layers 40
python -m src.run_evaluation --model models/efficientnet_b0_finetuned.h5
```

Evaluate a saved model from a Python session using `src.data_loader.build_tf_datasets` and `src.evaluate.evaluate_keras_model`. The metrics utility saves accuracy, macro precision, macro recall, macro F1, per-class metrics, and a confusion matrix. The app reads the generated fine-tuned summary automatically when it is available; otherwise it displays the checked benchmark fallback.

Launch the demo after a model has been trained:

```bash
streamlit run app/streamlit_app.py
```

## Outputs

- `outputs/image_metadata.csv`: image-level dimensions, channels, brightness, contrast, split, and duplicate flag.
- `outputs/dataset_summary.json`: measured dataset summary.
- `outputs/figures/`: 15 meaningful charts, including class balance, image properties, galleries, horizontal-flip examples, brightness/contrast examples, intensity distributions, and split composition.
- `outputs/chart_interpretations.csv`: chart rationale, data-derived insight, and practical impact.
- `outputs/model_results/model_comparison.csv`: executed one-epoch test-set comparison.
- `outputs/gradcam/custom_cnn_gradcam_example.png`: example Grad-CAM overlay.
- `models/`: generated `.keras` checkpoints and `.h5` exports after training.
- `notebooks/Brain_Tumor_MRI_Image_Classification.ipynb`: end-to-end submission notebook.

## Streamlit deployment

Streamlit Community Cloud should use the `app/requirements.txt` runtime dependency file and **Python 3.12**. The app entrypoint is `app/streamlit_app.py`. The root `requirements.txt` remains the broader development and notebook environment.

The deployment checkpoint is `models/efficientnet_b0_finetuned.h5`. It is intentionally allowed by `.gitignore` because the deployed app needs the model artifact after GitHub clones the repository. The checkpoint is below GitHub's 100 MB single-file limit; include it in the commit that deploys the app.

## Streamlit features

The NeuroScan interface provides a guided single-image workflow, batch prediction with CSV export, session-only scan history, confidence bands, image-quality checks, scan metadata, optional Grad-CAM attention maps, downloadable JSON reports, a model card, checkpoint validation, and an always-visible clinical safety notice. Launch it with `streamlit run app/streamlit_app.py`.

## Submission assets

- `outputs/brain_tumor_classifier_demo.mp4`: short project walkthrough covering the dataset profile, model comparison, Grad-CAM, and Streamlit deployment.
- `PROJECT_REPORT.md`: requirements reconciliation, measured results, limitations, and future work.

## Safety and limitations

This is an educational prototype, not a medical device or diagnostic system. It does not estimate clinical risk, replace radiologist review, or guarantee generalization to another hospital, scanner, population, or acquisition protocol. A responsible deployment would require external validation, calibration, subgroup analysis, privacy controls, human-in-the-loop review, and clinical/regulatory assessment.
