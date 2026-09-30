# Brain Tumor MRI Image Classification Project Report

## Project summary

This project builds a four-class brain MRI image classifier from the supplied folder-labeled dataset. The workflow follows the Labmentix submission template: data understanding, image-level profiling, meaningful visualization, preprocessing, model construction, evaluation planning, explainability, deployment, and conclusion. The implementation preserves the provided train, validation, and test split and uses the folder names as labels.

## Requirements reconciliation

The project brief requests a custom CNN, transfer learning, augmentation, statistical analysis, model comparison, explainable AI, and Streamlit. The sample notebook additionally requires a structured “Know Your Data” section, at least 15 meaningful charts with rationale and impact, hypothesis-testing discussion, preprocessing decisions, model evaluation metrics, model selection, feature/model explainability, saved artifacts, and a conclusion. The notebook in `notebooks/` covers these sections for image data. Text-only preprocessing and tabular feature-selection sections are explicitly marked as not applicable to this image-classification problem.

## Dataset inspection

The real scan found 2,443 JPG images: 1,695 train, 502 validation, and 246 test. It found four classes: `glioma` (805), `meningioma` (545), `no_tumor` (483), and `pituitary` (610). All 2,443 readable images are 640×640 RGB with a 1:1 aspect ratio. The scan found zero corrupted files and zero exact duplicate files using MD5 hashes. Average normalized grayscale brightness is 0.1716 and average within-image grayscale standard deviation is 0.1732.

The class proportions are not equal. Glioma is the largest class at 32.95%, while no tumor is the smallest at 19.77%. The split-level proportions are similar, which supports using the supplied validation and test folders for model comparison. The README included with the dataset says “three tumor types” while listing No Tumor; the observed four folder names and CSV label columns resolve the ambiguity.

## Exploratory analysis

The analysis script creates 15 charts in `outputs/figures/` and stores the explanation for each chart in `outputs/chart_interpretations.csv`. The charts cover univariate image counts and dimensions, bivariate class relationships with brightness, contrast, and aspect ratio, and multivariate/sample views. The sample gallery confirms four visually distinct folder categories for representative images, while the intensity and brightness plots also show that acquisition appearance differs by class. That difference is descriptive, not proof of a causal relationship, and it is a reason to validate on an external dataset.

## Preprocessing and augmentation

Images are resized to 224×224 for efficient batching. The custom CNN rescales inputs to [0, 1]. EfficientNet uses its application preprocessing. Training augmentation uses small horizontal and vertical flips, rotations, zoom, translations, brightness changes, and contrast changes. The augmentation examples are intentionally conservative because clinically meaningful structure should not be changed aggressively. The existing dataset split is retained to avoid leakage from re-splitting the supplied data.

## Models and evaluation

The custom CNN uses four convolutional blocks, batch normalization, global average pooling, and dropout. EfficientNetB0 provides the transfer-learning baseline with a new four-class softmax head and optional fine-tuning of the final layers. Early stopping, learning-rate reduction, and validation-loss checkpoints are used for reproducibility and overfitting control.

Evaluation reports accuracy, macro precision, macro recall, macro F1, per-class precision/recall/F1, and a confusion matrix. Macro recall and macro F1 are especially important here because the class counts differ. The executed one-epoch baselines produced the following test results:

| Model | Accuracy | Macro precision | Macro recall | Macro F1 |
|---|---:|---:|---:|---:|
| Custom CNN one epoch | 0.3821 | 0.1643 | 0.3056 | 0.2079 |
| EfficientNetB0 frozen head | 0.6341 | 0.7086 | 0.6179 | 0.6004 |
| EfficientNetB0 fine-tuned | 0.6911 | 0.7456 | 0.6829 | 0.6770 |

Fine-tuning the top EfficientNet layers improved macro F1 from 0.6004 to 0.6770 and accuracy from 0.6341 to 0.6911. Neither model is ready for clinical use. The custom CNN was interrupted after its first completed epoch because CPU training was unusually slow; the saved checkpoint was still evaluated. The selected deployment checkpoint uses verified ImageNet weights followed by low-learning-rate fine-tuning. Calibration and external validation are still required.

`src/tuning.py` provides a reproducible learning-rate search against the supplied validation folder. Conventional image-level k-fold cross-validation is intentionally not applied by default because the package does not contain patient identifiers; reshuffling related images across folds could overstate performance. If patient IDs become available, patient-level stratified folds should replace this fixed split for a stronger estimate.

## Statistical analysis

For image data, the primary statistical questions concern whether simple appearance summaries differ by class and whether such differences could create shortcut risk. The notebook defines appropriate descriptive comparisons and optional non-parametric tests when SciPy is installed. These tests describe the dataset and do not establish clinical validity. Model metrics, external validation, and calibration remain more important for deployment decisions.

## Explainability and deployment

`src/explainability.py` provides Grad-CAM for a selected convolutional layer. The Streamlit app displays the uploaded image, predicted class, confidence, and class probability chart, with an explicit clinical-use warning. The app should be used only for an educational demonstration until a trained model has been validated and the required governance controls are in place.

## Conclusion and future work

The project now has a reproducible implementation, measured data profile, 15-chart EDA, modular TensorFlow training path, evaluated model checkpoints, evaluation utilities, explainability helper, notebook, and deployment shell. Recommended future work includes external multi-site validation, patient-level deduplication if patient identifiers become available, probability calibration, uncertainty estimation, subgroup analysis, human review studies, and formal clinical/regulatory assessment.
