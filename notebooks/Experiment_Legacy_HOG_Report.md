# Experiment Report: Legacy HOG Face Recognition Model Evaluation

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Introduction and Motivation](#introduction-and-motivation)
3. [Background: HOG Face Detection and Recognition](#background-hog-face-detection-and-recognition)
4. [System Architecture](#system-architecture)
5. [Experimental Setup](#experimental-setup)
   - 5.1 [Software Environment and Dependencies](#software-environment-and-dependencies)
   - 5.2 [Model Configuration](#model-configuration)
   - 5.3 [Dataset Description](#dataset-description)
   - 5.4 [Reference Gallery](#reference-gallery)
   - 5.5 [Evaluation Parameters](#evaluation-parameters)
6. [Methodology](#methodology)
   - 6.1 [Pipeline Overview](#pipeline-overview)
   - 6.2 [Face Detection Stage](#face-detection-stage)
   - 6.3 [Embedding Extraction Stage](#embedding-extraction-stage)
   - 6.4 [Matching and Identification Stage](#matching-and-identification-stage)
   - 6.5 [Multi-Label Evaluation Strategy](#multi-label-evaluation-strategy)
   - 6.6 [Metrics Definitions](#metrics-definitions)
7. [Evaluation Functions: Detailed Walkthrough](#evaluation-functions-detailed-walkthrough)
   - 7.1 [Inference Loop](#inference-loop)
   - 7.2 [Score Aggregation](#score-aggregation)
   - 7.3 [Full Metrics Computation](#full-metrics-computation)
8. [Results and Analysis](#results-and-analysis)
   - 8.1 [Aggregate Metrics](#aggregate-metrics)
   - 8.2 [Per-Class Performance](#per-class-performance)
   - 8.3 [Precision–Recall Curves](#precisionrecall-curves)
   - 8.4 [ROC Curves](#roc-curves)
   - 8.5 [Per-Class Heatmaps](#per-class-heatmaps)
9. [Discussion](#discussion)
   - 9.1 [Strengths of the HOG Pipeline](#strengths-of-the-hog-pipeline)
   - 9.2 [Limitations and Failure Modes](#limitations-and-failure-modes)
   - 9.3 [Comparison Context with Other Models](#comparison-context-with-other-models)
10. [Reproducibility and Output Artefacts](#reproducibility-and-output-artefacts)
11. [Conclusions](#conclusions)
12. [Appendix: Parameter Reference](#appendix-parameter-reference)

---

## 1. Executive Summary

This document provides a comprehensive description of the experiment implemented in the Jupyter notebook `Experiment_legacy_HOG.ipynb`. The notebook evaluates the **HOG (Histogram of Oriented Gradients)** face recognition pipeline — specifically the `face_recognition_hog` model configuration — against a curated multi-label dataset of 1,536 images depicting four public figures: **Hugh Jackman**, **Donald Trump**, **Giorgia Meloni**, and **Lionel Messi**, plus a dedicated "None" class for images containing none of those identities. The evaluation follows the same rigorous methodology and metrics suite originally designed for the InsightFace baseline evaluation, enabling direct cross-model comparison. The experiment records subset accuracy, F-beta scores at both macro and micro averaging, precision, recall, precision-at-fixed-recall, recall-at-fixed-precision, per-class confusion statistics, ROC-AUC, and generates publication-quality Precision–Recall and ROC curves alongside per-class heatmap visualisations. All results and charts are persisted to a timestamped experiment directory for auditability and reproducibility.

---

## 2. Introduction and Motivation

Modern face recognition systems span a wide spectrum of complexity. On one end sit lightweight, CPU-oriented pipelines built on classical computer-vision descriptors; on the other end lie deep-learning approaches leveraging convolutional neural networks (CNNs), transformer architectures, and large-scale pre-training. Understanding where a legacy approach like HOG sits on this spectrum — and quantifying its accuracy rigorously — is essential when making deployment decisions in environments where GPU hardware is unavailable, latency budgets are tight, or model footprint matters.

The ImageEngine project is a modular face recognition platform that supports pluggable detection models (InsightFace variants, CNN, HOG), pluggable embedding models (InsightFace, `face_recognition` dlib embeddings, CLIP/ViT), and pluggable matching methods (cosine similarity, Euclidean distance, L2 distance). Each combination can be independently evaluated against a shared benchmark. This notebook specifically isolates the legacy HOG detector combined with `face_recognition` (dlib) embeddings and cosine-similarity matching, producing the same family of metrics used for the InsightFace baseline so that fair, apples-to-apples comparisons are possible.

The motivation for this experiment is threefold:

1. **Baseline benchmarking** — establish quantitative performance bounds for the simplest, fastest model in the system.
2. **Cross-model comparability** — produce results in an identical metric schema so they can be directly tabulated against CNN, InsightFace, and ViT model results from companion notebooks (`Experiment_legacy_CNN.ipynb`, `Experiment_insightface_only.ipynb`, `Experiment_legacy_ViT.ipynb`).
3. **Failure-mode analysis** — understand where HOG struggles (small faces, profile angles, complex backgrounds) to inform ensemble or fallback strategies.

---

## 3. Background: HOG Face Detection and Recognition

### 3.1 HOG Descriptor

The Histogram of Oriented Gradients is a feature descriptor introduced by Dalal and Triggs (2005) for object detection. It works by dividing an image into small connected regions called cells, computing a histogram of gradient orientations within each cell, and then normalising these histograms over larger spatial blocks to achieve illumination invariance. When applied to face detection, a sliding window scans the image at the configured scale, and a linear SVM classifier trained on HOG features determines whether each window contains a face.

### 3.2 The `face_recognition` Library

The Python `face_recognition` library, authored by Adam Geitgey, wraps dlib's face detection and face recognition models. For detection, it offers two back-ends:

- **HOG + Linear SVM** — fast, CPU-friendly, suitable for frontal faces at medium resolutions.
- **CNN (MMOD)** — a Max-Margin Object Detection CNN that is more accurate but requires significantly more compute.

For embedding extraction, the library uses dlib's ResNet-based face recognition model, which outputs a **128-dimensional** face descriptor. This descriptor can then be compared via Euclidean distance (the library's default) or cosine similarity (as used in this experiment) to determine identity.

### 3.3 Cosine Similarity vs. Euclidean Distance

Although the `face_recognition` library historically defaults to Euclidean distance for face comparison, this experiment explicitly uses **cosine similarity** to match the methodology of the InsightFace baseline evaluation notebook. Cosine similarity measures the angle between two embedding vectors rather than their absolute distance, making it more robust to embedding magnitude variations. The similarity score is computed as:

$$
\text{cosine\_similarity}(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{|\mathbf{a}| \cdot |\mathbf{b}|}
$$

A higher score indicates greater similarity. The identification threshold (0.30 in this experiment) determines the minimum score required to accept a match.

---

## 4. System Architecture

The experiment leverages the ImageEngine project's **modular classification architecture**, which decomposes face recognition into three independent, pluggable layers:

| Layer | Component | This Experiment |
|-------|-----------|-----------------|
| **Detection** | `FaceDetector` | `hog` — HOG + Linear SVM via `face_recognition` |
| **Embedding** | `EmbeddingExtractor` | `face_recognition` — dlib ResNet, 128-dim descriptors |
| **Matching** | `MatchingMethod` | `cosine_similarity` — normalised dot product |

The `UnifiedClassifier` class orchestrates these three layers. It is instantiated via the `get_classifier("unified", ...)` factory function, which accepts configuration parameters for each layer. This design means the exact same evaluation harness can test any permutation of detector, embedder, and matcher without code changes — only parameter swaps.

### 4.1 Key Classes Involved

- **`FaceDetector`** — supports InsightFace models (`buffalo_l`, `buffalo_m`, `buffalo_s`, `antelopev2`) and legacy models (`cnn`, `hog`). For HOG, it delegates to `face_recognition.face_locations(image, model="hog")`. Multi-pass detection is available: if no face is found at the default upsampling level, the detector retries with higher upsampling.
- **`FaceRecognitionEmbedder`** (subclass of `EmbeddingExtractor`) — extracts 128-dimensional face descriptors using dlib's pre-trained ResNet model via the `face_recognition` library.
- **`CosineSimilarityMatching`** (subclass of `MatchingMethod`) — computes cosine similarity between query and reference embeddings.
- **`UnifiedClassifier`** (subclass of `Classifier`) — ties everything together: loads reference images, extracts reference embeddings at initialisation, then at inference time detects faces, computes embeddings, matches against the reference gallery, and returns identified names with bounding boxes.

---

## 5. Experimental Setup

### 5.1 Software Environment and Dependencies

The experiment runs inside a Docker container based on Ubuntu 24.04 LTS. Key Python dependencies include:

- `face_recognition` — for HOG detection and dlib embedding extraction
- `opencv-python` (`cv2`) — for image I/O and manipulation
- `scikit-learn` — for computing precision, recall, F-beta, ROC curves, AUC, and multi-label binarisation
- `numpy` — for numerical operations on embeddings and scores
- `matplotlib` + `seaborn` — for generating PR curves, ROC curves, and heatmap visualisations
- `tqdm` — for progress bars during evaluation
- `pandas` — available for tabular analysis

The notebook sets the matplotlib backend to `'Agg'` (non-interactive) to ensure compatibility inside containerised or headless environments, while still rendering inline plots via Jupyter's `plt.show()`.

### 5.2 Model Configuration

The following model configuration is used throughout the experiment:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `MODEL_KEY` | `face_recognition_hog` | Identifies the pipeline variant |
| `DETECTION_MODEL` | `hog` | HOG + Linear SVM face detector |
| `EMBEDDING_MODEL` | `face_recognition` | dlib ResNet 128-dim embeddings |
| `MATCHING_METHOD` | `cosine_similarity` | Angle-based matching for cross-model parity |
| `MODEL_COLOR` | `#e74c3c` (red) | Visual colour in plots |

The classifier is instantiated once via `get_classifier("unified", celebrity_data, detection_model="hog", embedding_model="face_recognition", matching_method="cosine_similarity")` and deleted after evaluation to free memory.

### 5.3 Dataset Description

The evaluation uses the **four-people trainset** (`testsets/four-people-trainset.json`), a large, curated dataset comprising **1,536 images** distributed across five classes:

| Class | Image Count | Description |
|-------|-------------|-------------|
| Donald Trump | 351 | Press photos, public appearances, varied lighting and angles |
| Giorgia Meloni | 350 | Press photos, similar diversity |
| Hugh Jackman | 304 | Red-carpet events, candid photos, film stills |
| Lionel Messi | 369 | Match photos, press conferences, casual settings |
| None | 228 | Images containing no target identities or unrecognisable faces |

**Multi-label images** are present in the dataset. For example, press photographs of Donald Trump and Giorgia Meloni together carry both labels simultaneously. This is a realistic scenario in press-agency workflows and requires the evaluation to handle multi-label prediction correctly.

Each entry in the JSON testset contains:
- `image_name` — the file basename
- `path` — relative path from the project root to the image file
- `other_paths` — alternative paths for multi-label images appearing in multiple class folders
- `labels` — a list of identity labels present in the image

### 5.4 Reference Gallery

The reference gallery is loaded from `data/references.json` and contains one reference image per identity:

| Identity | Reference Image |
|----------|----------------|
| Hugh Jackman | `Images/references/HughJackman.jpg` |
| Donald Trump | `Images/references/DonaldTrump.jpg` |
| Giorgia Meloni | `Images/references/GiorgiaMeloni.jpg` |
| Lionel Messi | `Images/references/LionnelMessi.png` |

At initialisation, the classifier detects a face in each reference image, extracts its 128-dimensional embedding, and stores these as the reference gallery against which all test-image faces are compared. Using a single reference image per identity represents the most constrained gallery setting; augmenting the gallery with additional reference images could improve robustness.

### 5.5 Evaluation Parameters

The following hyperparameters govern evaluation, chosen to match the InsightFace baseline notebook exactly:

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `FACE_IDENTIFICATION_THRESHOLD` | 0.30 | Minimum cosine similarity to accept an identity match |
| `F1_BETA` | 0.4 | Beta parameter for F-beta score — weights precision more heavily than recall |
| `FIXED_RECALL_LEVEL` | 0.95 | Recall level at which Precision@Recall is reported |
| `FIXED_PRECISION_LEVEL` | 0.95 | Precision level at which Recall@Precision is reported |

The choice of β = 0.4 (< 1) reflects a system design philosophy that prioritises **precision over recall**: it is preferable to miss a celebrity in a photo than to misidentify someone. This is appropriate for press-agency workflows where false positives are costly (mis-labelled photos propagate errors downstream).

---

## 6. Methodology

### 6.1 Pipeline Overview

The evaluation follows a sequential pipeline for each image:

1. **Load image** via OpenCV (`cv2.imread`).
2. **Detect faces** using the HOG detector — returns a list of bounding boxes in `(top, right, bottom, left)` format.
3. **Extract embeddings** for each detected face using dlib's ResNet model via `face_recognition`.
4. **Compute cosine similarity** between each face embedding and every reference embedding.
5. **Identify** each face by selecting the reference with the highest similarity score, accepting only if that score exceeds the identification threshold (0.30).
6. **Aggregate** per-image predictions into a set of identified names (or `{"None"}` if no identifiable face was found).
7. **Collect** ground-truth labels and predicted labels across the entire dataset.
8. **Compute metrics** at both macro and micro averaging levels.

### 6.2 Face Detection Stage

The HOG face detector operates by:

1. Converting the image to grayscale internally.
2. Computing HOG feature descriptors at each position in a sliding-window scan.
3. Classifying each window with a pre-trained linear SVM.
4. Applying non-maximum suppression to merge overlapping detections.

The default upsampling factor is 2 (the `FaceDetector` class default), meaning the image is upscaled twice before scanning, improving detection of smaller faces at the cost of additional compute. If multi-pass detection is enabled and no faces are found, the detector retries with an upsampling factor of 3. In this notebook, the classifier is constructed with the default `UnifiedClassifier` settings, which include `detection_upsample=1` and `enable_multi_pass=False`.

### 6.3 Embedding Extraction Stage

Once faces are detected, the `FaceRecognitionEmbedder` extracts a 128-dimensional descriptor for each face. The dlib model internally:

1. Aligns the face using 5-point facial landmarks.
2. Resizes the aligned face to 150×150 pixels.
3. Passes it through a ResNet model trained with a metric-learning loss on millions of face images.
4. Outputs a 128-dimensional L2-normalised descriptor.

These descriptors are compact and fast to compute, making them suitable for real-time applications on CPU hardware.

### 6.4 Matching and Identification Stage

For each detected face embedding, the experiment computes cosine similarity against every reference embedding. The cosine similarity function is defined inline in the notebook:

$$
\text{sim}(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a}}{|\mathbf{a}| + \epsilon} \cdot \frac{\mathbf{b}}{|\mathbf{b}| + \epsilon}
$$

where ε = 10⁻⁸ prevents division by zero. The identity with the highest similarity score is selected, and the face is identified only if that score meets or exceeds the threshold (0.30). If no face exceeds the threshold, it is counted as an "unknown" face; the image's predicted label set may ultimately include only `"None"`.

This threshold of 0.30 is relatively permissive for cosine similarity (which ranges from -1 to +1), reflecting the fact that `face_recognition` dlib embeddings tend to produce lower cosine similarity scores than InsightFace's 512-dimensional embeddings.

### 6.5 Multi-Label Evaluation Strategy

Because images may contain multiple people (e.g., Donald Trump and Giorgia Meloni in the same press photo), the evaluation treats each image as a **multi-label classification** problem:

- **Ground truth**: a set of identity labels per image (e.g., `{"Donald Trump", "Giorgia Meloni"}`).
- **Prediction**: a set of identity labels per image, derived from all faces detected and identified in that image.

The `MultiLabelBinarizer` from scikit-learn converts these label sets into binary indicator matrices, enabling standard multi-label metrics to be computed. This correctly handles partial matches (e.g., detecting Trump but missing Meloni counts as partial true-positive for Trump and a false-negative for Meloni).

### 6.6 Metrics Definitions

The experiment computes the following metrics, all identical to the InsightFace baseline notebook:

#### Aggregate Metrics

| Metric | Definition |
|--------|------------|
| **Subset Accuracy** | Fraction of images where the predicted label set *exactly* matches the ground-truth label set. This is the strictest multi-label accuracy measure. |
| **F-beta (macro)** | F-beta score averaged across classes (unweighted). With β = 0.4, precision is weighted (1 + β²)/β² ≈ 7.25× more than recall. |
| **F-beta (micro)** | F-beta score computed globally across all class–sample pairs. |
| **Precision (macro/micro)** | Fraction of positive predictions that are correct, averaged per class (macro) or globally (micro). |
| **Recall (macro/micro)** | Fraction of actual positives that are correctly predicted, averaged per class (macro) or globally (micro). |
| **P@R=0.95 (macro/micro)** | Precision at 95% recall — measures how precise the model is when forced to achieve near-complete recall by sweeping the threshold. |
| **R@P=0.95 (macro/micro)** | Recall at 95% precision — measures how much recall the model retains while maintaining near-perfect precision. |
| **Identification Rate** | Fraction of detected faces that received a positive identity match (above threshold). |

#### Per-Class Metrics

For each class (each identity plus "None"), the notebook computes:

- **True Positives (TP)**, **False Positives (FP)**, **False Negatives (FN)**, **True Negatives (TN)** — the full confusion matrix entries.
- **Precision**, **Recall**, **F-beta** — derived from the confusion matrix.
- **ROC-AUC** — Area Under the Receiver Operating Characteristic curve, computed from continuous similarity scores.
- **P@R=0.95** — interpolated precision at the 95% recall operating point on the per-class PR curve.
- **R@P=0.95** — maximum recall achieved while maintaining 95% precision on the per-class PR curve.

The "None" class is handled specially: its score is defined as `1 - max(similarity scores)` — the complement of the best match score. This means an image receives a high "None" score when no reference identity matches well, which is the desired behaviour.

---

## 7. Evaluation Functions: Detailed Walkthrough

### 7.1 Inference Loop (`evaluate_model_with_scores`)

The core evaluation function iterates over every image in the training set and performs the following:

1. **Image Loading**: reads the image via `cv2.imread`. If the image cannot be loaded, it is recorded as a "no face" case and the prediction defaults to `{"None"}`.

2. **Face Detection**: calls `classifier.face_detector.detect_faces(img)`. If no faces are found, the image is again classified as `{"None"}` and the no-face counter increments.

3. **Embedding Extraction**: for each detected bounding box, extracts an embedding. The function checks whether InsightFace-style aligned embedding extraction is appropriate (it is not, for the HOG pipeline), and falls back to the standard `extract_embeddings_batch` method.

4. **Similarity Computation**: for each face embedding, computes cosine similarity against all reference embeddings. Per-identity, only the maximum similarity score is retained (relevant when multiple reference images exist per identity). These per-identity scores are collected in `image_scores`.

5. **Identity Assignment**: the best-matching identity is selected. If its score meets the threshold, the face is labelled with that identity; otherwise, it is counted as "unknown".

6. **Score Aggregation**: after processing all faces in an image, the maximum similarity score per identity across all faces becomes the image-level score for that identity. This handles multi-face images correctly — the face that best matches a given identity determines that identity's score.

The function returns a dictionary containing all predictions, ground-truth labels, per-image per-identity scores, and counters for total/identified/unknown/no-face counts.

### 7.2 Score Aggregation

The per-image score aggregation strategy is critical for computing threshold-dependent metrics like PR curves and ROC curves. For each image and each identity:

$$
\text{score}_{\text{image}, \text{identity}} = \max_{f \in \text{faces}} \text{cosine\_similarity}(\text{emb}_f, \text{ref}_{\text{identity}})
$$

If no faces were detected, all identity scores default to 0.0. This means images without detected faces will have maximum "None" scores (1 - 0 = 1.0), correctly pushing them toward the "None" class in the scoring framework.

### 7.3 Full Metrics Computation (`compute_full_metrics`)

This function takes the raw evaluation output and computes the comprehensive metrics suite:

1. **Multi-label binarisation**: converts ground-truth and predicted label lists into binary matrices using scikit-learn's `MultiLabelBinarizer`.

2. **Aggregate metrics**: computes subset accuracy, F-beta (macro/micro), precision (macro/micro), and recall (macro/micro) using scikit-learn functions with `zero_division=0` to handle classes with no predictions gracefully.

3. **Per-class analysis**: for each class, extracts TP/FP/FN/TN from the binarised matrices, computes precision, recall, F-beta, and accuracy. Then, using the continuous similarity scores, computes the full PR curve and ROC curve. From these curves, P@R=0.95 and R@P=0.95 are extracted via interpolation (for P@R) and filtering (for R@P).

4. **Macro/micro P@R and R@P**: macro values are simple averages of per-class values; micro values are computed from a flattened binary matrix and score vector across all classes and samples simultaneously.

The function returns both a summary dictionary (suitable for tabulation and cross-model comparison) and a detailed dictionary (containing raw curves, per-class metrics, and binarised matrices for further analysis).

---

## 8. Results and Analysis

### 8.1 Aggregate Metrics

The notebook prints a structured summary table containing all aggregate metrics. Key metrics to examine include:

- **Subset Accuracy** — the fraction of images where predictions exactly match ground truth. Given multi-label images, this can be quite strict.
- **F-beta (β=0.4)** — emphasises precision. A high F-beta at β < 1 indicates the model avoids false positives well.
- **Identification Rate** — the fraction of detected faces that the model successfully matched to an identity above the threshold.
- **P@R=0.95** — how precise the model is when pushed to 95% recall; low values indicate the model cannot achieve high recall without significant precision loss.
- **R@P=0.95** — how much recall remains when precision is held at 95%; low values indicate the model cannot achieve both high precision and high recall simultaneously.

### 8.2 Per-Class Performance

The per-class breakdown is printed as a formatted table with columns for precision, recall, F-beta, ROC-AUC, P@R=0.95, R@P=0.95, and the full confusion quadrant (TP, FP, FN, TN). This allows identification of which celebrities the HOG model handles well and which it struggles with.

Common patterns expected from HOG models:

- **Higher performance on frontal, well-lit faces** — Hugh Jackman red-carpet photos may score well.
- **Lower performance on profile views and small faces** — match photos of Lionel Messi at a distance may suffer from detection failures.
- **"None" class sensitivity** — depends heavily on the threshold; a low threshold (0.30) may produce false positives on "None" images.

### 8.3 Precision–Recall Curves

The notebook generates a combined PR curve plot with:

- **Macro-average PR curve** (thick navy line) — averaged across all classes using interpolation over a common recall grid.
- **Per-class PR curves** (thinner coloured lines) — showing how each identity's precision degrades as recall increases.
- **AUC annotations** — each curve is labelled with its Area Under the Curve.

The macro-average PR AUC is the single most informative number for overall ranking-quality assessment. Higher AUC indicates a model that maintains precision across a wider range of recall levels.

### 8.4 ROC Curves

Similarly, the notebook generates ROC curves:

- **Macro-average ROC curve** — interpolated across classes.
- **Per-class ROC curves** — showing TPR vs. FPR trade-offs per identity.
- **Random baseline** — the diagonal reference line (AUC = 0.5).

ROC-AUC is generally expected to be higher than PR-AUC, particularly when classes are imbalanced (as with the "None" class having fewer samples).

### 8.5 Per-Class Heatmaps

A four-panel horizontal heatmap visualises per-class precision, recall, F-beta, and ROC-AUC in a colour-coded grid. This provides an at-a-glance view of which identities are well-served by the HOG pipeline and which need attention. The `YlOrRd` colour map runs from yellow (low) to red (high), with annotations showing the exact numeric values.

---

## 9. Discussion

### 9.1 Strengths of the HOG Pipeline

1. **Speed** — HOG is an order of magnitude faster than CNN-based detectors on CPU hardware. For real-time applications on edge devices or budget servers, this matters.
2. **No GPU required** — the entire pipeline (detection, embedding, matching) runs on CPU, unlike InsightFace or ViT models that benefit significantly from GPU acceleration.
3. **Small memory footprint** — the dlib models are compact compared to ONNX-based InsightFace models (159–407 MB) or CLIP transformers.
4. **Simplicity** — fewer moving parts, fewer dependency issues, easier to debug and deploy.

### 9.2 Limitations and Failure Modes

1. **Limited pose invariance** — HOG detectors are trained primarily on frontal faces and struggle with profile views beyond approximately 30° from frontal. This leads to detection failures (no face found) rather than misidentification.
2. **Small face sensitivity** — without aggressive upsampling, faces occupying a small portion of the image may go undetected. The default upsampling in the `UnifiedClassifier` is 1, which is relatively conservative.
3. **Lower-dimensional embeddings** — at 128 dimensions, the dlib face descriptor has less representational capacity than InsightFace's 512-dimensional or ViT's 768-dimensional embeddings. This may lead to confusion between visually similar identities.
4. **Cosine similarity calibration** — the `face_recognition` embeddings were originally designed for Euclidean distance comparison. While cosine similarity works, the threshold calibration (0.30) is different from what would be used with Euclidean distance (typically ~0.6), and may not be optimal. The threshold was chosen for cross-model consistency rather than per-model optimisation.
5. **Single reference image** — using only one reference image per identity is a challenging setting. Any model will suffer from reference bias; if the reference image shows a person at a specific age, hairstyle, or lighting condition, test images that deviate significantly will score lower.

### 9.3 Comparison Context with Other Models

This experiment is designed to sit alongside evaluations of:

- **CNN (`Experiment_legacy_CNN.ipynb`)** — uses the `face_recognition` CNN detector instead of HOG. The CNN detector is more robust to pose and scale variations but significantly slower.
- **InsightFace (`Experiment_insightface_only.ipynb`)** — uses SCRFD detection and ArcFace 512-dimensional embeddings. Expected to outperform both legacy models significantly.
- **ViT (`Experiment_legacy_ViT.ipynb`)** — uses CLIP Vision Transformer embeddings (768-dim). Strong at generalised visual matching but may lag in face-specific tasks without fine-tuning.

The shared metric schema ensures results from all four notebooks can be directly compared in a summary table or radar chart. The consistent use of cosine similarity, identical F-beta parameter (0.4), identical threshold for the primary result (0.30), and identical P@R/R@P levels (0.95) eliminates confounding variables and isolates model-architecture differences.

---

## 10. Reproducibility and Output Artefacts

### 10.1 Timestamped Experiment Directory

Each run of the notebook creates a unique directory under `image_outputs/` with the naming convention:

```
eval_hog_YYYYMMDD_HHMMSS/
```

This directory contains:

| File | Description |
|------|-------------|
| `per_class_metrics_heatmap.png` | Four-panel heatmap of precision, recall, F-beta, ROC-AUC per class |
| `pr_roc_curves.png` | Combined PR and ROC curve plots with macro-average and per-class curves |

### 10.2 Determinism

The evaluation is deterministic in the following sense:

- **Detection**: HOG is deterministic given the same input and upsampling factor.
- **Embedding**: dlib's ResNet model produces identical outputs for identical inputs.
- **Matching**: cosine similarity is a pure mathematical operation.

Therefore, re-running the notebook on the same dataset with the same parameters will produce identical numerical results. The only non-determinism is in the timestamp of the output directory.

### 10.3 Data Dependencies

The notebook depends on the following external data files:

1. `testsets/four-people-trainset.json` — the evaluation dataset index (1,536 entries)
2. `data/references.json` — the reference gallery specification (4 identities)
3. `Images/references/` — the actual reference face images (4 files)
4. `Images/four-people Testset/` — the actual test images (organised by class directories)

All paths are relative to the project root (`/app`), and the notebook explicitly sets `os.chdir("/app")` to ensure path consistency.

---

## 11. Conclusions

The `Experiment_legacy_HOG.ipynb` notebook implements a rigorous, multi-faceted evaluation of the HOG-based face recognition pipeline. By using the same dataset, metric suite, and evaluation methodology as the InsightFace baseline, it enables meaningful cross-model comparison. The experiment's design reflects several best practices:

1. **Modular architecture** — leveraging the project's pluggable components means the evaluation code is not HOG-specific; only configuration parameters change.
2. **Multi-label awareness** — the evaluation correctly handles images containing multiple identities, avoiding the oversimplification of treating face recognition as single-label classification.
3. **Threshold-independent analysis** — by computing full PR and ROC curves from continuous similarity scores, the evaluation does not depend solely on the chosen threshold; decision-makers can select an operating point appropriate for their use case.
4. **Visual reporting** — heatmaps and curve plots provide intuitive summaries alongside numeric tables.
5. **Reproducibility** — timestamped outputs, deterministic processing, and explicit parameter documentation support experiment tracking.

The HOG pipeline represents the performance floor of the ImageEngine system. Its results inform decisions about when CPU-only deployment is acceptable, when more powerful models are needed, and what combination of detection and embedding models offers the best accuracy–latency trade-off for specific applications.

---

## 12. Appendix: Parameter Reference

### A. Notebook Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `MODEL_KEY` | `face_recognition_hog` | Internal model identifier |
| `MODEL_DISPLAY` | `HOG (face_recognition)` | Human-readable model name |
| `MODEL_COLOR` | `#e74c3c` | Plot colour (red) |
| `DETECTION_MODEL` | `hog` | Face detection backend |
| `EMBEDDING_MODEL` | `face_recognition` | Embedding extraction backend |
| `MATCHING_METHOD` | `cosine_similarity` | Similarity computation method |
| `FACE_IDENTIFICATION_THRESHOLD` | 0.30 | Minimum cosine similarity for identity acceptance |
| `F1_BETA` | 0.4 | F-beta weighting parameter |
| `FIXED_RECALL_LEVEL` | 0.95 | Recall level for P@R metric |
| `FIXED_PRECISION_LEVEL` | 0.95 | Precision level for R@P metric |

### B. Dataset Statistics

| Statistic | Value |
|-----------|-------|
| Total images | 1,536 |
| Donald Trump images | 351 |
| Giorgia Meloni images | 350 |
| Hugh Jackman images | 304 |
| Lionel Messi images | 369 |
| None images | 228 |
| Reference images per identity | 1 |
| Total reference identities | 4 |
| Multi-label images | Present (e.g., Trump + Meloni) |

### C. Embedding Specifications

| Property | Value |
|----------|-------|
| Model | dlib ResNet |
| Dimensionality | 128 |
| Normalisation | L2-normalised |
| Training data | Millions of face images (metric learning) |
| Input face size | 150×150 (after alignment) |
| Alignment | 5-point facial landmarks |

### D. Evaluation Metric Formulas

**F-beta Score:**
$$
F_\beta = (1 + \beta^2) \cdot \frac{\text{precision} \cdot \text{recall}}{\beta^2 \cdot \text{precision} + \text{recall}}
$$

**Cosine Similarity:**
$$
\cos(\theta) = \frac{\mathbf{a} \cdot \mathbf{b}}{||\mathbf{a}|| \cdot ||\mathbf{b}||}
$$

**Precision@Recall=r:**
$$
P@R = \text{interpolated precision at recall level } r \text{ on the PR curve}
$$

**Recall@Precision=p:**
$$
R@P = \max\{r : \text{precision}(r) \geq p\}
$$

### E. Visualisation Specifications

| Plot | Format | Resolution | Colour Map |
|------|--------|------------|------------|
| PR + ROC Curves | 18×8 inches, side-by-side | 150 DPI | Navy macro-avg, per-class colours |
| Per-Class Heatmap | 20×4 inches, four panels | 150 DPI | `YlOrRd` (sequential) |
| Plot annotations | AUC values in legends | — | — |

---

---

## 13. Notebook Cell-by-Cell Summary

The notebook consists of twelve cells — six code cells interleaved with six markdown narrative cells — structured as follows:

**Cell 1 (Markdown) — Title and Overview:** Introduces the experiment, names the model under test (`face_recognition_hog`), and lists all metrics that will be calculated. This cell serves as a self-contained abstract for anyone reviewing the notebook without running it.

**Cell 2 (Code) — Environment Setup:** Imports all dependencies (NumPy, pandas, OpenCV, matplotlib, seaborn, scikit-learn, tqdm), adds the project root to `sys.path`, and changes the working directory to `/app`. It then defines all constants — model key, detection/embedding/matching configuration, evaluation parameters (threshold, F-beta, fixed recall/precision levels), and creates a timestamped experiment output directory. The cell ends by printing a diagnostic summary confirming the setup is complete.

**Cell 3 (Markdown) — Section Header:** Introduces the training-set and reference-loading section.

**Cell 4 (Code) — Data Loading:** Loads the four-people trainset JSON and computes label-distribution statistics using `collections.Counter`. Loads celebrity reference data via the project's `load_celebrities_from_json` utility function, printing the name and reference image path for each identity. Extracts the list of all identity names for use in evaluation. This cell is essential for verifying the data pipeline before committing to a full evaluation run.

**Cell 5 (Markdown) — Section Header:** Introduces the evaluation function definitions.

**Cell 6 (Code) — Evaluation Function Definitions:** Defines two critical functions. `cosine_similarity(emb1, emb2)` normalises two embedding vectors and returns their dot product. `evaluate_model_with_scores(classifier, train_items, all_identities, threshold)` iterates over every test image, detects faces, extracts embeddings, computes per-identity cosine similarity scores, assigns identities above the threshold, and collects comprehensive statistics. `compute_full_metrics(eval_result, all_identities, f_beta, fixed_recall, fixed_precision)` transforms raw predictions and scores into the full metrics suite using scikit-learn's multi-label binarisation, computes aggregate and per-class metrics, generates PR and ROC curves per class, and extracts P@R and R@P values. This is the largest and most complex cell in the notebook, amounting to approximately 150 lines of carefully structured evaluation logic.

**Cell 7 (Markdown) — Section Header:** Introduces the evaluation execution section.

**Cell 8 (Code) — Model Evaluation:** Instantiates the classifier via `get_classifier("unified", ...)` with the HOG configuration, verifies that reference embeddings were loaded successfully, runs the full evaluation loop via `evaluate_model_with_scores`, computes comprehensive metrics via `compute_full_metrics`, and prints a formatted results summary. The classifier is explicitly deleted (`del classifier`) after evaluation to free memory resources, which is important in containerised environments with limited RAM.

**Cell 9 (Markdown) — Section Header:** Introduces per-class analysis.

**Cell 10 (Code) — Per-Class Analysis and Heatmap:** Prints a detailed per-class metrics table with precision, recall, F-beta, ROC-AUC, P@R, R@P, and confusion values for every class. Generates a four-panel heatmap visualisation using seaborn's `heatmap` function with the `YlOrRd` colour map, saving the result to the experiment directory at 150 DPI.

**Cell 11 (Markdown) — Section Header:** Introduces PR and ROC curves.

**Cell 12 (Code) — PR and ROC Curve Generation:** Computes per-class PR and ROC curves from continuous similarity scores, then generates macro-average curves by interpolating all per-class curves onto a common grid and averaging. Produces a side-by-side 18×8-inch figure with the PR curve (left) and ROC curve (right), each annotated with AUC values in the legend. The random-baseline diagonal is plotted on the ROC chart for reference. Saves the figure to the experiment directory.

**Cell 13 (Markdown) — Section Header:** Introduces the summary report.

**Cell 14 (Code) — Summary Report:** Prints a comprehensive final summary that recapitulates all configuration parameters, all aggregate metrics in a clean table format, and per-class F-beta scores. This provides a self-contained textual record of the experiment's outcome, suitable for copy-paste into comparison reports or version-control commit messages.

This sequential structure ensures that the notebook can be executed top-to-bottom in a single pass, with each cell depending only on variables defined in preceding cells. No out-of-order execution is required, and all side effects (file writes) occur only in the plot-generation and summary cells.

---

*This report was generated to document the experiment implemented in `Experiment_legacy_HOG.ipynb` as part of the ImageEngine face recognition evaluation suite.*
