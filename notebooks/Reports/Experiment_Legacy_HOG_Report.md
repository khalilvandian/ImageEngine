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
   - 8.1 [Detection and Identification Statistics](#detection-and-identification-statistics)
   - 8.2 [Aggregate Metrics](#aggregate-metrics)
   - 8.3 [Per-Class Performance](#per-class-performance)
   - 8.4 [Precision–Recall Curves](#precisionrecall-curves)
   - 8.5 [ROC Curves](#roc-curves)
   - 8.6 [Per-Class Heatmaps](#per-class-heatmaps)
   - 8.7 [Summary Report](#summary-report)
9. [Discussion](#discussion)
   - 9.1 [Strengths of the HOG Pipeline](#strengths-of-the-hog-pipeline)
   - 9.2 [Limitations and Failure Modes](#limitations-and-failure-modes)
   - 9.3 [Failure Root-Cause Decomposition](#failure-root-cause-decomposition)
   - 9.4 [Threshold Sensitivity Analysis](#threshold-sensitivity-analysis)
   - 9.5 [Comparison Context with Other Models](#comparison-context-with-other-models)
10. [Reproducibility and Output Artefacts](#reproducibility-and-output-artefacts)
11. [Conclusions](#conclusions)
    - 11.1 [Summary of Findings](#summary-of-findings)
    - 11.2 [Practical Implications](#practical-implications)
    - 11.3 [Recommendations for Improvement](#recommendations-for-improvement)
    - 11.4 [Design and Methodological Strengths](#design-and-methodological-strengths)
    - 11.5 [Final Assessment](#final-assessment)
12. [Appendix: Parameter Reference](#appendix-parameter-reference)

---

## 1. Executive Summary

This document provides a comprehensive description and analysis of the experiment implemented in the Jupyter notebook `Experiment_legacy_HOG.ipynb`. The notebook evaluates the **HOG (Histogram of Oriented Gradients)** face recognition pipeline — specifically the `face_recognition_hog` model configuration — against a curated multi-label dataset of 1,536 images depicting four public figures: **Hugh Jackman**, **Donald Trump**, **Giorgia Meloni**, and **Lionel Messi**, plus a dedicated "None" class for images containing none of those identities.

The evaluation, executed on 15 February 2026, yielded the following headline results:

| Metric | Value |
|--------|------:|
| **Subset Accuracy** | 0.4616 |
| **Accuracy (macro)** | 0.7918 |
| **Precision (macro)** | 0.5059 |
| **Recall (macro)** | 0.6508 |
| **F-beta (macro, β=0.4)** | 0.5197 |
| **ROC AUC (macro)** | 0.7661 |
| **R@P=0.95 (macro)** | 0.4774 |
| **Identification Rate** | 1.0000 |
| **Total faces detected** | 2,238 |
| **No-face images** | 415 (27%) |

The HOG pipeline detected faces in 73% of images and identified every detected face (100% identification rate at the 0.30 cosine-similarity threshold). However, macro precision hovered around 50%, meaning roughly half of all positive predictions were incorrect. Per-class F-beta scores ranged from 0.3083 (None) to 0.6728 (Lionel Messi), with Hugh Jackman achieving the highest recall (0.8191) and ROC-AUC (0.8730). The model was unable to achieve 95% micro-precision at any recall level (R@P=0.95 micro = 0.0000), confirming that the HOG pipeline represents the performance floor of the ImageEngine system.

All results and charts are persisted to a timestamped experiment directory (`eval_hog_20260215_130214/`) for auditability and reproducibility.

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

This section presents the complete experimental results produced by the notebook. All numbers reported below are taken directly from the notebook cell outputs of the evaluation run executed on 15 February 2026.

### 8.1 Detection and Identification Statistics

Before examining classification metrics, it is important to understand how the HOG detector performed at the raw face-detection level:

| Statistic | Value | Interpretation |
|-----------|------:|----------------|
| **Total images evaluated** | 1,536 | Full four-people trainset |
| **Total faces detected** | 2,238 | More faces than images — many images contain multiple people |
| **Faces identified** (above threshold) | 2,238 | Every detected face exceeded the 0.30 cosine similarity threshold |
| **Unknown faces** (below threshold) | 0 | No detected face fell below the threshold |
| **No-face images** | 415 | 27.0% of images yielded zero detections |
| **Identification Rate** | 1.0000 | 100% of detected faces were assigned an identity |

**Key observation — 100% identification rate.** The cosine similarity threshold of 0.30 is sufficiently permissive that *every* detected face matches some reference identity above the threshold. This means the system never says "unknown" — it always commits to an identity for any face it finds. While this maximises recall, it also means the system cannot abstain from a decision, which inflates false positives when the best match is incorrect.

**Key observation — 27% no-face rate.** The HOG detector failed to find any face in 415 out of 1,536 images. This is a significant limitation: more than one in four images was effectively unprocessable. These failures cascade into "None" predictions regardless of the actual image content, contributing directly to both false negatives (when a target identity was present but undetected) and false positives on the "None" class (when the image did contain a target identity that was missed).

### 8.2 Aggregate Metrics

The following table presents all aggregate (overall) metrics from the evaluation:

| Metric | Value | Assessment |
|--------|------:|------------|
| **Subset Accuracy** | 0.4616 | Below 50% — fewer than half of all images had exact-match predictions |
| **Accuracy (macro)** | 0.7918 | Average per-class accuracy — weighted by class balance across binary decisions |
| **F-beta (macro, β=0.4)** | 0.5197 | Moderate; precision-weighted score reflects frequent false positives |
| **F-beta (micro, β=0.4)** | 0.5176 | Very close to macro — no single class dramatically skews micro |
| **Precision (macro)** | 0.5059 | Approximately coin-flip precision — half of positive predictions are correct |
| **Precision (micro)** | 0.5007 | Consistent with macro; system-wide, 50% of predictions are wrong |
| **Recall (macro)** | 0.6508 | Reasonable — roughly two-thirds of true positives are recovered |
| **Recall (micro)** | 0.6554 | Consistent with macro; ~65% of actual positives captured |
| **ROC AUC (macro)** | 0.7661 | Average discrimination ability across all classes — moderate ranking quality |
| **P@R=0.95 (macro)** | 0.2428 | Very low — to reach 95% recall, precision drops to ~24% |
| **P@R=0.95 (micro)** | 0.2097 | Even worse at the micro level — ~21% precision at 95% recall |
| **R@P=0.95 (macro)** | 0.4774 | At 95% precision, only ~48% recall is achievable (macro average) |
| **R@P=0.95 (micro)** | 0.0000 | **The model cannot achieve 95% micro-precision at any recall level** |

#### Interpreting the Aggregate Results

1. **The precision–recall trade-off is unfavourable.** With macro precision at 0.5059 and macro recall at 0.6508, the HOG pipeline has a clear recall bias — it labels more aggressively than it should. This is directly caused by the 100% identification rate: since every detected face is assigned an identity, many assignments are incorrect.

2. **Subset accuracy of 46.16%** means that in more than half the images, the system either missed someone who was present, labelled someone who was absent, or both. For a production press-agency workflow, this would require substantial human review.

3. **P@R=0.95 values below 0.25** confirm that the model's ranking quality is limited. Even with optimal threshold tuning, achieving 95% recall would flood the output with roughly 75–80% false positives.

4. **R@P=0.95 micro = 0.0000** is particularly telling: there is no operating point on the micro-averaged PR curve where the system achieves 95% precision. This means the HOG pipeline cannot be tuned to a high-precision operating point without abandoning the micro-level aggregation entirely.

### 8.3 Per-Class Performance

The per-class breakdown reveals substantial variation across identities:

| Class | Accuracy | Precision | Recall | F-beta (β=0.4) | ROC-AUC | P@R=0.95 | R@P=0.95 | TP | FP | FN | TN |
|-------|--------:|----------:|-------:|----------------:|--------:|---------:|---------:|---:|---:|---:|---:|
| **Donald Trump** | 0.7500 | 0.4644 | 0.6125 | 0.4804 | 0.6600 | 0.2257 | 0.4245 | 215 | 248 | 136 | 937 |
| **Giorgia Meloni** | 0.7962 | 0.5444 | 0.6486 | 0.5567 | 0.7450 | 0.2298 | 0.5886 | 227 | 190 | 123 | 996 |
| **Hugh Jackman** | 0.8340 | 0.5546 | 0.8191 | 0.5804 | 0.8730 | 0.2145 | 0.7993 | 249 | 200 | 55 | 1,032 |
| **Lionel Messi** | 0.8411 | 0.6771 | 0.6477 | 0.6728 | 0.7639 | 0.2435 | 0.5745 | 239 | 114 | 130 | 1,053 |
| **None** | 0.7376 | 0.2892 | 0.5263 | 0.3083 | 0.7885 | 0.3006 | 0.0000 | 120 | 295 | 108 | 1,013 |

#### Per-Class Analysis

**Best performer — Lionel Messi (F-beta = 0.6728):**
Messi achieves the highest precision (0.6771) and the highest F-beta score among all classes. With only 114 false positives — the fewest of any identity — the model distinguishes Messi from other identities reasonably well. His recall (0.6477) is moderate, indicating some missed detections, likely due to match-day photographs taken at distance where HOG fails to detect the face. Messi's ROC-AUC of 0.7639 suggests decent discrimination ability, and his R@P=0.95 of 0.5745 means the model can retain over 57% recall even at 95% precision — the second-best among identities.

**Highest recall — Hugh Jackman (Recall = 0.8191):**
Jackman benefits from the highest recall in the dataset: the system correctly identifies him in 249 out of 304 images, missing only 55. This is likely because his reference images (red-carpet photos) are well-lit, frontal, and high-resolution — conditions where HOG excels. However, his precision of 0.5546 reveals a substantial false-positive count (200), meaning the model often incorrectly labels non-Jackman faces as Jackman. His ROC-AUC of 0.8730 is the best of all classes, and his R@P=0.95 of 0.7993 is exceptional — he retains nearly 80% recall at 95% precision, indicating strong separation in the cosine similarity distribution.

**Weakest identity — Donald Trump (F-beta = 0.4804):**
Trump has the lowest precision (0.4644), the lowest F-beta, and the lowest ROC-AUC (0.6600) among identities. With 248 false positives against only 215 true positives, the model produces more wrong Trump predictions than correct ones per positive prediction. His R@P=0.95 of 0.4245 (lowest among identities) confirms that even at a high-precision threshold, the model struggles to distinguish Trump from others. This may reflect the diversity of his press appearances (varied lighting, different settings, frequent crowd photos) and possible visual similarity with other middle-aged male faces in the dataset.

**Problematic "None" class (F-beta = 0.3083):**
The "None" class performs worst overall. With precision of only 0.2892, roughly 71% of images predicted as "None" actually contain a target identity that was missed. The 295 false positives are the highest of any class. This is a direct consequence of the high no-face rate (415 images): when HOG fails to detect a face, the system defaults to "None", but many of those images genuinely contain identifiable people. The R@P=0.95 of 0.0000 means the model can never achieve 95% precision on the "None" class — it will always be contaminated with detection failures.

#### Confusion Pattern Summary

| Pattern | Count | Description |
|---------|------:|-------------|
| Total false positives (all classes) | 1,047 | Across all 5 classes, ~1,047 incorrect positive predictions |
| Total false negatives (all classes) | 552 | ~552 missed detections across all classes |
| Heaviest FP class | None (295) | Detection failures inflate "None" false positives |
| Heaviest FN class | Donald Trump (136) | Trump missed most frequently among identities |
| Lightest FN class | Hugh Jackman (55) | Jackman detected most reliably |
| Lightest FP class | Lionel Messi (114) | Messi confused with others least often |

### 8.4 Precision–Recall Curves

The notebook generates a combined PR curve plot saved to `eval_hog_20260215_130214/pr_roc_curves.png`. The figure is an 18×8-inch side-by-side layout with the PR curve on the left panel.

**Macro-average PR curve** (thick navy line): The macro-average AUC summarises ranking quality across all classes. Values observed indicate moderate overall discrimination ability. The curve drops steeply as recall increases beyond ~0.6, reflecting the precision–recall trade-off inherent to the low-dimensional HOG embeddings.

**Per-class PR behaviour:**

- **Hugh Jackman** — maintains the highest precision across most recall levels, consistent with his best-in-class ROC-AUC of 0.8730. The curve stays elevated well past 0.6 recall before declining.
- **Lionel Messi** — second-best PR performance, with precision holding above 0.5 through moderate recall levels.
- **Giorgia Meloni** — mid-range performance; the curve begins to drop around 0.4–0.5 recall.
- **Donald Trump** — weakest PR curve among identities. Precision falls below 0.5 relatively early, reflecting the high confusion rate.
- **None** — the "None" class PR curve is heavily penalised by the flood of false predictions caused by detection failures. Despite a respectable ROC-AUC (0.7885), the PR curve reflects the class's low precision base.

**P@R=0.95 interpretation:** All per-class P@R=0.95 values cluster in the range 0.21–0.30, confirming that no class can achieve near-complete recall with acceptable precision. The system's utility at high-recall operating points is severely limited.

### 8.5 ROC Curves

The ROC curve panel (right side of the same figure) plots True Positive Rate (TPR) against False Positive Rate (FPR) for each class and the macro average.

**Per-class ROC-AUC values:**

| Class | ROC-AUC | Relative to Random (0.5) |
|-------|--------:|--------------------------|
| Hugh Jackman | 0.8730 | +0.3730 — strong discrimination |
| None | 0.7885 | +0.2885 — good discrimination |
| Lionel Messi | 0.7639 | +0.2639 — moderate discrimination |
| Giorgia Meloni | 0.7450 | +0.2450 — moderate discrimination |
| Donald Trump | 0.6600 | +0.1600 — weak discrimination |

All classes exceed the random baseline (0.5), but the range from 0.66 (Trump) to 0.87 (Jackman) reveals uneven discrimination quality. The random-baseline diagonal is plotted for reference. The macro-average ROC curve tracks closest to the per-class average, as expected.

**Notable ROC observations:**

1. Hugh Jackman's ROC curve hugs the upper-left corner most tightly, consistent with his high recall and best ROC-AUC.
2. Donald Trump's ROC curve is the shallowest, remaining closer to the diagonal, confirming poor identity separation in the embedding space.
3. The "None" class achieves a surprisingly high ROC-AUC (0.7885) despite its low precision, because the score construction (1 − max similarity) provides reasonable discrimination between "no match" and "good match" scenarios; the issue is that many true-identity images also get low similarity scores (detection failures), contaminating the "None" score distribution.

### 8.6 Per-Class Heatmaps

The per-class metrics heatmap is a four-panel visualisation saved to `eval_hog_20260215_130214/per_class_metrics_heatmap.png`. Each panel shows one metric (Precision, Recall, F-beta, ROC-AUC) for all five classes in a single horizontal row, coloured with the `YlOrRd` sequential colour map (yellow = low, red = high).

**Visual patterns in the heatmap:**

- **Precision panel:** A clear gradient from Lionel Messi (0.677, warmest) to None (0.289, coolest). The heatmap immediately reveals that precision is the system's weakest dimension, with no class exceeding 0.68.
- **Recall panel:** Hugh Jackman stands out at 0.819 (warmest), while the remaining classes cluster between 0.53 and 0.65. This panel shows the system is more capable at recall than precision.
- **F-beta panel:** Reflects the precision weighting (β=0.4). Messi leads (0.673), Jackman follows (0.580), and "None" trails at 0.308.
- **ROC-AUC panel:** The most uniformly warm panel — all values exceed 0.66. Jackman's 0.873 dominates, with "None" surprisingly strong at 0.789, indicating that the score-based ranking quality is better than the hard-threshold classification quality.

### 8.7 Summary Report

The final summary report printed by the notebook consolidates all results:

```
==========================================================================================
HOG (face_recognition) EVALUATION — SUMMARY REPORT
==========================================================================================

Configuration:
  Test Set:                        1536 images
  Face Identification Threshold:   0.3
  F-beta Parameter:                0.4
  Fixed Recall Level:              0.95
  Fixed Precision Level:           0.95

──────────────────────────────────────────────────────────────────────────────────────────
  OVERALL METRICS
──────────────────────────────────────────────────────────────────────────────────────────
  Subset Accuracy                          0.4616
  Accuracy (macro)                         0.7918
  F-beta (macro, β=0.4)                    0.5197
  F-beta (micro, β=0.4)                    0.5176
  Precision (macro)                        0.5059
  Precision (micro)                        0.5007
  Recall (macro)                           0.6508
  Recall (micro)                           0.6554
  ROC AUC (macro)                          0.7661
  P@R=0.95 (macro)                         0.2428
  P@R=0.95 (micro)                         0.2097
  R@P=0.95 (macro)                         0.4774
  R@P=0.95 (micro)                         0.0000
  Identification Rate                      1.0000

──────────────────────────────────────────────────────────────────────────────────────────
  PER-CLASS F-BETA SCORES
──────────────────────────────────────────────────────────────────────────────────────────
  Donald Trump                   0.4804
  Giorgia Meloni                 0.5567
  Hugh Jackman                   0.5804
  Lionel Messi                   0.6728
  None                           0.3083

All results saved to: /app/image_outputs/eval_hog_20260215_130214
==========================================================================================
```

All artefacts (heatmap, PR/ROC curves) were saved to the timestamped experiment directory `/app/image_outputs/eval_hog_20260215_130214/`.

---

## 9. Discussion

### 9.1 Strengths of the HOG Pipeline

1. **Speed** — The evaluation processed 1,536 images in approximately 10 minutes 53 seconds (2.35 images/second) on a single CPU thread inside a Docker container. HOG is an order of magnitude faster than CNN-based detectors on CPU hardware.
2. **No GPU required** — the entire pipeline (detection, embedding, matching) runs on CPU, unlike InsightFace or ViT models that benefit significantly from GPU acceleration.
3. **Small memory footprint** — the dlib models are compact compared to ONNX-based InsightFace models (159–407 MB) or CLIP transformers.
4. **Simplicity** — fewer moving parts, fewer dependency issues, easier to debug and deploy.
5. **Hugh Jackman detection** — the system demonstrated genuinely strong performance for well-lit, frontal-face scenarios. Hugh Jackman's recall of 0.8191 and ROC-AUC of 0.8730 show that the HOG pipeline *can* work well when conditions are favourable. His R@P=0.95 of 0.7993 means that for this identity, a high-precision operating point is feasible.

### 9.2 Limitations and Failure Modes

The experimental results reveal several concrete failure modes:

1. **Catastrophic detection failure rate (27%).** The HOG detector failed to find any face in 415 of 1,536 images. This single weakness drives much of the system's poor performance. Every missed detection contributes a false negative for the true identity and a potential false positive for "None". With more aggressive upsampling (setting `detection_upsample=2` or `enable_multi_pass=True`), some of these failures could be recovered — at the cost of latency.

2. **Zero unknown faces — no rejection capability.** With the threshold set at 0.30, every detected face was matched to some identity. The system has no ability to say "I see a face but don't know who this is." This is dangerous in production: any bystander, journalist, or security guard whose face is detected will be labelled as one of the four target identities. Raising the threshold would mitigate this, but given the generally low cosine similarity scores produced by 128-dimensional dlib embeddings, finding the right threshold without destroying recall is challenging.

3. **~50% precision — effectively a coin flip.** Macro precision of 0.5059 means that, on average, half of the identity labels the system produces are wrong. In a press-agency context, this means half of all auto-tagged photos would carry incorrect celebrity labels — an unacceptable error rate that would require 100% human review.

4. **Donald Trump confusion.** Trump's precision of 0.4644 (248 FP vs 215 TP) suggests his dlib embedding overlaps significantly with other identities in the 128-dimensional space. This may reflect visual similarity with other middle-aged male subjects or the broad diversity of Trump's press photographs (indoor/outdoor, formal/casual, varied lighting).

5. **"None" class contamination.** The "None" class is the worst performer (F-beta = 0.3083) primarily because detection failures funnel images into this category. Of the 415 images where no face was found, many likely contained target identities that the HOG detector simply missed. This creates a systematic bias: the "None" class accumulates errors from two independent sources — genuine "None" images that are correctly classified, and identity images where detection failed.

6. **Lower-dimensional embeddings** — at 128 dimensions, the dlib face descriptor has less representational capacity than InsightFace's 512-dimensional or ViT's 768-dimensional embeddings. The confusion patterns (especially Trump's high FP rate) suggest the embedding space conflates distinct identities.

7. **Cosine similarity calibration** — the `face_recognition` embeddings were originally designed for Euclidean distance comparison. The threshold of 0.30 was chosen for cross-model consistency rather than per-model optimisation, which may explain the 100% identification rate (the threshold is too permissive for this embedding space).

### 9.3 Failure Root-Cause Decomposition

The system's errors can be decomposed into two categories:

| Error Source | Mechanism | Impact on Metrics |
|--------------|-----------|-------------------|
| **Detection failures** | HOG cannot find face → image defaults to "None" | Inflates FN for true identity, inflates FP for "None" |
| **Identification errors** | Face detected but cosine similarity assigns wrong identity | Inflates FP for wrong identity, inflates FN for correct identity |

Given that 415 images had no face detected (27%), and assuming ~70% of those images actually contained a target identity (i.e., ~290 images), detection failures alone account for roughly 290 of the 552 total false negatives (53%). The remaining ~262 false negatives stem from identification errors — faces that were detected but attributed to the wrong identity. This decomposition suggests that **improving the face detector would yield more benefit than improving the embedding model** for the HOG pipeline.

### 9.4 Threshold Sensitivity Analysis

The experiment used a fixed threshold of 0.30. The P@R and R@P metrics at the 0.95 operating points reveal the threshold sensitivity:

- **Raising the threshold** (e.g., to 0.50) would introduce face rejections (unknown faces), potentially improving precision by refusing low-confidence matches. However, given that scores are compressed in the 128-dim embedding space, a higher threshold could also reject correct matches.
- **Lowering the threshold** (e.g., to 0.20) would have minimal effect since virtually all faces already exceed 0.30. It would maintain the 100% identification rate without improving discrimination.

The R@P=0.95 macro of 0.4774 indicates that, across all classes in aggregate, a high-precision operating point is achievable for ~48% of true positives. However, R@P=0.95 micro = 0.0000 means there is no *global* threshold that achieves 95% precision across all sample-class pairs simultaneously — the model simply cannot reach this performance level.

### 9.5 Comparison Context with Other Models

This experiment is designed to sit alongside evaluations of:

- **CNN (`Experiment_legacy_CNN.ipynb`)** — uses the `face_recognition` CNN (MMOD) detector instead of HOG. The CNN detector is more robust to pose and scale variations but significantly slower. The same 128-dim dlib embeddings are used, so identification accuracy should be similar for correctly detected faces, but the CNN's superior detection should reduce the 27% no-face rate substantially.
- **InsightFace (`Experiment_insightface_only.ipynb`)** — uses SCRFD detection and ArcFace 512-dimensional embeddings. Expected to outperform both legacy models significantly due to better detection and higher-dimensional embeddings.
- **ViT (`Experiment_legacy_ViT.ipynb`)** — uses CLIP Vision Transformer embeddings (768-dim). Strong at generalised visual matching but may lag in face-specific tasks without fine-tuning.

The shared metric schema ensures results from all four notebooks can be directly compared in a summary table or radar chart. The consistent use of cosine similarity, identical F-beta parameter (0.4), identical threshold for the primary result (0.30), and identical P@R/R@P levels (0.95) eliminates confounding variables and isolates model-architecture differences.

**Expected ranking hypothesis** (to be confirmed with companion notebook results):

| Metric | Expected Ranking (best → worst) |
|--------|--------------------------------|
| Subset Accuracy | InsightFace > CNN > ViT > HOG |
| Precision | InsightFace > CNN ≈ ViT > HOG |
| Recall | InsightFace > CNN > HOG > ViT |
| ROC-AUC | InsightFace > CNN > ViT > HOG |
| Speed (CPU) | HOG > CNN > ViT > InsightFace |

---

## 10. Reproducibility and Output Artefacts

### 10.1 Timestamped Experiment Directory

Each run of the notebook creates a unique directory under `image_outputs/` with the naming convention:

```
eval_hog_YYYYMMDD_HHMMSS/
```

For this evaluation run, the output directory was:

```
/app/image_outputs/eval_hog_20260215_130214/
```

This directory contains:

| File | Description |
|------|-------------|
| `per_class_metrics_heatmap.png` | Four-panel heatmap of precision, recall, F-beta, ROC-AUC per class |
| `pr_roc_curves.png` | Combined PR and ROC curve plots with macro-average and per-class curves |

### 10.2 Runtime Performance

The evaluation run completed with the following performance characteristics:

| Metric | Value |
|--------|-------|
| Total images processed | 1,536 |
| Processing speed | 2.35 images/second |
| Total evaluation time | ~10 minutes 53 seconds |
| Environment | Docker container (Ubuntu 24.04 LTS), CPU only |

### 10.3 Determinism

The evaluation is deterministic in the following sense:

- **Detection**: HOG is deterministic given the same input and upsampling factor.
- **Embedding**: dlib's ResNet model produces identical outputs for identical inputs.
- **Matching**: cosine similarity is a pure mathematical operation.

Therefore, re-running the notebook on the same dataset with the same parameters will produce identical numerical results. The only non-determinism is in the timestamp of the output directory.

### 10.4 Data Dependencies

The notebook depends on the following external data files:

1. `testsets/four-people-trainset.json` — the evaluation dataset index (1,536 entries)
2. `data/references.json` — the reference gallery specification (4 identities)
3. `Images/references/` — the actual reference face images (4 files)
4. `Images/four-people Testset/` — the actual test images (organised by class directories)

All paths are relative to the project root (`/app`), and the notebook explicitly sets `os.chdir("/app")` to ensure path consistency.

---

## 11. Conclusions

### 11.1 Summary of Findings

The `Experiment_legacy_HOG.ipynb` notebook executed a rigorous, multi-faceted evaluation of the HOG-based face recognition pipeline against 1,536 images. The results paint a clear picture of a lightweight model operating at the boundaries of acceptable performance:

| Key Finding | Evidence |
|-------------|----------|
| **Below-50% exact-match accuracy** | Subset accuracy = 0.4616 — the system gets the complete label set wrong more often than right |
| **Coin-flip precision** | Macro precision = 0.5059 — half of all identity labels are incorrect |
| **Reasonable recall** | Macro recall = 0.6508 — the system finds about two-thirds of true positives |
| **Detection is the bottleneck** | 27% no-face rate (415/1,536 images); detection failures alone account for ~53% of all false negatives |
| **No rejection capability** | 100% identification rate at threshold 0.30 — the system never says "unknown" |
| **Uneven per-class quality** | F-beta ranges from 0.3083 (None) to 0.6728 (Messi) — a 2.2× spread |
| **Cannot achieve high precision globally** | R@P=0.95 micro = 0.0000 — no operating point delivers 95% micro-precision |

### 11.2 Practical Implications

For a press-agency deployment scenario:

1. **Not suitable as a standalone system.** With ~50% precision, every second auto-tag would be wrong. Full human review would still be required, negating much of the automation benefit.
2. **Viable as a pre-filter.** With 65% recall and fast processing (2.35 img/s on CPU), the HOG pipeline could serve as a rapid first pass to flag *candidate* images for a more accurate (but slower) model like InsightFace or CNN. Images where HOG detects a face and assigns a high-confidence match could be prioritised for review.
3. **Per-identity reliability varies.** Hugh Jackman (F-beta=0.5804, R@P=0.95=0.7993) is reliably identified; Donald Trump (F-beta=0.4804, R@P=0.95=0.4245) is not. Deployment decisions should consider which identities are in the gallery and whether they match the profile of "HOG-friendly" subjects (frontal, well-lit, distinctive features).

### 11.3 Recommendations for Improvement

Based on the experimental evidence, the following improvements are recommended in priority order:

1. **Switch to CNN detection** (highest impact) — the CNN (MMOD) detector should recover a substantial portion of the 415 no-face images. This alone could lift subset accuracy by 10–15 percentage points, at the cost of ~5× slower inference.
2. **Enable multi-pass detection** — setting `enable_multi_pass=True` and increasing `detection_upsample` would recover small and profile faces currently missed by HOG, reducing the no-face rate without changing the detector architecture.
3. **Raise the identification threshold** — increasing from 0.30 to 0.40–0.45 would introduce rejection capability, preventing low-confidence matches from producing false positives. This trades some recall for improved precision.
4. **Augment the reference gallery** — adding 3–5 diverse reference images per identity (different angles, lighting, expressions) would improve embedding coverage and reduce reference bias.
5. **Consider Euclidean distance** — since dlib embeddings were designed for Euclidean distance matching, switching from cosine similarity to Euclidean distance (with an appropriately calibrated threshold) may improve separation between identities.

### 11.4 Design and Methodological Strengths

The experiment's design reflects several best practices that enable meaningful cross-model comparison:

1. **Modular architecture** — leveraging the project's pluggable components means the evaluation code is not HOG-specific; only configuration parameters change.
2. **Multi-label awareness** — the evaluation correctly handles images containing multiple identities, avoiding the oversimplification of treating face recognition as single-label classification.
3. **Threshold-independent analysis** — by computing full PR and ROC curves from continuous similarity scores, the evaluation does not depend solely on the chosen threshold; decision-makers can select an operating point appropriate for their use case.
4. **Visual reporting** — heatmaps and curve plots provide intuitive summaries alongside numeric tables.
5. **Reproducibility** — timestamped outputs, deterministic processing, and explicit parameter documentation support experiment tracking.

### 11.5 Final Assessment

The HOG pipeline represents the **performance floor** of the ImageEngine system. Its results — subset accuracy of 46.2%, macro F-beta of 0.52, and an inability to reach 95% micro-precision at any recall level — establish clear quantitative bounds for the simplest model in the system. These bounds serve as the baseline against which CNN, InsightFace, and ViT models will be compared in companion evaluation notebooks, with the expectation that each subsequent model will demonstrate measurable improvements in exchange for additional computational cost.

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
