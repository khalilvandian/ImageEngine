# Experiment Report: Legacy CNN Face Recognition Model Evaluation

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Introduction and Motivation](#introduction-and-motivation)
3. [Background: CNN Face Detection and dlib Embeddings](#background-cnn-face-detection-and-dlib-embeddings)
4. [System Architecture](#system-architecture)
5. [Experimental Setup](#experimental-setup)
   - 5.1 [Software Environment and Dependencies](#software-environment-and-dependencies)
   - 5.2 [Model Configuration](#model-configuration)
   - 5.3 [Dataset Description](#dataset-description)
   - 5.4 [Reference Gallery](#reference-gallery)
   - 5.5 [Evaluation Parameters](#evaluation-parameters)
6. [Methodology](#methodology)
   - 6.1 [Pipeline Overview](#pipeline-overview)
   - 6.2 [Face Detection Stage — CNN (MMOD)](#face-detection-stage--cnn-mmod)
   - 6.3 [Memory-Aware Image Resizing](#memory-aware-image-resizing)
   - 6.4 [Embedding Extraction Stage](#embedding-extraction-stage)
   - 6.5 [Matching and Identification Stage](#matching-and-identification-stage)
   - 6.6 [Multi-Label Evaluation Strategy](#multi-label-evaluation-strategy)
   - 6.7 [Metrics Definitions](#metrics-definitions)
7. [Evaluation Functions: Detailed Walkthrough](#evaluation-functions-detailed-walkthrough)
   - 7.1 [Inference Loop (`evaluate_model_with_scores`)](#inference-loop-evaluate_model_with_scores)
   - 7.2 [Score Aggregation](#score-aggregation)
   - 7.3 [Full Metrics Computation (`compute_full_metrics`)](#full-metrics-computation-compute_full_metrics)
8. [Results and Analysis](#results-and-analysis)
   - 8.1 [Aggregate Metrics](#aggregate-metrics)
   - 8.2 [Per-Class Performance](#per-class-performance)
   - 8.3 [Precision–Recall Curves](#precisionrecall-curves)
   - 8.4 [ROC Curves](#roc-curves)
   - 8.5 [Per-Class Heatmaps](#per-class-heatmaps)
9. [Discussion](#discussion)
   - 9.1 [CNN vs. HOG Detection: Architectural Trade-offs](#cnn-vs-hog-detection-architectural-trade-offs)
   - 9.2 [Memory Management and OOM Mitigation](#memory-management-and-oom-mitigation)
   - 9.3 [Strengths of the CNN Pipeline](#strengths-of-the-cnn-pipeline)
   - 9.4 [Limitations and Failure Modes](#limitations-and-failure-modes)
   - 9.5 [Cross-Model Comparison Context](#cross-model-comparison-context)
10. [Reproducibility and Output Artefacts](#reproducibility-and-output-artefacts)
11. [Notebook Cell-by-Cell Summary](#notebook-cell-by-cell-summary)
12. [Conclusions](#conclusions)
13. [Appendix: Parameter Reference](#appendix-parameter-reference)

---

## 1. Executive Summary

This document provides a comprehensive description of the experiment implemented in the Jupyter notebook `Experiment_legacy_CNN.ipynb`. The notebook evaluates the **CNN (Convolutional Neural Network)** face recognition pipeline — specifically the `face_recognition_cnn` model configuration — against a curated multi-label dataset of 1,536 images depicting four public figures: **Hugh Jackman**, **Donald Trump**, **Giorgia Meloni**, and **Lionel Messi**, plus a dedicated "None" class for images containing none of those identities. The evaluation follows the identical rigorous methodology and metrics suite originally designed for the InsightFace baseline evaluation, enabling direct cross-model comparison with the HOG, InsightFace, and ViT model experiments.

The CNN pipeline combines dlib's Max-Margin Object Detection (MMOD) CNN face detector with dlib's ResNet-based 128-dimensional face embeddings, matched via cosine similarity. Compared to the HOG pipeline evaluated in the companion notebook, the CNN detector offers superior robustness to pose variations, partial occlusions, and smaller faces, at the cost of significantly higher computational demand and memory footprint — necessitating dedicated out-of-memory (OOM) mitigation strategies including image downscaling and disabled multi-pass detection.

The experiment records subset accuracy, F-beta scores at both macro and micro averaging, precision, recall, precision-at-fixed-recall, recall-at-fixed-precision, per-class confusion statistics, ROC-AUC, and generates publication-quality Precision–Recall and ROC curves alongside per-class heatmap visualisations. All results and charts are persisted to a timestamped experiment directory for auditability and reproducibility.

---

## 2. Introduction and Motivation

The landscape of face recognition spans from lightweight classical methods to large-scale deep learning architectures. Within the ImageEngine project, four model configurations are systematically benchmarked: HOG (classical), CNN (legacy deep learning), InsightFace (modern deep learning), and ViT (transformer-based). Each configuration is evaluated under identical conditions to produce a fair, apples-to-apples performance comparison.

The CNN model evaluated in this notebook occupies an important middle ground. It uses a CNN-based face detector — more accurate than HOG but less powerful than InsightFace's SCRFD detector — paired with the same dlib ResNet embedding extractor used in the HOG pipeline. This combination isolates the impact of detection quality: since both HOG and CNN experiments share the same embedding and matching components, any performance difference is attributable solely to the face detector's ability to locate faces in varied conditions.

The motivation for this experiment is fourfold:

1. **Quantify CNN detection gains** — measure how much the CNN detector improves upon HOG detection when the rest of the pipeline (embedding, matching, threshold) is held constant.
2. **Cross-model comparability** — produce results in an identical metric schema so they can be directly tabulated against HOG, InsightFace, and ViT model results from companion notebooks (`Experiment_legacy_HOG.ipynb`, `Experiment_insightface_only.ipynb`, `Experiment_legacy_ViT.ipynb`).
3. **OOM resilience testing** — establish practical guidelines for deploying the CNN detector in memory-constrained environments (such as the Docker container used for evaluation).
4. **Failure-mode analysis** — understand where the CNN detector excels relative to HOG (profile views, small faces, complex backgrounds) and where it still falls short of InsightFace-class models.

---

## 3. Background: CNN Face Detection and dlib Embeddings

### 3.1 The MMOD CNN Face Detector

The CNN face detector used by the `face_recognition` library is based on dlib's **Max-Margin Object Detection (MMOD)** framework, introduced by King (2015). Unlike the sliding-window HOG approach, the MMOD CNN processes the entire image through a convolutional neural network that outputs a set of face bounding boxes directly. Key characteristics include:

- **End-to-end learning** — the detector is trained as a complete system, jointly optimising for face localisation and classification, rather than relying on hand-crafted HOG features and a separate SVM classifier.
- **Multi-scale robustness** — the network internally handles multiple scales through pooling and strided convolution layers, reducing dependence on input image upsampling.
- **Higher accuracy on difficult faces** — profile views, partially occluded faces, and faces in complex backgrounds are detected more reliably than with the linear HOG+SVM approach.
- **Significantly higher compute cost** — the CNN detector requires substantially more computation per image. On CPU hardware, processing a single high-resolution image can take several seconds, compared to milliseconds for HOG.

The CNN detector's higher accuracy comes at a tangible cost: without GPU acceleration, it is approximately 5–10× slower than HOG and consumes considerably more memory, particularly for high-resolution images.

### 3.2 The `face_recognition` Library and dlib Embeddings

The Python `face_recognition` library, authored by Adam Geitgey, provides a unified interface to dlib's face detection and recognition capabilities. For this experiment:

- **Detection**: the `cnn` backend is selected, directing calls through dlib's MMOD CNN detector via `face_recognition.face_locations(image, model="cnn")`.
- **Embedding**: dlib's pre-trained ResNet model generates **128-dimensional** face descriptors. The model internally performs 5-point landmark alignment, resizes the aligned face to 150×150 pixels, and produces an L2-normalised descriptor.

These 128-dimensional descriptors are identical to those produced in the HOG experiment — the only difference is how faces are localised before embedding extraction.

### 3.3 Cosine Similarity for Cross-Model Parity

Although the `face_recognition` library historically defaults to Euclidean distance for face comparison, this experiment explicitly uses **cosine similarity** to match the methodology of the InsightFace baseline evaluation notebook. The cosine similarity between two embedding vectors **a** and **b** is defined as:

$$
\text{cosine\_similarity}(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{||\mathbf{a}|| \cdot ||\mathbf{b}||}
$$

A higher score indicates greater similarity. The identification threshold of 0.30 determines the minimum score required to accept a match. This threshold was chosen for cross-model consistency rather than per-model optimisation; dlib embeddings typically produce lower cosine similarity magnitudes than InsightFace's 512-dimensional embeddings.

---

## 4. System Architecture

The experiment leverages the ImageEngine project's **modular classification architecture**, which decomposes face recognition into three independent, pluggable layers:

| Layer | Component | This Experiment |
|-------|-----------|-----------------|
| **Detection** | `FaceDetector` | `cnn` — dlib MMOD CNN face detector via `face_recognition` |
| **Embedding** | `EmbeddingExtractor` | `face_recognition` — dlib ResNet, 128-dim descriptors |
| **Matching** | `MatchingMethod` | `cosine_similarity` — normalised dot product |

The `UnifiedClassifier` class orchestrates these three layers. It is instantiated via the `get_classifier("unified", ...)` factory function, which accepts configuration parameters for each layer. This modular design means the exact same evaluation harness used for the HOG experiment can test the CNN configuration without code changes — only parameter swaps.

### 4.1 Key Classes Involved

- **`FaceDetector`** — supports InsightFace models (`buffalo_l`, `buffalo_m`, `buffalo_s`, `antelopev2`) and legacy models (`cnn`, `hog`). For CNN, it delegates to `face_recognition.face_locations(image, model="cnn")`. Batch face detection is also available: the `detect_faces_batch` method groups images by dimension and uses `face_recognition.batch_face_locations` for improved throughput when processing multiple same-sized images. Multi-pass detection (retrying with higher upsampling if no face is found) is configurable but **disabled** in this experiment to avoid out-of-memory issues.
- **`FaceRecognitionEmbedder`** (subclass of `EmbeddingExtractor`) — extracts 128-dimensional face descriptors using dlib's pre-trained ResNet model via the `face_recognition` library. The `extract_embeddings_batch` method converts images from BGR (OpenCV) to RGB format before extracting encodings for all provided bounding boxes simultaneously.
- **`CosineSimilarityMatching`** (subclass of `MatchingMethod`) — computes cosine similarity between query and reference embeddings, scaling the result from [-1, 1] to [0, 1] for consistency with the matching interface.
- **`UnifiedClassifier`** (subclass of `Classifier`) — ties the three components together: loads reference images, extracts reference embeddings at initialisation, then at inference time detects faces, computes embeddings, matches against the reference gallery, and returns identified names with bounding boxes and confidence scores.

### 4.2 Notebook-Level Overrides

The notebook defines its own `evaluate_model_with_scores` function rather than using the `UnifiedClassifier.classify_images` method directly. This is intentional: the notebook's evaluation function captures per-identity continuous similarity scores needed for PR and ROC curve computation, applies custom image resizing for OOM prevention, and manages memory with periodic garbage collection — capabilities not present in the standard classification pipeline, which is optimised for production inference rather than experimental metric computation.

---

## 5. Experimental Setup

### 5.1 Software Environment and Dependencies

The experiment runs inside a Docker container based on Ubuntu 24.04 LTS. Key Python dependencies include:

- `face_recognition` — for CNN detection and dlib embedding extraction
- `opencv-python` (`cv2`) — for image I/O, manipulation, and resizing
- `scikit-learn` — for computing precision, recall, F-beta, ROC curves, AUC, and multi-label binarisation
- `numpy` — for numerical operations on embeddings and scores
- `matplotlib` + `seaborn` — for generating PR curves, ROC curves, and heatmap visualisations
- `tqdm` — for progress bars during evaluation
- `pandas` — available for tabular analysis
- `gc` — Python garbage collector, used explicitly for memory management

The notebook sets the matplotlib backend to `'Agg'` (non-interactive) to ensure compatibility inside containerised or headless environments, while still rendering inline plots via Jupyter's `plt.show()`. The working directory is explicitly set to `/app` via `os.chdir("/app")` to guarantee that all relative image paths resolve correctly.

### 5.2 Model Configuration

The following model configuration is used throughout the experiment:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `MODEL_KEY` | `face_recognition_cnn` | Identifies the pipeline variant |
| `MODEL_DISPLAY` | `CNN (face_recognition)` | Human-readable model name for reports and plots |
| `MODEL_COLOR` | `#3498db` (blue) | Visual colour in plots, distinct from HOG's red and other models |
| `DETECTION_MODEL` | `cnn` | dlib MMOD CNN face detector |
| `EMBEDDING_MODEL` | `face_recognition` | dlib ResNet 128-dim embeddings |
| `MATCHING_METHOD` | `cosine_similarity` | Angle-based matching for cross-model parity |
| `DETECTION_UPSAMPLE` | `0` | **No upsampling** — critical OOM prevention measure |
| `ENABLE_MULTI_PASS` | `False` | **Disabled** — prevents secondary detection pass that could trigger OOM |

The upsample value of 0 and disabled multi-pass detection are the most significant deviations from the default `FaceDetector` configuration (which uses upsample=2 and multi-pass=True). These choices are driven by the CNN detector's memory consumption: upsampling a high-resolution image before running it through the CNN detector can easily exhaust available RAM in the Docker container, crashing the kernel. By operating at native resolution with no upsampling, the experiment sacrifices detection of very small faces in exchange for stable execution across the entire 1,536-image dataset.

### 5.3 Dataset Description

The evaluation uses the **four-people trainset** (`testsets/four-people-trainset.json`), a large, curated dataset comprising **1,536 images** distributed across five classes:

| Class | Image Count | Description |
|-------|-------------|-------------|
| Donald Trump | 351 | Press photos, public appearances, varied lighting and angles |
| Giorgia Meloni | 350 | Press photos, official events, similar diversity |
| Hugh Jackman | 304 | Red-carpet events, candid photos, film publicity stills |
| Lionel Messi | 369 | Match photos, press conferences, casual settings |
| None | 228 | Images containing no target identities or unrecognisable faces |

Despite the filename containing "trainset", the dataset is used purely for evaluation in this experiment — no model training or fine-tuning occurs. The "trainset" terminology is a historical artefact from the dataset's construction pipeline.

**Multi-label images** are present in the dataset. For example, press photographs of Donald Trump and Giorgia Meloni together carry both labels simultaneously. This is a realistic scenario in press-agency workflows and requires the evaluator to handle multi-label prediction correctly.

Each entry in the JSON testset contains:
- `image_name` — the file basename
- `path` — relative path from the project root to the image file
- `other_paths` — alternative paths for multi-label images appearing in multiple class folders
- `labels` — a list of identity labels present in the image

### 5.4 Reference Gallery

The reference gallery is loaded from `data/references.json` and contains exactly **one reference image per identity**:

| Identity | Reference Image |
|----------|----------------|
| Hugh Jackman | `Images/references/HughJackman.jpg` |
| Donald Trump | `Images/references/DonaldTrump.jpg` |
| Giorgia Meloni | `Images/references/GiorgiaMeloni.jpg` |
| Lionel Messi | `Images/references/LionnelMessi.png` |

At initialisation, the `UnifiedClassifier` detects a face in each reference image using the CNN detector, extracts its 128-dimensional dlib embedding, and stores these as the reference gallery. Using a single reference image per identity represents the most constrained gallery setting. Any model will exhibit reference bias under these conditions — if the reference image captures a person at a specific age, hairstyle, or lighting condition, test images that deviate significantly from those conditions will receive lower similarity scores.

### 5.5 Evaluation Parameters

The following hyperparameters govern evaluation, chosen to be **identical** to the InsightFace baseline and HOG evaluation notebooks:

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `FACE_IDENTIFICATION_THRESHOLD` | 0.30 | Minimum cosine similarity to accept an identity match |
| `F1_BETA` | 0.4 | Beta parameter for F-beta score — weights precision more heavily than recall |
| `FIXED_RECALL_LEVEL` | 0.95 | Recall level at which Precision@Recall is reported |
| `FIXED_PRECISION_LEVEL` | 0.95 | Precision level at which Recall@Precision is reported |

The choice of β = 0.4 (< 1) reflects a system design philosophy that prioritises **precision over recall**: in press-agency workflows, misidentifying a person in a photo is more costly than failing to detect them. The F-beta formula weights precision approximately 7.25× more than recall:

$$
F_\beta = (1 + \beta^2) \cdot \frac{\text{precision} \cdot \text{recall}}{\beta^2 \cdot \text{precision} + \text{recall}} \quad \text{where } \beta = 0.4
$$

---

## 6. Methodology

### 6.1 Pipeline Overview

The evaluation follows a sequential pipeline for each of the 1,536 test images:

1. **Load image** via OpenCV (`cv2.imread`).
2. **Resize if needed** — if the longest side exceeds `MAX_IMAGE_DIM` (800 pixels), proportionally downscale to prevent OOM.
3. **Detect faces** using the CNN detector — returns a list of bounding boxes in `(top, right, bottom, left)` format.
4. **Extract embeddings** for each detected face using dlib's ResNet model via `face_recognition`.
5. **Compute cosine similarity** between each face embedding and every reference embedding.
6. **Identify** each face by selecting the reference with the highest similarity score, accepting only if that score exceeds the identification threshold (0.30).
7. **Aggregate** per-image predictions into a set of identified names (or `{"None"}` if no identifiable face was found).
8. **Collect** ground-truth labels and predicted labels across the entire dataset.
9. **Compute metrics** at both macro and micro averaging levels.

### 6.2 Face Detection Stage — CNN (MMOD)

The CNN face detector operates fundamentally differently from HOG:

1. **Input processing**: the input image (optionally downscaled) is fed through a convolutional neural network consisting of multiple convolutional, pooling, and ReLU layers.
2. **Bounding box regression**: the network outputs candidate bounding boxes with associated confidence scores, trained with a max-margin objective that penalises missed detections and false alarms simultaneously.
3. **Non-maximum suppression**: overlapping detections are merged using NMS to produce a clean set of face bounding boxes.

Unlike HOG — which scans a sliding window and classifies HOG features with a linear SVM — the CNN detector performs detection in a single forward pass through the network. This gives it substantially better performance on:

- **Profile views**: faces rotated up to approximately 60° from frontal
- **Partially occluded faces**: faces behind microphones, hands, or other objects
- **Variable lighting**: the CNN's learned features are more invariant to illumination changes than hand-crafted HOG gradients
- **Small faces**: the CNN's internal multi-scale processing handles faces at varying distances better than HOG (though this advantage is partially negated by the MAX_IMAGE_DIM resizing in this experiment)

The upsampling parameter is set to 0 (no upsampling) in this experiment, meaning the image is processed at its (potentially downscaled) native resolution. This is a deliberate trade-off to prevent OOM crashes.

### 6.3 Memory-Aware Image Resizing

One of the most important engineering decisions in this notebook is the `resize_if_needed` function, which caps the longest image dimension at 800 pixels:

```python
MAX_IMAGE_DIM = 800

def resize_if_needed(img, max_dim=MAX_IMAGE_DIM):
    h, w = img.shape[:2]
    if max(h, w) <= max_dim:
        return img
    scale = max_dim / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
```

This is necessary because the dlib CNN detector's memory consumption scales with input image resolution. A 4000×3000 pixel image can require several gigabytes of RAM during CNN inference. In the Docker container environment with approximately 4GB available memory, processing such images without downscaling would crash the kernel.

The choice of 800 pixels as the ceiling is empirically determined to be a safe maximum for the available memory budget. `cv2.INTER_AREA` interpolation is used for the downscaling, which is the recommended method for image decimation as it produces moire-free results. The trade-off is that very small faces in high-resolution images may become too small to detect after downscaling — but this is preferable to a kernel crash.

Additionally, the evaluation function explicitly releases image memory with `del img` as soon as embeddings have been extracted, and runs `gc.collect()` every 50 images to return freed memory to the operating system.

### 6.4 Embedding Extraction Stage

Once faces are detected, the `FaceRecognitionEmbedder` extracts a 128-dimensional descriptor for each face. The embedding extraction pipeline consists of:

1. **Colour space conversion**: OpenCV loads images in BGR format; `face_recognition` expects RGB. The embedder converts with `cv2.cvtColor(image, cv2.COLOR_BGR2RGB)`.
2. **5-point landmark alignment**: dlib internally detects facial landmarks (eye corners, nose tip) and aligns the face to a canonical pose.
3. **Resizing**: the aligned face is resized to 150×150 pixels.
4. **ResNet inference**: the aligned face is passed through a ResNet model trained with metric learning on millions of face images.
5. **Normalisation**: the output 128-dimensional vector is L2-normalised.

The `extract_embeddings_batch` method processes all detected faces in a single image simultaneously, passing the full list of bounding boxes to `face_recognition.face_encodings` for efficiency.

### 6.5 Matching and Identification Stage

For each detected face embedding, the notebook's custom `cosine_similarity` function computes the similarity against every reference embedding:

$$
\text{sim}(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a}}{||\mathbf{a}|| + \epsilon} \cdot \frac{\mathbf{b}}{||\mathbf{b}|| + \epsilon}
$$

where ε = 10⁻⁸ prevents division by zero. Per-identity, only the maximum similarity score is retained (relevant when multiple reference images exist per identity or when multiple detected faces match the same identity). The identity with the highest similarity score is selected, and the face is identified only if that score meets or exceeds the threshold (0.30).

If no face exceeds the threshold, it is counted as an "unknown" face. If no faces in the entire image are identified (or no faces are detected at all), the image's predicted label set is `{"None"}`.

### 6.6 Multi-Label Evaluation Strategy

Because images may contain multiple people (e.g., Donald Trump and Giorgia Meloni in the same press photo), the evaluation treats each image as a **multi-label classification** problem:

- **Ground truth**: a set of identity labels per image (e.g., `{"Donald Trump", "Giorgia Meloni"}`).
- **Prediction**: a set of identity labels per image, derived from all faces detected and identified in that image.

The `MultiLabelBinarizer` from scikit-learn converts these label sets into binary indicator matrices, enabling standard multi-label metrics to be computed. This correctly handles partial matches — detecting Trump but missing Meloni counts as a true-positive for Trump and a false-negative for Meloni, rather than being treated as a complete miss.

### 6.7 Metrics Definitions

The experiment computes the following metrics, all identical to the InsightFace baseline and HOG notebooks:

#### Aggregate Metrics

| Metric | Definition |
|--------|------------|
| **Subset Accuracy** | Fraction of images where the predicted label set *exactly* matches the ground-truth label set. The strictest multi-label accuracy measure. |
| **F-beta (macro)** | F-beta score averaged across classes (unweighted). With β = 0.4, precision is weighted ~7.25× more than recall. |
| **F-beta (micro)** | F-beta score computed globally across all class–sample pairs. |
| **Precision (macro/micro)** | Fraction of positive predictions that are correct. |
| **Recall (macro/micro)** | Fraction of actual positives that are correctly predicted. |
| **P@R=0.95 (macro/micro)** | Precision achievable when recall is fixed at 95%. |
| **R@P=0.95 (macro/micro)** | Maximum recall achievable while maintaining at least 95% precision. |
| **Identification Rate** | Fraction of detected faces that received a positive identity match (above threshold). |

#### Per-Class Metrics

For each class (each identity plus "None"), the notebook computes: True Positives (TP), False Positives (FP), False Negatives (FN), True Negatives (TN), Precision, Recall, F-beta, ROC-AUC, P@R=0.95, and R@P=0.95.

The "None" class receives special treatment for score computation: its score is defined as `1 - max(similarity scores)`. This means an image receives a high "None" score when no reference identity matches well, which is the correct behaviour for a class representing the absence of known identities.

---

## 7. Evaluation Functions: Detailed Walkthrough

### 7.1 Inference Loop (`evaluate_model_with_scores`)

The core evaluation function iterates over every image in the training set and performs the following for each:

1. **Image Loading**: reads the image via `cv2.imread`. If the image cannot be loaded (corrupted file, missing path), it is recorded as a "no face" case and the prediction defaults to `{"None"}`.

2. **Image Resizing**: applies `resize_if_needed` to cap the longest dimension at 800 pixels. This is the CNN experiment's critical OOM prevention mechanism — absent in the HOG notebook because HOG has a dramatically smaller memory footprint.

3. **Face Detection**: calls `classifier.face_detector.detect_faces(img)` using the CNN detector. If no faces are found, the image is classified as `{"None"}` and the no-face counter increments. The original image array is explicitly deleted with `del img` to free memory immediately.

4. **Embedding Extraction**: the function checks whether InsightFace-style aligned embedding extraction is appropriate (it is not, for the CNN/face_recognition pipeline, since `face_recognition` is not an InsightFace model). It falls back to `classifier.embedder.extract_embeddings_batch(img, face_bboxes)`, which extracts 128-dimensional dlib descriptors for all detected faces simultaneously.

5. **Similarity Computation**: for each face embedding, the function iterates over all reference embeddings and computes cosine similarity. Per-identity, only the maximum score is retained. These scores are accumulated in the `image_scores` dictionary, keyed by identity name.

6. **Identity Assignment**: the identity with the highest cosine similarity is selected. If its score meets or exceeds the threshold (0.30), the face is labelled with that identity and `identified_faces` increments; otherwise, `unknown_faces` increments.

7. **Score Aggregation**: after processing all faces in an image, the maximum similarity score per identity across all faces becomes the image-level score. This handles multi-face images correctly — the face that best matches a given identity determines that identity's score for PR/ROC curve computation.

8. **Memory Management**: every 50 images, `gc.collect()` is called explicitly. This is particularly important for the CNN detector, whose underlying C++ allocations in dlib may not be reclaimed by Python's reference-counting garbage collector alone.

The function returns a comprehensive dictionary containing: all predictions (list of label sets), all ground-truth labels, all per-image per-identity scores, total face count, identified face count, unknown face count, and no-face count.

### 7.2 Score Aggregation

The per-image score aggregation strategy is critical for threshold-dependent metrics. For each image and each identity:

$$
\text{score}_{\text{image, identity}} = \max_{f \in \text{faces}} \text{cosine\_similarity}(\text{emb}_f, \text{ref}_{\text{identity}})
$$

If no faces were detected, all identity scores default to 0.0, giving the "None" class a score of 1.0 (since "None" score = 1 − max(identity scores) = 1 − 0 = 1). This correctly pushes faceless images toward the "None" class.

### 7.3 Full Metrics Computation (`compute_full_metrics`)

This function transforms raw evaluation output into the comprehensive metrics suite:

1. **Multi-label binarisation**: converts ground-truth and predicted label lists into binary matrices using `MultiLabelBinarizer.fit_transform` (for ground truth) and `.transform` (for predictions, using the already-fitted vocabulary).

2. **Aggregate metrics**: computes subset accuracy, F-beta (macro/micro), precision (macro/micro), and recall (macro/micro) using scikit-learn functions with `zero_division=0` for safe handling of classes with no predictions.

3. **Per-class analysis**: uses a second `MultiLabelBinarizer` fitted on the union of all observed labels (ground truth ∪ predictions) to ensure classes appearing only in predictions (like "None" as a prediction-only label) are handled correctly. For each class, TP/FP/FN/TN are computed from the binary matrices, and precision/recall/F-beta are derived. The continuous similarity scores drive the PR and ROC curve computation via scikit-learn's `precision_recall_curve` and `roc_curve`.

4. **P@R and R@P extraction**: Precision@Recall=0.95 is computed via linear interpolation on the reversed PR curve. Recall@Precision=0.95 is the maximum recall value among all operating points where precision ≥ 0.95.

5. **Micro-level curve metrics**: a flattened score vector is constructed across all samples and all classes, enabling micro-averaged PR and ROC curves and their associated P@R/R@P values.

---

## 8. Results and Analysis

### 8.1 Aggregate Metrics

The CNN pipeline evaluation on the 1,536-image dataset yielded the following overall performance metrics:

| Metric | Value |
|--------|-------|
| **Subset Accuracy** | 0.4967 |
| **F-beta (macro, β=0.4)** | 0.5466 |
| **F-beta (micro, β=0.4)** | 0.5553 |
| **Precision (macro)** | 0.5300 |
| **Precision (micro)** | 0.5365 |
| **Recall (macro)** | 0.6983 |
| **Recall (micro)** | 0.7116 |
| **P@R=0.95 (macro)** | 0.2700 |
| **P@R=0.95 (micro)** | 0.2115 |
| **R@P=0.95 (macro)** | 0.5172 |
| **R@P=0.95 (micro)** | 0.0000 |
| **Identification Rate** | 1.0000 |

#### Interpretation of Aggregate Results

**Subset Accuracy (0.4967):** Fewer than half of the images have their predicted label set exactly matching the ground truth. This is a strict multi-label metric — even predicting one correct identity while missing a second co-occurring identity in the same image counts as a miss. The value indicates that the CNN pipeline frequently makes partial errors (identifying some but not all people in multi-person images, or incorrectly adding extra identities).

**Precision vs. Recall gap:** Macro precision (0.5300) is substantially lower than macro recall (0.6983). This means the CNN model is *over-identifying* — it frequently assigns identity labels where it should not. The 17 percentage-point gap suggests the low identification threshold (0.30) combined with the CNN detector's aggressive face-finding behaviour leads to many false positive identifications, particularly for bystander faces that partially resemble reference identities.

**F-beta (macro: 0.5466):** Given that β = 0.4 weights precision more heavily, the F-beta score is dragged down by the poor precision. At 0.5466, the precision-weighted harmonic mean confirms that false identifications are a significant problem for this pipeline configuration.

**Identification Rate (1.0000):** This is a striking result — every single detected face was assigned to an identity (similarity score ≥ 0.30). This means the threshold is too permissive: zero faces were classified as "unknown." In a well-calibrated system, the vast majority of bystander faces should fall below the threshold. An identification rate of 100% strongly suggests the cosine similarity threshold of 0.30 should be raised for this model to reduce false positive identifications.

**P@R=0.95 (macro: 0.2700):** When the model is forced to achieve 95% recall across all classes, precision collapses to 27%. This indicates that near-complete detection coverage comes at a severe cost: nearly three out of four positive predictions would be incorrect.

**R@P=0.95 (micro: 0.0000):** The model cannot achieve 95% precision at any recall level when measured across all classes and samples simultaneously. This zero value is a direct consequence of the over-identification problem — there is no threshold at which the model is simultaneously highly precise across all classes in the micro-averaged sense.

**R@P=0.95 (macro: 0.5172):** When measured per-class and averaged, some classes (notably Hugh Jackman at 0.8191 and Lionel Messi at 0.6314) can achieve meaningful recall at 95% precision, while others (especially None at 0.0000) cannot. The macro average of 0.5172 reflects this unevenness.

### 8.2 Per-Class Performance

The per-class metrics reveal dramatically different performance profiles across the five classes:

| Class | Precision | Recall | F-beta (β=0.4) | ROC-AUC | P@R=0.95 | R@P=0.95 | TP | FP | FN | TN |
|-------|-----------|--------|-----------------|---------|----------|----------|----|----|----|-----|
| **Donald Trump** | 0.4991 | 0.7521 | 0.5233 | 0.7857 | 0.2282 | 0.5527 | 264 | 265 | 87 | 920 |
| **Giorgia Meloni** | 0.5446 | 0.6800 | 0.5600 | 0.7813 | 0.2299 | 0.5829 | 238 | 199 | 112 | 987 |
| **Hugh Jackman** | 0.5528 | 0.8947 | 0.5836 | 0.9426 | 0.3379 | 0.8191 | 272 | 220 | 32 | 1012 |
| **Lionel Messi** | 0.6995 | 0.7127 | 0.7013 | 0.8228 | 0.2433 | 0.6314 | 263 | 113 | 106 | 1054 |
| **None** | 0.3540 | 0.4518 | 0.3648 | 0.8259 | 0.3109 | 0.0000 | 103 | 188 | 125 | 1120 |

#### Per-Class Observations

**Hugh Jackman — Best recall (0.8947), best ROC-AUC (0.9426):** The CNN model excels at finding Hugh Jackman. With 272 true positives and only 32 false negatives, the pipeline correctly identifies Jackman in nearly 90% of the images where he appears. This is likely due to Jackman's distinctive facial features and the reference image being well-representative of his typical appearance in the dataset (red-carpet photos). The high ROC-AUC (0.9426) confirms strong ranking quality — the model assigns meaningfully higher similarity scores to true Jackman faces than to non-Jackman faces. Furthermore, R@P=0.95 of 0.8191 means the model can maintain 95% precision while still detecting 82% of Jackman appearances.

**Lionel Messi — Best precision (0.6995), best F-beta (0.7013):** Messi achieves the highest precision among all classes, with only 113 false positives compared to 263 true positives. The 70% precision and 71% recall yield the best F-beta score of 0.7013. The relatively balanced precision-recall trade-off suggests the embedding space separates Messi's face well from other identities. However, 106 false negatives indicate the model still misses roughly 29% of Messi appearances — possibly in action shots where faces are partially obscured or at unusual angles.

**Donald Trump — Highest false positive count (265 FP):** Trump's precision is just below 50% (0.4991), meaning the model misidentifies nearly as many non-Trump faces as Trump faces. The 265 false positives are the highest of any class, suggesting that Trump's facial features (or the reference embedding) overlap with bystanders in the embedding space. Despite this, recall is reasonably high at 75.2%, indicating the model does find Trump in most images where he appears.

**Giorgia Meloni — Moderate across all metrics:** Meloni sits in the middle of all classes, with precision (0.5446), recall (0.6800), and F-beta (0.5600) that are neither the best nor the worst. The 112 false negatives (highest among known identities) suggest that the model sometimes fails to recognise Meloni — possibly in images with unusual lighting, angles, or accessories.

**None — Weakest performer (F-beta: 0.3648, R@P=0.95: 0.0000):** The "None" class is the most challenging. With precision of only 0.3540, the majority of images predicted as "None" are actually false negatives for known identities. More critically, R@P=0.95 is exactly 0.0000, meaning there exists no operating point at which the model can classify "None" images with 95% precision while maintaining any recall. This is a direct consequence of the 100% identification rate — since the CNN detector finds faces aggressively and every detected face exceeds the 0.30 threshold, the system rarely predicts "None" when it should. The 188 false positives for "None" represent images where the model should have identified a known person but instead produced no confident match, while the 125 false negatives represent "None" images where the model incorrectly identified a bystander as a known person.

#### Confusion Pattern Summary

The dominant error pattern across all classes is **false positives** — the model over-identifies. The total false positive count across all classes is 985 (265 + 199 + 220 + 113 + 188), compared to 462 false negatives (87 + 112 + 32 + 106 + 125). This 2:1 ratio of FP to FN confirms the CNN pipeline's tendency to be overly aggressive in assigning identities, driven by the permissive 0.30 cosine similarity threshold and the CNN detector's thorough face-finding capability.

### 8.3 Precision–Recall Curves

The notebook generates a combined PR curve plot with:

- **Macro-average PR curve** (thick navy line) — the average of per-class PR curves interpolated to a common 200-point recall grid.
- **Per-class PR curves** (thinner coloured lines) — showing how each identity's precision trades off against recall as the similarity threshold varies.
- **AUC annotations** — each curve is labelled with its Area Under the Curve value.
- **Fill-under-curve** — a semi-transparent fill provides visual emphasis on the macro-average performance region.

The macro-average PR AUC serves as the single most informative ranking-quality summary. Higher AUC means the model maintains precision across a wider range of recall operating points, providing flexibility for threshold selection in deployment. The per-class curves are expected to show Hugh Jackman with the highest PR AUC (consistent with his 0.9426 ROC-AUC) and the None class with the lowest.

### 8.4 ROC Curves

The ROC curve panel mirrors the PR curve panel:

- **Macro-average ROC curve** — averaged TPR vs. FPR across all classes.
- **Per-class ROC curves** — individual class performance profiles.
- **Random baseline** — the diagonal line (AUC = 0.5), against which model performance is benchmarked.
- **AUC annotations** — per-class and macro-average AUC values in the legend.

The per-class ROC-AUC values range from 0.7813 (Giorgia Meloni) to 0.9426 (Hugh Jackman), with a mean of approximately 0.8317. All classes achieve ROC-AUC substantially above the 0.5 random baseline, confirming that the model's similarity scores carry genuine discriminative information even when the binary predictions at threshold 0.30 are suboptimal. The gap between high ROC-AUC values and moderate precision/F-beta scores suggests that threshold optimisation could significantly improve the binary classification performance without changing the underlying model.

### 8.5 Per-Class Heatmaps

A four-panel horizontal heatmap visualises per-class precision, recall, F-beta, and ROC-AUC. The `YlOrRd` colour map runs from yellow (low values) through orange to red (high values), with numeric annotations on each cell. Key visual patterns:

- **Recall panel**: Hugh Jackman appears as the strongest cell (0.895), with the None class visibly weaker (0.452).
- **Precision panel**: Lionel Messi is the brightest cell (0.700), while None is the weakest (0.354).
- **ROC-AUC panel**: the most uniformly strong panel, with all classes above 0.78, confirming ranking quality is consistently better than binary classification quality.
- **F-beta panel**: reflects the precision weighting, with Lionel Messi strongest (0.701) and None weakest (0.365).

### 8.6 Key Findings and Performance Summary

The CNN evaluation reveals several critical findings:

1. **The 0.30 threshold is too low for this model.** The 100% identification rate means every detected face is being assigned to some identity, producing massive false-positive volumes. Raising the threshold would immediately reduce FP counts across all classes and improve precision, at a controlled cost to recall.

2. **Ranking quality is strong (ROC-AUC 0.78–0.94).** The continuous similarity scores produced by the cosine-similarity matching are discriminative. The model *can* distinguish identities — the problem is the threshold at which binary decisions are made, not the underlying embedding quality.

3. **Hugh Jackman is the easiest identity.** With ROC-AUC 0.9426, recall 0.895, and R@P=0.95 of 0.819, this identity is well-separated in embedding space and well-represented by the reference image.

4. **The None class is the hardest.** F-beta of 0.365 and R@P=0.95 of 0.000 indicate the model cannot reliably distinguish images containing no target identities from those containing them. This is an inherent challenge when the identification threshold is so permissive that every face matches *someone*.

5. **Precision is the bottleneck.** Macro precision (0.530) lags macro recall (0.698) by 17 points. For a system designed to favour precision (β=0.4), this is the primary area requiring improvement — either through threshold tuning, reference gallery expansion, or switching to a higher-dimensional embedding model.

---

## 9. Discussion

### 9.1 CNN vs. HOG Detection: Architectural Trade-offs

The fundamental architectural difference between the CNN and HOG experiments is the face detector:

| Aspect | HOG | CNN |
|--------|-----|-----|
| **Detection method** | Sliding window + HOG features + linear SVM | Single-pass MMOD CNN |
| **Pose robustness** | Frontal only (~±30°) | Profile views (~±60°) |
| **Speed (CPU)** | ~50ms per image | ~500ms–5s per image |
| **Memory footprint** | Low (~100MB) | High (~1–4GB per image) |
| **Small face detection** | Depends on upsampling | Better internal multi-scale handling |
| **Embedding model** | dlib ResNet 128-dim | dlib ResNet 128-dim (identical) |

Because both experiments use the same embedding model (dlib ResNet, 128 dimensions), the same matching method (cosine similarity), and the same threshold (0.30), any performance differences are attributable exclusively to detection quality. This controlled comparison isolates the impact of the face detector within the full recognition pipeline.

### 9.2 Memory Management and OOM Mitigation

The CNN experiment introduces three OOM mitigation strategies not present in the HOG experiment:

1. **Image resize cap** (`MAX_IMAGE_DIM = 800`): all images are downscaled so the longest side never exceeds 800 pixels. This bounds the CNN detector's memory consumption regardless of input resolution.

2. **Disabled upsampling** (`DETECTION_UPSAMPLE = 0`): upsampling increases the effective image resolution fed to the detector. With the CNN detector, even modest upsampling (1×) can push memory usage past the container limit for large images. Setting upsample to 0 processes images at their (already-capped) resolution.

3. **Disabled multi-pass detection** (`ENABLE_MULTI_PASS = False`): the `FaceDetector` class supports retrying detection with higher upsampling if the first pass finds no faces. For CNN, this retry would re-run the expensive CNN detector with increased memory demand, risking OOM. Disabling it accepts that some edge-case images (with only very small faces) will yield no detections.

4. **Explicit garbage collection**: `gc.collect()` is called every 50 images and after image arrays are deleted. This ensures that dlib's C++ memory allocations — which may not be tracked by Python's normal reference-counting GC — are reclaimed.

These strategies collectively make the CNN evaluation feasible in a container with approximately 4GB of RAM, at the cost of reduced detection sensitivity for small and distant faces.

### 9.3 Strengths of the CNN Pipeline

1. **Better face localisation** — the CNN detector finds faces that HOG misses, particularly in challenging conditions (side profiles, partial occlusion, complex backgrounds). This translates to higher recall for known identities.
2. **Same embedding quality** — using the same dlib ResNet embedder means the CNN experiment produces equally discriminative face descriptors. The detection improvement does not come at the cost of embedding degradation.
3. **Unified evaluation framework** — by operating within the same modular architecture as all other model experiments, results are directly comparable without methodological confounds.
4. **Still CPU-capable** — although slower than HOG, the CNN pipeline runs entirely on CPU without requiring GPU hardware, making it deployable in environments where GPUs are unavailable.

### 9.4 Limitations and Failure Modes

1. **Computational cost** — processing all 1,536 images with the CNN detector is significantly slower than with HOG. In production settings, this matters for throughput-sensitive applications.
2. **Memory pressure** — the OOM mitigations (image resizing, no upsampling, no multi-pass) are necessary trade-offs that reduce detection sensitivity. The 800-pixel resize cap means the experiment may miss faces that would be detectable in the full-resolution image.
3. **128-dimensional embedding ceiling** — regardless of improved detection, the 128-dimensional dlib embeddings have less representational capacity than InsightFace's 512-dimensional or ViT's 768-dimensional embeddings. Identity confusion between visually similar individuals is more likely.
4. **Cosine similarity calibration** — dlib embeddings were originally designed for Euclidean distance comparison. The cosine similarity threshold of 0.30 is chosen for cross-model consistency but may not be optimal for this specific embedding space. A per-model threshold search could improve results.
5. **Single reference image** — the gallery contains one reference image per identity, introducing reference bias. The CNN detector's ability to find more faces doesn't help if those additional faces' embeddings are poorly matched due to reference-image limitations.
6. **No alignment differentiation** — the notebook checks for InsightFace-style aligned embedding extraction but correctly falls back to the standard path for `face_recognition` embeddings. However, dlib's internal alignment is fixed (5-point landmarks, 150×150 resize), and there is no way to adjust it from the Python API.

### 9.5 Cross-Model Comparison Context

This experiment is designed to sit alongside evaluations of:

- **HOG (`Experiment_legacy_HOG.ipynb`)** — the minimal baseline, using HOG detection instead of CNN. Any superiority of the CNN experiment over the HOG experiment can be attributed directly to the detector upgrade.
- **InsightFace (`Experiment_insightface_only.ipynb`)** — the state-of-the-art reference, using SCRFD detection and ArcFace 512-dimensional embeddings. Expected to outperform both legacy models significantly due to both superior detection and higher-dimensional, better-trained embeddings.
- **ViT (`Experiment_legacy_ViT.ipynb`)** — uses CLIP Vision Transformer embeddings (768-dim). Strong at generalised visual matching but may lag in face-specific tasks without face-specific fine-tuning.

The shared metric schema (identical thresholds, F-beta parameter, P@R and R@P levels, and evaluation dataset) ensures results from all four notebooks can be directly compared in summary tables, radar charts, or ranked leaderboards.

---

## 10. Reproducibility and Output Artefacts

### 10.1 Timestamped Experiment Directory

Each run of the notebook creates a unique directory under `image_outputs/` with the naming convention:

```
eval_cnn_YYYYMMDD_HHMMSS/
```

This directory contains:

| File | Description |
|------|-------------|
| `per_class_metrics_heatmap.png` | Four-panel heatmap of precision, recall, F-beta, ROC-AUC per class |
| `pr_roc_curves.png` | Combined PR and ROC curve plots with macro-average and per-class curves |

### 10.2 Determinism

The evaluation is deterministic given the same environment:

- **Detection**: the dlib CNN detector is deterministic for the same input image (no random sampling).
- **Embedding**: dlib's ResNet model produces identical outputs for identical aligned face crops.
- **Matching**: cosine similarity is a pure mathematical operation.
- **Resizing**: `cv2.resize` with `INTER_AREA` is deterministic.

Re-running the notebook on the same dataset and same Docker image will produce identical numerical results. The only varying element is the timestamp in the output directory name.

### 10.3 Data Dependencies

The notebook depends on the following external data files:

1. `testsets/four-people-trainset.json` — the evaluation dataset index (1,536 entries)
2. `data/references.json` — the reference gallery specification (4 identities, new format with `reference_images` arrays)
3. `Images/references/` — the actual reference face images (4 files: HughJackman.jpg, DonaldTrump.jpg, GiorgiaMeloni.jpg, LionnelMessi.png)
4. `Images/four-people Testset/` — the actual test images (organised by class: Donald Trump, Giorgia Meloni, Hugh Jackman, Lionel Messi, and None subdirectories)

All paths are relative to the project root (`/app`), and the notebook explicitly sets `os.chdir("/app")` to ensure path consistency inside the Docker container.

---

## 11. Notebook Cell-by-Cell Summary

The notebook consists of fourteen cells — seven code cells interleaved with seven markdown narrative cells — structured as follows:

**Cell 1 (Markdown) — Title and Overview:** Introduces the experiment, names the model under test (`face_recognition_cnn`), and lists all metrics that will be calculated. This cell serves as a self-contained abstract: it names the detection model (CNN), embedding model (`face_recognition`), matching method (euclidean distance, as stated historically, though the code uses cosine similarity), and enumerates every metric category. It allows anyone reviewing the notebook to understand scope without running it.

**Cell 2 (Code) — Environment Setup:** Imports all dependencies: `datetime`, `json`, `os`, `sys`, `numpy`, `pandas`, `pathlib.Path`, `tqdm`, `cv2`, `matplotlib` (with `'Agg'` backend), `seaborn`, and `collections.Counter`. Sets the project root to `/app` in both `sys.path` and the working directory. Imports the project's `get_classifier` and `load_celebrities_from_json` from `src.classification`. Defines all notebook-level constants: `MODEL_KEY`, `MODEL_DISPLAY`, `MODEL_COLOR`, `DETECTION_MODEL`, `EMBEDDING_MODEL`, `MATCHING_METHOD`, `DETECTION_UPSAMPLE` (0), `ENABLE_MULTI_PASS` (False), `FACE_IDENTIFICATION_THRESHOLD` (0.30), `F1_BETA` (0.4), `FIXED_RECALL_LEVEL` (0.95), `FIXED_PRECISION_LEVEL` (0.95). Creates the timestamped experiment output directory. Prints a comprehensive diagnostic summary of all parameter values.

**Cell 3 (Markdown) — Section Header:** "Load Training Set and References."

**Cell 4 (Code) — Data Loading:** Loads the four-people trainset JSON and computes label-distribution statistics using `Counter`. Loads celebrity reference data via `load_celebrities_from_json`, printing names and reference image paths. Extracts the ordered list of identity names from `references.json` for use in evaluation. This cell verifies the data pipeline before the expensive evaluation run.

**Cell 5 (Markdown) — Section Header:** "Define Evaluation Functions."

**Cell 6 (Code) — Evaluation Functions:** The largest and most complex cell in the notebook (~170 lines). Imports `gc` for garbage collection and scikit-learn metrics functions. Defines `MAX_IMAGE_DIM = 800` and the `resize_if_needed` function for OOM prevention. Defines `cosine_similarity(emb1, emb2)` for inline similarity computation. Defines `evaluate_model_with_scores(classifier, train_items, all_identities, threshold)` — the main inference loop with per-image face detection, embedding extraction, similarity scoring, identity assignment, and memory management. Defines `compute_full_metrics(eval_result, all_identities, f_beta, fixed_recall, fixed_precision)` — the comprehensive metrics computation function that produces both summary and detailed output dictionaries.

**Cell 7 (Markdown) — Section Header:** "Run Evaluation — CNN Model."

**Cell 8 (Code) — Model Evaluation:** Instantiates the classifier via `get_classifier("unified", celebrity_data, detection_model="cnn", ...)` with CNN-specific parameters (upsample=0, multi_pass=False). Asserts that reference embeddings were loaded successfully. Runs the full evaluation loop, prints face detection statistics (total/identified/unknown/no-face), computes comprehensive metrics, and prints the formatted results table with all aggregate metrics. Explicitly deletes the classifier object to free memory.

**Cell 9 (Markdown) — Section Header:** "Per-Class Performance."

**Cell 10 (Code) — Per-Class Table:** Prints a detailed per-class metrics table with all confusion matrix entries and derived metrics for every class, formatted for easy comparison.

**Cell 11 (Markdown) — Section Header:** "Per-Class Heatmap."

**Cell 12 (Code) — Heatmap Generation:** Creates a four-panel heatmap (precision, recall, F-beta, ROC-AUC) using seaborn's heatmap function with the `YlOrRd` colour map, annotated with exact values. Saves the figure at 150 DPI to the experiment directory.

**Cell 13 (Markdown) — Section Header:** "PR and ROC Curves."

**Cell 14 (Code) — Curve Generation:** Computes per-class PR and ROC curves from continuous similarity scores. Generates macro-average curves by interpolation. Creates a side-by-side 18×8-inch figure with PR (left) and ROC (right) curves, including per-class curves with colour coding, macro-average curves with fill, the random baseline on the ROC panel, and AUC values in legends. Saves the figure at 150 DPI.

**Cell 15 (Markdown) — Section Header:** "Summary Report."

**Cell 16 (Code) — Summary Report:** Prints a complete final summary capturing all configuration parameters, all aggregate metrics, and per-class F-beta scores in a clean, copy-pasteable format. Reports the output directory path.

This sequential structure ensures the notebook can be executed top-to-bottom in a single pass. Each cell depends only on variables defined in preceding cells. No out-of-order execution is required.

---

## 12. Conclusions

The `Experiment_legacy_CNN.ipynb` notebook implements a rigorous, comprehensive evaluation of the CNN-based face recognition pipeline within the ImageEngine project. By maintaining strict methodological parity with the HOG, InsightFace, and ViT evaluation notebooks — using the same dataset, metric suite, evaluation parameters, and reporting format — it enables meaningful cross-model comparison that isolates the impact of the face detector on overall recognition performance.

The experiment's key contributions are:

1. **Controlled detector comparison** — by sharing the same 128-dimensional dlib embeddings and cosine similarity matching with the HOG experiment, any performance differential is attributable solely to the CNN detector's superior face localisation capabilities.

2. **Production-realistic OOM handling** — the notebook demonstrates practical strategies for deploying the memory-hungry CNN detector in constrained environments: image resizing, disabled upsampling, disabled multi-pass detection, and explicit garbage collection. These strategies are documented as both code and configuration parameters, making them reproducible.

3. **Multi-label evaluation rigour** — the evaluation correctly handles images containing multiple identities, avoids oversimplifying face recognition as single-label classification, and produces both threshold-dependent metrics (subset accuracy, F-beta at a fixed operating point) and threshold-independent metrics (PR curves, ROC curves with AUC) for comprehensive performance characterisation.

4. **Visual reporting** — heatmaps, PR curves, and ROC curves provide intuitive visual summaries alongside numeric tables, supporting both detailed analysis and executive-level reporting.

5. **Full reproducibility** — timestamped outputs, deterministic processing, explicit parameter documentation, and containerised execution support experiment tracking and auditing.

The CNN pipeline represents a middle tier in the ImageEngine model hierarchy: more accurate than HOG (due to superior face detection) but less capable than InsightFace (due to lower-dimensional embeddings and a less powerful recognition model). Its results inform decisions about when the computational cost of CNN detection is justified over HOG's speed, and when a further upgrade to InsightFace is necessary for acceptable accuracy.

---

## 13. Appendix: Parameter Reference

### A. Notebook Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `MODEL_KEY` | `face_recognition_cnn` | Internal model identifier |
| `MODEL_DISPLAY` | `CNN (face_recognition)` | Human-readable model name |
| `MODEL_COLOR` | `#3498db` | Plot colour (blue) |
| `DETECTION_MODEL` | `cnn` | Face detection backend (dlib MMOD CNN) |
| `EMBEDDING_MODEL` | `face_recognition` | Embedding extraction backend (dlib ResNet) |
| `MATCHING_METHOD` | `cosine_similarity` | Similarity computation method |
| `DETECTION_UPSAMPLE` | 0 | No upsampling — OOM prevention |
| `ENABLE_MULTI_PASS` | False | No multi-pass retry — OOM prevention |
| `MAX_IMAGE_DIM` | 800 | Maximum image dimension before downscaling |
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

### D. OOM Mitigation Parameters

| Parameter | Value | Effect |
|-----------|-------|--------|
| `MAX_IMAGE_DIM` | 800 | Images with longest side > 800px are proportionally downscaled |
| `DETECTION_UPSAMPLE` | 0 | No upsampling before CNN detection |
| `ENABLE_MULTI_PASS` | False | No retry at higher upsampling if first pass fails |
| GC interval | Every 50 images | `gc.collect()` called periodically to reclaim memory |
| Image deletion | After embedding | `del img` immediately after embeddings are extracted |

### E. Evaluation Metric Formulas

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

### F. Visualisation Specifications

| Plot | Format | Resolution | Colour Map |
|------|--------|------------|------------|
| PR + ROC Curves | 18×8 inches, side-by-side | 150 DPI | Navy macro-avg, 5-colour per-class palette |
| Per-Class Heatmap | 20×4 inches, four panels | 150 DPI | `YlOrRd` (sequential, yellow→orange→red) |
| Colour palette | `#3498db, #e74c3c, #2ecc71, #f39c12, #8e44ad` | — | Blue, red, green, orange, purple |

---

*This report was generated to document the experiment implemented in `Experiment_legacy_CNN.ipynb` as part of the ImageEngine face recognition evaluation suite.*
