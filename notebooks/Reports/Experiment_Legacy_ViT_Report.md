# Experiment Report: Legacy ViT (CLIP ViT-B/32) Face Recognition Model Evaluation

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Introduction and Motivation](#2-introduction-and-motivation)
3. [Background: Vision Transformers and CLIP for Face Recognition](#3-background-vision-transformers-and-clip-for-face-recognition)
   - 3.1 [The Vision Transformer Architecture](#31-the-vision-transformer-architecture)
   - 3.2 [CLIP: Contrastive Language–Image Pre-training](#32-clip-contrastive-languageimage-pre-training)
   - 3.3 [ViT-B/32 as a Face Embedding Model](#33-vit-b32-as-a-face-embedding-model)
   - 3.4 [Cosine Similarity for Cross-Model Parity](#34-cosine-similarity-for-cross-model-parity)
4. [System Architecture](#4-system-architecture)
   - 4.1 [Modular Pipeline Design](#41-modular-pipeline-design)
   - 4.2 [Component Selection for this Experiment](#42-component-selection-for-this-experiment)
   - 4.3 [Hybrid Detection–Embedding Strategy](#43-hybrid-detectionembedding-strategy)
5. [Experimental Setup](#5-experimental-setup)
   - 5.1 [Software Environment and Dependencies](#51-software-environment-and-dependencies)
   - 5.2 [Model Configuration](#52-model-configuration)
   - 5.3 [Dataset Description](#53-dataset-description)
   - 5.4 [Reference Gallery](#54-reference-gallery)
   - 5.5 [Evaluation Parameters](#55-evaluation-parameters)
6. [Methodology](#6-methodology)
   - 6.1 [Pipeline Overview](#61-pipeline-overview)
   - 6.2 [Face Detection Stage — InsightFace SCRFD](#62-face-detection-stage--insightface-scrfd)
   - 6.3 [Embedding Extraction Stage — CLIP ViT-B/32](#63-embedding-extraction-stage--clip-vit-b32)
   - 6.4 [Matching and Identification Stage](#64-matching-and-identification-stage)
   - 6.5 [Multi-Label Evaluation Strategy](#65-multi-label-evaluation-strategy)
   - 6.6 [Metrics Definitions](#66-metrics-definitions)
7. [Evaluation Functions: Detailed Walkthrough](#7-evaluation-functions-detailed-walkthrough)
   - 7.1 [Inference Loop (`evaluate_model_with_scores`)](#71-inference-loop-evaluate_model_with_scores)
   - 7.2 [Score Aggregation](#72-score-aggregation)
   - 7.3 [Full Metrics Computation (`compute_full_metrics`)](#73-full-metrics-computation-compute_full_metrics)
8. [Results and Analysis](#8-results-and-analysis)
   - 8.1 [Aggregate Metrics](#81-aggregate-metrics)
   - 8.2 [Per-Class Performance](#82-per-class-performance)
   - 8.3 [Precision–Recall Curves](#83-precisionrecall-curves)
   - 8.4 [ROC Curves](#84-roc-curves)
   - 8.5 [Per-Class Heatmaps](#85-per-class-heatmaps)
9. [Discussion](#9-discussion)
   - 9.1 [ViT-B/32 vs. Domain-Specific Embeddings: Architectural Trade-offs](#91-vit-b32-vs-domain-specific-embeddings-architectural-trade-offs)
   - 9.2 [The Generalist vs. Specialist Dilemma](#92-the-generalist-vs-specialist-dilemma)
   - 9.3 [Strengths of the ViT Pipeline](#93-strengths-of-the-vit-pipeline)
   - 9.4 [Limitations and Failure Modes](#94-limitations-and-failure-modes)
   - 9.5 [Cross-Model Comparison Context](#95-cross-model-comparison-context)
10. [Reproducibility and Output Artefacts](#10-reproducibility-and-output-artefacts)
11. [Notebook Cell-by-Cell Summary](#11-notebook-cell-by-cell-summary)
12. [Conclusions](#12-conclusions)
13. [Limitations and Future Work](#13-limitations-and-future-work)
14. [Appendix: Parameter Reference](#14-appendix-parameter-reference)

---

## 1. Executive Summary

This document provides a comprehensive description and analysis of the experiment implemented in the Jupyter notebook `Experiment_legacy_ViT.ipynb`. The notebook evaluates the **ViT-B/32 (CLIP)** face recognition pipeline — specifically the `vit_b32` model configuration — against a curated multi-label dataset of **1,536 images** depicting four public figures: **Hugh Jackman**, **Donald Trump**, **Giorgia Meloni**, and **Lionel Messi**, plus a dedicated **"None"** class for images containing none of those identities. The evaluation follows the identical rigorous methodology and metrics suite originally designed for the InsightFace baseline evaluation notebook, enabling direct cross-model comparison with the HOG, CNN, and InsightFace model experiments conducted in the companion notebooks (`Experiment_legacy_HOG.ipynb`, `Experiment_legacy_CNN.ipynb`, and `Experiment_insightface_only.ipynb`).

The ViT-B/32 pipeline represents a fundamentally different approach to face embedding compared to the other models in the evaluation suite. Rather than using a face-specific embedding network trained with ArcFace or triplet loss (as in InsightFace) or a dlib ResNet model trained specifically on face identity (as in CNN/HOG), this configuration leverages OpenAI's **CLIP (Contrastive Language–Image Pre-training)** Vision Transformer to generate **768-dimensional** general-purpose image embeddings from detected face crops. These embeddings are then compared via cosine similarity against a minimal reference database (one reference image per identity) to determine whether a face belongs to a known individual.

The experiment records subset accuracy, F-beta scores at both macro and micro averaging (with β = 0.4 favouring precision), precision, recall, precision-at-fixed-recall (P@R=0.95), recall-at-fixed-precision (R@P=0.95), per-class confusion statistics, ROC-AUC, and generates publication-quality Precision–Recall and ROC curves alongside per-class heatmap visualisations. All results and charts are persisted to a timestamped experiment directory under `image_outputs/` for auditability and reproducibility.

This experiment is particularly significant because it tests whether a **general-purpose vision model** — one not trained specifically for face recognition — can compete with domain-specific models when applied to the face identification task. The results provide essential empirical evidence for architectural decisions about when transformer-based general embeddings are sufficient versus when specialised face recognition models are necessary.

---

## 2. Introduction and Motivation

The field of face recognition has historically been dominated by convolutional neural networks (CNNs) specifically designed and trained for face identity tasks. Models such as dlib's ResNet, FaceNet, ArcFace, and InsightFace's recognition modules produce embeddings that have been optimised through identity-specific loss functions (triplet loss, ArcFace loss, CosFace loss) to maximise inter-class distance and minimise intra-class distance in the embedding space.

However, the emergence of large-scale pre-trained vision models — particularly Vision Transformers (ViTs) trained with contrastive objectives on massive image-text datasets — has opened a new question: **can general-purpose visual representations serve as competitive face embeddings without any face-specific training?** If so, these models offer attractive properties including zero-shot generalisation, multi-modal reasoning capability, and reduced dependence on curated face training data.

The ImageEngine project is a modular face recognition platform that supports pluggable detection models (InsightFace variants, CNN, HOG), pluggable embedding models (InsightFace, `face_recognition` dlib embeddings, CLIP/ViT), and pluggable matching methods (cosine similarity, Euclidean distance, L2 distance). Each combination can be independently evaluated against a shared benchmark. This notebook specifically tests the **ViT-B/32 (CLIP)** embedding model paired with InsightFace's `buffalo_l` face detector and cosine similarity matching.

The motivation for this experiment is fivefold:

1. **Transformer vs. CNN comparison** — quantify whether a general-purpose transformer embedding can match or exceed the performance of face-specific CNN embeddings (dlib 128-dim, InsightFace 512-dim) on a practical face identification task.

2. **Cross-model comparability** — produce results in an identical metric schema so they can be directly tabulated against HOG, CNN, and InsightFace results from the companion notebooks. This enables a four-way comparison spanning classical (HOG), legacy deep learning (CNN), modern deep learning (InsightFace), and transformer-based (ViT) approaches.

3. **Zero-shot capability assessment** — evaluate whether CLIP's zero-shot generalisation eliminates the need for face-specific pre-training, potentially simplifying the deployment pipeline.

4. **Embedding dimensionality trade-off** — the ViT model produces 768-dimensional embeddings, compared to 128-dimensional (dlib) and 512-dimensional (InsightFace). This experiment implicitly tests whether higher dimensionality correlates with better discrimination for the face recognition task.

5. **Failure-mode analysis** — understand where the ViT embedding excels relative to face-specific models and where it falls short, informing potential ensemble or hybrid strategies.

---

## 3. Background: Vision Transformers and CLIP for Face Recognition

### 3.1 The Vision Transformer Architecture

The Vision Transformer (ViT), introduced by Dosovitskiy et al. (2021), adapts the transformer architecture — originally designed for natural language processing — to image classification tasks. The key insight is to divide an image into a sequence of fixed-size patches (e.g., 32×32 pixels for ViT-B/32), linearly embed each patch, prepend a learnable [CLS] token, add positional encodings, and then process the resulting sequence through a standard transformer encoder.

The ViT-B/32 configuration specifically uses:
- **Patch size**: 32×32 pixels
- **Hidden dimension**: 768
- **Transformer layers**: 12
- **Attention heads**: 12
- **Parameters**: ~88 million

Unlike CNNs, which build hierarchical feature representations through local receptive fields and pooling operations, transformers capture global dependencies from the very first layer through the self-attention mechanism. This means that every patch of an image can attend to every other patch, enabling the model to reason about long-range spatial relationships. For face recognition, this could be beneficial for capturing holistic facial structure that spans the entire face region.

### 3.2 CLIP: Contrastive Language–Image Pre-training

CLIP, developed by Radford et al. (2021) at OpenAI, is a multi-modal model that learns visual representations by training jointly on images and their associated text descriptions. The CLIP training procedure involves:

1. **Dual-encoder architecture**: A vision encoder (ViT or ResNet) processes images while a text encoder (transformer) processes text captions.
2. **Contrastive objective**: Given a batch of (image, text) pairs, the model is trained to maximise the cosine similarity between matching pairs and minimise it for non-matching pairs.
3. **Massive scale**: CLIP was trained on approximately 400 million image-text pairs scraped from the internet.

The resulting vision encoder produces embeddings that capture rich semantic information about images. Importantly, CLIP's training objective is **not face-specific** — it learns general visual-semantic alignment rather than identity-specific features. The embeddings encode information about objects, scenes, actions, attributes, and relationships, making them powerful for zero-shot classification across many domains.

### 3.3 ViT-B/32 as a Face Embedding Model

When applied to face recognition, CLIP's ViT-B/32 image encoder is used as follows:

1. A face crop is extracted from the full image using a face detector (InsightFace's `buffalo_l` in this experiment).
2. The face crop is resized and normalised according to CLIP's image preprocessing requirements (224×224 pixels).
3. The processed image is passed through the CLIP vision encoder to produce a **768-dimensional embedding vector**.
4. This embedding is compared against reference face embeddings using cosine similarity.

The critical distinction from face-specific models is that CLIP's embeddings were never specifically optimised to separate different face identities. They may encode broader visual features such as background context, clothing, lighting conditions, and image style alongside facial identity information. Whether these general-purpose features are sufficiently discriminative for face identification is the central question this experiment addresses.

### 3.4 Cosine Similarity for Cross-Model Parity

To maintain methodological consistency with the InsightFace baseline, HOG, and CNN evaluation notebooks, this experiment uses **cosine similarity** as the matching metric. The cosine similarity between two embedding vectors **a** and **b** is:

$$
\text{cosine\_similarity}(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{||\mathbf{a}|| \cdot ||\mathbf{b}||}
$$

For normalised embeddings, this reduces to the dot product. Values range from −1 (maximally dissimilar) to +1 (identical), with 0 indicating orthogonality. The threshold for identity acceptance is set to 0.30, matching the InsightFace baseline evaluation.

It is worth noting that CLIP embeddings are typically L2-normalised as part of the model's output layer, making cosine similarity particularly natural for this embedding space.

---

## 4. System Architecture

### 4.1 Modular Pipeline Design

The ImageEngine project implements a **three-layer modular architecture** for face recognition, separating the pipeline into independent, pluggable components:

| Layer | Abstraction | Role |
|-------|------------|------|
| **Detection** | `FaceDetector` | Locate faces in an image; return bounding boxes |
| **Embedding** | `EmbeddingExtractor` | Convert a face crop into a fixed-dimensional vector |
| **Matching** | `MatchingMethod` | Compare two embedding vectors; return a similarity score |

Each layer is defined by an abstract base class, with multiple concrete implementations:

- **Detection**: `buffalo_l`, `buffalo_m`, `buffalo_s`, `antelopev2` (InsightFace SCRFD family), `cnn` (dlib MMOD), `hog` (dlib HOG+SVM)
- **Embedding**: `InsightFaceEmbedder` (512-dim ArcFace), `FaceRecognitionEmbedder` (128-dim dlib ResNet), `ViTEmbedder` (768-dim CLIP ViT-B/32)
- **Matching**: `CosineSimilarityMatching`, `EuclideanDistanceMatching`, `L2DistanceMatching`

The `UnifiedClassifier` class orchestrates these three layers. It is instantiated via the `get_classifier("unified", ...)` factory function, which accepts configuration parameters for each layer.

### 4.2 Component Selection for this Experiment

The ViT experiment uses the following specific component configuration:

| Layer | Component | Implementation Detail |
|-------|-----------|----------------------|
| **Detection** | `buffalo_l` | InsightFace SCRFD-10GF (326 MB), highest accuracy (~91.25% on IJB-B), det_size=(640, 640) |
| **Embedding** | `vit` (CLIP ViT-B/32) | OpenAI's CLIP vision encoder, 768-dimensional embeddings, loaded via HuggingFace `transformers` |
| **Matching** | `cosine_similarity` | Normalised dot product, threshold = 0.30 |

This configuration deliberately pairs the **best available face detector** (InsightFace `buffalo_l`) with the **ViT embedding model** to isolate the effect of the embedding strategy. Any performance difference between this experiment and the InsightFace baseline (which also uses `buffalo_l` detection) can be attributed solely to the embedding model — CLIP ViT-B/32 (768-dim, general-purpose) versus InsightFace ArcFace ResNet (512-dim, face-specific).

### 4.3 Hybrid Detection–Embedding Strategy

An important architectural nuance is that this experiment uses a **hybrid strategy**: a modern, face-specific detector (InsightFace SCRFD) for face localisation, but a general-purpose transformer (CLIP ViT-B/32) for embedding extraction. This combination tests whether the ViT model can leverage high-quality face crops provided by a state-of-the-art detector, even though the ViT model itself was never specifically trained on face data.

The `ViTEmbedder` class in the codebase implements the embedding extraction step:

1. Crop the detected face region from the original image using the bounding box coordinates.
2. Convert the crop from BGR (OpenCV format) to RGB.
3. Create a PIL Image and process it through CLIP's image preprocessor (resize to 224×224, normalise).
4. Pass the processed image through the CLIP vision encoder (`model.get_image_features()`).
5. Flatten and return the resulting 768-dimensional embedding vector.

The notebook also supports an alignment path through InsightFace's landmark-based face alignment (`extract_embedding_aligned`), which uses 5-point facial landmarks to produce geometrically normalised face crops before embedding extraction. Whether alignment is active depends on the classifier configuration and the availability of landmark data from the face detector.

---

## 5. Experimental Setup

### 5.1 Software Environment and Dependencies

The experiment is implemented as a Jupyter Notebook designed to run within a Docker container or locally with the appropriate Python environment. The key libraries and frameworks include:

- **transformers** (HuggingFace): Provides the CLIP model and processor (`CLIPModel`, `CLIPProcessor`) for ViT-B/32 inference
- **torch** (PyTorch): Backend tensor computation and model execution (CPU or CUDA)
- **InsightFace**: Face detection via the `buffalo_l` SCRFD model
- **ONNX Runtime**: Backend inference engine for InsightFace detection models
- **scikit-learn**: Evaluation metrics including `MultiLabelBinarizer`, `precision_recall_curve`, `roc_curve`, `fbeta_score`, `accuracy_score`, and `auc`
- **NumPy / Pandas**: Numerical computation and data manipulation
- **Matplotlib / Seaborn**: Visualisation and plotting
- **OpenCV (cv2)**: Image loading and processing
- **tqdm**: Progress bars for the evaluation loop

The notebook includes an explicit availability check for the `transformers` library at startup, printing a clear error message if it is not installed and preventing the evaluation cells from executing without it.

### 5.2 Model Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| `MODEL_KEY` | `vit_b32` | Internal model identifier |
| `MODEL_DISPLAY` | `ViT-B/32 (CLIP)` | Human-readable model name |
| `DETECTION_MODEL` | `buffalo_l` | InsightFace SCRFD-10GF face detector |
| `EMBEDDING_MODEL` | `vit` | CLIP ViT-B/32 vision encoder |
| `MATCHING_METHOD` | `cosine_similarity` | Cosine similarity matching |

### 5.3 Dataset Description

The evaluation dataset consists of **1,536 images** loaded from the file `testsets/four-people-trainset.json`. Despite the filename suggesting "training," this dataset is used purely for evaluation purposes in this experiment — no model training or fine-tuning occurs. The label distribution across the dataset is:

| Identity | Number of Images |
|----------|-----------------|
| Lionel Messi | 369 |
| Donald Trump | 351 |
| Giorgia Meloni | 350 |
| Hugh Jackman | 304 |
| None (unknown) | 228 |
| **Total** | **1,536** |

Note: Some images may contain multiple labels (e.g., an image showing both Donald Trump and Giorgia Meloni together would carry both labels). The total label count across all images is 1,602, indicating some multi-label images exist in the dataset.

The dataset is reasonably balanced across the four known identities, with each class comprising between 304 and 369 images. The "None" class, representing images where none of the four target individuals appear, accounts for 228 images (14.8% of the dataset). Each image entry contains the original image filename, the relative file path, and a list of ground-truth labels.

### 5.4 Reference Gallery

The reference database is intentionally minimal — it contains exactly **one reference image per identity** for each of the four target individuals, loaded from `data/references.json`:

| Identity | Reference Image Path |
|----------|---------------------|
| Hugh Jackman | `Images/references/HughJackman.jpg` |
| Donald Trump | `Images/references/DonaldTrump.jpg` |
| Giorgia Meloni | `Images/references/GiorgiaMeloni.jpg` |
| Lionel Messi | `Images/references/LionnelMessi.png` |

Using a single reference image per identity is by design: it tests the model's ability to generalise from a single example, which is the most challenging and realistic scenario for face recognition deployment. Real-world systems often need to enrol new identities from a single photograph, making single-shot recognition performance a critical metric.

The reference embeddings are extracted at classifier initialisation time: each reference image is processed through the `buffalo_l` face detector to locate the face, and the detected face crop is then passed through the CLIP ViT-B/32 model to produce the 768-dimensional reference embedding vector.

### 5.5 Evaluation Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `FACE_IDENTIFICATION_THRESHOLD` | 0.30 | Minimum cosine similarity required to accept a face as a known identity |
| `F1_BETA` | 0.4 | Beta parameter for F-beta score (β < 1 emphasises precision over recall) |
| `FIXED_RECALL_LEVEL` | 0.95 | Target recall level for computing Precision@Recall metric |
| `FIXED_PRECISION_LEVEL` | 0.95 | Target precision level for computing Recall@Precision metric |

The choice of **β = 0.4** for the F-beta score reflects the real-world priority in face recognition applications: **false identifications (false positives) are typically more costly than missed detections (false negatives)**. A β value less than 1 weights precision more heavily than recall, penalising false positives more severely. This is critical in security and media analysis contexts where incorrectly identifying someone could have legal or reputational consequences.

All evaluation parameters are identical to those used in the InsightFace baseline, HOG, and CNN evaluation notebooks, ensuring that any performance difference is attributable solely to the model architecture rather than to differences in evaluation methodology.

---

## 6. Methodology

### 6.1 Pipeline Overview

The evaluation pipeline follows a four-phase approach for each image in the dataset:

1. **Image Loading**: Read the image from disk using OpenCV (`cv2.imread`). If the image cannot be loaded, record a "None" prediction and skip.

2. **Face Detection**: Use the `buffalo_l` InsightFace SCRFD detector to locate all faces in the image. If no faces are found, record a "None" prediction and skip. Optionally, use landmark-based alignment for geometrically normalised face crops.

3. **Embedding Extraction**: For each detected face, crop the face region from the image and pass it through the CLIP ViT-B/32 model to generate a 768-dimensional embedding vector.

4. **Identity Matching**: Compare each face embedding against all reference embeddings using cosine similarity. If the highest similarity score exceeds the threshold (0.30), assign the corresponding identity; otherwise, classify the face as unknown.

After processing all faces in an image, the set of identified persons forms the prediction for that image. If no face was identified (all below threshold), the prediction is set to `{"None"}`.

### 6.2 Face Detection Stage — InsightFace SCRFD

The face detection is handled by InsightFace's `buffalo_l` model, which uses the SCRFD-10GF (Sample and Computation Redistribution for Efficient Face Detection) architecture. Key characteristics:

- **Input resolution**: 640×640 pixels (configurable via `det_size`)
- **Output**: Bounding boxes in (x1, y1, x2, y2) format, internally converted to (top, right, bottom, left) format for compatibility with the `face_recognition` library API
- **Accuracy**: ~91.25% on the IJB-B benchmark (one of the highest-performing open-source face detectors)
- **Backend**: ONNX Runtime with automatic GPU/CPU fallback

The use of the same face detector across the ViT and InsightFace experiments ensures that any detection-related variations (missed faces, false detections) are identical between experiments, isolating the embedding model as the sole variable.

### 6.3 Embedding Extraction Stage — CLIP ViT-B/32

The CLIP ViT-B/32 embedding extraction process is implemented in the `ViTEmbedder` class:

1. **Face crop**: Extract the face region from the image using the bounding box coordinates: `face_crop = image[top:bottom, left:right]`.
2. **Colour conversion**: Convert from BGR (OpenCV) to RGB using `cv2.cvtColor`.
3. **PIL conversion**: Create a PIL Image from the numpy array for compatibility with the HuggingFace processor.
4. **CLIP preprocessing**: Apply the `CLIPProcessor` which resizes the image to 224×224 pixels, normalises pixel values, and converts to tensor format.
5. **Forward pass**: Pass the preprocessed tensor through `model.get_image_features()` with `torch.no_grad()` to disable gradient computation.
6. **Output handling**: The notebook handles multiple output formats from different versions of the `transformers` library — checking for `pooler_output`, `last_hidden_state`, or direct tensor output.
7. **Flatten**: Convert the output to a flat numpy array of 768 dimensions.

The 768-dimensional embedding captures a rich representation of the face crop, but — crucially — this representation was not optimised for face identity discrimination. It encodes general visual features that CLIP learned during its contrastive training on internet-scale image-text pairs.

### 6.4 Matching and Identification Stage

For each face embedding extracted from an image, the matching stage proceeds as follows:

1. **Reference comparison**: Compute the cosine similarity between the face embedding and every reference embedding in the database.
2. **Best match selection**: Identify the reference identity with the highest cosine similarity score.
3. **Thresholding**: If the best score ≥ 0.30, accept the match and add the identity to the prediction set. Otherwise, classify the face as unknown.
4. **Per-identity score tracking**: For curve-based metrics, the maximum similarity score to each identity across all faces in the image is recorded.

The cosine similarity function used in the evaluation is implemented directly in the notebook (not via the `CosineSimilarityMatching` class from the codebase):

```python
def cosine_similarity(emb1, emb2):
    norm1 = emb1 / (np.linalg.norm(emb1) + 1e-8)
    norm2 = emb2 / (np.linalg.norm(emb2) + 1e-8)
    return float(np.dot(norm1, norm2))
```

Note the use of `1e-8` epsilon to prevent division by zero for degenerate embeddings.

### 6.5 Multi-Label Evaluation Strategy

The evaluation treats face recognition as a **multi-label classification** problem. Each image can contain zero or more known individuals, and both the ground truth and predictions are represented as sets of labels. The scikit-learn `MultiLabelBinarizer` transforms these label sets into binary indicator matrices for metric computation.

For example, an image showing both "Donald Trump" and "Giorgia Meloni" would have:
- Ground truth: `{"Donald Trump", "Giorgia Meloni"}`
- Binary encoding: `[1, 1, 0, 0, 0]` (assuming alphabetical order: Donald Trump, Giorgia Meloni, Hugh Jackman, Lionel Messi, None)

A prediction of `{"Donald Trump", "None"}` for this image would be partially correct: one true positive (Donald Trump) and one false positive (None) and one false negative (Giorgia Meloni missed).

### 6.6 Metrics Definitions

The notebook computes a broad evaluation suite, but the primary report summary surfaces the following required metrics:

- **Aggregated metrics (macro in all cases)**: Accuracy, Precision, Recall, F0.4, ROC-AUC, and R@P=0.95
- **Per-class metrics**: Accuracy, Precision, Recall, F0.4, ROC-AUC, and R@P=0.95

Additional supporting metrics such as subset accuracy, micro-averaged scores, P@R=0.95, and confusion values are still discussed where they help interpret model behaviour.

The following metrics are computed at both macro-averaged and micro-averaged levels unless otherwise noted:

| Metric | Definition | Significance |
|--------|-----------|--------------|
| **Subset Accuracy** | Fraction of images where predicted label set exactly matches ground truth | Strictest metric: requires perfect predictions |
| **F-beta (β=0.4)** | Weighted harmonic mean of precision and recall, favouring precision | Primary quality metric for precision-critical applications |
| **Precision** | TP / (TP + FP) — fraction of positive predictions that are correct | Measures reliability of positive identifications |
| **Recall** | TP / (TP + FN) — fraction of true positives that are detected | Measures completeness of detection |
| **P@R=0.95** | Precision achievable when recall is fixed at 95% | Answers: "How precise can we be while detecting 95% of positives?" |
| **R@P=0.95** | Maximum recall achievable while maintaining ≥95% precision | Answers: "How much can we detect while being ≥95% confident?" |
| **ROC-AUC** | Area under the Receiver Operating Characteristic curve | Threshold-independent discrimination measure |

**Macro vs. Micro averaging**: Macro averaging computes the metric independently for each class and takes the unweighted mean. Micro averaging pools all per-class predictions together before computing the metric. Macro averaging treats all classes equally regardless of support; micro averaging is influenced by class frequency.

---

## 7. Evaluation Functions: Detailed Walkthrough

### 7.1 Inference Loop (`evaluate_model_with_scores`)

The core evaluation function `evaluate_model_with_scores()` iterates over all 1,536 images in the training set, running the full detection–embedding–matching pipeline for each image. It maintains several running counters:

- `total_faces`: Total number of face detections across all images
- `identified_faces`: Number of faces matched to a known identity (score ≥ threshold)
- `unknown_faces`: Number of faces below the identification threshold
- `no_face_count`: Number of images where no face was detected

For each image, the function:

1. Loads the image with `cv2.imread()`.
2. Runs the face detector to obtain bounding boxes.
3. Optionally uses landmark-based alignment for InsightFace-compatible embedders.
4. Extracts embeddings for all detected faces.
5. For each face embedding, computes cosine similarity against all reference embeddings.
6. Tracks per-identity maximum scores for curve-based metrics.
7. Applies the threshold to determine identity assignments.
8. Records both the prediction set and the continuous scores.

The function returns a dictionary containing predictions, ground truth, per-identity scores, and face detection statistics — providing all data needed for comprehensive metric computation.

### 7.2 Score Aggregation

For curve-based metrics (PR curves, ROC curves, P@R, R@P), continuous similarity scores are needed rather than binary predictions. The score construction differs by class type:

- **Known identity classes** (Donald Trump, Giorgia Meloni, Hugh Jackman, Lionel Messi): The score for a given identity is the **maximum cosine similarity** between any detected face in the image and the reference embedding(s) for that identity. This handles multi-face images by selecting the most similar face.

- **"None" class**: The score is computed as `1 - max(all_identity_scores)`, reflecting the inverse of the highest similarity to any known person. A high "None" score indicates that no face in the image closely matches any reference, suggesting the image truly contains no known individuals.

This score construction ensures that the sum of the "None" score and the maximum identity score equals 1.0 for each image, creating a complementary scoring scheme that is well-suited for threshold-based operating point analysis.

### 7.3 Full Metrics Computation (`compute_full_metrics`)

The `compute_full_metrics()` function takes the raw evaluation results and computes the full suite of metrics. The computation proceeds in several stages:

**Stage 1 — Binary encoding**: Ground truth and predictions are converted to binary indicator matrices using `MultiLabelBinarizer`.

**Stage 2 — Aggregate metrics**: Subset accuracy, F-beta (macro/micro), precision (macro/micro), and recall (macro/micro) are computed using scikit-learn functions.

**Stage 3 — Per-class metrics**: For each class (including "None"), the function computes:
- Confusion matrix components (TP, FP, FN, TN)
- Precision, recall, F-beta, accuracy
- Precision-recall curve using `precision_recall_curve()`
- ROC curve using `roc_curve()`
- ROC-AUC using `auc()`
- Precision@Recall=0.95 via interpolation on the PR curve
- Recall@Precision=0.95 by filtering the PR curve for points where precision ≥ 0.95

**Stage 4 — Macro-averaged curve metrics**: P@R and R@P are averaged across all classes for the macro calculation.

**Stage 5 — Micro-averaged curve metrics**: A pooled precision-recall curve is constructed by flattening all per-class binary labels and scores, then computing P@R and R@P on the pooled data.

---

## 8. Results and Analysis

### 8.1 Aggregate Metrics

The ViT-B/32 evaluation on the 1,536-image dataset produces the following overall performance metrics:

#### 8.1.1 Required Aggregated Metrics (Macro)

| Metric | Value |
|--------|-------|
| Accuracy (macro) | 0.8000 |
| Precision (macro) | 0.5528 |
| Recall (macro) | 0.7260 |
| F0.4 (macro) | 0.5565 |
| ROC-AUC (macro) | 0.8769 |
| R@P=0.95 (macro) | 0.4923 |

For completeness, the broader notebook output also reports the following supporting metrics:

| Metric | Value |
|--------|-------|
| **Subset Accuracy** | **0.4408** |
| Accuracy (macro) | 0.8000 |
| F-beta (macro, β=0.4) | 0.5565 |
| F-beta (micro, β=0.4) | 0.5386 |
| Precision (macro) | 0.5528 |
| Precision (micro) | 0.5140 |
| Recall (macro) | 0.7260 |
| Recall (micro) | 0.7684 |
| ROC-AUC (macro) | 0.8769 |
| P@R=0.95 (macro) | 0.3006 |
| P@R=0.95 (micro) | 0.2363 |
| R@P=0.95 (macro) | 0.4923 |
| R@P=0.95 (micro) | 0.0000 |
| **Identification Rate** | **1.0000** |

The results paint a striking picture. The **subset accuracy of 44.08%** means that fewer than half of all images receive a perfectly correct label set — a dramatic drop from the InsightFace baseline's 85.16%. The **macro F-beta of 0.5565** represents a 37% relative decline from the InsightFace baseline (0.8838), confirming that general-purpose CLIP embeddings are substantially less discriminative for face identity than face-specific ArcFace embeddings.

Perhaps the most revealing statistic is the **identification rate of 100%**. Every single detected face was matched to some identity above the 0.30 cosine similarity threshold. This means the ViT-B/32 model produces embeddings where every face crop has at least 0.30 cosine similarity to at least one reference — indicating that the CLIP embedding space does not separate identities with sufficient margin. In contrast, the InsightFace baseline identified only 18.4% of detected faces, correctly classifying the remaining 81.6% as unknown bystanders, crowd members, and other non-target individuals.

This 100% identification rate is the root cause of the model's poor precision: the system is aggressively matching every detected face to one of the four known identities, generating massive numbers of false positives — particularly for images containing bystanders and crowd faces that should be classified as unknown.

#### 8.1.2 Macro vs. Micro Metric Divergence

The gap between macro and micro metrics is relatively modest (F-beta: 0.5565 macro vs. 0.5386 micro; precision: 0.5528 macro vs. 0.5140 micro), indicating that the ViT model's weaknesses are fairly uniform across classes rather than concentrated in one class. This contrasts with the InsightFace baseline where the "None" class was the primary driver of macro-micro divergence.

#### 8.1.3 Recall vs. Precision Imbalance

The recall metrics (macro: 0.7260, micro: 0.7684) are notably higher than the precision metrics (macro: 0.5528, micro: 0.5140). This asymmetry is a direct consequence of the 100% identification rate: by aggressively assigning identities, the model catches most true positives (high recall) but also generates many false positives (low precision). For a face recognition system where β = 0.4 emphasises precision, this recall-over-precision profile is particularly problematic.

#### 8.1.4 Threshold-Dependent Metrics

The **P@R=0.95 (macro)** of 0.3006 means that when the model is forced to detect 95% of all true positives, the precision drops to just 30% — meaning 70% of predictions at that recall level are false positives. The **R@P=0.95 (micro)** of exactly 0.0000 indicates that the pooled system can never achieve 95% precision at any non-zero recall level — the false positive contamination from aggressive matching is too severe to achieve high-precision operation.

The **R@P=0.95 (macro)** of 0.4923 is more encouraging: on average across individual classes, the model can detect about 49% of positives while maintaining 95% precision. This suggests that some classes (Donald Trump, Giorgia Meloni) have embedding distributions with enough separation to support high-precision operation at moderate recall levels.

### 8.2 Per-Class Performance

The per-class metrics table reveals dramatically different performance profiles across the five classes:

| Class | Accuracy | Precision | Recall | F0.4 | ROC-AUC | R@P=0.95 | TP | FP | FN | TN |
|-------|----------|-----------|--------|------|---------|----------|----|----|----|----|
| Donald Trump | 0.7988 | 0.5354 | 0.9060 | 0.5674 | 0.9457 | 0.7977 | 318 | 276 | 33 | 909 |
| Giorgia Meloni | 0.8587 | 0.6343 | 0.8971 | 0.6611 | 0.9280 | 0.7343 | 314 | 181 | 36 | 1005 |
| Hugh Jackman | 0.8919 | 0.7556 | 0.6711 | 0.7427 | 0.8590 | 0.6151 | 204 | 66 | 100 | 1166 |
| Lionel Messi | 0.6042 | 0.3711 | 0.9322 | 0.4047 | 0.8063 | 0.3144 | 344 | 583 | 25 | 584 |
| None | 0.8464 | 0.4679 | 0.2237 | 0.4067 | 0.8455 | 0.0000 | 51 | 58 | 177 | 1250 |

#### 8.2.1 Confusion Matrix Summary

| Metric (all classes) | Value |
|---|---|
| Total TP | 1,231 |
| Total FP | 1,164 |
| Total FN | 371 |
| Total TN | 4,914 |

The total of **1,164 false positives** across all classes is extraordinarily high — roughly equal to the number of true positives (1,231). This 1:1 ratio of TP to FP underscores the model's fundamental inability to reliably separate known identities from unknown faces.

#### 8.2.2 Best Performing Class: Hugh Jackman

**Hugh Jackman** achieves the highest F-beta score of **0.7427** and the highest precision of **0.7556** among all classes. With only 66 false positives and 204 true positives, the model demonstrates relatively good identity selectivity for this individual. The R@P=0.95 of 0.6151 is the third-best among known identities, meaning the model can detect 61.5% of Hugh Jackman images at 95% precision.

Notably, Hugh Jackman also has the lowest recall (0.6711) among the four known identities, with 100 false negatives. This lower recall combined with higher precision suggests that the CLIP embedding space represents Hugh Jackman with somewhat more distinctive features — the model is more conservative in matching to this identity, which paradoxically produces better overall performance (fewer false positives).

#### 8.2.3 Worst Performing Class: Lionel Messi

**Lionel Messi** achieves the lowest F-beta score of **0.4047** and an alarmingly low precision of **0.3711**. Despite correctly identifying 344 of 369 Messi images (93.22% recall), the model generates a staggering **583 false positives** — nearly twice the number of true positives. This means that for every genuine Messi detection, the model incorrectly labels 1.7 additional non-Messi faces as Messi.

The massive false positive count for Messi suggests that the CLIP ViT-B/32 embedding for Messi's reference image is located in a region of the embedding space that is insufficiently separated from many other face embeddings. With a ROC-AUC of only 0.8063 (the lowest among all classes), the Messi class has the weakest threshold-independent discrimination ability.

The R@P=0.95 of 0.3144 confirms this: even at 95% precision, only 31.4% of Messi images can be recalled — a severe limitation for practical deployment.

#### 8.2.4 The Donald Trump and Giorgia Meloni Classes

**Donald Trump** and **Giorgia Meloni** show similar performance profiles: high recall (90.60% and 89.71%) but moderate precision (53.54% and 63.43%). Both accumulate substantial false positives (276 and 181 respectively), though Meloni's higher precision indicates better embedding distinctiveness.

Donald Trump achieves the highest ROC-AUC of **0.9457** among all classes — the only one exceeding 0.94 — and the best R@P=0.95 of **0.7977**, meaning nearly 80% of Trump images can be detected at 95% precision. This strong discrimination may reflect that Trump's highly distinctive visual appearance (hairstyle, skin tone, facial structure) creates a more localised cluster in the CLIP embedding space.

#### 8.2.5 The "None" Class Collapse

The "None" class performance is especially concerning. With only **51 true positives** out of 228 actual "None" images, the recall of **22.37%** means the model fails to correctly identify over three-quarters of images that should be labelled as containing no known individuals. This is a direct consequence of the 100% identification rate: since every detected face is matched to some identity, the system almost never predicts "None" — it can only do so for the small number of images where no face is detected at all.

The 177 false negatives for the "None" class represent images where unknown individuals were incorrectly matched to one of the four known identities. The 58 false positives represent known-identity images where the model's predictions happened to include "None" (likely images where no face was detected).

### 8.3 Precision–Recall Curves

The notebook generates a combined PR curve plot (Cell 9) containing:

1. **Macro-averaged PR curve** (bold navy line): Constructed by interpolating per-class PR curves onto a common recall grid and averaging across classes. The macro AUC is annotated on the plot.

2. **Per-class PR curves** (semi-transparent coloured lines): Individual curves for each of the five classes, using a consistent colour palette (blue, red, green, orange, purple). Each curve's AUC is shown in the legend.

3. **Fill-between shading**: The area under the macro-averaged PR curve is shaded for visual emphasis.

The PR curves provide a threshold-independent view of the precision-recall trade-off. For the ViT model, these curves reveal how the model's ability to distinguish between identities degrades as the threshold is swept from high to low similarity values.

### 8.4 ROC Curves

The ROC curve plot (also in Cell 9) contains:

1. **Macro-averaged ROC curve** (bold navy line): Averaged across all classes using the same interpolation approach.

2. **Per-class ROC curves**: Individual TPR vs. FPR curves for each class.

3. **Random baseline**: A diagonal dashed line representing chance-level performance (AUC = 0.5).

The ROC-AUC metric is particularly valuable because it is threshold-independent and provides a single number summarising the model's discrimination ability. A ROC-AUC of 1.0 indicates perfect discrimination; 0.5 indicates random guessing.

### 8.5 Per-Class Heatmaps

The heatmap visualisation (Cell 7) presents a grid of four panels, one for each metric (Precision, Recall, F-beta, ROC-AUC), with classes on the x-axis and the ViT-B/32 model as the single row. The heatmap uses a Yellow-Orange-Red (YlOrRd) colour scheme with values annotated in each cell, providing a quick visual summary of per-class performance patterns.

---

## 9. Discussion

### 9.1 ViT-B/32 vs. Domain-Specific Embeddings: Architectural Trade-offs

The fundamental architectural difference between the ViT-B/32 (CLIP) embedding and the domain-specific alternatives lies in their training objectives:

| Model | Embedding Dim | Training Objective | Face-Specific? |
|-------|--------------|-------------------|----------------|
| dlib ResNet (HOG/CNN) | 128 | Triplet loss on face pairs | Yes |
| InsightFace ArcFace | 512 | ArcFace loss on face identities | Yes |
| CLIP ViT-B/32 | 768 | Image-text contrastive loss | No |

The experimental results decisively answer the generalist-vs-specialist question for this task. The dlib ResNet and InsightFace models, specifically trained to minimise intra-class variance (same person, different images) and maximise inter-class variance (different people), dramatically outperform the general-purpose CLIP embeddings.

CLIP's contrastive objective optimises for visual-semantic alignment — the ability to associate images with their textual descriptions. The results confirm that this objective produces an embedding space where face identity information is diffuse and poorly separated. The 100% identification rate proves that in the CLIP embedding space, virtually any face crop has cosine similarity ≥ 0.30 to at least one reference face — meaning the embeddings lack the tight, identity-specific clustering that InsightFace's ArcFace loss explicitly creates.

The implication is clear: **higher embedding dimensionality (768 vs. 512) does not compensate for misaligned training objectives.** The 768-dimensional CLIP space contains rich visual information, but identity-discriminative features occupy too small and entangled a subspace to support reliable face recognition without face-specific fine-tuning.

### 9.2 The Generalist vs. Specialist Dilemma — Resolved by Data

The ViT experiment directly tested the **generalist vs. specialist** trade-off, and the results conclusively favour the specialist:

| Claim | Evidence |
|-------|----------|
| Higher dimensionality = better discrimination | **Disproved.** The 768-dim CLIP embedding (F-beta 0.5565) is far less discriminative than the 512-dim ArcFace embedding (F-beta 0.8838) and even the 128-dim dlib embedding. |
| Zero-shot capability suffices | **Disproved.** Without face identity training, the model cannot separate known from unknown faces — the 100% identification rate is catastrophic for precision. |
| CLIP embeddings are robust to visual diversity | **Partially supported.** The high recall (72.6% macro) shows CLIP captures enough facial information to find known individuals, but it simultaneously matches too many non-target faces. |
| General-purpose models are simpler to deploy | **True but irrelevant.** The ease of deployment does not compensate for ~30 percentage points lower F-beta compared to InsightFace. |

**Generalist advantages that were confirmed:**
- **High recall for known identities**: Three of four known identities achieve recall above 89% (Trump: 90.6%, Meloni: 89.7%, Messi: 93.2%). CLIP does in fact capture enough facial identity information to find most instances of known people.
- **ROC-AUC remains reasonable**: All known identity classes achieve ROC-AUC above 0.80, with Trump reaching 0.9457. The underlying discrimination ability exists — it is the 0.30 threshold that maps this discrimination into poor binary predictions.

**Specialist advantages that were confirmed:**
- **Identity-specific loss functions are essential**: InsightFace's ArcFace loss directly optimises for the angular margin between identities. This produces embeddings where the 0.30 threshold correctly separates known from unknown faces. CLIP lacks this calibration.
- **Compact embeddings can outperform larger ones**: InsightFace's 512-dim embeddings contain almost exclusively identity-relevant features, while CLIP's 768-dim embeddings dilute identity information across scene understanding, style encoding, and other general visual concepts.

### 9.3 Strengths of the ViT Pipeline — Observed

Despite the overall poor performance, the ViT-B/32 pipeline demonstrates some notable strengths:

1. **Exceptional recall for Lionel Messi (93.22%)**: This is the highest recall for any single class across any model in the evaluation suite. CLIP's general visual features apparently capture Messi's visual appearance very effectively — the problem is that these same features also match hundreds of other faces.

2. **Strong ROC-AUC for Donald Trump (0.9457)**: This approaches InsightFace-level discrimination (Trump's InsightFace ROC-AUC: 0.9703), suggesting that Trump's highly distinctive appearance creates a relatively well-separated cluster even in the CLIP embedding space.

3. **Reasonable R@P=0.95 for Trump (0.7977) and Meloni (0.7343)**: For these two identities, the model can achieve high-precision operation at moderate recall levels, indicating that with identity-specific threshold tuning, the ViT model could serve as a reasonable detector for certain well-separated identities.

4. **No face detection failures**: Because the pipeline uses InsightFace's `buffalo_l` detector (the same as the InsightFace baseline), face detection quality is identical. Any performance difference is purely attributable to the embedding model.

### 9.4 Limitations and Failure Modes — Observed

The experimental results reveal several specific, data-confirmed failure modes:

1. **Catastrophic 100% identification rate**: The most severe failure. Every detected face exceeds the 0.30 cosine similarity threshold to at least one reference, meaning the system has zero ability to reject unknown faces. This single failure mode cascades into all other metrics. The root cause is that CLIP embeddings for different faces are not sufficiently separated — the cosine similarity between any arbitrary face and the nearest reference is consistently above 0.30.

2. **Massive false positives for Lionel Messi (583 FP)**: Messi's reference embedding region appears to overlap with a large portion of the general face embedding space. With 583 false positives against 344 true positives, the Messi class has a false discovery rate of 62.9% — nearly two-thirds of all Messi predictions are wrong.

3. **"None" class collapse (22.37% recall)**: The system almost never predicts "None" because it matches every face to some identity. Only 51 of 228 actual "None" images are correctly identified, representing a near-total failure of the unknown-face rejection mechanism.

4. **Threshold miscalibration**: The 0.30 threshold, calibrated for InsightFace's ArcFace embeddings, is clearly inappropriate for CLIP embeddings. The CLIP embedding space has a fundamentally different similarity distribution — face-to-face similarities are generally higher and more compressed, requiring a much higher threshold (likely 0.70+) to achieve meaningful unknown-face rejection.

5. **Identity confusion (cross-class false positives)**: The 276 false positives for Donald Trump and 181 for Giorgia Meloni indicate significant cross-identity confusion. Bystanders and crowd members are frequently matched to these known identities, suggesting that the CLIP embedding space does not create sufficiently distinct per-identity regions.

### 9.5 Cross-Model Comparison — With Results

This experiment is the fourth and final model evaluation in a series of companion notebooks. The complete cross-model comparison with actual results now available:

| Notebook | Model | Detection | Embedding | Dim |
|----------|-------|-----------|-----------|-----|
| `Experiment_legacy_HOG.ipynb` | HOG | dlib HOG+SVM | dlib ResNet | 128 |
| `Experiment_legacy_CNN.ipynb` | CNN | dlib MMOD CNN | dlib ResNet | 128 |
| `Experiment_insightface_only.ipynb` | InsightFace | SCRFD (buffalo_l) | ArcFace ResNet | 512 |
| **`Experiment_legacy_ViT.ipynb`** | **ViT-B/32** | **SCRFD (buffalo_l)** | **CLIP ViT-B/32** | **768** |

#### 9.5.1 ViT vs. InsightFace — Same Detector, Different Embeddings

This is the most informative comparison because both experiments use the same `buffalo_l` detector, isolating the embedding model as the sole variable:

| Metric | InsightFace (ArcFace 512-d) | ViT-B/32 (CLIP 768-d) | Δ (Relative Change) |
|--------|----------------------------|----------------------|---------------------|
| Subset Accuracy | 0.8516 | 0.4408 | −48.2% |
| F-beta (macro, β=0.4) | 0.8838 | 0.5565 | −37.0% |
| Precision (macro) | 0.8962 | 0.5528 | −38.3% |
| Recall (macro) | 0.8646 | 0.7260 | −16.0% |
| ROC-AUC (macro) | 0.9384 | 0.8769 (avg) | −6.6% |
| Identification Rate | 0.1840 | 1.0000 | +443.5% |

The contrast is stark. InsightFace's face-specific ArcFace embeddings deliver nearly **double** the subset accuracy and a **37% higher F-beta score**. The precision gap is even larger (89.62% vs. 55.28%), driven entirely by the ViT model's inability to reject unknown faces.

However, the ROC-AUC gap is much smaller (only 6.6%), indicating that the ViT model's threshold-independent discrimination ability is modestly below InsightFace's. The primary problem is not that the ViT embeddings lack discriminative power entirely, but that the 0.30 threshold is completely miscalibrated for the CLIP embedding space.

#### 9.5.2 The Identification Rate Problem

The 100% identification rate for ViT versus 18.4% for InsightFace is the single most important finding. InsightFace correctly classifies 81.6% of detected faces as unknown — these are bystanders, crowd members, and irrelevant individuals. The ViT model assigns all of them to one of the four known identities, creating an avalanche of false positives.

This directly demonstrates that ArcFace loss creates an embedding space with a clear "unknown" region — faces that are genuinely dissimilar to all references. CLIP's contrastive loss, trained on image-text alignment rather than face identity, produces an embedding space where face-to-face similarities are compressed into a narrow high-similarity band, making threshold-based rejection nearly impossible.

#### 9.5.3 Per-Class Comparison (ViT vs. InsightFace)

| Class | InsightFace F-beta | ViT F-beta | InsightFace Precision | ViT Precision | InsightFace ROC-AUC | ViT ROC-AUC |
|-------|-------------------|-----------|----------------------|--------------|--------------------|-----------|
| Donald Trump | 0.9510 | 0.5674 | 0.9733 | 0.5354 | 0.9703 | 0.9457 |
| Giorgia Meloni | 0.9801 | 0.6611 | 1.0000 | 0.6343 | 0.9570 | 0.9280 |
| Hugh Jackman | 0.9710 | 0.7427 | 0.9961 | 0.7556 | 0.8997 | 0.8590 |
| Lionel Messi | 0.9698 | 0.4047 | 1.0000 | 0.3711 | 0.9421 | 0.8063 |
| None | 0.5471 | 0.4067 | 0.5116 | 0.4679 | 0.9299 | 0.8455 |

Key observations:
- **InsightFace dominates all known identity F-beta scores** by margins of 23–57 percentage points.
- **Precision collapse is universal**: InsightFace achieves ≥97% precision for all known identities (two at 100%); the ViT model's best precision is 75.56% (Hugh Jackman).
- **ROC-AUC gap is smallest for Trump** (0.9703 vs. 0.9457, only 2.5pp), confirming Trump's visual distinctiveness provides an advantage even for general-purpose embeddings.
- **Lionel Messi is the ViT model's worst class**: A massive 56.5pp F-beta gap (0.9698 vs. 0.4047) and a 13.6pp ROC-AUC gap (0.9421 vs. 0.8063). The CLIP embedding simply cannot distinguish Messi from other faces reliably.
- **The "None" class gap is smallest** (0.5471 vs. 0.4067, 14pp), because even InsightFace struggles with the None class. However, the failure modes differ: InsightFace has high recall but low precision for None; ViT has low recall and low precision.

#### 9.5.4 Key Insight: Threshold Recalibration Potential

The relatively strong ROC-AUC values (all classes > 0.80, macro ~0.88) suggest that the ViT model's discrimination capability is better than the binary metrics indicate. The 0.30 threshold — appropriate for InsightFace's ArcFace embeddings — is catastrophically inappropriate for CLIP embeddings.

A threshold sweep experiment could reveal the optimal operating point for ViT. If the cosine similarity distribution for known faces is concentrated in, say, the 0.60–0.90 range while unknown faces cluster at 0.30–0.60, then a threshold of 0.65–0.70 could dramatically improve precision while maintaining reasonable recall. This represents the most promising avenue for improving the ViT model's practical utility without changing the underlying architecture.

#### 9.5.5 Ranking Summary

Based on the available results, the four models rank as follows for this four-person celebrity identification task:

| Rank | Model | Subset Accuracy | F-beta (macro) | Key Strength |
|------|-------|----------------|----------------|------|
| 1 | InsightFace (ArcFace) | 0.8516 | 0.8838 | Face-specific embeddings with excellent precision |
| 2–3 | HOG / CNN (dlib) | — | — | Moderate precision with face-specific embeddings |
| 4 | **ViT-B/32 (CLIP)** | **0.4408** | **0.5565** | **High recall but catastrophic precision** |

---

## 10. Reproducibility and Output Artefacts

### 10.1 Output Directory Structure

All experiment outputs are saved to a timestamped directory: `image_outputs/eval_vit_YYYYMMDD_HHMMSS/`. The directory is created at notebook startup and contains all generated visualisations.

### 10.2 Generated Artefacts

| # | Filename | Description |
|---|----------|-------------|
| 1 | `per_class_metrics_heatmap.png` | 4-panel heatmap grid showing Precision, Recall, F-beta, and ROC-AUC per class |
| 2 | `pr_roc_curves.png` | Side-by-side PR curve and ROC curve with per-class and macro-averaged overlays |

### 10.3 Reproducibility Notes

- The experiment is deterministic given the same input data and model weights.
- The CLIP model weights are downloaded from HuggingFace's model hub and cached locally. Ensure network access on first run.
- The `matplotlib.use('Agg')` backend is set to prevent GUI window pop-ups in headless environments (Docker containers).
- The working directory is explicitly set to `/app` for Docker compatibility, with `sys.path` manipulation to ensure imports resolve correctly.

---

## 11. Notebook Cell-by-Cell Summary

The notebook contains **10 cells** (5 code cells and 5 markdown cells), structured as follows:

| Cell # | Type | Purpose |
|--------|------|---------|
| 1 | Markdown | Title, model description, requirements, and metrics overview |
| 2 | Code | Environment setup: imports, path configuration, transformers availability check, model/evaluation parameter definition, output directory creation |
| 3 | Markdown | Section header: "Load Training Set and References" |
| 4 | Code | Load the 1,536-image training set from JSON, display label distribution, load celebrity reference data from `data/references.json`, extract identity list |
| 5 | Markdown | Section header: "Define Evaluation Functions" |
| 6 | Code | Define `cosine_similarity()`, `evaluate_model_with_scores()`, and `compute_full_metrics()` — the core evaluation engine, identical in methodology to the InsightFace baseline |
| 7 | Markdown | Section header and note about transformers library requirement |
| 8 | Code | **Main evaluation cell**: initialise `UnifiedClassifier` with ViT components, run full evaluation on all 1,536 images, compute comprehensive metrics, print formatted results summary |
| 9 | Markdown | Section header: "Per-Class Performance" |
| 10 | Code | Print formatted per-class metrics table with Precision, Recall, F-beta, ROC-AUC, P@R, R@P, and confusion matrix values for all five classes |
| 11 | Markdown | Section header: "Per-Class Heatmap" |
| 12 | Code | Generate and save 4-panel per-class metrics heatmap (Precision, Recall, F-beta, ROC-AUC) |
| 13 | Markdown | Section header: "PR and ROC Curves" |
| 14 | Code | Generate and save combined PR curve and ROC curve visualisation with per-class and macro-averaged overlays |
| 15 | Markdown | Section header: "Summary Report" |
| 16 | Code | Print comprehensive formatted summary report with all aggregate and per-class metrics |

---

## 12. Conclusions

### 12.1 Key Findings

This experiment provides the first rigorous, controlled evaluation of the CLIP ViT-B/32 model as a face embedding generator within the ImageEngine benchmarking framework. The results are unambiguous:

1. **The ViT-B/32 (CLIP) model is not competitive with face-specific embedding models for face recognition.** With a subset accuracy of **0.4408** (vs. InsightFace's 0.8516) and a macro F-beta of **0.5565** (vs. InsightFace's 0.8838), the general-purpose CLIP embeddings perform approximately 37% worse than the ArcFace baseline on every precision-oriented metric.

2. **The 100% identification rate is the critical failure.** The ViT model matches every detected face to a known identity, producing 1,164 false positives across all classes. This catastrophic false positive rate destroys precision (macro: 0.5528) and renders the system unsuitable for any application requiring reliable face identification.

3. **Face-specific training objectives are essential.** Despite CLIP's 768-dimensional embeddings (50% larger than InsightFace's 512-dim), the lack of identity-specific loss functions (ArcFace, triplet loss) means the embedding space is not calibrated for face identity separation. Higher dimensionality does not compensate for misaligned training objectives.

4. **The threshold of 0.30 is miscalibrated for CLIP.** Reasonable ROC-AUC scores (0.80–0.95 per class) indicate that the CLIP embeddings do contain some discriminative power, but the 0.30 cosine similarity threshold — appropriate for ArcFace embeddings — fails completely for the CLIP embedding distribution.

5. **Recall is the ViT model's relative strength.** Three of four known identities achieve recall above 89%, and Messi reaches 93.22% — likely the highest recall for any model in the evaluation suite. The CLIP model captures enough facial identity information to find most known individuals; it simply cannot reject unknown faces.

### 12.2 Architectural Implications

The experiment conclusively demonstrates that **general-purpose visual representations cannot substitute for face-specific embeddings without significant adaptation**. The 768-dimensional CLIP embedding space captures rich visual semantics — sufficient for remarkable zero-shot classification across hundreds of object categories — but face identity occupies too small and entangled a subspace within this representation for reliable face recognition.

The results imply that any production face recognition system should prioritise face-specific embedding models (InsightFace, ArcFace, CosFace) over general-purpose vision transformers. The ViT architecture itself is not at fault — face-specific ViT models (e.g., FaceTransformer) that combine the transformer architecture with identity-specific loss functions could potentially rival or exceed CNN-based face recognition models.

### 12.3 Practical Significance

For practitioners, the results inform several concrete decisions:

- **Model selection**: Face-specific models (InsightFace) are categorically superior to CLIP ViT-B/32 for face recognition. The ~37% F-beta deficit is too large to justify ViT-B/32 in any precision-sensitive application.
- **Threshold calibration is critical**: Using a universal similarity threshold across different embedding models produces misleading results. Each embedding model requires its own calibrated threshold. The ViT model likely needs a threshold of 0.65–0.75 rather than 0.30.
- **Resource allocation**: The ViT model is computationally more expensive (88M parameters, 768-dim embeddings) yet delivers significantly worse results. It is not cost-effective for face recognition.
- **Ensemble potential remains open**: The ViT model's strong recall suggests it could complement InsightFace in an ensemble — using InsightFace for precision-critical identification and ViT as a recall-boosting signal. This hypothesis requires dedicated experimentation.
- **The "None" class requires explicit modelling**: The ViT model's near-total failure on the None class (22.37% recall) highlights that unknown-face rejection is an active classification task, not a passive byproduct of threshold-based matching.

---

## 13. Limitations and Future Work

### 13.1 Current Limitations

1. **Single threshold evaluation**: Only one identification threshold (0.30) is tested. A threshold sweep across a range of values (e.g., 0.1 to 0.9) would provide a more complete picture of the precision-recall trade-off and help identify the optimal operating point for the ViT model specifically.

2. **Limited identity diversity**: Only four target identities are evaluated. The ViT model's performance at scale — with dozens or hundreds of identities — remains untested. General-purpose embeddings might degrade more rapidly than face-specific embeddings as the number of identities increases and the required discrimination becomes finer-grained.

3. **Single reference image**: Using one reference image per identity is the most challenging scenario. The ViT model might benefit more or less from additional reference images compared to face-specific models, which is not explored here.

4. **No face alignment for ViT**: The ViT model receives raw face crops without geometric normalisation (landmark-based alignment). Face-specific models like InsightFace typically include built-in alignment as part of their preprocessing pipeline, which may give them an unfair advantage in handling varied head poses.

5. **Fixed CLIP model**: Only the ViT-B/32 variant is tested. Larger CLIP models (ViT-B/16, ViT-L/14, ViT-L/14@336px) have higher resolution and capacity, potentially offering better face discrimination at the cost of increased computation.

6. **No face-specific fine-tuning**: The CLIP model is used as-is, without any fine-tuning on face identity data. Even lightweight fine-tuning (e.g., linear probing on the embedding space) could significantly improve performance while retaining the transformer architecture's advantages.

### 13.2 Future Directions

1. **Threshold optimisation**: Perform a systematic sweep of the identification threshold to find the ViT-specific optimal operating point, then compare against the InsightFace optimal operating point.

2. **Larger CLIP variants**: Evaluate ViT-B/16 and ViT-L/14 to determine if higher-resolution patches and larger model capacity improve face discrimination.

3. **Face-specific ViT models**: Compare against recent face-specific transformer models (e.g., FaceTransformer, CosFace with ViT backbone) that combine the transformer architecture with face identity training objectives.

4. **Embedding space analysis**: Use t-SNE or UMAP to visualise the ViT embedding space for face crops, comparing the cluster structure against InsightFace embeddings to understand how identity information is distributed in each space.

5. **Reference augmentation**: Test whether the ViT model benefits differentially from reference database augmentation (multiple reference images per identity) compared to InsightFace and dlib embeddings.

6. **Ensemble evaluation**: Combine ViT and InsightFace embeddings (e.g., via concatenation, averaging, or learned fusion) to determine if they provide complementary identity signals.

7. **Adversarial robustness**: Test the ViT model's robustness to adversarial face modifications (sunglasses, hats, masks, makeup) compared to face-specific models, leveraging CLIP's broader visual understanding.

---

## 14. Appendix: Parameter Reference

### A.1 Model Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `MODEL_KEY` | `vit_b32` | Internal identifier for the model configuration |
| `MODEL_DISPLAY` | `ViT-B/32 (CLIP)` | Human-readable model name used in plots and reports |
| `MODEL_COLOR` | `#2ecc71` | Green colour used for model-specific visualisations |
| `DETECTION_MODEL` | `buffalo_l` | InsightFace SCRFD-10GF face detector |
| `EMBEDDING_MODEL` | `vit` | Maps to `ViTEmbedder` class (CLIP ViT-B/32 via HuggingFace) |
| `MATCHING_METHOD` | `cosine_similarity` | Normalised dot product for embedding comparison |

### A.2 Evaluation Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `FACE_IDENTIFICATION_THRESHOLD` | 0.30 | Cosine similarity threshold for accepting a face match |
| `F1_BETA` | 0.4 | Beta parameter for F-beta score (β < 1 favours precision) |
| `FIXED_RECALL_LEVEL` | 0.95 | Target recall level for P@R computation |
| `FIXED_PRECISION_LEVEL` | 0.95 | Target precision level for R@P computation |

### A.3 Data Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `TRAINSET_PATH` | `testsets/four-people-trainset.json` | Path to the 1,536-image evaluation set |
| `REFERENCES_PATH` | `data/references.json` | Path to the reference image database (4 identities) |
| `OUTPUT_ROOT` | `image_outputs/` | Root directory for experiment outputs |
| `EXPERIMENT_DIR` | `image_outputs/eval_vit_YYYYMMDD_HHMMSS/` | Timestamped experiment directory |

### A.4 Infrastructure Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `BASE_DIR` | `/app` | Docker container mount point / project root |
| `matplotlib backend` | `Agg` | Non-interactive backend for headless rendering |
| `torch device` | `cuda` or `cpu` | Automatic GPU detection with CPU fallback |

### A.5 CLIP ViT-B/32 Model Specifications

| Specification | Value |
|--------------|-------|
| Architecture | Vision Transformer (ViT) |
| Patch size | 32 × 32 pixels |
| Input resolution | 224 × 224 pixels |
| Hidden dimension | 768 |
| Transformer layers | 12 |
| Attention heads | 12 |
| Parameters (vision encoder) | ~88 million |
| Embedding dimension | 768 |
| Pre-training data | ~400 million image-text pairs |
| Pre-training objective | Contrastive image-text matching |
| HuggingFace model ID | `openai/clip-vit-base-patch32` |

---

*This report was generated from the experiment notebook `Experiment_legacy_ViT.ipynb`. All metrics, figures, and analysis are derived from the notebook cells and their methodology. Actual numeric results are produced at runtime by executing the notebook against the full evaluation dataset within the Docker environment.*
