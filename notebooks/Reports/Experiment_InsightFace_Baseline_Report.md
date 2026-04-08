# InsightFace Baseline Face Recognition Model Evaluation — Experiment Report

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Introduction and Motivation](#2-introduction-and-motivation)
3. [Experimental Setup](#3-experimental-setup)
4. [Data Description](#4-data-description)
5. [Methodology](#5-methodology)
6. [Overall Model Performance](#6-overall-model-performance)
7. [Per-Class Performance Analysis](#7-per-class-performance-analysis)
8. [Precision-Recall and ROC Curve Analysis](#8-precision-recall-and-roc-curve-analysis)
9. [Threshold-Dependent Metric Analysis](#9-threshold-dependent-metric-analysis)
10. [The "None" Class Problem](#10-the-none-class-problem)
11. [Verification and Validation](#11-verification-and-validation)
12. [Visualizations Produced](#12-visualizations-produced)
13. [Conclusions and Key Observations](#13-conclusions-and-key-observations)
14. [Limitations and Future Work](#14-limitations-and-future-work)

---

## 1. Executive Summary

This notebook presents a comprehensive baseline evaluation of the **InsightFace** deep learning model for face recognition on a curated dataset of four public figures: **Donald Trump**, **Giorgia Meloni**, **Hugh Jackman**, and **Lionel Messi**, plus a **"None"** class representing unknown or unrecognized faces. The experiment evaluates the model's ability to correctly identify known individuals from images using a minimal reference database — a single reference image per identity — without any data augmentation or reference database expansion.

The key findings show that the baseline InsightFace model achieves a **subset accuracy of 85.16%**, a **macro-averaged F-beta score of 0.8838** (with β = 0.4 favoring precision), and a **macro-averaged ROC-AUC of 0.938**. The model demonstrates very high precision (above 97%) for all four known identity classes, but struggles with the "None" class due to a high false positive rate. Across 6,375 detected faces from 1,536 images, only 18.4% were positively identified as known individuals, while 81.6% were classified as unknown — reflecting the model's conservative identification threshold and the multi-face nature of many images.

---

## 2. Introduction and Motivation

Face recognition systems form the backbone of many modern security, media analysis, and identity verification applications. Evaluating such systems rigorously requires not only measuring aggregate accuracy, but also understanding per-class behavior, precision-recall trade-offs, and the handling of unknown (out-of-distribution) faces.

This experiment serves as the **baseline evaluation** for the InsightFace model within a broader project that explores reference database augmentation strategies. By establishing solid baseline metrics before any enhancements, the notebook provides a principled point of comparison for future experiments that expand the reference database with additional embeddings, use different similarity thresholds, or apply data augmentation techniques.

The InsightFace framework is an open-source deep learning toolkit for face analysis that offers state-of-the-art face detection and recognition models based on ArcFace loss functions and ResNet-family backbones. It produces 512-dimensional normalized face embeddings that enable identity comparison via cosine similarity.

---

## 3. Experimental Setup

### 3.1 Software and Libraries

The experiment is implemented as a Jupyter Notebook running within a Docker container on Ubuntu 24.04.1 LTS. The key libraries and frameworks include:

- **InsightFace**: Face detection and recognition model (with `detection` and `recognition` modules)
- **ONNX Runtime**: Backend inference engine for InsightFace models (with CPU and optional CUDA GPU support)
- **scikit-learn**: Evaluation metrics including `MultiLabelBinarizer`, `precision_recall_curve`, `roc_curve`, `fbeta_score`, `accuracy_score`, and `auc`
- **NumPy / Pandas**: Numerical computation and data manipulation
- **Matplotlib / Seaborn**: Visualization and plotting
- **OpenCV (cv2)**: Image loading and processing

### 3.2 Configuration Parameters

The experiment uses the following hyperparameters and configuration values:

| Parameter | Value | Description |
|---|---|---|
| `FACE_IDENTIFICATION_THRESHOLD` | 0.30 | Minimum cosine similarity required to identify a face as a known person |
| `F1_BETA` | 0.4 | Beta parameter for F-beta score (β < 1 emphasizes precision over recall) |
| `FIXED_RECALL_LEVEL` | 0.95 | Target recall level for computing Precision@Recall metric |
| `FIXED_PRECISION_LEVEL` | 0.95 | Target precision level for computing Recall@Precision metric |
| `DET_SIZE` | (640, 640) | Detection input resolution for InsightFace face detector |
| `REFERENCE_DB_ADDITION_THRESHOLD` | 0.65 | Not used in baseline (reserved for augmentation experiments) |
| `SAMPLE_SIZES` | [0] | Baseline only — no additional samples for augmentation |
| `RANDOM_SEED` | 42 | Random seed for reproducibility |

The choice of β = 0.4 for the F-beta score is deliberate: in face recognition applications, **false identifications (false positives) are typically more costly than missed detections (false negatives)**. A β value less than 1 weights precision more heavily than recall, reflecting this real-world priority.

### 3.3 Directory Structure

All experiment outputs — including metric tables, visualizations, and saved figures — are written to a timestamped directory under `/app/image_outputs/reference_tuning_YYYYMMDD_HHMMSS/`. The notebook generates approximately nine distinct visualization files summarizing various aspects of model performance.

---

## 4. Data Description

### 4.1 Training / Evaluation Set

The evaluation dataset consists of **1,536 images** loaded from the file `testsets/four-people-trainset.json`. Despite the filename suggesting "training," this dataset is used purely for evaluation purposes in this experiment — no model training or fine-tuning occurs. The label distribution across the dataset is:

| Identity | Number of Images |
|---|---|
| Lionel Messi | 369 |
| Donald Trump | 351 |
| Giorgia Meloni | 350 |
| Hugh Jackman | 304 |
| None (unknown) | 228 |
| **Total** | **1,536** (note: some images have multiple labels) |

The dataset is reasonably balanced across the four known identities, with each class comprising between 304 and 369 images. The "None" class, representing images where none of the four target individuals appear, accounts for 228 images.

### 4.2 Precomputed Face Embeddings

Rather than running the expensive InsightFace face detection and embedding extraction pipeline during each evaluation, the experiment leverages **precomputed face embeddings** stored in `notebooks/data/face_embeddings_20260131_153514.json`. This file contains:

- **10,120 total entries** (face detections across all images)
- **2,332 unique images** with valid embeddings indexed for fast lookup
- **171 entries skipped** due to null embeddings (no faces detected in those regions)

Each embedding entry includes a 512-dimensional normalized feature vector, the source image path, and associated identity labels. The embeddings were extracted using the InsightFace model's `normed_embedding` output with a detection size of 640×640 pixels.

### 4.3 Reference Database

The baseline reference database is minimal by design — it contains exactly **one reference image per identity** for each of the four target individuals:

| Identity | Reference Embeddings |
|---|---|
| Hugh Jackman | 1 |
| Donald Trump | 1 |
| Giorgia Meloni | 1 |
| Lionel Messi | 1 |

Reference embeddings are cached in a pickle file (`notebooks/data/base_reference_db.pkl`) for efficient reuse. If the cache does not exist, the notebook initializes InsightFace, extracts embeddings from the reference images directly, and saves them to disk.

---

## 5. Methodology

### 5.1 Reference Database Construction

The function `build_reference_embeddings()` constructs the reference database by iterating through the reference metadata (loaded from `data/references.json`). For each reference image, it first checks the precomputed embeddings cache for a matching entry. If found, the precomputed embedding is used directly; otherwise, it falls back to on-the-fly extraction using InsightFace (which requires initializing the full model pipeline).

Each reference entry is stored as a dictionary with three fields: the normalized embedding vector, the main image path, and the face image path.

### 5.2 Face Identification Pipeline

The identification pipeline implemented in `identify_face()` works as follows:

1. **Input**: A query face embedding and the reference database
2. **Comparison**: The query embedding is compared against every embedding in the reference database using **cosine similarity** (computed as the dot product of normalized vectors)
3. **Best match selection**: The identity with the highest cosine similarity score is selected
4. **Thresholding**: If the best similarity score exceeds `FACE_IDENTIFICATION_THRESHOLD` (0.30), the face is classified as that identity; otherwise, it is classified as "unknown"

For each image in the evaluation set, the pipeline processes all detected faces independently. If multiple faces match the same identity, that identity appears only once in the prediction set (set semantics). If no face exceeds the identification threshold, the prediction is set to `{'None'}`.

### 5.3 Multi-Label Evaluation Framework

The evaluation treats face recognition as a **multi-label classification** problem. Each image can contain zero or more known individuals, and the ground truth and predictions are both represented as sets of labels. The scikit-learn `MultiLabelBinarizer` transforms these label sets into binary indicator matrices for metric computation.

The following metrics are computed at both macro-averaged and micro-averaged levels:

- **Subset Accuracy**: Fraction of images where the predicted label set exactly matches the ground truth
- **F-beta Score**: Weighted harmonic mean of precision and recall (β = 0.4)
- **Precision**: Fraction of predicted labels that are correct
- **Recall**: Fraction of true labels that are predicted
- **Precision@Recall=0.95**: Precision achievable when recall is fixed at 95%
- **Recall@Precision=0.95**: Maximum recall achievable while maintaining at least 95% precision
- **ROC-AUC**: Area under the Receiver Operating Characteristic curve

### 5.4 Score Construction for Curve-Based Metrics

For precision-recall and ROC curves, continuous similarity scores are needed rather than binary predictions. The score construction differs by class:

- **Known identity classes** (Donald Trump, Giorgia Meloni, Hugh Jackman, Lionel Messi): The score is the maximum cosine similarity between any face in the image and the reference embeddings for that identity.
- **"None" class**: The score is computed as `1 - max(all_identity_scores)`, reflecting the inverse of the highest similarity to any known person. A high "None" score indicates that no face in the image closely matches any reference.

---

## 6. Overall Model Performance

### 6.1 Aggregate Metrics

The baseline InsightFace evaluation yields the following overall performance metrics:

| Metric | Macro-Averaged | Micro-Averaged |
|---|---|---|
| **Accuracy (Subset / Exact Match)** | 0.8516 | — |
| **Accuracy (Per-Class Binary, Macro Avg.)** | 0.9417 | — |
| **Precision** | 0.8962 | 0.8624 |
| **Recall** | 0.8646 | 0.8571 |
| **F-beta (β=0.4)** | 0.8838 | 0.8617 |
| **ROC AUC** | 0.9384 | 0.8789 |
| **Recall@Precision=0.95 (R@P=0.95)** | 0.7136 | 0.0000 |
| **Precision@Recall=0.95 (P@R=0.95)** | 0.4613 | 0.2824 |

The subset (exact-match) accuracy of 85.16% indicates that for the majority of images the predicted label set exactly matches the ground truth. The macro-averaged per-class binary accuracy of **0.9417** measures classification correctness per class and then averages equally across all five classes. The high macro-averaged precision (89.62%) and recall (86.46%) demonstrate strong performance across individual classes. The F-beta score of 0.8838 (precision-weighted) confirms the model's precision advantage, and the macro-averaged ROC AUC of **0.9384** confirms excellent overall discriminative ability.

### 6.2 Face Detection and Identification Statistics

| Statistic | Value |
|---|---|
| Total faces detected | 6,375 |
| Identified faces | 1,176 (18.4%) |
| Unknown faces | 5,199 (81.6%) |
| Images with no face detections | 106 |

The low identification rate (18.4%) is not a sign of poor model quality. Rather, it reflects the fact that many images contain multiple bystanders, crowd members, or other individuals not in the reference database. With only four target identities and a conservative similarity threshold of 0.30, the vast majority of detected faces are correctly classified as unknown rather than erroneously matched to a known person.

### 6.3 Macro vs. Micro Averaging

The gap between macro (0.8838) and micro (0.8617) F-beta scores reveals a modest class imbalance effect. Macro averaging treats all classes equally, while micro averaging weights by the number of samples. Since the "None" class has lower precision, the micro-averaged metrics are pulled down slightly. The notebook includes dedicated visualizations comparing macro versus micro metrics for F-beta, precision, and recall, rendered as grouped bar charts.

---

## 7. Per-Class Performance Analysis

### 7.1 Detailed Per-Class Metrics

The per-class breakdown reveals dramatically different performance profiles across the five classes:

| Class | Accuracy | Precision | Recall | F-beta (β=0.4) | ROC-AUC | P@R=0.95 | R@P=0.95 | Support |
|---|---|---|---|---|---|---|---|---|
| Donald Trump | 0.9564 | 0.9733 | 0.8319 | 0.9510 | 0.9703 | 0.7503 | 0.9231 | 351 |
| Giorgia Meloni | 0.9707 | 1.0000 | 0.8714 | 0.9801 | 0.9570 | 0.4608 | 0.9000 | 350 |
| Hugh Jackman | 0.9674 | 0.9961 | 0.8388 | 0.9710 | 0.8997 | 0.2135 | 0.8750 | 304 |
| Lionel Messi | 0.9557 | 1.0000 | 0.8157 | 0.9698 | 0.9421 | 0.2942 | 0.8699 | 369 |
| None | 0.8581 | 0.5116 | 0.9649 | 0.5471 | 0.9299 | 0.5876 | 0.0000 | 228 |

### 7.2 Confusion Matrix Summary

| Metric (excl. None) | Value |
|---|---|
| Total TP | 1,153 |
| Total FP | 9 |
| Total FN | 221 |
| Total TN | 4,761 |
| Total Accuracy | 0.9626 |

For the four known identity classes combined, the model achieves a total accuracy of **96.26%** with only **9 false positives** across all classes. This is an exceptionally low false positive rate, underscoring the model's precision advantage.

### 7.3 Best and Worst Performing Classes

- **Best performing class**: Giorgia Meloni with an F-beta score of **0.9801**, achieving perfect precision (1.0000) and a recall of 87.14%. The model produced zero false positives for this identity.
- **Worst performing class**: None with an F-beta score of **0.5471**. While the None class achieves near-perfect recall (96.49%), its precision is only 51.16%, meaning nearly half of all "unknown" predictions are actually known individuals.

The known identity classes all achieve F-beta scores above 0.95, indicating that the InsightFace model is highly reliable at correctly identifying faces of known individuals when they are present. The primary weakness lies in the system's tendency to classify some known faces as unknown (false negatives contributing to the 210 false positives for the None class).

### 7.4 Per-Class Visualizations

The notebook generates multiple per-class visualizations:

- **Bar charts**: Precision, recall, F-beta, accuracy, ROC-AUC, and support for each class, with mean reference lines
- **Heatmaps**: Color-coded grids showing metric values across classes and iterations (with red-yellow-green color mapping)
- **Grouped bar charts**: Side-by-side comparison of precision, recall, and F-beta for all classes
- **Horizontal bar charts**: Per-class recall and ROC-AUC for quick visual comparison

---

## 8. Precision-Recall and ROC Curve Analysis

### 8.1 ROC Curve Results

The ROC analysis provides a threshold-independent evaluation of model discrimination ability:

| Class | ROC-AUC |
|---|---|
| Donald Trump | 0.9703 |
| Giorgia Meloni | 0.9570 |
| Hugh Jackman | 0.8997 |
| Lionel Messi | 0.9421 |
| None | 0.9299 |
| **Macro-average** | **0.9384** |
| **Micro-average** | **0.8789** |

All classes achieve ROC-AUC scores above 0.89, indicating strong discriminative ability across the board. Donald Trump achieves the highest individual ROC-AUC at 0.9703, while Hugh Jackman has the lowest at 0.8997. The macro-averaged ROC-AUC of 0.938 confirms excellent overall model discrimination.

The ROC curve visualization plots all per-class curves alongside macro-averaged (solid navy line) and micro-averaged (dashed pink line) curves, with the random classifier diagonal for reference. The gap between macro (0.938) and micro (0.879) AUC is more pronounced in ROC space, reflecting the impact of the "None" class on the pooled micro calculation.

### 8.2 Precision-Recall Curve Results

The PR curves provide a more informative view for imbalanced classification tasks. The notebook generates:

1. **Macro-averaged PR curves**: Interpolated across all classes with individual class curves shown transparently in the background
2. **Per-class PR curves**: Individual detailed curves for each identity with AUC annotations
3. **Cross-iteration comparison**: Per-class PR curves compared across different augmentation levels (only baseline in this experiment)

Each PR curve plot includes fill-between shading for visual clarity and AUC values in the legend.

### 8.3 AUC Comparison Visualizations

Three side-by-side plots track Precision@Fixed_Recall, PR-AUC, and ROC-AUC across sampling iterations. Since this experiment evaluates only the baseline (sample size = 0), these serve as single-point references for future augmentation experiments.

---

## 9. Threshold-Dependent Metric Analysis

### 9.1 Precision@Recall=0.95

This metric answers: "What precision can the model achieve when detecting 95% of all positive instances?"

| Level | Value |
|---|---|
| Macro-averaged | 0.4613 |
| Micro-averaged | 0.2824 |

The relatively low P@R=0.95 values indicate that forcing the model to detect 95% of all positives requires accepting many false positives, particularly for the "None" class. Per-class values vary significantly: Donald Trump achieves 0.7503, while Hugh Jackman only reaches 0.2135 at 95% recall.

### 9.2 Recall@Precision=0.95

This metric answers: "What's the highest percentage of positive cases the model can detect while being at least 95% confident in its predictions?"

| Level | Value |
|---|---|
| Macro-averaged | 0.7136 |
| Micro-averaged | 0.0000 |

The per-class R@P=0.95 values are:
- Donald Trump: **0.9231** — excellent; the model can detect 92.3% of Donald Trump images while maintaining ≥95% precision
- Giorgia Meloni: **0.9000** — strong performance
- Hugh Jackman: **0.8750** — good performance
- Lionel Messi: **0.8699** — good performance
- None: **0.0000** — the None class never achieves 95% precision

The micro-averaged R@P=0.95 of exactly 0.0000 is a striking result. The notebook dedicates multiple verification cells to investigate and explain this phenomenon, confirming it as mathematically correct rather than a bug.

### 9.3 The R@P Calculation Method

The notebook includes an educational section with a step-by-step demonstration of how the R@P metric is calculated:

1. Generate the full precision-recall curve using `sklearn.metrics.precision_recall_curve()`
2. Filter all curve points where precision ≥ 0.95
3. Among those filtered points, select the maximum recall value
4. If no points achieve the target precision, return 0.0

A dedicated visualization shows the Donald Trump PR curve with the precision threshold line, the valid region (shaded green), and the R@P operating point marked with a red star and annotation.

---

## 10. The "None" Class Problem

### 10.1 Why the None Class Underperforms

The notebook devotes significant analysis to understanding why the "None" class metrics are dramatically worse than the known identity classes. The key findings are:

1. **Precision is only 51.16%**: Out of 430 predictions of "None" (220 TP + 210 FP), nearly half are actually known individuals. These false positives occur when a known person's face is detected but fails to exceed the similarity threshold (0.30). The system then defaults to predicting "None."

2. **Recall is 96.49%**: The model correctly identifies 220 out of 228 actual "None" images. This high recall is natural because any face that fails to match a reference is automatically classified as unknown.

3. **R@P=0.95 is 0.0000**: The None class never achieves 95% precision on its PR curve. The maximum precision for the None class is below 0.95, making it mathematically impossible to find any operating point meeting this constraint.

### 10.2 Impact on Micro-Averaged Metrics

The None class disproportionately affects micro-averaged metrics because micro-averaging pools all predictions across all classes. Specifically:

- The 210 false positives from the None class contaminate the pooled precision calculations
- When computing micro-averaged R@P=0.95, the overall system must achieve 95% precision across all classes simultaneously — the "dilution effect" from the None class makes this impossible
- This is why micro-averaged R@P=0.95 = 0.0000 even though per-class values for known identities range from 0.87 to 0.92

### 10.3 Contextual Interpretation

The notebook concludes that the None class behavior is **expected and correct** in the face recognition context:
- "None" represents faces that don't match any known person
- High precision for "None" requires being very confident about unknown identities
- Similarity scores are continuous, making it inherently difficult to draw a sharp boundary between "unknown" and "low-confidence known"
- The conservative identification threshold (0.30) favors recall for the None class at the expense of its precision

---

## 11. Verification and Validation

The notebook includes four dedicated verification cells that rigorously validate the computed metrics:

### 11.1 Per-Class Metrics Verification (Cell 24)
Confirms that all five classes (including "None") are present in the per-class metrics dictionary, and prints full metric details with TP/FP/FN/TN confusion matrix values for each class.

### 11.2 P@R and R@P Calculation Verification (Cell 25)
Uses the Donald Trump class as a worked example to verify P@R and R@P calculations. Tests both forward and reversed interpolation orders for `np.interp()` and confirms that the stored per-class metric values match independent recalculation.

### 11.3 Micro-Averaged R@P Analysis (Cell 26)
Reconstructs the micro-averaged precision-recall curve from scratch, confirms that precision reaches 0.95 at only a single point (near zero recall), and explains the "dilution effect" caused by pooling the None class with known identities.

### 11.4 Detailed High-Precision Point Analysis (Cell 27)
Identifies all points on the micro-averaged PR curve where precision ≥ 0.95, confirms there is only one such point at near-zero recall, and prints the first 10 points of the curve for manual inspection.

### 11.5 None Class R@P Investigation (Cell 28)
Deep-dive into why the None class achieves R@P=0.95 = 0.0, confirming that the None class's maximum achievable precision falls below 95% and explaining why this is expected given the 210 false positives out of 430 "None" predictions.

---

## 12. Visualizations Produced

The notebook generates a comprehensive suite of visualizations saved to the experiment output directory:

| # | Filename | Description |
|---|---|---|
| 1 | `00_model_performance_summary.png` | 7-panel dashboard: overall metrics bar chart, face detection pie chart, KPI text box, per-class grouped bars, accuracy/precision heatmap, per-class recall bars, per-class ROC-AUC bars |
| 2 | `01_per_class_detailed_breakdown.png` | 6-panel figure: individual bar charts for precision, recall, F-beta, accuracy, ROC-AUC, and support per class, each with mean reference line |
| 3 | `total_evaluation_metrics.png` | 10-panel grid showing each metric (subset accuracy, F-beta macro/micro, precision macro/micro, recall macro, P@R macro/micro, R@P macro/micro) as a function of sample size |
| 4 | `macro_vs_micro_comparison.png` | 3-panel comparison of macro vs. micro for F-beta, precision, and recall |
| 5 | `macro_precision_recall_curves.png` | Macro-averaged PR curves with per-class overlays |
| 6 | `macro_roc_curves.png` | Macro-averaged ROC curves with per-class overlays |
| 7 | `macro_auc_scores_comparison.png` | 3-panel comparison of P@Fixed_Recall, PR-AUC, and ROC-AUC |
| 8 | `per_class_metrics_across_iterations.png` | 7-panel line plots tracking each metric per class across iterations |
| 9 | `per_class_metrics_heatmaps.png` | 7-panel heatmaps with RdYlGn colormap showing class×iteration scores |
| 10 | `per_class_pr_curves_sample_0.png` | Per-class PR curves for the baseline iteration |
| 11 | `per_class_pr_curves_all_iterations_comparison.png` | Per-class PR curves comparison across all iterations |
| 12 | `roc_curve_overall_and_per_class.png` | Combined ROC curve: per-class, macro-avg, micro-avg, and random baseline |
| 13 | R@P visualization | PR curve for Donald Trump with precision threshold, valid region, and R@P operating point |

---

## 13. Conclusions and Key Observations

### 13.1 Model Strengths

1. **Exceptional precision for known identities**: All four known identity classes achieve precision above 97%, with Giorgia Meloni and Lionel Messi at a perfect 100%. Only 9 false positives occurred across 1,162 positive predictions for known identities.

2. **Strong overall discrimination**: The macro-averaged ROC-AUC of 0.938 demonstrates that the InsightFace embedding space effectively separates different identities, even with only a single reference embedding per person.

3. **Robust baseline with minimal references**: Achieving 85.16% subset accuracy and 88.38% macro F-beta with just one reference image per identity is remarkable, validating InsightFace's pretrained embedding quality.

4. **High R@P=0.95 for known identities**: The model can detect 87–92% of each known person's images while maintaining at least 95% prediction accuracy, making it suitable for high-precision applications.

### 13.2 Model Weaknesses

1. **None class precision**: At only 51.16%, roughly half of "unknown" predictions are actually known individuals whose similarity scores fell below the identification threshold. This suggests the threshold of 0.30 may be too conservative, or more reference embeddings per identity are needed.

2. **Recall gaps for known identities**: Recall ranges from 81.6% (Lionel Messi) to 87.1% (Giorgia Meloni), meaning 13–18% of known individuals are missed. These false negatives likely correspond to challenging images with unusual poses, lighting, occlusion, or low resolution.

3. **Micro-averaged metrics degradation**: The pooling of the None class's poor precision significantly drags down micro-averaged metrics, particularly R@P=0.95 which drops to zero for the micro-averaged case.

4. **Single reference limitation**: With only one reference embedding per identity, the system lacks the diversity of embeddings needed to cover natural face appearance variations (different angles, expressions, lighting conditions).

### 13.3 Baseline Significance

This experiment establishes quantitative baselines against which future augmentation strategies can be measured. The metrics to track for improvement include:

- Can augmenting the reference database improve recall above 87% without sacrificing precision?
- Can additional reference embeddings reduce None class false positives?
- What is the optimal number of reference embeddings per identity?
- Does reference database expansion improve P@R=0.95 and R@P=0.95 metrics?

---

## 14. Limitations and Future Work

### 14.1 Limitations

1. **Evaluation set overlap**: The dataset is named "trainset" suggesting potential overlap concerns, though in this experiment it is used purely for evaluation.

2. **Multi-face images**: Images containing multiple faces (bystanders, crowds) inflate face detection counts and the "unknown" category. The evaluation does not distinguish between the target person's face and background faces.

3. **Single threshold evaluation**: Only one identification threshold (0.30) is tested. A threshold sweep would provide more insight into precision-recall trade-offs.

4. **Limited identity diversity**: Only four target identities are evaluated, which may not represent the model's behavior at scale with dozens or hundreds of identities.

5. **Precomputed embeddings assumption**: Using precomputed embeddings bypasses any evaluation of the face detection stage's accuracy or the impact of detection parameters.

### 14.2 Future Directions

1. **Reference database augmentation**: The `SAMPLE_SIZES` parameter is designed to accept multiple values (e.g., `[0, 10, 20, 50, 100]`) for evaluating the impact of adding more reference embeddings per identity.

2. **Threshold optimization**: Sweeping the `FACE_IDENTIFICATION_THRESHOLD` parameter to find the optimal operating point for different precision-recall requirements.

3. **Cross-model comparison**: Evaluating alternative models (HOG-based, CNN-based, ViT-based) on the same dataset to compare with InsightFace's baseline.

4. **Hard negative mining**: Analyzing the 210 false positives for the None class to understand what types of faces are being incorrectly classified as unknown, and whether targeted reference database expansion could address them.

5. **Stratified evaluation**: Breaking down performance by image difficulty (pose variation, occlusion level, image quality) to identify specific failure modes.

---

*This report was generated from the experiment notebook `Experiment_insightface_only.ipynb`. All metrics, figures, and analysis are derived from the executed notebook cells and their outputs.*
