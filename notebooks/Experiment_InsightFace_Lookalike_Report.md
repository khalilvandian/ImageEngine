# InsightFace Lookalike Classification Experiment: Distinguishing Javier Bardem from Jeffrey Dean Morgan

## A Comprehensive Evaluation of Deep Face Recognition Under Extreme Visual Similarity

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Introduction and Motivation](#2-introduction-and-motivation)
3. [Background and Related Work](#3-background-and-related-work)
4. [Experimental Design and Methodology](#4-experimental-design-and-methodology)
5. [Dataset Description and Analysis](#5-dataset-description-and-analysis)
6. [Model Architecture and Technical Details](#6-model-architecture-and-technical-details)
7. [Reference Embedding Construction](#7-reference-embedding-construction)
8. [Classification Pipeline and Decision Rule](#8-classification-pipeline-and-decision-rule)
9. [Results and Performance Analysis](#9-results-and-performance-analysis)
10. [Confusion Matrix Analysis](#10-confusion-matrix-analysis)
11. [Similarity Score Distribution Analysis](#11-similarity-score-distribution-analysis)
12. [ROC Curve and AUC Analysis](#12-roc-curve-and-auc-analysis)
13. [Precision-Recall Curve Analysis](#13-precision-recall-curve-analysis)
14. [Decision Margin Analysis](#14-decision-margin-analysis)
15. [Score Separation and Calibration](#15-score-separation-and-calibration)
16. [Threshold Sensitivity Analysis](#16-threshold-sensitivity-analysis)
17. [Per-Class Performance Breakdown](#17-per-class-performance-breakdown)
18. [Error Analysis](#18-error-analysis)
19. [Comparison with Baseline and Random Classifiers](#19-comparison-with-baseline-and-random-classifiers)
20. [Discussion](#20-discussion)
21. [Limitations](#21-limitations)
22. [Conclusions and Future Work](#22-conclusions-and-future-work)
23. [References](#23-references)

---

## 1. Executive Summary

This report presents a comprehensive evaluation of the InsightFace deep face recognition model applied to a uniquely challenging binary classification task: distinguishing between two celebrity lookalikes, Javier Bardem and Jeffrey Dean Morgan. These two actors are widely recognised in popular culture as being remarkably similar in facial appearance, making them an ideal stress test for modern face recognition systems.

The experiment was conducted on a dataset of 1,080 test images (465 of Javier Bardem and 615 of Jeffrey Dean Morgan) using a single reference embedding per person extracted from the InsightFace `buffalo_l` model. The classification decision was made purely on the basis of maximum cosine similarity — each test image was assigned the label of whichever reference embedding yielded the highest cosine similarity score, with no threshold and no "unknown" class.

The results are outstanding. The model achieved an overall accuracy of **98.61%** (1,065 out of 1,080 images correctly classified), a balanced accuracy of **98.73%**, a macro-averaged F1 score of **0.9859**, a Cohen's Kappa of **0.9718**, a Matthews Correlation Coefficient (MCC) of **0.9720**, and a ROC AUC of **0.9997**. Only 15 images were misclassified out of the entire test set, and 2 images had no detectable face. These results strongly demonstrate that InsightFace's deep embedding space captures identity-discriminative features that go well beyond superficial facial similarity, enabling near-perfect separation of two individuals that the human eye often confuses.

---

## 2. Introduction and Motivation

Face recognition has advanced dramatically in the past decade, driven by deep learning architectures that learn compact, discriminative facial embeddings from large-scale datasets. Models such as ArcFace, CosFace, and their successors — many of which are bundled in the InsightFace library — have achieved superhuman performance on standard benchmarks like LFW (Labeled Faces in the Wild), CFP-FP (Celebrities in Frontal-Profile), and MegaFace. These benchmarks typically report verification accuracy above 99.5%, leading to widespread adoption of face recognition technology in consumer electronics, border control, surveillance, and social media applications.

However, standard benchmarks often present a somewhat idealised view of face recognition difficulty. They typically evaluate the system's ability to distinguish between individuals who are visually distinct. The test pairs in LFW, for instance, rarely include individuals who share striking facial resemblance. A far more revealing and practically meaningful test is to evaluate the system on **lookalikes** — pairs of individuals who share strikingly similar facial features, bone structure, complexion, hair style, and overall appearance. The lookalike scenario represents the worst-case adversarial condition for a face recognition system operating in the wild.

Javier Bardem (born 1 March 1969, Las Palmas de Gran Canaria, Spain) and Jeffrey Dean Morgan (born 22 April 1966, Seattle, Washington, USA) are two accomplished actors who have been frequently compared in popular media due to their remarkable physical resemblance. Both have strong jawlines, similar facial proportions, dark hair, and comparable complexions. Internet memes, social media posts, and entertainment articles have extensively documented their similarity, with many people genuinely unable to tell them apart in photographs, particularly when images are taken under varying lighting conditions, angles, or expressions.

This experiment was designed to answer a specific research question: **Can a state-of-the-art deep face recognition model, using only a single reference image per person, reliably distinguish between two individuals who are visually near-identical to the human eye?**

The motivation is both scientific and practical. Scientifically, it probes the representational capacity of deep facial embeddings — whether they encode identity information that transcends surface-level visual similarity. Practically, it addresses a real-world concern: in surveillance, law enforcement, and identity verification systems, the ability to distinguish between lookalikes is critical. False identifications of lookalikes can have serious legal and personal consequences.

---

## 3. Background and Related Work

### 3.1 Deep Face Recognition

Modern face recognition systems operate in three stages: face detection, face alignment, and face embedding extraction. The detection stage locates faces in an image and produces bounding boxes. The alignment stage normalises the detected face to a canonical pose using facial landmark positions. The embedding stage passes the aligned face through a deep convolutional neural network (CNN) to produce a compact feature vector (typically 128 or 512 dimensions) in a learned metric space where distances correspond to identity similarity.

The training objective for these embedding networks is usually a variation of metric learning. ArcFace (Deng et al., 2019), which is the backbone used by InsightFace's `buffalo_l` model, uses an additive angular margin loss that enforces an angular gap between identity clusters in the embedding space. This additive margin in the angular domain translates to tighter, more discriminative clusters compared to earlier approaches like softmax loss, contrastive loss, or triplet loss.

### 3.2 InsightFace and the Buffalo_L Model

InsightFace is an open-source 2D and 3D deep face analysis toolbox maintained by the DeepInsight team. It provides a unified pipeline for face detection (using RetinaFace or SCRFD detectors), face alignment, face recognition (using ArcFace-based models), and face attribute analysis (age, gender, etc.).

The `buffalo_l` model pack, which is the default and largest model in InsightFace's model zoo, includes:
- **Detection model**: SCRFD (Sample and Computation Redistribution for Efficient Face Detection) with 10GF computational budget
- **Recognition model**: ArcFace with a ResNet-100 backbone trained on the Glint360K dataset (approximately 360,000 identities and 17 million images)
- **Embedding dimensionality**: 512-dimensional normalised vectors

The recognition model produces L2-normalised embeddings, meaning that cosine similarity between two embeddings is equivalent to their dot product. This is the similarity metric used throughout this experiment.

### 3.3 Lookalike Recognition Challenges

The problem of distinguishing lookalikes has received relatively limited attention in the face recognition literature compared to general face verification and identification. Notable works include:
- The **Disguised Faces in the Wild (DFW)** challenge, which evaluates recognition under disguise
- Studies on **kinship recognition** (recognising family resemblance), which inherently involves similar-looking individuals
- Work on **doppelgänger detection** by researchers at Cornell and elsewhere

Most standard face recognition benchmarks do not specifically target the lookalike problem, which makes this experiment a valuable contribution to understanding model capabilities under extreme similarity conditions. The lookalike scenario is particularly important because it tests the model's representational limits — if a model can distinguish between two individuals that even humans routinely confuse, it demonstrates that the learned embeddings capture identity information at a level of granularity that exceeds conscious human perception. This has profound implications for both the practical deployment of face recognition systems and for our scientific understanding of what information deep neural networks extract from facial images.

Moreover, the lookalike problem has practical relevance beyond academic curiosity. In judicial contexts, defendants have been wrongly accused based on resemblance to the actual perpetrator. In entertainment and media, celebrity lookalikes are routinely used as stand-ins and impersonators. In social engineering and fraud, attackers sometimes attempt to impersonate individuals by exploiting their physical resemblance to authorised personnel. A face recognition system that can reliably distinguish between lookalikes provides a critical safeguard against all these scenarios.

---

## 4. Experimental Design and Methodology

### 4.1 Task Formulation

The experiment is formulated as a **binary classification task** with the following characteristics:
- **Classes**: Javier Bardem and Jeffrey Dean Morgan
- **Classification rule**: For each test image, compute the cosine similarity of the detected face embedding to each reference embedding. Assign the label of the reference with the **highest similarity**.
- **No threshold**: Unlike typical face verification systems that use a similarity threshold to decide "match" vs "no match," this experiment always assigns one of the two labels. There is no "unknown" or "none" class.
- **No rejection**: Every image receives a classification, even if the model is uncertain.

This design choice was deliberate. By forcing a binary decision without an abstention option, we directly measure the model's ability to discriminate between the two identities. Any rejection mechanism or threshold would obscure the fundamental question of discriminative power.

### 4.2 Evaluation Protocol

The evaluation follows a standard classification evaluation protocol:
1. **Load test set**: 1,080 images with ground-truth labels
2. **Load reference images**: One reference image per person
3. **Initialise InsightFace**: Using the `buffalo_l` model with SCRFD detection and ArcFace recognition
4. **For each test image**:
   a. Detect all faces in the image
   b. Extract the embedding of each detected face
   c. Compute cosine similarity between each detected face and each reference embedding
   d. Select the identity with the highest similarity across all detected faces
   e. Record the prediction, similarity scores, and decision margin
5. **Compute metrics**: Accuracy, balanced accuracy, precision, recall, F1, Cohen's Kappa, MCC, ROC AUC, Average Precision, and confusion matrix

### 4.3 Software and Hardware

- **Framework**: InsightFace (Python) with ONNX Runtime backend
- **Model**: buffalo_l (SCRFD-10GF detector + ArcFace ResNet-100 recogniser)
- **Detection size**: 640×640 pixels
- **Execution provider**: CPUExecutionProvider (CPU-based inference)
- **Environment**: Docker container running Ubuntu 24.04.1 LTS with Python 3.12
- **Processing speed**: Approximately 3.49 images per second (total evaluation time ~5 minutes 9 seconds for 1,080 images)

---

## 5. Dataset Description and Analysis

### 5.1 Test Set Composition

The test set, named "Lookalike-Testset," consists of **1,080 images** distributed as follows:

| Identity | Count | Proportion |
|---|---|---|
| Javier Bardem | 465 | 43.1% |
| Jeffrey Dean Morgan | 615 | 56.9% |

Each image is labelled with exactly one identity. The images are sourced from various internet sources and represent a wide variety of conditions:
- **Pose variations**: Frontal, three-quarter view, profile
- **Lighting conditions**: Studio lighting, outdoor natural light, red carpet events, press conferences
- **Expression variations**: Neutral, smiling, speaking, emotive
- **Age variations**: Images spanning multiple decades of each actor's career
- **Image quality variations**: High-resolution professional photographs to lower-quality paparazzi shots
- **Accessories**: With and without glasses, facial hair variations
- **Context**: Headshots, upper-body, images with other people in the background

### 5.2 Class Imbalance

The dataset exhibits a mild class imbalance, with Jeffrey Dean Morgan having approximately 32% more images than Javier Bardem (615 vs 465). This imbalance ratio (1.32:1) is relatively modest and is accounted for in the evaluation through the use of balanced accuracy and macro-averaged metrics alongside standard accuracy.

### 5.3 Reference Images

Each identity is represented by a **single reference image**:
- **Javier Bardem**: `HavierBardem.jpg` (note: the filename contains a misspelling, with the original intended name `JavierBardem.jpg` not found, so the system automatically falls back to the alternate filename)
- **Jeffrey Dean Morgan**: `JeffreyDeanMorgan.jpg`

Using only a single reference image per person is a deliberately constrained setup. In practice, recognition systems typically use multiple reference images to build a more robust template (averaging embeddings or using a nearest-neighbor approach across references). By restricting to a single reference, we test the model under the most challenging conditions — any weakness in the reference image (unusual angle, expression, or lighting) will directly impact classification quality.

---

## 6. Model Architecture and Technical Details

### 6.1 Face Detection: SCRFD

The SCRFD (Sample and Computation Redistribution for Efficient Face Detection) model is used for face detection. Key characteristics include:
- **Architecture**: Anchor-free single-stage detector based on a modified feature pyramid network
- **Input resolution**: 640×640 pixels
- **Output**: Bounding boxes, confidence scores, and 5-point facial landmarks for each detected face
- **Computational budget**: 10 GFlops (the "large" variant)

SCRFD is designed for high detection accuracy even on small faces. In this experiment, the detected faces are generally well-centred and of sufficient size, as the test images predominantly feature the subject as the primary figure.

### 6.2 Face Recognition: ArcFace with ResNet-100

The recognition component uses an ArcFace model with a ResNet-100 backbone:
- **Backbone**: ResNet-100 (100-layer residual network)
- **Training data**: Glint360K dataset (~360K identities, ~17M images)
- **Loss function**: ArcFace (Additive Angular Margin Loss)
- **Embedding dimension**: 512
- **Normalisation**: L2-normalised embeddings (unit vectors on a 512-dimensional hypersphere)

The ArcFace loss function is defined as:

$$L = -\frac{1}{N} \sum_{i=1}^{N} \log \frac{e^{s \cdot \cos(\theta_{y_i} + m)}}{e^{s \cdot \cos(\theta_{y_i} + m)} + \sum_{j \neq y_i} e^{s \cdot \cos \theta_j}}$$

where $s$ is the feature scale (typically 64), $m$ is the angular margin (typically 0.5 radians), $\theta_{y_i}$ is the angle between the feature vector and the weight vector of the ground-truth class, and $N$ is the batch size. This margin penalty forces the network to learn more compact intra-class clusters and wider inter-class gaps compared to standard softmax-based training.

### 6.3 Cosine Similarity Metric

Since ArcFace produces L2-normalised embeddings $\mathbf{e}_1, \mathbf{e}_2 \in \mathbb{R}^{512}$ with $\|\mathbf{e}\|_2 = 1$, the cosine similarity reduces to the dot product:

$$\text{sim}(\mathbf{e}_1, \mathbf{e}_2) = \frac{\mathbf{e}_1 \cdot \mathbf{e}_2}{\|\mathbf{e}_1\| \|\mathbf{e}_2\|} = \mathbf{e}_1 \cdot \mathbf{e}_2$$

This similarity ranges from -1 (perfectly opposite) to +1 (identical). In practice, face embeddings from different identities typically yield similarities in the range of 0.0 to 0.4, while same-identity pairs typically yield similarities in the range of 0.4 to 0.8 or higher, depending on image quality and conditions.

---

## 7. Reference Embedding Construction

### 7.1 Process

The reference embedding construction process follows these steps for each of the two target identities:

1. Load the reference image from disk using OpenCV (`cv2.imread`)
2. Run the InsightFace detector on the image to locate all faces
3. Select the **largest detected face** (by bounding box area) — this heuristic assumes the primary subject is the most prominent face in the image
4. Extract the 512-dimensional L2-normalised embedding from the selected face
5. Store the embedding as the reference template for that identity

### 7.2 Reference Quality Considerations

The reference embedding is the cornerstone of the classification system. Its quality depends on:
- **Image quality**: Resolution, sharpness, and exposure of the reference photograph
- **Pose**: How close to frontal the reference face is
- **Expression**: Neutral expressions tend to produce more generalisable embeddings
- **Lighting**: Even, diffuse lighting produces more representative embeddings

In this experiment, both reference images successfully yielded face detections and embeddings. The InsightFace face alignment module (which uses the 5-point landmarks from SCRFD to align faces to a canonical template) ensures consistent preprocessing regardless of the input pose, which helps stabilise the embedding quality.

### 7.3 Single-Reference Limitation

Using a single reference embedding per person means the system has no redundancy. If the reference image is atypical (e.g., an unusual expression or angle), the classification boundary may be suboptimal. Despite this limitation, the system achieved 98.61% accuracy, suggesting that the reference images are sufficiently representative of each actor's facial identity in the embedding space.

---

## 8. Classification Pipeline and Decision Rule

### 8.1 Per-Image Classification

For each test image, the classification pipeline operates as follows:

1. **Image loading**: Read image from disk using OpenCV
2. **Face detection**: Run SCRFD detector to find all faces in the image
3. **Handle edge cases**:
   - If the image cannot be loaded: assign the first identity as a fallback (with zero similarity scores)
   - If no faces are detected: assign the first identity as a fallback (with zero similarity scores)
4. **Similarity computation**: For every detected face and every reference embedding, compute cosine similarity
5. **Identity-level aggregation**: For each identity, take the **maximum similarity** across all detected faces
6. **Decision**: Select the identity with the highest maximum similarity
7. **Margin computation**: Calculate the difference between the top-1 and top-2 similarity scores

### 8.2 Multi-Face Handling

The pipeline handles images containing multiple faces by computing similarities for all detected faces and selecting the best match across all of them. This is important because some test images may contain the subject alongside other people (e.g., at events). The maximum-similarity aggregation ensures that the most relevant face drives the classification decision.

### 8.3 Decision Margin

The decision margin is defined as:

$$\text{margin} = \text{sim}_{\text{top-1}} - \text{sim}_{\text{top-2}}$$

This represents the model's "confidence" in its decision. A large margin indicates that one identity is clearly preferred over the other, while a small margin indicates uncertainty. The margin is a critical diagnostic metric for understanding where the model struggles.

---

## 9. Results and Performance Analysis

### 9.1 Overall Classification Metrics

The model achieved the following overall performance on the 1,080-image test set:

| Metric | Value |
|---|---|
| **Accuracy** | 0.9861 (98.61%) |
| **Balanced Accuracy** | 0.9873 (98.73%) |
| **Precision (macro)** | 0.9847 |
| **Recall (macro)** | 0.9873 |
| **F1 Score (macro)** | 0.9859 |
| **Precision (weighted)** | 0.9864 |
| **Recall (weighted)** | 0.9861 |
| **F1 Score (weighted)** | 0.9861 |
| **Cohen's Kappa** | 0.9718 |
| **Matthews Correlation Coefficient** | 0.9720 |
| **ROC AUC** | 0.9997 |
| **Average Precision** | 0.9996 |

### 9.2 Interpretation of Key Metrics

**Accuracy (98.61%)**: Out of 1,080 test images, 1,065 were classified correctly. Only 15 images were misclassified. This is an exceptional result given the extreme visual similarity between the two subjects.

**Balanced Accuracy (98.73%)**: This metric accounts for class imbalance by averaging the recall of each class. The fact that it is slightly higher than raw accuracy indicates that the model performs marginally better on the minority class (Javier Bardem), confirming that the class imbalance does not adversely affect performance.

**F1 Score (macro: 0.9859)**: The harmonic mean of precision and recall, averaged across classes, confirms that the model achieves simultaneously high precision and high recall for both identities. There is no significant trade-off between the two.

**Cohen's Kappa (0.9718)**: This measures agreement beyond what would be expected by chance. A value above 0.80 is generally considered "almost perfect agreement." At 0.9718, the model demonstrates near-perfect reliability that far exceeds random chance.

**Matthews Correlation Coefficient (0.9720)**: MCC is considered one of the most balanced measures of binary classification quality, as it takes into account all four cells of the confusion matrix. A value of +1 indicates perfect prediction, 0 indicates no better than random, and -1 indicates total disagreement. At 0.9720, the model is performing at an exceptionally high level.

**ROC AUC (0.9997)**: The area under the Receiver Operating Characteristic curve measures the model's ability to discriminate between the two classes across all possible thresholds. A value of 0.9997 (out of a maximum of 1.0000) indicates near-perfect discriminative ability. This means that a randomly chosen Javier Bardem image will have a higher "Bardem score" than a randomly chosen Jeffrey Dean Morgan image 99.97% of the time.

**Average Precision (0.9996)**: Similar to ROC AUC but more sensitive to performance in high-precision regimes, the Average Precision of 0.9996 confirms that the model maintains near-perfect precision even at very high recall levels.

### 9.3 Processing Statistics

- **Total images processed**: 1,080
- **Images with no face detected**: 2 (0.19%)
- **Processing errors**: 0
- **Processing speed**: 3.49 images/second
- **Total evaluation time**: 5 minutes 9 seconds

The 2 images where no face was detected represent a minor limitation of the SCRFD detector. These images likely contain heavily occluded faces, extreme poses, or poor image quality that prevent reliable face detection. They were assigned to the fallback class, contributing to the error count.

### 9.4 Performance Context

To contextualise these results, it is helpful to compare them against known benchmarks. On the standard LFW benchmark, ArcFace achieves 99.83% verification accuracy. However, LFW primarily tests the model's ability to distinguish between visually dissimilar individuals, making it a relatively easy benchmark by modern standards. The 98.61% accuracy achieved here on lookalikes is remarkably close to the LFW result, despite the dramatically higher difficulty of the task. This suggests that the InsightFace embedding space retains strong discriminative power even at the extreme end of facial similarity.

Furthermore, the 0.9997 ROC AUC achieved in this experiment exceeds the AUC reported by many face recognition systems on standard verification benchmarks, highlighting the exceptional quality of the ArcFace embeddings for this specific identity pair. The combination of high accuracy, high AUC, and well-calibrated confidence scores paints a picture of a mature, reliable recognition system that degrades gracefully as task difficulty increases.

---

## 10. Confusion Matrix Analysis

The confusion matrix provides the most detailed view of classification performance:

### 10.1 Raw Counts

|  | Predicted: Javier Bardem | Predicted: Jeffrey Dean Morgan |
|---|---|---|
| **True: Javier Bardem** | 463 (TP) | 2 (FN) |
| **True: Jeffrey Dean Morgan** | 13 (FP) | 602 (TN) |

### 10.2 Row-Normalised (Recall per Class)

|  | Predicted: Javier Bardem | Predicted: Jeffrey Dean Morgan |
|---|---|---|
| **True: Javier Bardem** | 99.57% | 0.43% |
| **True: Jeffrey Dean Morgan** | 2.11% | 97.89% |

### 10.3 Interpretation

The confusion matrix reveals an important asymmetry:
- **Javier Bardem recall: 99.57%** — Out of 465 Bardem images, only 2 were misclassified as Morgan. The model is extremely reliable at recognising Bardem.
- **Jeffrey Dean Morgan recall: 97.89%** — Out of 615 Morgan images, 13 were misclassified as Bardem. While still excellent, this is somewhat lower than Bardem's recall.

This asymmetry suggests that the Jeffrey Dean Morgan test images exhibit greater variability in the embedding space, causing a small number of them to fall closer to Bardem's reference embedding than to Morgan's. This could be due to:
- Greater pose/expression/lighting diversity in Morgan's test images
- The Morgan reference embedding being slightly less centrally positioned in Morgan's identity cluster
- A possible systematic bias in the model's learned representation that makes certain Morgan-like features overlap more with Bardem's embedding neighbourhood

The fact that this asymmetry is small (only 11 additional misclassifications) and that overall performance remains above 98% confirms that the model handles this challenging pair remarkably well.

---

## 11. Similarity Score Distribution Analysis

### 11.1 Descriptive Statistics

The cosine similarity scores across all 1,080 test images show the following distribution:

| Statistic | Score (Bardem) | Score (Morgan) | Margin | Max Score |
|---|---|---|---|---|
| **Count** | 1,080 | 1,080 | 1,080 | 1,080 |
| **Mean** | 0.3105 | 0.3793 | 0.4620 | 0.5759 |
| **Std** | 0.2192 | 0.2795 | 0.1318 | 0.1374 |
| **Min** | -0.1020 | -0.0372 | 0.0000 | 0.0000 |
| **25th percentile** | 0.1245 | 0.0920 | 0.3930 | 0.5201 |
| **Median** | 0.1854 | 0.4383 | 0.4779 | 0.5957 |
| **75th percentile** | 0.5528 | 0.6488 | 0.5397 | 0.6610 |
| **Max** | 0.7066 | 0.8590 | 0.7905 | 0.8590 |

### 11.2 Score Distribution by True Label

The "Similarity to Own Reference" distribution reveals how well each person's images match their own reference:
- **Javier Bardem images** show a roughly normal distribution of similarity to the Bardem reference, centred around 0.5–0.6, with a moderate spread
- **Jeffrey Dean Morgan images** show a similar pattern with similarity to the Morgan reference, also centred around 0.5–0.7, but with a slightly wider spread

Both distributions show healthy separation from zero, indicating that the model reliably detects each person's identity signature. The overlap between the two distributions is minimal, which explains the high classification accuracy.

### 11.3 Score Space Scatter Plot

The scatter plot of Bardem similarity versus Morgan similarity reveals the two-dimensional score landscape:
- **Javier Bardem images** (blue dots) cluster in the lower-right quadrant (high Bardem similarity, low Morgan similarity)
- **Jeffrey Dean Morgan images** (red dots) cluster in the upper-left quadrant (low Bardem similarity, high Morgan similarity)
- The **decision boundary** (the diagonal line where score_Bardem = score_Morgan) cleanly separates the two clusters, with only a handful of points near or on the wrong side

This visualisation powerfully demonstrates that the InsightFace embedding space creates a linearly separable representation of these two lookalike identities.

### 11.4 Margin Distribution

The decision margin (difference between top-1 and top-2 similarity) shows a stark contrast between correctly and incorrectly classified images:
- **Correct predictions (1,065 images)**: Mean margin = 0.4678, indicating high confidence
- **Incorrect predictions (15 images)**: Mean margin = 0.0541, indicating the model was barely leaning one way

This dramatic difference (approximately 8.6× larger margin for correct predictions) confirms that the model's errors are concentrated in the "uncertainty zone" — images where the two similarity scores are nearly equal. The model is well-calibrated in the sense that high-confidence predictions are overwhelmingly correct.

---

## 12. ROC Curve and AUC Analysis

### 12.1 ROC Curve Construction

The ROC curve was constructed by treating the problem as binary classification with Javier Bardem as the positive class. The discriminant function is the **score difference**: $\Delta s = s_{\text{Bardem}} - s_{\text{Morgan}}$. Higher values of $\Delta s$ indicate greater likelihood of the image being Javier Bardem.

### 12.2 ROC AUC Result

The ROC AUC is **0.9997**, which is exceptionally close to the theoretical maximum of 1.0000. This means that if we randomly select one Bardem image and one Morgan image, there is a 99.97% probability that the Bardem image will have a higher score difference than the Morgan image.

### 12.3 Operating Point Analysis

The operating point of the current system (using threshold = 0, i.e., classify as Bardem if $s_{\text{Bardem}} > s_{\text{Morgan}}$) is marked on the ROC curve at:
- **True Positive Rate (Sensitivity)**: 0.996 — 99.6% of Bardem images are correctly identified
- **False Positive Rate (1 - Specificity)**: 0.021 — 2.1% of Morgan images are incorrectly identified as Bardem

This operating point is already in the extreme upper-left corner of the ROC space, very close to the ideal point (0, 1). The ROC curve hugs the upper-left corner almost perfectly, reflecting the near-complete separability of the two classes.

### 12.4 Comparison to Random Classifier

A random classifier would produce a diagonal ROC curve with AUC = 0.5. The observed AUC of 0.9997 represents a 99.94% improvement over random chance, confirming that the model captures genuine identity-discriminative information rather than relying on spurious correlations.

---

## 13. Precision-Recall Curve Analysis

### 13.1 Precision-Recall Curve

The Precision-Recall (PR) curve evaluates the trade-off between precision (positive predictive value) and recall (sensitivity) for the Javier Bardem class as the decision threshold varies.

### 13.2 Average Precision Result

The Average Precision (AP) is **0.9996**, indicating that the model maintains near-perfect precision across virtually all recall levels. The PR curve is nearly a perfect rectangle — it stays at precision ≈ 1.0 until recall reaches approximately 0.99, then drops sharply. This means the model can identify almost all Bardem images before making any false positive predictions.

### 13.3 Baseline Comparison

The baseline precision (random classifier) equals the prevalence of the positive class: 0.431 (since 465 out of 1,080 images are Bardem). The model's AP of 0.9996 far exceeds this baseline, confirming substantial information gain from the face recognition system.

### 13.4 Practical Implications

The near-perfect PR curve has important practical implications. In applications where false identifications must be minimised (e.g., law enforcement), the model can operate at a high threshold to achieve virtually 100% precision while still maintaining high recall. Conversely, in applications where missing a person is unacceptable (e.g., security screening), the model can operate at a lower threshold to achieve near-100% recall with minimal precision loss.

---

## 14. Decision Margin Analysis

### 14.1 Margin Statistics

| Subset | Mean Margin | Median Margin |
|---|---|---|
| All images | 0.4620 | 0.4779 |
| Correct predictions | 0.4678 | — |
| Incorrect predictions | 0.0541 | — |

### 14.2 Cumulative Distribution Function (CDF)

The CDF of decision margins for correct vs incorrect predictions reveals dramatically different profiles:
- **Correct predictions**: The CDF rises gradually, with most margins falling between 0.3 and 0.7. This indicates broad, confident separation.
- **Incorrect predictions**: The CDF rises steeply near zero, with 80% of incorrect predictions having margins below 0.1 and 100% below 0.25. This confirms that errors occur exclusively in the narrow band of ambiguity.

### 14.3 Accuracy vs Coverage Trade-off

By introducing a rejection mechanism — refusing to classify images with margins below a threshold — we can trade coverage (fraction of images classified) for accuracy:
- At **margin threshold 0.0**: 100% coverage, 98.61% accuracy (baseline)
- At **margin threshold 0.1**: ~98% coverage, ~99.3% accuracy
- At **margin threshold 0.2**: ~96% coverage, ~99.8% accuracy
- At **margin threshold 0.3**: ~92% coverage, approaching 100% accuracy

This analysis reveals that the model's errors are highly concentrated in low-margin predictions. A simple confidence-based rejection policy could virtually eliminate errors while retaining the vast majority of the dataset. In production systems, such margin-based rejection would allow uncertain cases to be flagged for human review, significantly improving overall system reliability.

### 14.4 Practical Margin Threshold Recommendations

For different application scenarios:
- **High-throughput screening** (accept all predictions): Use threshold = 0, achieving 98.61% accuracy
- **Balanced deployment** (moderate rejection): Use threshold = 0.1, achieving ~99.3% accuracy on ~98% of images
- **High-security applications** (minimise errors): Use threshold = 0.25, achieving ~99.9% accuracy on ~95% of images

---

## 15. Score Separation and Calibration

### 15.1 Score Difference Distributions

The score difference ($s_{\text{Bardem}} - s_{\text{Morgan}}$) provides a one-dimensional projection that captures the essential discriminative information:

- **True Javier Bardem images**: Mean score difference = +0.4549 ± 0.1003 (positive, as expected)
- **True Jeffrey Dean Morgan images**: Mean score difference = -0.4648 ± 0.1589 (negative, as expected)

### 15.2 Distribution Overlap

The overlap between the two score difference distributions is extremely small:
- **Javier Bardem images misclassified** (score_B < score_M): Only **0.4%** (2 out of 465)
- **Jeffrey Dean Morgan images misclassified** (score_M < score_B): Only **2.0%** (13 out of 615, noting that this includes the 2 no-face cases)

The score difference histogram shows two well-separated bell-shaped distributions with the decision boundary at zero falling cleanly between them. The Bardem distribution is slightly tighter (std = 0.1003 vs 0.1589 for Morgan), which is consistent with the higher recall for Bardem images.

### 15.3 Calibration Quality

The separation between the two distributions can be quantified by the **d-prime** (d') metric borrowed from signal detection theory:

$$d' = \frac{|\mu_1 - \mu_2|}{\sqrt{\frac{\sigma_1^2 + \sigma_2^2}{2}}}$$

Using the observed statistics:
$$d' = \frac{|0.4549 - (-0.4648)|}{\sqrt{\frac{0.1003^2 + 0.1589^2}{2}}} = \frac{0.9197}{\sqrt{\frac{0.01006 + 0.02525}{2}}} = \frac{0.9197}{\sqrt{0.01765}} = \frac{0.9197}{0.1329} \approx 6.92$$

A d' of 6.92 is extraordinarily high. In psychophysics, d' values above 4.0 are considered near-perfect discrimination. This confirms that the InsightFace embedding space provides massive separation between these two lookalike identities, well beyond what would be needed for reliable classification.

---

## 16. Threshold Sensitivity Analysis

### 16.1 Threshold Sweep

The default classification threshold is 0 (classify as Bardem if $\Delta s \geq 0$, Morgan otherwise). A sweep of thresholds across the full range of observed score differences reveals how sensitive accuracy is to this choice.

### 16.2 Optimal Threshold

The optimal decision threshold was found at **0.0485**, yielding an accuracy of **99.54%** (an improvement of +0.93 percentage points over the default threshold of 0).

This small positive shift in the optimal threshold suggests a slight asymmetry in the score distributions: the Morgan reference may be marginally more "attractive" in the embedding space, causing some borderline Morgan images to produce slightly positive score differences (i.e., marginally closer to Bardem). By shifting the threshold to +0.0485, these borderline cases are correctly reassigned to Morgan.

### 16.3 Threshold Stability

The accuracy vs threshold curve shows a broad plateau of high accuracy (above 98%) spanning approximately from -0.2 to +0.3. This indicates that the system is **highly robust** to threshold selection — even significant perturbations from the optimal threshold barely affect performance. This robustness is a direct consequence of the wide score separation documented in the previous section.

### 16.4 Practical Implications

The near-zero optimal threshold shift (+0.0485) means that the default policy of "assign to the nearest reference" is already close to optimal. The small improvement from threshold tuning (+0.93 pp) does not justify the added complexity and potential overfitting risk of threshold optimisation in most practical applications. However, in safety-critical deployments, even small improvements matter, and the optimal threshold could be calibrated using a held-out validation set.

It is also noteworthy that the accuracy plateau extends into negative threshold territory (down to approximately -0.2) without catastrophic degradation. This resilience to threshold perturbation means that the system would remain functional even if the operating conditions shifted (e.g., due to camera changes, lighting changes, or reference image updates) in ways that subtly altered the score distribution. This operational robustness is a highly desirable property for deployed systems that must function reliably over extended periods without constant recalibration.

---

## 17. Per-Class Performance Breakdown

### 17.1 Detailed Per-Class Metrics

| Metric | Javier Bardem | Jeffrey Dean Morgan |
|---|---|---|
| **True Positives (TP)** | 463 | 602 |
| **False Positives (FP)** | 13 | 2 |
| **False Negatives (FN)** | 2 | 13 |
| **True Negatives (TN)** | 602 | 463 |
| **Precision** | 0.9727 | 0.9967 |
| **Recall** | 0.9957 | 0.9789 |
| **F1 Score** | 0.9841 | 0.9877 |
| **Support** | 465 | 615 |

### 17.2 Class-Level Interpretation

**Javier Bardem**:
- **Precision (0.9727)**: Of all images predicted as Bardem, 97.27% were actually Bardem. The 13 false positives are Morgan images misclassified as Bardem.
- **Recall (0.9957)**: Of all actual Bardem images, 99.57% were correctly identified. Only 2 Bardem images were missed.
- **F1 (0.9841)**: Excellent harmonic mean of precision and recall.

**Jeffrey Dean Morgan**:
- **Precision (0.9967)**: Of all images predicted as Morgan, 99.67% were actually Morgan. Only 2 false positives (Bardem images misclassified as Morgan).
- **Recall (0.9789)**: Of all actual Morgan images, 97.89% were correctly identified. 13 Morgan images were missed.
- **F1 (0.9877)**: Slightly higher F1 than Bardem due to the exceptional precision.

### 17.3 Asymmetry Analysis

The most notable asymmetry is in the error distribution:
- Bardem → Morgan misclassifications: 2
- Morgan → Bardem misclassifications: 13

This 6.5:1 ratio suggests that Morgan's embedding distribution has a heavier tail extending toward Bardem's reference point. Possible explanations include:
1. **Greater variability in Morgan's test images**: More diverse poses, expressions, or image quality
2. **Reference image quality**: Morgan's reference may not perfectly represent his cluster centre
3. **Intrinsic facial feature distribution**: Certain Morgan expressions or angles may produce embeddings that genuinely resemble Bardem's canonical appearance

Despite this asymmetry, both classes achieve F1 scores above 0.98, demonstrating that the model is reliable for both identities.

---

## 18. Error Analysis

### 18.1 Error Summary

Out of 1,080 test images, 15 were misclassified:
- **2 Javier Bardem** images misclassified as Jeffrey Dean Morgan
- **13 Jeffrey Dean Morgan** images misclassified as Javier Bardem

### 18.2 Error Margin Analysis

**Bardem → Morgan errors (2 images)**:
- Mean margin: 0.1891
- Median margin: 0.1891
- Max margin: 0.2434
- Min margin: 0.1349

These errors have relatively moderate margins (0.13–0.24), suggesting that the model was somewhat confident in its (incorrect) decision. These may represent genuinely ambiguous images where Bardem's appearance in that particular photograph closely resembles Morgan.

**Morgan → Bardem errors (13 images)**:
- Mean margin: 0.0333
- Median margin: 0.0347
- Max margin: 0.0871
- Min margin: 0.0000

These errors have very low margins (most below 0.05), indicating the model was barely leaning toward Bardem over Morgan. These are "coin flip" decisions where the score difference is negligible. The minimum margin of 0.0000 means at least one image had essentially identical similarity to both references.

### 18.3 Error Characteristics

The errors fall into two distinct categories:

**Category 1 — Near-boundary errors (13 Morgan → Bardem)**: These are images where the model's two similarity scores are nearly tied. The score for Bardem slightly exceeds the score for Morgan, causing a misclassification. These errors are the "natural" errors of a system operating at the boundary of its discriminative ability. They would be easily caught by a margin-based rejection policy.

**Category 2 — Moderate-confidence errors (2 Bardem → Morgan)**: These are more concerning because the model is somewhat confident in the wrong direction. They may represent images with unusual characteristics (extreme pose, poor quality, or occlusion) that shift the embedding away from Bardem's typical cluster position toward Morgan's reference.

### 18.4 No-Face Detection Cases

Two images resulted in no face detection. These were assigned to the fallback class (Javier Bardem) and may or may not have been correct. Images with no detectable face typically suffer from:
- Extreme head pose (facing away from camera)
- Heavy occlusion (hand over face, sunglasses, etc.)
- Very small face size in a wide-angle shot
- Extremely poor image quality or heavy compression artefacts

---

## 19. Comparison with Baseline and Random Classifiers

### 19.1 Random Classifier Baseline

A random classifier that assigns labels based on class prevalence would achieve:
- **Accuracy**: $0.431^2 + 0.569^2 = 0.186 + 0.324 = 0.510$ (50.96%)
- **Cohen's Kappa**: 0.0
- **MCC**: 0.0
- **ROC AUC**: 0.50

### 19.2 Majority Classifier Baseline

A majority classifier that always predicts "Jeffrey Dean Morgan" (the majority class) would achieve:
- **Accuracy**: 56.94% (615/1080)
- **Balanced Accuracy**: 50.00%
- **Cohen's Kappa**: 0.0
- **MCC**: 0.0

### 19.3 Performance Gain Over Baselines

| Metric | Random | Majority | InsightFace | Gain over best baseline |
|---|---|---|---|---|
| Accuracy | 51.0% | 56.9% | 98.6% | +41.7 pp |
| Balanced Accuracy | 50.0% | 50.0% | 98.7% | +48.7 pp |
| Cohen's Kappa | 0.0 | 0.0 | 0.972 | +0.972 |
| MCC | 0.0 | 0.0 | 0.972 | +0.972 |
| ROC AUC | 0.50 | — | 0.9997 | +0.4997 |

The InsightFace model achieves massive improvements over all baseline methods, confirming that the deep embeddings capture genuine identity information rather than exploiting dataset artefacts or class imbalance.

---

## 20. Discussion

### 20.1 Key Findings

This experiment demonstrates several important findings:

**Finding 1: Deep face embeddings can reliably distinguish lookalikes.** Despite Javier Bardem and Jeffrey Dean Morgan being widely considered near-identical in appearance, the InsightFace model achieves 98.61% accuracy using only a single reference image per person. This strongly suggests that deep face recognition models learn identity representations that capture subtle differences invisible or easily overlooked by the human visual system.

**Finding 2: The embedding space provides massive class separation.** The d-prime of approximately 6.92 between the two score difference distributions indicates that the ArcFace embedding space creates well-separated identity clusters even for lookalikes. This separation is far larger than what would be minimally necessary for reliable classification.

**Finding 3: Errors are concentrated in a narrow uncertainty band.** The 15 misclassified images all have low decision margins (mean 0.054 for Morgan errors, 0.189 for Bardem errors), while correctly classified images have a mean margin of 0.468. This 8.6× difference demonstrates that the model's confidence is well-calibrated — it "knows when it doesn't know."

**Finding 4: The system is robust to threshold selection.** The accuracy vs threshold curve shows a broad plateau of near-optimal performance, meaning the system does not require careful threshold tuning to perform well. The default threshold of 0 (nearest-reference assignment) achieves 98.61% accuracy, while the optimal threshold of 0.0485 achieves 99.54% — a modest 0.93 pp improvement.

**Finding 5: Asymmetric error patterns reveal identity-specific challenges.** The 13:2 ratio of Morgan→Bardem vs Bardem→Morgan errors suggests that Jeffrey Dean Morgan's facial variability in the test set is broader, leading to some images falling into the ambiguous zone. This type of analysis can inform reference image selection strategies in deployed systems.

### 20.2 What Makes InsightFace Succeed on Lookalikes?

Several properties of the ArcFace training paradigm contribute to the model's success:

1. **Angular margin loss**: The additive angular margin in ArcFace explicitly penalises intra-class spread and encourages inter-class separation. This means the model is trained to find discriminative features even between similar identities.

2. **Large-scale training data**: The Glint360K dataset contains approximately 360,000 identities. With such diversity, the model has likely encountered many pairs of similar-looking individuals during training, forcing it to develop representations sensitive to fine-grained identity differences.

3. **High embedding dimensionality**: The 512-dimensional embedding space provides ample capacity to encode subtle identity-specific features. In a 512-dimensional space, even small angular differences between embeddings correspond to meaningful identity distinctions.

4. **Robust preprocessing**: The face alignment pipeline normalises pose, scale, and position before embedding extraction, ensuring that the neural network sees a consistently preprocessed face. This removes nuisance variability that could confuse the model.

### 20.3 Implications for Real-World Deployment

The results have several implications for deploying face recognition systems in real-world scenarios:

1. **Lookalike discrimination is feasible**: Systems can distinguish between look-alikes with high accuracy, even using single reference images. This is reassuring for identity verification and access control applications.

2. **Confidence-based rejection improves reliability**: By refusing to classify low-margin predictions, systems can achieve near-perfect accuracy on the remaining images. This is the recommended deployment strategy for applications where errors have serious consequences.

3. **Reference image quality matters**: The asymmetric error pattern suggests that reference image selection affects class-specific performance. Investment in high-quality, representative reference images can further improve accuracy.

4. **Single-reference systems are surprisingly effective**: Even with only one reference embedding per person, accuracy exceeds 98%. Multi-reference systems (using embeddings from multiple images per person) would likely push accuracy even higher, potentially eliminating all or nearly all errors.

### 20.4 The Nature of Facial Similarity

This experiment raises interesting questions about the nature of facial similarity. Javier Bardem and Jeffrey Dean Morgan look extremely similar to human observers, yet the model distinguishes them with 98.61% accuracy. This suggests that human perception of facial similarity relies heavily on coarse features (face shape, hair, complexion, overall "gestalt") while deep networks additionally capture fine-grained microstructural features that humans may not consciously perceive.

Research in cognitive psychology has shown that humans process faces holistically — as integrated wholes rather than collections of independent features. This holistic processing is efficient for everyday recognition but may miss subtle differences that are captured by the high-dimensional representations learned by deep networks. The success of InsightFace on this task is a compelling demonstration that machine perception can exceed human perception in specific domains.

The 512-dimensional embedding space likely encodes features at multiple levels of abstraction. While lower-level features may capture texture, skin patterns, and local geometry, higher-level features may encode identity-specific combinations of facial proportions, eye spacing, nose shape, and jawline curvature that collectively form a unique "facial fingerprint" in embedding space. Even though Bardem and Morgan share many coarse-level features, their fine-grained identity fingerprints are distinct enough to create well-separated clusters.

### 20.5 Implications for Forensic and Security Applications

The success of this experiment has direct implications for forensic science and security applications. Law enforcement agencies frequently deal with cases where suspects resemble other individuals, and eyewitness misidentification is one of the leading causes of wrongful convictions. The ability of a deep face recognition system to distinguish between lookalikes with over 98% accuracy — and with well-calibrated confidence scores — suggests that such systems could serve as valuable forensic tools, provided appropriate protocols are established.

In border control and airport security, the system's ability to operate with a single reference image is particularly relevant, as passport photographs typically provide only one canonical view of an individual. The fact that InsightFace achieves 98.61% accuracy in this single-reference scenario on lookalikes (a worst-case scenario) suggests that performance on typical, non-lookalike individuals would be substantially higher, likely exceeding 99.9%.

However, the deployment of such systems must be accompanied by proper safeguards. The 15 errors in this experiment — while statistically minor — could have serious consequences if they occurred in a law enforcement context. The margin-based rejection mechanism identified in Section 14 provides a natural safeguard: by flagging low-confidence predictions for human review, the system can operate at near-zero error rates on the predictions it does make, while ensuring that ambiguous cases receive appropriate human oversight.

### 20.6 The Role of Training Data Diversity

The exceptional performance of InsightFace on this lookalike task is intimately connected to the diversity and scale of its training data. The Glint360K dataset, with approximately 360,000 distinct identities and 17 million images, provides the model with exposure to an enormous range of facial variations. During training, the model must learn to distinguish between all 360,000 identities simultaneously, many of which inevitably share visual similarities. This forces the ArcFace loss function to push the model toward representations that capture the most identity-discriminative features, rather than relying on superficial cues.

It is worth noting that both Javier Bardem and Jeffrey Dean Morgan are internationally famous actors who have appeared in numerous films, television shows, and public events. It is plausible (though unconfirmed) that images of both actors appear in the training data, either in the Glint360K dataset or in earlier pretraining stages. If so, the model may have explicitly learned to distinguish between them during training. Even if their specific images are not in the training data, the model's experience with hundreds of thousands of other identity pairs — including many that share visual similarities — has equipped it with the general capability to make fine-grained distinctions.

### 20.7 Statistical Significance of Results

Given the large test set (1,080 images), the observed accuracy of 98.61% has narrow confidence intervals. Using the normal approximation to the binomial distribution, the 95% confidence interval for the true accuracy is:

$$\hat{p} \pm z_{0.975} \sqrt{\frac{\hat{p}(1 - \hat{p})}{n}} = 0.9861 \pm 1.96 \sqrt{\frac{0.9861 \times 0.0139}{1080}} = 0.9861 \pm 0.0070$$

This gives a 95% confidence interval of [0.9791, 0.9931], meaning we can be 95% confident that the model's true accuracy on this type of lookalike task lies between 97.91% and 99.31%. The lower bound of 97.91% is itself an excellent result, and the narrow width of the interval (1.4 percentage points) reflects the large sample size.

Similarly, the ROC AUC of 0.9997 has extremely narrow confidence bounds due to the near-perfect separation. A DeLong test or bootstrap approach could be used to formally compute confidence intervals for the AUC, but the point estimate of 0.9997 leaves essentially no room for concern about the model's discriminative ability.

### 20.8 Comparison with Human Performance

While this experiment does not include a formal human evaluation, informal evidence from popular culture suggests that human accuracy on this specific lookalike pair would be substantially lower than 98.61%. Internet surveys, social media polls, and entertainment articles consistently report that a large fraction of respondents cannot reliably distinguish between Bardem and Morgan in photographs, particularly when contextual cues (body type, clothing, setting) are removed.

A formal human study would be a valuable follow-up to this work. Such a study could present the same 1,080 test images to human participants (with and without reference images) and measure their classification accuracy, response time, and confidence. Comparing human and machine performance on the same test set would provide a rigorous assessment of whether InsightFace truly exceeds human-level discrimination ability for this lookalike pair.

Anecdotal evidence suggests that humans rely heavily on contextual and extra-facial cues when distinguishing lookalikes: knowing which movies each actor has appeared in, recognising their typical attire or companions, or using voice and mannerism cues. When restricted to static facial images alone — as in this experiment — human performance would likely be significantly impaired, while the model's performance remains robust because it operates solely on facial geometry and texture features.

---

## 21. Limitations

### 21.1 Single Reference Image

The most significant limitation is the use of only one reference image per person. While the results are impressive, they are potentially fragile: if the reference image is atypical, performance could degrade significantly. A more robust system would use multiple reference images to build a centroid or exemplar-based template.

### 21.2 Binary Classification Only

The experiment is limited to two identities. In a real-world scenario, the system would need to distinguish between many more individuals, potentially including other lookalikes. The binary setting may overestimate performance compared to a multi-class scenario.

### 21.3 No Threshold / No Rejection

By forcing a binary decision on every image, the system cannot express uncertainty. In practice, a "none of the above" option would be important for images that don't contain either person. The no-threshold design is appropriate for evaluating discriminative ability but would need modification for deployment.

### 21.4 Dataset Characteristics

The test dataset consists of internet-sourced celebrity images, which tend to be reasonably well-lit and well-composed. Performance might differ on surveillance-quality images, images taken at extreme distances, or images with significant motion blur or compression artefacts.

### 21.5 No Temporal Analysis

The experiment does not account for temporal changes in appearance. Both actors have changed their look over the years (facial hair, weight, ageing), and the experiment does not analyse whether errors correlate with specific time periods or appearance variations.

### 21.6 CPU-Only Inference

The experiment was run on CPU, achieving 3.49 images per second. While sufficient for this evaluation, real-world deployment at scale would require GPU acceleration to achieve the throughput needed for applications like real-time surveillance.

### 21.7 Misspelled Reference File

The reference image for Javier Bardem was stored with a misspelled filename (`HavierBardem.jpg` instead of `JavierBardem.jpg`). While the code handled this gracefully with a fallback mechanism, this kind of data management issue could cause silent failures in less carefully designed systems.

### 21.8 Potential Data Leakage

It is possible that images in the test set (or very similar images) appeared in the Glint360K training dataset used to train the ArcFace model. Since both actors are internationally famous, their images are widely available on the internet, and face recognition training datasets are typically constructed through web scraping. If significant data leakage exists, the reported accuracy may be an optimistic estimate of the model's generalisation ability on truly unseen images. However, because the model operates through embedding comparison rather than direct memorisation, the impact of data leakage is likely modest — the model would perform similarly on new images of the same individuals.

### 21.9 Environmental and Demographic Factors

The test dataset primarily consists of images from Western media contexts (red carpet events, press conferences, film premieres, professional photoshoots). Performance might differ on images taken in different cultural contexts, with different photographic traditions, or under conditions more typical of surveillance or consumer photography. Additionally, both subjects are middle-aged Caucasian males, and the results of this experiment should not be generalised without caution to lookalike pairs from other demographic groups, as face recognition systems are known to exhibit varying performance across demographic categories.

---

## 22. Conclusions and Future Work

### 22.1 Conclusions

This experiment has comprehensively evaluated the InsightFace face recognition model on the challenging task of distinguishing between two celebrity lookalikes, Javier Bardem and Jeffrey Dean Morgan. The key conclusions are:

1. **InsightFace achieves exceptional accuracy (98.61%) on the lookalike classification task**, correctly classifying 1,065 out of 1,080 test images using only a single reference image per person.

2. **The model demonstrates near-perfect discriminative ability**, with a ROC AUC of 0.9997 and Average Precision of 0.9996, indicating that the embedding space provides massive separation between the two lookalike identities.

3. **All standard classification metrics are above 0.97**, including balanced accuracy (0.9873), macro F1 (0.9859), Cohen's Kappa (0.9718), and MCC (0.9720), confirming robust and reliable performance that far exceeds any baseline.

4. **Errors are concentrated in a narrow uncertainty zone**, with misclassified images having a mean decision margin of only 0.054, compared to 0.468 for correctly classified images. This well-calibrated confidence enables effective margin-based rejection policies.

5. **The error pattern is asymmetric**, with 13 Morgan→Bardem errors versus 2 Bardem→Morgan errors, suggesting class-specific variability in embedding distributions.

6. **Threshold sensitivity is minimal**, with the default threshold achieving near-optimal performance and only a 0.93 percentage point improvement available through optimal threshold tuning.

7. **The ArcFace deep face embedding captures identity-discriminative features that transcend superficial visual similarity**, enabling reliable distinction between individuals that the human eye frequently confuses.

### 22.2 Future Work

Several directions could extend and strengthen this work:

1. **Multi-reference evaluation**: Test performance with 2, 5, 10, and 20 reference images per person to quantify the benefit of reference diversity. Centroid-based templates (averaging multiple embeddings) and nearest-neighbour approaches (comparing against the closest reference) should both be evaluated. The expected outcome is that multi-reference systems will reduce the error rate, particularly for the Morgan→Bardem direction where the single Morgan reference appears suboptimal.

2. **Cross-model comparison**: Evaluate other face recognition models (e.g., FaceNet, VGGFace2, AdaFace, MagFace) on the same lookalike testset to compare their discriminative capabilities. This would reveal whether the exceptional performance observed here is specific to ArcFace or is a general property of modern deep face recognition models. Models trained with different loss functions (contrastive loss, triplet loss, curriculum loss) and different backbone architectures (Vision Transformers, EfficientNet, MobileNet) should be included.

3. **Additional lookalike pairs**: Extend the evaluation to other known lookalike pairs (e.g., Keira Knightley and Natalie Portman, Zooey Deschanel and Katy Perry, Will Ferrell and Chad Smith, Matt Damon and Mark Wahlberg) to assess generalisability. A benchmark dataset of 10–20 lookalike pairs, curated with balanced class distributions and consistent image quality, would be a valuable contribution to the face recognition community.

4. **Ablation studies**: Investigate the effect of detection model, input resolution, embedding dimensionality, and alignment method on lookalike classification accuracy. For example, testing at 320×320 vs 640×640 detection resolution, or using 128-dimensional vs 512-dimensional embeddings, would reveal which components are most critical for fine-grained discrimination.

5. **Multi-class setting**: Extend to a gallery of 50+ individuals (including multiple lookalike pairs) to evaluate performance in a realistic open-set identification scenario. This is more representative of real-world deployment where the system must handle many identities simultaneously, and the confusion space is much larger.

6. **Hard negative mining**: Analyse the embedding space to identify which facial features most differentiate the two lookalikes, potentially using gradient-based attribution methods (e.g., GradCAM, Integrated Gradients, or SHAP) on the recognition model. Understanding which regions of the face the model attends to when discriminating between Bardem and Morgan would provide insights into the learned representation and could inform the design of more interpretable face recognition systems.

7. **Temporal analysis**: Correlate errors with image metadata (year, event type) to understand whether certain periods of each actor's career are more challenging for the model. Both actors have undergone appearance changes over time (weight fluctuations, ageing, facial hair styles), and understanding how these temporal factors affect recognition could inform more robust reference image selection strategies.

8. **Real-time deployment**: Package the system as a real-time classification pipeline with confidence-based rejection and human-in-the-loop review for uncertain cases. This would demonstrate the practical viability of the approach and allow measurement of end-to-end latency, throughput, and user experience.

9. **Adversarial robustness**: Evaluate whether adversarial perturbations can exploit the narrow margin between the two identities to cause targeted misclassifications. Given that the mean margin for errors is only 0.054, even small adversarial perturbations might be sufficient to flip borderline predictions. Understanding the adversarial vulnerability of lookalike classification systems is important for security-sensitive applications.

10. **Embedding space visualisation**: Apply t-SNE or UMAP to the 512-dimensional embeddings of all test images to visualise the cluster structure and identify patterns in the misclassified images. This visualisation could reveal whether errors occur at specific locations in the embedding space (e.g., at the boundary between the two clusters) or are scattered randomly, providing further insight into the nature of the model's failure modes.

---

## 23. References

1. Deng, J., Guo, J., Xue, N., & Zafeiriou, S. (2019). ArcFace: Additive Angular Margin Loss for Deep Face Recognition. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 4690–4699.

2. Guo, Y., Zhang, L., Hu, Y., He, X., & Gao, J. (2016). MS-Celeb-1M: A Dataset and Benchmark for Large-Scale Face Recognition. *European Conference on Computer Vision (ECCV)*, 87–102.

3. Huang, G. B., Ramesh, M., Berg, T., & Learned-Miller, E. (2007). Labeled Faces in the Wild: A Database for Studying Face Recognition in Unconstrained Environments. Technical Report 07-49, University of Massachusetts, Amherst.

4. Guo, J., Deng, J., Lattas, A., & Zafeiriou, S. (2021). Sample and Computation Redistribution for Efficient Face Detection. *arXiv preprint arXiv:2105.04714*.

5. An, X., Deng, J., Guo, J., Feng, Z., Zhu, X., Yang, J., & Liu, T. (2021). Killing Two Birds with One Stone: Efficient and Robust Training of Face Recognition CNNs by Partial FC. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*.

6. Schroff, F., Kalenichenko, D., & Philbin, J. (2015). FaceNet: A Unified Embedding for Face Recognition and Clustering. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 815–823.

7. Parkhi, O. M., Vedaldi, A., & Zisserman, A. (2015). Deep Face Recognition. *Proceedings of the British Machine Vision Conference (BMVC)*.

8. Kim, M., Jain, A. K., & Liu, X. (2022). AdaFace: Quality Adaptive Margin for Face Recognition. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*.

9. Meng, Q., Zhao, S., Huang, Z., & Zhou, F. (2021). MagFace: A Universal Representation for Face Recognition and Quality Assessment. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*.

10. Taigman, Y., Yang, M., Ranzato, M., & Wolf, L. (2014). DeepFace: Closing the Gap to Human-Level Performance in Face Verification. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 1701–1708.

---

## Appendix A: Complete Classification Report

```
                     precision    recall  f1-score   support

      Javier Bardem     0.9727    0.9957    0.9841       465
Jeffrey Dean Morgan     0.9967    0.9789    0.9877       615

           accuracy                         0.9861      1080
          macro avg     0.9847    0.9873    0.9859      1080
       weighted avg     0.9864    0.9861    0.9861      1080
```

## Appendix B: Confusion Matrix (Raw Counts)

```
                        Predicted:           Predicted:
                        Javier Bardem        Jeffrey Dean Morgan
True: Javier Bardem          463                    2
True: Jeffrey D. Morgan       13                  602
```

## Appendix C: Similarity Score Summary Statistics

```
       score_bardem  score_morgan     margin  max_score
count     1080.0000     1080.0000  1080.0000  1080.0000
mean         0.3105        0.3793     0.4620     0.5759
std          0.2192        0.2795     0.1318     0.1374
min         -0.1020       -0.0372     0.0000     0.0000
25%          0.1245        0.0920     0.3930     0.5201
50%          0.1854        0.4383     0.4779     0.5957
75%          0.5528        0.6488     0.5397     0.6610
max          0.7066        0.8590     0.7905     0.8590
```

## Appendix D: Generated Visualisations

The following visualisations were generated during the experiment and saved to the output directory:

1. **00_dashboard.png** — Comprehensive summary dashboard with 7 sub-plots covering all major analyses
2. **confusion_matrix.png** — Side-by-side confusion matrices (raw counts and row-normalised)
3. **similarity_distributions.png** — Three-panel plot: own-reference similarity distributions, margin histogram (correct vs incorrect), and score space scatter
4. **roc_curve.png** — ROC curve with AUC annotation and operating point marker
5. **precision_recall_curve.png** — Precision-Recall curve with AP annotation and prevalence baseline
6. **per_class_performance.png** — Three-panel per-class analysis: precision/recall/F1 bars, TP/FP/FN counts, and per-class recall
7. **margin_analysis.png** — Two-panel margin analysis: CDF comparison and accuracy-coverage trade-off curve
8. **score_separation.png** — Score difference histogram by true class with decision boundary
9. **threshold_sensitivity.png** — Accuracy vs decision threshold sweep with default and optimal markers

## Appendix E: Experimental Configuration

```
Model:              InsightFace buffalo_l
Detection Model:    SCRFD-10GF
Recognition Model:  ArcFace ResNet-100
Detection Size:     640 × 640 pixels
Embedding Dim:      512
Similarity Metric:  Cosine similarity (= dot product for L2-normalised vectors)
Decision Rule:      argmax(cosine_similarity) — no threshold
Reference Images:   1 per person
Test Images:        1,080 total (465 Bardem, 615 Morgan)
Execution Provider: CPUExecutionProvider
Processing Speed:   3.49 images/second
Total Runtime:      ~5 min 9 sec
```

## Appendix F: Error Distribution Summary

```
Error Type                                    Count    Mean Margin    Range
─────────────────────────────────────────────────────────────────────────────
Bardem → Morgan (false Morgan)                  2       0.1891      [0.1349, 0.2434]
Morgan → Bardem (false Bardem)                 13       0.0333      [0.0000, 0.0871]
No face detected (fallback assignment)          2       0.0000       N/A
─────────────────────────────────────────────────────────────────────────────
Total errors                                   15
```

---

*Report generated from experiment notebook: `Experiment_insightface_only_lookalike_testset.ipynb`*
*Experiment date: February 22, 2026*
*Report date: February 23, 2026*

---

## Appendix G: Glossary of Key Terms

- **ArcFace**: An additive angular margin loss function for training deep face recognition networks, which enforces enhanced discriminability by adding an angular margin penalty to the target logit during softmax-based classification training.
- **Balanced Accuracy**: The arithmetic mean of recall values computed for each class independently, which compensates for class imbalance in the test set.
- **Cohen's Kappa**: A statistic that measures inter-rater agreement for categorical items, adjusted for agreement expected by chance alone.
- **Cosine Similarity**: A measure of similarity between two non-zero vectors defined as the cosine of the angle between them, ranging from negative one to positive one.
- **Decision Margin**: The difference between the highest and second-highest similarity scores, used as a proxy for prediction confidence.
- **Embedding**: A mapping of high-dimensional data into a lower-dimensional vector space where geometric relationships encode semantic similarities.
- **F1 Score**: The harmonic mean of precision and recall, providing a single metric that balances both concerns.
- **InsightFace**: An open-source deep face analysis toolkit providing state-of-the-art face detection, recognition, and analysis models.
- **Matthews Correlation Coefficient (MCC)**: A balanced measure of binary classification quality that returns a value between negative one and positive one, where positive one represents perfect prediction.
- **ROC AUC**: The area under the Receiver Operating Characteristic curve, measuring a classifier's ability to distinguish between classes across all decision thresholds.
- **SCRFD**: Sample and Computation Redistribution for Efficient Face Detection, a single-stage face detector optimised for accuracy under computational constraints.
