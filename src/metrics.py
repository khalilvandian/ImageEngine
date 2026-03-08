import os
import json
import io
import base64
import datetime

import numpy as np
import face_recognition
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    fbeta_score, confusion_matrix,
    precision_recall_curve, roc_curve, auc,
)
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm

from src.classification import FaceDetector, get_classifier, load_celebrities_from_json
from src.image_utils import draw_bounding_boxes
from src.logging_utils import setup_logger

logger = setup_logger()

def load_test_set(json_path):
    """
    Loads the test set from a JSON file.

    Args:
        json_path (str): The path to the test set JSON file.

    Returns:
        tuple: A tuple containing two lists: image_paths and ground_truth_labels (list of lists of strings).
    """
    image_paths = []
    ground_truth_labels = []

    try:
        with open(json_path, 'r') as f:
            test_set_data = json.load(f)
    except FileNotFoundError as e:
        logger.error(f"Test set file not found at {json_path}: {e}", exc_info=True)
        return [], []
    except json.JSONDecodeError as e:
        logger.error(f"Could not decode JSON from {json_path}: {e}", exc_info=True)
        return [], []

    for item in test_set_data:
        image_paths.append(item["path"])
        # Return the full list of labels
        if item["labels"]:
            ground_truth_labels.append(item["labels"])
        else:
            ground_truth_labels.append([])

    return image_paths, ground_truth_labels

def run_classification_on_test_set(classifier, image_paths, output_image_dir=None):
    """
    Runs classification on the test set and returns the predictions.
    Optionally saves annotated images to a specified directory.

    Args:
        classifier: An instance of a Classifier.
        image_paths (list): A list of image paths.
        output_image_dir (str, optional): Directory to save annotated images. Defaults to None.

    Returns:
        list: A list of lists, where each inner list contains detected names for an image.
    """
    total_images = len(image_paths)
    logger.info(f"Running classification on {total_images} test images with {classifier.name}")

    # If output_image_dir is provided, create a unique subdirectory for this test run
    full_output_dir = None
    if output_image_dir:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_folder_name = f"{timestamp}_{classifier.name}_test_tolerance_{getattr(classifier, 'tolerance', 'N-A')}_threshold_{getattr(classifier, 'threshold', 'N-A')}"
        full_output_dir = os.path.join(output_image_dir, experiment_folder_name)
        os.makedirs(full_output_dir, exist_ok=True)
        logger.info(f"Test output images will be saved to: {full_output_dir}")

    # Batch classify to allow classifiers to serialize detection/identification internally
    results_by_image = classifier.classify_images(image_paths)

    all_detections = []
    for image_path in image_paths:
        results = results_by_image.get(image_path, [])
        detected_names = [r.get('name', 'Unknown') for r in results]
        all_detections.append(detected_names)
        
        if full_output_dir:
            try:
                annotated_image = draw_bounding_boxes(image_path, results)
                filename = os.path.basename(image_path)
                output_path = os.path.join(full_output_dir, filename)
                annotated_image.save(output_path)
                logger.debug(f"Saved annotated test image to {output_path}")
            except Exception as e:
                logger.error(f"Failed to save annotated image for {image_path}: {e}", exc_info=True)

    logger.info("Completed test set classification.")
    return all_detections


def run_face_detection_on_test_set(detector_model, image_paths, output_image_dir=None, detection_upsample=None, enable_multi_pass=False):
    """Runs face detection only and optionally saves annotated images.

    Returns a tuple of detections map and the output directory used (or None).
    """
    total_images = len(image_paths)
    logger.info(f"Running face detection on {total_images} images with detector={detector_model}")

    if detection_upsample is None:
        try:
            detection_upsample = int(os.getenv("FR_UPSAMPLE", "1"))
        except ValueError:
            detection_upsample = 1

    try:
        image_batch_size = int(os.getenv("FR_IMAGE_BATCH", "4"))
    except ValueError:
        image_batch_size = 4

    try:
        detection_batch_size = int(os.getenv("FR_DETECT_BATCH", "4"))
    except ValueError:
        detection_batch_size = 4

    detector = FaceDetector(
        model=detector_model,
        upsample=detection_upsample,
        enable_multi_pass=enable_multi_pass,
    )

    full_output_dir = None
    if output_image_dir:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_folder_name = f"{timestamp}_{detector_model}_only-faceDetector_upsample_{detection_upsample}"
        full_output_dir = os.path.join(output_image_dir, experiment_folder_name)
        os.makedirs(full_output_dir, exist_ok=True)
        logger.info(f"Face detection output images will be saved to: {full_output_dir}")

    detections_map = {}
    prog_detect = tqdm(total=total_images, desc=f"Detecting faces ({detector_model})", unit="img")

    for start in range(0, total_images, image_batch_size):
        batch_paths = image_paths[start:start + image_batch_size]
        images = []
        valid_paths = []

        for path in batch_paths:
            try:
                images.append(face_recognition.load_image_file(path))
                valid_paths.append(path)
            except FileNotFoundError:
                logger.error(f"Image not found: {path}", exc_info=True)
                detections_map[path] = []

        if not images:
            prog_detect.update(len(batch_paths))
            continue

        face_locations_batch = detector.detect_faces_batch(
            images,
            batch_size=min(detection_batch_size, len(images)),
        )

        for i, image_path in enumerate(valid_paths):
            locations = face_locations_batch[i] if i < len(face_locations_batch) else []
            detections_map[image_path] = [
                {"name": "Face", "location": loc} for loc in locations
            ]

        prog_detect.update(len(batch_paths))

    prog_detect.close()

    if full_output_dir:
        for image_path, detections in detections_map.items():
            try:
                annotated_image = draw_bounding_boxes(image_path, detections)
                filename = os.path.basename(image_path)
                output_path = os.path.join(full_output_dir, filename)
                annotated_image.save(output_path)
                logger.debug(f"Saved face detection image to {output_path}")
            except Exception as e:
                logger.error(f"Failed to save annotated detection image for {image_path}: {e}", exc_info=True)

    logger.info("Completed face detection run.")
    return detections_map, full_output_dir

def normalize_detections_for_metrics(ground_truth_labels, all_detections):
    """
    Normalizes detections to handle 'Unknown' labels appropriately.
    
    Logic:
    - If ground truth is empty (no known celebrities) and detection is ['Unknown'], 
      normalize to [] (correctly identified as no known celebrities)
    - If ground truth has known celebrities and detection contains 'Unknown' mixed with other labels,
      remove 'Unknown' (it's correct that other people exist, but we only care about known celebrities)
    - If ground truth has known celebrities and detection is only ['Unknown'],
      remove 'Unknown' (no known celebrities detected when expected)
    
    Args:
        ground_truth_labels (list of lists): The ground truth labels.
        all_detections (list of lists): The detected names for each image.
    
    Returns:
        list: Normalized detections.
    """
    normalized_detections = []
    
    for gt, det in zip(ground_truth_labels, all_detections):
        if not gt:  # Ground truth is empty (no known celebrities in image)
            # If we only detected "Unknown", that's correct -> normalize to []
            if det == ["Unknown"]:
                normalized_detections.append([])
            else:
                # If we detected known celebrities when there are none, keep as is (will be marked as false positive)
                normalized_detections.append(det)
        else:
            # Ground truth has known celebrities
            # Remove "Unknown" from detections as it represents other faces that are not in our celebrity list
            # This is correct behavior and shouldn't penalize the metrics
            filtered_det = [name for name in det if name != "Unknown"]
            normalized_detections.append(filtered_det)
    
    return normalized_detections

def calculate_metrics(ground_truth_labels, all_detections):
    """
    Calculates classification metrics for multi-label multi-class data.
    
    Handles 'Unknown' labels intelligently:
    - 'Unknown' when ground truth is empty is considered correct (no known celebrities)
    - 'Unknown' mixed with known celebrities is filtered out (correctly identifies other people exist)

    Args:
        ground_truth_labels (list of lists): The ground truth labels.
        all_detections (list of lists): The detected names for each image.

    Returns:
        dict: A dictionary containing the calculated metrics.
    """
    # Handle edge cases early to avoid sklearn target validation errors.
    if not ground_truth_labels or not all_detections:
        logger.warning("Missing ground truth or predictions; returning empty metrics.")
        return {
            "accuracy (exact match)": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "area_under_curve": "N/A (Multi-label)",
        }

    # Normalize detections to handle "Unknown" labels appropriately
    normalized_detections = normalize_detections_for_metrics(ground_truth_labels, all_detections)
    logger.debug(f"Normalized detections from {all_detections} to {normalized_detections}")

    mlb = MultiLabelBinarizer()
    
    # Fit on both true labels and detected labels to ensure all classes are covered
    # (including those detected but not in GT, and vice-versa)
    # We combine them just for fitting the classes, then transform separately
    all_labels = []
    for labels in ground_truth_labels:
        all_labels.extend(labels)
    for labels in normalized_detections:
        all_labels.extend(labels)

    classes = sorted(list(set(all_labels)))

    # If there are no labels at all, sklearn accuracy/precision will raise because
    # it cannot infer a valid target type. Return zeros with a clear signal instead.
    if not classes:
        logger.warning("No labels found in ground truth or predictions; returning zeroed metrics.")
        return {
            "accuracy (exact match)": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "area_under_curve": "N/A (Multi-label)",
        }

    mlb.fit([classes])

    y_true = mlb.transform(ground_truth_labels)
    y_pred = mlb.transform(normalized_detections)

    # Accuracy in multi-label is "Subset Accuracy" (strict match)
    accuracy = accuracy_score(y_true, y_pred)
    
    # Weighted average calculates metrics for each label, and finds their average weighted by support
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    return {
        "accuracy (exact match)": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "area_under_curve": "N/A (Multi-label)"
    }

def plot_confusion_matrix(ground_truth_labels, all_detections):
    """
    Generates a placeholder for Confusion Matrix as it's not suitable for Multi-Label.

    Args:
        ground_truth_labels (list): The ground truth labels.
        all_detections (list of lists): The detected names.

    Returns:
        str: Base64 encoded PNG image of the placeholder.
    """
    plt.figure(figsize=(6, 4))
    plt.text(0.5, 0.5, "Standard Confusion Matrix\nnot applicable for\nMulti-Label Classification", 
             horizontalalignment='center', verticalalignment='center')
    plt.title("Confusion Matrix")
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def plot_roc_curve(ground_truth_labels, predictions):
    """
    Generates an ROC curve plot.
    
    Note: ROC Curve is not easily applicable to this multi-class string-label logic 
    without probability scores. Returning a placeholder.

    Args:
        ground_truth_labels (list): The ground truth labels.
        predictions (list): The predicted labels.

    Returns:
        str: Base64 encoded PNG image of a placeholder.
    """
    plt.figure(figsize=(6, 4))
    plt.text(0.5, 0.5, "ROC Curve not available for Multi-class text labels", 
             horizontalalignment='center', verticalalignment='center')
    plt.title("ROC Curve")
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def save_test_output_to_csv(image_paths, all_detections, ground_truth_labels, model_name, output_dir="test_outputs"):
    """
    Saves the test output to a CSV file.

    Args:
        image_paths (list): A list of image paths.
        all_detections (list of lists): A list of lists of detected labels.
        ground_truth_labels (list of lists): A list of lists of ground truth labels.
        model_name (str): The name of the model being tested.
        output_dir (str): The directory to save the CSV file in.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{model_name}_{timestamp}.csv"
    filepath = os.path.join(output_dir, filename)

    # Convert list of lists to string representation for CSV
    detections_str = [", ".join(d) for d in all_detections]
    ground_truth_str = [", ".join(g) for g in ground_truth_labels]

    df = pd.DataFrame({
        "image_path": image_paths,
        "detected_labels": detections_str,
        "true_labels": ground_truth_str
    })

    df.to_csv(filepath, index=False)
    return filepath

def load_test_config(config_path):
    """
    Loads the test configuration from a JSON file.

    Args:
        config_path (str): The path to the config file.

    Returns:
        dict: The test configuration.
    """
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except FileNotFoundError as e:
        logger.error(f"Test config file not found at {config_path}: {e}", exc_info=True)
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Could not decode JSON from {config_path}: {e}", exc_info=True)
        return None


# ======================================================================
# Advanced evaluation metrics (F-beta, macro/micro P/R, P@R, R@P, curves)
# ======================================================================

def calculate_advanced_metrics(
    ground_truth_labels,
    all_detections,
    per_image_scores,
    identity_names,
    f_beta: float = 1.0,
    fixed_recall_level: float = 0.80,
    fixed_precision_level: float = 0.80,
):
    """
    Compute a comprehensive set of evaluation metrics for multi-label
    face recognition.

    Parameters
    ----------
    ground_truth_labels : list[list[str]]
        Per-image ground truth label sets.
    all_detections : list[list[str]]
        Per-image raw detection lists (may contain 'Unknown').
    per_image_scores : list[dict[str, float]]
        Continuous per-identity similarity scores for every image
        (used for ROC / PR curves).
    identity_names : list[str]
        Sorted list of known identity names (keys of the reference DB).
    f_beta : float
        Beta parameter for F-beta score (default 1.0 = F1).
    fixed_recall_level : float
        The recall level at which to report precision.
    fixed_precision_level : float
        The precision level at which to report recall.

    Returns
    -------
    dict with keys:
        scalar_metrics   – dict of aggregate metric values
        per_class_metrics – dict[class_name -> metric_dict]
        curve_data       – dict with ROC & PR curve arrays for plotting
    """
    normalized = normalize_detections_for_metrics(ground_truth_labels, all_detections)

    # ----- Build label matrices -----
    mlb = MultiLabelBinarizer()

    all_labels = set()
    for labels in ground_truth_labels:
        all_labels.update(labels)
    for labels in normalized:
        all_labels.update(labels)
    # Always include the identity names (some may not appear in GT or preds)
    all_labels.update(identity_names)
    classes = sorted(all_labels)

    if not classes:
        return _empty_advanced_metrics()

    mlb.fit([classes])
    y_true = mlb.transform(ground_truth_labels)
    y_pred = mlb.transform(normalized)
    class_names = list(mlb.classes_)

    # ----- Scalar metrics -----
    subset_accuracy = float(accuracy_score(y_true, y_pred))

    fbeta_macro = float(fbeta_score(y_true, y_pred, beta=f_beta, average='macro', zero_division=0))
    fbeta_micro = float(fbeta_score(y_true, y_pred, beta=f_beta, average='micro', zero_division=0))

    precision_macro = float(precision_score(y_true, y_pred, average='macro', zero_division=0))
    precision_micro = float(precision_score(y_true, y_pred, average='micro', zero_division=0))
    recall_macro = float(recall_score(y_true, y_pred, average='macro', zero_division=0))
    recall_micro = float(recall_score(y_true, y_pred, average='micro', zero_division=0))

    precision_weighted = float(precision_score(y_true, y_pred, average='weighted', zero_division=0))
    recall_weighted = float(recall_score(y_true, y_pred, average='weighted', zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))

    # ----- Per-class metrics + curve data -----
    per_class = {}
    roc_curves = {}   # class_name -> (fpr, tpr, roc_auc)
    pr_curves = {}    # class_name -> (precision_arr, recall_arr, pr_auc)

    p_at_r_per_class = {}  # precision@fixed_recall per class
    r_at_p_per_class = {}  # recall@fixed_precision per class

    beta2 = f_beta ** 2

    for class_idx, class_name in enumerate(class_names):
        y_true_binary = y_true[:, class_idx]
        y_pred_binary = y_pred[:, class_idx]

        tp = int(np.sum((y_pred_binary == 1) & (y_true_binary == 1)))
        fp = int(np.sum((y_pred_binary == 1) & (y_true_binary == 0)))
        fn = int(np.sum((y_pred_binary == 0) & (y_true_binary == 1)))
        tn = int(np.sum((y_pred_binary == 0) & (y_true_binary == 0)))

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
        if (prec + rec) > 0:
            fb = (1 + beta2) * prec * rec / (beta2 * prec + rec)
        else:
            fb = 0.0

        # Continuous scores for this class
        # For "None" class, use (1 - max identity similarity) as the score:
        # high score ⇒ no known identity matched well ⇒ likely "None".
        if class_name == "None":
            y_scores = np.array([
                1.0 - max(s.values()) if s else 1.0
                for s in per_image_scores
            ])
        else:
            y_scores = np.array([
                s.get(class_name, 0.0) for s in per_image_scores
            ])

        # Check for the degenerate case where the class never appears or
        # we don't have scores (all zeros).
        has_positive = int(y_true_binary.sum()) > 0
        has_negative = int((1 - y_true_binary).sum()) > 0

        # PR curve
        if has_positive:
            prec_curve, rec_curve, _ = precision_recall_curve(y_true_binary, y_scores)
            pr_auc_val = float(auc(rec_curve, prec_curve))

            # Precision @ fixed recall
            p_at_r = float(np.interp(fixed_recall_level, rec_curve[::-1], prec_curve[::-1], left=0.0, right=0.0))

            # Recall @ fixed precision
            valid_recalls = rec_curve[prec_curve >= fixed_precision_level]
            r_at_p = float(valid_recalls.max()) if len(valid_recalls) > 0 else 0.0
        else:
            prec_curve = np.array([1.0, 0.0])
            rec_curve = np.array([0.0, 1.0])
            pr_auc_val = 0.0
            p_at_r = 0.0
            r_at_p = 0.0

        pr_curves[class_name] = (prec_curve, rec_curve, pr_auc_val)
        p_at_r_per_class[class_name] = p_at_r
        r_at_p_per_class[class_name] = r_at_p

        # ROC curve
        if has_positive and has_negative:
            fpr, tpr, _ = roc_curve(y_true_binary, y_scores)
            roc_auc_val = float(auc(fpr, tpr))
        else:
            fpr = np.array([0.0, 1.0])
            tpr = np.array([0.0, 1.0])
            roc_auc_val = 0.0

        roc_curves[class_name] = (fpr, tpr, roc_auc_val)

        per_class[class_name] = {
            "precision": prec,
            "recall": rec,
            "f_beta": fb,
            "accuracy": acc,
            "support": tp + fn,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "pr_auc": pr_auc_val,
            "roc_auc": roc_auc_val,
            "precision_at_fixed_recall": p_at_r,
            "recall_at_fixed_precision": r_at_p,
        }

    # ----- Macro-averaged P@R and R@P -----
    valid_p_at_r = [v for v in p_at_r_per_class.values()]
    valid_r_at_p = [v for v in r_at_p_per_class.values()]
    p_at_r_macro = float(np.mean(valid_p_at_r)) if valid_p_at_r else 0.0
    r_at_p_macro = float(np.mean(valid_r_at_p)) if valid_r_at_p else 0.0

    # Micro-averaged P@R and R@P (flatten all classes)
    y_true_flat = y_true.ravel()
    y_scores_flat = []
    for sample_idx in range(len(per_image_scores)):
        for c_name in class_names:
            if c_name == "None":
                s = per_image_scores[sample_idx]
                y_scores_flat.append(1.0 - max(s.values()) if s else 1.0)
            else:
                y_scores_flat.append(per_image_scores[sample_idx].get(c_name, 0.0))
    y_scores_flat = np.array(y_scores_flat)

    if int(y_true_flat.sum()) > 0:
        prec_micro_c, rec_micro_c, _ = precision_recall_curve(y_true_flat, y_scores_flat)
        p_at_r_micro = float(np.interp(fixed_recall_level, rec_micro_c[::-1], prec_micro_c[::-1], left=0.0, right=0.0))
        valid_rec_micro = rec_micro_c[prec_micro_c >= fixed_precision_level]
        r_at_p_micro = float(valid_rec_micro.max()) if len(valid_rec_micro) > 0 else 0.0
    else:
        p_at_r_micro = 0.0
        r_at_p_micro = 0.0

    # ----- Macro-averaged ROC AUC and PR AUC -----
    roc_aucs = [roc_curves[c][2] for c in class_names if roc_curves[c][2] > 0]
    pr_aucs = [pr_curves[c][2] for c in class_names if pr_curves[c][2] > 0]
    macro_roc_auc = float(np.mean(roc_aucs)) if roc_aucs else 0.0
    macro_pr_auc = float(np.mean(pr_aucs)) if pr_aucs else 0.0

    scalar_metrics = {
        "subset_accuracy": subset_accuracy,
        f"f_beta_macro (β={f_beta})": fbeta_macro,
        f"f_beta_micro (β={f_beta})": fbeta_micro,
        "precision_macro": precision_macro,
        "precision_micro": precision_micro,
        "recall_macro": recall_macro,
        "recall_micro": recall_micro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
        f"precision@recall={fixed_recall_level} (macro)": p_at_r_macro,
        f"precision@recall={fixed_recall_level} (micro)": p_at_r_micro,
        f"recall@precision={fixed_precision_level} (macro)": r_at_p_macro,
        f"recall@precision={fixed_precision_level} (micro)": r_at_p_micro,
        "macro_roc_auc": macro_roc_auc,
        "macro_pr_auc": macro_pr_auc,
    }

    return {
        "scalar_metrics": scalar_metrics,
        "per_class_metrics": per_class,
        "curve_data": {
            "roc_curves": roc_curves,
            "pr_curves": pr_curves,
            "class_names": class_names,
            "y_true": y_true,
            "y_pred": y_pred,
        },
    }


def _empty_advanced_metrics():
    return {
        "scalar_metrics": {
            "subset_accuracy": 0.0,
            "f_beta_macro (β=1.0)": 0.0,
            "f_beta_micro (β=1.0)": 0.0,
            "precision_macro": 0.0,
            "precision_micro": 0.0,
            "recall_macro": 0.0,
            "recall_micro": 0.0,
            "precision_weighted": 0.0,
            "recall_weighted": 0.0,
            "f1_weighted": 0.0,
        },
        "per_class_metrics": {},
        "curve_data": {},
    }


def plot_roc_curves_figure(curve_data):
    """
    Generate a matplotlib Figure with per-class and macro-averaged ROC curves.

    Parameters
    ----------
    curve_data : dict
        The ``curve_data`` dict returned by ``calculate_advanced_metrics``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    roc_curves = curve_data.get("roc_curves", {})
    class_names = curve_data.get("class_names", [])

    if not roc_curves:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "No ROC data available", ha="center", va="center", fontsize=14)
        return fig

    colors = plt.cm.tab10.colors
    fig, ax = plt.subplots(figsize=(9, 7))

    # Per-class curves
    for idx, class_name in enumerate(class_names):
        if class_name not in roc_curves:
            continue
        fpr, tpr, roc_auc_val = roc_curves[class_name]
        color = colors[idx % len(colors)]
        ax.plot(fpr, tpr, linewidth=1.5, alpha=0.55, color=color,
                label=f"{class_name} (AUC = {roc_auc_val:.3f})")

    # Macro-averaged ROC curve
    mean_fpr = np.linspace(0, 1, 200)
    mean_tpr = np.zeros_like(mean_fpr)
    n_valid = 0
    for class_name in class_names:
        if class_name not in roc_curves:
            continue
        fpr, tpr, roc_auc_val = roc_curves[class_name]
        if roc_auc_val > 0:
            mean_tpr += np.interp(mean_fpr, fpr, tpr)
            n_valid += 1
    if n_valid > 0:
        mean_tpr /= n_valid
        macro_auc = float(auc(mean_fpr, mean_tpr))
        ax.plot(mean_fpr, mean_tpr, linewidth=3, color="navy",
                label=f"Macro-avg (AUC = {macro_auc:.3f})")
        ax.fill_between(mean_fpr, mean_tpr, alpha=0.12, color="navy")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Positive Rate", fontsize=12, fontweight="bold")
    ax.set_title("ROC Curve — Per Class & Macro Average", fontsize=14, fontweight="bold")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3, linestyle="--")
    fig.tight_layout()
    return fig


def plot_pr_curves_figure(curve_data):
    """
    Generate a matplotlib Figure with per-class and macro-averaged
    Precision-Recall curves.

    Parameters
    ----------
    curve_data : dict
        The ``curve_data`` dict returned by ``calculate_advanced_metrics``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    pr_curves = curve_data.get("pr_curves", {})
    class_names = curve_data.get("class_names", [])

    if not pr_curves:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "No PR data available", ha="center", va="center", fontsize=14)
        return fig

    colors = plt.cm.tab10.colors
    fig, ax = plt.subplots(figsize=(9, 7))

    all_precisions = []
    all_recalls = []

    for idx, class_name in enumerate(class_names):
        if class_name not in pr_curves:
            continue
        prec, rec, pr_auc_val = pr_curves[class_name]
        color = colors[idx % len(colors)]
        ax.plot(rec, prec, linewidth=1.5, alpha=0.55, color=color,
                label=f"{class_name} (AUC = {pr_auc_val:.3f})")
        if pr_auc_val > 0:
            all_precisions.append(prec)
            all_recalls.append(rec)

    # Macro-averaged PR curve
    if all_precisions:
        mean_recall = np.linspace(0, 1, 200)
        mean_precision = np.zeros_like(mean_recall)
        for prec, rec in zip(all_precisions, all_recalls):
            mean_precision += np.interp(mean_recall, rec[::-1], prec[::-1])
        mean_precision /= len(all_precisions)
        macro_pr_auc = float(auc(mean_recall, mean_precision))

        ax.plot(mean_recall, mean_precision, linewidth=3, color="navy",
                label=f"Macro-avg (AUC = {macro_pr_auc:.3f})")
        ax.fill_between(mean_recall, mean_precision, alpha=0.12, color="navy")

    ax.set_xlabel("Recall", fontsize=12, fontweight="bold")
    ax.set_ylabel("Precision", fontsize=12, fontweight="bold")
    ax.set_title("Precision–Recall Curve — Per Class & Macro Average", fontsize=14, fontweight="bold")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(True, alpha=0.3, linestyle="--")
    fig.tight_layout()
    return fig
