import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import MultiLabelBinarizer
from src.classification import get_classifier, load_celebrities_from_json
import json
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import pandas as pd
import datetime
from src.image_utils import draw_bounding_boxes
from src.logging_utils import setup_logger
from tqdm import tqdm

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

    all_detections = []
    
    # If output_image_dir is provided, create a unique subdirectory for this test run
    if output_image_dir:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_folder_name = f"{timestamp}_{classifier.name}_test_tolerance_{getattr(classifier, 'tolerance', 'N-A')}_threshold_{getattr(classifier, 'threshold', 'N-A')}"
        full_output_dir = os.path.join(output_image_dir, experiment_folder_name)
        os.makedirs(full_output_dir, exist_ok=True)
        logger.info(f"Test output images will be saved to: {full_output_dir}")

    progress = tqdm(image_paths, desc="Classifying", unit="img")
    for image_path in progress:
        results = classifier.detect_celebrity(image_path)
        detected_names = [result['name'] for result in results]
        all_detections.append(detected_names)
        progress.set_postfix({'last': os.path.basename(image_path)})
        
        if output_image_dir:
            annotated_image = draw_bounding_boxes(image_path, results)
            filename = os.path.basename(image_path)
            output_path = os.path.join(full_output_dir, filename)
            annotated_image.save(output_path)
            logger.debug(f"Saved annotated test image to {output_path}")

    logger.info("Completed test set classification.")
    return all_detections

def calculate_metrics(ground_truth_labels, all_detections):
    """
    Calculates classification metrics for multi-label multi-class data.

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

    mlb = MultiLabelBinarizer()
    
    # Fit on both true labels and detected labels to ensure all classes are covered
    # (including those detected but not in GT, and vice-versa)
    # We combine them just for fitting the classes, then transform separately
    all_labels = []
    for labels in ground_truth_labels:
        all_labels.extend(labels)
    for labels in all_detections:
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
    y_pred = mlb.transform(all_detections)

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
