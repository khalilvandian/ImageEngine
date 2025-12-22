import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, RocCurveDisplay
from classification import get_classifier, load_celebrities_from_json
import json
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import pandas as pd
import datetime
from image_utils import draw_bounding_boxes
from logging_utils import setup_logger

logger = setup_logger()

def load_test_set(json_path):
    """
    Loads the test set from a JSON file.

    Args:
        json_path (str): The path to the test set JSON file.

    Returns:
        tuple: A tuple containing two lists: image_paths and ground_truth_labels.
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
        if "Hugh Jackman" in item["labels"]:
            ground_truth_labels.append(1)
        else:
            ground_truth_labels.append(0)

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
        list: A list of predictions (1 for positive, 0 for negative).
    """
    predictions = []
    
    # If output_image_dir is provided, create a unique subdirectory for this test run
    if output_image_dir:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_folder_name = f"{timestamp}_{classifier.name}_test_tolerance_{getattr(classifier, 'tolerance', 'N-A')}_threshold_{getattr(classifier, 'threshold', 'N-A')}"
        full_output_dir = os.path.join(output_image_dir, experiment_folder_name)
        os.makedirs(full_output_dir, exist_ok=True)
        logger.info(f"Test output images will be saved to: {full_output_dir}")

    for image_path in image_paths:
        results = classifier.detect_celebrity(image_path)
        # Extract names from the list of result dictionaries
        detected_names = [result['name'] for result in results]
        if "Hugh Jackman" in detected_names:
            predictions.append(1)
        else:
            predictions.append(0)
        
        # Save annotated image if output_image_dir is provided
        if output_image_dir:
            annotated_image = draw_bounding_boxes(image_path, results)
            filename = os.path.basename(image_path)
            output_path = os.path.join(full_output_dir, filename)
            annotated_image.save(output_path)
            logger.info(f"Saved annotated test image to {output_path}")

    return predictions

def calculate_metrics(ground_truth_labels, predictions):
    """
    Calculates classification metrics.

    Args:
        ground_truth_labels (list): The ground truth labels.
        predictions (list): The predicted labels.

    Returns:
        dict: A dictionary containing the calculated metrics.
    """
    accuracy = accuracy_score(ground_truth_labels, predictions)
    precision = precision_score(ground_truth_labels, predictions, zero_division=0)
    recall = recall_score(ground_truth_labels, predictions, zero_division=0)
    f1 = f1_score(ground_truth_labels, predictions, zero_division=0)
    
    try:
        auc = roc_auc_score(ground_truth_labels, predictions)
    except ValueError as e:
        logger.warning(f"Could not calculate AUC, likely due to only one class being present in the data. Error: {e}", exc_info=True)
        auc = "N/A"

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "area_under_curve": auc
    }

def plot_confusion_matrix(ground_truth_labels, predictions):
    """
    Generates a confusion matrix plot.

    Args:
        ground_truth_labels (list): The ground truth labels.
        predictions (list): The predicted labels.

    Returns:
        str: Base64 encoded PNG image of the confusion matrix.
    """
    cm = confusion_matrix(ground_truth_labels, predictions)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Negative", "Positive"],
                yticklabels=["Negative", "Positive"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def plot_roc_curve(ground_truth_labels, predictions):
    """
    Generates an ROC curve plot.

    Args:
        ground_truth_labels (list): The ground truth labels.
        predictions (list): The predicted labels.

    Returns:
        str: Base64 encoded PNG image of the ROC curve.
    """
    plt.figure(figsize=(6, 4))
    RocCurveDisplay.from_predictions(ground_truth_labels, predictions)
    plt.title("ROC Curve")
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def save_test_output_to_csv(image_paths, predictions, ground_truth_labels, model_name, output_dir="test_outputs"):
    """
    Saves the test output to a CSV file.

    Args:
        image_paths (list): A list of image paths.
        predictions (list): A list of predicted labels.
        ground_truth_labels (list): A list of ground truth labels.
        model_name (str): The name of the model being tested.
        output_dir (str): The directory to save the CSV file in.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{model_name}_{timestamp}.csv"
    filepath = os.path.join(output_dir, filename)

    df = pd.DataFrame({
        "image_path": image_paths,
        "predicted_label": predictions,
        "true_label": ground_truth_labels
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
