import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, RocCurveDisplay
from classification import get_classifier, load_celebrities_from_json
import json
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

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
    except FileNotFoundError:
        return [], []

    for item in test_set_data:
        image_paths.append(item["path"])
        if "Hugh Jackman" in item["labels"]:
            ground_truth_labels.append(1)
        else:
            ground_truth_labels.append(0)

    return image_paths, ground_truth_labels

def run_classification_on_test_set(classifier, image_paths):
    """
    Runs classification on the test set and returns the predictions.

    Args:
        classifier: An instance of a Classifier.
        image_paths (list): A list of image paths.

    Returns:
        list: A list of predictions (1 for positive, 0 for negative).
    """
    predictions = []
    for image_path in image_paths:
        result = classifier.detect_celebrity(image_path)
        if "Hugh Jackman" in result:
            predictions.append(1)
        else:
            predictions.append(0)
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
    except ValueError:
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
    except FileNotFoundError:
        return None
