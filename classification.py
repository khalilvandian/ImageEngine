"""
This module provides a modular classification system for detecting celebrities in images.
It is designed to be extensible, allowing for the addition of new models and methods for comparison.

The module includes:
- An abstract base class `Classifier` that defines the interface for all classifiers.
- `FaceRecognitionClassifier`: A classifier that uses the `face_recognition` library. 
  It can be configured to use either 'cnn' or 'hog' models.
- `ViTClassifier`: A classifier that uses a pre-trained Vision Transformer (ViT) model 
  for face embedding and comparison.
- `get_classifier`: A factory function to easily create instances of different classifiers.
"""

import json
from abc import ABC, abstractmethod
import os
import face_recognition
from logging_utils import setup_logger

logger = setup_logger()

def load_celebrities_from_json(json_path):
    """
    Loads celebrity data from a JSON file.

    Args:
        json_path (str): The path to the JSON file.

    Returns:
        list: A list of dictionaries, where each dictionary contains
              "name" and "reference_image_path" for a celebrity.
    """
    logger.info(f"Loading celebrity data from {json_path}")
    try:
        with open(json_path, 'r') as f:
            celebrities_data = json.load(f)
        logger.info(f"Successfully loaded {len(celebrities_data)} celebrities from {json_path}")
        return celebrities_data
    except FileNotFoundError:
        logger.error(f"JSON file not found at {json_path}")
        return []
    except json.JSONDecodeError:
        logger.error(f"Could not decode JSON from {json_path}")
        return []

# --- Abstract Base Class for Classifiers ---

class Classifier(ABC):
    """
    Abstract base class for a classifier. It defines the common interface
    that all concrete classifier implementations must follow.
    """
    def __init__(self, name):
        """
        Initializes the classifier.
        
        Args:
            name (str): The name of the classifier model.
        """
        self.name = name

    @abstractmethod
    def detect_celebrity(self, image_path):
        """
        Analyzes a single image to determine if known celebrities are present.
        This method must be implemented by all subclasses.

        Args:
            image_path (str): The path to the image file.

        Returns:
            list: A list containing the names of detected celebrities, otherwise an empty list.
        """
        pass

    def classify_images(self, image_paths):
        """
        Takes a list of image paths and returns a dictionary with the classification results.

        Args:
            image_paths (list): A list of strings, where each string is a path to an image.

        Returns:
            dict: A dictionary where keys are image paths and values are lists of detected celebrity names.
        """
        output = {}
        for path in image_paths:
            output[path] = self.detect_celebrity(path)
        return output

# --- Face Recognition Classifier ---

class FaceRecognitionClassifier(Classifier):
    """
    A classifier that uses the 'face_recognition' library. This is based on the logic
    from the original classification.py and lapressHughJackmanDetector.py.
    """
    def __init__(self, name, celebrity_data, tolerance=0.6, model="cnn"):
        """
        Initializes the FaceRecognitionClassifier.

        Args:
            name (str): The name for this classifier instance.
            celebrity_data (list): A list of dictionaries, each with "name" and "reference_image_path".
            tolerance (float): How much distance between faces to consider it a match. Lower is stricter.
            model (str): The face detection model to use ('cnn' or 'hog').
        """
        super().__init__(name)
        logger.info(f"Initializing FaceRecognitionClassifier with model: {model}, tolerance: {tolerance}")
        self.tolerance = tolerance
        self.model = model
        self.known_face_encodings = []
        self.known_face_names = []

        for celebrity in celebrity_data:
            logger.info(f"Loading reference encoding for {celebrity['name']} from {celebrity['reference_image_path']}")
            encodings = self._load_reference_encoding(celebrity["reference_image_path"])
            if encodings:
                self.known_face_encodings.extend(encodings)
                self.known_face_names.extend([celebrity["name"]] * len(encodings))
                logger.info(f"Successfully loaded {len(encodings)} encodings for {celebrity['name']}")
            else:
                logger.warning(f"Could not load reference encoding for {celebrity['name']} from {celebrity['reference_image_path']}")

        if not self.known_face_encodings:
            logger.warning(f"Could not initialize {self.name}. No reference face encodings loaded.")
        else:
            logger.info(f"Successfully initialized {self.name} with {len(self.known_face_encodings)} total reference encodings.")

    def _load_reference_encoding(self, reference_image_path):
        try:
            logger.info(f"Loading reference image from {reference_image_path}")
            reference_image = face_recognition.load_image_file(reference_image_path)
            reference_face_encodings = face_recognition.face_encodings(reference_image)
            if reference_face_encodings:
                logger.info(f"Found {len(reference_face_encodings)} face(s) in {reference_image_path}")
                return reference_face_encodings
            else:
                logger.warning(f"Could not find a face in the reference image: {reference_image_path}")
                return []
        except FileNotFoundError:
            logger.error(f"Reference image not found at {reference_image_path}")
            return []

    def detect_celebrity(self, image_path):
        logger.info(f"Detecting celebrities in {image_path} using {self.name}")
        if not self.known_face_encodings:
            logger.warning("No known face encodings to compare against.")
            return []

        try:
            logger.info(f"Loading image from {image_path}")
            image = face_recognition.load_image_file(image_path)
        except FileNotFoundError:
            logger.error(f"Input image not found at {image_path}")
            return []

        logger.info(f"Finding face locations in {image_path} using model: {self.model}")
        face_locations = face_recognition.face_locations(image, number_of_times_to_upsample=1, model=self.model)
        logger.info(f"Found {len(face_locations)} face(s) in {image_path}")
        face_encodings = face_recognition.face_encodings(image, face_locations)

        detected_celebrities = []
        for i, face_encoding in enumerate(face_encodings):
            logger.info(f"Comparing face {i+1}/{len(face_encodings)} with known encodings.")
            matches = face_recognition.compare_faces(
                self.known_face_encodings,
                face_encoding,
                tolerance=self.tolerance
            )
            # Find all matches for the current face
            for j, is_match in enumerate(matches):
                if is_match:
                    celebrity_name = self.known_face_names[j]
                    logger.info(f"Match found for face {i+1}: {celebrity_name}")
                    detected_celebrities.append(celebrity_name)
        
        # Return unique names
        unique_celebrities = list(set(detected_celebrities))
        logger.info(f"Detected celebrities in {image_path}: {unique_celebrities}")
        return unique_celebrities

# --- Vision Transformer (ViT) Classifier ---

# Lazily import heavy libraries for ViT to avoid errors if they are not installed.
try:
    import torch
    import timm
    from PIL import Image
    import numpy as np
    from numpy.linalg import norm
    VIT_LIBRARIES_AVAILABLE = True
except ImportError:
    VIT_LIBRARIES_AVAILABLE = False

class ViTClassifier(Classifier):
    """
    A classifier that uses a Vision Transformer (ViT) model. This is based on the
    logic from the vit32_hugh_jackman_detector.ipynb notebook.
    """
    def __init__(self, name, celebrity_data, threshold=0.8):
        """
        Initializes the ViTClassifier.

        Args:
            name (str): The name for this classifier instance.
            celebrity_data (list): A list of dictionaries, each with "name" and "reference_image_path".
            threshold (float): Cosine similarity threshold for a match.
        """
        super().__init__(name)
        
        if not VIT_LIBRARIES_AVAILABLE:
            logger.error("Required libraries for ViTClassifier (torch, timm, Pillow, numpy) are not installed.")
            raise ImportError("Required libraries for ViTClassifier (torch, timm, Pillow, numpy) are not installed.")

        logger.info(f"Initializing ViTClassifier with threshold: {threshold}")
        self.threshold = threshold
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"ViTClassifier will use device: {self.device}")

        logger.info(f"Loading ViT model for '{self.name}'...")
        self.model = timm.create_model('vit_base_patch32_224_in21k', pretrained=True)
        self.model.to(self.device)
        self.model.eval()
        data_config = timm.data.resolve_data_config({}, model=self.model)
        self.transform = timm.data.create_transform(**data_config)
        logger.info("ViT model loaded.")
        
        self.reference_embeddings = []
        self.reference_names = []

        for celebrity in celebrity_data:
            logger.info(f"Generating reference embedding for {celebrity['name']} from {celebrity['reference_image_path']}")
            embedding = self._generate_reference_embedding(celebrity["reference_image_path"])
            if embedding is not None:
                self.reference_embeddings.append(embedding)
                self.reference_names.append(celebrity["name"])
                logger.info(f"Successfully generated embedding for {celebrity['name']}")
            else:
                logger.warning(f"Could not generate reference embedding for {celebrity['name']} from {celebrity['reference_image_path']}")

        if not self.reference_embeddings:
            logger.warning(f"Could not initialize {self.name}. No reference embeddings generated.")
        else:
            logger.info(f"Successfully initialized {self.name} with {len(self.reference_embeddings)} total reference embeddings.")

    def _get_embedding(self, face_image_pil):
        logger.info("Generating embedding for face image.")
        img_tensor = self.transform(face_image_pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.model.forward_features(img_tensor)
            embedding = embedding[:, 0]
        return embedding.cpu().numpy().flatten()

    def _generate_reference_embedding(self, reference_image_path):
        try:
            logger.info(f"Loading reference image from {reference_image_path}")
            reference_image = face_recognition.load_image_file(reference_image_path)
            face_locations = face_recognition.face_locations(reference_image)
            if face_locations:
                logger.info(f"Found {len(face_locations)} face(s) in {reference_image_path}")
                top, right, bottom, left = face_locations[0]
                reference_face_image = reference_image[top:bottom, left:right]
                reference_face_pil = Image.fromarray(reference_face_image)
                return self._get_embedding(reference_face_pil)
            else:
                logger.warning(f'No face found in the reference image at {reference_image_path}')
                return None
        except FileNotFoundError:
            logger.error(f'Reference image not found at {reference_image_path}')
            return None

    def detect_celebrity(self, image_path):
        logger.info(f"Detecting celebrities in {image_path} using {self.name}")
        if not self.reference_embeddings:
            logger.warning("No reference embeddings to compare against.")
            return []
        
        try:
            logger.info(f"Loading image from {image_path}")
            target_image = face_recognition.load_image_file(image_path)
            face_locations = face_recognition.face_locations(target_image)
            logger.info(f"Found {len(face_locations)} face(s) in {image_path}")
            
            if not face_locations:
                return []
            
            detected_celebrities = []
            for i, (top, right, bottom, left) in enumerate(face_locations):
                logger.info(f"Processing face {i+1}/{len(face_locations)}")
                face_image = target_image[top:bottom, left:right]
                face_pil = Image.fromarray(face_image)
                
                embedding = self._get_embedding(face_pil)
                
                for j, ref_embedding in enumerate(self.reference_embeddings):
                    cosine_similarity = np.dot(ref_embedding, embedding) / (norm(ref_embedding) * norm(embedding))
                    logger.info(f"Comparing with {self.reference_names[j]}: cosine similarity = {cosine_similarity}")
                    
                    if cosine_similarity > self.threshold:
                        celebrity_name = self.reference_names[j]
                        logger.info(f"Match found for face {i+1}: {celebrity_name}")
                        detected_celebrities.append(celebrity_name)
            
            unique_celebrities = list(set(detected_celebrities))
            logger.info(f"Detected celebrities in {image_path}: {unique_celebrities}")
            return unique_celebrities
            
        except FileNotFoundError:
            logger.error(f'Image not found at {image_path}')
            return []

# --- Classifier Factory ---

def get_classifier(classifier_type, celebrity_data):
    """
    Factory function to get a classifier instance. This provides a single point
    of entry for creating different types of classifiers.

    Args:
        classifier_type (str): The type of classifier to create. 
                               Options: "face_recognition_cnn", "face_recognition_hog", "vit_b32".
        celebrity_data (list): A list of dictionaries, each with "name" and "reference_image_path".

    Returns:
        Classifier: An instance of a Classifier subclass, or None if unavailable.
    """
    logger.info(f"Getting classifier of type: {classifier_type}")
    if classifier_type == "face_recognition_cnn":
        return FaceRecognitionClassifier(
            name="face_recognition_cnn",
            celebrity_data=celebrity_data,
            model="cnn"
        )
    elif classifier_type == "face_recognition_hog":
        return FaceRecognitionClassifier(
            name="face_recognition_hog",
            celebrity_data=celebrity_data,
            model="hog"
        )
    elif classifier_type == "vit_b32":
        if not VIT_LIBRARIES_AVAILABLE:
            logger.warning("ViT libraries not found. ViT classifier is unavailable.")
            return None
        return ViTClassifier(
            name="vit_b32",
            celebrity_data=celebrity_data,
            threshold=0.6
        )
    else:
        logger.error(f"Unknown classifier type: {classifier_type}")
        raise ValueError(f"Unknown classifier type: {classifier_type}")

# --- Example Usage ---
if __name__ == '__main__':
    # Configuration for the example
    CELEBRITIES_JSON = "celebrities.json"
    TEST_IMAGES = [
        "Images/3899/062770.jpg",       # An image expected to contain Hugh Jackman
        "Images/not_3899/017031.jpg"  # An image not expected to contain Hugh Jackman
    ]

    celebrity_data = load_celebrities_from_json(CELEBRITIES_JSON)
    if not celebrity_data:
        logger.error("No celebrity data loaded. Exiting example.")
    else:
        logger.info("--- Testing Face Recognition (CNN) ---")
        cnn_classifier = get_classifier("face_recognition_cnn", celebrity_data)
        if cnn_classifier:
            results_cnn = cnn_classifier.classify_images(TEST_IMAGES)
            logger.info(f"CNN Results: {results_cnn}")

        logger.info("\n--- Testing ViT-B/32 ---")
        vit_classifier = get_classifier("vit_b32", celebrity_data)
        if vit_classifier:
            results_vit = vit_classifier.classify_images(TEST_IMAGES)
            logger.info(f"ViT Results: {results_vit}")

        logger.info("\n--- Testing Face Recognition (HOG) ---")
        hog_classifier = get_classifier("face_recognition_hog", celebrity_data)
        if hog_classifier:
            results_hog = hog_classifier.classify_images(TEST_IMAGES)
            logger.info(f"HOG Results: {results_hog}")