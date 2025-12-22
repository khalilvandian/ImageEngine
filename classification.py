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
import tempfile
import face_recognition
from logging_utils import setup_logger
import numpy as np

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
    except FileNotFoundError as e:
        logger.error(f"JSON file not found at {json_path}: {e}", exc_info=True)
        return []
    except json.JSONDecodeError as e:
        logger.error(f"Could not decode JSON from {json_path}: {e}", exc_info=True)
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

    def detect_celebrity(self, image_path):
        """
        Analyzes a single image to determine if known celebrities are present.
        This method now calls the batch-processing `classify_images` method.

        Args:
            image_path (str): The path to the image file.

        Returns:
            list: A list of dictionaries containing 'name' and 'location' of detected celebrities.
        """
        return self.classify_images([image_path]).get(image_path, [])

    @abstractmethod
    def classify_images(self, image_paths):
        """
        Takes a list of image paths and returns a dictionary with the classification results.
        This method must be implemented by all subclasses to handle batch processing.

        Args:
            image_paths (list): A list of strings, where each string is a path to an image.

        Returns:
            dict: A dictionary where keys are image paths and values are lists of detected celebrity results.
        """
        pass

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
        except FileNotFoundError as e:
            logger.error(f"Reference image not found at {reference_image_path}: {e}", exc_info=True)
            return []

    def classify_images(self, image_paths):
        logger.info(f"Batch detecting celebrities in {len(image_paths)} images using {self.name}")
        if not self.known_face_encodings:
            logger.warning("No known face encodings to compare against.")
            # Still process images to find all faces, just label them as "Unknown"
            # return {path: [] for path in image_paths}

        images = [face_recognition.load_image_file(p) for p in image_paths]
        
        logger.info(f"Finding face locations in batch using model: {self.model}")
        batch_face_locations = face_recognition.batch_face_locations(images, number_of_times_to_upsample=1, batch_size=128)

        output = {}
        for i, image_path in enumerate(image_paths):
            face_locations = batch_face_locations[i]
            image = images[i]
            
            logger.info(f"Found {len(face_locations)} face(s) in {image_path}")
            face_encodings = face_recognition.face_encodings(image, face_locations)

            all_detections = []
            for j, face_encoding in enumerate(face_encodings):
                name = "Unknown"
                if self.known_face_encodings:
                    logger.info(f"Comparing face {j+1}/{len(face_encodings)} in {image_path} with known encodings.")
                    matches = face_recognition.compare_faces(
                        self.known_face_encodings,
                        face_encoding,
                        tolerance=self.tolerance
                    )
                    
                    face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)

                    if matches[best_match_index]:
                        name = self.known_face_names[best_match_index]
                        logger.info(f"Match found for face {j+1} in {image_path}: {name}")
                
                all_detections.append({
                    "name": name,
                    "location": face_locations[j]
                })
            
            output[image_path] = all_detections
            logger.info(f"Detected faces in {image_path}: {all_detections}")

        return output

# --- Vision Transformer (ViT) Classifier ---

# Lazily import heavy libraries for ViT to avoid errors if they are not installed.
try:
    import torch
    import timm
    from PIL import Image
    import numpy as np
    from numpy.linalg import norm
    VIT_LIBRARIES_AVAILABLE = True
except ImportError as e:
    logger.error(f"Failed to import ViT libraries. ViTClassifier will be unavailable. Error: {e}", exc_info=True)
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

    def classify_images(self, image_paths):
        logger.info(f"Batch detecting celebrities in {len(image_paths)} images using {self.name}")

        output = {path: [] for path in image_paths}
        
        try:
            images = [face_recognition.load_image_file(p) for p in image_paths]
        except FileNotFoundError as e:
            logger.error(f"Image not found: {e}", exc_info=True)
            return output

        face_locations_by_image = [face_recognition.face_locations(img) for img in images]

        # Initialize output with all detected faces as "Unknown"
        for i, image_path in enumerate(image_paths):
            for face_location in face_locations_by_image[i]:
                output[image_path].append({
                    "name": "Unknown",
                    "location": face_location
                })

        face_batch = []
        face_indices = [] # To map faces back to their original images

        for i, (image, face_locations) in enumerate(zip(images, face_locations_by_image)):
            for j, (top, right, bottom, left) in enumerate(face_locations):
                face_image = image[top:bottom, left:right]
                face_pil = Image.fromarray(face_image)
                face_batch.append(self.transform(face_pil))
                face_indices.append({'image_index': i, 'face_location': (top, right, bottom, left), 'face_in_image_index': j})
        
        if not face_batch:
            logger.info("No faces found in any of the images.")
            return output

        if not self.reference_embeddings:
            logger.warning("No reference embeddings to compare against.")
            return output

        face_tensors = torch.stack(face_batch).to(self.device)
        
        logger.info(f"Processing a batch of {len(face_tensors)} faces.")
        with torch.no_grad():
            embeddings = self.model.forward_features(face_tensors)
            embeddings = embeddings[:, 0].cpu().numpy()

        for i, embedding in enumerate(embeddings):
            cosine_similarities = [np.dot(ref_embedding, embedding) / (norm(ref_embedding) * norm(embedding)) for ref_embedding in self.reference_embeddings]
            best_match_index = np.argmax(cosine_similarities)

            if cosine_similarities[best_match_index] > self.threshold:
                celebrity_name = self.reference_names[best_match_index]
                original_image_index = face_indices[i]['image_index']
                face_in_image_index = face_indices[i]['face_in_image_index']
                image_path = image_paths[original_image_index]
                
                logger.info(f"Match found for a face in {image_path}: {celebrity_name} with similarity {cosine_similarities[best_match_index]}")
                
                output[image_path][face_in_image_index]['name'] = celebrity_name

        for path, results in output.items():
            logger.info(f"Detected faces in {path}: {results}")
            
        return output

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