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
from src.logging_utils import setup_logger
import numpy as np
from tqdm import tqdm

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

# --- Face Detector Class ---

class FaceDetector:
    """
    A modular face detector that can use different detection models (CNN or HOG).
    Separates face detection from face recognition/identification.
    """
    def __init__(self, model="cnn", upsample=2, enable_multi_pass=True):
        """
        Initializes the FaceDetector.
        
        Args:
            model (str): The face detection model to use ('cnn' or 'hog').
                        CNN is more accurate but slower, HOG is faster but less accurate.
            upsample (int): How many times to upsample the image for detection.
                           Higher values (2-3) detect smaller faces but are slower.
                           Recommended: 2 for general use, 3 for very small faces, 1 for speed.
            enable_multi_pass (bool): If True and no faces found, retry with higher upsampling.
        """
        if model not in ["cnn", "hog"]:
            raise ValueError(f"Invalid face detection model: {model}. Must be 'cnn' or 'hog'.")
        self.model = model
        self.upsample = upsample
        self.enable_multi_pass = enable_multi_pass
        logger.info(f"FaceDetector initialized with model: {model}, upsample: {upsample}, multi-pass: {enable_multi_pass}")
    
    def detect_faces(self, image, number_of_times_to_upsample=None):
        """
        Detects faces in a single image.
        
        Args:
            image: A loaded image (numpy array from face_recognition.load_image_file).
            number_of_times_to_upsample (int, optional): Override the default upsampling value.
        
        Returns:
            list: A list of face locations as (top, right, bottom, left) tuples.
        """
        upsample = number_of_times_to_upsample if number_of_times_to_upsample is not None else self.upsample
        try:
            face_locations = face_recognition.face_locations(
                image, model=self.model, number_of_times_to_upsample=upsample
            )
        except Exception as e:
            logger.error(f"Face detection failed with model={self.model}: {e}", exc_info=True)
            return []
        
        # Multi-pass detection: if no faces found and multi-pass enabled, try with higher upsampling
        if not face_locations and self.enable_multi_pass and upsample < 3:
            logger.debug(
                f"No faces found with upsample={upsample}, retrying with upsample={upsample + 1}"
            )
            try:
                face_locations = face_recognition.face_locations(
                    image, model=self.model, number_of_times_to_upsample=upsample + 1
                )
            except Exception as e:
                logger.error(f"Face detection retry failed with model={self.model}: {e}", exc_info=True)
                return []
        
        return face_locations
    
    def detect_faces_batch(self, images, batch_size=32, number_of_times_to_upsample=None):
        """
        Detects faces in a batch of images for efficiency.
        
        Args:
            images (list): A list of loaded images (numpy arrays).
            batch_size (int): Number of images to process at once.
            number_of_times_to_upsample (int, optional): Override the default upsampling value.
        
        Returns:
            list: A list of lists, where each inner list contains face locations for that image.
        """
        upsample = number_of_times_to_upsample if number_of_times_to_upsample is not None else self.upsample
        
        if self.model == "cnn":
            # Allow disabling batch mode entirely via env var
            disable_batch = os.getenv("FR_CNN_BATCH", "1") != "1"
            group_batch_default = 4
            try:
                group_batch_size = int(os.getenv("FR_CNN_GROUP_BATCH", str(group_batch_default)))
            except ValueError:
                group_batch_size = group_batch_default
            if disable_batch:
                return [self.detect_faces(img, number_of_times_to_upsample=upsample) for img in images]
            # CNN batch detector requires all images in the batch to have same dimensions.
            # Group images by (height, width) and process each group separately.
            dims_to_indices = {}
            for idx, img in enumerate(images):
                try:
                    h, w = img.shape[0], img.shape[1]
                except Exception:
                    h, w = None, None
                dims_to_indices.setdefault((h, w), []).append(idx)

            results = [[] for _ in images]
            for (h, w), group_idxs in dims_to_indices.items():
                group_imgs = [images[i] for i in group_idxs]
                if h is None or w is None or not group_imgs:
                    # Fallback to per-image detection for malformed entries
                    for i in group_idxs:
                        results[i] = self.detect_faces(images[i], number_of_times_to_upsample=upsample)
                    continue

                try:
                    group_results = face_recognition.batch_face_locations(
                        group_imgs,
                        number_of_times_to_upsample=upsample,
                        batch_size=min(group_batch_size, len(group_imgs)),
                    )
                except Exception as e:
                    logger.warning(
                        f"CNN batch detector failed for group ({h}x{w}) ({e}). Using per-image CNN.",
                        exc_info=True,
                    )
                    group_results = [
                        self.detect_faces(img, number_of_times_to_upsample=upsample) for img in group_imgs
                    ]

                # Multi-pass retry for empty detections within this group
                if self.enable_multi_pass and upsample < 3:
                    retry_local_idxs = [i for i, locs in enumerate(group_results) if not locs]
                    if retry_local_idxs:
                        retry_imgs = [group_imgs[i] for i in retry_local_idxs]
                        try:
                            retry_results = face_recognition.batch_face_locations(
                                retry_imgs,
                                number_of_times_to_upsample=upsample + 1,
                                batch_size=min(batch_size, len(retry_imgs)),
                            )
                            for j, local_idx in enumerate(retry_local_idxs):
                                if retry_results[j]:
                                    group_results[local_idx] = retry_results[j]
                        except Exception as e:
                            logger.warning(
                                f"CNN retry batch failed for group ({h}x{w}) ({e}). Using per-image retry.",
                                exc_info=True,
                            )
                            for local_idx in retry_local_idxs:
                                try:
                                    group_results[local_idx] = face_recognition.face_locations(
                                        group_imgs[local_idx], model=self.model, number_of_times_to_upsample=upsample + 1
                                    )
                                except Exception as e2:
                                    logger.error(
                                        f"CNN per-image retry failed for ({h}x{w}) image index {local_idx}: {e2}",
                                        exc_info=True,
                                    )
                                    group_results[local_idx] = []

                # Place group results back into overall results list
                for local_idx, global_idx in enumerate(group_idxs):
                    results[global_idx] = group_results[local_idx]

            return results
        else:
            # HOG doesn't have native batch support, process sequentially
            return [self.detect_faces(img, number_of_times_to_upsample=upsample) for img in images]

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
    def __init__(self, name, celebrity_data, tolerance=0.6, face_detection_model="cnn", detection_upsample=2, enable_multi_pass=True):
        """
        Initializes the FaceRecognitionClassifier.

        Args:
            name (str): The name for this classifier instance.
            celebrity_data (list): A list of dictionaries, each with "name" and "reference_image_path".
            tolerance (float): How much distance between faces to consider it a match. Lower is stricter.
            face_detection_model (str): The face detection model to use ('cnn' or 'hog').
            detection_upsample (int): Upsampling factor for face detection (1-3, default: 2).
            enable_multi_pass (bool): Enable multi-pass detection for missed faces (default: True).
        """
        super().__init__(name)
        logger.info(f"Initializing FaceRecognitionClassifier with detection model: {face_detection_model}, tolerance: {tolerance}, upsample: {detection_upsample}")
        self.tolerance = tolerance
        self.face_detector = FaceDetector(model=face_detection_model, upsample=detection_upsample, enable_multi_pass=enable_multi_pass)
        self.known_face_encodings = []
        self.known_face_names = []

        for celebrity in celebrity_data:
            logger.debug(f"Loading reference encoding for {celebrity['name']} from {celebrity['reference_image_path']}")
            encodings = self._load_reference_encoding(celebrity["reference_image_path"])
            if encodings:
                self.known_face_encodings.extend(encodings)
                self.known_face_names.extend([celebrity["name"]] * len(encodings))
                logger.debug(f"Successfully loaded {len(encodings)} encodings for {celebrity['name']}")
            else:
                logger.warning(f"Could not load reference encoding for {celebrity['name']} from {celebrity['reference_image_path']}")

        if not self.known_face_encodings:
            logger.warning(f"Could not initialize {self.name}. No reference face encodings loaded.")
        else:
            logger.info(f"Successfully initialized {self.name} with {len(self.known_face_encodings)} total reference encodings.")

    def _load_reference_encoding(self, reference_image_path):
        try:
            logger.debug(f"Loading reference image from {reference_image_path}")
            reference_image = face_recognition.load_image_file(reference_image_path)
            reference_face_encodings = face_recognition.face_encodings(reference_image)
            if reference_face_encodings:
                logger.debug(f"Found {len(reference_face_encodings)} face(s) in {reference_image_path}")
                return reference_face_encodings
            else:
                logger.warning(f"Could not find a face in the reference image: {reference_image_path}")
                return []
        except FileNotFoundError as e:
            logger.error(f"Reference image not found at {reference_image_path}: {e}", exc_info=True)
            return []

    def classify_images(self, image_paths, image_batch_size=None, detection_batch_size=None):
        logger.info(f"Batch detecting celebrities in {len(image_paths)} images using {self.name}")
        if not self.known_face_encodings:
            logger.warning("No known face encodings to compare against.")

        output = {}

        # Allow environment overrides for batch sizes
        if image_batch_size is None:
            try:
                image_batch_size = int(os.getenv("FR_IMAGE_BATCH", "4"))
            except ValueError:
                image_batch_size = 4
        if detection_batch_size is None:
            try:
                detection_batch_size = int(os.getenv("FR_DETECT_BATCH", "4"))
            except ValueError:
                detection_batch_size = 4

        # Phase 1: detect faces and store locations only
        locations_map = {}
        prog_detect = tqdm(total=len(image_paths), desc=f"Detecting faces ({self.name})", unit="img")
        for start in range(0, len(image_paths), image_batch_size):
            batch_paths = image_paths[start:start + image_batch_size]

            images = []
            valid_paths = []
            for path in batch_paths:
                try:
                    images.append(face_recognition.load_image_file(path))
                    valid_paths.append(path)
                except FileNotFoundError:
                    logger.error(f"Image not found: {path}", exc_info=True)
                    locations_map[path] = []

            if not images:
                continue

            logger.info(
                f"Finding face locations for batch size {len(images)} using detection model: {self.face_detector.model} "
                f"with upsample={self.face_detector.upsample}"
            )
            batch_face_locations = self.face_detector.detect_faces_batch(
                images,
                batch_size=min(detection_batch_size, len(images)),
            )

            for i, image_path in enumerate(valid_paths):
                locations_map[image_path] = batch_face_locations[i]

            del images
            del batch_face_locations
            prog_detect.update(len(valid_paths))
        prog_detect.close()

        # Optionally free detector reference
        try:
            del self.face_detector
        except Exception:
            pass

        # Phase 2: identification using stored locations
        prog_ident = tqdm(total=len(image_paths), desc=f"Identifying faces ({self.name})", unit="img")
        for image_path in image_paths:
            face_locations = locations_map.get(image_path, [])
            if not face_locations:
                output[image_path] = []
                continue

            try:
                image = face_recognition.load_image_file(image_path)
            except FileNotFoundError:
                logger.error(f"Image not found: {image_path}", exc_info=True)
                output[image_path] = []
                continue

            logger.debug(f"Found {len(face_locations)} face(s) in {image_path}")
            face_encodings = face_recognition.face_encodings(image, face_locations)

            all_detections = []
            for j, face_encoding in enumerate(face_encodings):
                name = "Unknown"
                if self.known_face_encodings:
                    matches = face_recognition.compare_faces(
                        self.known_face_encodings,
                        face_encoding,
                        tolerance=self.tolerance,
                    )
                    face_distances = face_recognition.face_distance(
                        self.known_face_encodings, face_encoding
                    )
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = self.known_face_names[best_match_index]

                all_detections.append({
                    "name": name,
                    "location": face_locations[j],
                })

            output[image_path] = all_detections
            prog_ident.update(1)
            del image

        prog_ident.close()
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
    def __init__(self, name, celebrity_data, threshold=0.8, face_detection_model="cnn", detection_upsample=2, enable_multi_pass=True):
        """
        Initializes the ViTClassifier.

        Args:
            name (str): The name for this classifier instance.
            celebrity_data (list): A list of dictionaries, each with "name" and "reference_image_path".
            threshold (float): Cosine similarity threshold for a match.
            face_detection_model (str): The face detection model to use ('cnn' or 'hog'). Defaults to 'cnn'.
            detection_upsample (int): Upsampling factor for face detection (1-3, default: 2).
            enable_multi_pass (bool): Enable multi-pass detection for missed faces (default: True).
        """
        super().__init__(name)
        
        if not VIT_LIBRARIES_AVAILABLE:
            logger.error("Required libraries for ViTClassifier (torch, timm, Pillow, numpy) are not installed.")
            raise ImportError("Required libraries for ViTClassifier (torch, timm, Pillow, numpy) are not installed.")

        logger.info(f"Initializing ViTClassifier with threshold: {threshold}, detection model: {face_detection_model}, upsample: {detection_upsample}")
        self.threshold = threshold
        self.face_detector = FaceDetector(model=face_detection_model, upsample=detection_upsample, enable_multi_pass=enable_multi_pass)
        
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
            logger.debug(f"Generating reference embedding for {celebrity['name']} from {celebrity['reference_image_path']}")
            embedding = self._generate_reference_embedding(celebrity["reference_image_path"])
            if embedding is not None:
                self.reference_embeddings.append(embedding)
                self.reference_names.append(celebrity["name"])
                logger.debug(f"Successfully generated embedding for {celebrity['name']}")
            else:
                logger.warning(f"Could not generate reference embedding for {celebrity['name']} from {celebrity['reference_image_path']}")

        if not self.reference_embeddings:
            logger.warning(f"Could not initialize {self.name}. No reference embeddings generated.")
        else:
            logger.info(f"Successfully initialized {self.name} with {len(self.reference_embeddings)} total reference embeddings.")

    def _get_embedding(self, face_image_pil):
        logger.debug("Generating embedding for face image.")
        img_tensor = self.transform(face_image_pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.model.forward_features(img_tensor)
            embedding = embedding[:, 0]
        return embedding.cpu().numpy().flatten()

    def _generate_reference_embedding(self, reference_image_path):
        try:
            logger.debug(f"Loading reference image from {reference_image_path}")
            reference_image = face_recognition.load_image_file(reference_image_path)
            face_locations = self.face_detector.detect_faces(reference_image)
            if face_locations:
                logger.debug(f"Found {len(face_locations)} face(s) in {reference_image_path}")
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

    def classify_images(self, image_paths, image_batch_size=None, face_batch_size=None, detection_batch_size=None):
        logger.info(f"Batch detecting celebrities in {len(image_paths)} images using {self.name}")

        output = {path: [] for path in image_paths}

        if not self.reference_embeddings:
            logger.warning("No reference embeddings to compare against.")
            return output

        # Allow environment overrides for batch sizes
        if image_batch_size is None:
            try:
                image_batch_size = int(os.getenv("VIT_IMAGE_BATCH", "4"))
            except ValueError:
                image_batch_size = 4
        if face_batch_size is None:
            try:
                face_batch_size = int(os.getenv("VIT_FACE_BATCH", "8"))
            except ValueError:
                face_batch_size = 8
        if detection_batch_size is None:
            try:
                detection_batch_size = int(os.getenv("FR_DETECT_BATCH", "4"))
            except ValueError:
                detection_batch_size = 4

        def process_face_batch(face_tensors, face_indices, batch_paths):
            if not face_tensors:
                return
            face_tensor = torch.stack(face_tensors).to(self.device)
            logger.debug(f"Processing a batch of {len(face_tensor)} faces.")
            with torch.no_grad():
                embeddings = self.model.forward_features(face_tensor)
                embeddings = embeddings[:, 0].cpu().numpy()

            for embedding, info in zip(embeddings, face_indices):
                cosine_similarities = [
                    np.dot(ref_embedding, embedding) / (norm(ref_embedding) * norm(embedding))
                    for ref_embedding in self.reference_embeddings
                ]
                best_match_index = np.argmax(cosine_similarities)

                if cosine_similarities[best_match_index] > self.threshold:
                    celebrity_name = self.reference_names[best_match_index]
                    image_path = batch_paths[info['image_index']]
                    face_in_image_index = info['face_in_image_index']
                    logger.debug(
                        f"Match found for a face in {image_path}: {celebrity_name} "
                        f"with similarity {cosine_similarities[best_match_index]}"
                    )
                    output[image_path][face_in_image_index]['name'] = celebrity_name

            del face_tensor
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()

        # Phase 1: detect faces and store locations only
        prog_detect = tqdm(total=len(image_paths), desc=f"Detecting faces ({self.name})", unit="img")
        locations_map = {}
        for start in range(0, len(image_paths), image_batch_size):
            batch_paths = image_paths[start:start + image_batch_size]
            images = []
            for path in batch_paths:
                try:
                    images.append(face_recognition.load_image_file(path))
                except FileNotFoundError:
                    logger.error(f"Image not found: {path}", exc_info=True)
                    locations_map[path] = []
                    images.append(None)

            valid_images = [img for img in images if img is not None]
            if not valid_images:
                continue

            logger.info(
                f"Detecting faces for batch size {len(valid_images)} using detection model: {self.face_detector.model} "
                f"with upsample={self.face_detector.upsample}"
            )
            face_locations_by_image = self.face_detector.detect_faces_batch(
                valid_images,
                batch_size=min(detection_batch_size, len(valid_images)),
            )

            valid_idx = 0
            for path, image in zip(batch_paths, images):
                if image is None:
                    continue
                face_locations = face_locations_by_image[valid_idx]
                valid_idx += 1
                locations_map[path] = face_locations

            del images
            del face_locations_by_image
            prog_detect.update(len(batch_paths))

        prog_detect.close()

        # Optionally free detector reference
        try:
            del self.face_detector
        except Exception:
            pass

        # Phase 2: build face tensors and run embeddings separately
        prog_ident = tqdm(total=len(image_paths), desc=f"Embedding & identification ({self.name})", unit="img")
        for start in range(0, len(image_paths), image_batch_size):
            batch_paths = image_paths[start:start + image_batch_size]
            face_tensors = []
            face_indices = []

            for img_idx, path in enumerate(batch_paths):
                face_locations = locations_map.get(path, [])
                if not face_locations:
                    continue
                try:
                    image = face_recognition.load_image_file(path)
                except FileNotFoundError:
                    logger.error(f"Image not found: {path}", exc_info=True)
                    continue

                # Initialize output with Unknown
                if not output[path]:
                    for face_location in face_locations:
                        output[path].append({
                            "name": "Unknown",
                            "location": face_location,
                        })

                for face_in_image_index, (top, right, bottom, left) in enumerate(face_locations):
                    face_image = image[top:bottom, left:right]
                    face_pil = Image.fromarray(face_image)
                    face_tensors.append(self.transform(face_pil))
                    face_indices.append({
                        'image_index': img_idx,
                        'face_in_image_index': face_in_image_index,
                    })

                    if len(face_tensors) >= face_batch_size:
                        process_face_batch(face_tensors, face_indices, batch_paths)
                        face_tensors = []
                        face_indices = []

            # Process remaining faces
            process_face_batch(face_tensors, face_indices, batch_paths)
            prog_ident.update(len(batch_paths))

        prog_ident.close()

        for path, results in output.items():
            logger.debug(f"Detected faces in {path}: {results}")

        return output

# --- Classifier Factory ---

def get_classifier(classifier_type, celebrity_data, face_detection_model=None, detection_upsample=None, enable_multi_pass=None):
    """
    Factory function to get a classifier instance. This provides a single point
    of entry for creating different types of classifiers.

    Args:
        classifier_type (str): The type of classifier to create. 
                               Options: "face_recognition_cnn", "face_recognition_hog", "vit_b32".
        celebrity_data (list): A list of dictionaries, each with "name" and "reference_image_path".
        face_detection_model (str, optional): The face detection model to use ('cnn' or 'hog').
                                             If None, uses 'cnn' for face_recognition_cnn and vit_b32,
                                             and 'hog' for face_recognition_hog.

    Returns:
        Classifier: An instance of a Classifier subclass, or None if unavailable.
    """
    logger.info(f"Getting classifier of type: {classifier_type}")
    if classifier_type == "face_recognition_cnn":
        detection_model = face_detection_model if face_detection_model else "cnn"
        # Resolve upsample/multi-pass with env overrides if not explicitly provided
        if detection_upsample is None:
            try:
                detection_upsample = int(os.getenv("FR_UPSAMPLE", "1"))
            except ValueError:
                detection_upsample = 1
        if enable_multi_pass is None:
            enable_multi_pass = False
        return FaceRecognitionClassifier(
            name="face_recognition_cnn",
            celebrity_data=celebrity_data,
            face_detection_model=detection_model,
            detection_upsample=detection_upsample,
            enable_multi_pass=enable_multi_pass,
        )
    elif classifier_type == "face_recognition_hog":
        detection_model = face_detection_model if face_detection_model else "hog"
        if detection_upsample is None:
            try:
                detection_upsample = int(os.getenv("FR_UPSAMPLE", "1"))
            except ValueError:
                detection_upsample = 1
        if enable_multi_pass is None:
            enable_multi_pass = False
        return FaceRecognitionClassifier(
            name="face_recognition_hog",
            celebrity_data=celebrity_data,
            face_detection_model=detection_model,
            detection_upsample=detection_upsample,
            enable_multi_pass=enable_multi_pass,
        )
    elif classifier_type == "vit_b32":
        if not VIT_LIBRARIES_AVAILABLE:
            logger.warning("ViT libraries not found. ViT classifier is unavailable.")
            return None
        detection_model = face_detection_model if face_detection_model else "cnn"
        # Use lower upsampling for ViT to reduce memory usage
        return ViTClassifier(
            name="vit_b32",
            celebrity_data=celebrity_data,
            threshold=0.6,
            face_detection_model=detection_model,
            detection_upsample=1,  # Lower for memory efficiency
            enable_multi_pass=False  # Disable multi-pass to save memory
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