"""
Modular Classification System for Detecting Celebrities in Images

This module provides a highly composable architecture where:
- Detection models (buffalo_l, buffalo_m, cnn, hog, etc.) are pluggable
- Embedding models (insightface, face_recognition, vit, etc.) are pluggable  
- Matching methods (cosine_similarity, euclidean_distance, etc.) are pluggable

This enables testing ANY combination of detection + embedding + matching method.

Key Components:
- FaceDetector: Pluggable face detection (multiple model support)
- EmbeddingExtractor: Pluggable embedding generation (multiple model support)
- MatchingMethod: Pluggable similarity/distance computation
- UnifiedClassifier: Orchestrates the pipeline with configurable components
- get_classifier: Factory for creating configured classifiers
"""

import json
from abc import ABC, abstractmethod
import os
import tempfile
import face_recognition
from src.logging_utils import setup_logger
import numpy as np
from tqdm import tqdm
import cv2
from typing import List, Dict, Tuple, Optional, Any

# InsightFace imports (optional, with fallback)
try:
    from insightface.app import FaceAnalysis
    import onnxruntime as ort
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False

# Vision Transformer imports (optional)
try:
    import torch
    from transformers import CLIPProcessor, CLIPModel
    VIT_LIBRARIES_AVAILABLE = True
except ImportError:
    VIT_LIBRARIES_AVAILABLE = False

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
    A modular face detector supporting multiple detection models:
    - CNN/HOG (face_recognition library)
    - InsightFace model zoo: buffalo_l, buffalo_m, buffalo_s, antelopev2
    
    Separates face detection from face recognition/identification.
    """
    
    # InsightFace model zoo names
    INSIGHTFACE_MODELS = {"buffalo_l", "buffalo_m", "buffalo_s", "antelopev2"}
    LEGACY_MODELS = {"cnn", "hog"}
    VALID_MODELS = INSIGHTFACE_MODELS | LEGACY_MODELS
    
    def __init__(self, model="buffalo_l", upsample=2, enable_multi_pass=True, det_size=(640, 640)):
        """
        Initializes the FaceDetector.
        
        Args:
            model (str): The face detection model to use. Options:
                        InsightFace Model Zoo (recommended):
                        - 'buffalo_l': SCRFD-10GF (326MB, default) - highest accuracy, ~91.25% IJB-B
                        - 'buffalo_m': SCRFD-2.5GF (313MB) - same accuracy as buffalo_l, faster
                        - 'buffalo_s': SCRFD-500MF (159MB) - lightweight, lower accuracy ~71.87%
                        - 'antelopev2': SCRFD-10GF (407MB) - alternative high-end model
                        Legacy (face_recognition):
                        - 'cnn': CNN detector, more accurate but slower
                        - 'hog': HOG detector, faster but less accurate
            upsample (int): How many times to upsample the image (for CNN/HOG only).
                           Higher values (2-3) detect smaller faces but are slower.
                           Recommended: 2 for general use, 3 for small faces, 1 for speed.
            enable_multi_pass (bool): If True and no faces found, retry with higher upsampling.
            det_size (tuple): Detection size for InsightFace models (width, height).
                             Default (640, 640) provides good balance. Larger sizes detect smaller faces.
        """
        if model not in self.VALID_MODELS:
            raise ValueError(f"Invalid face detection model: {model}. Must be one of {self.VALID_MODELS}")
        
        if model in self.INSIGHTFACE_MODELS and not INSIGHTFACE_AVAILABLE:
            logger.warning(f"InsightFace not available, falling back to CNN model")
            model = "cnn"
        
        self.model = model
        self.upsample = upsample
        self.enable_multi_pass = enable_multi_pass
        self.det_size = det_size
        self.insightface_app = None
        
        # Initialize InsightFace if selected
        if self.model in self.INSIGHTFACE_MODELS:
            self.insightface_app = self._init_insightface()
        
        logger.info(f"FaceDetector initialized with model: {model}, det_size: {det_size}, multi-pass: {enable_multi_pass}")
    
    def _init_insightface(self):
        """
        Initializes InsightFace model zoo with GPU/CPU fallback logic.
        Automatically uses the specified model from the model zoo (buffalo_l, buffalo_m, buffalo_s, antelopev2).
        
        Returns:
            FaceAnalysis: Initialized InsightFace app for face detection with the specified model.
        """
        available = ort.get_available_providers()
        prefer_cuda = "CUDAExecutionProvider" in available
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if prefer_cuda else ["CPUExecutionProvider"]
        ctx_id = 0 if prefer_cuda else -1
        
        try:
            # FaceAnalysis automatically downloads and uses the model from model zoo
            app = FaceAnalysis(name=self.model, allowed_modules=["detection"], providers=providers)
            app.prepare(ctx_id=ctx_id, det_size=self.det_size)
            logger.info(f"InsightFace ({self.model}) initialized with providers={providers}, ctx_id={ctx_id}, det_size={self.det_size}")
            return app
        except Exception as e:
            if prefer_cuda:
                logger.warning(f"GPU initialization failed ({e}); retrying on CPU only.")
                providers = ["CPUExecutionProvider"]
                ctx_id = -1
                app = FaceAnalysis(allowed_modules=["detection"], providers=providers)
                app.prepare(ctx_id=ctx_id, det_size=self.det_size)
                logger.info(f"InsightFace initialized with providers={providers}, ctx_id={ctx_id}, det_size={self.det_size}")
                return app
            raise
    
    def detect_faces(self, image, number_of_times_to_upsample=None):
        """
        Detects faces in a single image.
        
        Args:
            image: A loaded image (numpy array from face_recognition.load_image_file or cv2.imread).
            number_of_times_to_upsample (int, optional): Override the default upsampling value (for cnn/hog only).
        
        Returns:
            list: A list of face locations as (top, right, bottom, left) tuples.
        """
        if self.model in self.INSIGHTFACE_MODELS:
            return self._detect_faces_insightface(image)
        
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
    
    def _detect_faces_insightface(self, image):
        """
        Detects faces using InsightFace model.
        
        Args:
            image: A loaded image (numpy array, RGB or BGR format).
        
        Returns:
            list: A list of face locations as (top, right, bottom, left) tuples.
        """
        if self.insightface_app is None:
            logger.error("InsightFace app not initialized")
            return []
        
        try:
            # InsightFace expects BGR format (OpenCV format).
            # Default: assume input is BGR from cv2.imread.
            # If input is RGB (from face_recognition.load_image_file), caller should convert upstream.
            image_bgr = image
            
            faces = self.insightface_app.get(image_bgr)

            # Convert InsightFace bbox format (x1, y1, x2, y2) to face_recognition format (top, right, bottom, left)
            face_locations = []
            for face in faces:
                x1, y1, x2, y2 = [int(v) for v in face.bbox]
                # Convert to (top, right, bottom, left)
                face_locations.append((y1, x2, y2, x1))

            return face_locations
        except Exception as e:
            logger.error(f"InsightFace detection failed: {e}", exc_info=True)
            return []

    def detect_faces_with_landmarks(self, image):
        """
        Detect faces and return landmarks and detection scores when available.

        Returns a list of dicts with keys:
        - 'bbox': (top, right, bottom, left)
        - 'kps': numpy array of shape (5, 2) or None
        - 'det_score': float or None
        """
        try:
            if self.model in self.INSIGHTFACE_MODELS:
                if self.insightface_app is None:
                    logger.error("InsightFace app not initialized")
                    return []

                # Ensure BGR for InsightFace
                if len(image.shape) == 3 and image.shape[2] == 3:
                    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                else:
                    image_bgr = image

                faces = self.insightface_app.get(image_bgr)
                results = []
                for face in faces:
                    x1, y1, x2, y2 = [int(v) for v in face.bbox]
                    bbox_trbl = (y1, x2, y2, x1)
                    kps = None
                    try:
                        if face.kps is not None:
                            kps = np.array(face.kps, dtype=np.float32)
                    except Exception:
                        kps = None
                    det_score = None
                    try:
                        det_score = float(face.det_score)
                    except Exception:
                        det_score = None
                    results.append({
                        "bbox": bbox_trbl,
                        "kps": kps,
                        "det_score": det_score,
                    })
                return results
            else:
                # Legacy detectors do not provide landmarks/scores
                bboxes = self.detect_faces(image)
                return [{"bbox": b, "kps": None, "det_score": None} for b in bboxes]
        except Exception as e:
            logger.error(f"Detection with landmarks failed: {e}", exc_info=True)
            return []
    
    def detect_faces_batch(self, images, batch_size=32, number_of_times_to_upsample=None):
        """
        Detects faces in a batch of images for efficiency.
        
        Args:
            images (list): A list of loaded images (numpy arrays).
            batch_size (int): Number of images to process at once (for CNN/HOG only).
            number_of_times_to_upsample (int, optional): Override the default upsampling value (for cnn/hog only).
        
        Returns:
            list: A list of lists, where each inner list contains face locations for that image.
        """
        # InsightFace processes images individually but is fast enough
        if self.model in self.INSIGHTFACE_MODELS:
            return [self._detect_faces_insightface(img) for img in images]
        
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


# ===== MODULAR ARCHITECTURE: PLUGGABLE COMPONENTS =====

# --- Matching Methods (Pluggable) ---

class MatchingMethod(ABC):
    """Abstract base class for similarity/distance matching methods."""
    
    @abstractmethod
    def compare(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compare two embeddings and return a similarity/distance score.
        
        Returns:
            float: A score where higher = more similar (for similarity metrics)
                   or lower = more similar (for distance metrics).
                   Should be normalized to roughly 0-1 range for consistency.
        """
        pass


class CosineSimilarityMatching(MatchingMethod):
    """Cosine similarity matching (for normalized embeddings)."""
    
    def compare(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Returns cosine similarity in range [-1, 1], scaled to [0, 1]."""
        norm1 = embedding1 / (np.linalg.norm(embedding1) + 1e-8)
        norm2 = embedding2 / (np.linalg.norm(embedding2) + 1e-8)
        # Scale from [-1, 1] to [0, 1] so higher is better
        return 0.5 + 0.5 * np.dot(norm1, norm2)


class EuclideanDistanceMatching(MatchingMethod):
    """Euclidean distance matching (lower is better, inverted to match interface)."""
    
    def compare(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Returns inverted normalized euclidean distance so higher = more similar."""
        distance = np.linalg.norm(embedding1 - embedding2)
        # Normalize to roughly 0-1 range (most faces have distance < 1.5)
        return np.exp(-distance)  # Gaussian falloff, closer faces score higher


class L2DistanceMatching(MatchingMethod):
    """L2 (Euclidean) distance with thresholding."""
    
    def compare(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Returns 1 - normalized L2 distance."""
        distance = np.sqrt(np.sum((embedding1 - embedding2) ** 2))
        # Most faces have L2 distance < 2.0, normalize accordingly
        return max(0, 1.0 - (distance / 2.0))


# --- Embedding Extractors (Pluggable) ---

class EmbeddingExtractor(ABC):
    """Abstract base class for face embedding extraction."""
    
    @abstractmethod
    def extract_embedding(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """
        Extract embedding from a face crop.
        
        Args:
            image: Image in BGR format (OpenCV)
            bbox: Bounding box as (top, right, bottom, left)
        
        Returns:
            Embedding vector or None if extraction failed
        """
        pass
    
    @abstractmethod
    def extract_embeddings_batch(self, image: np.ndarray, bboxes: List[Tuple[int, int, int, int]]) -> List[Optional[np.ndarray]]:
        """Extract embeddings for multiple faces in one image."""
        pass


class InsightFaceEmbedder(EmbeddingExtractor):
    """InsightFace embedding extractor (512-dim embeddings)."""
    
    def __init__(self, model_name: str = "buffalo_l"):
        if not INSIGHTFACE_AVAILABLE:
            raise ImportError("InsightFace not available")
        
        logger.info(f"Initializing InsightFaceEmbedder({model_name})")
        self.model_name = model_name
        
        available = ort.get_available_providers()
        prefer_cuda = "CUDAExecutionProvider" in available

        # Try requested model first (GPU if available), then CPU, finally fall back to buffalo_l CPU if download fails.
        attempts = []
        if prefer_cuda:
            attempts.append((model_name, ["CUDAExecutionProvider", "CPUExecutionProvider"], 0))
        attempts.append((model_name, ["CPUExecutionProvider"], -1))
        if model_name != "buffalo_l":
            attempts.append(("buffalo_l", ["CPUExecutionProvider"], -1))
        
        last_error = None
        for attempt_model, providers, ctx_id in attempts:
            try:
                self.app = FaceAnalysis(name=attempt_model, allowed_modules=["recognition"], providers=providers)
                self.app.prepare(ctx_id=ctx_id, det_size=(640, 640))
                self.model_name = attempt_model
                logger.info(
                    f"InsightFaceEmbedder initialized (model={attempt_model}, providers={providers}, ctx_id={ctx_id})"
                )
                break
            except Exception as e:
                last_error = e
                logger.warning(
                    f"InsightFaceEmbedder init failed for model={attempt_model}, providers={providers}, ctx_id={ctx_id}: {e}"
                )
                continue
        else:
            # Only reached if every attempt failed.
            raise last_error or RuntimeError("InsightFaceEmbedder initialization failed")
    
    def extract_embedding(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        try:
            top, right, bottom, left = bbox
            face_crop = image[top:bottom, left:right]
            if face_crop.size == 0:
                return None
            embedding = self.app.rec_model.get_feat(face_crop)
            return embedding
        except Exception as e:
            logger.debug(f"Error extracting InsightFace embedding: {e}")
            return None
    
    def extract_embeddings_batch(self, image: np.ndarray, bboxes: List[Tuple[int, int, int, int]]) -> List[Optional[np.ndarray]]:
        return [self.extract_embedding(image, bbox) for bbox in bboxes]

    def extract_embedding_aligned(self, image: np.ndarray, face_info: Dict[str, Any], image_size: int = 112) -> Optional[np.ndarray]:
        """
        Extract embedding using 5-point landmark alignment if available.

        face_info should contain keys: 'bbox', 'kps', 'det_score'.
        """
        try:
            kps = face_info.get("kps")
            bbox = face_info.get("bbox")
            if kps is None:
                # Fallback to bbox-only crop
                if bbox is None:
                    return None
                top, right, bottom, left = bbox
                face_crop = image[top:bottom, left:right]
                if face_crop.size == 0:
                    return None
                return self.app.rec_model.get_feat(face_crop)

            # Ensure BGR image for InsightFace
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            else:
                image_bgr = image

            # Landmark-based alignment
            try:
                from insightface.utils.face_align import norm_crop
            except Exception as ie:
                logger.debug(f"Alignment import failed, using bbox crop: {ie}")
                if bbox is None:
                    return None
                top, right, bottom, left = bbox
                face_crop = image[top:bottom, left:right]
                if face_crop.size == 0:
                    return None
                return self.app.rec_model.get_feat(face_crop)

            aligned = norm_crop(image_bgr, kps.astype(np.float32), image_size=image_size, mode='arcface')
            if aligned is None or aligned.size == 0:
                return None
            return self.app.rec_model.get_feat(aligned)
        except Exception as e:
            logger.debug(f"Error extracting aligned InsightFace embedding: {e}")
            return None


class FaceRecognitionEmbedder(EmbeddingExtractor):
    """face_recognition library embedder (128-dim dlib embeddings)."""
    
    def extract_embedding(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        try:
            # face_recognition expects RGB, convert from BGR
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # Extract encoding for this specific bbox
            top, right, bottom, left = bbox
            encodings = face_recognition.face_encodings(image_rgb, [(top, right, bottom, left)])
            if encodings:
                return encodings[0]
            return None
        except Exception as e:
            logger.debug(f"Error extracting face_recognition embedding: {e}")
            return None
    
    def extract_embeddings_batch(self, image: np.ndarray, bboxes: List[Tuple[int, int, int, int]]) -> List[Optional[np.ndarray]]:
        try:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            encodings = face_recognition.face_encodings(image_rgb, bboxes)
            # Pad with None if face_recognition returns fewer encodings than bboxes
            while len(encodings) < len(bboxes):
                encodings.append(None)
            return encodings[:len(bboxes)]
        except Exception as e:
            logger.debug(f"Error extracting face_recognition embeddings batch: {e}")
            return [None] * len(bboxes)


class ViTEmbedder(EmbeddingExtractor):
    """Vision Transformer embedder (768-dim CLIP embeddings)."""
    
    def __init__(self):
        if not VIT_LIBRARIES_AVAILABLE:
            raise ImportError("Vision Transformer libraries not available")
        
        logger.info("Initializing ViT embedder (CLIP ViT-B/32)")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model.eval()
    
    def extract_embedding(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        try:
            top, right, bottom, left = bbox
            face_crop = image[top:bottom, left:right]
            if face_crop.size == 0:
                return None
            
            # Convert BGR to RGB for PIL
            face_crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            from PIL import Image
            face_pil = Image.fromarray(face_crop_rgb)
            
            with torch.no_grad():
                inputs = self.processor(images=face_pil, return_tensors="pt").to(self.device)
                embedding = self.model.get_image_features(**inputs)
            
            return embedding.cpu().numpy().flatten()
        except Exception as e:
            logger.debug(f"Error extracting ViT embedding: {e}")
            return None
    
    def extract_embeddings_batch(self, image: np.ndarray, bboxes: List[Tuple[int, int, int, int]]) -> List[Optional[np.ndarray]]:
        return [self.extract_embedding(image, bbox) for bbox in bboxes]


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

# --- Unified Classifier (Pluggable Detection + Embedding + Matching) ---

class UnifiedClassifier(Classifier):
    """
    A unified, highly modular classifier that accepts pluggable components for:
    - Face detection (detection_model: buffalo_l, buffalo_m, buffalo_s, cnn, hog, etc.)
    - Face embedding (embedding_model: insightface, face_recognition, vit, etc.)
    - Matching method (matching_method: cosine_similarity, euclidean_distance, l2_distance)
    
    This allows ANY combination of detection + embedding + matching to be tested.
    
    Example combinations:
    - buffalo_l detection + insightface embedding + cosine_similarity
    - buffalo_s detection + face_recognition embedding + euclidean_distance  
    - cnn detection + vit embedding + l2_distance
    """
    
    def __init__(
        self,
        name: str,
        celebrity_data: List[Dict[str, str]],
        detection_model: str = "buffalo_l",
        embedding_model: str = "insightface",
        matching_method: str = "cosine_similarity",
        threshold: float = 0.6,
        detection_upsample: int = 1,
        enable_multi_pass: bool = False,
        use_alignment: bool = False,
    ):
        """
        Initialize the UnifiedClassifier with pluggable components.
        
        Args:
            name: Classifier name
            celebrity_data: List of {name, reference_image_path} dicts
            detection_model: "buffalo_l" (default), "buffalo_m", "buffalo_s", "cnn", "hog"
            embedding_model: "insightface" (default), "face_recognition", "vit"
            matching_method: "cosine_similarity" (default), "euclidean_distance", "l2_distance"
            threshold: Similarity threshold for match (0-1)
            detection_upsample: Upsampling for cnn/hog only (1-3)
            enable_multi_pass: Multi-pass detection for cnn/hog only
        """
        super().__init__(name)
        
        logger.info(
            f"Initializing UnifiedClassifier: {name}\n"
            f"  Detection: {detection_model}\n"
            f"  Embedding: {embedding_model}\n"
            f"  Matching: {matching_method}\n"
            f"  Threshold: {threshold}"
        )
        
        self.detection_model = detection_model
        self.embedding_model = embedding_model
        self.matching_method_name = matching_method
        self.threshold = threshold
        self.use_alignment = use_alignment
        
        # Initialize detector
        self.face_detector = FaceDetector(
            model=detection_model,
            upsample=detection_upsample,
            enable_multi_pass=enable_multi_pass
        )
        
        # Initialize embedding extractor
        self.embedder = self._create_embedder(embedding_model)
        
        # Initialize matching method
        self.matcher = self._create_matcher(matching_method)
        
        # Load reference embeddings
        self.reference_embeddings: List[np.ndarray] = []
        self.reference_names: List[str] = []
        self._load_reference_embeddings(celebrity_data)
        
        if not self.reference_embeddings:
            logger.warning(f"No reference embeddings loaded for {name}")
        else:
            logger.info(
                f"Successfully initialized {name} with "
                f"{len(self.reference_embeddings)} reference embeddings"
            )
    
    def _create_embedder(self, embedding_model: str) -> EmbeddingExtractor:
        """Create the appropriate embedding extractor."""
        if embedding_model == "insightface":
            return InsightFaceEmbedder("buffalo_l")
        elif embedding_model == "insightface_buffalo_l":
            return InsightFaceEmbedder("buffalo_l")
        elif embedding_model == "insightface_buffalo_m":
            return InsightFaceEmbedder("buffalo_m")
        elif embedding_model == "insightface_buffalo_s":
            return InsightFaceEmbedder("buffalo_s")
        elif embedding_model == "insightface_antelopev2":
            return InsightFaceEmbedder("antelopev2")
        elif embedding_model == "face_recognition":
            return FaceRecognitionEmbedder()
        elif embedding_model == "vit":
            return ViTEmbedder()
        else:
            logger.warning(f"Unknown embedding model {embedding_model}, defaulting to face_recognition")
            return FaceRecognitionEmbedder()
    
    def _create_matcher(self, matching_method: str) -> MatchingMethod:
        """Create the appropriate matching method."""
        if matching_method == "cosine_similarity":
            return CosineSimilarityMatching()
        elif matching_method == "euclidean_distance":
            return EuclideanDistanceMatching()
        elif matching_method == "l2_distance":
            return L2DistanceMatching()
        else:
            logger.warning(f"Unknown matching method {matching_method}, defaulting to cosine_similarity")
            return CosineSimilarityMatching()
    
    def _load_reference_embeddings(self, celebrity_data: List[Dict[str, str]]):
        """Load embeddings from reference images."""
        for celeb in celebrity_data:
            try:
                img = cv2.imread(celeb["reference_image_path"])
                if img is None:
                    logger.warning(f"Could not load: {celeb['reference_image_path']}")
                    continue
                
                # Detect face in reference image
                bboxes = self.face_detector.detect_faces(img)
                if not bboxes:
                    logger.warning(f"No face detected in {celeb['reference_image_path']}")
                    continue
                
                # Extract embedding
                embedding = self.embedder.extract_embedding(img, bboxes[0])
                if embedding is not None:
                    self.reference_embeddings.append(embedding)
                    self.reference_names.append(celeb["name"])
                    logger.debug(f"Loaded embedding for {celeb['name']}")
                else:
                    logger.warning(f"Could not extract embedding for {celeb['name']}")
            except Exception as e:
                logger.warning(f"Error loading {celeb['name']}: {e}")
    
    def classify_images(self, image_paths: List[str], image_batch_size: Optional[int] = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Classify images using detection + embedding + matching pipeline.
        
        Args:
            image_paths: List of image file paths
            image_batch_size: Batch size for processing
        
        Returns:
            Dict mapping image paths to lists of detected celebrities
        """
        logger.info(f"Classifying {len(image_paths)} images with {self.name}")
        
        if not self.reference_embeddings:
            logger.warning("No reference embeddings available")
            return {path: [] for path in image_paths}
        
        if image_batch_size is None:
            image_batch_size = int(os.getenv("UNIFIED_BATCH", "4"))
        
        output = {}
        reference_embeddings_array = np.array(self.reference_embeddings)
        
        prog = tqdm(total=len(image_paths), desc=f"Classifying ({self.name})", unit="img")
        
        for image_path in image_paths:
            try:
                img = cv2.imread(image_path)
                if img is None:
                    logger.error(f"Could not load: {image_path}")
                    output[image_path] = []
                    prog.update(1)
                    continue
                
                # Phase 1: Detect faces
                face_bboxes = self.face_detector.detect_faces(img)
                if not face_bboxes:
                    output[image_path] = []
                    prog.update(1)
                    continue
                
                logger.debug(f"Found {len(face_bboxes)} face(s)")
                
                # Phase 2: Extract embeddings (optionally with alignment if available)
                if self.use_alignment and isinstance(self.embedder, InsightFaceEmbedder) and self.face_detector.model in self.face_detector.INSIGHTFACE_MODELS:
                    face_infos = self.face_detector.detect_faces_with_landmarks(img)
                    embeddings = []
                    for fi in face_infos:
                        emb = self.embedder.extract_embedding_aligned(img, fi)
                        embeddings.append(emb)
                    # use aligned bbox list to keep locations in sync
                    face_bboxes = [fi.get("bbox") for fi in face_infos]
                else:
                    embeddings = self.embedder.extract_embeddings_batch(img, face_bboxes)
                
                all_detections = []
                for face_idx, (bbox, embedding) in enumerate(zip(face_bboxes, embeddings)):
                    if embedding is None:
                        logger.debug(f"Could not extract embedding for face {face_idx}")
                        continue
                    
                    # Phase 3: Match against references
                    scores = []
                    for ref_embedding in reference_embeddings_array:
                        score = self.matcher.compare(embedding, ref_embedding)
                        scores.append(score)
                    
                    scores = np.array(scores)
                    best_idx = np.argmax(scores)
                    best_score = scores[best_idx]
                    
                    if best_score >= self.threshold:
                        name = self.reference_names[best_idx]
                    else:
                        name = "Unknown"
                    
                    all_detections.append({
                        "name": name,
                        "location": bbox,
                        "confidence": float(best_score),
                    })
                
                output[image_path] = all_detections
                prog.update(1)
                
            except Exception as e:
                logger.error(f"Error processing {image_path}: {e}", exc_info=True)
                output[image_path] = []
                prog.update(1)
        
        prog.close()
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

def get_classifier(
    classifier_type: str, 
    celebrity_data: List[Dict[str, str]], 
    detection_model: Optional[str] = None,
    embedding_model: Optional[str] = None,
    matching_method: Optional[str] = None,
    threshold: Optional[float] = None,
    detection_upsample: Optional[int] = None,
    enable_multi_pass: Optional[bool] = None,
):
    """
    Factory function to create a classifier with pluggable detection, embedding, and matching.
    
    USAGE PATTERNS:
    ---------------
    1. Simple presets (backward compatible):
       get_classifier("face_recognition_cnn", celebrity_data)
       get_classifier("insightface", celebrity_data)
       get_classifier("vit_b32", celebrity_data)
    
    2. Custom combinations (NEW - fully modular):
       get_classifier(
           "unified",
           celebrity_data,
           detection_model="buffalo_l",
           embedding_model="insightface",
           matching_method="cosine_similarity"
       )
       
       get_classifier(
           "unified",
           celebrity_data,
           detection_model="hog",
           embedding_model="face_recognition",
           matching_method="euclidean_distance"
       )
    
    Args:
        classifier_type: "unified", "face_recognition_cnn", "face_recognition_hog", 
                        "vit_b32", or "insightface"
        celebrity_data: List of {name, reference_image_path} dicts
        detection_model: "buffalo_l", "buffalo_m", "buffalo_s", "cnn", "hog" (optional)
        embedding_model: "insightface", "face_recognition", "vit" (optional)
        matching_method: "cosine_similarity", "euclidean_distance", "l2_distance" (optional)
        threshold: Similarity threshold 0-1 (optional, defaults vary by type)
        detection_upsample: Upsampling for cnn/hog detection (optional)
        enable_multi_pass: Multi-pass detection for cnn/hog (optional)
    
    Returns:
        Classifier instance
    """
    logger.info(f"Getting classifier: {classifier_type}")
    
    # NEW: Unified classifier with full modularity
    if classifier_type == "unified":
        return UnifiedClassifier(
            name=f"unified_{detection_model or 'default'}_{embedding_model or 'default'}",
            celebrity_data=celebrity_data,
            detection_model=detection_model or "buffalo_l",
            embedding_model=embedding_model or "insightface",
            matching_method=matching_method or "cosine_similarity",
            threshold=threshold or 0.6,
            detection_upsample=detection_upsample or 1,
            enable_multi_pass=enable_multi_pass or False,
        )
    
    # BACKWARD COMPATIBLE: Legacy classifier types
    elif classifier_type == "face_recognition_cnn":
        return UnifiedClassifier(
            name="face_recognition_cnn",
            celebrity_data=celebrity_data,
            detection_model=detection_model or "cnn",
            embedding_model="face_recognition",
            matching_method="euclidean_distance",
            threshold=threshold or 0.6,
            detection_upsample=detection_upsample or 2,
            enable_multi_pass=enable_multi_pass or True,
        )
    
    elif classifier_type == "face_recognition_hog":
        return UnifiedClassifier(
            name="face_recognition_hog",
            celebrity_data=celebrity_data,
            detection_model=detection_model or "hog",
            embedding_model="face_recognition",
            matching_method="euclidean_distance",
            threshold=threshold or 0.6,
            detection_upsample=detection_upsample or 1,
            enable_multi_pass=enable_multi_pass or False,
        )
    
    elif classifier_type == "vit_b32":
        if not VIT_LIBRARIES_AVAILABLE:
            logger.warning("ViT libraries not available, falling back to face_recognition_cnn")
            return get_classifier("face_recognition_cnn", celebrity_data)
        return UnifiedClassifier(
            name="vit_b32",
            celebrity_data=celebrity_data,
            detection_model=detection_model or "buffalo_l",
            embedding_model="vit",
            matching_method="cosine_similarity",
            threshold=threshold or 0.8,
            detection_upsample=detection_upsample or 1,
            enable_multi_pass=enable_multi_pass or False,
            use_alignment=False,
        )
    
    elif classifier_type.startswith("insightface"):
        if not INSIGHTFACE_AVAILABLE:
            logger.warning("InsightFace not available, falling back to face_recognition_cnn")
            return get_classifier("face_recognition_cnn", celebrity_data)
        
        # Parse insightface variants: "insightface", "insightface_buffalo_m", etc.
        model_variant = "buffalo_l"  # default
        if "_" in classifier_type:
            model_variant = classifier_type.split("_", 1)[1]
        
        return UnifiedClassifier(
            name=f"insightface_{model_variant}",
            celebrity_data=celebrity_data,
            detection_model=detection_model or model_variant,
            embedding_model=f"insightface_{model_variant}",
            matching_method="cosine_similarity",
            threshold=threshold or 0.6,
            detection_upsample=detection_upsample or 1,
            enable_multi_pass=enable_multi_pass or False,
            use_alignment=True,
        )
    
    else:
        logger.error(f"Unknown classifier type: {classifier_type}")
        raise ValueError(f"Unknown classifier type: {classifier_type}")

# --- Example Usage ---
if __name__ == '__main__':
    CELEBRITIES_JSON = "celebrities.json"
    TEST_IMAGES = [
        "Images/3899/062770.jpg",
        "Images/not_3899/017031.jpg"
    ]

    celebrity_data = load_celebrities_from_json(CELEBRITIES_JSON)
    if not celebrity_data:
        logger.error("No celebrity data loaded.")
    else:
        # Example 1: Legacy classifier types (backward compatible)
        logger.info("=== BACKWARD COMPATIBLE USAGE ===")
        cnn_clf = get_classifier("face_recognition_cnn", celebrity_data)
        insightface_clf = get_classifier("insightface", celebrity_data)
        
        # Example 2: Custom modular combinations (NEW)
        logger.info("\n=== NEW MODULAR COMBINATIONS ===")
        
        # Fast detection + accurate embeddings
        fast_accurate = get_classifier(
            "unified",
            celebrity_data,
            detection_model="buffalo_s",  # lightweight detection
            embedding_model="insightface_buffalo_l",  # accurate embeddings
            matching_method="cosine_similarity"
        )
        logger.info(f"Created: {fast_accurate.name}")
        
        # Legacy detection + modern embeddings
        legacy_modern = get_classifier(
            "unified",
            celebrity_data,
            detection_model="hog",  # CPU-friendly legacy
            embedding_model="insightface",  # modern 512-dim
            matching_method="cosine_similarity"
        )
        logger.info(f"Created: {legacy_modern.name}")
        
        # Vision Transformer with HOG detection
        vit_hog = get_classifier(
            "unified",
            celebrity_data,
            detection_model="hog",
            embedding_model="vit",
            matching_method="l2_distance"
        )
        logger.info(f"Created: {vit_hog.name}")
        
        logger.info("\n✓ Modular classification architecture ready!")