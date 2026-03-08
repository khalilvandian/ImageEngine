"""
InsightFace-only face recognition engine.

Uses InsightFace for both detection and recognition (embedding extraction),
following the approach from Experiment_insightface_only.ipynb.

Supports:
- Reference database management (load, augment, persist via pickle cache)
- Face classification with cosine similarity scores
- Two-threshold system for human-in-the-loop annotation workflows:
    * similarity >= upper_threshold  → Confident identification (auto-confirmed)
    * lower_threshold <= sim < upper → Augment reference DB under matched label
    * similarity < lower_threshold   → Queue for human annotation
"""

import json
import pickle
import datetime
import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from tqdm import tqdm

from insightface.app import FaceAnalysis
import onnxruntime as ort

from src.logging_utils import setup_logger

logger = setup_logger()

BASE_DIR = Path("/app")


class InsightFaceEngine:
    """
    InsightFace-based face recognition engine.

    Provides detection + recognition using a single InsightFace model
    (e.g. buffalo_l). Manages a reference database of known identities
    with support for augmentation and persistence.
    """

    def __init__(
        self,
        model_name: str = "buffalo_l",
        det_size: Tuple[int, int] = (640, 640),
        cache_dir: str = "data/cache",
    ):
        """
        Args:
            model_name: InsightFace model zoo name
                        ('buffalo_l', 'buffalo_m', 'buffalo_s', 'antelopev2')
            det_size:   Detection input size (width, height)
            cache_dir:  Directory for caching reference embeddings
        """
        self.model_name = model_name
        self.det_size = det_size
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize InsightFace
        self.app = self._init_insightface()

        # Reference database: {identity_name: [{"embedding": ndarray, "source": str}]}
        self.reference_db: Dict[str, List[Dict[str, Any]]] = {}

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------

    def _init_insightface(self) -> FaceAnalysis:
        """Initialize InsightFace with GPU → CPU fallback."""
        available = ort.get_available_providers()
        prefer_cuda = "CUDAExecutionProvider" in available
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if prefer_cuda
            else ["CPUExecutionProvider"]
        )
        ctx_id = 0 if prefer_cuda else -1

        try:
            app = FaceAnalysis(
                name=self.model_name,
                allowed_modules=["detection", "recognition"],
                providers=providers,
            )
            app.prepare(ctx_id=ctx_id, det_size=self.det_size)
            logger.info(
                f"InsightFace ({self.model_name}) initialized: "
                f"providers={providers}, det_size={self.det_size}"
            )
            return app
        except Exception as e:
            if prefer_cuda:
                logger.warning(f"GPU init failed ({e}); retrying CPU…")
                providers = ["CPUExecutionProvider"]
                ctx_id = -1
                app = FaceAnalysis(
                    name=self.model_name,
                    allowed_modules=["detection", "recognition"],
                    providers=providers,
                )
                app.prepare(ctx_id=ctx_id, det_size=self.det_size)
                logger.info("InsightFace initialised on CPU")
                return app
            raise

    # ------------------------------------------------------------------
    # Reference database management
    # ------------------------------------------------------------------

    def load_references(
        self,
        references_path: str = "data/references.json",
        force_extract: bool = False,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Load reference database from JSON and extract embeddings.
        Uses a pickle‐cached version on disk when available.

        Args:
            references_path: Path to the references JSON file.
            force_extract:   If True, ignore cache and re‐extract.

        Returns:
            The reference database dict.
        """
        ref_path = Path(references_path)
        if not ref_path.is_absolute():
            ref_path = BASE_DIR / ref_path

        with open(ref_path, "r", encoding="utf-8") as f:
            references = json.load(f)

        cache_path = self.cache_dir / f"reference_db_{self.model_name}.pkl"

        # --- try cache ---
        if cache_path.exists() and not force_extract:
            try:
                with open(cache_path, "rb") as f:
                    cached_db = pickle.load(f)
                # Verify every name in the JSON has cached embeddings
                all_cached = all(
                    ref.get("name") in cached_db and cached_db[ref["name"]]
                    for ref in references
                    if ref.get("name")
                )
                if all_cached:
                    self.reference_db = cached_db
                    total = sum(len(v) for v in self.reference_db.values())
                    logger.info(
                        f"Loaded cached reference DB "
                        f"({len(self.reference_db)} identities, {total} embeddings)"
                    )
                    return self.reference_db
                else:
                    logger.info("Cache incomplete – re‐extracting…")
            except Exception as e:
                logger.warning(f"Cache load failed ({e}), re‐extracting…")

        # --- extract embeddings from images ---
        self.reference_db = {}

        for ref in references:
            name = ref.get("name")
            if not name:
                continue
            self.reference_db[name] = []

            for img_ref in ref.get("reference_images", []):
                img_path_str = img_ref.get("image_path", "")
                img_path = (
                    BASE_DIR / img_path_str
                    if not img_path_str.startswith("/")
                    else Path(img_path_str)
                )

                if not img_path.exists():
                    logger.warning(f"Reference image not found: {img_path}")
                    continue

                img = cv2.imread(str(img_path))
                if img is None:
                    logger.warning(f"Failed to load: {img_path}")
                    continue

                faces = self.app.get(img)
                if not faces:
                    logger.warning(f"No face detected in reference: {img_path}")
                    continue

                for face in faces:
                    self.reference_db[name].append(
                        {
                            "embedding": face.normed_embedding,
                            "source": str(img_path),
                        }
                    )
                logger.info(
                    f"Extracted {len(faces)} face(s) from {img_path_str} ({name})"
                )

        # --- persist cache ---
        self._save_cache(cache_path)

        total = sum(len(v) for v in self.reference_db.values())
        logger.info(
            f"Reference DB ready: {len(self.reference_db)} identities, "
            f"{total} embeddings"
        )
        return self.reference_db

    def _save_cache(self, cache_path: Optional[Path] = None):
        if cache_path is None:
            cache_path = self.cache_dir / f"reference_db_{self.model_name}.pkl"
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(self.reference_db, f)
            logger.info(f"Saved reference DB cache → {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to cache reference DB: {e}")

    def save_reference_db(self):
        """Persist the current (possibly augmented) reference DB."""
        self._save_cache()

    def save_reference_db_versioned(
        self,
        upper_threshold: float = 0.50,
        lower_threshold: float = 0.25,
        testset_name: Optional[str] = None,
    ):
        """
        Save the reference DB with a versioned filename and also update the
        standard cache.  Filename format:

            reference_db_{model}_{testset}_ut{upper}_lt{lower}_{datetime}.pkl

        Parameters
        ----------
        upper_threshold : float
            The upper (identification) threshold used during this session.
        lower_threshold : float
            The lower (augmentation floor) threshold used during this session.
        testset_name : str, optional
            Name of the test set (just the filename, no path).  When *None*
            only classify-mode data was used – the tag becomes ``classify``.
        """
        from datetime import datetime as _dt

        # Always persist the standard (latest) cache
        self._save_cache()

        # Build versioned filename
        ts_tag = "classify"
        if testset_name:
            ts_tag = Path(testset_name).stem  # strip .json

        ut_str = f"{upper_threshold:.2f}".replace(".", "")
        lt_str = f"{lower_threshold:.2f}".replace(".", "")
        dt_str = _dt.now().strftime("%Y%m%d_%H%M%S")

        versioned_name = (
            f"reference_db_{self.model_name}_{ts_tag}"
            f"_ut{ut_str}_lt{lt_str}_{dt_str}.pkl"
        )
        versioned_path = self.cache_dir / versioned_name
        self._save_cache(cache_path=versioned_path)
        logger.info(f"Versioned reference DB saved → {versioned_path}")

    def get_reference_summary(self) -> Dict[str, int]:
        """Return {identity_name: num_embeddings} for the current DB."""
        return {
            name: len(embs)
            for name, embs in self.reference_db.items()
        }

    def list_versioned_dbs(self) -> List[str]:
        """Return a sorted list of versioned reference DB filenames in the cache dir."""
        import glob

        pattern = str(self.cache_dir / "reference_db_*.pkl")
        paths = sorted(glob.glob(pattern), reverse=True)  # newest first
        return paths

    def load_reference_db_from_file(self, pkl_path: str):
        """
        Load a specific reference DB pickle file, replacing the current DB
        in memory.  Does *not* overwrite the default cache.
        """
        with open(pkl_path, "rb") as f:
            self.reference_db = pickle.load(f)
        total = sum(len(v) for v in self.reference_db.values())
        logger.info(
            f"Loaded reference DB from {pkl_path} "
            f"({len(self.reference_db)} identities, {total} embeddings)"
        )

    def reset_to_base_references(
        self, references_path: str = "data/references.json"
    ):
        """
        Re-extract embeddings from the original references JSON, ignoring
        any cached / augmented DB.  This gives a clean baseline with only
        the original identities (e.g. 4 celebrities).
        """
        self.load_references(references_path, force_extract=True)
        logger.info("Reference DB reset to base (original references).")

    # ------------------------------------------------------------------
    # Face identification
    # ------------------------------------------------------------------

    def identify_face(
        self, embedding: np.ndarray
    ) -> Tuple[Optional[str], float]:
        """
        Identify a face by **average** cosine similarity against the
        reference DB.

        For each identity, the similarity is computed against every
        stored embedding and then averaged.  The identity with the
        highest average similarity is returned.

        Returns:
            (best_match_name, best_avg_similarity).
            If the DB is empty returns (None, -1.0).
        """
        best_match: Optional[str] = None
        best_similarity = -1.0

        for identity, emb_list in self.reference_db.items():
            if not emb_list:
                continue
            total_sim = sum(
                float(np.dot(embedding, emb_obj["embedding"]))
                for emb_obj in emb_list
            )
            avg_sim = total_sim / len(emb_list)
            if avg_sim > best_similarity:
                best_similarity = avg_sim
                best_match = identity

        return best_match, best_similarity

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def classify_image(self, image_path: str, store_images: bool = True) -> List[Dict[str, Any]]:
        """
        Detect and classify all faces in a single image.

        Args:
            image_path:    Path to the image file.
            store_images:  If True, include face_crop_rgb and main_image_rgb
                           in the result dicts (needed for UI display but
                           heavy on memory for large batches).

        Returns a list of dicts, one per detected face:
            name          – matched identity (or None)
            similarity    – cosine similarity to best match
            bbox          – (top, right, bottom, left)
            bbox_xyxy     – (x1, y1, x2, y2)
            embedding     – normed embedding vector
            face_crop_rgb – face crop as RGB numpy array (only if store_images)
            main_image_rgb – full source image as RGB numpy array (only if store_images)
            image_path    – source image path
        """
        logger.debug(f"[classify_image] Loading image: {image_path}")
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"Could not load image: {image_path}")
            return []

        h, w = img.shape[:2]
        logger.debug(f"[classify_image] Image loaded: {w}x{h} pixels")

        faces = self.app.get(img)
        logger.debug(f"[classify_image] Detected {len(faces)} face(s) in {Path(image_path).name}")
        results: List[Dict[str, Any]] = []

        # Convert the full image to RGB once for display
        main_image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if store_images else None

        for i, face in enumerate(faces):
            embedding = face.normed_embedding
            best_match, best_sim = self.identify_face(embedding)
            logger.debug(
                f"[classify_image]   Face #{i}: best_match='{best_match}', "
                f"similarity={best_sim:.6f}, det_score={float(face.det_score):.4f}"
            )

            x1, y1, x2, y2 = [int(v) for v in face.bbox]
            # Clamp to image boundaries
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            logger.debug(f"[classify_image]   Face #{i}: bbox=({x1},{y1},{x2},{y2}), crop_size={x2-x1}x{y2-y1}")

            entry: Dict[str, Any] = {
                "name": best_match,
                "similarity": best_sim,
                "bbox": (y1, x2, y2, x1),  # (top, right, bottom, left)
                "bbox_xyxy": (x1, y1, x2, y2),
                "embedding": embedding,
                "image_path": image_path,
            }

            if store_images:
                face_crop = img[y1:y2, x1:x2]
                entry["face_crop_rgb"] = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                entry["main_image_rgb"] = main_image_rgb

            results.append(entry)

        return results

    def classify_image_from_array(self, img: np.ndarray, source_label: str = "upload", store_images: bool = True) -> List[Dict[str, Any]]:
        """
        Classify faces in an already-loaded image (BGR numpy array).
        """
        faces = self.app.get(img)
        results: List[Dict[str, Any]] = []

        main_image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if store_images else None

        for face in faces:
            embedding = face.normed_embedding
            best_match, best_sim = self.identify_face(embedding)

            x1, y1, x2, y2 = [int(v) for v in face.bbox]
            h, w = img.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            entry: Dict[str, Any] = {
                "name": best_match,
                "similarity": best_sim,
                "bbox": (y1, x2, y2, x1),
                "bbox_xyxy": (x1, y1, x2, y2),
                "embedding": embedding,
                "image_path": source_label,
            }

            if store_images:
                face_crop = img[y1:y2, x1:x2]
                entry["face_crop_rgb"] = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                entry["main_image_rgb"] = main_image_rgb

            results.append(entry)

        return results

    def classify_images(
        self, image_paths: List[str], show_progress: bool = True
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Classify faces in multiple images. Returns {path: [results]}."""
        output: Dict[str, List[Dict[str, Any]]] = {}
        iterator = tqdm(image_paths, desc="Classifying", unit="img") if show_progress else image_paths
        for path in iterator:
            output[path] = self.classify_image(path)
        return output

    # ------------------------------------------------------------------
    # Two-threshold classification for human-in-the-loop workflows
    # ------------------------------------------------------------------

    def classify_with_thresholds(
        self,
        image_paths: List[str],
        upper_threshold: float = 0.65,
        lower_threshold: float = 0.30,
        review_top_k: int = 10,
        show_progress: bool = True,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Classify images and split results into three categories:

        1. **identified** (similarity ≥ upper_threshold):
           Confident match – automatically registered as identified people.

        2. **augment** (lower_threshold ≤ similarity < upper_threshold):
           Moderate match – added to the reference DB under the matched label
           to improve future recognition.

        3. **review** (similarity < lower_threshold):
           Low confidence – queued for human annotation.  Only the
           *review_top_k* faces closest to the lower threshold are kept
           so the review queue stays manageable.

        Memory-efficient: processes images one at a time and only stores
        images (face_crop_rgb / main_image_rgb) for augment & review faces.
        Identified faces keep image_path for reference but drop the heavy
        pixel data.

        Returns a dict with keys "identified", "augment", "review",
        each mapping to a list of face result dicts.
        """
        import heapq
        import time as _time

        logger.info(
            f"[classify_with_thresholds] Starting two-threshold classification: "
            f"{len(image_paths)} image(s), upper={upper_threshold}, lower={lower_threshold}, "
            f"review_top_k={review_top_k}"
        )
        _cwt_start = _time.time()

        categorised: Dict[str, List[Dict[str, Any]]] = {
            "identified": [],
            "augment": [],
            "review": [],
        }
        _total_faces = 0
        _discarded_review = 0

        # We use a min-heap of size review_top_k so we can efficiently
        # keep only the top-K review faces (highest similarity) without
        # accumulating all of them in memory.
        review_heap: list = []  # (similarity, counter, face_dict)
        _counter = 0  # tie-breaker so heapq never compares dicts

        iterator = (
            tqdm(image_paths, desc="Classifying", unit="img")
            if show_progress
            else image_paths
        )

        for path in iterator:
            _img_start = _time.time()
            # First pass: classify without storing images (lightweight)
            faces = self.classify_image(path, store_images=False)
            _img_elapsed = _time.time() - _img_start
            _total_faces += len(faces)

            logger.info(
                f"[classify_with_thresholds] {Path(path).name}: "
                f"{len(faces)} face(s) detected in {_img_elapsed:.3f}s"
            )

            for face in faces:
                sim = face["similarity"]
                match_name = face.get("name", "Unknown")

                if sim >= upper_threshold:
                    # Identified – attach images for display
                    self._attach_images(face)
                    face["category"] = "identified"
                    categorised["identified"].append(face)
                    logger.info(
                        f"[classify_with_thresholds]   -> IDENTIFIED: '{match_name}' "
                        f"(sim={sim:.6f} >= upper={upper_threshold})"
                    )

                elif sim >= lower_threshold:
                    # Augment – reload images for this face
                    self._attach_images(face)
                    face["category"] = "augment"
                    categorised["augment"].append(face)
                    logger.info(
                        f"[classify_with_thresholds]   -> AUGMENT: '{match_name}' "
                        f"(sim={sim:.6f}, {lower_threshold} <= sim < {upper_threshold})"
                    )

                else:
                    # Review candidate – keep only top-K by similarity
                    _counter += 1
                    if len(review_heap) < review_top_k:
                        self._attach_images(face)
                        face["category"] = "review"
                        heapq.heappush(review_heap, (sim, _counter, face))
                        logger.info(
                            f"[classify_with_thresholds]   -> REVIEW (queued): '{match_name}' "
                            f"(sim={sim:.6f} < lower={lower_threshold})"
                        )
                    elif sim > review_heap[0][0]:
                        # This face is better than the worst in the heap
                        self._attach_images(face)
                        face["category"] = "review"
                        _evicted_sim = review_heap[0][0]
                        heapq.heapreplace(review_heap, (sim, _counter, face))
                        logger.debug(
                            f"[classify_with_thresholds]   -> REVIEW (replaced): '{match_name}' "
                            f"(sim={sim:.6f}, evicted sim={_evicted_sim:.6f})"
                        )
                    else:
                        _discarded_review += 1
                        logger.debug(
                            f"[classify_with_thresholds]   -> DISCARDED: '{match_name}' "
                            f"(sim={sim:.6f}, below review heap min)"
                        )

        # Extract review faces from heap, sorted best-first
        categorised["review"] = [
            entry[2] for entry in sorted(review_heap, reverse=True)
        ]

        _cwt_elapsed = _time.time() - _cwt_start
        logger.info(
            f"[classify_with_thresholds] Completed in {_cwt_elapsed:.3f}s: "
            f"{_total_faces} total faces across {len(image_paths)} images — "
            f"identified={len(categorised['identified'])}, "
            f"augment={len(categorised['augment'])}, "
            f"review={len(categorised['review'])}, "
            f"discarded_review={_discarded_review}"
        )

        # Log similarity distribution per category
        for cat_name, cat_faces in categorised.items():
            if cat_faces:
                sims = [f["similarity"] for f in cat_faces]
                logger.info(
                    f"[classify_with_thresholds] {cat_name} similarity stats: "
                    f"min={min(sims):.6f}, max={max(sims):.6f}, "
                    f"mean={sum(sims)/len(sims):.6f}"
                )

        return categorised

    def _attach_images(self, face: Dict[str, Any]):
        """
        Reload the source image and attach face_crop_rgb to a face dict
        that was classified without store_images.

        NOTE: main_image_rgb is NOT stored to avoid OOM on large batches.
        Use load_main_image() to lazily load it when needed for display.
        """
        image_path = face.get("image_path", "")
        img = cv2.imread(image_path)
        if img is None:
            face["face_crop_rgb"] = None
            return

        x1, y1, x2, y2 = face["bbox_xyxy"]
        h, w = img.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        crop = img[y1:y2, x1:x2]
        face["face_crop_rgb"] = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB) if crop.size > 0 else None

    @staticmethod
    def load_main_image(face: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        Lazily load the full source image as an RGB numpy array.

        Call this only when you need to display the main image (e.g. in
        the UI for the current page).  The result is NOT cached on the
        face dict so memory is reclaimed after display.
        """
        image_path = face.get("image_path", "")
        if not image_path:
            return None
        img = cv2.imread(image_path)
        if img is None:
            return None
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # ------------------------------------------------------------------
    # Reference DB mutation helpers
    # ------------------------------------------------------------------

    def add_embedding_to_identity(
        self,
        identity_name: str,
        embedding: np.ndarray,
        source: str = "augmented",
    ):
        """Add an embedding to an existing identity."""
        if identity_name not in self.reference_db:
            self.reference_db[identity_name] = []
        self.reference_db[identity_name].append(
            {"embedding": embedding, "source": source}
        )
        logger.info(
            f"Added embedding to '{identity_name}' "
            f"(total: {len(self.reference_db[identity_name])})"
        )

    def create_new_identity(
        self,
        name: str,
        embedding: np.ndarray,
        source: str = "user_created",
    ):
        """Create a new identity (or append to existing if name exists)."""
        if name not in self.reference_db:
            self.reference_db[name] = []
        self.reference_db[name].append(
            {"embedding": embedding, "source": source}
        )
        logger.info(f"Created/augmented identity '{name}' → {len(self.reference_db[name])} embedding(s)")

    def auto_augment_references(
        self,
        augment_faces: List[Dict[str, Any]],
    ) -> int:
        """
        Automatically add the 'augment' category faces to the reference DB
        under their matched identity labels.

        Returns the number of embeddings added.
        """
        logger.info(
            f"[auto_augment] Starting auto-augmentation with {len(augment_faces)} face(s)"
        )
        _before_counts = {k: len(v) for k, v in self.reference_db.items()}
        added = 0
        for face in augment_faces:
            name = face.get("name")
            embedding = face.get("embedding")
            source = face.get("image_path", "augmented")
            if name and embedding is not None:
                self.add_embedding_to_identity(name, embedding, source=source)
                logger.info(
                    f"[auto_augment] Added embedding to '{name}' from {Path(source).name} "
                    f"(sim={face.get('similarity', 0):.6f})"
                )
                added += 1
        _after_counts = {k: len(v) for k, v in self.reference_db.items()}
        logger.info(f"[auto_augment] Augmentation complete: {added} embeddings added")
        for name in sorted(set(list(_before_counts.keys()) + list(_after_counts.keys()))):
            before = _before_counts.get(name, 0)
            after = _after_counts.get(name, 0)
            if after != before:
                logger.info(f"[auto_augment]   '{name}': {before} -> {after} embeddings (+{after - before})")
        return added

    # ------------------------------------------------------------------
    # Test‐set evaluation (mirrors the notebook approach)
    # ------------------------------------------------------------------

    def _compute_all_identity_scores(
        self, embedding: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute average cosine similarity of *embedding* against every
        identity in the reference DB.

        Returns ``{identity_name: avg_similarity}`` (empty dict when
        the DB is empty).
        """
        scores: Dict[str, float] = {}
        for identity, emb_list in self.reference_db.items():
            if not emb_list:
                scores[identity] = 0.0
                continue
            total_sim = sum(
                float(np.dot(embedding, emb_obj["embedding"]))
                for emb_obj in emb_list
            )
            scores[identity] = total_sim / len(emb_list)
        return scores

    def evaluate_testset(
        self,
        testset_path: str,
        identification_threshold: float = 0.30,
    ) -> Dict[str, Any]:
        """
        Pure evaluation: run predictions on a test set and compute metrics.
        No categorisation / augmentation logic.

        Returns dict with:
            predictions        – list of list of predicted names
            ground_truth       – list of list of true labels
            image_paths        – list of image paths
            metrics            – dict of aggregate metrics
            all_faces          – flat list of every face result dict
            per_image_scores   – list of {identity: max_score} dicts
                                 (continuous scores for ROC / PR curves)
            identity_names     – sorted list of identity names in the ref DB
        """
        import time as _time

        logger.info(
            f"[evaluate_testset] Starting evaluation: testset={testset_path}, "
            f"identification_threshold={identification_threshold}"
        )
        logger.info(
            f"[evaluate_testset] Reference DB: {len(self.reference_db)} identities, "
            f"{sum(len(v) for v in self.reference_db.values())} embeddings"
        )
        _eval_start = _time.time()

        with open(testset_path, "r", encoding="utf-8") as f:
            items = json.load(f)

        logger.info(f"[evaluate_testset] Loaded {len(items)} test items from {testset_path}")

        # Log test set label distribution
        _label_counts: Dict[str, int] = {}
        for item in items:
            for lbl in item.get("labels", []):
                _label_counts[lbl] = _label_counts.get(lbl, 0) + 1
        logger.info(f"[evaluate_testset] Test set label distribution:")
        for lbl, cnt in sorted(_label_counts.items()):
            logger.info(f"[evaluate_testset]   - {lbl}: {cnt} image(s)")

        identity_names = sorted(self.reference_db.keys())
        logger.info(f"[evaluate_testset] Identity names in reference DB: {identity_names}")

        image_paths = []
        ground_truth = []
        predictions = []
        per_image_scores: List[Dict[str, float]] = []
        all_faces: List[Dict[str, Any]] = []
        _total_faces = 0
        _correct_images = 0

        for item_idx, item in enumerate(tqdm(items, desc="Evaluating", unit="img")):
            img_path = item.get("path", "")
            labels = item.get("labels", [])

            # Resolve path
            full_path = (
                str(BASE_DIR / img_path)
                if not img_path.startswith("/")
                else img_path
            )
            image_paths.append(full_path)

            # Empty ground truth → "None" class (no known person expected)
            gt_labels = labels if labels else ["None"]
            ground_truth.append(gt_labels)

            _img_start = _time.time()
            faces = self.classify_image(full_path, store_images=False)
            _img_elapsed = _time.time() - _img_start
            _total_faces += len(faces)
            identified_set: set = set()

            logger.info(
                f"[evaluate_testset] Image {item_idx+1}/{len(items)}: {Path(full_path).name} — "
                f"{len(faces)} face(s) detected in {_img_elapsed:.3f}s, GT={gt_labels}"
            )

            # Track per-identity max scores across all faces in an image
            image_identity_scores: Dict[str, List[float]] = {
                name: [] for name in identity_names
            }

            for face_idx, face in enumerate(faces):
                # Attach ground truth for later display
                face["ground_truth"] = labels
                face["original_filename"] = Path(full_path).name

                if face["similarity"] >= identification_threshold and face["name"]:
                    face["predicted_label"] = face["name"]
                    identified_set.add(face["name"])
                    logger.info(
                        f"[evaluate_testset]   Face #{face_idx}: IDENTIFIED as '{face['name']}' "
                        f"(sim={face['similarity']:.6f} >= thresh={identification_threshold})"
                    )
                else:
                    face["predicted_label"] = "None"
                    logger.info(
                        f"[evaluate_testset]   Face #{face_idx}: UNIDENTIFIED — "
                        f"best_match='{face.get('name', 'N/A')}', "
                        f"sim={face['similarity']:.6f} < thresh={identification_threshold}"
                    )

                all_faces.append(face)

                # Compute similarity to ALL identities for this face
                identity_scores = self._compute_all_identity_scores(face["embedding"])
                for name in identity_names:
                    score = identity_scores.get(name, 0.0)
                    image_identity_scores[name].append(score)
                # Log top-3 identity scores for this face
                _sorted_scores = sorted(identity_scores.items(), key=lambda x: x[1], reverse=True)[:3]
                logger.debug(
                    f"[evaluate_testset]   Face #{face_idx} top-3 scores: "
                    + ", ".join(f"{n}={s:.4f}" for n, s in _sorted_scores)
                )

            # Image-level predictions: only identified names, or "None"
            # when no face could be identified (mirrors notebook approach)
            if identified_set:
                pred = sorted(identified_set)
                predictions.append(pred)
            else:
                pred = ["None"]
                predictions.append(pred)

            _is_correct = set(gt_labels) == set(pred)
            if _is_correct:
                _correct_images += 1
            logger.info(
                f"[evaluate_testset]   Result: pred={pred}, gt={gt_labels}, "
                f"correct={_is_correct}"
            )

            # Use max score per identity across all faces in the image
            max_scores = {
                name: max(scores) if scores else 0.0
                for name, scores in image_identity_scores.items()
            }
            per_image_scores.append(max_scores)

        _eval_elapsed = _time.time() - _eval_start
        logger.info(
            f"[evaluate_testset] Evaluation complete in {_eval_elapsed:.3f}s: "
            f"{_total_faces} faces across {len(items)} images, "
            f"{_correct_images}/{len(items)} images correct "
            f"({_correct_images/len(items)*100:.1f}%)" if len(items) > 0 else "(no items)"
        )

        # Compute metrics using the existing metrics module
        from src.metrics import calculate_metrics, normalize_detections_for_metrics

        normalised_preds = normalize_detections_for_metrics(ground_truth, predictions)
        metrics = calculate_metrics(ground_truth, predictions)

        return {
            "predictions": predictions,
            "ground_truth": ground_truth,
            "image_paths": image_paths,
            "metrics": metrics,
            "all_faces": all_faces,
            "per_image_scores": per_image_scores,
            "identity_names": identity_names,
        }

    def augment_from_testset(
        self,
        testset_path: str,
        upper_threshold: float = 0.50,
        lower_threshold: float = 0.25,
        review_top_k: int = 10,
    ) -> Dict[str, Any]:
        """
        Load images from a test-set JSON and categorise all detected faces
        using the two-threshold system for augmentation / human review.

        Unlike *evaluate_testset*, this method does **not** compute
        evaluation metrics — it is purely focused on reference DB
        augmentation.

        Returns dict with:
            categorised  – {"identified": [...], "augment": [...], "review": [...]}
            total_faces  – int, total number of faces detected
        """
        logger.info(
            f"[augment_from_testset] Loading augmentation test set: {testset_path}, "
            f"upper={upper_threshold}, lower={lower_threshold}, review_top_k={review_top_k}"
        )
        with open(testset_path, "r", encoding="utf-8") as f:
            items = json.load(f)

        logger.info(f"[augment_from_testset] Loaded {len(items)} items from test set")

        image_paths = [
            (
                str(BASE_DIR / item.get("path", ""))
                if not item.get("path", "").startswith("/")
                else item.get("path", "")
            )
            for item in items
        ]

        result = self.classify_with_thresholds(
            image_paths,
            upper_threshold=upper_threshold,
            lower_threshold=lower_threshold,
            review_top_k=review_top_k,
        )
        logger.info(
            f"[augment_from_testset] Augmentation analysis complete: "
            f"identified={len(result.get('identified', []))}, "
            f"augment={len(result.get('augment', []))}, "
            f"review={len(result.get('review', []))}"
        )
        return result
