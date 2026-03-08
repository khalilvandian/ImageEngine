import logging
import os
import json
import datetime
import time
from typing import Any, Dict, List, Optional


def setup_logger(log_dir="logs", console_level=logging.INFO, file_level=logging.DEBUG):
    """Return a logger that keeps verbose details in a file and a quieter view in the console."""
    logger = logging.getLogger(__name__)
    logger.setLevel(min(console_level, file_level))
    logger.propagate = False

    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(console_level)

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        log_filename = os.path.join(log_dir, f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        file_handler = logging.FileHandler(log_filename)
        file_handler.setLevel(file_level)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger


class SessionLogger:
    """
    Structured session logger that accumulates all data from an InsightFace
    tab run and writes a JSON report file for post-hoc analysis.

    Captures: configuration, model parameters, reference DB state, test set
    details, per-image results, augmentation actions, metrics, timing, and
    any other relevant data.

    Usage:
        session = SessionLogger(mode="evaluate")
        session.log_config(...)
        session.log_reference_db(...)
        ...
        session.save()  # writes JSON to logs/
    """

    def __init__(self, mode: str, log_dir: str = "logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        self._start_time = time.time()
        self._start_dt = datetime.datetime.now()
        self._timers: Dict[str, float] = {}
        self._logger = setup_logger()

        self.data: Dict[str, Any] = {
            "session_id": self._start_dt.strftime("%Y%m%d_%H%M%S_%f"),
            "mode": mode,
            "start_time": self._start_dt.isoformat(),
            "end_time": None,
            "total_duration_seconds": None,
            "configuration": {},
            "model": {},
            "reference_db": {},
            "test_set": {},
            "processing": {
                "images_processed": 0,
                "total_faces_detected": 0,
                "per_image_results": [],
                "timing": {},
            },
            "categorisation": {
                "identified_count": 0,
                "augment_count": 0,
                "review_count": 0,
                "identified_faces": [],
                "augment_faces": [],
                "review_faces": [],
            },
            "augmentation": {
                "enabled": False,
                "embeddings_added": 0,
                "actions": [],
                "reference_db_before": {},
                "reference_db_after": {},
            },
            "human_review": {
                "enabled": False,
                "decisions": [],
            },
            "evaluation": {
                "metrics": {},
                "advanced_metrics": {},
                "per_class_metrics": {},
                "predictions_vs_ground_truth": [],
            },
            "errors": [],
            "warnings": [],
        }

    # ── Timer helpers ────────────────────────────────────────────────

    def start_timer(self, name: str):
        """Start a named timer."""
        self._timers[name] = time.time()
        self._logger.debug(f"[SessionLog] Timer started: {name}")

    def stop_timer(self, name: str) -> float:
        """Stop a named timer and return elapsed seconds."""
        if name not in self._timers:
            return 0.0
        elapsed = time.time() - self._timers[name]
        self.data["processing"]["timing"][name] = round(elapsed, 4)
        self._logger.debug(f"[SessionLog] Timer stopped: {name} = {elapsed:.4f}s")
        del self._timers[name]
        return elapsed

    # ── Configuration logging ────────────────────────────────────────

    def log_config(
        self,
        model_name: str,
        upper_threshold: float,
        lower_threshold: float,
        enable_augment: bool = False,
        enable_review: bool = False,
        review_top_k: int = 10,
        identification_threshold: Optional[float] = None,
        f_beta: Optional[float] = None,
        fixed_recall: Optional[float] = None,
        fixed_precision: Optional[float] = None,
        **extra,
    ):
        """Log all configuration parameters."""
        self.data["configuration"] = {
            "model_name": model_name,
            "upper_threshold": upper_threshold,
            "lower_threshold": lower_threshold,
            "enable_augment": enable_augment,
            "enable_review": enable_review,
            "review_top_k": review_top_k,
            "identification_threshold": identification_threshold,
            "f_beta": f_beta,
            "fixed_recall": fixed_recall,
            "fixed_precision": fixed_precision,
            **extra,
        }
        self._logger.info(
            f"[SessionLog] Configuration: model={model_name}, "
            f"upper_thresh={upper_threshold}, lower_thresh={lower_threshold}, "
            f"augment={enable_augment}, review={enable_review}, "
            f"review_top_k={review_top_k}"
        )
        if identification_threshold is not None:
            self._logger.info(
                f"[SessionLog] Evaluation params: ident_thresh={identification_threshold}, "
                f"f_beta={f_beta}, fixed_recall={fixed_recall}, fixed_precision={fixed_precision}"
            )

    def log_model(
        self,
        model_name: str,
        det_size: tuple = (640, 640),
        providers: Optional[List[str]] = None,
        cache_dir: str = "data/cache",
    ):
        """Log model details."""
        self.data["model"] = {
            "name": model_name,
            "detection_input_size": list(det_size),
            "onnx_providers": providers or [],
            "cache_dir": cache_dir,
        }
        self._logger.info(
            f"[SessionLog] Model: {model_name}, det_size={det_size}, providers={providers}"
        )

    # ── Reference DB logging ─────────────────────────────────────────

    def log_reference_db(
        self,
        summary: Dict[str, int],
        source: str = "in-memory",
        total_embeddings: Optional[int] = None,
    ):
        """Log reference DB state."""
        total = total_embeddings or sum(summary.values())
        self.data["reference_db"] = {
            "source": source,
            "identities": len(summary),
            "total_embeddings": total,
            "per_identity": dict(summary),
        }
        self._logger.info(
            f"[SessionLog] Reference DB: source={source}, "
            f"{len(summary)} identities, {total} embeddings"
        )
        for name, count in summary.items():
            self._logger.info(f"[SessionLog]   - {name}: {count} embedding(s)")

    # ── Test set logging ─────────────────────────────────────────────

    def log_test_set(
        self,
        path: str,
        num_images: int,
        labels_summary: Optional[Dict[str, int]] = None,
        raw_items: Optional[List[Dict]] = None,
    ):
        """Log test set details."""
        self.data["test_set"] = {
            "path": path,
            "filename": os.path.basename(path),
            "num_images": num_images,
            "labels_summary": labels_summary or {},
        }
        if raw_items:
            # Count unique labels
            label_counts: Dict[str, int] = {}
            for item in raw_items:
                for lbl in item.get("labels", []):
                    label_counts[lbl] = label_counts.get(lbl, 0) + 1
            if not label_counts:
                label_counts["(no labels)"] = num_images
            self.data["test_set"]["labels_summary"] = label_counts
            self._logger.info(f"[SessionLog] Test set labels distribution:")
            for lbl, cnt in sorted(label_counts.items()):
                self._logger.info(f"[SessionLog]   - {lbl}: {cnt} image(s)")

        self._logger.info(
            f"[SessionLog] Test set: {path} ({num_images} images)"
        )

    # ── Per-image result logging ─────────────────────────────────────

    def log_image_result(
        self,
        image_path: str,
        faces_detected: int,
        face_details: List[Dict[str, Any]],
        processing_time: Optional[float] = None,
    ):
        """Log results for a single image."""
        compact_faces = []
        for f in face_details:
            compact_faces.append({
                "name": f.get("name"),
                "similarity": round(float(f.get("similarity", 0)), 6),
                "category": f.get("category", "unknown"),
                "bbox_xyxy": list(f["bbox_xyxy"]) if "bbox_xyxy" in f else None,
                "predicted_label": f.get("predicted_label"),
                "ground_truth": f.get("ground_truth"),
            })

        entry = {
            "image_path": image_path,
            "filename": os.path.basename(image_path),
            "faces_detected": faces_detected,
            "faces": compact_faces,
            "processing_time_seconds": round(processing_time, 4) if processing_time else None,
        }
        self.data["processing"]["per_image_results"].append(entry)
        self.data["processing"]["images_processed"] += 1
        self.data["processing"]["total_faces_detected"] += faces_detected

    # ── Categorisation logging ───────────────────────────────────────

    def log_categorisation(self, categorised: Dict[str, List[Dict[str, Any]]]):
        """Log the categorisation results summary and per-face details."""
        for cat_name in ("identified", "augment", "review"):
            faces = categorised.get(cat_name, [])
            count = len(faces)
            self.data["categorisation"][f"{cat_name}_count"] = count

            compact = []
            for f in faces:
                compact.append({
                    "name": f.get("name"),
                    "similarity": round(float(f.get("similarity", 0)), 6),
                    "image_path": f.get("image_path", ""),
                    "filename": os.path.basename(f.get("image_path", "")),
                    "bbox_xyxy": list(f["bbox_xyxy"]) if "bbox_xyxy" in f else None,
                })
            self.data["categorisation"][f"{cat_name}_faces"] = compact

        total = sum(self.data["categorisation"][f"{c}_count"] for c in ("identified", "augment", "review"))
        self._logger.info(
            f"[SessionLog] Categorisation: {total} total faces — "
            f"identified={self.data['categorisation']['identified_count']}, "
            f"augment={self.data['categorisation']['augment_count']}, "
            f"review={self.data['categorisation']['review_count']}"
        )

        # Log similarity distributions per category
        for cat_name in ("identified", "augment", "review"):
            faces = categorised.get(cat_name, [])
            if faces:
                sims = [f.get("similarity", 0) for f in faces]
                self._logger.info(
                    f"[SessionLog]   {cat_name}: count={len(sims)}, "
                    f"sim_min={min(sims):.4f}, sim_max={max(sims):.4f}, "
                    f"sim_mean={sum(sims)/len(sims):.4f}"
                )

    # ── Augmentation logging ─────────────────────────────────────────

    def log_augmentation_start(self, ref_db_summary: Dict[str, int]):
        """Log reference DB state before augmentation."""
        self.data["augmentation"]["reference_db_before"] = dict(ref_db_summary)
        self._logger.info(
            f"[SessionLog] Augmentation: reference DB before — "
            f"{sum(ref_db_summary.values())} total embeddings"
        )

    def log_augmentation_action(self, identity: str, source: str, similarity: float):
        """Log a single augmentation action."""
        self.data["augmentation"]["actions"].append({
            "identity": identity,
            "source": source,
            "similarity": round(similarity, 6),
            "timestamp": datetime.datetime.now().isoformat(),
        })

    def log_augmentation_end(self, ref_db_summary: Dict[str, int], embeddings_added: int):
        """Log reference DB state after augmentation."""
        self.data["augmentation"]["reference_db_after"] = dict(ref_db_summary)
        self.data["augmentation"]["embeddings_added"] = embeddings_added
        self._logger.info(
            f"[SessionLog] Augmentation complete: {embeddings_added} embeddings added, "
            f"reference DB now has {sum(ref_db_summary.values())} total embeddings"
        )

    # ── Human review logging ─────────────────────────────────────────

    def log_review_decision(self, face_index: int, action: str, assigned_name: Optional[str], similarity: float):
        """Log a human review decision."""
        self.data["human_review"]["decisions"].append({
            "face_index": face_index,
            "action": action,
            "assigned_name": assigned_name,
            "similarity": round(similarity, 6),
            "timestamp": datetime.datetime.now().isoformat(),
        })
        self._logger.info(
            f"[SessionLog] Review decision: face#{face_index} — "
            f"action='{action}', assigned='{assigned_name}', sim={similarity:.4f}"
        )

    # ── Evaluation logging ───────────────────────────────────────────

    def log_evaluation_metrics(self, metrics: Dict[str, Any]):
        """Log aggregate evaluation metrics."""
        self.data["evaluation"]["metrics"] = _make_serialisable(metrics)
        self._logger.info(f"[SessionLog] Evaluation metrics:")
        for k, v in metrics.items():
            self._logger.info(f"[SessionLog]   {k}: {v}")

    def log_advanced_metrics(self, adv: Dict[str, Any]):
        """Log advanced evaluation metrics."""
        self.data["evaluation"]["advanced_metrics"] = _make_serialisable(
            adv.get("scalar_metrics", {})
        )
        self.data["evaluation"]["per_class_metrics"] = _make_serialisable(
            adv.get("per_class_metrics", {})
        )
        self._logger.info(f"[SessionLog] Advanced metrics (scalar):")
        for k, v in adv.get("scalar_metrics", {}).items():
            self._logger.info(f"[SessionLog]   {k}: {v}")
        self._logger.info(f"[SessionLog] Per-class metrics:")
        for cls, m in adv.get("per_class_metrics", {}).items():
            self._logger.info(
                f"[SessionLog]   {cls}: precision={m.get('precision', 'N/A')}, "
                f"recall={m.get('recall', 'N/A')}, f_beta={m.get('f_beta', 'N/A')}, "
                f"roc_auc={m.get('roc_auc', 'N/A')}, pr_auc={m.get('pr_auc', 'N/A')}"
            )

    def log_predictions_vs_ground_truth(
        self,
        image_paths: List[str],
        ground_truth: List[List[str]],
        predictions: List[List[str]],
    ):
        """Log per-image predictions vs ground truth."""
        entries = []
        correct = 0
        for path, gt, pred in zip(image_paths, ground_truth, predictions):
            match = set(gt) == set(pred)
            if match:
                correct += 1
            entries.append({
                "image": os.path.basename(path),
                "ground_truth": gt,
                "predictions": pred,
                "correct": match,
            })
        self.data["evaluation"]["predictions_vs_ground_truth"] = entries
        total = len(entries)
        self._logger.info(
            f"[SessionLog] Predictions vs ground truth: {correct}/{total} correct "
            f"({correct/total*100:.1f}%)" if total > 0 else
            "[SessionLog] No predictions to compare"
        )

    # ── Error / warning logging ──────────────────────────────────────

    def log_error(self, message: str, context: Optional[Dict] = None):
        """Log an error."""
        self.data["errors"].append({
            "message": message,
            "context": context,
            "timestamp": datetime.datetime.now().isoformat(),
        })
        self._logger.error(f"[SessionLog] {message}")

    def log_warning(self, message: str, context: Optional[Dict] = None):
        """Log a warning."""
        self.data["warnings"].append({
            "message": message,
            "context": context,
            "timestamp": datetime.datetime.now().isoformat(),
        })
        self._logger.warning(f"[SessionLog] {message}")

    # ── Finalise and save ────────────────────────────────────────────

    def save(self) -> str:
        """
        Finalise the session and write the JSON report file.
        Returns the path to the saved file.
        """
        end_dt = datetime.datetime.now()
        self.data["end_time"] = end_dt.isoformat()
        self.data["total_duration_seconds"] = round(time.time() - self._start_time, 4)

        filename = (
            f"insightface_session_{self.data['mode']}_"
            f"{self.data['session_id']}.json"
        )
        filepath = os.path.join(self.log_dir, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, default=str)

        self._logger.info(
            f"[SessionLog] Session report saved → {filepath} "
            f"(duration: {self.data['total_duration_seconds']}s)"
        )
        return filepath


def _make_serialisable(obj: Any) -> Any:
    """Recursively convert numpy/non-JSON types for JSON serialisation."""
    import numpy as np

    if isinstance(obj, dict):
        return {k: _make_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serialisable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
