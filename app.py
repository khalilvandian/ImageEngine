import os
import datetime
import tempfile
import streamlit as st

from src.logging_utils import setup_logger

logger = setup_logger()


def ensure_dirs():
    if not os.path.exists("logs"):
        os.makedirs("logs")
    if not os.path.exists("image_outputs"):
        os.makedirs("image_outputs")


def resolve_uploaded_or_default(uploaded_file, default_path, tempdir, filename):
    """Return a usable file path from an upload or fall back to the default path."""
    if uploaded_file:
        target_path = os.path.join(tempdir, filename)
        with open(target_path, "wb") as f:
            f.write(uploaded_file.read())
        return target_path
    return default_path


def classify_ui():
    st.header("Image Classification")

    # Celebrity data upload
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Configuration")
    with col2:
        celebrities_json_file = st.file_uploader(
            "Celebrities JSON",
            type=["json"],
            key="celebrities_json_upload_classify",
        )
    
    # Modular configuration - always shown
    st.markdown("**Select your model combination:**")
    col_det, col_emb, col_match = st.columns(3)
    
    with col_det:
        st.markdown("**Detection Model**")
        detection_model = st.selectbox(
            "Choose how to detect faces",
            options=[
                "buffalo_l",
                "buffalo_m",
                "buffalo_s",
                "antelopev2",
                "cnn",
                "hog"
            ],
            index=0,
            label_visibility="collapsed",
            help="buffalo_l: InsightFace SCRFD-10GF (highest accuracy)\nbuffalo_m: InsightFace SCRFD-2.5GF (balanced)\nbuffalo_s: InsightFace lightweight (fastest)\nantelopev2: InsightFace alternative\ncnn: face_recognition CNN (accurate)\nhog: face_recognition HOG (CPU-friendly)"
        )
    
    with col_emb:
        st.markdown("**Embedding Model**")
        embedding_model = st.selectbox(
            "Choose how to extract features",
            options=[
                "insightface",
                "insightface_buffalo_m",
                "insightface_buffalo_s",
                "insightface_antelopev2",
                "face_recognition_cnn",
                "face_recognition_hog",
                "vit"
            ],
            index=0,
            label_visibility="collapsed",
            help="insightface: 512-dim ResNet50 embeddings\nface_recognition_cnn: 128-dim dlib with CNN\nface_recognition_hog: 128-dim dlib with HOG\nvit: 768-dim CLIP Vision Transformer"
        )
    
    with col_match:
        st.markdown("**Matching Method**")
        matching_method = st.selectbox(
            "Choose how to compare embeddings",
            options=[
                "cosine_similarity",
                "euclidean_distance",
                "l2_distance"
            ],
            index=0,
            label_visibility="collapsed",
            help="cosine_similarity: Best for InsightFace, ViT (normalized dot product)\neuclidean_distance: Best for face_recognition (L2 distance inverted)\nl2_distance: L2 with Gaussian falloff"
        )
    
    # Threshold slider
    threshold = st.slider(
        "Similarity Threshold (higher = stricter matching)",
        min_value=0.0,
        max_value=1.0,
        value=0.6,
        step=0.05,
        help="Faces with similarity below this threshold will be marked as 'Unknown'"
    )
    
    # File upload
    multiple = st.toggle("Upload multiple files", value=False)
    uploaded_files = st.file_uploader(
        "Select image file(s)", type=["jpg", "jpeg", "png"], accept_multiple_files=multiple
    )

    if st.button("Classify"):
        logger.info("=" * 60)
        logger.info("Classify button clicked - starting classification workflow")
        logger.info("=" * 60)
        
        # Lazy import - only load when needed
        logger.info("Loading classification modules...")
        from src.classification import get_classifier, load_celebrities_from_json
        from src.image_utils import draw_bounding_boxes
        logger.info("Classification modules loaded successfully")
        
        ensure_dirs()
        st.info("Starting classification…")
        if not uploaded_files:
            logger.warning("No files selected by user")
            st.error("No files selected.")
            return
        
        logger.info(f"Files uploaded: {len(uploaded_files) if isinstance(uploaded_files, list) else 1}")

        with tempfile.TemporaryDirectory() as tempdir:
            logger.info("Resolving celebrity data JSON file path...")
            celebrities_json_path = resolve_uploaded_or_default(
                celebrities_json_file,
                default_path="data/celebrities.json",
                tempdir=tempdir,
                filename="celebrities.json",
            )
            logger.info(f"Using celebrity data from: {celebrities_json_path}")

            logger.info("Loading celebrity reference data...")
            celebrity_data = load_celebrities_from_json(celebrities_json_path)
            if not celebrity_data:
                logger.error("Failed to load celebrity data from JSON")
                st.error("No celebrity data loaded from JSON. Check path and content.")
                return
            logger.info(f"Loaded {len(celebrity_data)} celebrities successfully")

            logger.info(f"Initializing unified classifier")
            logger.info(f"  Detection model: {detection_model}")
            logger.info(f"  Embedding model: {embedding_model}")
            logger.info(f"  Matching method: {matching_method}")
            logger.info(f"  Threshold: {threshold}")
            classifier = get_classifier(
                "unified",
                celebrity_data,
                detection_model=detection_model,
                embedding_model=embedding_model,
                matching_method=matching_method,
                threshold=threshold
            )
            logger.info("Classifier initialized successfully")
            if classifier is None:
                logger.error(f"Failed to initialize unified classifier")
                st.error(
                    f"Could not initialize classifier. Check logs for details."
                )
                return

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            logger.info(f"Timestamp: {timestamp}")
            experiment_folder_name = (
                f"{timestamp}_{detection_model}_{embedding_model}_{matching_method}_"
                f"threshold_{threshold}"
            )
            output_dir = os.path.join("image_outputs", experiment_folder_name)
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Output directory created: {output_dir}")
            st.write(f"Output images will be saved to: {output_dir}")

            logger.info("Processing uploaded files...")
            image_paths = []
            path_to_name = {}

            for uploaded in uploaded_files if isinstance(uploaded_files, list) else [uploaded_files]:
                temp_path = os.path.join(tempdir, uploaded.name)
                with open(temp_path, "wb") as f:
                    f.write(uploaded.read())
                image_paths.append(temp_path)
                path_to_name[temp_path] = uploaded.name
                logger.info(f"Saved uploaded file: {uploaded.name}")

            logger.info(f"Starting classification on {len(image_paths)} image(s)...")
            output_by_path = classifier.classify_images(image_paths)
            logger.info("Classification complete")

            logger.info("Processing results and generating annotated images...")
            st.subheader("Results")
            for image_path, results in output_by_path.items():
                original_filename = path_to_name[image_path]
                logger.info(f"Processing results for: {original_filename}")
                logger.info(f"  Found {len(results)} face(s)")
                for result in results:
                    logger.info(f"    - {result['name']} (confidence: {result.get('confidence', 'N/A')})")
                annotated_image = draw_bounding_boxes(image_path, results)
                output_path = os.path.join(output_dir, original_filename)
                annotated_image.save(output_path)
                logger.info(f"Saved annotated image: {output_path}")

                st.markdown(f"**{original_filename}**")
                if results:
                    for result in results:
                        st.write(f"- {result['name']} at location {result['location']}")
                else:
                    st.write("- No celebrities detected.")
                st.image(annotated_image, caption=f"Annotated: {original_filename}")

        logger.info("=" * 60)
        logger.info("Classification workflow completed successfully")
        logger.info("=" * 60)
        st.success("Classification complete.")


def testset_ui():
    st.header("Model Testing")

    col1, col2, col3 = st.columns(3)
    with col1:
        test_set_upload = st.file_uploader(
            "Test Set JSON (defaults to testsets/test_set.json)",
            type=["json"],
            key="test_set_upload",
        )
    with col2:
        celebrities_json_file = st.file_uploader(
            "Celebrities JSON (defaults to data/celebrities.json)",
            type=["json"],
            key="celebrities_json_upload_test",
        )
    with col3:
        st.write("")  # Spacing
    
    # Modular configuration - always shown
    st.markdown("**Select your model combination:**")
    col_det, col_emb, col_match = st.columns(3)
    
    with col_det:
        st.markdown("**Detection Model**")
        detection_model = st.selectbox(
            "Choose how to detect faces",
            options=[
                "buffalo_l",
                "buffalo_m",
                "buffalo_s",
                "antelopev2",
                "cnn",
                "hog"
            ],
            index=0,
            label_visibility="collapsed",
            help="buffalo_l: InsightFace SCRFD-10GF (highest accuracy)\nbuffalo_m: InsightFace SCRFD-2.5GF (balanced)\nbuffalo_s: InsightFace lightweight (fastest)\nantelopev2: InsightFace alternative\ncnn: face_recognition CNN (accurate)\nhog: face_recognition HOG (CPU-friendly)",
            key="test_detection_model"
        )
    
    with col_emb:
        st.markdown("**Embedding Model**")
        embedding_model = st.selectbox(
            "Choose how to extract features",
            options=[
                "insightface",
                "insightface_buffalo_m",
                "insightface_buffalo_s",
                "insightface_antelopev2",
                "face_recognition_cnn",
                "face_recognition_hog",
                "vit"
            ],
            index=0,
            label_visibility="collapsed",
            help="insightface: 512-dim ResNet50 embeddings\nface_recognition_cnn: 128-dim dlib with CNN\nface_recognition_hog: 128-dim dlib with HOG\nvit: 768-dim CLIP Vision Transformer",
            key="test_embedding_model"
        )
    
    with col_match:
        st.markdown("**Matching Method**")
        matching_method = st.selectbox(
            "Choose how to compare embeddings",
            options=[
                "cosine_similarity",
                "euclidean_distance",
                "l2_distance"
            ],
            index=0,
            label_visibility="collapsed",
            help="cosine_similarity: Best for InsightFace, ViT (normalized dot product)\neuclidean_distance: Best for face_recognition (L2 distance inverted)\nl2_distance: L2 with Gaussian falloff",
            key="test_matching_method"
        )
    
    # Threshold slider
    threshold = st.slider(
        "Similarity Threshold (higher = stricter matching)",
        min_value=0.0,
        max_value=1.0,
        value=0.6,
        step=0.05,
        help="Faces with similarity below this threshold will be marked as 'Unknown'",
        key="test_threshold"
    )

    if st.button("Run Tests"):
        logger.info("=" * 60)
        logger.info("Run Tests button clicked - starting testing workflow")
        logger.info("=" * 60)
        
        # Lazy import - only load when needed
        logger.info("Loading classification and metrics modules...")
        from src.classification import get_classifier, load_celebrities_from_json
        from src.metrics import (
            load_test_set,
            run_classification_on_test_set,
            calculate_metrics,
            plot_confusion_matrix,
            plot_roc_curve,
            save_test_output_to_csv,
        )
        logger.info("Modules loaded successfully")
        
        ensure_dirs()
        st.info("Running tests… this may take a while")

        with tempfile.TemporaryDirectory() as tempdir:
            logger.info("Resolving test set JSON file path...")
            test_set_path = resolve_uploaded_or_default(
                test_set_upload,
                default_path="testsets/test_set.json",
                tempdir=tempdir,
                filename="test_set.json",
            )
            logger.info(f"Using test set from: {test_set_path}")

            logger.info("Loading test set...")
            image_paths, ground_truth_labels = load_test_set(test_set_path)
            logger.info(f"Loaded {len(image_paths)} images for testing")
            st.write(f"Loaded {len(image_paths)} images for testing.")

            celebrities_json_path = resolve_uploaded_or_default(
                celebrities_json_file,
                default_path="data/celebrities.json",
                tempdir=tempdir,
                filename="celebrities_test.json",
            )

            logger.info("Loading celebrity reference data...")
            celebrity_data = load_celebrities_from_json(celebrities_json_path)
            if not celebrity_data:
                logger.error("Failed to load celebrity data from JSON")
                st.error("No celebrity data loaded from JSON. Check path and content.")
                return
            logger.info(f"Loaded {len(celebrity_data)} celebrities successfully")

            logger.info(f"Initializing unified test classifier")
            logger.info(f"  Detection model: {detection_model}")
            logger.info(f"  Embedding model: {embedding_model}")
            logger.info(f"  Matching method: {matching_method}")
            logger.info(f"  Threshold: {threshold}")
            classifier = get_classifier(
                "unified",
                celebrity_data,
                detection_model=detection_model,
                embedding_model=embedding_model,
                matching_method=matching_method,
                threshold=threshold
            )
            logger.info("Test classifier initialized successfully")
            if classifier is None:
                logger.error(f"Failed to initialize unified classifier")
                st.error(
                    f"Could not initialize classifier. Check logs for details."
                )
                return

            logger.info("Running classification on test set...")
            predictions = run_classification_on_test_set(
                classifier, image_paths, output_image_dir="image_outputs"
            )
            logger.info("Test classification complete")

            logger.info("Calculating metrics...")
            metrics = calculate_metrics(ground_truth_labels, predictions)
            logger.info("Metrics calculated:")
            for metric, value in metrics.items():
                logger.info(f"  {metric}: {value}")
            st.subheader("Test Results")
            for metric, value in metrics.items():
                st.write(f"**{metric.replace('_', ' ').title()}**: {value}")

            logger.info("Saving test output to CSV...")
            csv_filepath = save_test_output_to_csv(
                image_paths, predictions, ground_truth_labels, f"{detection_model}_{embedding_model}_{matching_method}"
            )
            logger.info(f"Test output saved to: {csv_filepath}")
            st.write(f"Test output saved to: {csv_filepath}")

            logger.info("Generating confusion matrix...")
            confusion_matrix_base64 = plot_confusion_matrix(
                ground_truth_labels, predictions
            )
            st.subheader("Confusion Matrix")
            st.image(f"data:image/png;base64,{confusion_matrix_base64}")

            logger.info("Generating ROC curve...")
            roc_curve_base64 = plot_roc_curve(ground_truth_labels, predictions)
            st.subheader("ROC Curve")
            st.image(f"data:image/png;base64,{roc_curve_base64}")
            
            logger.info("=" * 60)
            logger.info("Testing workflow completed successfully")
            logger.info("=" * 60)


def main():
    logger.info("Starting Streamlit app...")
    st.set_page_config(page_title="Image Classification App", layout="wide")
    st.title("Image Classification App")
    logger.info("App interface loaded successfully")

    tabs = st.tabs(["Classify", "Testset"])
    with tabs[0]:
        classify_ui()
    with tabs[1]:
        testset_ui()


if __name__ == "__main__":
    main()
