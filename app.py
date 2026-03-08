import os
import datetime
import tempfile
import streamlit as st
import matplotlib.pyplot as plt

from src.logging_utils import setup_logger, SessionLogger

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
                "face_recognition",
                "vit"
            ],
            index=0,
            label_visibility="collapsed",
            help="insightface: 512-dim ResNet50 embeddings\nface_recognition: 128-dim dlib (HOG/CNN detector chosen separately)\nvit: 768-dim CLIP Vision Transformer"
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
                default_path="data/references.json",
                tempdir=tempdir,
                filename="references.json",
            )
            logger.info(f"Using celebrity data from: {celebrities_json_path}")

            logger.info("Loading celebrity reference data...")
            celebrity_data = load_celebrities_from_json(celebrities_json_path)
            if not celebrity_data:
                logger.error("Failed to load celebrity data from JSON")
                st.error("No celebrity data loaded from JSON. Check path and content.")
                return
            logger.info(f"Loaded {len(celebrity_data)} celebrities successfully")

            # Match UI selections to classifier defaults (cnn benefits from higher upsample + multi-pass)
            detection_upsample = 2 if detection_model == "cnn" else 1
            enable_multi_pass = detection_model == "cnn"

            logger.info(f"Initializing unified classifier")
            logger.info(f"  Detection model: {detection_model}")
            logger.info(f"  Embedding model: {embedding_model}")
            logger.info(f"  Matching method: {matching_method}")
            logger.info(f"  Threshold: {threshold}")
            logger.info(f"  Detection upsample: {detection_upsample}")
            logger.info(f"  Multi-pass: {enable_multi_pass}")
            classifier = get_classifier(
                "unified",
                celebrity_data,
                detection_model=detection_model,
                embedding_model=embedding_model,
                matching_method=matching_method,
                threshold=threshold,
                detection_upsample=detection_upsample,
                enable_multi_pass=enable_multi_pass
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
            "Celebrities JSON (defaults to data/references.json)",
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
                "face_recognition",
                "vit"
            ],
            index=0,
            label_visibility="collapsed",
            help="insightface: 512-dim ResNet50 embeddings\nface_recognition: 128-dim dlib (HOG/CNN detector chosen separately)\nvit: 768-dim CLIP Vision Transformer",
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
                default_path="data/references.json",
                tempdir=tempdir,
                filename="references_test.json",
            )

            logger.info("Loading celebrity reference data...")
            celebrity_data = load_celebrities_from_json(celebrities_json_path)
            if not celebrity_data:
                logger.error("Failed to load celebrity data from JSON")
                st.error("No celebrity data loaded from JSON. Check path and content.")
                return
            logger.info(f"Loaded {len(celebrity_data)} celebrities successfully")

            detection_upsample = 2 if detection_model == "cnn" else 1
            enable_multi_pass = detection_model == "cnn"

            logger.info(f"Initializing unified test classifier")
            logger.info(f"  Detection model: {detection_model}")
            logger.info(f"  Embedding model: {embedding_model}")
            logger.info(f"  Matching method: {matching_method}")
            logger.info(f"  Threshold: {threshold}")
            logger.info(f"  Detection upsample: {detection_upsample}")
            logger.info(f"  Multi-pass: {enable_multi_pass}")
            classifier = get_classifier(
                "unified",
                celebrity_data,
                detection_model=detection_model,
                embedding_model=embedding_model,
                matching_method=matching_method,
                threshold=threshold,
                detection_upsample=detection_upsample,
                enable_multi_pass=enable_multi_pass
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


def _render_face_card(face, show_main_image=True):
    """Render a single face with its source image side by side."""
    main_img = face.get("image_path", "") if show_main_image else None
    if main_img:
        c_main, c_face = st.columns([2, 1])
        with c_main:
            st.image(
                main_img,
                caption=face.get("original_filename", os.path.basename(face.get("image_path", ""))),
                width="stretch",
            )
        with c_face:
            st.image(
                face["face_crop_rgb"],
                caption=(
                    f"{face.get('name', 'Unknown')}\n"
                    f"sim={face['similarity']:.3f}"
                ),
                width="stretch",
            )
    else:
        st.image(
            face["face_crop_rgb"],
            caption=(
                f"{face.get('name', 'Unknown')}\n"
                f"sim={face['similarity']:.3f}\n"
                f"{face.get('original_filename', '')}"
            ),
            width="stretch",
        )


def _render_categorised_results(
    categorised, engine, prefix="if",
    enable_augment=False, enable_review=False,
    upper_threshold=0.50, lower_threshold=0.25,
    testset_name=None,
):
    """
    Shared renderer for the three-category (identified / augment / review)
    results display with human feedback, used by both Classify and Evaluate modes.

    Each face entry is shown alongside its source image so users can tell
    where it came from.
    """
    PAGE_SIZE = 10

    n_id = len(categorised["identified"])
    n_aug = len(categorised["augment"])
    n_rev = len(categorised["review"])
    total = n_id + n_aug + n_rev

    st.markdown("---")
    st.subheader(f"Results — {total} face(s) detected")
    met_cols = st.columns(3)
    met_cols[0].metric("Identified", n_id)
    met_cols[1].metric("Auto-augment", n_aug)
    met_cols[2].metric("Needs Review", n_rev)

    # Helper: does this set of faces have image paths for lazy loading?
    has_main = any(
        face.get("image_path")
        for cat in categorised.values()
        for face in cat
    )

    # ── Section 1: Identified ────────────────────────────────────────
    with st.expander(f"✅ Identified ({n_id})", expanded=n_id > 0):
        if not categorised["identified"]:
            st.info("No faces met the upper threshold.")
        else:
            # Pagination
            total_pages_id = max(1, (n_id + PAGE_SIZE - 1) // PAGE_SIZE)
            page_id = st.number_input(
                "Page", min_value=1, max_value=total_pages_id, value=1,
                key=f"{prefix}_id_page",
            )
            start = (page_id - 1) * PAGE_SIZE
            end = min(start + PAGE_SIZE, n_id)
            st.caption(f"Showing {start + 1}–{end} of {n_id}")

            for face in categorised["identified"][start:end]:
                _render_face_card(face, show_main_image=has_main)
                gt = face.get("ground_truth")
                if gt is not None:
                    st.caption(f"Ground truth: {', '.join(gt) if gt else '(none)'}")
                st.markdown("---")

    # ── Section 2: Augment ───────────────────────────────────────────
    aug_label = (
        f"🔄 Auto-augment Reference DB ({n_aug})"
        if enable_augment
        else f"🔄 Auto-augment Reference DB ({n_aug}) — disabled"
    )
    with st.expander(aug_label, expanded=enable_augment and n_aug > 0):
        if not categorised["augment"]:
            st.info("No faces in the augmentation range.")
        elif not enable_augment:
            st.info(
                "Augmentation is disabled. Enable the checkbox above to "
                "review and apply these entries."
            )
        else:
            st.caption(
                "These faces will be added to the reference database under "
                "their matched identity to improve future recognition."
            )
            # Pagination
            total_pages_aug = max(1, (n_aug + PAGE_SIZE - 1) // PAGE_SIZE)
            page_aug = st.number_input(
                "Page", min_value=1, max_value=total_pages_aug, value=1,
                key=f"{prefix}_aug_page",
            )
            start = (page_aug - 1) * PAGE_SIZE
            end = min(start + PAGE_SIZE, n_aug)
            st.caption(f"Showing {start + 1}–{end} of {n_aug}")

            for face in categorised["augment"][start:end]:
                _render_face_card(face, show_main_image=has_main)
                gt = face.get("ground_truth")
                if gt is not None:
                    st.caption(f"Ground truth: {', '.join(gt) if gt else '(none)'}")
                st.markdown("---")

    # ── Section 3: Human Review ──────────────────────────────────────
    decisions_key = f"{prefix}_review_decisions"
    rev_label = (
        f"🔍 Human Review ({n_rev})"
        if enable_review
        else f"🔍 Human Review ({n_rev}) — disabled"
    )
    with st.expander(rev_label, expanded=enable_review and n_rev > 0):
        if not categorised["review"]:
            st.info("No faces below the lower threshold.")
        elif not enable_review:
            st.info(
                "Human review is disabled. Enable the checkbox above to "
                "review faces and provide annotations."
            )
        else:
            st.caption(
                "These faces could not be confidently matched. "
                "Please review each one and assign an identity or mark as new."
            )
            decisions = st.session_state.get(decisions_key, {})

            # Pagination
            total_pages_rev = max(1, (n_rev + PAGE_SIZE - 1) // PAGE_SIZE)
            page_rev = st.number_input(
                "Page", min_value=1, max_value=total_pages_rev, value=1,
                key=f"{prefix}_rev_page",
            )
            start = (page_rev - 1) * PAGE_SIZE
            end = min(start + PAGE_SIZE, n_rev)
            st.caption(f"Showing {start + 1}–{end} of {n_rev}")

            for idx in range(start, end):
                face = categorised["review"][idx]
                st.markdown("---")

                # Show main image + face crop side by side (lazy load via path)
                main_img_path = face.get("image_path", "") if has_main else None
                if main_img_path:
                    img_col, face_col, action_col = st.columns([2, 1, 3])
                    with img_col:
                        st.image(
                            main_img_path,
                            caption=face.get(
                                "original_filename",
                                os.path.basename(face.get("image_path", "")),
                            ),
                            width="stretch",
                        )
                    with face_col:
                        st.image(
                            face["face_crop_rgb"],
                            caption=f"sim={face['similarity']:.3f}",
                            width="stretch",
                        )
                else:
                    face_col_only, action_col = st.columns([1, 3])
                    with face_col_only:
                        st.image(
                            face["face_crop_rgb"],
                            caption=f"sim={face['similarity']:.3f}",
                            width="stretch",
                        )

                with action_col:
                    suggestion = face["name"] if face["name"] else "Unknown"
                    st.write(
                        f"**Best match:** {suggestion} "
                        f"(similarity {face['similarity']:.3f})"
                    )
                    source_name = face.get(
                        "original_filename", os.path.basename(face.get("image_path", ""))
                    )
                    st.write(f"**Source:** {source_name}")
                    gt = face.get("ground_truth")
                    if gt is not None:
                        st.write(
                            f"**Ground truth:** {', '.join(gt) if gt else '(none)'}"
                        )

                    action = st.radio(
                        "Action",
                        [
                            "Skip (do nothing)",
                            f"Confirm as '{suggestion}'",
                            "Assign to existing identity",
                            "Create new identity",
                        ],
                        key=f"{prefix}_review_action_{idx}",
                        index=0,
                    )

                    assigned_name = None
                    if action.startswith("Assign"):
                        existing_names = sorted(engine.reference_db.keys())
                        if existing_names:
                            assigned_name = st.selectbox(
                                "Choose identity",
                                existing_names,
                                key=f"{prefix}_review_assign_{idx}",
                            )
                        else:
                            st.warning("No existing identities.")
                    elif action.startswith("Create"):
                        assigned_name = st.text_input(
                            "New identity name",
                            key=f"{prefix}_review_newname_{idx}",
                        )

                    decisions[idx] = {
                        "action": action,
                        "assigned_name": assigned_name,
                        "suggestion": suggestion,
                    }
            st.session_state[decisions_key] = decisions

    # ── Apply Changes Button ─────────────────────────────────────────
    st.markdown("---")
    if st.button("💾 Apply Changes to Reference DB", key=f"{prefix}_apply_btn"):
        logger.info("=" * 60)
        logger.info(f"[Apply Changes] User clicked Apply Changes (prefix={prefix})")
        logger.info(f"[Apply Changes] enable_augment={enable_augment}, enable_review={enable_review}")
        logger.info(f"[Apply Changes] upper_threshold={upper_threshold}, lower_threshold={lower_threshold}")
        logger.info("=" * 60)

        _before_summary = engine.get_reference_summary()
        logger.info(f"[Apply Changes] Reference DB BEFORE: {sum(_before_summary.values())} embeddings")
        for name, count in _before_summary.items():
            logger.info(f"[Apply Changes]   - {name}: {count}")

        added_count = 0

        # 1. Auto-augment (only when enabled)
        if enable_augment:
            logger.info(f"[Apply Changes] Auto-augmenting with {n_aug} face(s)...")
            added_count += engine.auto_augment_references(categorised["augment"])
            logger.info(f"[Apply Changes] Auto-augmentation added {added_count} embedding(s)")
        else:
            logger.info("[Apply Changes] Auto-augmentation is disabled — skipping")

        # 2. Process human review decisions (only when enabled)
        if enable_review:
            decisions = st.session_state.get(decisions_key, {})
            logger.info(f"[Apply Changes] Processing {len(decisions)} human review decision(s)...")
            for idx, face in enumerate(categorised["review"]):
                dec = decisions.get(idx)
                if dec is None:
                    continue
                action = dec["action"]
                embedding = face["embedding"]
                source = face.get("image_path", "review")

                logger.info(
                    f"[Apply Changes] Review face #{idx}: action='{action}', "
                    f"sim={face['similarity']:.6f}, source={os.path.basename(source)}"
                )

                if action.startswith("Confirm"):
                    engine.add_embedding_to_identity(
                        dec["suggestion"], embedding, source=source
                    )
                    logger.info(f"[Apply Changes]   -> Confirmed as '{dec['suggestion']}'")
                    added_count += 1
                elif action.startswith("Assign") and dec.get("assigned_name"):
                    engine.add_embedding_to_identity(
                        dec["assigned_name"], embedding, source=source
                    )
                    logger.info(f"[Apply Changes]   -> Assigned to '{dec['assigned_name']}'")
                    added_count += 1
                elif action.startswith("Create") and dec.get("assigned_name"):
                    engine.create_new_identity(
                        dec["assigned_name"], embedding, source=source
                    )
                    logger.info(f"[Apply Changes]   -> Created new identity '{dec['assigned_name']}'")
                    added_count += 1
                else:
                    logger.info(f"[Apply Changes]   -> Skipped (action='{action}')")
        else:
            logger.info("[Apply Changes] Human review is disabled — skipping")

        # Persist — versioned save with metadata
        engine.save_reference_db_versioned(
            upper_threshold=upper_threshold,
            lower_threshold=lower_threshold,
            testset_name=testset_name,
        )

        _after_summary = engine.get_reference_summary()
        logger.info(f"[Apply Changes] Reference DB AFTER: {sum(_after_summary.values())} embeddings")
        for name, count in _after_summary.items():
            _delta = count - _before_summary.get(name, 0)
            if _delta > 0:
                logger.info(f"[Apply Changes]   - {name}: {count} (+{_delta})")
            else:
                logger.info(f"[Apply Changes]   - {name}: {count}")

        logger.info(f"[Apply Changes] Total embeddings added: {added_count}")

        notes = []
        if not enable_augment:
            notes.append("auto-augmentation disabled")
        if not enable_review:
            notes.append("human review disabled")
        note_str = f" ({', '.join(notes)})" if notes else ""
        st.success(
            f"Applied changes: {added_count} embedding(s) added{note_str}. "
            f"Reference DB saved."
        )
        logger.info(f"[Apply Changes] Complete: {added_count} embedding(s) added{note_str}")
        logger.info("=" * 60)
        # Refresh
        st.rerun()


def _get_insightface_engine(model_name="buffalo_l"):
    """
    Return a cached InsightFaceEngine, using st.cache_resource so the heavy
    ONNX model load happens only once per model_name (survives reruns and
    widget changes like threshold sliders).
    """
    # Use cache_resource for the expensive part (model loading)
    @st.cache_resource(show_spinner=False)
    def _load_engine(name: str):
        from src.insightface_engine import InsightFaceEngine

        engine = InsightFaceEngine(model_name=name)
        engine.load_references()
        return engine

    return _load_engine(model_name)


# ---------- Per-class sample gallery helper ---------------------------


def _load_face_crop(image_path, bbox_xyxy):
    """Load an image and crop the face region. Returns a PIL Image or None."""
    from PIL import Image

    try:
        img = Image.open(image_path).convert("RGB")
    except Exception:
        return None

    if bbox_xyxy is not None:
        x1, y1, x2, y2 = bbox_xyxy
        w, h = img.size
        # Clamp to image bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 > x1 and y2 > y1:
            img = img.crop((x1, y1, x2, y2))
    return img


def _pick_samples(sorted_faces, k=5):
    """
    From a list of face dicts already sorted by similarity (ascending),
    pick *k* from the bottom (lowest), *k* from the top (highest),
    and *k* from near the median.  Returns (bottom, median, top) lists.
    """
    n = len(sorted_faces)
    if n == 0:
        return [], [], []

    bottom = sorted_faces[:k]
    top = sorted_faces[-k:] if n > k else sorted_faces[:]
    mid_start = max(0, n // 2 - k // 2)
    mid_end = min(n, mid_start + k)
    median = sorted_faces[mid_start:mid_end]
    return bottom, median, top


def _render_face_row(faces, label):
    """Render a horizontal row of face crops with similarity captions."""
    if not faces:
        st.info(f"No {label} samples available.")
        return
    cols = st.columns(len(faces))
    for col, face in zip(cols, faces):
        with col:
            img = _load_face_crop(face["image_path"], face.get("bbox_xyxy"))
            if img is not None:
                st.image(img, width="stretch")
            else:
                st.warning("Image not found")
            gt = ", ".join(face.get("ground_truth", [])) or "—"
            st.caption(
                f"sim **{face['similarity']:.3f}**  \n"
                f"GT: {gt}  \n"
                f"_{face.get('original_filename', '')}_"
            )


def _render_per_class_gallery(face_samples, class_names):
    """
    For each class, show an expandable section with 15 sample faces:
    top-5, median-5, bottom-5 by similarity.
    """
    # Group faces by threshold-aware predicted label
    by_class = {}
    for f in face_samples:
        label = f.get("predicted_label", "Unknown")
        by_class.setdefault(label, []).append(f)

    for cls_name in class_names:
        cls_faces = by_class.get(cls_name, [])
        # Sort by similarity ascending
        cls_faces_sorted = sorted(cls_faces, key=lambda x: x["similarity"])
        n = len(cls_faces_sorted)

        with st.expander(f"**{cls_name}** — {n} face(s) classified", expanded=False):
            if n == 0:
                st.info("No faces classified as this identity.")
                continue

            bottom, median, top = _pick_samples(cls_faces_sorted, k=5)

            st.markdown("**Top similarity (most confident)**")
            _render_face_row(list(reversed(top)), "top")

            st.markdown("**Median similarity**")
            _render_face_row(median, "median")

            st.markdown("**Lowest similarity (least confident)**")
            _render_face_row(bottom, "bottom")


def _render_misclassification_gallery(face_samples, class_names, details):
    """
    For each class show up to 10 misclassified samples (false-positives and
    false-negatives) inside an expander.  Each sample renders the full image
    alongside the face crop.

    Misclassification types shown per class:
    - **FP (False Positive)**: a face was labelled as this class but the
      class is NOT in the image's ground truth.
    - **FN (False Negative)**: the class IS in the image's ground truth but
      no face in that image was labelled as this class.
    """

    # ---- build per-image prediction sets from face_samples ----
    # We need image-level view to detect FN (missed labels).
    from collections import defaultdict
    images_by_path: dict = defaultdict(lambda: {
        "preds": set(), "gt": [], "faces": [],
    })
    for f in face_samples:
        path = f["image_path"]
        images_by_path[path]["preds"].add(f["predicted_label"])
        images_by_path[path]["gt"] = f["ground_truth"]
        images_by_path[path]["faces"].append(f)

    for cls_name in class_names:
        # --- False Positives: faces labelled as cls but cls NOT in GT ---
        fp_faces = []
        for f in face_samples:
            pred = f["predicted_label"]
            gt = f["ground_truth"] if f["ground_truth"] else ["None"]
            if pred == cls_name and cls_name not in gt:
                fp_faces.append(f)

        # --- False Negatives: images where cls IS in GT but wasn't predicted ---
        fn_images = []
        for path, info in images_by_path.items():
            gt_labels = info["gt"] if info["gt"] else ["None"]
            if cls_name in gt_labels and cls_name not in info["preds"]:
                fn_images.append(info)

        total_errors = len(fp_faces) + len(fn_images)
        if total_errors == 0:
            continue  # skip classes with no errors

        with st.expander(
            f"**{cls_name}** — {total_errors} error(s)  "
            f"({len(fp_faces)} FP, {len(fn_images)} FN)",
            expanded=False,
        ):
            # ---- False Positives ----
            if fp_faces:
                st.markdown(
                    f"**False Positives** — face labelled *{cls_name}* "
                    f"but class not in ground truth ({len(fp_faces)} total, "
                    f"showing up to 10)"
                )
                for f in fp_faces[:10]:
                    _render_error_sample(f, error_type="FP")
                st.divider()

            # ---- False Negatives ----
            if fn_images:
                st.markdown(
                    f"**False Negatives** — *{cls_name}* expected in "
                    f"ground truth but not predicted ({len(fn_images)} total, "
                    f"showing up to 10)"
                )
                for img_info in fn_images[:10]:
                    # Pick the face with the highest similarity to show
                    # what the model *did* predict instead
                    best_face = max(img_info["faces"], key=lambda x: x["similarity"])
                    _render_error_sample(best_face, error_type="FN", expected_label=cls_name)


def _render_error_sample(face, error_type="FP", expected_label=None):
    """
    Render one misclassification sample: full image + face crop side by side
    with metadata.
    """
    from PIL import Image

    col_img, col_face, col_info = st.columns([2, 1, 2])

    with col_img:
        try:
            full_img = Image.open(face["image_path"]).convert("RGB")
            st.image(full_img, caption="Full image", width="stretch")
        except Exception:
            st.warning("Image not found")

    with col_face:
        crop = _load_face_crop(face["image_path"], face.get("bbox_xyxy"))
        if crop is not None:
            st.image(crop, caption="Face crop", width="stretch")
        else:
            st.warning("Crop unavailable")

    with col_info:
        gt = ", ".join(face.get("ground_truth", [])) or "—"
        pred = face.get("predicted_label", "—")
        sim = face.get("similarity", 0.0)
        fname = face.get("original_filename", "")

        if error_type == "FP":
            st.markdown(
                f"**Type:** False Positive  \n"
                f"**Predicted:** {pred}  \n"
                f"**Ground Truth:** {gt}  \n"
                f"**Similarity:** {sim:.3f}  \n"
                f"**File:** _{fname}_"
            )
        else:
            st.markdown(
                f"**Type:** False Negative  \n"
                f"**Expected:** {expected_label}  \n"
                f"**Predicted instead:** {pred}  \n"
                f"**Ground Truth:** {gt}  \n"
                f"**Similarity:** {sim:.3f}  \n"
                f"**File:** _{fname}_"
            )

    st.divider()


def insightface_ui():
    """
    InsightFace-only recognition tab with two-threshold human-in-the-loop
    annotation workflow.

    Three zones based on cosine similarity to the best reference match:

    1. **Identified** (similarity ≥ upper threshold):
       High-confidence matches – automatically recognised as known people.

    2. **Reference Augmentation** (lower ≤ sim < upper):
       Moderate matches – added to the reference DB under the matched label
       to improve future recognition.

    3. **Human Review** (similarity < lower threshold):
       Uncertain – queued for manual annotation.
    """
    logger.info("=" * 70)
    logger.info("[InsightFace Tab] Rendering InsightFace UI")
    logger.info("=" * 70)
    st.header("InsightFace Recognition & Human Feedback")

    # ── Configuration ────────────────────────────────────────────────
    with st.expander("⚙️ Configuration", expanded=True):
        cfg_col1, cfg_col2, cfg_col3 = st.columns(3)

        with cfg_col1:
            if_model = st.selectbox(
                "InsightFace Model",
                options=["buffalo_l", "buffalo_m", "buffalo_s", "antelopev2"],
                index=0,
                key="if_model",
                help=(
                    "buffalo_l: SCRFD-10GF (highest accuracy)\n"
                    "buffalo_m: SCRFD-2.5GF (balanced)\n"
                    "buffalo_s: lightweight (fastest)\n"
                    "antelopev2: alternative high-end"
                ),
            )

        with cfg_col2:
            upper_threshold = st.slider(
                "Upper Threshold (auto-identify)",
                min_value=0.0,
                max_value=1.0,
                value=0.50,
                step=0.05,
                key="if_upper",
                help=(
                    "Faces with similarity ≥ this value are treated as confident "
                    "identifications and automatically added as recognised people."
                ),
            )

        with cfg_col3:
            lower_threshold = st.slider(
                "Lower Threshold (augmentation floor)",
                min_value=0.0,
                max_value=1.0,
                value=0.25,
                step=0.05,
                key="if_lower",
                help=(
                    "Faces with similarity between the lower and upper thresholds "
                    "are added to the reference DB under their matched label.\n"
                    "Faces below this value are sent for human review."
                ),
            )

        if lower_threshold >= upper_threshold:
            st.warning("Lower threshold should be less than the upper threshold.")

        ref_upload = st.file_uploader(
            "References JSON (defaults to data/references.json)",
            type=["json"],
            key="if_ref_upload",
        )

        # ── Augmentation / review toggles ──────────────────────────
        tog_col1, tog_col2 = st.columns(2)
        with tog_col1:
            enable_augment = st.checkbox(
                "Enable auto-augmentation",
                value=False,
                key="if_enable_augment",
                help=(
                    "When enabled, faces in the augmentation range will be "
                    "added to the reference database under their matched "
                    "label when you click 'Apply Changes'."
                ),
            )
        with tog_col2:
            enable_review = st.checkbox(
                "Enable human review",
                value=False,
                key="if_enable_review",
                help=(
                    "When enabled, low-confidence faces are shown for manual "
                    "annotation. You can confirm, reassign or create new "
                    "identities."
                ),
            )

        if enable_review:
            review_top_k = st.number_input(
                "Review Top-K (max faces for human review)",
                min_value=1,
                max_value=500,
                value=10,
                step=1,
                key="if_review_top_k",
                help=(
                    "Only the K faces closest to (but below) the lower "
                    "threshold are shown for review. This keeps the review "
                    "queue manageable by focusing on the most promising cases."
                ),
            )
        else:
            review_top_k = 10  # default, unused when review is disabled

    # ── Mode selector ────────────────────────────────────────────────
    mode = st.radio(
        "Mode",
        ["Classify Images", "Reference Augmentation", "Evaluate Test Set"],
        horizontal=True,
        key="if_mode",
    )
    logger.info(f"[InsightFace Tab] Mode selected: {mode}")
    logger.info(
        f"[InsightFace Tab] Configuration: model={if_model}, "
        f"upper_threshold={upper_threshold}, lower_threshold={lower_threshold}, "
        f"enable_augment={enable_augment}, enable_review={enable_review}, "
        f"review_top_k={review_top_k}"
    )

    # ── Model initialisation ─────────────────────────────────────────
    # Only load the heavy ONNX model when the user explicitly requests it.
    # Once loaded, it stays cached (st.cache_resource) across all reruns
    # so changing thresholds / uploading files does NOT retrigger loading.
    engine = None
    model_ready = st.session_state.get("if_model_ready", False)
    loaded_model = st.session_state.get("if_loaded_model", None)

    # If the user switched to a different model, mark as not ready
    if loaded_model is not None and loaded_model != if_model:
        model_ready = False
        st.session_state["if_model_ready"] = False

    if model_ready and loaded_model == if_model:
        engine = _get_insightface_engine(if_model)
        logger.info(f"[InsightFace Tab] Engine already loaded: model={if_model}")
    else:
        logger.info(f"[InsightFace Tab] Engine not yet loaded. Waiting for user to initialize.")
        init_col1, init_col2 = st.columns([1, 3])
        with init_col1:
            if st.button("🚀 Initialize Model", key="if_init_btn"):
                logger.info(f"[InsightFace Tab] User clicked Initialize Model: {if_model}")
                with st.spinner(f"Loading InsightFace ({if_model})… this only happens once."):
                    engine = _get_insightface_engine(if_model)
                logger.info(f"[InsightFace Tab] Model initialized successfully: {if_model}")
                st.session_state["if_model_ready"] = True
                st.session_state["if_loaded_model"] = if_model
                st.rerun()
        with init_col2:
            st.info(
                f"Click **Initialize Model** to load **{if_model}**. "
                "The model stays cached — changing thresholds won't reload it."
            )
        return  # nothing else to show until the model is loaded

    # ── Reload references if user uploaded a custom file ─────────────
    if ref_upload is not None:
        logger.info("[InsightFace Tab] User uploaded custom references JSON — reloading references")
        with tempfile.TemporaryDirectory() as td:
            ref_path = os.path.join(td, "references.json")
            with open(ref_path, "wb") as f:
                f.write(ref_upload.read())
            engine.load_references(ref_path, force_extract=True)
        logger.info("[InsightFace Tab] Custom references loaded successfully")

    # ── Reference DB summary ─────────────────────────────────────────
    summary = engine.get_reference_summary()
    logger.info(
        f"[InsightFace Tab] Reference DB summary: {len(summary)} identities, "
        f"{sum(summary.values())} total embeddings"
    )
    for name, count in summary.items():
        logger.debug(f"[InsightFace Tab]   - {name}: {count} embedding(s)")
    with st.expander("📚 Reference Database", expanded=False):
        if summary:
            for name, count in summary.items():
                st.write(f"- **{name}**: {count} embedding(s)")
        else:
            st.info("No reference identities loaded.")

    # ==================================================================
    # MODE 1: Classify Images
    # ==================================================================
    if mode == "Classify Images":
        uploaded_files = st.file_uploader(
            "Upload image(s)",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key="if_upload_images",
        )

        if st.button("Classify & Analyse", key="if_classify_btn"):
            if not uploaded_files:
                st.error("Please upload at least one image.")
                return

            logger.info("=" * 60)
            logger.info("[InsightFace Classify] Starting classification workflow")
            logger.info("=" * 60)

            session = SessionLogger(mode="classify")
            session.log_config(
                model_name=if_model,
                upper_threshold=upper_threshold,
                lower_threshold=lower_threshold,
                enable_augment=enable_augment,
                enable_review=enable_review,
                review_top_k=review_top_k,
            )
            session.log_model(
                model_name=engine.model_name,
                det_size=engine.det_size,
                providers=[p for p in (engine.app.session_options.providers if hasattr(engine.app, 'session_options') else [])],
                cache_dir=str(engine.cache_dir),
            )
            session.log_reference_db(
                summary=engine.get_reference_summary(),
                source="in-memory (classify mode)",
            )
            session.data["augmentation"]["enabled"] = enable_augment
            session.data["human_review"]["enabled"] = enable_review

            logger.info(f"[InsightFace Classify] Files uploaded: {len(uploaded_files)}")
            for uf in uploaded_files:
                logger.info(f"[InsightFace Classify]   - {uf.name} ({uf.size} bytes)")

            ensure_dirs()

            # Save uploads to a temp dir and classify
            with tempfile.TemporaryDirectory() as td:
                paths = []
                name_map = {}
                for uf in uploaded_files:
                    p = os.path.join(td, uf.name)
                    with open(p, "wb") as f:
                        f.write(uf.read())
                    paths.append(p)
                    name_map[p] = uf.name

                logger.info(f"[InsightFace Classify] Running two-threshold classification on {len(paths)} image(s)...")
                session.start_timer("classification")
                with st.spinner("Running InsightFace classification…"):
                    categorised = engine.classify_with_thresholds(
                        paths,
                        upper_threshold=upper_threshold,
                        lower_threshold=lower_threshold,
                        review_top_k=review_top_k,
                    )
                _cls_time = session.stop_timer("classification")
                logger.info(f"[InsightFace Classify] Classification completed in {_cls_time:.3f}s")

                # Attach original filename for display
                for cat in categorised.values():
                    for face in cat:
                        face["original_filename"] = name_map.get(
                            face["image_path"], face["image_path"]
                        )

                # Log categorisation results to session
                session.log_categorisation(categorised)

                # Log per-image details
                for path in paths:
                    _img_faces = []
                    for cat in categorised.values():
                        for f in cat:
                            if f.get("image_path") == path:
                                _img_faces.append(f)
                    session.log_image_result(
                        image_path=path,
                        faces_detected=len(_img_faces),
                        face_details=_img_faces,
                    )

                st.session_state["if_results"] = categorised
                st.session_state["if_review_decisions"] = {}

            # Save session log
            _session_path = session.save()
            logger.info(f"[InsightFace Classify] Session log saved: {_session_path}")
            st.session_state["if_session_log_path"] = _session_path

            logger.info("=" * 60)
            logger.info("[InsightFace Classify] Classification workflow complete")
            logger.info("=" * 60)

        # ── Display results ──────────────────────────────────────────
        categorised = st.session_state.get("if_results")
        if categorised is None:
            return

        _render_categorised_results(
            categorised, engine, prefix="if",
            enable_augment=enable_augment,
            enable_review=enable_review,
            upper_threshold=upper_threshold,
            lower_threshold=lower_threshold,
            testset_name=None,
        )

    # ==================================================================
    # MODE 2: Reference Augmentation
    # ==================================================================
    elif mode == "Reference Augmentation":
        st.subheader("Reference Augmentation")
        st.caption(
            "Run images through the two-threshold system to augment the "
            "reference database. No evaluation metrics are computed — this "
            "mode is purely for improving the reference DB."
        )

        # ── Reference DB selection for augmentation ──────────────────
        versioned_dbs = engine.list_versioned_dbs()
        aug_db_options = [
            "Current (in-memory)",
            "Base (original 4 identities)",
        ] + versioned_dbs

        selected_aug_db = st.selectbox(
            "Reference database to augment from",
            options=aug_db_options,
            index=0,
            key="if_aug_db_select",
            help=(
                "Choose the starting reference database for augmentation.\n"
                "• 'Current (in-memory)' – the active DB (may already include augmentations).\n"
                "• 'Base (original 4 identities)' – re-extracts from "
                "data/references.json (clean baseline).\n"
                "• Versioned snapshots include thresholds, testset, and timestamp."
            ),
        )

        if selected_aug_db == "Base (original 4 identities)":
            if st.button("📂 Load base reference DB", key="if_aug_load_base_btn"):
                logger.info("[InsightFace Augment] Loading base reference DB (original identities)")
                with st.spinner("Extracting base references…"):
                    engine.reset_to_base_references()
                logger.info(
                    f"[InsightFace Augment] Base DB loaded: "
                    f"{sum(len(v) for v in engine.reference_db.values())} embeddings "
                    f"across {len(engine.reference_db)} identities"
                )
                st.success(
                    f"Loaded base DB — "
                    f"{sum(len(v) for v in engine.reference_db.values())} embeddings "
                    f"across {len(engine.reference_db)} identities"
                )
                st.rerun()
        elif selected_aug_db != "Current (in-memory)":
            if st.button("📂 Load selected reference DB", key="if_aug_load_db_btn"):
                logger.info(f"[InsightFace Augment] Loading versioned reference DB: {selected_aug_db}")
                engine.load_reference_db_from_file(selected_aug_db)
                st.success(
                    f"Loaded: {os.path.basename(selected_aug_db)} — "
                    f"{sum(len(v) for v in engine.reference_db.values())} embeddings"
                )
                st.rerun()

        # Show the DB that will be used
        aug_db_summary = engine.get_reference_summary()
        with st.expander("📚 Active Reference DB for Augmentation", expanded=False):
            if aug_db_summary:
                for name, count in aug_db_summary.items():
                    st.write(f"- **{name}**: {count} embedding(s)")
            else:
                st.info("No reference identities loaded.")

        aug_source = st.radio(
            "Image source",
            ["Upload images", "Use test set file"],
            horizontal=True,
            key="if_aug_source",
        )

        if aug_source == "Upload images":
            aug_files = st.file_uploader(
                "Upload image(s) for augmentation",
                type=["jpg", "jpeg", "png"],
                accept_multiple_files=True,
                key="if_aug_upload_images",
            )

            if st.button("Run Augmentation Analysis", key="if_aug_classify_btn"):
                if not aug_files:
                    st.error("Please upload at least one image.")
                    return

                logger.info("=" * 60)
                logger.info("[InsightFace Augment] Starting augmentation analysis (uploaded images)")
                logger.info("=" * 60)

                session = SessionLogger(mode="augment_upload")
                session.log_config(
                    model_name=if_model,
                    upper_threshold=upper_threshold,
                    lower_threshold=lower_threshold,
                    enable_augment=enable_augment,
                    enable_review=enable_review,
                    review_top_k=review_top_k,
                )
                session.log_model(
                    model_name=engine.model_name,
                    det_size=engine.det_size,
                    cache_dir=str(engine.cache_dir),
                )
                session.log_reference_db(
                    summary=engine.get_reference_summary(),
                    source=f"augmentation starting DB ({selected_aug_db})",
                )
                session.data["augmentation"]["enabled"] = enable_augment
                session.data["human_review"]["enabled"] = enable_review

                logger.info(f"[InsightFace Augment] Files uploaded for augmentation: {len(aug_files)}")
                for uf in aug_files:
                    logger.info(f"[InsightFace Augment]   - {uf.name} ({uf.size} bytes)")

                ensure_dirs()
                with tempfile.TemporaryDirectory() as td:
                    paths = []
                    name_map = {}
                    for uf in aug_files:
                        p = os.path.join(td, uf.name)
                        with open(p, "wb") as f:
                            f.write(uf.read())
                        paths.append(p)
                        name_map[p] = uf.name

                    logger.info(f"[InsightFace Augment] Running two-threshold analysis on {len(paths)} image(s)...")
                    session.start_timer("augmentation_analysis")
                    with st.spinner("Analysing images for augmentation…"):
                        categorised = engine.classify_with_thresholds(
                            paths,
                            upper_threshold=upper_threshold,
                            lower_threshold=lower_threshold,
                            review_top_k=review_top_k,
                        )
                    _aug_time = session.stop_timer("augmentation_analysis")
                    logger.info(f"[InsightFace Augment] Analysis completed in {_aug_time:.3f}s")

                    for cat in categorised.values():
                        for face in cat:
                            face["original_filename"] = name_map.get(
                                face["image_path"], face["image_path"]
                            )

                    session.log_categorisation(categorised)

                st.session_state["if_aug_results"] = categorised
                st.session_state["if_aug_review_decisions"] = {}

                _session_path = session.save()
                logger.info(f"[InsightFace Augment] Session log saved: {_session_path}")
                st.session_state["if_aug_session_log_path"] = _session_path

                logger.info("=" * 60)
                logger.info("[InsightFace Augment] Augmentation analysis complete (uploaded images)")
                logger.info("=" * 60)

        else:
            # Test set source
            import glob

            available_testsets = sorted(
                p
                for p in glob.glob(os.path.join("testsets", "*.json"))
                if not os.path.basename(p).startswith("test_config")
            )

            ts_col1, ts_col2 = st.columns([2, 1])
            with ts_col1:
                selected_aug_testset = st.selectbox(
                    "Browse available test sets",
                    options=available_testsets if available_testsets else ["(no test sets found)"],
                    index=(
                        available_testsets.index("testsets/four-people-trainset-sample.json")
                        if "testsets/four-people-trainset-sample.json" in available_testsets
                        else 0
                    ),
                    key="if_aug_testset_browse",
                    help="Select a test set JSON file from the testsets/ directory.",
                )
            with ts_col2:
                ts_aug_upload = st.file_uploader(
                    "Or upload a test set JSON",
                    type=["json"],
                    key="if_aug_testset_upload",
                )

            if st.button("Run Augmentation Analysis", key="if_aug_ts_btn"):
                logger.info("=" * 60)
                logger.info("[InsightFace Augment] Starting augmentation analysis (test set)")
                logger.info("=" * 60)

                session = SessionLogger(mode="augment_testset")
                session.log_config(
                    model_name=if_model,
                    upper_threshold=upper_threshold,
                    lower_threshold=lower_threshold,
                    enable_augment=enable_augment,
                    enable_review=enable_review,
                    review_top_k=review_top_k,
                )
                session.log_model(
                    model_name=engine.model_name,
                    det_size=engine.det_size,
                    cache_dir=str(engine.cache_dir),
                )
                session.log_reference_db(
                    summary=engine.get_reference_summary(),
                    source=f"augmentation starting DB ({selected_aug_db})",
                )
                session.data["augmentation"]["enabled"] = enable_augment
                session.data["human_review"]["enabled"] = enable_review

                ensure_dirs()
                with tempfile.TemporaryDirectory() as td:
                    if ts_aug_upload:
                        ts_path = os.path.join(td, "testset.json")
                        with open(ts_path, "wb") as f:
                            f.write(ts_aug_upload.read())
                        ts_display_name = ts_aug_upload.name
                    else:
                        ts_path = selected_aug_testset
                        ts_display_name = os.path.basename(selected_aug_testset)

                    logger.info(f"[InsightFace Augment] Test set: {ts_display_name} ({ts_path})")

                    # Log test set details
                    import json as _json
                    try:
                        with open(ts_path, "r") as _tf:
                            _ts_items = _json.load(_tf)
                        session.log_test_set(
                            path=ts_path,
                            num_images=len(_ts_items),
                            raw_items=_ts_items,
                        )
                    except Exception as _e:
                        logger.warning(f"[InsightFace Augment] Could not parse test set for logging: {_e}")

                    session.start_timer("augmentation_analysis")
                    with st.spinner("Analysing test set images for augmentation…"):
                        categorised = engine.augment_from_testset(
                            ts_path,
                            upper_threshold=upper_threshold,
                            lower_threshold=lower_threshold,
                            review_top_k=review_top_k,
                        )
                    _aug_time = session.stop_timer("augmentation_analysis")
                    logger.info(f"[InsightFace Augment] Test set analysis completed in {_aug_time:.3f}s")

                    session.log_categorisation(categorised)

                st.session_state["if_aug_results"] = categorised
                st.session_state["if_aug_review_decisions"] = {}
                st.session_state["if_aug_testset_name"] = ts_display_name

                _session_path = session.save()
                logger.info(f"[InsightFace Augment] Session log saved: {_session_path}")
                st.session_state["if_aug_session_log_path"] = _session_path

                logger.info("=" * 60)
                logger.info("[InsightFace Augment] Augmentation analysis complete (test set)")
                logger.info("=" * 60)

        # ── Display augmentation results ─────────────────────────────
        aug_categorised = st.session_state.get("if_aug_results")
        if aug_categorised is not None:
            testset_name = st.session_state.get("if_aug_testset_name")
            _render_categorised_results(
                aug_categorised, engine, prefix="if_aug",
                enable_augment=enable_augment,
                enable_review=enable_review,
                upper_threshold=upper_threshold,
                lower_threshold=lower_threshold,
                testset_name=testset_name,
            )

    # ==================================================================
    # MODE 3: Evaluate Test Set
    # ==================================================================
    else:
        st.subheader("Evaluate Test Set")
        st.caption(
            "Run evaluation against a test set and compute metrics. "
            "Choose which reference database to use for evaluation."
        )

        # ── Reference DB selection for evaluation ────────────────────
        versioned_dbs = engine.list_versioned_dbs()
        db_options = [
            "Current (in-memory)",
            "Base (original 4 identities)",
        ] + versioned_dbs

        selected_db = st.selectbox(
            "Reference database to evaluate with",
            options=db_options,
            index=0,
            key="if_eval_db_select",
            help=(
                "Choose the reference database to use.\n"
                "• 'Current (in-memory)' – the active DB (may include augmentations).\n"
                "• 'Base (original 4 identities)' – re-extracts from "
                "data/references.json (clean baseline).\n"
                "• Versioned snapshots include thresholds, testset, and timestamp."
            ),
        )

        if selected_db == "Base (original 4 identities)":
            if st.button("📂 Load base reference DB", key="if_eval_load_base_btn"):
                logger.info("[InsightFace Evaluate] Loading base reference DB (original identities)")
                with st.spinner("Extracting base references…"):
                    engine.reset_to_base_references()
                logger.info(
                    f"[InsightFace Evaluate] Base DB loaded: "
                    f"{sum(len(v) for v in engine.reference_db.values())} embeddings "
                    f"across {len(engine.reference_db)} identities"
                )
                st.success(
                    f"Loaded base DB — "
                    f"{sum(len(v) for v in engine.reference_db.values())} embeddings "
                    f"across {len(engine.reference_db)} identities"
                )
                st.rerun()
        elif selected_db != "Current (in-memory)":
            if st.button("📂 Load selected reference DB", key="if_eval_load_db_btn"):
                logger.info(f"[InsightFace Evaluate] Loading versioned reference DB: {selected_db}")
                engine.load_reference_db_from_file(selected_db)
                st.success(
                    f"Loaded: {os.path.basename(selected_db)} — "
                    f"{sum(len(v) for v in engine.reference_db.values())} embeddings"
                )
                st.rerun()

        # Show the DB that will be used
        eval_summary = engine.get_reference_summary()
        with st.expander("📚 Active Reference DB for Evaluation", expanded=False):
            if eval_summary:
                for name, count in eval_summary.items():
                    st.write(f"- **{name}**: {count} embedding(s)")
            else:
                st.info("No reference identities loaded.")

        # ── Test set selection ───────────────────────────────────────
        import glob

        available_testsets = sorted(
            p
            for p in glob.glob(os.path.join("testsets", "*.json"))
            if not os.path.basename(p).startswith("test_config")
        )

        ts_col1, ts_col2 = st.columns([2, 1])
        with ts_col1:
            selected_testset = st.selectbox(
                "Browse available test sets",
                options=available_testsets if available_testsets else ["(no test sets found)"],
                index=(
                    available_testsets.index("testsets/four-people-trainset-sample.json")
                    if "testsets/four-people-trainset-sample.json" in available_testsets
                    else 0
                ),
                key="if_eval_testset_browse",
                help="Select a test set JSON file from the testsets/ directory.",
            )
        with ts_col2:
            ts_upload = st.file_uploader(
                "Or upload a test set JSON",
                type=["json"],
                key="if_eval_testset_upload",
            )

        ident_thresh = st.slider(
            "Identification Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.30,
            step=0.05,
            key="if_ident_thresh",
            help="Minimum similarity for a face to be labelled as a known identity during evaluation.",
        )

        # Advanced metric parameters
        adv_col1, adv_col2, adv_col3 = st.columns(3)
        with adv_col1:
            f_beta_param = st.number_input(
                "F-beta (β)",
                min_value=0.1,
                max_value=5.0,
                value=1.0,
                step=0.1,
                key="if_fbeta",
                help="β parameter for F-beta score. β=1 → F1, β<1 weights precision more, β>1 weights recall more.",
            )
        with adv_col2:
            fixed_recall = st.number_input(
                "Fixed Recall Level",
                min_value=0.0,
                max_value=1.0,
                value=0.80,
                step=0.05,
                key="if_fixed_recall",
                help="Recall level at which to report Precision@Recall.",
            )
        with adv_col3:
            fixed_precision = st.number_input(
                "Fixed Precision Level",
                min_value=0.0,
                max_value=1.0,
                value=0.80,
                step=0.05,
                key="if_fixed_precision",
                help="Precision level at which to report Recall@Precision.",
            )

        if st.button("Run Evaluation", key="if_eval_btn"):
            logger.info("=" * 70)
            logger.info("[InsightFace Evaluate] Starting evaluation workflow")
            logger.info("=" * 70)

            session = SessionLogger(mode="evaluate")
            session.log_config(
                model_name=if_model,
                upper_threshold=upper_threshold,
                lower_threshold=lower_threshold,
                enable_augment=enable_augment,
                enable_review=enable_review,
                review_top_k=review_top_k,
                identification_threshold=ident_thresh,
                f_beta=f_beta_param,
                fixed_recall=fixed_recall,
                fixed_precision=fixed_precision,
            )
            session.log_model(
                model_name=engine.model_name,
                det_size=engine.det_size,
                cache_dir=str(engine.cache_dir),
            )
            session.log_reference_db(
                summary=engine.get_reference_summary(),
                source=f"evaluation DB ({selected_db})",
            )

            ensure_dirs()

            with tempfile.TemporaryDirectory() as td:
                if ts_upload:
                    ts_path = os.path.join(td, "testset.json")
                    with open(ts_path, "wb") as f:
                        f.write(ts_upload.read())
                    _ts_display = ts_upload.name
                else:
                    ts_path = selected_testset
                    _ts_display = os.path.basename(selected_testset)

                logger.info(f"[InsightFace Evaluate] Test set: {_ts_display} ({ts_path})")
                logger.info(f"[InsightFace Evaluate] Identification threshold: {ident_thresh}")
                logger.info(f"[InsightFace Evaluate] F-beta={f_beta_param}, fixed_recall={fixed_recall}, fixed_precision={fixed_precision}")
                logger.info(f"[InsightFace Evaluate] Reference DB: {selected_db}")

                # Log test set details
                import json as _json
                try:
                    with open(ts_path, "r") as _tf:
                        _ts_items = _json.load(_tf)
                    session.log_test_set(
                        path=ts_path,
                        num_images=len(_ts_items),
                        raw_items=_ts_items,
                    )
                    logger.info(f"[InsightFace Evaluate] Test set loaded: {len(_ts_items)} images")
                except Exception as _e:
                    logger.warning(f"[InsightFace Evaluate] Could not parse test set for logging: {_e}")

                session.start_timer("evaluation")
                with st.spinner("Evaluating test set…"):
                    eval_results = engine.evaluate_testset(
                        ts_path,
                        identification_threshold=ident_thresh,
                    )
                _eval_time = session.stop_timer("evaluation")
                logger.info(f"[InsightFace Evaluate] Evaluation completed in {_eval_time:.3f}s")

            # Log basic metrics
            logger.info("[InsightFace Evaluate] Base metrics:")
            for k, v in eval_results.get("metrics", {}).items():
                logger.info(f"[InsightFace Evaluate]   {k}: {v}")
            session.log_evaluation_metrics(eval_results.get("metrics", {}))

            # Log per-image predictions vs ground truth
            session.log_predictions_vs_ground_truth(
                image_paths=eval_results["image_paths"],
                ground_truth=eval_results["ground_truth"],
                predictions=eval_results["predictions"],
            )

            # Log per-image result details
            _face_idx_map: dict = {}
            for face in eval_results.get("all_faces", []):
                ip = face.get("image_path", "")
                _face_idx_map.setdefault(ip, []).append(face)
            for ip in eval_results.get("image_paths", []):
                _faces = _face_idx_map.get(ip, [])
                session.log_image_result(
                    image_path=ip,
                    faces_detected=len(_faces),
                    face_details=_faces,
                )

            # Compute advanced metrics
            logger.info("[InsightFace Evaluate] Computing advanced metrics...")
            session.start_timer("advanced_metrics")
            from src.metrics import calculate_advanced_metrics

            adv = calculate_advanced_metrics(
                ground_truth_labels=eval_results["ground_truth"],
                all_detections=eval_results["predictions"],
                per_image_scores=eval_results["per_image_scores"],
                identity_names=eval_results["identity_names"],
                f_beta=f_beta_param,
                fixed_recall_level=fixed_recall,
                fixed_precision_level=fixed_precision,
            )
            _adv_time = session.stop_timer("advanced_metrics")
            logger.info(f"[InsightFace Evaluate] Advanced metrics computed in {_adv_time:.3f}s")

            # Log advanced metrics
            session.log_advanced_metrics(adv)

            st.session_state["if_eval_metrics"] = eval_results["metrics"]
            st.session_state["if_eval_advanced"] = adv
            st.session_state["if_eval_details"] = {
                "image_paths": eval_results["image_paths"],
                "ground_truth": eval_results["ground_truth"],
                "predictions": eval_results["predictions"],
            }

            # Store lightweight face metadata for per-class sample gallery
            face_samples = []
            for f in eval_results.get("all_faces", []):
                face_samples.append({
                    "image_path": f.get("image_path", ""),
                    "bbox_xyxy": tuple(int(v) for v in f["bbox_xyxy"]) if "bbox_xyxy" in f else None,
                    "similarity": float(f.get("similarity", 0.0)),
                    "name": f.get("name"),
                    "predicted_label": f.get("predicted_label", "Unknown"),
                    "ground_truth": f.get("ground_truth", []),
                    "original_filename": f.get("original_filename", ""),
                })
            st.session_state["if_eval_face_samples"] = face_samples

            # Save session log
            _session_path = session.save()
            logger.info(f"[InsightFace Evaluate] Session log saved: {_session_path}")
            st.session_state["if_eval_session_log_path"] = _session_path

            logger.info("=" * 70)
            logger.info("[InsightFace Evaluate] Evaluation workflow complete")
            logger.info("=" * 70)

        # ── Display evaluation results ───────────────────────────────
        adv = st.session_state.get("if_eval_advanced")
        if adv is not None:
            st.subheader("Evaluation Results")

            # --- Scalar Metrics ---
            scalar = adv.get("scalar_metrics", {})
            st.markdown("#### Aggregate Metrics")
            # Group scalars into a nice 3-column layout
            scalar_items = list(scalar.items())
            cols_per_row = 3
            for row_start in range(0, len(scalar_items), cols_per_row):
                row_items = scalar_items[row_start : row_start + cols_per_row]
                cols = st.columns(len(row_items))
                for col, (name, val) in zip(cols, row_items):
                    with col:
                        display_name = name.replace("_", " ").title()
                        # Keep special characters like β, @, = intact
                        for token in ["β", "@", "="]:
                            display_name = display_name.replace(
                                token.title() if token.isalpha() else token, token
                            )
                        if isinstance(val, float):
                            st.metric(label=display_name, value=f"{val:.4f}")
                        else:
                            st.metric(label=display_name, value=str(val))

            # --- Per-Class Metrics Table ---
            per_class = adv.get("per_class_metrics", {})
            if per_class:
                st.markdown("#### Per-Class Metrics")
                import pandas as pd

                pc_rows = []
                for cls_name, m in per_class.items():
                    pc_rows.append(
                        {
                            "Class": cls_name,
                            "Precision": f"{m['precision']:.4f}",
                            "Recall": f"{m['recall']:.4f}",
                            "F-beta": f"{m['f_beta']:.4f}",
                            "Accuracy": f"{m['accuracy']:.4f}",
                            "PR AUC": f"{m['pr_auc']:.4f}",
                            "ROC AUC": f"{m['roc_auc']:.4f}",
                            "P@R": f"{m['precision_at_fixed_recall']:.4f}",
                            "R@P": f"{m['recall_at_fixed_precision']:.4f}",
                            "Support": m["support"],
                            "TP": m["tp"],
                            "FP": m["fp"],
                            "FN": m["fn"],
                            "TN": m["tn"],
                        }
                    )
                st.dataframe(
                    pd.DataFrame(pc_rows),
                    width="stretch",
                    hide_index=True,
                )

            # --- Curves ---
            curve_data = adv.get("curve_data", {})
            if curve_data:
                from src.metrics import plot_roc_curves_figure, plot_pr_curves_figure

                st.markdown("#### ROC Curve")
                roc_fig = plot_roc_curves_figure(curve_data)
                st.pyplot(roc_fig)
                plt.close(roc_fig)

                st.markdown("#### Precision–Recall Curve")
                pr_fig = plot_pr_curves_figure(curve_data)
                st.pyplot(pr_fig)
                plt.close(pr_fig)

            # --- Per-class sample gallery ---
            face_samples = st.session_state.get("if_eval_face_samples", [])
            if face_samples and per_class:
                st.markdown("#### Per-Class Sample Gallery")
                st.caption(
                    "For each class: 5 highest-similarity faces, 5 near the median, "
                    "and 5 lowest-similarity faces still labelled as that class."
                )
                _render_per_class_gallery(face_samples, sorted(per_class.keys()))

            # --- Misclassification analysis ---
            if face_samples and per_class:
                st.markdown("#### Misclassification Analysis")
                st.caption(
                    "Per-class breakdown of errors: faces wrongly assigned a label "
                    "(False Positives) and images where a label was missed "
                    "(False Negatives). Up to 10 samples per error type per class."
                )
                _render_misclassification_gallery(
                    face_samples,
                    sorted(per_class.keys()),
                    st.session_state.get("if_eval_details", {}),
                )

            # --- Per-image predictions ---
            details = st.session_state.get("if_eval_details", {})
            with st.expander("Per-image predictions", expanded=False):
                import pandas as pd

                rows = []
                for path, gt, pred in zip(
                    details.get("image_paths", []),
                    details.get("ground_truth", []),
                    details.get("predictions", []),
                ):
                    rows.append(
                        {
                            "Image": os.path.basename(path),
                            "Ground Truth": ", ".join(gt) if gt else "(none)",
                            "Predictions": ", ".join(pred) if pred else "(none)",
                        }
                    )
                st.dataframe(pd.DataFrame(rows), width="stretch")


def main():
    logger.info("Starting Streamlit app...")
    st.set_page_config(page_title="Image Classification App", layout="wide")
    st.title("Image Classification App")
    logger.info("App interface loaded successfully")

    tabs = st.tabs(["Classify", "Testset", "InsightFace"])
    with tabs[0]:
        classify_ui()
    with tabs[1]:
        testset_ui()
    with tabs[2]:
        insightface_ui()


if __name__ == "__main__":
    main()
