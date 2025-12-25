import os
import datetime
import tempfile
import streamlit as st

from src.classification import get_classifier, load_celebrities_from_json
from src.image_utils import draw_bounding_boxes
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

    col1, col2 = st.columns(2)
    with col1:
        classifier_type = st.selectbox(
            "Select Classifier Type",
            options=["face_recognition_cnn", "face_recognition_hog", "vit_b32"],
            index=0,
        )
    with col2:
        celebrities_json_file = st.file_uploader(
            "Celebrities JSON (defaults to data/celebrities.json)",
            type=["json"],
            key="celebrities_json_upload_classify",
        )

    multiple = st.toggle("Upload multiple files", value=False)
    uploaded_files = st.file_uploader(
        "Select file(s)", type=["jpg", "jpeg", "png"], accept_multiple_files=multiple
    )

    if st.button("Classify"):
        ensure_dirs()
        st.info("Starting classification…")
        if not uploaded_files:
            st.error("No files selected.")
            return

        with tempfile.TemporaryDirectory() as tempdir:
            celebrities_json_path = resolve_uploaded_or_default(
                celebrities_json_file,
                default_path="data/celebrities.json",
                tempdir=tempdir,
                filename="celebrities.json",
            )

            celebrity_data = load_celebrities_from_json(celebrities_json_path)
            if not celebrity_data:
                st.error("No celebrity data loaded from JSON. Check path and content.")
                return

            classifier = get_classifier(classifier_type, celebrity_data)
            if classifier is None:
                st.error(
                    f"Could not initialize classifier: {classifier_type}. Check logs for details."
                )
                return

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_folder_name = (
                f"{timestamp}_{classifier.name}_tolerance_{getattr(classifier, 'tolerance', 'N-A')}_"
                f"threshold_{getattr(classifier, 'threshold', 'N-A')}"
            )
            output_dir = os.path.join("image_outputs", experiment_folder_name)
            os.makedirs(output_dir, exist_ok=True)
            st.write(f"Output images will be saved to: {output_dir}")

            image_paths = []
            path_to_name = {}

            for uploaded in uploaded_files if isinstance(uploaded_files, list) else [uploaded_files]:
                temp_path = os.path.join(tempdir, uploaded.name)
                with open(temp_path, "wb") as f:
                    f.write(uploaded.read())
                image_paths.append(temp_path)
                path_to_name[temp_path] = uploaded.name

            output_by_path = classifier.classify_images(image_paths)

            st.subheader("Results")
            for image_path, results in output_by_path.items():
                annotated_image = draw_bounding_boxes(image_path, results)
                original_filename = path_to_name[image_path]
                output_path = os.path.join(output_dir, original_filename)
                annotated_image.save(output_path)

                st.markdown(f"**{original_filename}**")
                if results:
                    for result in results:
                        st.write(f"- {result['name']} at location {result['location']}")
                else:
                    st.write("- No celebrities detected.")
                st.image(annotated_image, caption=f"Annotated: {original_filename}")

        st.success("Classification complete.")


def testset_ui():
    from src.metrics import (
        load_test_set,
        run_classification_on_test_set,
        calculate_metrics,
        plot_confusion_matrix,
        plot_roc_curve,
        save_test_output_to_csv,
    )

    st.header("Model Testing")

    col1, col2, col3 = st.columns(3)
    with col1:
        model_to_test = st.selectbox(
            "Select Model to Test",
            options=["face_recognition_cnn", "face_recognition_hog", "vit_b32"],
            index=0,
        )
    with col2:
        test_set_upload = st.file_uploader(
            "Test Set JSON (defaults to testsets/test_set.json)",
            type=["json"],
            key="test_set_upload",
        )
    with col3:
        celebrities_json_file = st.file_uploader(
            "Celebrities JSON (defaults to data/celebrities.json)",
            type=["json"],
            key="celebrities_json_upload_test",
        )

    if st.button("Run Tests"):
        ensure_dirs()
        st.info("Running tests… this may take a while")

        with tempfile.TemporaryDirectory() as tempdir:
            test_set_path = resolve_uploaded_or_default(
                test_set_upload,
                default_path="testsets/test_set.json",
                tempdir=tempdir,
                filename="test_set.json",
            )

            image_paths, ground_truth_labels = load_test_set(test_set_path)
            st.write(f"Loaded {len(image_paths)} images for testing.")

            celebrities_json_path = resolve_uploaded_or_default(
                celebrities_json_file,
                default_path="data/celebrities.json",
                tempdir=tempdir,
                filename="celebrities_test.json",
            )

            celebrity_data = load_celebrities_from_json(celebrities_json_path)
            if not celebrity_data:
                st.error("No celebrity data loaded from JSON. Check path and content.")
                return

            classifier = get_classifier(model_to_test, celebrity_data)
            if classifier is None:
                st.error(
                    f"Could not initialize classifier: {model_to_test}. Check logs for details."
                )
                return

            predictions = run_classification_on_test_set(
                classifier, image_paths, output_image_dir="image_outputs"
            )

            metrics = calculate_metrics(ground_truth_labels, predictions)
            st.subheader("Test Results")
            for metric, value in metrics.items():
                st.write(f"**{metric.replace('_', ' ').title()}**: {value}")

            csv_filepath = save_test_output_to_csv(
                image_paths, predictions, ground_truth_labels, model_to_test
            )
            st.write(f"Test output saved to: {csv_filepath}")

            confusion_matrix_base64 = plot_confusion_matrix(
                ground_truth_labels, predictions
            )
            st.subheader("Confusion Matrix")
            st.image(f"data:image/png;base64,{confusion_matrix_base64}")

            roc_curve_base64 = plot_roc_curve(ground_truth_labels, predictions)
            st.subheader("ROC Curve")
            st.image(f"data:image/png;base64,{roc_curve_base64}")


def main():
    st.set_page_config(page_title="Image Classification App", layout="wide")
    st.title("Image Classification App")

    tabs = st.tabs(["Classify", "Testset"])
    with tabs[0]:
        classify_ui()
    with tabs[1]:
        testset_ui()


if __name__ == "__main__":
    main()
