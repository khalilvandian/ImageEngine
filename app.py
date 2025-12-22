import marimo

__generated_with = "0.17.4"
app = marimo.App()


@app.cell
def _(mo):
    mo.md("""# Image Classification App""")
    return


@app.cell
def _():
    import marimo as mo
    import os
    import datetime
    from classification import get_classifier, load_celebrities_from_json
    from image_utils import draw_bounding_boxes
    import tempfile
    from logging_utils import setup_logger
    return (
        get_classifier,
        load_celebrities_from_json,
        mo,
        os,
        setup_logger,
        tempfile,
    )


@app.cell
def _(setup_logger):
    logger = setup_logger()
    return (logger,)


@app.cell
def _(os):
    LOG_DIR = "logs"
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    return


@app.cell
def _(mo):
    input_type = mo.ui.dropdown(
        ["File", "Multiple Files"],
        value="File",
        label="Select Input Type"
    )
    return (input_type,)


@app.cell
def _(mo):
    classifier_type = mo.ui.dropdown(
        options=["face_recognition_cnn", "face_recognition_hog", "vit_b32"],
        value="face_recognition_cnn",
        label="Select Classifier Type"
    )
    celebrities_json_path = mo.ui.text(
        value="celebrities.json",
        label="Path to Celebrities JSON File"
    )
    return celebrities_json_path, classifier_type


@app.cell
def _(input_type, mo):
    file_selector = mo.ui.file(
        kind="button",
        multiple=(input_type.value == "Multiple Files"),
        label="Select file(s)"
    )
    return (file_selector,)


@app.cell
def _(
    celebrities_json_path,
    classifier_type,
    datetime,
    draw_bounding_boxes,
    file_selector,
    get_classifier,
    input_type,
    load_celebrities_from_json,
    logger,
    mo,
    os,
    tempfile,
):
    def classify():
        logger.info("Starting classification process.")
        logger.info(f"Input Type: {input_type.value}")
        logger.info(f"Classifier Type: {classifier_type.value}")
        logger.info(f"Celebrities JSON Path: {celebrities_json_path.value}")

        celebrity_data = load_celebrities_from_json(celebrities_json_path.value)
        if not celebrity_data:
            logger.error("No celebrity data loaded from JSON. Check path and content.")
            mo.output.append(mo.md("### <font color='red'>Error</font>\nNo celebrity data loaded from JSON. Check path and content."))
            return {}

        raw_results = file_selector.value

        if not raw_results:
            logger.error("No files selected.")
            mo.output.append(mo.md("### <font color='red'>Error</font>\nNo files selected."))
            return {}

        if not isinstance(raw_results, (list, tuple)):
            raw_results = [raw_results]

        classifier = get_classifier(
            classifier_type.value,
            celebrity_data
        )

        if classifier is None:
            logger.error(f"Could not initialize classifier: {classifier_type.value}. Check logs for details.")
            mo.output.append(mo.md(f"### <font color='red'>Error</font>\nCould not initialize classifier: {classifier_type.value}. Check logs for details."))
            return {}

        # Create a unique output directory for this experiment
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_folder_name = f"{timestamp}_{classifier.name}_tolerance_{getattr(classifier, 'tolerance', 'N-A')}_threshold_{getattr(classifier, 'threshold', 'N-A')}"
        output_dir = os.path.join("image_outputs", experiment_folder_name)
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Output images will be saved to: {output_dir}")

        # Create temporary files to get paths, as the model expects paths.
        with tempfile.TemporaryDirectory() as tempdir:
            image_paths = []
            path_to_name = {}

            logger.info("Saving uploaded images to temporary directory.")
            for result in raw_results:
                temp_path = os.path.join(tempdir, result.name)
                with open(temp_path, "wb") as temp_f:
                    temp_f.write(result.contents)

                image_paths.append(temp_path)
                path_to_name[temp_path] = result.name
                logger.info(f"{result.name} saved to temp path {temp_path}")

            logger.info(f"Temporary image paths: {image_paths}")

            output_by_path = classifier.classify_images(image_paths)

            # Draw bounding boxes and save images
            for image_path, results in output_by_path.items():
                annotated_image = draw_bounding_boxes(image_path, results)
                original_filename = path_to_name[image_path]
                output_path = os.path.join(output_dir, original_filename)
                annotated_image.save(output_path)
                logger.info(f"Saved annotated image to {output_path}")

            output_by_name = {path_to_name[path]: labels for path, labels in output_by_path.items()}

            logger.info(f"Classification output: {output_by_name}")
            mo.md("Classification complete.")

            return output_by_name

    classify_button = mo.ui.run_button(label="Classify")
    return classify, classify_button


@app.cell
def _(
    celebrities_json_path,
    classifier_type,
    classify_button,
    file_selector,
    input_type,
    mo,
):
    mo.md("## Select Input")

    mo.vstack(
        [
            input_type,
            classifier_type,
            celebrities_json_path,
            file_selector,
            classify_button
        ]
    )
    return


@app.cell
def _(classify, classify_button, mo):
    if classify_button.value:
        _output = mo.md("## Classification Results")
        op = classify()
        for path, results in op.items():
            mo.output.append(mo.md(f"**{path}**"))
            if results:
                for result in results:
                    mo.output.append(mo.md(f"- **{result['name']}** at location {result['location']}"))
            else:
                mo.output.append(mo.md("- No celebrities detected."))
    return


@app.cell
def _(mo):
    mo.md(r"""## Model Testing""")
    return


@app.cell
def _(mo):
    model_to_test = mo.ui.dropdown(
        ["face_recognition_cnn", "face_recognition_hog", "vit_b32"],
        value="face_recognition_cnn",
        label="Select Model to Test"
    )
    test_config_path = mo.ui.text(
        value="test_config.json",
        label="Path to Test Config JSON"
    )
    run_tests_button = mo.ui.run_button(label="Run Tests")
    return model_to_test, run_tests_button, test_config_path


@app.cell
def _(mo, model_to_test, run_tests_button, test_config_path):
    mo.vstack(
        [
            model_to_test,
            test_config_path,
            run_tests_button,
        ]
    )
    return


@app.cell
def _(
    celebrities_json_path,
    get_classifier,
    load_celebrities_from_json,
    logger,
    mo,
    model_to_test,
    run_tests_button,
    test_config_path,
):
    def run_tests():
        from metrics import load_test_config, load_test_set, run_classification_on_test_set, calculate_metrics, plot_confusion_matrix, plot_roc_curve, save_test_output_to_csv

        logger.info("Starting model testing process.")

        # Load test config
        test_config = load_test_config(test_config_path.value)
        if not test_config:
            logger.error("Could not load test config.")
            mo.output.append(mo.md("### <font color='red'>Error</font>\nCould not load test config."))
            return

        # Load test set
        image_paths, ground_truth_labels = load_test_set(test_config["test_set_json_path"])
        logger.info(f"Loaded {len(image_paths)} images for testing.")

        # Load celebrity data
        celebrity_data = load_celebrities_from_json(celebrities_json_path.value)
        if not celebrity_data:
            logger.error("No celebrity data loaded from JSON. Check path and content.")
            mo.output.append(mo.md("### <font color='red'>Error</font>\nNo celebrity data loaded from JSON. Check path and content."))
            return

        # Get classifier
        classifier = get_classifier(model_to_test.value, celebrity_data)
        if classifier is None:
            logger.error(f"Could not initialize classifier: {model_to_test.value}. Check logs for details.")
            mo.output.append(mo.md(f"### <font color='red'>Error</font>\nCould not initialize classifier: {model_to_test.value}. Check logs for details."))
            return

        # Run classification
        predictions = run_classification_on_test_set(classifier, image_paths, output_image_dir="image_outputs")

        # Calculate metrics
        metrics = calculate_metrics(ground_truth_labels, predictions)
        logger.info(f"Calculated metrics: {metrics}")

        # Save test output to CSV
        csv_filepath = save_test_output_to_csv(image_paths, predictions, ground_truth_labels, model_to_test.value)
        logger.info(f"Test output saved to {csv_filepath}")

        # Display metrics
        mo.output.append(mo.md("### Test Results"))
        for metric, value in metrics.items():
            mo.output.append(mo.md(f"**{metric.replace('_', ' ').title()}**: {value}"))
        
        mo.output.append(mo.md(f"**Test output saved to**: [{csv_filepath}]({csv_filepath})"))

        # Generate and display confusion matrix
        confusion_matrix_base64 = plot_confusion_matrix(ground_truth_labels, predictions)
        mo.output.append(mo.md("### Confusion Matrix"))
        mo.output.append(mo.image(src=f"data:image/png;base64,{confusion_matrix_base64}"))

        # Generate and display ROC curve
        roc_curve_base64 = plot_roc_curve(ground_truth_labels, predictions)
        mo.output.append(mo.md("### ROC Curve"))
        mo.output.append(mo.image(src=f"data:image/png;base64,{roc_curve_base64}"))

    if run_tests_button.value:
        run_tests()
    return


if __name__ == "__main__":
    app.run()
