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
    classifier_type = mo.ui.text(
        value="face_recognition_cnn",
        label="Classifier Type (e.g., face_recognition_cnn, face_recognition_hog, vit_b32)"
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

            classifier = get_classifier(
                classifier_type.value,
                celebrity_data
            )

            if classifier is None:
                logger.error(f"Could not initialize classifier: {classifier_type.value}. Check logs for details.")
                mo.output.append(mo.md(f"### <font color='red'>Error</font>\nCould not initialize classifier: {classifier_type.value}. Check logs for details."))
                return {}

            output_by_path = classifier.classify_images(image_paths)

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
        for path, labels in op.items():
            mo.output.append(mo.md(f"**{path}**"))
            for label in labels:
                mo.output.append(mo.md(f"- {label}"))
    return


if __name__ == "__main__":
    app.run()
