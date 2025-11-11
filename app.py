import marimo

__generated_with = "0.17.0"
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
    from classification import classify_images
    return classify_images, datetime, mo, os


@app.cell
def _(os):
    LOG_DIR = "logs"
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    return (LOG_DIR,)


@app.cell
def _(mo):
    input_type = mo.ui.dropdown(
        ["File", "Multiple Files"],
        value="File",
        label="Select Input Type"
    )
    return (input_type,)


@app.cell
def _(input_type, mo):
    file_selector = mo.ui.file(
        kind="button",
        multiple=(input_type.value == "Multiple Files"),
        label="Select file(s)"
    )
    return (file_selector,)


@app.cell
def _(LOG_DIR, classify_images, datetime, file_selector, input_type, mo, os):
    import tempfile
    
    def classify():
        log_filename = os.path.join(LOG_DIR, f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        with open(log_filename, "w") as f:
            f.write(f"Input Type: {input_type.value}\n")

            raw_results = file_selector.value

            if not raw_results:
                mo.output.append(mo.md("### <font color='red'>Error</font>\nNo files selected."))
                return {}

            if not isinstance(raw_results, (list, tuple)):
                raw_results = [raw_results]

            # Create temporary files to get paths, as the model expects paths.
            with tempfile.TemporaryDirectory() as tempdir:
                image_paths = []
                path_to_name = {}

                f.write("\n--- Input Images ---\n")
                for result in raw_results:
                    temp_path = os.path.join(tempdir, result.name)
                    with open(temp_path, "wb") as temp_f:
                        temp_f.write(result.contents)
                    
                    image_paths.append(temp_path)
                    path_to_name[temp_path] = result.name
                    f.write(f"{result.name} (saved to temp path {temp_path})\n")

                print("Temporary image paths:", image_paths)

                output_by_path = classify_images(image_paths)
                
                output_by_name = {path_to_name[path]: labels for path, labels in output_by_path.items()}

                f.write("\n--- Classification Output ---\n")
                for name, labels in output_by_name.items():
                    f.write(f"{name}: {labels}\n")

                mo.md(f"Classification complete. Log saved to: {log_filename}")

                return output_by_name

    classify_button = mo.ui.run_button(label="Classify")
    return classify, classify_button


@app.cell
def _(classify_button, file_selector, input_type, mo):
    mo.md("## Select Input")

    mo.vstack(
        [
            input_type,
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
