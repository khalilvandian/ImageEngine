import marimo

__generated_with = "0.17.4"
app = marimo.App(width="medium")

@app.cell
def _():
    import os
    from datetime import datetime
    from classification import classify_images
    return os, datetime, classify_images

@app.cell
def _(os):
    LOG_DIR = "logs"
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    return LOG_DIR

@app.cell
def _(marimo, classify_images, datetime, os, LOG_DIR):
    marimo.md("# Image Classification App")

    classification_output = marimo.state(None)

    input_type = marimo.ui.dropdown(
        options=["Folder", "File", "Multiple Files"],
        value="Folder",
        label="Select Input Type"
    )

    if input_type.value == "Folder":
        folder_path = marimo.ui.text(label="Enter Folder Path")
        inputs = [folder_path]
    elif input_type.value == "File":
        file_path = marimo.ui.text(label="Enter File Path")
        inputs = [file_path]
    elif input_type.value == "Multiple Files":
        files_path = marimo.ui.text(label="Enter File Paths (comma-separated)")
        inputs = [files_path]

    classify_button = marimo.ui.button(label="Classify")

    @classify_button.on_click
    def classify():
        """
        Classification function
        """
        log_filename = os.path.join(LOG_DIR, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        with open(log_filename, "w") as f:
            f.write(f"Input Type: {input_type.value}\n")
            image_paths = []
            if input_type.value == "Folder":
                folder = folder_path.value
                f.write(f"Folder Path: {folder}\n")
                if os.path.isdir(folder):
                    image_paths = [os.path.join(folder, p) for p in os.listdir(folder)]
            elif input_type.value == "File":
                file = file_path.value
                f.write(f"File Path: {file}\n")
                if os.path.isfile(file):
                    image_paths = [file]
            elif input_type.value == "Multiple Files":
                files = files_path.value
                f.write(f"File Paths: {files}\n")
                image_paths = [p.strip() for p in files.split(",")]

            f.write("\n--- Input Images ---\n")
            for path in image_paths:
                f.write(f"{path}\n")

            output = classify_images(image_paths)
            classification_output.set(output)

            f.write("\n--- Classification Output ---\n")
            for path, labels in output.items():
                f.write(f"{path}: {labels}\n")

        marimo.md(f"Classification complete. Log saved to: {log_filename}")

    marimo.md("## Select Input")
    marimo.md(input_type)
    for i in inputs:
        marimo.md(i)
    marimo.md(classify_button)

    if classification_output.value:
        marimo.md("## Classification Results")
        for path, labels in classification_output.value.items():
            marimo.md(f"**{path}**")
            for label in labels:
                marimo.md(f"- {label}")
    return (classification_output, classify, classify_button, file_path, files_path, folder_path, input_type, inputs)

if __name__ == "__main__":
    app.run()
