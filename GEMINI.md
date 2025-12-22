# Gemini Code Assistant Context

This document provides a comprehensive overview of the ImageEngine project, its structure, and how to run and extend it.

## Project Overview

ImageEngine is a Python-based application for detecting and recognizing celebrities in images. It provides an interactive web interface built with [Marimo](https://marimo.io/), allowing users to upload images and see the results of various classification models. The project is designed to be run in a Docker container with GPU support to accelerate the machine learning models.

### Key Technologies

*   **Backend:** Python
*   **Web Framework:** Marimo
*   **Machine Learning:**
    *   `face_recognition`: For face detection and recognition using both CNN and HOG models.
    *   `dlib`: A dependency for `face_recognition`, built from source with CUDA support for GPU acceleration.
    *   `PyTorch`: Used by the Vision Transformer (ViT) and TransFace models.
    *   `timm`: (PyTorch Image Models) Used to create the ViT model.
    *   `scikit-learn`: For calculating performance metrics.
    *   `modelscope`: Used for the TransFace model.
*   **Containerization:** Docker, Docker Compose
*   **Image Processing:** Pillow
*   **Data Handling:** pandas, numpy

### Architecture

The application is structured around a central `app.py` which creates the Marimo-based user interface. The core classification logic is abstracted into `classification.py`, which defines a `Classifier` base class and several concrete implementations:

*   **`FaceRecognitionClassifier`:** Uses the `face_recognition` library.
*   **`ViTClassifier`:** A Vision Transformer model.
*   **`TransFaceClassifier`:** A TransFace model from ModelScope.

A factory function, `get_classifier`, is used to instantiate the desired classifier. The application also includes a testing section to evaluate model performance using a predefined test set.

## Building and Running

### With Docker (Recommended)

The project is designed to be run with Docker and Docker Compose, which handles the complex dependencies and GPU configuration.

1.  **Prerequisites:**
    *   Docker installed
    *   NVIDIA Docker Toolkit installed for GPU support

2.  **Build and Run:**
    Execute the following command from the project root:
    ```bash
    docker-compose up --build
    ```

3.  **Access the Application:**
    Open your web browser and navigate to `http://localhost:2718`.

### Without Docker (Advanced)

Running without Docker is not recommended due to the specific `dlib` compilation requirements. However, if you wish to proceed:

1.  **Install System Dependencies:**
    *   `cmake`
    *   A C++ compiler
    *   NVIDIA CUDA Toolkit (if using GPU)

2.  **Install Python Dependencies:**
    It is highly recommended to use a virtual environment.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dlib with CUDA support:**
    Follow the instructions in the `Dockerfile` to clone and build `dlib` from source with CUDA enabled.

4.  **Install Python packages:**
    ```bash
    pip install -r requirements.txt
    ```

5.  **Run the application:**
    ```bash
    marimo run app.py
    ```

## Development Conventions

### Code Structure

*   **`app.py`:** Main Marimo application file containing the UI and event handling logic.
*   **`classification.py`:** Core classification logic, including the `Classifier` abstract base class and its implementations.
*   **`image_utils.py`:** Utility functions for image manipulation (e.g., drawing bounding boxes).
*   **`logging_utils.py`:** Configures the application's logger.
*   **`metrics.py`:** Functions for calculating and plotting model performance metrics.
*   **`*.json`:** Configuration files for celebrities and test sets.
*   **`Images/`:** Contains the image datasets.
*   **`image_outputs/`:** Default directory for saving annotated images.
*   **`test_outputs/`:** Default directory for saving test result CSVs.

### Adding a New Classifier

To add a new classifier, follow these steps:

1.  Create a new class in `classification.py` that inherits from `Classifier`.
2.  Implement the `classify_images` method.
3.  Add the new classifier type to the `get_classifier` factory function in `classification.py`.
4.  Add the new classifier to the dropdown options in `app.py`.

### Testing

The application includes a "Model Testing" section in the UI. This section uses the configuration from `test_config.json` to load a test set and evaluate the selected model. The results, including metrics and plots, are displayed in the UI and saved to the `test_outputs` directory.
