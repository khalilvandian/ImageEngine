# ImageEngine Project Context

This document provides a comprehensive overview of the ImageEngine project, its structure, and how to run and extend it.

## Project Overview

ImageEngine is a Python-based application for detecting and recognizing celebrities in images. It provides an interactive web interface built with [Streamlit](https://streamlit.io/), allowing users to upload images and see the results of various classification models. The project is designed to be run in a Docker container with GPU support to accelerate the machine learning models.

### Key Technologies

*   **Backend:** Python
*   **Web Framework:** Streamlit
*   **Machine Learning:**
    *   `face_recognition`: For face detection and recognition using both CNN and HOG models.
    *   `dlib`: A dependency for `face_recognition`, built from source with CUDA support for GPU acceleration.
    *   `PyTorch`: Used by the Vision Transformer (ViT) model.
    *   `timm`: (PyTorch Image Models) Used to create the ViT model.
    *   `scikit-learn`: For calculating performance metrics.
*   **Containerization:** Docker, Docker Compose
*   **Image Processing:** Pillow
*   **Data Handling:** pandas, numpy, tqdm

### Architecture

The application is structured around a central `app.py` which creates the Streamlit-based user interface. The core logic is located in the `src/` directory:

*   **`src/classification.py`:** Core classification logic.
    *   `FaceDetector`: A modular class for face detection using CNN or HOG, supporting multi-pass and batch processing.
    *   `Classifier`: Abstract base class for all classifiers.
    *   `FaceRecognitionClassifier`: Uses the `face_recognition` library.
    *   `ViTClassifier`: A Vision Transformer model using `timm`.
    *   `get_classifier`: Factory function to instantiate classifiers.
*   **`src/metrics.py`:** Functions for calculating metrics (accuracy, precision, recall, F1) and generating plots (confusion matrix, ROC curve).
*   **`src/image_utils.py`:** Utility functions for image manipulation like drawing bounding boxes.
*   **`src/logging_utils.py`:** Centralized logging configuration.

The project also includes a CLI test runner:
*   **`run_tests.py`:** Allows running evaluations on test sets from the command line without the UI.

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
    Open your web browser and navigate to `http://localhost:8501`.

### Without Docker (Advanced)

Running without Docker is not recommended due to the specific `dlib` compilation requirements. However, if you wish to proceed:

1.  **Install System Dependencies:**
    *   `cmake`, C++ compiler, NVIDIA CUDA Toolkit (if using GPU).

2.  **Install Python Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the application:**
    ```bash
    streamlit run app.py
    ```

4.  **Run CLI tests:**
    ```bash
    python run_tests.py --model hog --quick
    ```

## Development Conventions

### Code Structure

*   **`app.py`:** Main Streamlit application file.
*   **`src/`:** Contains the modularized source code.
*   **`data/celebrities.json`:** Default configuration for celebrity reference images.
*   **`testsets/`:** Contains test set definitions in JSON format.
*   **`Images/`:** Contains image datasets.
*   **`image_outputs/`:** Default directory for saving annotated images.
*   **`test_outputs/`:** Default directory for saving test result CSVs.

### Batch Processing & Memory Management

The project implements batch processing for both face detection and recognition to optimize GPU usage. Batch sizes and other parameters can be controlled via environment variables:
*   `FR_IMAGE_BATCH`: Number of images to load at once.
*   `FR_DETECT_BATCH`: Batch size for the CNN face detector.
*   `FR_UPSAMPLE`: Number of times to upsample for detection (default 1).
*   `VIT_IMAGE_BATCH` / `VIT_FACE_BATCH`: Batch sizes for ViT processing.

### Adding a New Classifier

To add a new classifier:
1.  Inherit from `Classifier` in `src/classification.py`.
2.  Implement `classify_images(self, image_paths)`.
3.  Register it in `get_classifier` factory.
4.  Add the option to the dropdown in `app.py`.