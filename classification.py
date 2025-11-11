import face_recognition
import os

# --- Configuration ---
KNOWN_NAME = "Hugh Jackman"
REFERENCE_IMAGE_PATH = "Images/references/104711.jpg"
TOLERANCE = 0.6

# --- Load Known Face Data ---
known_face_encodings = []
try:
    reference_image = face_recognition.load_image_file(REFERENCE_IMAGE_PATH)
    reference_face_encodings = face_recognition.face_encodings(reference_image)
    if reference_face_encodings:
        known_face_encodings = [reference_face_encodings[0]]
    else:
        print(f"Warning: Could not find a face in the reference image: {REFERENCE_IMAGE_PATH}")
except FileNotFoundError:
    print(f"Error: Reference image not found at {REFERENCE_IMAGE_PATH}")
    # The app will classify everything as not Hugh Jackman.


def detect_celebrity(image_path):
    """
    Analyzes an image to see if the known celebrity is present.
    Returns a list with the celebrity's name if found, otherwise an empty list.
    """
    if not known_face_encodings:
        print("Warning: No known face encoding loaded. Cannot perform detection.")
        return []

    try:
        image = face_recognition.load_image_file(image_path)
    except FileNotFoundError:
        print(f"Error: Input image not found at {image_path}")
        return []

    face_locations = face_recognition.face_locations(image, number_of_times_to_upsample=1, model="cnn")
    face_encodings = face_recognition.face_encodings(image, face_locations)

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(
            known_face_encodings,
            face_encoding,
            tolerance=TOLERANCE
        )
        if True in matches:
            return [KNOWN_NAME]

    return []


def classify_images(image_paths):
    """
    This function takes a list of image paths and returns a dictionary
    with the classification results for the known celebrity.
    """
    output = {}
    for path in image_paths:
        output[path] = detect_celebrity(path)
    return output
