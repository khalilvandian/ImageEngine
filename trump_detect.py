import marimo

__generated_with = "0.17.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return


@app.cell
def _():
    import face_recognition
    import cv2
    import numpy as np
    import os
    return cv2, face_recognition


@app.cell
def _(cv2, face_recognition):


    # =================================================================
    #               🖼️ CONFIGURATION VARIABLES
    # =================================================================

    # --- 1. Reference Image Settings (The person to be detected) ---
    # NOTE: This image MUST contain a clear picture of Donald Trump's face.
    KNOWN_NAME = "Donald Trump"
    REFERENCE_IMAGE_PATH = "images/reference_trump.jpg" 

    # --- 2. Input/Output Image Settings (The image to analyze) ---
    INPUT_IMAGE_PATH = "images/yes_trump_2.jpg"
    OUTPUT_IMAGE_PATH = "output_detection.jpg"

    # --- 3. Model/Tolerance Settings ---
    TOLERANCE = 0.6  # Lower is stricter (0.6 is typical for face_recognition)

    # =================================================================
    #               🧠 CORE DETECTION LOGIC
    # =================================================================

    # --- 1. Load Known Face Data ---
    try:
        # Load the reference image
        trump_image = face_recognition.load_image_file(REFERENCE_IMAGE_PATH)

        # Get the face encoding for the reference image
        trump_face_encodings = face_recognition.face_encodings(trump_image)

        if not trump_face_encodings:
            print(f"❌ Error: Could not find a face in the reference image: {REFERENCE_IMAGE_PATH}")
            exit()

        known_face_encodings = [trump_face_encodings[0]]

    except FileNotFoundError:
        print(f"❌ Error: Reference image '{REFERENCE_IMAGE_PATH}' not found.")
        print(f"       Please ensure your reference image is named '{REFERENCE_IMAGE_PATH}' and is in the correct directory.")
        exit()


    def visualize_trump_detection(input_path, output_path, known_name):
        """
        Analyzes an image, draws a bounding box around the target face if found,
        saves the result, and returns a 'Yes' or 'No' answer.
        """
        print(f"Analyzing input image: {input_path}...")

        try:
            # Load the image in RGB format (required by face_recognition)
            image = face_recognition.load_image_file(input_path)
        except FileNotFoundError:
            return f"❌ Error: Input image '{input_path}' not found."

        # Convert the image to BGR for OpenCV processing/saving
        visual_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Find all face locations and face encodings in the image
        face_locations = face_recognition.face_locations(image)
        face_encodings = face_recognition.face_encodings(image, face_locations)

        trump_found = False

        # Process all detected faces
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Compare the unknown face with the known face encoding
            matches = face_recognition.compare_faces(
                known_face_encodings, 
                face_encoding, 
                tolerance=TOLERANCE
            )

            name = "Unknown Person"

            if True in matches:
                # A match was found
                name = known_name
                trump_found = True

                # --- Drawing Logic for Identified Person (Green Box) ---
                color = (0, 255, 0) # Green in BGR
                cv2.rectangle(visual_image, (left, top), (right, bottom), color, 2)
                cv2.rectangle(visual_image, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(visual_image, name, (left + 6, bottom - 6), font, 1.0, (0, 0, 0), 1)

            else:
                 # --- Drawing Logic for Unknown Person (Red Box) ---
                color = (0, 0, 255) # Red in BGR
                cv2.rectangle(visual_image, (left, top), (right, bottom), color, 1)


        # Save the resulting image
        cv2.imwrite(output_path, visual_image)
        print(f"✅ Detection complete. Visual output saved to: {output_path}")

        # The final required response
        return "Yes" if trump_found else "No"


    # =================================================================
    #               🚀 EXECUTION
    # =================================================================

    final_result = visualize_trump_detection(INPUT_IMAGE_PATH, OUTPUT_IMAGE_PATH, KNOWN_NAME)

    print(f"\nModel Response (Trump Found): **{final_result}**")
    return


if __name__ == "__main__":
    app.run()
