import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import onnxruntime
import time

def main():
    print(f"ONNX Runtime Device: {onnxruntime.get_device()}")
    print(f"Available Providers: {onnxruntime.get_available_providers()}")

    # Initialize FaceAnalysis with CUDA
    print("\nInitializing FaceAnalysis with CUDAExecutionProvider...")
    try:
        app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        app.prepare(ctx_id=0, det_size=(640, 640))
        print("✅ FaceAnalysis initialized successfully.")
    except Exception as e:
        print(f"❌ Failed to initialize FaceAnalysis: {e}")
        return

    # Create a dummy image (black square) just to test the forward pass
    img = np.zeros((640, 640, 3), dtype=np.uint8)
    
    print("Running dummy detection...")
    start = time.time()
    faces = app.get(img)
    end = time.time()
    
    print(f"Detection ran in {end - start:.4f} seconds.")
    print("InsightFace GPU feasibility test complete.")

if __name__ == "__main__":
    main()

