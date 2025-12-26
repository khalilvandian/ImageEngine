import sys
import os
import pprint

print("--- Python Path ---")
pprint.pprint(sys.path)

print("\n--- Environment Variables (LD_LIBRARY_PATH) ---")
print(os.environ.get("LD_LIBRARY_PATH", "Not Set"))

print("\n--- NumPy Info ---")
try:
    import numpy
    print(f"NumPy Version: {numpy.__version__}")
    print(f"NumPy Path: {numpy.__file__}")
except ImportError as e:
    print(f"NumPy Import Failed: {e}")

print("\n--- OpenCV Import Attempt ---")
try:
    # Try finding the module spec before importing, to see where it WOULD come from
    import importlib.util
    spec = importlib.util.find_spec("cv2")
    if spec:
        print(f"CV2 Spec Origin: {spec.origin}")
    else:
        print("CV2 Spec not found!")

    import cv2
    print(f"OpenCV Version: {cv2.__version__}")
    print(f"OpenCV Path: {cv2.__file__}")
except ImportError as e:
    print(f"OpenCV Import Failed: {e}")
    # Inspect the directory where cv2 is supposed to be
    if spec and spec.origin:
        dir_path = os.path.dirname(spec.origin)
        print(f"\nListing {dir_path}:")
        print(os.listdir(dir_path))
