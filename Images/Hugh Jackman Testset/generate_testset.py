import os
import json

base_dir = "." # Current directory
output_filename = "hughJackmanTest.json"
image_base_path = "Images/Hugh Jackman Testset"

all_images = []

# Process images in '1' directory (Hugh Jackman)
dir_1_path = os.path.join(base_dir, "1")
if os.path.exists(dir_1_path) and os.path.isdir(dir_1_path):
    for filename in os.listdir(dir_1_path):
        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp")):
            all_images.append({
                "image_name": filename,
                "path": os.path.join(image_base_path, "1", filename).replace("\\", "/"),
                "labels": ["Hugh Jackman"]
            })

# Process images in '0' directory (No label)
dir_0_path = os.path.join(base_dir, "0")
if os.path.exists(dir_0_path) and os.path.isdir(dir_0_path):
    for filename in os.listdir(dir_0_path):
        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp")):
            all_images.append({
                "image_name": filename,
                "path": os.path.join(image_base_path, "0", filename).replace("\\", "/"),
                "labels": []
            })

# Write to JSON file
with open(output_filename, "w") as f:
    json.dump(all_images, f, indent=4)

print(f"Successfully created {output_filename}")
