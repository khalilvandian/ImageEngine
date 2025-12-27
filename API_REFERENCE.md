# Quick Reference: Modular Classification API

## Import
```python
from src.classification import get_classifier, load_celebrities_from_json
```

## Load Reference Data
```python
celebrities = load_celebrities_from_json('data/celebrities.json')
```

## Create Classifier

### Pattern 1: Preset Configurations (Backward Compatible)
```python
# Traditional face_recognition with CNN
clf = get_classifier('face_recognition_cnn', celebrities)

# Traditional face_recognition with HOG
clf = get_classifier('face_recognition_hog', celebrities)

# InsightFace (buffalo_l detection + buffalo_l embedding)
clf = get_classifier('insightface', celebrities)

# InsightFace variants
clf = get_classifier('insightface_buffalo_m', celebrities)
clf = get_classifier('insightface_buffalo_s', celebrities)

# Vision Transformer
clf = get_classifier('vit_b32', celebrities)
```

### Pattern 2: Custom Modular Combinations (NEW)
```python
clf = get_classifier(
    'unified',  # Use unified classifier
    celebrities,
    detection_model='buffalo_l',  # Choose detection
    embedding_model='insightface',  # Choose embedding
    matching_method='cosine_similarity',  # Choose matching
    threshold=0.6  # Optional: similarity threshold
)
```

## Classify Images
```python
image_paths = ['path/to/image1.jpg', 'path/to/image2.jpg']
results = clf.classify_images(image_paths)

# Results format:
# {
#     'path/to/image1.jpg': [
#         {'name': 'Hugh Jackman', 'location': (top, right, bottom, left), 'confidence': 0.85},
#         {'name': 'Unknown', 'location': (top, right, bottom, left), 'confidence': 0.42}
#     ],
#     'path/to/image2.jpg': []
# }
```

## Model Options

### Detection Models
- `'buffalo_l'` - SCRFD-10GF (highest accuracy)
- `'buffalo_m'` - SCRFD-2.5GF (balanced)
- `'buffalo_s'` - SCRFD-500MF (fastest)
- `'antelopev2'` - Alternative high-end
- `'cnn'` - face_recognition CNN
- `'hog'` - face_recognition HOG (CPU-friendly)

### Embedding Models
- `'insightface'` or `'insightface_buffalo_l'` - 512-dim ResNet50
- `'insightface_buffalo_m'` - 512-dim
- `'insightface_buffalo_s'` - 512-dim MobileNet
- `'insightface_antelopev2'` - 512-dim ResNet100
- `'face_recognition'` - 128-dim dlib
- `'vit'` - 768-dim CLIP Vision Transformer

### Matching Methods
- `'cosine_similarity'` - For normalized embeddings (InsightFace, ViT)
- `'euclidean_distance'` - For face_recognition embeddings
- `'l2_distance'` - L2 with normalization

## Example Combinations

### Speed-Optimized
```python
clf = get_classifier(
    'unified', celebrities,
    detection_model='buffalo_s',  # Fast detection
    embedding_model='insightface_buffalo_s',  # Fast embedding
    matching_method='cosine_similarity'
)
```

### Accuracy-Optimized
```python
clf = get_classifier(
    'unified', celebrities,
    detection_model='buffalo_l',  # Accurate detection
    embedding_model='insightface_buffalo_l',  # Accurate embedding
    matching_method='cosine_similarity'
)
```

### CPU-Friendly
```python
clf = get_classifier(
    'unified', celebrities,
    detection_model='hog',  # CPU detection
    embedding_model='face_recognition',  # CPU embedding
    matching_method='euclidean_distance'
)
```

### Mixed Modern-Legacy
```python
clf = get_classifier(
    'unified', celebrities,
    detection_model='buffalo_l',  # Modern detection
    embedding_model='face_recognition',  # Legacy embedding
    matching_method='euclidean_distance'
)
```

### Vision Transformer
```python
clf = get_classifier(
    'unified', celebrities,
    detection_model='buffalo_l',  # Accurate detection
    embedding_model='vit',  # Transformer embeddings
    matching_method='cosine_similarity',
    threshold=0.8  # Higher threshold for ViT
)
```

## Additional Parameters

```python
clf = get_classifier(
    'unified',
    celebrities,
    detection_model='cnn',
    embedding_model='face_recognition',
    matching_method='euclidean_distance',
    threshold=0.6,  # Similarity threshold (0-1)
    detection_upsample=2,  # For CNN/HOG only (1-3)
    enable_multi_pass=True  # For CNN/HOG only
)
```

## Testing Systematic Combinations

```python
# Test all detection models with InsightFace embeddings
for det_model in ['buffalo_l', 'buffalo_m', 'buffalo_s', 'cnn', 'hog']:
    clf = get_classifier(
        'unified', celebrities,
        detection_model=det_model,
        embedding_model='insightface',
        matching_method='cosine_similarity'
    )
    results = clf.classify_images(test_images)
    print(f"{det_model}: {len(results)} images classified")

# Test all embedding models with buffalo_l detection
for emb_model in ['insightface', 'face_recognition', 'vit']:
    clf = get_classifier(
        'unified', celebrities,
        detection_model='buffalo_l',
        embedding_model=emb_model,
        matching_method='cosine_similarity' if emb_model != 'face_recognition' else 'euclidean_distance'
    )
    results = clf.classify_images(test_images)
    print(f"{emb_model}: {len(results)} images classified")
```

## Performance Tips

1. **Use buffalo_s for speed**: Lightweight detection, good accuracy
2. **Use buffalo_l for accuracy**: Highest detection quality
3. **Use hog for CPU**: When GPU unavailable
4. **Match embedding type to matching method**:
   - InsightFace embeddings → cosine_similarity
   - face_recognition embeddings → euclidean_distance
   - ViT embeddings → cosine_similarity or l2_distance
5. **Adjust thresholds**: Lower = stricter, higher = more lenient
   - face_recognition: 0.5-0.6
   - InsightFace: 0.5-0.65
   - ViT: 0.7-0.9

## Error Handling

```python
try:
    clf = get_classifier('unified', celebrities, ...)
    results = clf.classify_images(image_paths)
except ImportError as e:
    print(f"Missing library: {e}")
except ValueError as e:
    print(f"Invalid parameter: {e}")
except Exception as e:
    print(f"Error: {e}")
```
