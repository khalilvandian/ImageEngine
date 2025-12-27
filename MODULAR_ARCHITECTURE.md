# Modular Classification Architecture

## Overview

The classification system has been completely refactored to support **pluggable components** allowing ANY combination of:
- **Detection models** (buffalo_l, buffalo_m, buffalo_s, cnn, hog, etc.)
- **Embedding models** (insightface variants, face_recognition, vit)
- **Matching methods** (cosine_similarity, euclidean_distance, l2_distance)

This enables systematic testing of different model combinations without code changes.

---

## Architecture Components

### 1. Detection Layer (`FaceDetector`)
**Responsibility**: Locate faces in images

**Supported Models**:
- `buffalo_l` - InsightFace SCRFD-10GF (highest accuracy, 326MB)
- `buffalo_m` - InsightFace SCRFD-2.5GF (313MB)
- `buffalo_s` - InsightFace SCRFD-500MF (lightweight, 159MB)
- `antelopev2` - InsightFace alternative (407MB)
- `cnn` - face_recognition CNN detector
- `hog` - face_recognition HOG detector (CPU-friendly)

### 2. Embedding Layer (`EmbeddingExtractor`)
**Responsibility**: Extract feature vectors from face crops

**Supported Models**:
- `insightface` (buffalo_l) - 512-dim ResNet50 embeddings
- `insightface_buffalo_m` - 512-dim embeddings
- `insightface_buffalo_s` - 512-dim embeddings (MobileNet)
- `insightface_antelopev2` - 512-dim ResNet100 embeddings
- `face_recognition` - 128-dim dlib embeddings
- `vit` - 768-dim Vision Transformer (CLIP) embeddings

### 3. Matching Layer (`MatchingMethod`)
**Responsibility**: Compare embeddings for identification

**Supported Methods**:
- `cosine_similarity` - Normalized dot product (best for InsightFace, ViT)
- `euclidean_distance` - L2 distance (best for face_recognition)
- `l2_distance` - Normalized L2 with thresholding

---

## Usage

### Basic Usage (Backward Compatible)

```python
from src.classification import get_classifier, load_celebrities_from_json

celebrities = load_celebrities_from_json('data/celebrities.json')

# Legacy classifiers still work
clf_cnn = get_classifier('face_recognition_cnn', celebrities)
clf_hog = get_classifier('face_recognition_hog', celebrities)
clf_insightface = get_classifier('insightface', celebrities)
clf_vit = get_classifier('vit_b32', celebrities)
```

### Advanced Usage (Modular Combinations)

```python
# Combination 1: Fast detection + accurate embeddings
fast_accurate = get_classifier(
    'unified',
    celebrities,
    detection_model='buffalo_s',  # lightweight, fast
    embedding_model='insightface_buffalo_l',  # accurate embeddings
    matching_method='cosine_similarity',
    threshold=0.6
)

# Combination 2: Legacy detection + modern embeddings
legacy_modern = get_classifier(
    'unified',
    celebrities,
    detection_model='hog',  # CPU-friendly
    embedding_model='insightface',  # modern 512-dim
    matching_method='cosine_similarity',
    threshold=0.55
)

# Combination 3: Vision Transformer with HOG detection
vit_hog = get_classifier(
    'unified',
    celebrities,
    detection_model='hog',
    embedding_model='vit',
    matching_method='l2_distance',
    threshold=0.8
)

# Combination 4: Buffalo_l detection + face_recognition embeddings
mixed = get_classifier(
    'unified',
    celebrities,
    detection_model='buffalo_l',  # accurate detection
    embedding_model='face_recognition',  # 128-dim embeddings
    matching_method='euclidean_distance',
    threshold=0.6
)
```

---

## Example Combinations to Test

| Detection | Embedding | Matching | Use Case |
|-----------|-----------|----------|----------|
| buffalo_s | insightface_buffalo_l | cosine | Fast detection, accurate recognition |
| buffalo_l | insightface_buffalo_s | cosine | Accurate detection, lightweight recognition |
| hog | insightface | cosine | CPU-friendly detection, modern embeddings |
| cnn | face_recognition | euclidean | Traditional pipeline |
| buffalo_l | vit | cosine | High-end detection + Transformer |
| hog | vit | l2 | CPU detection + Transformer |

---

## Benefits

### 1. **Flexibility**
- Mix and match any detection/embedding/matching combination
- No code changes needed to test new configurations
- Systematic evaluation of model trade-offs

### 2. **Reusability**
- `FaceDetector` can be used standalone
- `EmbeddingExtractor` can be used independently
- Components are loosely coupled

### 3. **Maintainability**
- Clear separation of concerns
- Single responsibility per class
- Easy to add new models

### 4. **Performance Optimization**
- Choose lightweight detection for speed
- Choose accurate embeddings for quality
- Balance based on requirements

### 5. **Research & Experimentation**
- Systematically compare combinations
- Identify optimal configurations
- Understand component impact

---

## Implementation Details

### Class Hierarchy

```
Classifier (ABC)
└── UnifiedClassifier
    ├── FaceDetector (pluggable detection)
    ├── EmbeddingExtractor (pluggable embedding)
    │   ├── InsightFaceEmbedder
    │   ├── FaceRecognitionEmbedder
    │   └── ViTEmbedder
    └── MatchingMethod (pluggable matching)
        ├── CosineSimilarityMatching
        ├── EuclideanDistanceMatching
        └── L2DistanceMatching
```

### Pipeline Flow

```
1. DETECTION: Image → FaceDetector → Bounding boxes
2. EMBEDDING: Face crops → EmbeddingExtractor → Feature vectors
3. MATCHING: Query embeddings vs Reference embeddings → Similarity scores
4. IDENTIFICATION: Scores vs Threshold → Celebrity names
```

---

## Migration Guide

### Old Code
```python
# Separate classifiers for each model
cnn_clf = FaceRecognitionClassifier(name="cnn", ...)
insightface_clf = InsightFaceClassifier(name="insightface", ...)
vit_clf = ViTClassifier(name="vit", ...)
```

### New Code
```python
# Unified classifier with pluggable components
clf = UnifiedClassifier(
    name="custom",
    celebrity_data=celebrities,
    detection_model="buffalo_l",
    embedding_model="insightface",
    matching_method="cosine_similarity"
)
```

### Backward Compatibility
All old classifier types (`face_recognition_cnn`, `face_recognition_hog`, `insightface`, `vit_b32`) still work via `get_classifier()`:

```python
# These still work exactly as before
clf1 = get_classifier("face_recognition_cnn", celebrities)
clf2 = get_classifier("insightface", celebrities)
clf3 = get_classifier("vit_b32", celebrities)
```

---

## Future Enhancements

### Potential Additions
1. **More Detection Models**: YOLOv8-Face, RetinaFace, MTCNN
2. **More Embedding Models**: ArcFace, CosFace, FaceNet variants
3. **More Matching Methods**: Learned metrics, attention-based matching
4. **Ensemble Methods**: Combine multiple models for voting
5. **Dynamic Model Selection**: Auto-select based on image characteristics

### Extension Pattern
```python
# Adding a new embedding model
class NewEmbedder(EmbeddingExtractor):
    def extract_embedding(self, image, bbox):
        # Your implementation
        return embedding
    
    def extract_embeddings_batch(self, image, bboxes):
        # Your batch implementation
        return embeddings

# Register in UnifiedClassifier._create_embedder()
```

---

## Summary

The new modular architecture enables:
- ✅ ANY combination of detection + embedding + matching
- ✅ Backward compatibility with existing code
- ✅ Clean separation of concerns
- ✅ Easy extensibility
- ✅ Systematic model comparison
- ✅ Performance optimization flexibility

**This is exactly what you requested**: A unified classifier where you can pass detection model, embedding model, and matching method as arguments, enabling testing of every possible combination.
