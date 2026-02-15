.PHONY: help test test-if test-if-m test-if-s test-modular test-detector test-legacy-cnn test-legacy-hog test-legacy-vit test-legacy-all test-legacy-quick test-custom clean-outputs

# Default test parameters
TESTSET ?= testsets/test_set.json
CELEBRITIES ?= data/references.json
SAVE_IMAGES ?= 1
FR_UPSAMPLE ?= 1
DETECTOR_MODEL ?= cnn

# MODULAR PIPELINE PARAMETERS (for test-unified command)
DETECTION ?= buffalo_l
EMBEDDING ?= insightface
MATCHING ?= cosine_similarity
THRESHOLD ?= 0.6

help:
	@echo "======================================================================="
	@echo "           MODULAR FACE RECOGNITION - TEST COMMANDS"
	@echo "======================================================================="
	@echo ""
	@echo "FULLY MODULAR (any detection + embedding + matching combination):"
	@echo "  make test-unified DETECTION=<det> EMBEDDING=<emb> MATCHING=<match>"
	@echo ""
	@echo "    Detection options:   buffalo_l, buffalo_m, buffalo_s, cnn, hog"
	@echo "    Embedding options:   insightface, insightface_buffalo_m, insightface_buffalo_s,"
	@echo "                         face_recognition, vit"
	@echo "    Matching options:    cosine_similarity, euclidean_distance, l2_distance"
	@echo "    Threshold:           any float 0.0-1.0 (default: 0.6)"
	@echo ""
	@echo "  Examples:"
	@echo "    make test-unified DETECTION=buffalo_l EMBEDDING=insightface"
	@echo "    make test-unified DETECTION=hog EMBEDDING=face_recognition MATCHING=euclidean_distance"
	@echo "    make test-unified DETECTION=cnn EMBEDDING=vit MATCHING=cosine_similarity THRESHOLD=0.8"
	@echo ""
	@echo "PRESET SHORTCUTS (Recommended modular combinations - shown below for quick testing):"
	@echo "  make test-if           - buffalo_l detection + insightface embedding + cosine"
	@echo "  make test-if-m         - buffalo_m detection + insightface_buffalo_m + cosine"
	@echo "  make test-if-s         - buffalo_s detection + insightface_buffalo_s + cosine"
	@echo "  make test-modular      - Compare all InsightFace variants (if + if-m + if-s)"
	@echo ""
	@echo "LEGACY ARCHITECTURE (Deprecated - no alignment):"
	@echo "  make test-legacy-cnn   - [LEGACY] CNN detector + face_recognition"
	@echo "  make test-legacy-hog   - [LEGACY] HOG detector + face_recognition (faster)"
	@echo "  make test-legacy-vit   - [LEGACY] Buffalo-L detector + ViT-B/32"
	@echo "  make test-legacy-all   - [LEGACY] Run all legacy models"
	@echo "  make test-legacy-quick - [LEGACY] Quick test (HOG only)"
	@echo ""
	@echo "UTILITIES:"
	@echo "  make test-detector     - Face detection-only test (no recognition)"
	@echo "  make test-custom       - Run test with custom testset (use TESTSET=path)"
	@echo "  make clean-outputs     - Clean test output directories"
	@echo ""
	@echo "ENVIRONMENT VARIABLES (Global):"
	@echo "  TESTSET=path/to/test.json    - Custom test set (default: testsets/test_set.json)"
	@echo "  CELEBRITIES=path/to/celebs   - Custom celebrities file (default: data/celebrities.json)"
	@echo "  SAVE_IMAGES=1|0              - Save annotated images (default: 1)"
	@echo "  FR_UPSAMPLE=N                - Detection upsampling for CNN/HOG (default: 1)"
	@echo ""
	@echo "  DETECTION=<model>            - For test-unified (default: buffalo_l)"
	@echo "  EMBEDDING=<model>            - For test-unified (default: insightface)"
	@echo "  MATCHING=<method>            - For test-unified (default: cosine_similarity)"
	@echo "  THRESHOLD=<float>            - For test-unified (default: 0.6)"
	@echo ""
	@echo "QUICK START - Explore Modularity:"
	@echo "  make test-if                          # Default modular (buffalo_l+insightface)"
	@echo "  make test-unified DETECTION=cnn       # Switch detector only"
	@echo "  make test-unified EMBEDDING=vit       # Switch embedding only"
	@echo "  make test-unified DETECTION=hog EMBEDDING=face_recognition MATCHING=euclidean_distance"
	@echo "  make test-unified DETECTION=buffalo_s EMBEDDING=insightface_buffalo_m THRESHOLD=0.7"
	@echo ""
	@echo "======================================================================="

test-cnn:
	@echo "[DEPRECATED - use 'make test-legacy-cnn' instead]"
	@make test-legacy-cnn

test-hog:
	@echo "[DEPRECATED - use 'make test-legacy-hog' instead]"
	@make test-legacy-hog

test-vit:
	@echo "[DEPRECATED - use 'make test-legacy-vit' instead]"
	@make test-legacy-vit

test-all:
	@echo "[DEPRECATED - use 'make test-legacy-all' instead]"
	@make test-legacy-all

test-quick:
	@echo "[DEPRECATED - use 'make test-legacy-quick' instead]"
	@make test-legacy-quick

# NEW MODULAR ARCHITECTURE COMMANDS (Primary)
test-unified:
	@echo "Running MODULAR test: detection=$(DETECTION) embedding=$(EMBEDDING) matching=$(MATCHING) threshold=$(THRESHOLD)"
	@python run_tests.py --model unified --detection $(DETECTION) --embedding $(EMBEDDING) --matching $(MATCHING) --threshold $(THRESHOLD) --testset $(TESTSET) --celebrities $(CELEBRITIES) $(if $(filter 0,$(SAVE_IMAGES)),--no-images,)

test-if:
	@echo "Running InsightFace (buffalo_l) with landmark alignment..."
	@python -c "from src.metrics import load_test_set, run_classification_on_test_set, calculate_metrics, save_test_output_to_csv; \
	from src.classification import get_classifier, load_celebrities_from_json; \
	import json; \
	image_paths, ground_truth = load_test_set('$(TESTSET)'); \
	print(f'Loaded {len(image_paths)} images'); \
	celebrities = load_celebrities_from_json('$(CELEBRITIES)'); \
	classifier = get_classifier('insightface', celebrities); \
	predictions = run_classification_on_test_set(classifier, image_paths, output_image_dir=('image_outputs' if '$(SAVE_IMAGES)' != '0' else None)); \
	metrics = calculate_metrics(ground_truth, predictions); \
	csv_path = save_test_output_to_csv(image_paths, predictions, ground_truth, 'insightface_buffalo_l_aligned'); \
	print('\n=== TEST RESULTS (InsightFace + Alignment) ==='); \
	print(json.dumps(metrics, indent=2)); \
	print(f'\nResults saved to: {csv_path}')"

test-if-m:
	@echo "Running InsightFace (buffalo_m) with landmark alignment..."
	@python -c "from src.metrics import load_test_set, run_classification_on_test_set, calculate_metrics, save_test_output_to_csv; \
	from src.classification import get_classifier, load_celebrities_from_json; \
	import json; \
	image_paths, ground_truth = load_test_set('$(TESTSET)'); \
	print(f'Loaded {len(image_paths)} images'); \
	celebrities = load_celebrities_from_json('$(CELEBRITIES)'); \
	classifier = get_classifier('insightface_buffalo_m', celebrities); \
	predictions = run_classification_on_test_set(classifier, image_paths, output_image_dir=('image_outputs' if '$(SAVE_IMAGES)' != '0' else None)); \
	metrics = calculate_metrics(ground_truth, predictions); \
	csv_path = save_test_output_to_csv(image_paths, predictions, ground_truth, 'insightface_buffalo_m_aligned'); \
	print('\n=== TEST RESULTS (InsightFace Buffalo-M + Alignment) ==='); \
	print(json.dumps(metrics, indent=2)); \
	print(f'\nResults saved to: {csv_path}')"

test-if-s:
	@echo "Running InsightFace (buffalo_s) with landmark alignment..."
	@python -c "from src.metrics import load_test_set, run_classification_on_test_set, calculate_metrics, save_test_output_to_csv; \
	from src.classification import get_classifier, load_celebrities_from_json; \
	import json; \
	image_paths, ground_truth = load_test_set('$(TESTSET)'); \
	print(f'Loaded {len(image_paths)} images'); \
	celebrities = load_celebrities_from_json('$(CELEBRITIES)'); \
	classifier = get_classifier('insightface_buffalo_s', celebrities); \
	predictions = run_classification_on_test_set(classifier, image_paths, output_image_dir=('image_outputs' if '$(SAVE_IMAGES)' != '0' else None)); \
	metrics = calculate_metrics(ground_truth, predictions); \
	csv_path = save_test_output_to_csv(image_paths, predictions, ground_truth, 'insightface_buffalo_s_aligned'); \
	print('\n=== TEST RESULTS (InsightFace Buffalo-S + Alignment) ==='); \
	print(json.dumps(metrics, indent=2)); \
	print(f'\nResults saved to: {csv_path}')"

test-modular:
	@echo "Running comprehensive modular architecture tests..."
	@python -c "from src.metrics import load_test_set, run_classification_on_test_set, calculate_metrics, save_test_output_to_csv; \
	from src.classification import get_classifier, load_celebrities_from_json; \
	import json; \
	image_paths, ground_truth = load_test_set('$(TESTSET)'); \
	celebrities = load_celebrities_from_json('$(CELEBRITIES)'); \
	models = ['insightface', 'insightface_buffalo_m', 'insightface_buffalo_s']; \
	results = {}; \
	for model in models: \
	  print(f'\n🔄 Testing {model}...'); \
	  clf = get_classifier(model, celebrities); \
	  preds = run_classification_on_test_set(clf, image_paths, output_image_dir=('image_outputs' if '$(SAVE_IMAGES)' != '0' else None)); \
	  metrics = calculate_metrics(ground_truth, preds); \
	  results[model] = metrics; \
	  save_test_output_to_csv(image_paths, preds, ground_truth, model); \
	print('\n=== SUMMARY: MODULAR VARIANTS ==='); \
	print(json.dumps({k: v for k,v in results.items()}, indent=2))"

# LEGACY ARCHITECTURE COMMANDS (Deprecated)
test-legacy-cnn:
	@echo "Running [LEGACY] CNN face recognition test..."
	@FR_UPSAMPLE=$(FR_UPSAMPLE) python -c "from src.metrics import load_test_set, run_classification_on_test_set, calculate_metrics, save_test_output_to_csv; \
	from src.classification import get_classifier, load_celebrities_from_json; \
	import json; \
	image_paths, ground_truth = load_test_set('$(TESTSET)'); \
	print(f'Loaded {len(image_paths)} images'); \
	celebrities = load_celebrities_from_json('$(CELEBRITIES)'); \
	classifier = get_classifier('face_recognition_cnn', celebrities); \
	predictions = run_classification_on_test_set(classifier, image_paths, output_image_dir=('image_outputs' if '$(SAVE_IMAGES)' != '0' else None)); \
	metrics = calculate_metrics(ground_truth, predictions); \
	csv_path = save_test_output_to_csv(image_paths, predictions, ground_truth, 'face_recognition_cnn'); \
	print('\n=== TEST RESULTS (LEGACY CNN) ==='); \
	print(json.dumps(metrics, indent=2)); \
	print(f'\nResults saved to: {csv_path}')"

test-legacy-hog:
	@echo "Running [LEGACY] HOG face recognition test..."
	@python -c "from src.metrics import load_test_set, run_classification_on_test_set, calculate_metrics, save_test_output_to_csv; \
	from src.classification import get_classifier, load_celebrities_from_json; \
	import json; \
	image_paths, ground_truth = load_test_set('$(TESTSET)'); \
	print(f'Loaded {len(image_paths)} images'); \
	celebrities = load_celebrities_from_json('$(CELEBRITIES)'); \
	classifier = get_classifier('face_recognition_hog', celebrities); \
	predictions = run_classification_on_test_set(classifier, image_paths, output_image_dir=('image_outputs' if '$(SAVE_IMAGES)' != '0' else None)); \
	metrics = calculate_metrics(ground_truth, predictions); \
	csv_path = save_test_output_to_csv(image_paths, predictions, ground_truth, 'face_recognition_hog'); \
	print('\n=== TEST RESULTS (LEGACY HOG) ==='); \
	print(json.dumps(metrics, indent=2)); \
	print(f'\nResults saved to: {csv_path}')"

test-legacy-vit:
	@echo "Running [LEGACY] ViT-B/32 test..."
	@python -c "from src.metrics import load_test_set, run_classification_on_test_set, calculate_metrics, save_test_output_to_csv; \
	from src.classification import get_classifier, load_celebrities_from_json; \
	import json; \
	image_paths, ground_truth = load_test_set('$(TESTSET)'); \
	print(f'Loaded {len(image_paths)} images'); \
	celebrities = load_celebrities_from_json('$(CELEBRITIES)'); \
	classifier = get_classifier('vit_b32', celebrities); \
	predictions = run_classification_on_test_set(classifier, image_paths, output_image_dir=('image_outputs' if '$(SAVE_IMAGES)' != '0' else None)); \
	metrics = calculate_metrics(ground_truth, predictions); \
	csv_path = save_test_output_to_csv(image_paths, predictions, ground_truth, 'vit_b32'); \
	print('\n=== TEST RESULTS (LEGACY ViT) ==='); \
	print(json.dumps(metrics, indent=2)); \
	print(f'\nResults saved to: {csv_path}')"

test-legacy-all: test-legacy-cnn test-legacy-hog test-legacy-vit
	@echo ""
	@echo "====================================="
	@echo "All LEGACY tests completed!"
	@echo "====================================="

test-legacy-quick: test-legacy-hog

test-detector:
	@echo "Running face detector-only test with model=$(DETECTOR_MODEL)..."
	@FR_UPSAMPLE=$(FR_UPSAMPLE) python -c "from src.metrics import load_test_set, run_face_detection_on_test_set; import json; testset='$(TESTSET)'; detector_model='$(DETECTOR_MODEL)'; save_images='$(SAVE_IMAGES)'!='0'; image_paths,_=load_test_set(testset); print(f'Loaded {len(image_paths)} images'); detections_map, output_dir = run_face_detection_on_test_set(detector_model, image_paths, output_image_dir=('image_outputs' if save_images else None)); total_faces = sum(len(v) for v in detections_map.values()); max_faces = max((len(v) for v in detections_map.values()), default=0); print(f'Total detected faces: {total_faces}'); print(f'Max faces in one image: {max_faces}'); print(f'Annotated images saved to: {output_dir}' if output_dir else '')"

test-custom:
	@echo "Running test with custom testset: $(TESTSET) [LEGACY]"
	@make test-legacy-hog TESTSET=$(TESTSET)

clean-outputs:
	@echo "Cleaning test outputs..."
	@rm -rf test_outputs/*
	@rm -rf image_outputs/*
	@echo "Test outputs cleaned!"
