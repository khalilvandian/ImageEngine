.PHONY: help test test-cnn test-hog test-vit test-all test-quick test-detector clean-outputs

# Default test parameters
TESTSET ?= testsets/test_set.json
CELEBRITIES ?= data/celebrities.json
SAVE_IMAGES ?= 1
FR_UPSAMPLE ?= 1
DETECTOR_MODEL ?= cnn

help:
	@echo "Available commands:"
	@echo "  make test-cnn          - Run tests with CNN face recognition model"
	@echo "  make test-hog          - Run tests with HOG face recognition model (faster)"
	@echo "  make test-vit          - Run tests with ViT-B/32 model"
	@echo "  make test-all          - Run tests with all three models sequentially"
	@echo "  make test-quick        - Run quick test (HOG model only)"
	@echo "  make test-detector     - Run face detector only (cnn/hog)"
	@echo "  make test-custom       - Run test with custom testset (use TESTSET=path/to/test.json)"
	@echo "  make clean-outputs     - Clean test output directories"
	@echo ""
	@echo "Examples:"
	@echo "  make test-cnn"
	@echo "  make test-custom TESTSET=testsets/hughJackmanTest.json"
	@echo "  make test-all TESTSET=testsets/four-people_testset.json"

test-cnn:
	@echo "Running CNN face recognition test..."
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
	print('\n=== TEST RESULTS (CNN) ==='); \
	print(json.dumps(metrics, indent=2)); \
	print(f'\nResults saved to: {csv_path}')"

test-hog:
	@echo "Running HOG face recognition test..."
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
	print('\n=== TEST RESULTS (HOG) ==='); \
	print(json.dumps(metrics, indent=2)); \
	print(f'\nResults saved to: {csv_path}')"

test-vit:
	@echo "Running ViT-B/32 test..."
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
	print('\n=== TEST RESULTS (ViT) ==='); \
	print(json.dumps(metrics, indent=2)); \
	print(f'\nResults saved to: {csv_path}')"

test-all: test-cnn test-hog test-vit
	@echo ""
	@echo "====================================="
	@echo "All tests completed!"
	@echo "====================================="

test-quick: test-hog

test-detector:
	@echo "Running face detector-only test with model=$(DETECTOR_MODEL)..."
	@FR_UPSAMPLE=$(FR_UPSAMPLE) python -c "from src.metrics import load_test_set, run_face_detection_on_test_set; import json; testset='$(TESTSET)'; detector_model='$(DETECTOR_MODEL)'; save_images='$(SAVE_IMAGES)'!='0'; image_paths,_=load_test_set(testset); print(f'Loaded {len(image_paths)} images'); detections_map, output_dir = run_face_detection_on_test_set(detector_model, image_paths, output_image_dir=('image_outputs' if save_images else None)); total_faces = sum(len(v) for v in detections_map.values()); max_faces = max((len(v) for v in detections_map.values()), default=0); print(f'Total detected faces: {total_faces}'); print(f'Max faces in one image: {max_faces}'); print(f'Annotated images saved to: {output_dir}' if output_dir else '')"

test-custom:
	@echo "Running test with custom testset: $(TESTSET)"
	@make test-hog TESTSET=$(TESTSET)

clean-outputs:
	@echo "Cleaning test outputs..."
	@rm -rf test_outputs/*
	@rm -rf image_outputs/*
	@echo "Test outputs cleaned!"
