#!/usr/bin/env python3
"""
Quick test runner script for running model tests without Streamlit.
Usage:
    python run_tests.py --model cnn --testset testsets/test_set.json
    python run_tests.py --model all
    python run_tests.py --quick
"""

import argparse
import os
import json
import sys
from src.metrics import (
    load_test_set,
    run_classification_on_test_set,
    run_face_detection_on_test_set,
    calculate_metrics,
    save_test_output_to_csv,
)
from src.classification import get_classifier, load_celebrities_from_json
from src.logging_utils import setup_logger

logger = setup_logger()
# Limit thread usage to reduce memory contention
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("FR_SERIALIZE", "1")
os.environ.setdefault("VIT_SERIALIZE", "1")
os.environ.setdefault("FR_IMAGE_BATCH", "4")
os.environ.setdefault("FR_DETECT_BATCH", "4")
os.environ.setdefault("FR_CNN_GROUP_BATCH", "4")


def run_test(model_name, testset_path, celebrities_path, output_dir="image_outputs", save_images=True):
    """Run a single test with the specified model."""
    print(f"\n{'='*60}")
    print(f"Running {model_name.upper()} model test")
    print(f"{'='*60}")
    
    # Load test set
    image_paths, ground_truth = load_test_set(testset_path)
    print(f"✓ Loaded {len(image_paths)} images from {testset_path}")
    
    if not image_paths:
        print("❌ No images found in test set!")
        return None
    
    # Load celebrity data
    celebrities = load_celebrities_from_json(celebrities_path)
    if not celebrities:
        print(f"❌ No celebrity data loaded from {celebrities_path}")
        return None
    print(f"✓ Loaded {len(celebrities)} celebrities")
    
    # Initialize classifier
    classifier = get_classifier(model_name, celebrities)
    if classifier is None:
        print(f"❌ Could not initialize classifier: {model_name}")
        return None
    
    # Run predictions
    print(f"\n🔄 Running predictions...")
    predictions = run_classification_on_test_set(
        classifier,
        image_paths,
        output_image_dir=output_dir if save_images else None,
    )
    
    # Calculate metrics
    metrics = calculate_metrics(ground_truth, predictions)
    
    # Save results
    csv_path = save_test_output_to_csv(
        image_paths, predictions, ground_truth, model_name
    )
    
    # Display results
    print(f"\n{'='*60}")
    print(f"TEST RESULTS: {model_name.upper()}")
    print(f"{'='*60}")
    for metric, value in metrics.items():
        metric_label = metric.replace("_", " ").title()
        print(f"{metric_label:25s}: {value:.4f}" if isinstance(value, float) else f"{metric_label:25s}: {value}")
    print(f"\n📄 Results saved to: {csv_path}")
    print(f"{'='*60}\n")
    
    return metrics


def run_detector_test(detector_model, testset_path, output_dir="image_outputs", save_images=True):
    """Run detection-only test using FaceDetector."""
    print(f"\n{'='*60}")
    print(f"Running FACE DETECTOR ({detector_model.upper()}) test")
    print(f"{'='*60}")

    image_paths, _ = load_test_set(testset_path)
    print(f"✓ Loaded {len(image_paths)} images from {testset_path}")

    if not image_paths:
        print("❌ No images found in test set!")
        return None

    print("\n🔄 Running face detection...")
    detections_map, output_dir_used = run_face_detection_on_test_set(
        detector_model,
        image_paths,
        output_dir if save_images else None,
    )

    total_faces = sum(len(v) for v in detections_map.values())
    max_faces = max((len(v) for v in detections_map.values()), default=0)
    print(f"\nTotal detected faces: {total_faces}")
    print(f"Max faces in a single image: {max_faces}")
    if save_images and output_dir_used:
        print(f"Annotated images saved under: {output_dir_used}")

    print(f"{'='*60}\n")
    return detections_map


def main():
    parser = argparse.ArgumentParser(
        description="Run model tests without Streamlit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Legacy models (backward compatible)
  python run_tests.py --quick
  python run_tests.py --model cnn
  python run_tests.py --model vit
  
  # New modular architecture (with alignment)
  python run_tests.py --model insightface
  python run_tests.py --model insightface_buffalo_m
  
  # Test all models (legacy + modular variants)
  python run_tests.py --model all
  
  # Use custom testset
  python run_tests.py --model insightface --testset testsets/hughJackmanTest.json
  
  # Use custom celebrities file
  python run_tests.py --model cnn --celebrities data/custom_celebs.json
  
  # Detection-only test
  python run_tests.py --detector-only --detector-model cnn
        """
    )
    
    parser.add_argument(
        "--model",
        type=str,
        choices=["cnn", "hog", "vit", "insightface", "insightface_buffalo_m", "insightface_buffalo_s", "unified", "all"],
        default="hog",
        help="Model to test (default: hog). Use 'unified' for custom detection+embedding+matching combinations.",
    )

    parser.add_argument(
        "--detection",
        type=str,
        default=None,
        help="Detection model for 'unified' mode: buffalo_l, buffalo_m, buffalo_s, cnn, hog, antelopev2",
    )

    parser.add_argument(
        "--embedding",
        type=str,
        default=None,
        help="Embedding model for 'unified' mode: insightface, insightface_buffalo_m, insightface_buffalo_s, face_recognition, vit",
    )

    parser.add_argument(
        "--matching",
        type=str,
        default=None,
        help="Matching method for 'unified' mode: cosine_similarity, euclidean_distance, l2_distance",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Similarity threshold for 'unified' mode (0.0-1.0, default varies by embedding)",
    )

    parser.add_argument(
        "--detector-only",
        action="store_true",
        help="Run detection-only test (skip recognition)",
    )

    parser.add_argument(
        "--detector-model",
        type=str,
        choices=["cnn", "hog"],
        default="cnn",
        help="Face detector model for detection-only mode (default: cnn)",
    )
    
    parser.add_argument(
        "--testset",
        type=str,
        default="testsets/test_set.json",
        help="Path to test set JSON (default: testsets/test_set.json)",
    )
    
    parser.add_argument(
        "--celebrities",
        type=str,
        default="data/celebrities.json",
        help="Path to celebrities JSON (default: data/celebrities.json)",
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="image_outputs",
        help="Directory to save output images (default: image_outputs)",
    )

    parser.add_argument(
        "--no-images",
        action="store_true",
        help="Disable saving annotated images (images saved by default)",
    )
    
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick test with HOG model (same as --model hog)",
    )

    parser.add_argument(
        "--upsample",
        type=int,
        default=None,
        help="Face detection upsample for CNN/HOG (e.g., 2)",
    )

    parser.add_argument(
        "--alignment",
        action="store_true",
        help="Enable landmark-based alignment for InsightFace (enabled by default for insightface models)",
    )
    
    args = parser.parse_args()
    
    # Handle quick mode
    if args.quick:
        args.model = "hog"
        print("🚀 Quick test mode: Using HOG model for fast testing\n")

    # Apply upsample override if provided
    if args.upsample is not None:
        os.environ["FR_UPSAMPLE"] = str(args.upsample)

    # Detector-only run short-circuits classification
    if args.detector_only:
        run_detector_test(
            args.detector_model,
            args.testset,
            args.output_dir,
            save_images=not args.no_images,
        )
        print("✅ Detection-only testing completed successfully!")
        return
    
    # Map model names
    model_map = {
        "cnn": "face_recognition_cnn",
        "hog": "face_recognition_hog",
        "vit": "vit_b32",
        "insightface": "insightface",
        "insightface_buffalo_m": "insightface_buffalo_m",
        "insightface_buffalo_s": "insightface_buffalo_s",
    }
    
    try:
        if args.model == "unified":
            # UNIFIED MODE: fully modular combination
            if not args.detection or not args.embedding or not args.matching:
                logger.error("Unified mode requires --detection, --embedding, and --matching parameters")
                print("❌ Error: Unified mode requires --detection, --embedding, and --matching")
                print("   Example: python run_tests.py --model unified --detection buffalo_l --embedding insightface --matching cosine_similarity")
                sys.exit(1)
            
            print(f"\n{'='*80}")
            print(f"🔧 MODULAR TEST: detection={args.detection} | embedding={args.embedding} | matching={args.matching} | threshold={args.threshold or 'default'}")
            print(f"{'='*80}\n")
            
            # Load test set
            image_paths, ground_truth = load_test_set(args.testset)
            print(f"✓ Loaded {len(image_paths)} images from {args.testset}")
            
            if not image_paths:
                print("❌ No images found in test set!")
                return None
            
            # Load celebrity data
            celebrities = load_celebrities_from_json(args.celebrities)
            if not celebrities:
                print(f"❌ No celebrity data loaded from {args.celebrities}")
                return None
            print(f"✓ Loaded {len(celebrities)} celebrities")
            
            # Initialize classifier with modular parameters
            classifier = get_classifier(
                "unified",
                celebrities,
                detection_model=args.detection,
                embedding_model=args.embedding,
                matching_method=args.matching,
                threshold=args.threshold or 0.6,
            )
            
            if classifier is None:
                print(f"❌ Could not initialize classifier with given parameters")
                return None
            
            # Run predictions
            print(f"\n🔄 Running predictions...")
            predictions = run_classification_on_test_set(
                classifier,
                image_paths,
                output_image_dir=args.output_dir if not args.no_images else None,
            )
            
            # Calculate metrics
            metrics = calculate_metrics(ground_truth, predictions)
            
            # Save results
            csv_path = save_test_output_to_csv(
                image_paths, predictions, ground_truth, 
                f"unified_{args.detection}_{args.embedding}_{args.matching}"
            )
            
            # Display results
            print(f"\n{'='*80}")
            print(f"TEST RESULTS: Modular Pipeline")
            print(f"{'='*80}")
            print(f"Detection:    {args.detection}")
            print(f"Embedding:    {args.embedding}")
            print(f"Matching:     {args.matching}")
            print(f"Threshold:    {args.threshold or 'default'}")
            print(f"{'-'*80}")
            for metric, value in metrics.items():
                metric_label = metric.replace("_", " ").title()
                print(f"{metric_label:25s}: {value:.4f}" if isinstance(value, float) else f"{metric_label:25s}: {value}")
            print(f"\n📄 Results saved to: {csv_path}")
            print(f"{'='*80}\n")
            
            return metrics
        
        elif args.model == "all":
            print("🔄 Running tests with all models...\n")
            results = {}
            # Test legacy models
            legacy_models = ["cnn", "hog", "vit"]
            # Test new modular InsightFace variants
            modular_models = ["insightface", "insightface_buffalo_m", "insightface_buffalo_s"]
            
            all_models = legacy_models + modular_models
            
            for short_name in all_models:
                full_name = model_map[short_name]
                metrics = run_test(
                    full_name,
                    args.testset,
                    args.celebrities,
                    args.output_dir,
                    save_images=not args.no_images,
                )
                if metrics:
                    results[short_name] = metrics
            
            # Summary comparison
            if len(results) > 1:
                print(f"\n{'='*80}")
                print("SUMMARY COMPARISON - ALL MODELS")
                print(f"{'='*80}")
                print(f"{'Model':<25} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1 Score':<12}")
                print("-" * 80)
                for model, metrics in results.items():
                    print(f"{model.upper():<25} {metrics.get('accuracy', 0):<12.4f} "
                          f"{metrics.get('precision', 0):<12.4f} "
                          f"{metrics.get('recall', 0):<12.4f} "
                          f"{metrics.get('f1_score', 0):<12.4f}")
                
                # Separate legacy from modular
                print(f"\n{'LEGACY MODELS':<25} {'─'*56}")
                legacy_results = {k: v for k, v in results.items() if k in legacy_models}
                if legacy_results:
                    for model, metrics in legacy_results.items():
                        print(f"{model.upper():<25} Acc: {metrics.get('accuracy', 0):.4f} | "
                              f"Prec: {metrics.get('precision', 0):.4f} | F1: {metrics.get('f1_score', 0):.4f}")
                
                print(f"\n{'MODULAR (InsightFace + Alignment)':<25} {'─'*56}")
                modular_results = {k: v for k, v in results.items() if k in modular_models}
                if modular_results:
                    for model, metrics in modular_results.items():
                        print(f"{model.upper():<25} Acc: {metrics.get('accuracy', 0):.4f} | "
                              f"Prec: {metrics.get('precision', 0):.4f} | F1: {metrics.get('f1_score', 0):.4f}")
                
                print(f"{'='*80}\n")
        else:
            full_model_name = model_map[args.model]
            run_test(
                full_model_name,
                args.testset,
                args.celebrities,
                args.output_dir,
                save_images=not args.no_images,
            )
        
        print("✅ Testing completed successfully!")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Testing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        logger.error(f"Test error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
