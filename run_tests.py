#!/usr/bin/env python3
"""
Quick test runner script for running model tests without Streamlit.
Usage:
    python run_tests.py --model cnn --testset testsets/test_set.json
    python run_tests.py --model all
    python run_tests.py --quick
"""

import argparse
import json
import sys
from src.metrics import (
    load_test_set,
    run_classification_on_test_set,
    calculate_metrics,
    save_test_output_to_csv,
)
from src.classification import get_classifier, load_celebrities_from_json
from src.logging_utils import setup_logger

logger = setup_logger()


def run_test(model_name, testset_path, celebrities_path, output_dir="test_outputs"):
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
        classifier, image_paths, output_image_dir=output_dir
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


def main():
    parser = argparse.ArgumentParser(
        description="Run model tests without Streamlit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run quick test (HOG model)
  python run_tests.py --quick
  
  # Test specific model
  python run_tests.py --model cnn
  python run_tests.py --model vit
  
  # Test all models
  python run_tests.py --model all
  
  # Use custom testset
  python run_tests.py --model hog --testset testsets/hughJackmanTest.json
  
  # Use custom celebrities file
  python run_tests.py --model cnn --celebrities data/custom_celebs.json
        """
    )
    
    parser.add_argument(
        "--model",
        type=str,
        choices=["cnn", "hog", "vit", "all"],
        default="hog",
        help="Model to test (default: hog)",
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
        default="test_outputs",
        help="Directory to save output images (default: test_outputs)",
    )
    
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick test with HOG model (same as --model hog)",
    )
    
    args = parser.parse_args()
    
    # Handle quick mode
    if args.quick:
        args.model = "hog"
        print("🚀 Quick test mode: Using HOG model for fast testing\n")
    
    # Map model names
    model_map = {
        "cnn": "face_recognition_cnn",
        "hog": "face_recognition_hog",
        "vit": "vit_b32",
    }
    
    try:
        if args.model == "all":
            print("🔄 Running tests with all models...\n")
            results = {}
            for short_name, full_name in model_map.items():
                metrics = run_test(
                    full_name, args.testset, args.celebrities, args.output_dir
                )
                if metrics:
                    results[short_name] = metrics
            
            # Summary comparison
            if len(results) > 1:
                print(f"\n{'='*60}")
                print("SUMMARY COMPARISON")
                print(f"{'='*60}")
                print(f"{'Model':<15} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1 Score':<12}")
                print("-" * 60)
                for model, metrics in results.items():
                    print(f"{model.upper():<15} {metrics.get('accuracy', 0):<12.4f} "
                          f"{metrics.get('precision', 0):<12.4f} "
                          f"{metrics.get('recall', 0):<12.4f} "
                          f"{metrics.get('f1_score', 0):<12.4f}")
                print(f"{'='*60}\n")
        else:
            full_model_name = model_map[args.model]
            run_test(full_model_name, args.testset, args.celebrities, args.output_dir)
        
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
