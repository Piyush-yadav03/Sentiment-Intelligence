"""Evaluate and compare trained models"""

import sys
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import RESULTS
from model import TinyTransformerStudent
from evaluate import evaluate_model, measure_latency, get_classification_report
from metrics import MetricsTracker, compute_model_comparison
from utils import seed_everything, get_checkpoint_size_mb


def load_model_and_tokenizer(checkpoint_path, tokenizer_path, max_length=96):
    """Load trained student model"""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    model = TinyTransformerStudent(
        vocab_size=len(tokenizer),
        pad_id=tokenizer.pad_token_id or 0,
        max_length=max_length
    )
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    
    return model, tokenizer


def main():
    seed_everything(42)
    device = torch.device("cpu")
    
    print("=" * 60)
    print("MODEL EVALUATION AND COMPARISON")
    print("=" * 60)
    
    # Check if models exist
    baseline_path = RESULTS / "student_baseline.pt"
    distilled_path = RESULTS / "student_distilled.pt"
    tokenizer_path = RESULTS / "tokenizer"
    metrics_path = RESULTS / "metrics.json"
    
    if not baseline_path.exists():
        print("❌ Baseline model not found. Run train_baseline.py first.")
        return
    
    if not distilled_path.exists():
        print("❌ Distilled model not found. Run train_distilled.py first.")
        return
    
    if not tokenizer_path.exists():
        print("❌ Tokenizer not found. Run training scripts first.")
        return
    
    # Load existing metrics if available
    if metrics_path.exists():
        with open(metrics_path, "r") as f:
            existing_metrics = json.load(f)
        print(f"\nLoaded existing metrics from {metrics_path}")
        
        # Extract test data info
        test_size = existing_metrics.get("test_examples", 300)
        max_length = existing_metrics.get("max_length", 96)
    else:
        print("\n⚠️  No existing metrics found. Using defaults.")
        test_size = 300
        max_length = 96
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    # Load test data (need to recreate from same parameters)
    print(f"\nNote: To evaluate, you need test data. This script shows how to")
    print(f"load and compare already-trained models. Full evaluation requires")
    print(f"running src/train.py which trains both models and evaluates them.")
    
    # Load models
    print("\nLoading baseline model...")
    baseline_model, _ = load_model_and_tokenizer(baseline_path, tokenizer_path, max_length)
    baseline_model.to(device)
    
    print("Loading distilled model...")
    distilled_model, _ = load_model_and_tokenizer(distilled_path, tokenizer_path, max_length)
    distilled_model.to(device)
    
    # Get checkpoint sizes
    baseline_size = get_checkpoint_size_mb(baseline_path)
    distilled_size = get_checkpoint_size_mb(distilled_path)
    
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    print(f"\nBaseline checkpoint: {baseline_size:.2f} MB")
    print(f"Distilled checkpoint: {distilled_size:.2f} MB")
    
    if metrics_path.exists():
        print("\nLoaded metrics from previous training run:")
        print(json.dumps(existing_metrics, indent=2))
        
        # Compute comparison
        if "baseline_student" in existing_metrics and "distilled_student" in existing_metrics:
            comparison = compute_model_comparison(
                existing_metrics["baseline_student"],
                existing_metrics["distilled_student"]
            )
            
            print("\n" + "=" * 60)
            print("PERFORMANCE COMPARISON")
            print("=" * 60)
            
            if "accuracy_diff" in comparison:
                print(f"Accuracy difference: {comparison['accuracy_diff_pct']:+.2f}%")
            
            if "f1_diff" in comparison:
                print(f"Macro-F1 difference: {comparison['f1_diff_pct']:+.2f}%")
            
            if "latency_speedup" in comparison:
                print(f"Latency speedup: {comparison['latency_speedup']:.2f}x")
            
            if "size_reduction_pct" in comparison:
                print(f"Size reduction: {comparison['size_reduction_pct']:.1f}%")
    else:
        print("\n⚠️  Run src/train.py to generate full evaluation metrics")
    
    print("\n" + "=" * 60)
    print("✅ Evaluation complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Review results in results/metrics.json")
    print("2. Launch Streamlit app: streamlit run app.py")
    print("3. Explore Model Comparison page for visualizations")


if __name__ == "__main__":
    main()
