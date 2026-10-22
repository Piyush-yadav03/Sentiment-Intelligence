"""Metrics computation and tracking"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
from dataclasses import asdict


class MetricsTracker:
    """Track and save experiment metrics"""
    
    def __init__(self):
        self.metrics = {}
    
    def add_experiment_config(self, config: Dict[str, Any]):
        """Add experiment configuration"""
        self.metrics.update(config)
    
    def add_model_metrics(self, model_name: str, metrics: Dict[str, float]):
        """Add metrics for a specific model"""
        self.metrics[model_name] = metrics
    
    def add_limitation(self, limitation: str):
        """Add a research limitation"""
        if "limitations" not in self.metrics:
            self.metrics["limitations"] = []
        self.metrics["limitations"].append(limitation)
    
    def set_limitations(self, limitations: List[str]):
        """Set all limitations at once"""
        self.metrics["limitations"] = limitations
    
    def save(self, path: Path):
        """Save metrics to JSON file"""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.metrics, f, indent=2)
        print(f"Metrics saved to {path}")
    
    def load(self, path: Path) -> Dict[str, Any]:
        """Load metrics from JSON file"""
        if not path.exists():
            raise FileNotFoundError(f"Metrics file not found: {path}")
        
        with open(path, "r", encoding="utf-8") as f:
            self.metrics = json.load(f)
        return self.metrics
    
    def get_summary(self) -> str:
        """Get human-readable summary"""
        lines = ["=== Experiment Summary ==="]
        
        if "train_examples" in self.metrics:
            lines.append(f"Training examples: {self.metrics['train_examples']}")
        if "test_examples" in self.metrics:
            lines.append(f"Test examples: {self.metrics['test_examples']}")
        
        for model_name in ["baseline_student", "distilled_student"]:
            if model_name in self.metrics:
                m = self.metrics[model_name]
                lines.append(f"\n{model_name.replace('_', ' ').title()}:")
                lines.append(f"  Accuracy: {m.get('accuracy', 0)*100:.2f}%")
                lines.append(f"  Macro-F1: {m.get('macro_f1', 0)*100:.2f}%")
                lines.append(f"  Latency: {m.get('latency_ms_per_sample', 0):.2f} ms/sample")
                lines.append(f"  Size: {m.get('checkpoint_mb', 0):.2f} MB")
        
        return "\n".join(lines)


def load_metrics(path: Path) -> Optional[Dict[str, Any]]:
    """Load metrics from JSON file, return None if not found"""
    if not path.exists():
        return None
    
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading metrics: {e}")
        return None


def compute_model_comparison(
    baseline_metrics: Dict[str, float],
    distilled_metrics: Dict[str, float],
) -> Dict[str, float]:
    """
    Compute comparison metrics between baseline and distilled models.
    
    Returns:
        Dictionary with comparison metrics
    """
    comparison = {}
    
    # Accuracy difference
    if "accuracy" in baseline_metrics and "accuracy" in distilled_metrics:
        comparison["accuracy_diff"] = (
            distilled_metrics["accuracy"] - baseline_metrics["accuracy"]
        )
        comparison["accuracy_diff_pct"] = comparison["accuracy_diff"] * 100
    
    # F1 difference
    if "macro_f1" in baseline_metrics and "macro_f1" in distilled_metrics:
        comparison["f1_diff"] = (
            distilled_metrics["macro_f1"] - baseline_metrics["macro_f1"]
        )
        comparison["f1_diff_pct"] = comparison["f1_diff"] * 100
    
    # Latency speedup
    if "latency_ms_per_sample" in baseline_metrics and "latency_ms_per_sample" in distilled_metrics:
        baseline_latency = baseline_metrics["latency_ms_per_sample"]
        distilled_latency = distilled_metrics["latency_ms_per_sample"]
        
        if distilled_latency > 0:
            comparison["latency_speedup"] = baseline_latency / distilled_latency
        else:
            comparison["latency_speedup"] = 1.0
    
    # Size reduction
    if "checkpoint_mb" in baseline_metrics and "checkpoint_mb" in distilled_metrics:
        baseline_size = baseline_metrics["checkpoint_mb"]
        distilled_size = distilled_metrics["checkpoint_mb"]
        
        if baseline_size > 0:
            comparison["size_reduction_pct"] = (
                (baseline_size - distilled_size) / baseline_size * 100
            )
        else:
            comparison["size_reduction_pct"] = 0.0
    
    return comparison


def format_metric(value: float, metric_type: str = "percent") -> str:
    """Format a metric value for display"""
    if metric_type == "percent":
        return f"{value * 100:.2f}%"
    elif metric_type == "latency":
        return f"{value:.2f} ms"
    elif metric_type == "size":
        return f"{value:.2f} MB"
    elif metric_type == "throughput":
        return f"{value:.2f} samples/s"
    else:
        return f"{value:.4f}"
