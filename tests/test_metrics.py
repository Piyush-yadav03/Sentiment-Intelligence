"""Tests for metrics computation and tracking"""

import sys
from pathlib import Path
import json
import tempfile
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from metrics import (
    MetricsTracker,
    load_metrics,
    compute_model_comparison,
    format_metric,
)


def test_metrics_tracker_creation():
    """Test MetricsTracker can be created"""
    tracker = MetricsTracker()
    assert tracker is not None
    assert tracker.metrics == {}


def test_metrics_tracker_add_config():
    """Test adding experiment configuration"""
    tracker = MetricsTracker()
    config = {
        "train_size": 800,
        "test_size": 300,
        "epochs": 3,
        "temperature": 2.0,
    }
    
    tracker.add_experiment_config(config)
    
    assert "train_size" in tracker.metrics
    assert tracker.metrics["train_size"] == 800
    assert tracker.metrics["temperature"] == 2.0


def test_metrics_tracker_add_model_metrics():
    """Test adding model-specific metrics"""
    tracker = MetricsTracker()
    
    baseline_metrics = {
        "accuracy": 0.85,
        "macro_f1": 0.83,
        "latency_ms_per_sample": 5.2,
    }
    
    tracker.add_model_metrics("baseline_student", baseline_metrics)
    
    assert "baseline_student" in tracker.metrics
    assert tracker.metrics["baseline_student"]["accuracy"] == 0.85


def test_metrics_tracker_add_limitations():
    """Test adding research limitations"""
    tracker = MetricsTracker()
    
    tracker.add_limitation("Domain shift between teacher and evaluation data")
    tracker.add_limitation("Results based on subset, not full dataset")
    
    assert "limitations" in tracker.metrics
    assert len(tracker.metrics["limitations"]) == 2


def test_metrics_tracker_save_load():
    """Test saving and loading metrics"""
    tracker = MetricsTracker()
    tracker.add_experiment_config({"train_size": 100, "test_size": 50})
    tracker.add_model_metrics("student", {"accuracy": 0.9})
    
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "metrics.json"
        
        # Save
        tracker.save(path)
        assert path.exists()
        
        # Load
        loaded = load_metrics(path)
        assert loaded is not None
        assert loaded["train_size"] == 100
        assert loaded["student"]["accuracy"] == 0.9


def test_metrics_tracker_get_summary():
    """Test getting human-readable summary"""
    tracker = MetricsTracker()
    tracker.add_experiment_config({"train_examples": 800, "test_examples": 300})
    tracker.add_model_metrics("baseline_student", {
        "accuracy": 0.85,
        "macro_f1": 0.83,
        "latency_ms_per_sample": 5.2,
        "checkpoint_mb": 1.5,
    })
    
    summary = tracker.get_summary()
    
    assert isinstance(summary, str)
    assert "800" in summary  # train examples
    assert "85.00%" in summary  # accuracy


def test_load_metrics_missing_file():
    """Test loading metrics from non-existent file"""
    result = load_metrics(Path("/nonexistent/metrics.json"))
    assert result is None


def test_compute_model_comparison():
    """Test computing comparison metrics"""
    baseline = {
        "accuracy": 0.80,
        "macro_f1": 0.78,
        "latency_ms_per_sample": 10.0,
        "checkpoint_mb": 2.0,
    }
    
    distilled = {
        "accuracy": 0.82,
        "macro_f1": 0.80,
        "latency_ms_per_sample": 8.0,
        "checkpoint_mb": 1.5,
    }
    
    comparison = compute_model_comparison(baseline, distilled)
    
    assert "accuracy_diff" in comparison
    assert "f1_diff" in comparison
    assert "latency_speedup" in comparison
    assert "size_reduction_pct" in comparison
    
    # Check values
    assert comparison["accuracy_diff"] == pytest.approx(0.02, abs=1e-6)
    assert comparison["latency_speedup"] == pytest.approx(1.25, abs=1e-6)
    assert comparison["size_reduction_pct"] == pytest.approx(25.0, abs=1e-6)


def test_compute_model_comparison_zero_latency():
    """Test comparison handles zero latency gracefully"""
    baseline = {"latency_ms_per_sample": 10.0}
    distilled = {"latency_ms_per_sample": 0.0}
    
    comparison = compute_model_comparison(baseline, distilled)
    
    # Should not crash and should give default speedup
    assert "latency_speedup" in comparison
    assert comparison["latency_speedup"] == 1.0


def test_compute_model_comparison_zero_size():
    """Test comparison handles zero size gracefully"""
    baseline = {"checkpoint_mb": 0.0}
    distilled = {"checkpoint_mb": 1.5}
    
    comparison = compute_model_comparison(baseline, distilled)
    
    assert "size_reduction_pct" in comparison
    assert comparison["size_reduction_pct"] == 0.0


def test_format_metric_percent():
    """Test formatting percentage metrics"""
    result = format_metric(0.8542, "percent")
    assert result == "85.42%"


def test_format_metric_latency():
    """Test formatting latency metrics"""
    result = format_metric(5.234, "latency")
    assert result == "5.23 ms"


def test_format_metric_size():
    """Test formatting size metrics"""
    result = format_metric(1.567, "size")
    assert result == "1.57 MB"


def test_format_metric_throughput():
    """Test formatting throughput metrics"""
    result = format_metric(123.456, "throughput")
    assert result == "123.46 samples/s"


def test_format_metric_default():
    """Test formatting with default format"""
    result = format_metric(0.123456, "other")
    assert result == "0.1235"


def test_metrics_tracker_set_limitations():
    """Test setting all limitations at once"""
    tracker = MetricsTracker()
    
    limitations = [
        "Limitation 1",
        "Limitation 2",
        "Limitation 3"
    ]
    
    tracker.set_limitations(limitations)
    
    assert tracker.metrics["limitations"] == limitations


def test_metrics_json_serializable():
    """Test that metrics can be serialized to JSON"""
    tracker = MetricsTracker()
    tracker.add_experiment_config({"train_size": 100})
    tracker.add_model_metrics("student", {"accuracy": 0.9})
    
    # Should not raise
    json_str = json.dumps(tracker.metrics)
    assert isinstance(json_str, str)
    
    # Should be loadable
    loaded = json.loads(json_str)
    assert loaded["train_size"] == 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
