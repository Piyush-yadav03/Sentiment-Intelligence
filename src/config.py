"""Centralized configuration for SentimentLab AI"""

from pathlib import Path
from dataclasses import dataclass
from typing import Optional

# Project paths
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
RESULTS = ROOT / "results"
ARTIFACTS = ROOT / "artifacts"

# Model configuration
TEACHER_MODEL = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
DATASET_NAME = "imdb"

# Default training hyperparameters
DEFAULT_CONFIG = {
    "train_size": 800,
    "test_size": 300,
    "epochs": 3,
    "batch_size": 8,
    "learning_rate": 3e-4,
    "weight_decay": 0.01,
    "max_length": 96,
    "temperature": 2.0,
    "alpha": 0.5,  # Weight for hard-label loss
    "seed": 42,
    "gradient_clip": 1.0,
}

# Student model architecture
STUDENT_CONFIG = {
    "d_model": 128,
    "nhead": 4,
    "num_layers": 2,
    "dim_feedforward": 256,
    "dropout": 0.1,
}

# Smoke test configuration (for quick validation)
SMOKE_TEST_CONFIG = {
    "train_size": 200,
    "test_size": 100,
    "epochs": 1,
    "batch_size": 4,
    "max_length": 96,
}


@dataclass
class ExperimentConfig:
    """Configuration for a training experiment"""
    train_size: int = DEFAULT_CONFIG["train_size"]
    test_size: int = DEFAULT_CONFIG["test_size"]
    epochs: int = DEFAULT_CONFIG["epochs"]
    batch_size: int = DEFAULT_CONFIG["batch_size"]
    learning_rate: float = DEFAULT_CONFIG["learning_rate"]
    weight_decay: float = DEFAULT_CONFIG["weight_decay"]
    max_length: int = DEFAULT_CONFIG["max_length"]
    temperature: float = DEFAULT_CONFIG["temperature"]
    alpha: float = DEFAULT_CONFIG["alpha"]
    seed: int = DEFAULT_CONFIG["seed"]
    gradient_clip: float = DEFAULT_CONFIG["gradient_clip"]
    
    def validate(self):
        """Validate configuration parameters"""
        if self.train_size < 2:
            raise ValueError("train_size must be >= 2")
        if self.test_size < 2:
            raise ValueError("test_size must be >= 2")
        if self.epochs < 1:
            raise ValueError("epochs must be >= 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if not 0 <= self.alpha <= 1:
            raise ValueError("alpha must be between 0 and 1")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if self.max_length < 1:
            raise ValueError("max_length must be >= 1")
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            "train_size": self.train_size,
            "test_size": self.test_size,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "max_length": self.max_length,
            "temperature": self.temperature,
            "alpha": self.alpha,
            "seed": self.seed,
            "gradient_clip": self.gradient_clip,
        }
