"""Inference utilities for trained models"""

import time
from pathlib import Path
from typing import Dict, Optional, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, PreTrainedTokenizer

from model import TinyTransformerStudent


class SentimentPredictor:
    """Unified interface for sentiment prediction"""
    
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 96,
        device: torch.device = None,
    ):
        """
        Args:
            model: Trained model
            tokenizer: Tokenizer
            max_length: Maximum sequence length
            device: Device to run inference on
        """
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device or torch.device("cpu")
        
        self.model.to(self.device)
        self.model.eval()
    
    @torch.no_grad()
    def predict(self, text: str) -> Dict[str, float]:
        """
        Predict sentiment for a single text.
        
        Args:
            text: Input text
        
        Returns:
            Dictionary with 'negative' and 'positive' probabilities
        """
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding="max_length"
        )
        
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        
        # Forward pass
        start_time = time.perf_counter()
        logits = self.model(input_ids, attention_mask)
        inference_time = (time.perf_counter() - start_time) * 1000  # ms
        
        # Get probabilities
        probs = torch.softmax(logits[0], dim=-1).cpu().numpy()
        
        return {
            "negative": float(probs[0]),
            "positive": float(probs[1]),
            "latency_ms": inference_time,
        }
    
    def predict_label(self, text: str) -> Tuple[str, float]:
        """
        Predict sentiment label and confidence.
        
        Returns:
            Tuple of (label, confidence)
        """
        result = self.predict(text)
        
        if result["positive"] >= result["negative"]:
            return "Positive", result["positive"]
        else:
            return "Negative", result["negative"]


def load_teacher_model(
    model_name: str = "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    device: torch.device = None,
) -> Tuple[PreTrainedTokenizer, torch.nn.Module]:
    """
    Load pre-trained teacher model.
    
    Args:
        model_name: HuggingFace model name
        device: Device to load model on
    
    Returns:
        Tuple of (tokenizer, model)
    """
    device = device or torch.device("cpu")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(device)
    model.eval()
    
    return tokenizer, model


def load_student_model(
    checkpoint_path: Path,
    tokenizer_path: Path,
    max_length: int = 96,
    device: torch.device = None,
) -> Tuple[PreTrainedTokenizer, TinyTransformerStudent]:
    """
    Load trained student model from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint (.pt file)
        tokenizer_path: Path to tokenizer directory
        max_length: Maximum sequence length used during training
        device: Device to load model on
    
    Returns:
        Tuple of (tokenizer, model)
    """
    device = device or torch.device("cpu")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    # Create model
    model = TinyTransformerStudent(
        vocab_size=len(tokenizer),
        pad_id=tokenizer.pad_token_id or 0,
        max_length=max_length
    )
    
    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    
    return tokenizer, model


def create_predictor_from_checkpoint(
    checkpoint_path: Path,
    tokenizer_path: Path,
    max_length: int = 96,
    device: torch.device = None,
) -> Optional[SentimentPredictor]:
    """
    Create a SentimentPredictor from saved checkpoint.
    
    Returns None if checkpoint doesn't exist.
    """
    if not checkpoint_path.exists() or not tokenizer_path.exists():
        return None
    
    try:
        tokenizer, model = load_student_model(
            checkpoint_path,
            tokenizer_path,
            max_length,
            device
        )
        return SentimentPredictor(model, tokenizer, max_length, device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
