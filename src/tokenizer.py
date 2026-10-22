"""Tokenizer utilities for SentimentLab AI"""

from pathlib import Path
from typing import Optional
from transformers import AutoTokenizer, PreTrainedTokenizer


def load_or_create_tokenizer(
    model_name: str,
    save_path: Optional[Path] = None
) -> PreTrainedTokenizer:
    """
    Load tokenizer from model name or saved path.
    
    Args:
        model_name: HuggingFace model name
        save_path: Optional path to save/load tokenizer
    
    Returns:
        Loaded tokenizer
    """
    # Try to load from saved path first
    if save_path and save_path.exists():
        try:
            print(f"Loading tokenizer from {save_path}")
            return AutoTokenizer.from_pretrained(save_path)
        except Exception as e:
            print(f"Could not load tokenizer from {save_path}: {e}")
            print("Loading from model name instead...")
    
    # Load from model name
    print(f"Loading tokenizer from {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Save if path provided
    if save_path:
        save_path.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(save_path)
        print(f"Saved tokenizer to {save_path}")
    
    return tokenizer


def validate_tokenizer(tokenizer: PreTrainedTokenizer):
    """
    Validate tokenizer has required attributes.
    
    Args:
        tokenizer: Tokenizer to validate
    
    Raises:
        ValueError if tokenizer is invalid
    """
    if tokenizer.pad_token is None and tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer must have a pad_token or pad_token_id")
    
    if len(tokenizer) == 0:
        raise ValueError("Tokenizer vocabulary is empty")
    
    # Test tokenization
    try:
        test_output = tokenizer("Test sentence", return_tensors="pt")
        if "input_ids" not in test_output or "attention_mask" not in test_output:
            raise ValueError("Tokenizer output missing required keys")
    except Exception as e:
        raise ValueError(f"Tokenizer test failed: {e}")
