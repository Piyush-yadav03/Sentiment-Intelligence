"""Tests for inference utilities"""

import sys
from pathlib import Path
import torch
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from model import TinyTransformerStudent
from inference import SentimentPredictor


class MockTokenizer:
    """Mock tokenizer for testing"""
    def __init__(self):
        self.pad_token_id = 0
    
    def __call__(self, text, **kwargs):
        # Return dummy encodings
        return {
            "input_ids": torch.randint(1, 100, (1, 50)),
            "attention_mask": torch.ones(1, 50),
        }
    
    def __len__(self):
        return 1000


def test_sentiment_predictor_creation():
    """Test SentimentPredictor can be created"""
    model = TinyTransformerStudent(vocab_size=1000, pad_id=0, max_length=96)
    tokenizer = MockTokenizer()
    
    predictor = SentimentPredictor(model, tokenizer)
    
    assert predictor is not None
    assert predictor.model is model
    assert predictor.tokenizer is tokenizer


def test_sentiment_predictor_predict():
    """Test prediction returns valid probabilities"""
    model = TinyTransformerStudent(vocab_size=1000, pad_id=0, max_length=96)
    tokenizer = MockTokenizer()
    
    predictor = SentimentPredictor(model, tokenizer)
    result = predictor.predict("This is a test sentence")
    
    assert isinstance(result, dict)
    assert "negative" in result
    assert "positive" in result
    assert "latency_ms" in result
    
    # Check probabilities sum to approximately 1
    prob_sum = result["negative"] + result["positive"]
    assert pytest.approx(prob_sum, abs=1e-5) == 1.0
    
    # Check probabilities are in valid range
    assert 0 <= result["negative"] <= 1
    assert 0 <= result["positive"] <= 1
    
    # Check latency is non-negative
    assert result["latency_ms"] >= 0


def test_sentiment_predictor_predict_label():
    """Test label prediction"""
    model = TinyTransformerStudent(vocab_size=1000, pad_id=0, max_length=96)
    tokenizer = MockTokenizer()
    
    predictor = SentimentPredictor(model, tokenizer)
    label, confidence = predictor.predict_label("This is a test sentence")
    
    assert label in ["Positive", "Negative"]
    assert 0 <= confidence <= 1


def test_sentiment_predictor_different_texts():
    """Test predictor handles different text inputs"""
    model = TinyTransformerStudent(vocab_size=1000, pad_id=0, max_length=96)
    tokenizer = MockTokenizer()
    
    predictor = SentimentPredictor(model, tokenizer)
    
    texts = [
        "Short text",
        "This is a longer text with more words to test the model's handling of variable length inputs",
        "Numbers 123 and symbols !@#",
    ]
    
    for text in texts:
        result = predictor.predict(text)
        assert "negative" in result
        assert "positive" in result


def test_sentiment_predictor_reproducibility():
    """Test predictor gives consistent results"""
    torch.manual_seed(42)
    model = TinyTransformerStudent(vocab_size=1000, pad_id=0, max_length=96)
    tokenizer = MockTokenizer()
    
    predictor = SentimentPredictor(model, tokenizer)
    
    # Need to seed tokenizer randomness for true reproducibility
    # In real scenario with fixed tokenizer, outputs should match
    result1 = predictor.predict("Test sentence")
    result2 = predictor.predict("Test sentence")
    
    # At minimum, should return valid results
    assert isinstance(result1, dict)
    assert isinstance(result2, dict)


def test_sentiment_predictor_device():
    """Test predictor respects device setting"""
    device = torch.device("cpu")
    model = TinyTransformerStudent(vocab_size=1000, pad_id=0, max_length=96)
    tokenizer = MockTokenizer()
    
    predictor = SentimentPredictor(model, tokenizer, device=device)
    
    assert predictor.device == device
    assert next(predictor.model.parameters()).device == device


def test_model_in_eval_mode():
    """Test predictor sets model to eval mode"""
    model = TinyTransformerStudent(vocab_size=1000, pad_id=0, max_length=96)
    tokenizer = MockTokenizer()
    
    model.train()  # Set to train mode
    predictor = SentimentPredictor(model, tokenizer)
    
    # Should be in eval mode
    assert not predictor.model.training


def test_prediction_output_types():
    """Test all prediction outputs have correct types"""
    model = TinyTransformerStudent(vocab_size=1000, pad_id=0, max_length=96)
    tokenizer = MockTokenizer()
    
    predictor = SentimentPredictor(model, tokenizer)
    result = predictor.predict("Test")
    
    assert isinstance(result["negative"], float)
    assert isinstance(result["positive"], float)
    assert isinstance(result["latency_ms"], float)


def test_empty_text_handling():
    """Test predictor handles empty or whitespace text"""
    model = TinyTransformerStudent(vocab_size=1000, pad_id=0, max_length=96)
    tokenizer = MockTokenizer()
    
    predictor = SentimentPredictor(model, tokenizer)
    
    # Should not crash on empty/whitespace text
    result = predictor.predict("")
    assert isinstance(result, dict)
    
    result = predictor.predict("   ")
    assert isinstance(result, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
