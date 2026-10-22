"""Tests for student model architecture"""

import sys
from pathlib import Path
import torch
import pytest

# Add src to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from model import TinyTransformerStudent


def test_model_creation():
    """Test model can be created with valid parameters"""
    vocab_size = 1000
    pad_id = 0
    max_length = 96
    
    model = TinyTransformerStudent(
        vocab_size=vocab_size,
        pad_id=pad_id,
        max_length=max_length
    )
    
    assert model is not None
    assert isinstance(model, torch.nn.Module)


def test_model_forward_pass():
    """Test model forward pass with valid inputs"""
    vocab_size = 1000
    pad_id = 0
    max_length = 96
    batch_size = 4
    seq_length = 50
    
    model = TinyTransformerStudent(vocab_size, pad_id, max_length)
    model.eval()
    
    # Create dummy inputs
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
    attention_mask = torch.ones(batch_size, seq_length)
    
    # Forward pass
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
    
    # Check output shape
    assert logits.shape == (batch_size, 2), f"Expected shape ({batch_size}, 2), got {logits.shape}"


def test_model_output_type():
    """Test model outputs are tensors with correct properties"""
    model = TinyTransformerStudent(vocab_size=1000, pad_id=0, max_length=96)
    model.eval()
    
    input_ids = torch.randint(0, 1000, (2, 50))
    attention_mask = torch.ones(2, 50)
    
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
    
    assert isinstance(logits, torch.Tensor)
    assert logits.dtype == torch.float32
    assert not torch.isnan(logits).any(), "Output contains NaN"
    assert not torch.isinf(logits).any(), "Output contains Inf"


def test_model_with_padding():
    """Test model handles padding correctly"""
    model = TinyTransformerStudent(vocab_size=1000, pad_id=0, max_length=96)
    model.eval()
    
    batch_size = 2
    seq_length = 50
    
    # Create inputs with padding
    input_ids = torch.randint(1, 1000, (batch_size, seq_length))
    input_ids[:, 40:] = 0  # Pad last 10 tokens
    
    attention_mask = torch.ones(batch_size, seq_length)
    attention_mask[:, 40:] = 0  # Mask padding
    
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
    
    assert logits.shape == (batch_size, 2)
    assert not torch.isnan(logits).any()


def test_model_handles_max_length_truncation():
    """Test model truncates sequences longer than max_length"""
    max_length = 96
    model = TinyTransformerStudent(vocab_size=1000, pad_id=0, max_length=max_length)
    model.eval()
    
    # Create input longer than max_length
    long_seq_length = 150
    input_ids = torch.randint(0, 1000, (2, long_seq_length))
    attention_mask = torch.ones(2, long_seq_length)
    
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
    
    # Should still produce valid output
    assert logits.shape == (2, 2)
    assert not torch.isnan(logits).any()


def test_model_parameter_count():
    """Test model has reasonable parameter count"""
    model = TinyTransformerStudent(vocab_size=1000, pad_id=0, max_length=96)
    
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Should be compact (less than 1M parameters)
    assert param_count < 1_000_000, f"Model too large: {param_count:,} parameters"
    assert param_count > 10_000, f"Model too small: {param_count:,} parameters"


def test_model_gradient_flow():
    """Test gradients flow through the model"""
    model = TinyTransformerStudent(vocab_size=1000, pad_id=0, max_length=96)
    model.train()
    
    input_ids = torch.randint(0, 1000, (2, 50))
    attention_mask = torch.ones(2, 50)
    labels = torch.tensor([0, 1])
    
    # Forward pass
    logits = model(input_ids, attention_mask)
    loss = torch.nn.functional.cross_entropy(logits, labels)
    
    # Backward pass
    loss.backward()
    
    # Check gradients exist
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"


def test_model_different_batch_sizes():
    """Test model works with different batch sizes"""
    model = TinyTransformerStudent(vocab_size=1000, pad_id=0, max_length=96)
    model.eval()
    
    batch_sizes = [1, 4, 8, 16]
    seq_length = 50
    
    for batch_size in batch_sizes:
        input_ids = torch.randint(0, 1000, (batch_size, seq_length))
        attention_mask = torch.ones(batch_size, seq_length)
        
        with torch.no_grad():
            logits = model(input_ids, attention_mask)
        
        assert logits.shape == (batch_size, 2), f"Failed for batch_size={batch_size}"


def test_model_reproducibility():
    """Test model produces same output with same inputs"""
    torch.manual_seed(42)
    model = TinyTransformerStudent(vocab_size=1000, pad_id=0, max_length=96)
    model.eval()
    
    input_ids = torch.randint(0, 1000, (2, 50))
    attention_mask = torch.ones(2, 50)
    
    with torch.no_grad():
        output1 = model(input_ids, attention_mask)
        output2 = model(input_ids, attention_mask)
    
    assert torch.allclose(output1, output2), "Model output not reproducible"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
