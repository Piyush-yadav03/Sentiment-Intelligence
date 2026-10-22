"""Tests for knowledge distillation loss functions"""

import sys
from pathlib import Path
import torch
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from distillation import (
    distillation_loss,
    baseline_loss,
    validate_distillation_inputs,
)


def test_baseline_loss():
    """Test baseline cross-entropy loss"""
    batch_size = 4
    num_classes = 2
    
    logits = torch.randn(batch_size, num_classes)
    labels = torch.randint(0, num_classes, (batch_size,))
    
    loss = baseline_loss(logits, labels)
    
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0  # Scalar
    assert loss.item() >= 0  # Loss should be non-negative
    assert not torch.isnan(loss), "Loss is NaN"


def test_distillation_loss_basic():
    """Test distillation loss with valid inputs"""
    batch_size = 4
    num_classes = 2
    
    student_logits = torch.randn(batch_size, num_classes)
    teacher_logits = torch.randn(batch_size, num_classes)
    labels = torch.randint(0, num_classes, (batch_size,))
    
    total_loss, hard_loss, soft_loss = distillation_loss(
        student_logits,
        teacher_logits,
        labels,
        temperature=2.0,
        alpha=0.5
    )
    
    assert isinstance(total_loss, torch.Tensor)
    assert isinstance(hard_loss, torch.Tensor)
    assert isinstance(soft_loss, torch.Tensor)
    
    assert not torch.isnan(total_loss), "Total loss is NaN"
    assert not torch.isnan(hard_loss), "Hard loss is NaN"
    assert not torch.isnan(soft_loss), "Soft loss is NaN"
    
    assert total_loss.item() >= 0
    assert hard_loss.item() >= 0
    assert soft_loss.item() >= 0


def test_distillation_loss_alpha_boundaries():
    """Test distillation loss with alpha at boundaries"""
    batch_size = 4
    student_logits = torch.randn(batch_size, 2)
    teacher_logits = torch.randn(batch_size, 2)
    labels = torch.randint(0, 2, (batch_size,))
    
    # Alpha = 0 (only soft loss)
    total_loss_0, hard_loss_0, soft_loss_0 = distillation_loss(
        student_logits, teacher_logits, labels, temperature=2.0, alpha=0.0
    )
    assert torch.allclose(total_loss_0, soft_loss_0, rtol=1e-5)
    
    # Alpha = 1 (only hard loss)
    total_loss_1, hard_loss_1, soft_loss_1 = distillation_loss(
        student_logits, teacher_logits, labels, temperature=2.0, alpha=1.0
    )
    assert torch.allclose(total_loss_1, hard_loss_1, rtol=1e-5)


def test_distillation_loss_temperature_effect():
    """Test temperature affects soft loss"""
    batch_size = 4
    student_logits = torch.randn(batch_size, 2)
    teacher_logits = torch.randn(batch_size, 2)
    labels = torch.randint(0, 2, (batch_size,))
    
    _, _, soft_loss_t1 = distillation_loss(
        student_logits, teacher_logits, labels, temperature=1.0, alpha=0.5
    )
    
    _, _, soft_loss_t4 = distillation_loss(
        student_logits, teacher_logits, labels, temperature=4.0, alpha=0.5
    )
    
    # Different temperatures should give different soft losses
    assert not torch.allclose(soft_loss_t1, soft_loss_t4)


def test_distillation_loss_gradient_flow():
    """Test gradients flow through distillation loss"""
    student_logits = torch.randn(4, 2, requires_grad=True)
    teacher_logits = torch.randn(4, 2)
    labels = torch.randint(0, 2, (4,))
    
    total_loss, _, _ = distillation_loss(
        student_logits, teacher_logits, labels
    )
    
    total_loss.backward()
    
    assert student_logits.grad is not None
    assert not torch.isnan(student_logits.grad).any()


def test_validate_distillation_inputs_valid():
    """Test validation passes for valid inputs"""
    student_logits = torch.randn(4, 2)
    teacher_logits = torch.randn(4, 2)
    labels = torch.randint(0, 2, (4,))
    
    # Should not raise
    validate_distillation_inputs(student_logits, teacher_logits, labels)


def test_validate_distillation_inputs_shape_mismatch():
    """Test validation fails for shape mismatch"""
    student_logits = torch.randn(4, 2)
    teacher_logits = torch.randn(4, 3)  # Wrong number of classes
    labels = torch.randint(0, 2, (4,))
    
    with pytest.raises(ValueError, match="must match"):
        validate_distillation_inputs(student_logits, teacher_logits, labels)


def test_validate_distillation_inputs_batch_mismatch():
    """Test validation fails for batch size mismatch"""
    student_logits = torch.randn(4, 2)
    teacher_logits = torch.randn(4, 2)
    labels = torch.randint(0, 2, (5,))  # Wrong batch size
    
    with pytest.raises(ValueError, match="batch size"):
        validate_distillation_inputs(student_logits, teacher_logits, labels)


def test_validate_distillation_inputs_nan():
    """Test validation fails for NaN values"""
    student_logits = torch.tensor([[1.0, 2.0], [float('nan'), 3.0]])
    teacher_logits = torch.randn(2, 2)
    labels = torch.tensor([0, 1])
    
    with pytest.raises(ValueError, match="NaN or Inf"):
        validate_distillation_inputs(student_logits, teacher_logits, labels)


def test_distillation_loss_invalid_alpha():
    """Test distillation loss rejects invalid alpha"""
    student_logits = torch.randn(4, 2)
    teacher_logits = torch.randn(4, 2)
    labels = torch.randint(0, 2, (4,))
    
    with pytest.raises(ValueError, match="alpha"):
        distillation_loss(student_logits, teacher_logits, labels, alpha=1.5)
    
    with pytest.raises(ValueError, match="alpha"):
        distillation_loss(student_logits, teacher_logits, labels, alpha=-0.1)


def test_distillation_loss_invalid_temperature():
    """Test distillation loss rejects invalid temperature"""
    student_logits = torch.randn(4, 2)
    teacher_logits = torch.randn(4, 2)
    labels = torch.randint(0, 2, (4,))
    
    with pytest.raises(ValueError, match="temperature"):
        distillation_loss(student_logits, teacher_logits, labels, temperature=0.0)
    
    with pytest.raises(ValueError, match="temperature"):
        distillation_loss(student_logits, teacher_logits, labels, temperature=-1.0)


def test_distillation_loss_is_finite():
    """Test distillation loss components are finite"""
    for _ in range(10):  # Test multiple random cases
        student_logits = torch.randn(8, 2)
        teacher_logits = torch.randn(8, 2)
        labels = torch.randint(0, 2, (8,))
        
        total_loss, hard_loss, soft_loss = distillation_loss(
            student_logits, teacher_logits, labels
        )
        
        assert torch.isfinite(total_loss), "Total loss not finite"
        assert torch.isfinite(hard_loss), "Hard loss not finite"
        assert torch.isfinite(soft_loss), "Soft loss not finite"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
