"""Knowledge distillation loss functions"""

import torch
import torch.nn.functional as F


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 2.0,
    alpha: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute knowledge distillation loss.
    
    Loss = alpha * L_hard + (1 - alpha) * L_soft
    
    where:
    - L_hard is cross-entropy between student logits and true labels
    - L_soft is KL divergence between temperature-scaled teacher and student distributions
    
    Args:
        student_logits: Student model logits [batch_size, num_classes]
        teacher_logits: Teacher model logits [batch_size, num_classes]
        labels: True class labels [batch_size]
        temperature: Temperature for softening distributions (T > 1)
        alpha: Weight for hard-label loss (0 <= alpha <= 1)
    
    Returns:
        Tuple of (total_loss, hard_loss, soft_loss)
    """
    # Validate inputs
    if not 0 <= alpha <= 1:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    
    # Hard-label loss (standard cross-entropy)
    hard_loss = F.cross_entropy(student_logits, labels)
    
    # Soft-target loss (KL divergence with temperature scaling)
    # Note: We use log_softmax for student and softmax for teacher
    # KL(P||Q) where P=teacher, Q=student
    soft_loss = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=-1),
        F.softmax(teacher_logits / temperature, dim=-1),
        reduction="batchmean"
    ) * (temperature ** 2)  # Scale by T^2 as per Hinton et al.
    
    # Combined loss
    total_loss = alpha * hard_loss + (1 - alpha) * soft_loss
    
    return total_loss, hard_loss, soft_loss


def baseline_loss(
    student_logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """
    Compute baseline loss (standard cross-entropy).
    
    Args:
        student_logits: Student model logits [batch_size, num_classes]
        labels: True class labels [batch_size]
    
    Returns:
        Cross-entropy loss
    """
    return F.cross_entropy(student_logits, labels)


def validate_distillation_inputs(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
):
    """
    Validate inputs for distillation loss.
    
    Raises:
        ValueError if inputs are invalid
    """
    # Check shapes match
    if student_logits.shape != teacher_logits.shape:
        raise ValueError(
            f"Student logits shape {student_logits.shape} must match "
            f"teacher logits shape {teacher_logits.shape}"
        )
    
    # Check batch sizes match
    if student_logits.shape[0] != labels.shape[0]:
        raise ValueError(
            f"Logits batch size {student_logits.shape[0]} must match "
            f"labels batch size {labels.shape[0]}"
        )
    
    # Check for NaN or Inf
    if torch.isnan(student_logits).any() or torch.isinf(student_logits).any():
        raise ValueError("Student logits contain NaN or Inf")
    
    if torch.isnan(teacher_logits).any() or torch.isinf(teacher_logits).any():
        raise ValueError("Teacher logits contain NaN or Inf")
