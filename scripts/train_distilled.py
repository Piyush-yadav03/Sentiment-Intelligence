"""Train distilled student model (hard labels + teacher soft targets)"""

import sys
import argparse
import time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import TEACHER_MODEL, RESULTS, DEFAULT_CONFIG
from data import prepare_datasets, DistilledSentimentDataset
from model import TinyTransformerStudent
from distillation import distillation_loss
from evaluate import evaluate_model, measure_latency
from metrics import MetricsTracker
from utils import seed_everything, count_parameters, get_checkpoint_size_mb


@torch.no_grad()
def generate_teacher_logits(teacher_model, data_loader, device):
    """Generate teacher logits for all training examples"""
    teacher_model.to(device)
    teacher_model.eval()
    
    all_logits = []
    
    for batch in tqdm(data_loader, desc="Generating teacher logits"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        
        outputs = teacher_model(input_ids=input_ids, attention_mask=attention_mask)
        all_logits.append(outputs.logits.cpu())
    
    return torch.cat(all_logits, dim=0)


def train_distilled(
    model,
    train_loader,
    epochs,
    learning_rate,
    weight_decay,
    gradient_clip,
    temperature,
    alpha,
    device,
):
    """Train student with distillation loss"""
    model.to(device)
    model.train()
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    history = []
    
    for epoch in range(epochs):
        epoch_total_loss = 0.0
        epoch_hard_loss = 0.0
        epoch_soft_loss = 0.0
        num_batches = 0
        
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch in progress:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            teacher_logits = batch["teacher_logits"].to(device)
            
            # Forward pass
            student_logits = model(input_ids, attention_mask)
            
            # Distillation loss
            total_loss, hard_loss, soft_loss = distillation_loss(
                student_logits,
                teacher_logits,
                labels,
                temperature=temperature,
                alpha=alpha
            )
            
            # Backward pass
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            
            optimizer.step()
            
            # Track losses
            epoch_total_loss += total_loss.item()
            epoch_hard_loss += hard_loss.item()
            epoch_soft_loss += soft_loss.item()
            num_batches += 1
            
            progress.set_postfix({
                "total": total_loss.item(),
                "hard": hard_loss.item(),
                "soft": soft_loss.item()
            })
        
        avg_total = epoch_total_loss / num_batches
        avg_hard = epoch_hard_loss / num_batches
        avg_soft = epoch_soft_loss / num_batches
        
        history.append({
            "epoch": epoch + 1,
            "total_loss": avg_total,
            "hard_loss": avg_hard,
            "soft_loss": avg_soft
        })
        
        print(f"Epoch {epoch+1}/{epochs} - Total: {avg_total:.4f}, "
              f"Hard: {avg_hard:.4f}, Soft: {avg_soft:.4f}")
    
    return model, history


def main():
    parser = argparse.ArgumentParser(description="Train distilled student model")
    parser.add_argument("--train-size", type=int, default=DEFAULT_CONFIG["train_size"])
    parser.add_argument("--test-size", type=int, default=DEFAULT_CONFIG["test_size"])
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG["epochs"])
    parser.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_CONFIG["learning_rate"])
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_CONFIG["weight_decay"])
    parser.add_argument("--max-length", type=int, default=DEFAULT_CONFIG["max_length"])
    parser.add_argument("--temperature", type=float, default=DEFAULT_CONFIG["temperature"])
    parser.add_argument("--alpha", type=float, default=DEFAULT_CONFIG["alpha"])
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG["seed"])
    parser.add_argument("--gradient-clip", type=float, default=DEFAULT_CONFIG["gradient_clip"])
    
    args = parser.parse_args()
    
    # Validate
    if args.train_size < 2 or args.test_size < 2:
        raise ValueError("Dataset sizes must be >= 2")
    if args.epochs < 1:
        raise ValueError("Epochs must be >= 1")
    if not 0 <= args.alpha <= 1:
        raise ValueError("Alpha must be between 0 and 1")
    if args.temperature <= 0:
        raise ValueError("Temperature must be positive")
    
    # Setup
    seed_everything(args.seed)
    device = torch.device("cpu")
    RESULTS.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("DISTILLED STUDENT TRAINING")
    print("=" * 60)
    print(f"Training size: {args.train_size}")
    print(f"Test size: {args.test_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Temperature: {args.temperature}")
    print(f"Alpha (hard loss weight): {args.alpha}")
    print(f"Device: {device}")
    print("=" * 60)
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL)
    tokenizer.save_pretrained(RESULTS / "tokenizer")
    
    # Prepare datasets
    print("\nPreparing datasets...")
    train_dataset, test_dataset = prepare_datasets(
        tokenizer,
        args.train_size,
        args.test_size,
        args.max_length,
        args.seed
    )
    
    # Load teacher model
    print("\nLoading teacher model...")
    teacher_model = AutoModelForSequenceClassification.from_pretrained(TEACHER_MODEL)
    
    # Generate teacher logits
    print("\nGenerating teacher logits for training data...")
    train_eval_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    
    start_time = time.time()
    teacher_logits = generate_teacher_logits(teacher_model, train_eval_loader, device)
    teacher_time = time.time() - start_time
    
    print(f"Teacher logits generated in {teacher_time:.1f}s")
    torch.save(teacher_logits, RESULTS / "teacher_logits.pt")
    
    # Create distilled dataset
    distilled_dataset = DistilledSentimentDataset(train_dataset, teacher_logits)
    
    distilled_loader = DataLoader(
        distilled_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )
    
    # Create model
    print("\nCreating distilled student model...")
    model = TinyTransformerStudent(
        vocab_size=len(tokenizer),
        pad_id=tokenizer.pad_token_id or 0,
        max_length=args.max_length
    )
    
    param_count = count_parameters(model)
    print(f"Model parameters: {param_count:,}")
    
    # Train
    print("\nTraining distilled student...")
    start_time = time.time()
    
    model, history = train_distilled(
        model,
        distilled_loader,
        args.epochs,
        args.learning_rate,
        args.weight_decay,
        args.gradient_clip,
        args.temperature,
        args.alpha,
        device
    )
    
    train_time = time.time() - start_time
    print(f"\nTraining completed in {train_time:.1f}s")
    
    # Save model
    checkpoint_path = RESULTS / "student_distilled.pt"
    torch.save({"state_dict": model.cpu().state_dict()}, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")
    
    # Evaluate
    print("\nEvaluating distilled student...")
    model.to(device)
    
    eval_metrics = evaluate_model(model, test_loader, device)
    latency_metrics = measure_latency(model, test_loader, device)
    
    eval_metrics.update(latency_metrics)
    eval_metrics["checkpoint_mb"] = get_checkpoint_size_mb(checkpoint_path)
    eval_metrics["parameter_count"] = param_count
    eval_metrics["training_time_seconds"] = train_time
    
    # Display results
    print("\n" + "=" * 60)
    print("DISTILLED STUDENT RESULTS")
    print("=" * 60)
    print(f"Accuracy: {eval_metrics['accuracy']*100:.2f}%")
    print(f"Precision: {eval_metrics['precision']*100:.2f}%")
    print(f"Recall: {eval_metrics['recall']*100:.2f}%")
    print(f"Macro-F1: {eval_metrics['macro_f1']*100:.2f}%")
    print(f"Latency: {eval_metrics['latency_ms_per_sample']:.2f} ms/sample")
    print(f"Checkpoint size: {eval_metrics['checkpoint_mb']:.2f} MB")
    print(f"Parameters: {param_count:,}")
    print("=" * 60)
    
    print("\n✅ Distilled training complete!")
    print("Use run_evaluation.py to compare baseline and distilled models")


if __name__ == "__main__":
    main()
