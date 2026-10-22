"""Train baseline student model (hard labels only)"""

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
from data import prepare_datasets
from model import TinyTransformerStudent
from distillation import baseline_loss
from evaluate import evaluate_model, measure_latency
from metrics import MetricsTracker
from utils import seed_everything, count_parameters, get_checkpoint_size_mb


def train_baseline(
    model,
    train_loader,
    epochs,
    learning_rate,
    weight_decay,
    gradient_clip,
    device,
):
    """Train student with hard labels only"""
    model.to(device)
    model.train()
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    history = []
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch in progress:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # Forward pass
            logits = model(input_ids, attention_mask)
            loss = baseline_loss(logits, labels)
            
            # Backward pass
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            
            optimizer.step()
            
            # Track loss
            epoch_loss += loss.item()
            num_batches += 1
            
            progress.set_postfix({"loss": loss.item()})
        
        avg_loss = epoch_loss / num_batches
        history.append({"epoch": epoch + 1, "loss": avg_loss})
        
        print(f"Epoch {epoch+1}/{epochs} - Average Loss: {avg_loss:.4f}")
    
    return model, history


def main():
    parser = argparse.ArgumentParser(description="Train baseline student model")
    parser.add_argument("--train-size", type=int, default=DEFAULT_CONFIG["train_size"])
    parser.add_argument("--test-size", type=int, default=DEFAULT_CONFIG["test_size"])
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG["epochs"])
    parser.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_CONFIG["learning_rate"])
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_CONFIG["weight_decay"])
    parser.add_argument("--max-length", type=int, default=DEFAULT_CONFIG["max_length"])
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG["seed"])
    parser.add_argument("--gradient-clip", type=float, default=DEFAULT_CONFIG["gradient_clip"])
    
    args = parser.parse_args()
    
    # Validate
    if args.train_size < 2 or args.test_size < 2:
        raise ValueError("Dataset sizes must be >= 2")
    if args.epochs < 1:
        raise ValueError("Epochs must be >= 1")
    
    # Setup
    seed_everything(args.seed)
    device = torch.device("cpu")
    RESULTS.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("BASELINE STUDENT TRAINING")
    print("=" * 60)
    print(f"Training size: {args.train_size}")
    print(f"Test size: {args.test_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
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
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )
    
    # Create model
    print("\nCreating baseline student model...")
    model = TinyTransformerStudent(
        vocab_size=len(tokenizer),
        pad_id=tokenizer.pad_token_id or 0,
        max_length=args.max_length
    )
    
    param_count = count_parameters(model)
    print(f"Model parameters: {param_count:,}")
    
    # Train
    print("\nTraining baseline student...")
    start_time = time.time()
    
    model, history = train_baseline(
        model,
        train_loader,
        args.epochs,
        args.learning_rate,
        args.weight_decay,
        args.gradient_clip,
        device
    )
    
    train_time = time.time() - start_time
    print(f"\nTraining completed in {train_time:.1f}s")
    
    # Save model
    checkpoint_path = RESULTS / "student_baseline.pt"
    torch.save({"state_dict": model.cpu().state_dict()}, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")
    
    # Evaluate
    print("\nEvaluating baseline student...")
    model.to(device)
    
    eval_metrics = evaluate_model(model, test_loader, device)
    latency_metrics = measure_latency(model, test_loader, device)
    
    eval_metrics.update(latency_metrics)
    eval_metrics["checkpoint_mb"] = get_checkpoint_size_mb(checkpoint_path)
    eval_metrics["parameter_count"] = param_count
    eval_metrics["training_time_seconds"] = train_time
    
    # Display results
    print("\n" + "=" * 60)
    print("BASELINE STUDENT RESULTS")
    print("=" * 60)
    print(f"Accuracy: {eval_metrics['accuracy']*100:.2f}%")
    print(f"Precision: {eval_metrics['precision']*100:.2f}%")
    print(f"Recall: {eval_metrics['recall']*100:.2f}%")
    print(f"Macro-F1: {eval_metrics['macro_f1']*100:.2f}%")
    print(f"Latency: {eval_metrics['latency_ms_per_sample']:.2f} ms/sample")
    print(f"Checkpoint size: {eval_metrics['checkpoint_mb']:.2f} MB")
    print(f"Parameters: {param_count:,}")
    print("=" * 60)
    
    print("\n✅ Baseline training complete!")
    print("Next: Run train_distilled.py or use src/train.py for both models")


if __name__ == "__main__":
    main()
