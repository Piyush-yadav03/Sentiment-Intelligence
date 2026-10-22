"""Model evaluation utilities"""

import time
import torch
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, float]:
    """
    Evaluate model on a dataset.
    
    Args:
        model: Model to evaluate
        dataloader: DataLoader for evaluation data
        device: Device to run evaluation on
    
    Returns:
        Dictionary of metrics
    """
    model.eval()
    model.to(device)
    
    all_predictions = []
    all_labels = []
    
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        # Forward pass
        logits = model(input_ids, attention_mask)
        predictions = logits.argmax(dim=-1)
        
        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average="macro", zero_division=0)
    recall = recall_score(all_labels, all_predictions, average="macro", zero_division=0)
    macro_f1 = f1_score(all_labels, all_predictions, average="macro", zero_division=0)
    
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "macro_f1": float(macro_f1),
    }


@torch.no_grad()
def measure_latency(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device = torch.device("cpu"),
    num_warmup: int = 2,
    num_measure: int = 10,
) -> Dict[str, float]:
    """
    Measure inference latency.
    
    Args:
        model: Model to measure
        dataloader: DataLoader for test data
        device: Device to run inference on
        num_warmup: Number of warmup batches
        num_measure: Number of batches to measure
    
    Returns:
        Dictionary with latency metrics
    """
    model.eval()
    model.to(device)
    
    times = []
    total_samples = 0
    
    for i, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        batch_size = input_ids.shape[0]
        
        # Warmup
        if i < num_warmup:
            model(input_ids, attention_mask)
            continue
        
        # Measure
        start_time = time.perf_counter()
        model(input_ids, attention_mask)
        end_time = time.perf_counter()
        
        times.append(end_time - start_time)
        total_samples += batch_size
        
        if len(times) >= num_measure:
            break
    
    if not times:
        return {"latency_ms_per_sample": 0.0, "throughput_samples_per_sec": 0.0}
    
    total_time = sum(times)
    latency_per_sample = total_time / max(total_samples, 1)
    throughput = total_samples / total_time if total_time > 0 else 0
    
    return {
        "latency_ms_per_sample": latency_per_sample * 1000,  # Convert to ms
        "throughput_samples_per_sec": throughput,
    }


def get_confusion_matrix(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device = torch.device("cpu"),
) -> np.ndarray:
    """Get confusion matrix for model predictions"""
    model.eval()
    model.to(device)
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]
            
            logits = model(input_ids, attention_mask)
            predictions = logits.argmax(dim=-1).cpu()
            
            all_predictions.extend(predictions.numpy())
            all_labels.extend(labels.numpy())
    
    return confusion_matrix(all_labels, all_predictions)


def get_classification_report(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device = torch.device("cpu"),
    class_names: List[str] = None,
) -> str:
    """Get detailed classification report"""
    model.eval()
    model.to(device)
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]
            
            logits = model(input_ids, attention_mask)
            predictions = logits.argmax(dim=-1).cpu()
            
            all_predictions.extend(predictions.numpy())
            all_labels.extend(labels.numpy())
    
    if class_names is None:
        class_names = ["Negative", "Positive"]
    
    return classification_report(
        all_labels,
        all_predictions,
        target_names=class_names,
        zero_division=0
    )
