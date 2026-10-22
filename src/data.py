"""Data loading and preprocessing utilities"""

import random
from typing import List, Tuple, Dict, Any
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import PreTrainedTokenizer


class SentimentDataset(Dataset):
    """Dataset for sentiment analysis"""
    
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 96,
    ):
        """
        Args:
            texts: List of text samples
            labels: List of labels (0 for negative, 1 for positive)
            tokenizer: Tokenizer to use
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Pre-tokenize all samples
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        )
    
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }


class DistilledSentimentDataset(Dataset):
    """Dataset with teacher logits for knowledge distillation"""
    
    def __init__(
        self,
        base_dataset: SentimentDataset,
        teacher_logits: torch.Tensor,
    ):
        """
        Args:
            base_dataset: Base sentiment dataset
            teacher_logits: Pre-computed teacher logits [N, num_classes]
        """
        self.base_dataset = base_dataset
        self.teacher_logits = teacher_logits
        
        if len(base_dataset) != len(teacher_logits):
            raise ValueError(
                f"Dataset size ({len(base_dataset)}) must match "
                f"teacher logits size ({len(teacher_logits)})"
            )
    
    def __len__(self) -> int:
        return len(self.base_dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.base_dataset[idx]
        item["teacher_logits"] = self.teacher_logits[idx]
        return item


def load_imdb_balanced(
    split: str,
    size: int,
    seed: int = 42
) -> Tuple[List[str], List[int]]:
    """
    Load balanced IMDb dataset subset.
    
    Args:
        split: 'train' or 'test'
        size: Total number of samples (will be split 50/50 between classes)
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (texts, labels)
    """
    if split not in ["train", "test"]:
        raise ValueError(f"Invalid split: {split}. Must be 'train' or 'test'")
    
    if size < 2:
        raise ValueError(f"Size must be at least 2, got {size}")
    
    # Load dataset
    dataset = load_dataset("imdb", split=split)
    
    # Get half from each class
    half = size // 2
    
    # Separate by class
    negative_samples = [
        text for text, label in zip(dataset["text"], dataset["label"])
        if label == 0
    ][:half]
    
    positive_samples = [
        text for text, label in zip(dataset["text"], dataset["label"])
        if label == 1
    ][:half]
    
    # Combine and shuffle
    texts = negative_samples + positive_samples
    labels = [0] * len(negative_samples) + [1] * len(positive_samples)
    
    # Shuffle with seed
    indices = list(range(len(labels)))
    random.Random(seed + size).shuffle(indices)
    
    texts = [texts[i] for i in indices]
    labels = [labels[i] for i in indices]
    
    return texts, labels


def prepare_datasets(
    tokenizer: PreTrainedTokenizer,
    train_size: int,
    test_size: int,
    max_length: int,
    seed: int = 42,
) -> Tuple[SentimentDataset, SentimentDataset]:
    """
    Prepare training and test datasets.
    
    Args:
        tokenizer: Tokenizer to use
        train_size: Number of training samples
        test_size: Number of test samples
        max_length: Maximum sequence length
        seed: Random seed
    
    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    print(f"Loading IMDb dataset (first run downloads dataset)...")
    
    # Load balanced subsets
    train_texts, train_labels = load_imdb_balanced("train", train_size, seed)
    test_texts, test_labels = load_imdb_balanced("test", test_size, seed)
    
    print(f"Train: {len(train_texts)} samples ({sum(train_labels)} positive)")
    print(f"Test: {len(test_texts)} samples ({sum(test_labels)} positive)")
    
    # Create datasets
    train_dataset = SentimentDataset(train_texts, train_labels, tokenizer, max_length)
    test_dataset = SentimentDataset(test_texts, test_labels, tokenizer, max_length)
    
    return train_dataset, test_dataset
