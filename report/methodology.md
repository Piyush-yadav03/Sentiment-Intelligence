# SentimentLab AI: Detailed Methodology Documentation

## Table of Contents
1. [Overview](#overview)
2. [Knowledge Distillation Framework](#knowledge-distillation-framework)
3. [Data Pipeline](#data-pipeline)
4. [Training Procedures](#training-procedures)
5. [Evaluation Protocol](#evaluation-protocol)
6. [Hyperparameter Selection](#hyperparameter-selection)
7. [Reproducibility Guidelines](#reproducibility-guidelines)

---

## Overview

This document provides detailed technical methodology for the SentimentLab AI project, complementing the main project report with implementation-level details.

**Research Paradigm:** Supervised learning with knowledge distillation  
**Task:** Binary sentiment classification (positive/negative)  
**Approach:** Transfer knowledge from large teacher to compact student  
**Evaluation:** Comparative study of baseline vs. distilled student

---

## Knowledge Distillation Framework

### Theoretical Foundation

Knowledge distillation transfers learned representations from a "teacher" model to a "student" model. Unlike traditional training that uses only hard labels (one-hot vectors), distillation leverages soft probability distributions that encode the teacher's understanding of class relationships.

### Mathematical Formulation

#### Hard-Label Loss (Baseline)

For baseline training, we use standard cross-entropy:

```
L_hard = -Σ_i y_i · log(p_i)
```

Where:
- `y_i` is the one-hot true label (1 for correct class, 0 otherwise)
- `p_i = softmax(z_i)` where `z_i` are student logits
- This loss only teaches which class is correct

#### Soft-Target Loss (Distillation Component)

For distillation, we add a KL divergence term:

```
L_soft = KL(P_teacher || P_student)
       = Σ_i P_teacher,i · log(P_teacher,i / P_student,i)
```

Where:
- `P_teacher,i = softmax(z_teacher,i / T)`
- `P_student,i = softmax(z_student,i / T)`
- `T` is temperature (T > 1 creates softer distributions)

**Why KL Divergence?**
- Measures how student distribution differs from teacher
- Minimizing KL makes student mimic teacher's confidence patterns
- Asymmetric: penalizes student errors more when teacher is confident

#### Combined Loss

```
L_total = α · L_hard + (1 - α) · T² · L_soft
```

**Components:**
- `α ∈ [0, 1]`: weight for hard-label loss
- `(1 - α)`: weight for soft-target loss
- `T²`: scaling factor to match gradient magnitudes

**Rationale for T² Scaling:**  
Without T² scaling, gradients from soft targets would be approximately T² times smaller than hard-label gradients, making the soft loss ineffective.

### Temperature Scaling

Temperature `T` controls distribution softness:

**T = 1 (No Scaling):**
```
softmax([2, 1, 0]) = [0.66, 0.24, 0.09]
```

**T = 2 (Softer):**
```
softmax([2/2, 1/2, 0/2]) = [0.49, 0.31, 0.19]
```

**T = 4 (Very Soft):**
```
softmax([2/4, 1/4, 0/4]) = [0.41, 0.34, 0.25]
```

Higher T reveals more information about class relationships that hard labels hide.

### Implementation Details

**PyTorch Implementation:**

```python
def distillation_loss(student_logits, teacher_logits, labels, T, alpha):
    # Hard loss
    L_hard = F.cross_entropy(student_logits, labels)
    
    # Soft loss
    p_teacher = F.softmax(teacher_logits / T, dim=-1)
    log_p_student = F.log_softmax(student_logits / T, dim=-1)
    L_soft = F.kl_div(log_p_student, p_teacher, reduction='batchmean')
    
    # Combined
    L_total = alpha * L_hard + (1 - alpha) * T**2 * L_soft
    return L_total, L_hard, L_soft
```

**Key Decisions:**
- Use `log_softmax` for numerical stability
- `reduction='batchmean'` divides by batch size
- Separate tracking of hard and soft losses for monitoring

---

## Data Pipeline

### Dataset: IMDb Movie Reviews

**Source:** `datasets.load_dataset("imdb")`  
**Statistics:**
- Full training: 25,000 reviews
- Full test: 25,000 reviews
- Class balance: 50% positive, 50% negative

### Balanced Subset Sampling

**Algorithm:**

```python
def load_imdb_balanced(split, size, seed):
    dataset = load_dataset("imdb", split=split)
    
    # Separate by class
    negative = [text for text, label in zip(dataset["text"], dataset["label"]) if label == 0]
    positive = [text for text, label in zip(dataset["text"], dataset["label"]) if label == 1]
    
    # Take half from each class
    half = size // 2
    negative_sample = negative[:half]
    positive_sample = positive[:half]
    
    # Combine and shuffle
    texts = negative_sample + positive_sample
    labels = [0] * len(negative_sample) + [1] * len(positive_sample)
    
    indices = list(range(len(labels)))
    random.Random(seed + size).shuffle(indices)
    
    return [texts[i] for i in indices], [labels[i] for i in indices]
```

**Design Choices:**
- Equal class representation prevents bias
- Fixed seed ensures reproducibility
- Seed modified by size prevents overlap if sampling multiple sizes

### Tokenization

**Tokenizer:** DistilBERT WordPiece tokenizer

**Process:**
1. Load pre-trained tokenizer: `AutoTokenizer.from_pretrained(teacher_model)`
2. Tokenize with settings:
   - `truncation=True`: Cut sequences longer than max_length
   - `padding="max_length"`: Pad shorter sequences
   - `max_length=96`: Fixed sequence length
   - `return_tensors="pt"`: Return PyTorch tensors

**Output:**
- `input_ids`: Token IDs [batch_size, max_length]
- `attention_mask`: 1 for real tokens, 0 for padding [batch_size, max_length]

**Example:**
```
Text: "This movie was great!"
Token IDs: [101, 2023, 3185, 2001, 2307, 999, 102, 0, 0, ...]
Attention:  [1,    1,    1,    1,    1,   1,   1,  0, 0, ...]
            [CLS] this movie was great !  [SEP] [PAD] [PAD] ...
```

### DataLoader Configuration

```python
train_loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,      # Random order each epoch
    num_workers=0,     # Single process (CPU-friendly)
    pin_memory=False   # No GPU pinning needed
)
```

---

## Training Procedures

### Baseline Student Training

**Objective:** Train student using only hard labels

**Algorithm:**
```
FOR each epoch:
    FOR each batch in train_loader:
        1. Forward pass: logits = student(input_ids, attention_mask)
        2. Compute loss: L = cross_entropy(logits, labels)
        3. Backward pass: L.backward()
        4. Clip gradients: clip_grad_norm_(parameters, max_norm=1.0)
        5. Update weights: optimizer.step()
        6. Zero gradients: optimizer.zero_grad()
```

**Optimizer:** AdamW
- Learning rate: 3e-4
- Weight decay: 0.01 (L2 regularization)
- Betas: (0.9, 0.999) (default)
- Epsilon: 1e-8 (default)

**Why AdamW?**
- Adaptive learning rates per parameter
- Decoupled weight decay (better than L2 penalty)
- Standard for Transformer training

### Distilled Student Training

**Objective:** Train student using hard labels + teacher soft targets

**Preparation:**
1. Generate teacher logits for all training examples:
   ```python
   teacher.eval()
   with torch.no_grad():
       for batch in train_eval_loader:
           logits = teacher(batch["input_ids"], batch["attention_mask"]).logits
           teacher_logits.append(logits)
   ```

2. Cache to disk: `torch.save(teacher_logits, "results/teacher_logits.pt")`

3. Create distilled dataset with teacher logits attached to each example

**Training Algorithm:**
```
FOR each epoch:
    FOR each batch in distilled_loader:
        1. Forward pass: student_logits = student(input_ids, attention_mask)
        2. Compute combined loss:
           L_hard = cross_entropy(student_logits, labels)
           L_soft = KL_div(student_logits/T, teacher_logits/T) * T²
           L_total = α * L_hard + (1-α) * L_soft
        3. Backward pass: L_total.backward()
        4. Clip gradients
        5. Update weights
        6. Log both losses
```

**Why Pre-compute Teacher Logits?**
- Teacher inference is expensive (large model)
- Logits don't change during student training
- Caching speeds up training significantly
- Enables exact reproduction

### Gradient Clipping

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**Purpose:**
- Prevents exploding gradients
- Stabilizes training on small datasets
- Standard practice for Transformers

### Training Monitoring

**Metrics Tracked Per Batch:**
- Total loss
- Hard-label loss (if distilling)
- Soft-target loss (if distilling)

**Metrics Tracked Per Epoch:**
- Average loss
- Training time
- Learning rate (if using scheduler)

**Early Stopping:**
Not implemented in current version—trains for fixed number of epochs.

---

## Evaluation Protocol

### Test Set Evaluation

**Procedure:**
1. Set model to evaluation mode: `model.eval()`
2. Disable gradient computation: `with torch.no_grad():`
3. For each batch:
   - Forward pass: `logits = model(input_ids, attention_mask)`
   - Get predictions: `preds = logits.argmax(dim=-1)`
   - Collect predictions and true labels
4. Compute metrics on all predictions

### Classification Metrics

**Accuracy:**
```
accuracy = (number of correct predictions) / (total predictions)
```

**Precision (per class):**
```
precision_class_i = TP_i / (TP_i + FP_i)
```

**Recall (per class):**
```
recall_class_i = TP_i / (TP_i + FN_i)
```

**F1 Score (per class):**
```
F1_i = 2 * (precision_i * recall_i) / (precision_i + recall_i)
```

**Macro-F1:**
```
macro_F1 = (F1_negative + F1_positive) / 2
```

**Why Macro-F1?**
- Treats both classes equally important
- Better than accuracy for imbalanced datasets
- Standard metric for classification evaluation

**Implementation:**
```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

accuracy = accuracy_score(true_labels, predictions)
precision = precision_score(true_labels, predictions, average='macro')
recall = recall_score(true_labels, predictions, average='macro')
macro_f1 = f1_score(true_labels, predictions, average='macro')
```

### Latency Measurement

**Objective:** Measure average inference time per example on CPU

**Protocol:**
1. **Warmup:** Run 2-3 batches to warm up CPU caches
2. **Measurement:** Time next 10 batches using `time.perf_counter()`
3. **Calculation:** `latency_per_sample = total_time / total_samples`
4. **Reporting:** Report in milliseconds (ms)

**Implementation:**
```python
import time

model.eval()
times = []
total_samples = 0

with torch.no_grad():
    for i, batch in enumerate(test_loader):
        if i < 2:  # Warmup
            model(batch["input_ids"], batch["attention_mask"])
            continue
        
        if i >= 12:  # Measured enough
            break
        
        start = time.perf_counter()
        model(batch["input_ids"], batch["attention_mask"])
        end = time.perf_counter()
        
        times.append(end - start)
        total_samples += len(batch["labels"])

latency_ms = (sum(times) / total_samples) * 1000
```

**Why This Protocol?**
- Warmup eliminates cache-cold effects
- Multiple batches smooth out variance
- `perf_counter()` is high-resolution timer
- CPU-only: no GPU synchronization needed

### Model Size Measurement

**Checkpoint Size:**
```python
checkpoint_mb = checkpoint_path.stat().st_size / (1024 ** 2)
```

**Parameter Count:**
```python
param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
```

---

## Hyperparameter Selection

### Rationale for Default Values

| Hyperparameter | Value | Rationale |
|---|---|---|
| Learning Rate | 3e-4 | Standard for small Transformers; not too aggressive |
| Weight Decay | 0.01 | Mild regularization; prevents overfitting |
| Batch Size | 8 | CPU memory constraint; provides stable gradients |
| Epochs | 3 | Sufficient for small dataset convergence |
| Max Length | 96 | Balances context vs. computation; covers most reviews |
| Temperature | 2.0 | Standard in literature; softens distributions well |
| Alpha | 0.5 | Equal weight to hard and soft losses; balanced approach |
| Gradient Clip | 1.0 | Prevents gradient explosions; standard for Transformers |

### Sensitivity Analysis (Recommended Future Work)

**Temperature Sweep:**
- Values: {1.0, 2.0, 4.0, 8.0}
- Expected: Higher T reveals more dark knowledge but may introduce noise

**Alpha Sweep:**
- Values: {0.25, 0.5, 0.75}
- Expected: Lower α emphasizes teacher mimicry; higher α emphasizes correctness

**Learning Rate Sweep:**
- Values: {1e-4, 3e-4, 1e-3}
- Expected: Lower LR more stable but slower; higher LR faster but may diverge

---

## Reproducibility Guidelines

### Random Seed Control

**Set All Seeds:**
```python
import random
import numpy as np
import torch

seed = 42

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
```

**Dataset Sampling:**
```python
# Use seed + size to avoid overlap between different size samples
random.Random(seed + train_size).shuffle(indices)
```

### Deterministic Operations

**PyTorch Settings:**
```python
torch.backends.cudnn.deterministic = True  # If using GPU
torch.backends.cudnn.benchmark = False     # If using GPU
```

**Note:** Some operations (especially on CPU) are inherently deterministic.

### Configuration Persistence

**Save Hyperparameters with Checkpoints:**
```python
torch.save({
    'state_dict': model.state_dict(),
    'config': {
        'vocab_size': len(tokenizer),
        'max_length': max_length,
        'seed': seed,
        'temperature': temperature,
        'alpha': alpha,
        # ... all other hyperparameters
    }
}, checkpoint_path)
```

**Save Experiment Metadata:**
```json
{
  "teacher_model": "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
  "dataset": "IMDb",
  "train_examples": 800,
  "test_examples": 300,
  "epochs": 3,
  "batch_size": 8,
  "learning_rate": 0.0003,
  "max_length": 96,
  "temperature": 2.0,
  "alpha": 0.5,
  "seed": 42,
  "device": "CPU"
}
```

### Version Control

**Pin Dependencies:**
```
torch==2.2.0
transformers==4.44.0
datasets==2.20.0
# ... etc
```

**Document Environment:**
- Python version
- Operating system
- CPU model
- RAM amount

### Reproducibility Checklist

✓ Fixed random seeds for Python, NumPy, PyTorch  
✓ Deterministic data splitting  
✓ Consistent tokenization  
✓ Saved configuration files  
✓ Pinned dependency versions  
✓ Documented hardware specs  
✓ Cached teacher logits (same across runs)  
✓ Identical training order  

---

## Conclusion

This methodology document provides complete technical details for reproducing the SentimentLab AI experiments. Key aspects:

1. **Knowledge Distillation:** Mathematically rigorous formulation with proper T² scaling
2. **Data Pipeline:** Balanced sampling, proper tokenization, reproducible splits
3. **Training:** Separate baseline and distilled procedures with identical hyperparameters
4. **Evaluation:** Comprehensive metrics covering quality and efficiency
5. **Reproducibility:** Extensive controls for exact replication

For practical implementation, refer to the source code in `src/` directory. For theoretical background, see the main project report. For experimental results, see `results/metrics.json` after training.
