# Training Scripts

Standalone scripts for training and evaluation.

## Quick Start

**Full training pipeline (recommended):**
```powershell
# Train both baseline and distilled models with evaluation
python src/train.py --train-size 800 --test-size 300 --epochs 3
```

**Individual training:**
```powershell
# Train baseline student only
python scripts/train_baseline.py --train-size 800 --test-size 300 --epochs 3

# Train distilled student only
python scripts/train_distilled.py --train-size 800 --test-size 300 --epochs 3

# Compare trained models
python scripts/run_evaluation.py
```

## Scripts

### train_baseline.py
Trains the baseline student model using only hard labels (ground truth).

**Arguments:**
- `--train-size`: Number of training samples (default: 800)
- `--test-size`: Number of test samples (default: 300)
- `--epochs`: Number of training epochs (default: 3)
- `--batch-size`: Batch size (default: 8)
- `--learning-rate`: Learning rate (default: 3e-4)
- `--max-length`: Maximum sequence length (default: 96)
- `--seed`: Random seed (default: 42)

**Output:**
- `results/student_baseline.pt`: Model checkpoint
- `results/tokenizer/`: Saved tokenizer
- Console: Training progress and evaluation metrics

### train_distilled.py
Trains the distilled student model using hard labels + teacher soft targets.

**Additional Arguments:**
- `--temperature`: Distillation temperature (default: 2.0)
- `--alpha`: Hard loss weight, 0-1 (default: 0.5)

**Output:**
- `results/student_distilled.pt`: Model checkpoint
- `results/teacher_logits.pt`: Cached teacher outputs
- `results/tokenizer/`: Saved tokenizer
- Console: Training progress with hard/soft loss breakdown

### run_evaluation.py
Loads trained models and displays comparison metrics.

**Requirements:**
- Both `student_baseline.pt` and `student_distilled.pt` must exist
- `metrics.json` should exist (created by src/train.py)

**Output:**
- Console: Model comparison summary
- References existing metrics.json

## Notes

- **Recommended workflow**: Use `src/train.py` for complete experiments
- These scripts are useful for:
  - Training models separately
  - Experimenting with one approach at a time
  - Understanding the training process step-by-step
- All scripts use CPU by default (modify for GPU if available)
- First run downloads IMDb dataset and teacher model
