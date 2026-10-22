# 🎯 How to Improve Model Accuracy

## 📊 Your Current Results

**Baseline Student:** 50.0% accuracy  
**Distilled Student:** 55.0% accuracy  

**Why so low?**
- ❌ Only 50 training samples (way too small!)
- ❌ Only 1 epoch (not enough training)
- ❌ Only 20 test samples (unreliable evaluation)

**This was just a smoke test to verify the code works!**

---

## 🚀 3 Ways to Improve Accuracy

### Option 1: Quick Improvement ⚡ (RECOMMENDED)

**Run this:**
```powershell
.\train_improved.ps1
```

**Or manually:**
```powershell
.\.venv\Scripts\python.exe src/train.py --train-size 800 --test-size 300 --epochs 3 --batch-size 8
```

**Expected Results:**
- ✅ Baseline: 75-80% accuracy
- ✅ Distilled: 77-85% accuracy
- ⏱️ Time: 10-15 minutes
- 💾 Good for project reports

---

### Option 2: Best Accuracy 🏆

**Run this:**
```powershell
.\train_best_accuracy.ps1
```

**Or manually:**
```powershell
.\.venv\Scripts\python.exe src/train.py --train-size 2000 --test-size 500 --epochs 5 --batch-size 16 --max-length 128
```

**Expected Results:**
- ✅ Baseline: 80-85% accuracy
- ✅ Distilled: 82-88% accuracy
- ⏱️ Time: 30-45 minutes
- 💾 Excellent for project reports

---

### Option 3: Advanced Optimization 🧪

**Run this:**
```powershell
.\train_advanced.ps1
```

**Or manually:**
```powershell
.\.venv\Scripts\python.exe src/train.py --train-size 1500 --test-size 400 --epochs 4 --batch-size 12 --max-length 128 --temperature 3.0 --alpha 0.4
```

**Expected Results:**
- ✅ Baseline: 78-84% accuracy
- ✅ Distilled: 82-90% accuracy
- ⏱️ Time: 25-40 minutes
- 💾 Shows hyperparameter optimization

**Why these settings work:**
- **Temperature 3.0**: Softer teacher distributions reveal more knowledge
- **Alpha 0.4**: More weight to teacher soft targets
- **Max length 128**: Captures more context from reviews
- **1500 samples**: Good balance of accuracy vs. time

---

## 📈 Understanding the Parameters

### Training Size
- **50 samples**: Only for testing (50-55% accuracy) ❌
- **200 samples**: Minimum viable (60-70% accuracy) ⚠️
- **800 samples**: Good for reports (75-85% accuracy) ✅
- **2000 samples**: Excellent results (80-88% accuracy) 🏆
- **5000+ samples**: Research quality (85-90% accuracy) 🎓

### Epochs
- **1 epoch**: Underfitting, poor accuracy ❌
- **3 epochs**: Good baseline ✅
- **5 epochs**: Better convergence 🏆
- **10+ epochs**: Diminishing returns, may overfit ⚠️

### Temperature (T)
- **1.0**: Hard targets (like baseline) ❌
- **2.0**: Standard soft targets (default) ✅
- **3.0-4.0**: Softer, reveals more knowledge 🏆
- **8.0+**: Too soft, adds noise ❌

### Alpha (α)
- **0.25**: Heavy teacher influence (good if teacher is accurate) 🎓
- **0.4**: Balanced with more teacher weight ✅
- **0.5**: Equal hard and soft (default) ✅
- **0.75**: More emphasis on correctness ✅
- **1.0**: No distillation (same as baseline) ❌

### Max Length
- **64**: Fast but may lose context ⚠️
- **96**: Good balance (default) ✅
- **128**: More context, better accuracy 🏆
- **256**: Expensive, diminishing returns ⚠️

---

## 🎯 My Recommendation

**Start here (10-15 minutes):**
```powershell
.\train_improved.ps1
```

**Then try this for best results (30-45 minutes):**
```powershell
.\train_best_accuracy.ps1
```

**Compare the results!** This shows the impact of:
- More training data
- More epochs
- Longer sequences

---

## 📊 Tracking Your Experiments

After each run, document in `report/experiment_log.md`:

```markdown
### Experiment #2: Improved Configuration

**Date:** [Today's date]
**Configuration:**
- Train size: 800
- Test size: 300
- Epochs: 3
- Batch size: 8

**Results:**
| Model | Accuracy | Macro-F1 |
|---|---|---|
| Baseline | XX.X% | XX.X% |
| Distilled | XX.X% | XX.X% |

**Observations:**
- Accuracy improved from 50% to XX%
- Distillation helped by X.X%
- [Your observations]
```

---

## 🔬 Advanced: Hyperparameter Tuning

Want to experiment? Try these:

**Vary Temperature:**
```powershell
# T=2.0 (default)
.\.venv\Scripts\python.exe src/train.py --train-size 800 --test-size 300 --epochs 3 --temperature 2.0

# T=4.0 (softer)
.\.venv\Scripts\python.exe src/train.py --train-size 800 --test-size 300 --epochs 3 --temperature 4.0
```

**Vary Alpha:**
```powershell
# α=0.25 (more teacher)
.\.venv\Scripts\python.exe src/train.py --train-size 800 --test-size 300 --epochs 3 --alpha 0.25

# α=0.75 (more hard labels)
.\.venv\Scripts\python.exe src/train.py --train-size 800 --test-size 300 --epochs 3 --alpha 0.75
```

---

## ⚡ Quick Commands

**Current accuracy:** Check `results/metrics.json`

**Run improved training:**
```powershell
.\train_improved.ps1
```

**View results after training:**
```powershell
Get-Content results/metrics.json | ConvertFrom-Json | ConvertTo-Json -Depth 10
```

**Refresh web app:**
Just reload the browser page!

---

## 🎓 For Your Report

After running improved training, update your report with:

1. **Comparison Table:**
   | Configuration | Baseline Acc | Distilled Acc | Improvement |
   |---|---|---|---|
   | Smoke test (50 samples) | 50.0% | 55.0% | +5.0% |
   | Improved (800 samples) | XX.X% | XX.X% | +X.X% |

2. **Discussion:**
   - "Initial smoke test showed 50-55% accuracy with only 50 training samples"
   - "After training on 800 samples for 3 epochs, accuracy improved to XX-XX%"
   - "Distillation provided X.X% improvement over baseline"
   - "This demonstrates that knowledge distillation is effective for this task"

3. **Learning Curve Analysis:**
   - Show how accuracy improved with more data
   - Discuss the impact of hyperparameters
   - Compare your results to literature (DistilBERT retained 97% of BERT performance)

---

## 🚀 Ready to Improve?

**Start with this command:**
```powershell
.\train_improved.ps1
```

Then check your new results in `results/metrics.json` and refresh the web app!

**You should see 75-85% accuracy!** 📈✨
