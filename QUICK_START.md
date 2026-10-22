# 🚀 Quick Start Guide for SentimentLab AI

## ✅ Setup Complete!

Your environment is ready! All dependencies are installed and tests are passing.

---

## 🎯 What to Do Next (Step by Step)

### Step 1: Run a Quick Smoke Test (2-5 minutes)

Open PowerShell in this folder and run:

```powershell
.\.venv\Scripts\python.exe src/train.py --train-size 50 --test-size 20 --epochs 1 --batch-size 4 --max-length 64
```

**What this does:**
- Downloads IMDb dataset (first run only, ~80MB)
- Downloads teacher model (first run only, ~260MB)
- Trains baseline student
- Trains distilled student
- Creates `results/metrics.json` with your results

**Expected time:** 2-5 minutes on a modern laptop

---

### Step 2: Launch the Web Application

After training completes:

```powershell
.\.venv\Scripts\streamlit.exe run app.py
```

Your browser will open to `http://localhost:8501`

**Explore all 6 pages:**
1. 📊 Overview - Project introduction
2. 🎯 Live Analysis - Try predicting sentiment!
3. 📈 Model Comparison - See your results visualized
4. 📁 Batch Analysis - Upload CSV files
5. 🔬 Research Lab - Understand the methodology
6. ℹ️ About - Full documentation

---

### Step 3: Run Full Experiment (15-30 minutes)

For your project report, run a more complete experiment:

```powershell
.\.venv\Scripts\python.exe src/train.py --train-size 800 --test-size 300 --epochs 3 --batch-size 8 --max-length 96
```

This gives you better results to report in your project.

---

## 📝 Your Action Checklist

- [x] ✅ Virtual environment created
- [x] ✅ Dependencies installed  
- [x] ✅ All 47 tests passing
- [ ] ⏳ Run smoke test (50 samples)
- [ ] ⏳ Explore web application
- [ ] ⏳ Run full experiment (800 samples)
- [ ] ⏳ Take screenshots of web app
- [ ] ⏳ Copy results to report template
- [ ] ⏳ Fill your personal information in report

---

## 📊 Understanding Your Results

After training, check `results/metrics.json`:

```json
{
  "baseline_student": {
    "accuracy": 0.XX,
    "macro_f1": 0.XX,
    "latency_ms_per_sample": X.XX,
    "checkpoint_mb": X.XX
  },
  "distilled_student": {
    "accuracy": 0.XX,
    "macro_f1": 0.XX,
    "latency_ms_per_sample": X.XX,
    "checkpoint_mb": X.XX
  }
}
```

**Key Questions to Answer:**
1. Did distillation improve accuracy? By how much?
2. Is the student faster than the teacher?
3. How much smaller is the student?
4. What's the trade-off between accuracy and efficiency?

---

## 🎓 For Your Project Report

### What to Document

1. **Your Hardware Specs:**
   - CPU: [Your processor model]
   - RAM: [Your RAM amount]
   - OS: Windows 11

2. **Your Results:** (from `results/metrics.json`)
   - Baseline accuracy: ___%
   - Distilled accuracy: ___%
   - Speed improvement: ___x
   - Size reduction: ___%

3. **Your Observations:**
   - Did distillation help in your case?
   - What were the trade-offs?
   - Any unexpected findings?

### Report Location

Fill in the template: `report/report_template.md`

---

## 💡 Helpful Commands

### Run Tests Anytime:
```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -v
```

### Check What's Installed:
```powershell
.\.venv\Scripts\python.exe -m pip list
```

### Train with Different Settings:
```powershell
# Smaller dataset
.\.venv\Scripts\python.exe src/train.py --train-size 200 --test-size 100 --epochs 2

# Different temperature
.\.venv\Scripts\python.exe src/train.py --train-size 800 --test-size 300 --temperature 4.0

# Different alpha
.\.venv\Scripts\python.exe src/train.py --train-size 800 --test-size 300 --alpha 0.25
```

---

## 🆘 If Something Goes Wrong

### "Out of Memory"
```powershell
# Use smaller batch size
.\.venv\Scripts\python.exe src/train.py --train-size 200 --test-size 100 --batch-size 2
```

### "Training Too Slow"
- This is normal on CPU!
- Smoke test (50 samples): 2-5 minutes
- Standard (800 samples): 15-30 minutes
- Start small and work your way up

### "Module Not Found"
```powershell
# Reinstall dependencies
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

---

## 🎉 You're Ready!

Everything is set up and working. Your next command should be:

```powershell
.\.venv\Scripts\python.exe src/train.py --train-size 50 --test-size 20 --epochs 1 --batch-size 4
```

Then explore the web app:

```powershell
.\.venv\Scripts\streamlit.exe run app.py
```

**Good luck with your project!** 🚀

---

**Questions?** Check the main README.md or the troubleshooting section in `pages/6_About_Project.py`
