# Experiment Log: SentimentLab AI

## Purpose

This document tracks all experimental runs, configurations, results, and observations. Maintain chronological records of training experiments to support reproducibility and analysis.

---

## Experiment Template

For each experiment, record:

```markdown
### Experiment #X: [Short Description]

**Date:** YYYY-MM-DD  
**Time:** HH:MM  
**Duration:** [Training time]

**Configuration:**
- Train size: 
- Test size:
- Epochs:
- Batch size:
- Learning rate:
- Max length:
- Temperature:
- Alpha:
- Seed:
- Command: [Exact command used]

**Hardware:**
- CPU:
- RAM:
- OS:

**Results:**
| Model | Accuracy | Macro-F1 | Latency (ms) | Size (MB) |
|---|---:|---:|---:|---:|
| Baseline | X.XX | X.XX | X.XX | X.XX |
| Distilled | X.XX | X.XX | X.XX | X.XX |

**Observations:**
- [Any interesting findings]
- [Convergence behavior]
- [Unexpected results]

**Issues:**
- [Any problems encountered]
- [Warnings or errors]

**Files Generated:**
- `results/student_baseline.pt`
- `results/student_distilled.pt`
- `results/metrics.json`
- `results/teacher_logits.pt`

**Next Steps:**
- [What to try next]
```

---

## Experiments

### Experiment #1: Smoke Test (Setup Validation)

**Date:** [Fill after running]  
**Time:** [Fill after running]  
**Duration:** [Fill after running]

**Configuration:**
- Train size: 200
- Test size: 100
- Epochs: 1
- Batch size: 4
- Learning rate: 3e-4
- Max length: 96
- Temperature: 2.0
- Alpha: 0.5
- Seed: 42
- Command: `python src/train.py --train-size 200 --test-size 100 --epochs 1 --batch-size 4 --max-length 96`

**Hardware:**
- CPU: [Fill your CPU]
- RAM: [Fill your RAM]
- OS: [Fill your OS]

**Results:**
[Fill from results/metrics.json after running]

**Observations:**
- Purpose: Verify setup and pipeline functionality
- Expected: Training completes without errors
- Expected: Models produce reasonable (not random) predictions
- Expected: Latency measurements complete successfully

**Issues:**
[Note any problems]

**Files Generated:**
- `results/student_baseline.pt`
- `results/student_distilled.pt`
- `results/metrics.json`
- `results/teacher_logits.pt`
- `results/tokenizer/`

**Next Steps:**
- If successful: proceed to larger experiment
- If failed: debug issues and retry

---

### Experiment #2: Standard Configuration

**Date:** [Fill after running]  
**Time:** [Fill after running]  
**Duration:** [Fill after running]

**Configuration:**
- Train size: 800
- Test size: 300
- Epochs: 3
- Batch size: 8
- Learning rate: 3e-4
- Max length: 96
- Temperature: 2.0
- Alpha: 0.5
- Seed: 42
- Command: `python src/train.py --train-size 800 --test-size 300 --epochs 3 --batch-size 8 --max-length 96`

**Hardware:**
- CPU: [Fill]
- RAM: [Fill]
- OS: [Fill]

**Results:**
[Fill from results/metrics.json]

**Observations:**
- Purpose: Main experiment for report results
- Compare baseline vs. distilled performance
- Measure accuracy-efficiency trade-offs

**Issues:**
[Note any problems]

**Analysis:**
- Did distillation improve accuracy?
- By how much?
- Was the improvement worth the additional complexity?
- How does student compare to teacher?

**Next Steps:**
- Use these results in project report
- Generate visualizations for presentation
- Consider hyperparameter tuning if time permits

---

### Experiment #3: Temperature Sweep (Optional)

**Purpose:** Investigate effect of temperature on distillation quality

**Planned Configurations:**

| Exp | T | Alpha | Train | Test | Epochs |
|---|---|---|---|---|---|
| 3a | 1.0 | 0.5 | 800 | 300 | 3 |
| 3b | 2.0 | 0.5 | 800 | 300 | 3 |
| 3c | 4.0 | 0.5 | 800 | 300 | 3 |
| 3d | 8.0 | 0.5 | 800 | 300 | 3 |

**Expected Outcome:**
- T=1.0: Approaches hard-label training
- T=2.0-4.0: Optimal range for knowledge transfer
- T=8.0: May become too soft, introduces noise

**Results:**
[Fill as experiments complete]

---

### Experiment #4: Alpha Sweep (Optional)

**Purpose:** Investigate balance between hard and soft losses

**Planned Configurations:**

| Exp | T | Alpha | Train | Test | Epochs |
|---|---|---|---|---|---|
| 4a | 2.0 | 0.25 | 800 | 300 | 3 |
| 4b | 2.0 | 0.50 | 800 | 300 | 3 |
| 4c | 2.0 | 0.75 | 800 | 300 | 3 |

**Expected Outcome:**
- Alpha=0.25: More teacher influence
- Alpha=0.50: Balanced
- Alpha=0.75: More emphasis on correctness

**Results:**
[Fill as experiments complete]

---

### Experiment #5: Architecture Variations (Optional)

**Purpose:** Test different student model sizes

**Planned Configurations:**

| Exp | Layers | d_model | Heads | Train | Test | Epochs |
|---|---|---|---|---|---|---|
| 5a | 1 | 128 | 4 | 800 | 300 | 3 |
| 5b | 2 | 128 | 4 | 800 | 300 | 3 |
| 5c | 3 | 128 | 4 | 800 | 300 | 3 |
| 5d | 2 | 64 | 2 | 800 | 300 | 3 |
| 5e | 2 | 256 | 8 | 800 | 300 | 3 |

**Expected Outcome:**
- Larger models: better accuracy, slower inference
- Smaller models: lower accuracy, faster inference
- Find optimal accuracy-efficiency point

**Results:**
[Fill as experiments complete]

---

## Observations and Insights

### General Findings

[After completing experiments, summarize key insights]

**Distillation Effectiveness:**
- [Does distillation help in your experiments?]
- [By how much?]
- [Under what conditions?]

**Hyperparameter Sensitivity:**
- [Which hyperparameters matter most?]
- [Are results sensitive to small changes?]

**Trade-offs:**
- [What are the accuracy-efficiency trade-offs?]
- [When would you recommend baseline vs. distilled?]

### Unexpected Results

[Document anything surprising]

### Failed Experiments

[Record failures for learning purposes]

| Exp | Issue | Cause | Solution |
|---|---|---|---|
| | | | |

### Best Configuration

**For Accuracy:**
- Configuration: [Fill]
- Result: [Fill]

**For Efficiency:**
- Configuration: [Fill]
- Result: [Fill]

**Balanced:**
- Configuration: [Fill]
- Result: [Fill]

---

## Comparison with Literature

[After completing experiments, compare with published results]

**DistilBERT (Sanh et al., 2019):**
- Retained 97% of BERT performance
- 40% fewer parameters
- 60% faster

**Our Results:**
- Retained [X]% of teacher performance
- [Y]% fewer parameters
- [Z]x faster

**Differences:**
- We distill on task-specific data (IMDb)
- They distill during pre-training (general text)
- We use smaller student architecture
- Different evaluation metrics and dataset

---

## Lessons Learned

### Technical Lessons

1. [What worked well?]
2. [What didn't work?]
3. [What would you do differently?]

### Research Lessons

1. [What did you learn about knowledge distillation?]
2. [What did you learn about sentiment analysis?]
3. [What are the limitations of this approach?]

### Software Engineering Lessons

1. [What tools were most useful?]
2. [What challenges did you face?]
3. [How would you improve the implementation?]

---

## Future Experiments

### Immediate Extensions

1. **Domain-Matched Teacher:**
   - Fine-tune teacher on IMDb training set
   - Eliminate SST-2/IMDb domain shift
   - Expected improvement: [Estimate]

2. **Multiple Seeds:**
   - Run experiments with seeds: {42, 123, 456, 789, 2024}
   - Calculate mean and standard deviation
   - Report confidence intervals

3. **Larger Dataset:**
   - Use full IMDb training set (25K samples)
   - Train for more epochs (5-10)
   - Expected improvement: [Estimate]

### Advanced Extensions

1. **Feature-Based Distillation:**
   - Transfer intermediate layer representations
   - Not just final logits
   - May improve student quality

2. **Online Distillation:**
   - Train teacher and student jointly
   - Teacher updates during student training
   - May find better teacher for this task

3. **Multi-Task Distillation:**
   - Distill from multiple teachers
   - Different sentiment datasets
   - Improve generalization

4. **Compression Comparison:**
   - Compare distillation vs. quantization
   - Compare distillation vs. pruning
   - Hybrid approaches

---

## Conclusion

This experiment log tracks the complete experimental journey for SentimentLab AI. Key takeaways:

1. **Reproducibility:** All experiments documented with exact configurations
2. **Analysis:** Systematic investigation of hyperparameters and trade-offs
3. **Learning:** Both successes and failures recorded for future reference
4. **Extension:** Clear path forward for future work

Remember: Research is iterative. Not all experiments will succeed, but each one teaches something valuable. Document everything, even (especially!) the failures.

---

**Last Updated:** [Fill date]  
**Total Experiments Completed:** [Fill number]  
**Time Invested:** [Fill hours]  
**Best Model:** [Fill path and metrics]
