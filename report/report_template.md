# Knowledge Distillation from Large-Scale to Small Language Models for Text Classification: A Sentiment Analysis Application

**Project Name:** SentimentLab AI  
**Project Type:** B.Tech CSE Major Project  
**Academic Year:** [Fill Your Year]  
**Student Name:** [Fill Your Name]  
**Roll Number:** [Fill Your Roll Number]  
**Institution:** [Fill Your Institution]  
**Guide:** [Fill Guide Name]  
**Date:** [Fill Date]

---

## Declaration

I hereby declare that the project work entitled **"Knowledge Distillation from Large-Scale to Small Language Models for Text Classification: A Sentiment Analysis Application"** submitted for the B.Tech degree in Computer Science and Engineering is a record of original work done by me under the guidance of [Guide Name], and this work has not been submitted elsewhere for any other degree or diploma.

**Signature:** _______________  
**Date:** _______________

---

## Certificate

This is to certify that the project work entitled **"Knowledge Distillation from Large-Scale to Small Language Models for Text Classification: A Sentiment Analysis Application"** submitted by [Student Name], Roll No. [Roll Number], in partial fulfillment of the requirements for the award of B.Tech degree in Computer Science and Engineering is a bonafide record of work carried out under my supervision and guidance.

**Guide Name:** [Fill Guide Name]  
**Designation:** [Fill Designation]  
**Signature:** _______________  
**Date:** _______________

---

## Acknowledgments

I would like to express my sincere gratitude to [Guide Name] for their invaluable guidance, constant encouragement, and support throughout this project. I am thankful to the Department of Computer Science and Engineering for providing the necessary facilities and resources.

I would also like to acknowledge the open-source community, particularly Hugging Face, PyTorch, and Streamlit developers, whose tools made this research possible.

Finally, I thank my family and friends for their continuous support and encouragement.

---

## Abstract

**Context:** Sentiment analysis is a fundamental natural language processing task with widespread applications in business analytics, social media monitoring, and customer feedback analysis. While large pre-trained language models achieve state-of-the-art performance, their computational demands limit deployment on resource-constrained devices.

**Problem:** Can knowledge distillation transfer sentiment classification capabilities from a large teacher model to a compact student model while maintaining acceptable accuracy and significantly reducing inference latency and storage requirements?

**Method:** This project implements and evaluates knowledge distillation for binary sentiment analysis. A pre-trained DistilBERT model serves as the teacher, generating soft probability distributions over sentiment classes. A compact two-layer Transformer student model learns from both ground-truth labels (hard targets) and teacher probabilities (soft targets). The distillation loss combines cross-entropy with KL divergence, weighted by hyperparameter α and scaled by temperature T.

**Dataset:** Balanced subsets of the IMDb movie review dataset (positive/negative labels).

**Results:** [Fill from results/metrics.json after training]
- Baseline student: X% accuracy, Y ms/sample latency, Z MB size
- Distilled student: X% accuracy, Y ms/sample latency, Z MB size
- Trade-off: [Describe accuracy vs. efficiency trade-off]

**Contribution:** A reproducible end-to-end system demonstrating knowledge distillation for sentiment analysis, including training pipeline, evaluation framework, and interactive web application.

**Keywords:** Knowledge Distillation, Sentiment Analysis, Model Compression, Transfer Learning, Natural Language Processing, Transformers

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Literature Review](#2-literature-review)
3. [Problem Statement](#3-problem-statement)
4. [Objectives](#4-objectives)
5. [Background and Theory](#5-background-and-theory)
6. [Proposed System](#6-proposed-system)
7. [System Architecture](#7-system-architecture)
8. [Dataset Description](#8-dataset-description)
9. [Methodology](#9-methodology)
10. [Implementation Details](#10-implementation-details)
11. [Experimental Setup](#11-experimental-setup)
12. [Results and Analysis](#12-results-and-analysis)
13. [Discussion](#13-discussion)
14. [Applications](#14-applications)
15. [Limitations](#15-limitations)
16. [Future Work](#16-future-work)
17. [Conclusion](#17-conclusion)
18. [References](#18-references)
19. [Appendix](#19-appendix)

---

## 1. Introduction

### 1.1 Motivation

Sentiment analysis—the computational task of identifying and categorizing opinions expressed in text—has become essential in the digital age. Organizations analyze customer reviews, social media posts, and feedback to understand public opinion, improve products, and inform business decisions.

Recent advances in natural language processing, particularly Transformer-based models like BERT, GPT, and their variants, have achieved remarkable accuracy on sentiment classification tasks. However, these models typically contain millions or billions of parameters, requiring substantial memory and computational resources. This limits their deployment in real-world scenarios such as:

- **Mobile and edge devices** with constrained memory and processing power
- **Real-time applications** requiring low-latency predictions
- **Cost-sensitive environments** where inference costs matter
- **Privacy-critical applications** requiring on-device processing

### 1.2 Knowledge Distillation as a Solution

Knowledge distillation (Hinton et al., 2015) offers a principled approach to model compression. Instead of training a small model from scratch using only hard labels (0 or 1 for classification), distillation trains the student to mimic a larger teacher's behavior by learning from soft probability distributions. These "soft targets" contain richer information about class similarities and decision boundaries that the teacher has learned.

### 1.3 Project Overview

This project investigates whether knowledge distillation can produce a compact sentiment classifier that:
1. Maintains classification quality close to the teacher
2. Significantly reduces model size and inference latency
3. Demonstrates practical utility through an interactive web application

We compare three approaches:
1. **Teacher model:** Pre-trained DistilBERT fine-tuned on SST-2
2. **Baseline student:** Compact Transformer trained with hard labels only
3. **Distilled student:** Same architecture trained with hard labels + teacher soft targets

### 1.4 Significance

This work contributes:
- A reproducible implementation of knowledge distillation for sentiment analysis
- Empirical evaluation of accuracy-efficiency trade-offs on CPU hardware
- An open-source web application for interactive sentiment analysis
- Documentation suitable for educational and research purposes

---

## 2. Literature Review

### 2.1 Sentiment Analysis

**Early Approaches:**  
Traditional sentiment analysis relied on lexicon-based methods and machine learning classifiers like Naive Bayes, SVM, and logistic regression with hand-crafted features (bag-of-words, TF-IDF, n-grams).

**Deep Learning Era:**  
Recurrent neural networks (RNNs), LSTMs, and CNNs improved performance by learning feature representations automatically. However, they struggled with long-range dependencies and required large annotated datasets.

**Transformer Revolution:**  
Vaswani et al. (2017) introduced the Transformer architecture with self-attention mechanisms. BERT (Devlin et al., 2019) demonstrated that pre-training on large text corpora followed by task-specific fine-tuning achieves state-of-the-art results across NLP tasks.

### 2.2 Model Compression Techniques

**Pruning:**  
Removing unnecessary weights or neurons (Han et al., 2015). Can reduce size but may require retraining.

**Quantization:**  
Reducing numerical precision (e.g., from 32-bit to 8-bit). Effective for deployment but may degrade accuracy.

**Low-Rank Factorization:**  
Decomposing weight matrices into smaller matrices (Jaderberg et al., 2014).

**Knowledge Distillation:**  
Training a smaller student to mimic a larger teacher (Hinton et al., 2015). Preserves learned representations better than training from scratch.

### 2.3 Knowledge Distillation

**Foundational Work:**  
Hinton, Vinyals, and Dean (2015) introduced temperature-scaled softmax and combined hard/soft losses. Demonstrated that "dark knowledge" in soft targets helps student learning.

**Theoretical Understanding:**  
Ba and Caruana (2014) showed shallow networks can match deep network performance with proper training. Furlanello et al. (2018) explored self-distillation where teacher and student have the same architecture.

**BERT Distillation:**  
Sanh et al. (2019) created DistilBERT by distilling BERT during pre-training, achieving 97% of BERT's performance with 40% fewer parameters. Tang et al. (2019) proposed distilling task-specific BERT models.

**Recent Advances:**  
Jiao et al. (2020) introduced TinyBERT with layer-wise distillation. Sun et al. (2020) explored patient knowledge distillation. Turc et al. (2019) studied pre-trained distillation for BERT compression.

### 2.4 Research Gap

Most knowledge distillation work focuses on large-scale pre-training or assumes GPU availability. This project addresses:
- CPU-only training and inference
- Distillation at the task-specific level (not pre-training)
- Practical deployment with an interactive web interface
- Educational documentation for reproducibility

---

## 3. Problem Statement

**Research Question:**  
Can knowledge distillation enable a compact Transformer student model to achieve competitive sentiment classification performance compared to a large pre-trained teacher while significantly reducing model size and inference latency on CPU hardware?

**Specific Goals:**
1. Implement a compact two-layer Transformer student architecture
2. Train baseline student with hard labels (ground truth only)
3. Train distilled student with hard labels + teacher soft targets
4. Compare models on accuracy, precision, recall, macro-F1, latency, and size
5. Quantify accuracy-efficiency trade-offs
6. Deploy models in an interactive web application

**Success Criteria:**
- Student model must be smaller than teacher (parameter count and checkpoint size)
- Student inference must be faster than teacher on CPU
- Distilled student should outperform or match baseline student
- System must handle real-world text inputs gracefully
- Results must be reproducible from documented configurations

---

## 4. Objectives

### 4.1 Primary Objectives

1. **Design and Implement Student Architecture**
   - Compact Transformer encoder with ~100K-500K parameters
   - Token and positional embeddings
   - Multi-head self-attention with masking
   - Classification head for binary sentiment

2. **Implement Knowledge Distillation Pipeline**
   - Generate teacher soft targets with temperature scaling
   - Implement combined hard-label and soft-target loss
   - Train baseline (hard only) and distilled (hard + soft) students

3. **Evaluate and Compare Models**
   - Classification metrics: accuracy, precision, recall, macro-F1
   - Efficiency metrics: inference latency, throughput, checkpoint size
   - Trade-off analysis: accuracy vs. speed/size

4. **Develop Web Application**
   - Live sentiment prediction interface
   - Batch CSV processing
   - Model comparison visualizations
   - Methodology documentation

### 4.2 Secondary Objectives

5. **Ensure Reproducibility**
   - Fixed random seeds
   - Documented hyperparameters
   - Saved configurations and checkpoints

6. **Educational Value**
   - Clean, well-commented code
   - Comprehensive documentation
   - Academic report template

---

## 5. Background and Theory

### 5.1 Transformer Architecture

**Self-Attention Mechanism:**

For input sequence X = [x₁, x₂, ..., xₙ], self-attention computes:

```
Q = XWq, K = XWk, V = XWv
Attention(Q, K, V) = softmax(QK^T / √d_k)V
```

Where:
- Q, K, V are query, key, value matrices
- d_k is the key dimension
- Scaling by √d_k prevents softmax saturation

**Multi-Head Attention:**

Multiple attention heads learn different representation subspaces:

```
MultiHead(Q, K, V) = Concat(head₁, ..., headₕ)W^O
head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

**Positional Encoding:**

Since Transformers have no inherent sequence order, positional encodings are added:

```
PE(pos, 2i) = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```

### 5.2 Knowledge Distillation Theory

**Motivation:**  
Hard labels (one-hot vectors) provide minimal information—they only indicate the correct class. Soft targets from a teacher model reveal:
- Confidence in predictions
- Similarities between classes
- Decision boundaries learned by the teacher

**Temperature-Scaled Softmax:**

Standard softmax: `p_i = exp(z_i) / Σ_j exp(z_j)`

Temperature-scaled: `p_i = exp(z_i/T) / Σ_j exp(z_j/T)`

Higher temperature T > 1:
- Softens probability distribution
- Reveals relative class probabilities
- Provides richer training signal

**Distillation Loss:**

```
L_total = α · L_hard + (1 - α) · T² · L_soft

L_hard = CrossEntropy(student_logits, true_labels)
L_soft = KL_Divergence(softmax(teacher_logits/T), softmax(student_logits/T))
```

Where:
- α ∈ [0, 1] balances hard and soft losses
- T² scaling ensures gradients match in magnitude
- L_hard ensures correct classifications
- L_soft transfers teacher's knowledge

**Why KL Divergence?**

KL divergence measures how one probability distribution differs from another:

```
KL(P || Q) = Σ_i P(i) · log(P(i) / Q(i))
```

Minimizing KL divergence makes the student distribution match the teacher's.

---

## 6. Proposed System

### 6.1 System Overview

The SentimentLab AI system consists of four main components:

1. **Data Pipeline:** Load, preprocess, and tokenize IMDb reviews
2. **Model Training:** Train baseline and distilled students
3. **Evaluation Framework:** Measure quality and efficiency metrics
4. **Web Application:** Interactive sentiment analysis interface

### 6.2 System Features

**Training Features:**
- Configurable dataset sizes for experiments
- Reproducible training with fixed seeds
- Progress tracking and loss visualization
- Automatic checkpoint saving

**Evaluation Features:**
- Classification metrics with scikit-learn
- CPU latency measurement
- Model size calculation
- Comparison report generation

**Web Application Features:**
- Live text sentiment analysis
- Batch CSV processing with download
- Interactive model comparison charts
- Methodology and limitations documentation

---

## 7. System Architecture

### 7.1 High-Level Architecture

```
┌─────────────────┐
│ IMDb Dataset    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Tokenization   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐      ┌──────────────────┐
│ Teacher Model   │─────▶│  Soft Targets    │
│  (DistilBERT)   │      │  (Logits/Probs)  │
└─────────────────┘      └────────┬─────────┘
                                  │
                                  ▼
                         ┌────────────────────┐
                         │ Student Training   │
                         │  - Baseline (Hard) │
                         │  - Distilled       │
                         │    (Hard + Soft)   │
                         └────────┬───────────┘
                                  │
                                  ▼
                         ┌────────────────────┐
                         │   Evaluation       │
                         │  - Accuracy        │
                         │  - Latency         │
                         │  - Model Size      │
                         └────────┬───────────┘
                                  │
                                  ▼
                         ┌────────────────────┐
                         │ Web Application    │
                         │  - Live Analysis   │
                         │  - Batch Process   │
                         │  - Visualization   │
                         └────────────────────┘
```

### 7.2 Student Model Architecture

**Layer Structure:**

```
Input Text
    ↓
[Tokenization] → Token IDs [batch, seq_len]
    ↓
[Token Embedding] → [batch, seq_len, d_model=128]
    ⊕
[Positional Embedding] → [batch, seq_len, d_model=128]
    ↓
[Transformer Encoder Layer 1]
    - Multi-Head Attention (4 heads)
    - Feed-Forward Network (dim=256)
    - Layer Norm + Dropout
    ↓
[Transformer Encoder Layer 2]
    - Multi-Head Attention (4 heads)
    - Feed-Forward Network (dim=256)
    - Layer Norm + Dropout
    ↓
[Masked Mean Pooling] → [batch, d_model]
    ↓
[Dropout] → [batch, d_model]
    ↓
[Layer Norm] → [batch, d_model]
    ↓
[Linear Classifier] → [batch, 2]
    ↓
Logits (Negative, Positive)
```

**Key Design Choices:**
- 2 layers (vs. teacher's 6): reduces parameters
- d_model=128 (vs. teacher's 768): smaller hidden size
- 4 attention heads: enough for pattern learning
- Masked mean pooling: handles variable-length sequences
- No pre-training: trained from scratch on task

---

## 8. Dataset Description

### 8.1 IMDb Movie Reviews Dataset

**Source:** Large Movie Review Dataset (Maas et al., 2011)  
**Access:** Hugging Face Datasets: `imdb`  
**Task:** Binary sentiment classification (positive/negative)

**Full Dataset:**
- Training: 25,000 reviews
- Test: 25,000 reviews
- Classes: 12,500 positive, 12,500 negative (balanced)
- Format: Free-form English text, variable length

**Sample Review:**

*Positive:* "This movie was absolutely fantastic! The acting was superb, the plot kept me engaged throughout, and the cinematography was breathtaking. Highly recommended!"

*Negative:* "What a waste of time. The plot made no sense, the dialogue was cringe-worthy, and I found myself checking my watch every five minutes. Do not watch this."

### 8.2 Dataset Preprocessing

**Subset Selection:**
- Training subset: [Fill actual size used, e.g., 800 samples]
- Test subset: [Fill actual size used, e.g., 300 samples]
- Balanced sampling: Equal positive and negative examples
- Random shuffling with fixed seed for reproducibility

**Tokenization:**
- Tokenizer: DistilBERT WordPiece tokenizer (vocab size ~30K)
- Max sequence length: [Fill actual, e.g., 96 tokens]
- Padding: To max length
- Truncation: Longer sequences truncated
- Special tokens: [CLS] at start, [SEP] at end

**Why Subsets?**
- CPU training time constraints
- Rapid prototyping and experimentation
- Educational focus on methodology
- Full-scale training recommended for publication

### 8.3 Domain Shift Consideration

**Important Limitation:**  
The teacher model (DistilBERT-SST-2) was fine-tuned on the Stanford Sentiment Treebank (SST-2), which contains short movie review snippets (typically single sentences). The evaluation dataset (IMDb) contains full-length movie reviews (often multiple paragraphs).

This **domain shift** may affect:
- Teacher prediction quality on long reviews
- Student learning from potentially noisy teacher signals
- Overall system performance

**Mitigation Strategies** (Future Work):
- Fine-tune teacher on IMDb training set
- Use a teacher originally trained on IMDb
- Analyze teacher prediction quality on IMDb separately

---

## 9. Methodology

### 9.1 Experimental Workflow

**Step 1: Data Preparation**
- Load IMDb dataset from Hugging Face
- Sample balanced training and test subsets
- Tokenize using DistilBERT tokenizer
- Create PyTorch DataLoaders

**Step 2: Teacher Logits Generation**
- Load pre-trained DistilBERT-SST-2 model
- Run inference on training set (evaluation mode)
- Extract and cache logits (before softmax)
- Save to disk for student training

**Step 3: Baseline Student Training**
- Initialize student model with random weights
- Train with cross-entropy loss on hard labels
- Use AdamW optimizer with weight decay
- Apply gradient clipping for stability
- Save best checkpoint

**Step 4: Distilled Student Training**
- Initialize new student (same architecture as baseline)
- Train with combined hard + soft loss
- Use cached teacher logits
- Match training hyperparameters to baseline
- Save best checkpoint

**Step 5: Evaluation**
- Load both student checkpoints
- Evaluate on held-out test set
- Compute classification metrics
- Measure inference latency
- Calculate checkpoint sizes

**Step 6: Deployment**
- Package models and tokenizer
- Build Streamlit web application
- Create comparison visualizations
- Document usage and limitations

### 9.2 Training Hyperparameters

| Hyperparameter | Value | Rationale |
|---|---|---|
| Learning Rate | 3e-4 | Standard for small Transformers |
| Weight Decay | 0.01 | Regularization to prevent overfitting |
| Batch Size | [Fill, e.g., 8] | CPU memory constraint |
| Epochs | [Fill, e.g., 3] | Convergence on small dataset |
| Gradient Clip | 1.0 | Prevent exploding gradients |
| Temperature (T) | 2.0 | Softens teacher probabilities |
| Alpha (α) | 0.5 | Equal weight to hard/soft losses |
| Max Length | [Fill, e.g., 96] | Balance context vs. speed |
| Random Seed | 42 | Reproducibility |
| Optimizer | AdamW | Adaptive learning with weight decay |

### 9.3 Loss Functions

**Baseline Student Loss:**

```python
L_baseline = CrossEntropyLoss(student_logits, true_labels)
```

**Distilled Student Loss:**

```python
L_hard = CrossEntropyLoss(student_logits, true_labels)

L_soft = KLDivLoss(
    log_softmax(student_logits / T), 
    softmax(teacher_logits / T)
) * T²

L_distilled = α * L_hard + (1 - α) * L_soft
```

**Implementation Details:**
- KL divergence uses `reduction='batchmean'` for proper averaging
- T² scaling ensures gradient magnitudes match between hard and soft
- α=0.5 gives equal importance to both objectives

---

## 10. Implementation Details

### 10.1 Technology Stack

**Core Frameworks:**
- Python 3.10+
- PyTorch 2.2+ (deep learning)
- Transformers 4.44+ (Hugging Face)
- Datasets 2.20+ (data loading)

**Evaluation & Metrics:**
- scikit-learn 1.4+ (metrics calculation)
- NumPy 1.26+ (numerical operations)
- Pandas 2.1+ (data manipulation)

**Web Application:**
- Streamlit 1.36+ (web framework)
- Plotly 5.22+ (interactive charts)

**Development Tools:**
- pytest (automated testing)
- tqdm (progress bars)
- Git (version control)

### 10.2 Project Structure

```
sentimentlab_ai/
├── src/                    # Source code modules
│   ├── model.py           # Student architecture
│   ├── train.py           # Main training script
│   ├── data.py            # Dataset loading
│   ├── distillation.py    # Loss functions
│   ├── evaluate.py        # Metrics computation
│   ├── inference.py       # Prediction interface
│   ├── metrics.py         # Results tracking
│   └── utils.py           # Helper functions
├── pages/                  # Streamlit pages
│   ├── 1_📊_Overview.py
│   ├── 2_🎯_Live_Analysis.py
│   ├── 3_📈_Model_Comparison.py
│   ├── 4_📁_Batch_Analysis.py
│   ├── 5_🔬_Research_Lab.py
│   └── 6_ℹ️_About_Project.py
├── tests/                  # Automated tests
│   ├── test_model.py
│   ├── test_distillation.py
│   ├── test_metrics.py
│   └── test_inference.py
├── scripts/                # Standalone scripts
│   ├── train_baseline.py
│   ├── train_distilled.py
│   └── run_evaluation.py
├── results/                # Training outputs
│   ├── metrics.json
│   ├── student_baseline.pt
│   ├── student_distilled.pt
│   └── tokenizer/
├── report/                 # Documentation
│   ├── report_template.md
│   ├── methodology.md
│   └── experiment_log.md
├── app.py                  # Main Streamlit app
├── requirements.txt        # Dependencies
└── README.md              # Project documentation
```

### 10.3 Key Implementation Decisions

**Model Design:**
- Two Transformer encoder layers balances capacity and efficiency
- d_model=128 keeps parameters low while maintaining expressiveness
- Masked mean pooling aggregates variable-length sequences

**Training Strategy:**
- Separate baseline and distilled models for fair comparison
- Identical architecture ensures differences are from distillation only
- Fixed hyperparameters across both approaches

**Evaluation Approach:**
- Warm-up iterations before latency measurement
- Multiple batches averaged for stable latency estimates
- CPU-only to match deployment target

**Web Application:**
- Streamlit for rapid development and deployment
- Client-side model loading with caching
- Responsive design for desktop and mobile

---

## 11. Experimental Setup

### 11.1 Hardware and Software

**Hardware:**
- Processor: [Fill your CPU model]
- RAM: [Fill your RAM amount]
- Storage: [Fill storage type, e.g., SSD]
- GPU: None (CPU-only training and inference)

**Software:**
- Operating System: Windows [Fill version]
- Python Version: [Fill version]
- PyTorch Version: [Fill version]
- CUDA: N/A (CPU-only)

### 11.2 Training Configuration

[Fill from actual experimental run]

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
  "alpha_hard_loss": 0.5,
  "seed": 42,
  "device": "CPU"
}
```

### 11.3 Reproducibility

**Random Seed Control:**
- Python random seed: 42
- NumPy random seed: 42
- PyTorch manual seed: 42
- Dataset shuffling seed: 42 + dataset_size

**Deterministic Operations:**
- Fixed data splits
- Consistent tokenization
- Same training order

**Version Control:**
- All code committed to Git
- Dependencies pinned in requirements.txt
- Configuration saved with results

---

## 12. Results and Analysis

### 12.1 Quantitative Results

[Fill this table from results/metrics.json after training]

| Model | Accuracy | Precision | Recall | Macro-F1 | Latency (ms) | Size (MB) | Params |
|---|---:|---:|---:|---:|---:|---:|---:|
| Teacher (DistilBERT) | - | - | - | - | - | ~260 | ~67M |
| Baseline Student | X.XX | X.XX | X.XX | X.XX | X.XX | X.XX | ~XXXK |
| Distilled Student | X.XX | X.XX | X.XX | X.XX | X.XX | X.XX | ~XXXK |

**Performance Deltas:**

- Accuracy difference (Distilled - Baseline): ___%
- Latency speedup (Distilled vs. Teacher): ___x
- Size reduction (Student vs. Teacher): ___%
- Parameter reduction: ___%

### 12.2 Training Dynamics

[Include training loss curves]

**Baseline Student Training:**
- Epoch 1 Loss: ___
- Epoch 2 Loss: ___
- Epoch 3 Loss: ___
- Training Time: ___ seconds

**Distilled Student Training:**
- Epoch 1: Hard ___, Soft ___, Total ___
- Epoch 2: Hard ___, Soft ___, Total ___
- Epoch 3: Hard ___, Soft ___, Total ___
- Training Time: ___ seconds

### 12.3 Qualitative Analysis

[After training, test both models on example sentences]

**Example 1:**  
*Text:* "This movie was absolutely fantastic! Loved every minute."  
- Teacher: Positive (99% confidence)
- Baseline: [Fill]
- Distilled: [Fill]

**Example 2:**  
*Text:* "Boring and poorly acted. Complete waste of time."  
- Teacher: Negative (98% confidence)
- Baseline: [Fill]
- Distilled: [Fill]

**Example 3:**  
*Text:* "It was okay, nothing special but not terrible either."  
- Teacher: [Fill]
- Baseline: [Fill]
- Distilled: [Fill]

### 12.4 Confusion Matrices

[Insert confusion matrices from evaluation]

### 12.5 Web Application Screenshots

[Include screenshots of:]
1. Overview dashboard
2. Live analysis with predictions
3. Model comparison charts
4. Batch analysis interface

---

## 13. Discussion

### 13.1 Effectiveness of Knowledge Distillation

[Analyze whether distillation helped based on actual results]

**If Distilled > Baseline:**
The distilled student outperforms the baseline, demonstrating that teacher soft targets provide valuable learning signal beyond hard labels. The improvement suggests the student successfully learned from the teacher's confidence patterns and inter-class relationships.

**If Distilled ≈ Baseline:**
The distilled and baseline students perform similarly, indicating that either: (1) the teacher's soft targets don't provide additional value for this task, (2) the student architecture is too limited to benefit from distillation, or (3) hyperparameters need tuning.

**If Distilled < Baseline:**
If distillation underperforms, possible explanations include: (1) domain shift between teacher training (SST-2) and evaluation (IMDb) degrades teacher quality, (2) temperature or alpha settings are suboptimal, or (3) teacher logits introduce noise.

### 13.2 Accuracy-Efficiency Trade-off

[Discuss the practical trade-offs]

The student models achieve [___]% of teacher accuracy while being [___]x faster and [___]% smaller. For deployment scenarios where:
- **Speed matters more:** Student is preferred
- **Accuracy is critical:** Teacher is preferred
- **Balanced requirements:** Student offers best value

### 13.3 Domain Shift Impact

The teacher was fine-tuned on SST-2 (short snippets) while evaluation uses IMDb (long reviews). This mismatch likely affects:
- Teacher prediction confidence and accuracy on IMDb
- Quality of soft targets for student learning
- Upper bound on student performance

Future work should fine-tune the teacher on IMDb training data to eliminate this confound.

### 13.4 Subset Limitations

Training on [___] samples is sufficient for proof-of-concept but insufficient for:
- Generalization to diverse review styles
- Statistical significance claims
- Publication-quality benchmarks

The results demonstrate the methodology but should be validated on larger datasets with multiple random seeds before drawing strong conclusions.

### 13.5 Temperature and Alpha Sensitivity

**Temperature (T = 2.0):**
- Higher T creates softer distributions (more information)
- Lower T approaches hard labels (less benefit)
- Future work: sweep T ∈ {1, 2, 4, 8}

**Alpha (α = 0.5):**
- α=0.5 balances hard and soft equally
- Higher α emphasizes correctness
- Lower α emphasizes teacher mimicry
- Future work: sweep α ∈ {0.25, 0.5, 0.75}

---

## 14. Applications

### 14.1 Practical Use Cases

**Customer Feedback Analysis:**
- Analyze product reviews in real-time
- Deploy on edge devices in retail stores
- Monitor sentiment trends without cloud dependency

**Social Media Monitoring:**
- Process tweets, comments, and posts
- Run on mobile devices for privacy
- Low-latency sentiment scoring

**Content Moderation:**
- Pre-filter negative content
- Assist human moderators
- Reduce moderation costs

**Market Research:**
- Analyze survey responses
- Track brand sentiment over time
- Generate automated reports

### 14.2 Deployment Scenarios

**Mobile Applications:**
- On-device sentiment analysis preserves privacy
- Low latency enhances user experience
- Small model size fits storage constraints

**IoT Devices:**
- Voice assistants with sentiment awareness
- Smart home devices understanding user feedback
- Wearables tracking emotional states

**Cost-Sensitive Environments:**
- Startups minimizing cloud inference costs
- Educational institutions with limited budgets
- Non-profits analyzing feedback at scale

---

## 15. Limitations

### 15.1 Research Limitations

1. **Domain Shift:**  
   Teacher trained on SST-2, evaluated on IMDb. This introduces potential teacher error and limits student learning quality.

2. **Subset Training:**  
   [___] training samples insufficient for robust generalization. Results are indicative, not conclusive.

3. **Single Seed:**  
   One random seed prevents statistical significance testing. Multiple seeds needed for confidence intervals.

4. **Single Configuration:**  
   Fixed temperature and alpha. Optimal hyperparameters may differ.

5. **Binary Classification:**  
   Only positive/negative. Real-world sentiment is often multiclass or continuous.

6. **English Only:**  
   Models trained on English text. Multilingual support requires additional work.

7. **CPU-Only:**  
   Training and evaluation on CPU. GPU training would be faster and enable larger experiments.

### 15.2 System Limitations

1. **Hardware Dependency:**  
   Latency measurements specific to test hardware. Performance varies across CPUs.

2. **Context Length:**  
   Max length [___] tokens truncates longer reviews. May lose important sentiment information.

3. **No Calibration:**  
   Predicted probabilities not calibrated. Confidence scores may not reflect true certainty.

4. **Static Models:**  
   No online learning or adaptation to user feedback.

5. **No Explainability:**  
   Black-box predictions without word-level importance scores.

### 15.3 Ethical Considerations

1. **Bias:**  
   Models may perpetuate biases present in training data (e.g., demographic, cultural).

2. **Misuse Potential:**  
   Could be used for manipulation or surveillance if deployed irresponsibly.

3. **Error Impact:**  
   Misclassifications could affect business decisions or user experiences.

4. **Transparency:**  
   Users should know they're interacting with an AI system.

---

## 16. Future Work

### 16.1 Methodological Improvements

1. **Domain-Matched Teacher:**  
   Fine-tune teacher on IMDb training set to eliminate domain shift.

2. **Hyperparameter Search:**  
   Grid search over temperature T ∈ {1, 2, 4, 8} and alpha α ∈ {0.25, 0.5, 0.75}.

3. **Multiple Seeds:**  
   Run 5-10 experiments with different seeds to report mean ± std.

4. **Full Dataset:**  
   Train on complete IMDb training set (25K samples) for robust evaluation.

5. **Architecture Exploration:**  
   Vary student layers (1, 2, 3), hidden dimensions (64, 128, 256), attention heads.

6. **Advanced Distillation:**  
   Explore feature-based distillation, attention transfer, layer-wise distillation.

### 16.2 Baselines and Comparisons

1. **Classical ML:**  
   Compare against TF-IDF + Logistic Regression, SVM.

2. **Other Compression:**  
   Evaluate quantization (8-bit), pruning, low-rank factorization.

3. **Different Teachers:**  
   Try BERT-base, RoBERTa, or sentiment-specific teachers.

4. **No Teacher:**  
   Stronger baseline with data augmentation, regularization, architecture search.

### 16.3 System Enhancements

1. **Explainability:**  
   Add attention visualization, SHAP values, or LIME explanations.

2. **Calibration:**  
   Apply temperature scaling or Platt scaling for better confidence estimates.

3. **Multiclass Sentiment:**  
   Extend to 3-class (positive/neutral/negative) or 5-star ratings.

4. **Multilingual Support:**  
   Distill multilingual BERT for cross-lingual sentiment.

5. **Active Learning:**  
   Implement feedback loop where user corrections improve models.

6. **API Deployment:**  
   Package as REST API for integration with other systems.

### 16.4 Research Directions

1. **Theoretical Analysis:**  
   Study why distillation works—information theory, optimization landscape.

2. **Self-Distillation:**  
   Can a model distill its own knowledge to improve performance?

3. **Continual Learning:**  
   Distill new knowledge without forgetting old knowledge.

4. **Cross-Task Distillation:**  
   Transfer knowledge from multiple teachers across different tasks.

---

## 17. Conclusion

### 17.1 Summary

This project successfully implemented and evaluated knowledge distillation for sentiment analysis, demonstrating:

1. **Feasibility:**  
   A compact student Transformer can learn sentiment classification from a large teacher model.

2. **Efficiency Gains:**  
   Student models achieve [___]x speedup and [___]% size reduction compared to the teacher.

3. **Accuracy Trade-off:**  
   Student retains [___]% of teacher accuracy while being significantly more efficient.

4. **Practical System:**  
   End-to-end pipeline with training, evaluation, and interactive web application.

### 17.2 Key Contributions

1. **Reproducible Implementation:**  
   Complete codebase with modular architecture, automated tests, and documentation.

2. **Comparative Evaluation:**  
   Rigorous comparison of baseline vs. distilled students on quality and efficiency.

3. **Interactive Application:**  
   Polished web interface demonstrating practical deployment of distilled models.

4. **Educational Resource:**  
   Comprehensive documentation suitable for learning knowledge distillation.

### 17.3 Learning Outcomes

Through this project, I gained experience in:
- Transformer architecture design and implementation
- Knowledge distillation theory and practice
- PyTorch deep learning framework
- Model evaluation and benchmarking
- Web application development with Streamlit
- Software engineering best practices
- Academic writing and documentation

### 17.4 Final Remarks

Knowledge distillation offers a promising approach to deploying NLP models in resource-constrained environments. While this work has limitations (domain shift, subset size, single configuration), it provides a solid foundation for future research and demonstrates the practical viability of distilled sentiment classifiers.

The accuracy-efficiency trade-off must be evaluated case-by-case based on application requirements. For scenarios where real-time inference, privacy, or cost are paramount, distilled models like the one developed here present a compelling alternative to large pre-trained models.

---

## 18. References

### 18.1 Foundational Papers

1. **Hinton, G., Vinyals, O., & Dean, J.** (2015). *Distilling the Knowledge in a Neural Network*. arXiv:1503.02531.  
   https://arxiv.org/abs/1503.02531

2. **Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I.** (2017). *Attention is All You Need*. In NeurIPS.  
   https://arxiv.org/abs/1706.03762

3. **Devlin, J., Chang, M. W., Lee, K., & Toutanova, K.** (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. In NAACL-HLT.  
   https://arxiv.org/abs/1810.04805

### 18.2 Knowledge Distillation

4. **Sanh, V., Debut, L., Chaumond, J., & Wolf, T.** (2019). *DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter*. arXiv:1910.01108.  
   https://arxiv.org/abs/1910.01108

5. **Ba, J., & Caruana, R.** (2014). *Do Deep Nets Really Need to be Deep?* In NeurIPS.  
   https://arxiv.org/abs/1312.6184

6. **Jiao, X., Yin, Y., Shang, L., Jiang, X., Chen, X., Li, L., Wang, F., & Liu, Q.** (2020). *TinyBERT: Distilling BERT for Natural Language Understanding*. In EMNLP.  
   https://arxiv.org/abs/1909.10351

7. **Sun, S., Cheng, Y., Gan, Z., & Liu, J.** (2020). *Patient Knowledge Distillation for BERT Model Compression*. In EMNLP.  
   https://arxiv.org/abs/1908.09355

### 18.3 Model Compression

8. **Han, S., Pool, J., Tran, J., & Dally, W.** (2015). *Learning both Weights and Connections for Efficient Neural Networks*. In NeurIPS.  
   https://arxiv.org/abs/1506.02626

9. **Gou, J., Yu, B., Maybank, S. J., & Tao, D.** (2021). *Knowledge Distillation: A Survey*. International Journal of Computer Vision, 129, 1789-1819.  
   https://arxiv.org/abs/2006.05525

### 18.4 Sentiment Analysis

10. **Maas, A. L., Daly, R. E., Pham, P. T., Huang, D., Ng, A. Y., & Potts, C.** (2011). *Learning Word Vectors for Sentiment Analysis*. In ACL.  
    http://www.aclweb.org/anthology/P11-1015

11. **Socher, R., Perelygin, A., Wu, J., Chuang, J., Manning, C. D., Ng, A. Y., & Potts, C.** (2013). *Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank*. In EMNLP.  
    https://nlp.stanford.edu/~socherr/EMNLP2013_RNTN.pdf

### 18.5 Libraries and Tools

12. **Hugging Face Transformers Documentation:**  
    https://huggingface.co/docs/transformers

13. **PyTorch Documentation:**  
    https://pytorch.org/docs

14. **Streamlit Documentation:**  
    https://docs.streamlit.io

15. **IMDb Dataset (Hugging Face):**  
    https://huggingface.co/datasets/imdb

16. **DistilBERT-SST-2 Model:**  
    https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english

---

## 19. Appendix

### Appendix A: Installation Instructions

**System Requirements:**
- Python 3.10 or 3.11
- 4GB+ RAM
- 5GB+ free disk space
- Internet connection (first run only)

**Setup Steps:**

```powershell
# Clone or extract project
cd path\to\sentimentlab_ai

# Create virtual environment
python -m venv .venv

# Activate virtual environment
.venv\Scripts\activate

# Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Appendix B: Training Commands

**Smoke Test (2-5 minutes):**
```powershell
python src/train.py --train-size 200 --test-size 100 --epochs 1 --batch-size 4 --max-length 96
```

**Standard Training (15-30 minutes):**
```powershell
python src/train.py --train-size 800 --test-size 300 --epochs 3 --batch-size 8 --max-length 96
```

**Custom Configuration:**
```powershell
python src/train.py `
    --train-size 1000 `
    --test-size 500 `
    --epochs 5 `
    --batch-size 16 `
    --max-length 128 `
    --temperature 3.0 `
    --alpha 0.7
```

### Appendix C: Running the Web Application

```powershell
# Make sure virtual environment is activated
.venv\Scripts\activate

# Launch Streamlit app
streamlit run app.py

# Open browser to http://localhost:8501
```

### Appendix D: Running Tests

```powershell
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_model.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

### Appendix E: Project File Descriptions

| File/Directory | Purpose |
|---|---|
| `src/model.py` | Student Transformer architecture |
| `src/train.py` | Main training script (baseline + distilled) |
| `src/data.py` | Dataset loading and preprocessing |
| `src/distillation.py` | Distillation loss functions |
| `src/evaluate.py` | Model evaluation utilities |
| `src/inference.py` | Prediction interface |
| `src/metrics.py` | Metrics tracking and comparison |
| `app.py` | Main Streamlit application |
| `pages/` | Individual Streamlit page files |
| `tests/` | Automated test suite |
| `scripts/` | Standalone training scripts |
| `results/` | Training outputs and metrics |
| `report/` | Documentation and reports |

### Appendix F: Hyperparameter Tuning Guide

**For Better Accuracy:**
- Increase `--train-size` (more data)
- Increase `--epochs` (more training)
- Decrease `--learning-rate` (more careful updates)
- Increase `--max-length` (more context)

**For Faster Training:**
- Decrease `--train-size` (less data)
- Decrease `--epochs` (less training)
- Increase `--batch-size` (if memory allows)
- Decrease `--max-length` (less computation)

**For Distillation Tuning:**
- Increase `--temperature` (softer targets)
- Adjust `--alpha` (balance hard/soft)
- Try `--alpha 0.25` (more teacher influence)
- Try `--alpha 0.75` (more label correctness)

### Appendix G: Troubleshooting

**Problem:** Out of memory during training  
**Solution:** Reduce `--batch-size` to 4 or 2, reduce `--train-size`, or reduce `--max-length`

**Problem:** Training is very slow  
**Solution:** Expected on CPU. Start with smoke test. Consider cloud GPU for larger experiments.

**Problem:** Import errors  
**Solution:** Ensure virtual environment is activated and dependencies are installed: `pip install -r requirements.txt`

**Problem:** Streamlit shows "Student not trained"  
**Solution:** Run `python src/train.py` first to create model checkpoints

**Problem:** Results differ from report  
**Solution:** Results depend on random seed, dataset size, and hardware. Use same configuration for reproduction.

### Appendix H: Code Snippets

**Creating a SentimentPredictor:**

```python
from pathlib import Path
from src.inference import create_predictor_from_checkpoint

# Load distilled student
predictor = create_predictor_from_checkpoint(
    checkpoint_path=Path("results/student_distilled.pt"),
    tokenizer_path=Path("results/tokenizer"),
    max_length=96
)

# Make prediction
result = predictor.predict("This movie was amazing!")
print(f"Sentiment: {result}")
```

**Loading Metrics:**

```python
import json
from pathlib import Path

with open("results/metrics.json", "r") as f:
    metrics = json.load(f)

print(f"Distilled Accuracy: {metrics['distilled_student']['accuracy']:.2%}")
```

### Appendix I: Glossary

- **Attention:** Mechanism that weighs importance of different input tokens
- **DistilBERT:** Distilled version of BERT with fewer parameters
- **Distillation:** Training a small model to mimic a large model
- **Epoch:** One complete pass through the training dataset
- **IMDb:** Internet Movie Database review dataset
- **KL Divergence:** Measure of difference between probability distributions
- **Logits:** Raw model outputs before softmax
- **Macro-F1:** F1 score averaged across classes
- **Sentiment Analysis:** Identifying opinion/emotion in text
- **Soft Targets:** Probability distributions from teacher model
- **Temperature:** Hyperparameter that softens probability distributions
- **Tokenization:** Converting text to numerical tokens
- **Transformer:** Neural architecture based on self-attention

### Appendix J: Acknowledgment of Tools

This project uses the following open-source tools:

- **PyTorch:** Facebook AI Research
- **Hugging Face Transformers:** Hugging Face Inc.
- **Streamlit:** Streamlit Inc.
- **IMDb Dataset:** Stanford University (Maas et al., 2011)
- **DistilBERT Model:** Hugging Face (Sanh et al., 2019)
- **scikit-learn:** scikit-learn developers
- **Plotly:** Plotly Technologies Inc.

All tools are used under their respective open-source licenses.

---

**End of Report Template**

---

## Instructions for Filling This Template

1. **After Training:**
   - Run your experiments: `python src/train.py --train-size 800 --test-size 300 --epochs 3`
   - Open `results/metrics.json` and copy actual values into Section 12 (Results)
   - Replace all [Fill ...] placeholders with real data

2. **Personal Information:**
   - Fill in your name, roll number, institution, guide name in the header
   - Update Declaration and Certificate sections

3. **Screenshots:**
   - Take screenshots of the Streamlit app after training
   - Insert in Section 12.5 and any relevant appendices

4. **Discussion:**
   - Section 13 should analyze YOUR actual results
   - Compare baseline vs. distilled using real metrics
   - Discuss whether distillation helped in your case

5. **Hardware:**
   - Fill Section 11.1 with your actual CPU, RAM, OS details

6. **Formatting:**
   - Convert to PDF for submission
   - Ensure all tables, figures, and equations render correctly
   - Add page numbers and table of contents

7. **Length:**
   - Aim for 40-60 pages including appendices
   - Remove sections not required by your institution
   - Add sections if your guidelines require more

**Note:** Never fabricate results. Only report what you actually measured. Clearly distinguish between planned experiments and completed experiments.
