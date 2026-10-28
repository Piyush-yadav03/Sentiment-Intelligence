"""Research Lab Page"""

from pathlib import Path
import json
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"

st.set_page_config(page_title="Research Lab - SentimentLab AI", page_icon="🔬", layout="wide")

# Apply custom styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
.stApp{background:radial-gradient(ellipse at 15% 0%,#17274a 0%,#0a1020 48%,#080d18 100%);font-family:'DM Sans',sans-serif}
h1,h2,h3{font-family:'Space Grotesk',sans-serif;letter-spacing:-.03em}
.hero{padding:1.7rem;border:1px solid #2a3b5a;border-radius:22px;background:linear-gradient(120deg,#1a2d50,#101b2f);margin-bottom:1.2rem}
.eyebrow{color:#65e6c4;font-size:.75rem;font-weight:700;letter-spacing:.16em}
.hero h1{font-size:2.5rem;color:#f4f7ff;margin:.45rem 0}
.hero p{color:#aebcda}
</style>
""", unsafe_allow_html=True)

def hero(title, sub, tag):
    st.markdown(
        f'<div class="hero"><div class="eyebrow">{tag}</div><h1>{title}</h1><p>{sub}</p></div>',
        unsafe_allow_html=True
    )

def load_metrics():
    metrics_path = OUT / "metrics.json"
    if metrics_path.exists():
        return json.loads(metrics_path.read_text())
    return None

# Hero
hero(
    "Research lab",
    "Explore the experimental setup and methodology for your major project.",
    "METHODOLOGY"
)

metrics = load_metrics()

# Experiment configuration
st.markdown("### ⚙️ Experiment Configuration")

if metrics:
    config = {k: metrics[k] for k in [
        "teacher_model", "dataset", "device", "seed",
        "train_examples", "test_examples", "epochs",
        "batch_size", "max_length", "temperature", "alpha_hard_loss"
    ] if k in metrics}
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.json(config)
    
    with col2:
        st.markdown("#### Key Parameters")
        st.markdown(f"""
        - **Dataset:** {config.get('dataset', 'N/A')}
        - **Training samples:** {config.get('train_examples', 'N/A'):,}
        - **Test samples:** {config.get('test_examples', 'N/A'):,}
        - **Epochs:** {config.get('epochs', 'N/A')}
        - **Temperature (T):** {config.get('temperature', 'N/A')}
        - **Alpha (α):** {config.get('alpha_hard_loss', 'N/A')}
        - **Device:** {config.get('device', 'N/A')}
        """)
    
    st.markdown("---")
    
    # Limitations
    st.markdown("### ⚠️ Research Limitations")
    if "limitations" in metrics:
        for limitation in metrics["limitations"]:
            st.warning(f"⚠️ {limitation}")
else:
    st.info("Run training to populate actual experiment settings and limitations.")
    st.code("python src/train.py --train-size 800 --test-size 300 --epochs 3", language="powershell")

# Distillation objective
st.markdown("---")
st.markdown("### 🎓 Knowledge Distillation Objective")

st.latex(r"L_{total} = \alpha L_{hard} + (1-\alpha) T^2 \cdot D_{KL}(P_{teacher} \parallel P_{student})")

st.markdown("""
Where:
- **$L_{hard}$**: Cross-entropy loss with true labels (ground truth)
- **$L_{soft}$**: KL divergence between teacher and student probability distributions
- **$T$**: Temperature parameter that smooths probability distributions
- **$\\alpha$**: Weight balancing hard and soft losses (0 ≤ α ≤ 1)
- **$P_{teacher}$**: $\\text{softmax}(\\text{logits}_{teacher} / T)$
- **$P_{student}$**: $\\text{softmax}(\\text{logits}_{student} / T)$
""")

st.markdown("#### Why Temperature Scaling?")
st.markdown("""
Higher temperatures (T > 1) create softer probability distributions, revealing relationships 
between classes that the teacher has learned. This "dark knowledge" helps the student learn 
not just what's correct, but also what's similar or related.
""")

st.markdown("#### Loss Components")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Hard Loss (α · L_hard)**")
    st.markdown("""
    - Standard cross-entropy with true labels
    - Ensures student learns correct classifications
    - Direct supervision from ground truth
    - Typical weight: α = 0.5
    """)

with col2:
    st.markdown("**Soft Loss ((1-α) · T² · L_soft)**")
    st.markdown("""
    - KL divergence with teacher's soft targets
    - Transfers teacher's learned similarities
    - Scaled by T² to match gradient magnitudes
    - Typical weight: (1-α) = 0.5
    """)

# Methodology
st.markdown("---")
st.markdown("### 🔍 Experimental Methodology")

st.markdown("""
#### Training Process
1. **Data Preparation**: Load balanced subsets from IMDb train/test splits
2. **Teacher Inference**: Generate soft targets using pre-trained DistilBERT
3. **Baseline Training**: Train student using only hard labels
4. **Distilled Training**: Train student using hard labels + teacher soft targets
5. **Evaluation**: Compare both students on held-out test set

#### Metrics Tracked
- **Quality**: Accuracy, Precision, Recall, Macro-F1
- **Efficiency**: Inference latency (ms/sample), checkpoint size (MB)
- **Trade-offs**: Accuracy vs. efficiency comparison

#### Reproducibility Controls
- Fixed random seed: {seed}
- Deterministic data splits
- Consistent tokenization and max length
- Same optimizer and learning rate for both students
""".format(seed=metrics.get('seed', 42) if metrics else 42))

# Future experiments
st.markdown("---")
st.markdown("### 🚀 Future Experiments")

st.markdown("""
To strengthen this research, consider:

1. **Temperature Sweep**: Test T ∈ {1, 2, 4, 8} to find optimal softness
2. **Alpha Sweep**: Test α ∈ {0.25, 0.5, 0.75} to balance hard/soft losses
3. **Dataset Size**: Increase to 5K+ samples for more robust results
4. **Multiple Seeds**: Run 3-5 seeds to report mean ± std for statistical confidence
5. **Domain Matching**: Fine-tune teacher on IMDb to reduce domain shift
6. **Architecture Variations**: Try different student sizes (1-layer, 3-layer, varying d_model)
7. **Comparison Baselines**: Add TF-IDF + Logistic Regression, Bag-of-Words + MLP
8. **Quantization**: Compare distillation vs. quantization for compression
9. **Error Analysis**: Study where teacher and student disagree
10. **Calibration**: Measure confidence calibration using reliability diagrams
""")

# Reproducibility instructions
st.markdown("---")
st.markdown("### 🔁 Reproducing Results")

st.markdown("**Smoke Test (Quick Validation)**")
st.code("""python src/train.py --train-size 200 --test-size 100 --epochs 1 --batch-size 4 --max-length 96""", language="powershell")

st.markdown("**Standard Experiment**")
st.code("""python src/train.py --train-size 800 --test-size 300 --epochs 3 --batch-size 8 --max-length 96 --temperature 2.0 --alpha 0.5""", language="powershell")

st.markdown("**Custom Configuration**")
st.code("""python src/train.py `
    --train-size 1000 `
    --test-size 500 `
    --epochs 5 `
    --batch-size 16 `
    --max-length 128 `
    --temperature 3.0 `
    --alpha 0.7""", language="powershell")

st.info("💡 **Note**: CPU training can take significant time. Start with the smoke test to verify everything works, then scale up gradually.")
