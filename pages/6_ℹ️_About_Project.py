"""About Project Page"""

import streamlit as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEACHER = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"

st.set_page_config(page_title="About - SentimentLab AI", page_icon="ℹ️", layout="wide")

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

# Hero
hero(
    "About SentimentLab AI",
    "A reproducible college major project on transferring classification behavior from a teacher to a smaller student.",
    "PROJECT GUIDE"
)

# Setup instructions
st.markdown("### 🚀 Setup Instructions (Windows PowerShell)")

st.markdown("""
Follow these steps to set up and run the project on your Windows machine:

#### 1. Prerequisites
- Python 3.10 or 3.11 installed
- At least 4GB RAM available
- Internet connection for initial setup (downloads models and dataset)
""")

st.code(r"""# Navigate to project directory
cd "path\to\sentimentlab_ai"

# Create virtual environment
python -m venv .venv

# Activate virtual environment
.venv\Scripts\activate

# Upgrade pip
python -m pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt""", language="powershell")

st.markdown("#### 2. Train Models")

st.markdown("**Quick Smoke Test** (validates setup, ~2-5 minutes on CPU):")
st.code(r"""python src/train.py --train-size 200 --test-size 100 --epochs 1 --batch-size 4 --max-length 96""", language="powershell")

st.markdown("**Standard Training** (for meaningful results, ~15-30 minutes on CPU):")
st.code(r"""python src/train.py --train-size 800 --test-size 300 --epochs 3 --batch-size 8 --max-length 96""", language="powershell")

st.markdown("#### 3. Launch Web Application")
st.code(r"""streamlit run app.py""", language="powershell")

st.markdown("The app will open in your browser at `http://localhost:8501`")

# Research design
st.markdown("---")
st.markdown("### 🎓 Research Design")

col1, col2 = st.columns(2)

with col1:
    st.markdown(f"""
    **Teacher Model**
    - Model: `{TEACHER}`
    - Pre-trained on SST-2 sentiment dataset
    - Used to generate soft targets
    - Not trained or fine-tuned
    
    **Student Model**
    - Compact 2-layer Transformer encoder
    - ~128 dimensional embeddings
    - 4 attention heads
    - Trained from scratch
    """)

with col2:
    st.markdown("""
    **Training Approaches**
    1. **Baseline**: Hard labels only (cross-entropy)
    2. **Distilled**: Hard labels + teacher soft targets
    
    **Dataset**
    - IMDb movie reviews
    - Binary classification (positive/negative)
    - Balanced subsets for training and testing
    
    **Evaluation Metrics**
    - Accuracy, Macro-F1, Precision, Recall
    - CPU inference latency (ms/sample)
    - Model checkpoint size (MB)
    """)

# Important notes
st.markdown("---")
st.markdown("### ⚠️ Important Research Considerations")

st.warning("""
**Domain Shift**: The teacher is fine-tuned on SST-2 (short movie review snippets) 
while the experiment evaluates on IMDb (full-length reviews). This domain mismatch 
must be discussed in your project report as a limitation.
""")

st.warning("""
**Subset Results**: Training on subsets (200-1000 samples) is for validation and 
rapid prototyping. These are NOT full-benchmark results. For publication-quality 
results, use larger datasets and multiple random seeds.
""")

st.info("""
**CPU Performance**: This project is designed for CPU-only hardware. Training times 
will be longer than GPU-based experiments. Start with small configurations and scale up gradually.
""")

# Report structure
st.markdown("---")
st.markdown("### 📝 Project Report Structure")

st.markdown("""
Your B.Tech CSE major project report should include:

1. **Title Page** - Project title, your name, institution, date
2. **Abstract** - Concise summary of objective, method, and results
3. **Introduction** - Background on sentiment analysis and model compression
4. **Problem Statement** - Research question and motivation
5. **Objectives** - Specific goals of the project
6. **Literature Review** - Survey of knowledge distillation and related work
7. **Methodology** - Detailed description of approach
   - Teacher model architecture and selection
   - Student model design
   - Distillation loss formulation
   - Training procedure
8. **Implementation** - Technical details
   - Software stack and dependencies
   - Dataset preparation
   - Hyperparameter selection
9. **Experimental Setup** - Hardware, configurations, reproducibility
10. **Results** - Tables, graphs, and metrics from actual experiments
11. **Discussion** - Analysis of results, comparison to baselines
12. **Limitations** - Honest discussion of constraints and weaknesses
13. **Future Work** - Extensions and improvements
14. **Conclusion** - Summary of findings and contributions
15. **References** - Citations to papers, libraries, datasets
16. **Appendix** - Code snippets, additional tables, setup instructions
""")

st.markdown("📄 See `report/report_template.md` for a detailed template.")

# References
st.markdown("---")
st.markdown("### 📚 Key References")

st.markdown("""
**Foundational Papers:**
- Hinton, G., Vinyals, O., & Dean, J. (2015). *Distilling the Knowledge in a Neural Network*. 
  arXiv:1503.02531
- Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). *DistilBERT, a distilled version of BERT: 
  smaller, faster, cheaper and lighter*. arXiv:1910.01108

**Libraries & Tools:**
- Hugging Face Transformers: https://huggingface.co/docs/transformers
- PyTorch: https://pytorch.org/docs
- Streamlit: https://docs.streamlit.io
- IMDb Dataset: https://huggingface.co/datasets/imdb

**Related Work:**
- Ba, J., & Caruana, R. (2014). *Do Deep Nets Really Need to be Deep?*
- Buciluǎ, C., Caruana, R., & Niculescu-Mizil, A. (2006). *Model Compression*
""")

# Troubleshooting
st.markdown("---")
st.markdown("### 🔧 Troubleshooting")

with st.expander("❌ Import errors or missing modules"):
    st.markdown("""
    **Solution:**
    ```powershell
    # Make sure virtual environment is activated
    .venv\Scripts\activate
    
    # Reinstall dependencies
    pip install --upgrade -r requirements.txt
    ```
    """)

with st.expander("⚠️ Out of memory during training"):
    st.markdown("""
    **Solution:**
    - Reduce `--batch-size` to 4 or 2
    - Reduce `--train-size` and `--test-size`
    - Reduce `--max-length` to 64
    - Close other applications
    """)

with st.expander("🐌 Training is very slow"):
    st.markdown("""
    **Expected:** CPU training is slower than GPU. For reference:
    - Smoke test (200 samples, 1 epoch): ~2-5 minutes
    - Standard (800 samples, 3 epochs): ~15-30 minutes
    
    **To speed up:**
    - Start with smoke test configuration
    - Use smaller batch sizes (paradoxically can be faster on CPU)
    - Consider cloud GPU (Google Colab, Kaggle) for larger experiments
    """)

with st.expander("🔄 Streamlit shows 'Student not trained'"):
    st.markdown("""
    **Cause:** Training hasn't been run yet or failed
    
    **Solution:**
    ```powershell
    # Run training first
    python src/train.py --train-size 200 --test-size 100 --epochs 1 --batch-size 4
    
    # Check results directory
    ls results/
    
    # Should see: student_baseline.pt, student_distilled.pt, metrics.json, tokenizer/
    ```
    """)

# Contact and contributions
st.markdown("---")
st.markdown("### 📬 Project Information")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **Project Type:** B.Tech CSE Major Project
    
    **Keywords:** 
    - Knowledge Distillation
    - Sentiment Analysis
    - Model Compression
    - Transfer Learning
    - Natural Language Processing
    """)

with col2:
    st.markdown("""
    **Technology Stack:**
    - Python 3.10+
    - PyTorch
    - Hugging Face Transformers
    - Streamlit
    - scikit-learn
    - Plotly
    """)

st.markdown("---")
st.success("""
🎉 **Ready to start?** Follow the setup instructions above, run the training command, 
and explore the web interface to see knowledge distillation in action!
""")
