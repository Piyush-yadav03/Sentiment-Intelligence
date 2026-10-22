"""Overview Dashboard Page"""

from pathlib import Path
import json
import sys
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"

st.set_page_config(page_title="Overview - SentimentLab AI", page_icon="📊", layout="wide")

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
.panel{background:#111b30;border:1px solid #2a3b5a;padding:1.1rem;border-radius:17px}
</style>
""", unsafe_allow_html=True)

def hero(title, sub, tag="SENTIMENT INTELLIGENCE"):
    st.markdown(
        f'<div class="hero"><div class="eyebrow">{tag}</div><h1>{title}</h1><p>{sub}</p></div>',
        unsafe_allow_html=True
    )

def load_metrics():
    metrics_path = OUT / "metrics.json"
    if metrics_path.exists():
        return json.loads(metrics_path.read_text())
    return None

# Hero section
hero(
    "Understand the sentiment behind every sentence.",
    "An interactive research platform comparing a pretrained teacher with a compact student trained through knowledge distillation."
)

# Key metrics
metrics = load_metrics()
cols = st.columns(4)

if metrics:
    values = [
        ("TRAIN EXAMPLES", f'{metrics["train_examples"]:,}'),
        ("TEST EXAMPLES", f'{metrics["test_examples"]:,}'),
        ("STUDENT ACCURACY", f'{metrics["distilled_student"]["accuracy"]*100:.1f}%'),
        ("STUDENT LATENCY", f'{metrics["distilled_student"]["latency_ms_per_sample"]:.1f} ms')
    ]
else:
    values = [
        ("TASK", "Sentiment"),
        ("TEACHER", "DistilBERT"),
        ("STUDENT", "Tiny Transformer"),
        ("HARDWARE", "CPU-first")
    ]

for col, (key, value) in zip(cols, values):
    with col:
        st.metric(key, value)

# Main content
col1, col2 = st.columns([1.2, 1])

with col1:
    st.markdown("### How it works")
    st.markdown("""
    <div class="panel">
    <b>01 · Teacher</b>
    <p>Produces soft sentiment predictions for training reviews.</p>
    <hr>
    <b>02 · Distillation</b>
    <p>Student learns from true labels and teacher probabilities.</p>
    <hr>
    <b>03 · Evaluation</b>
    <p>Compare the baseline and distilled student on held-out reviews.</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("### Explore")
    st.markdown("""
    - **Live Analysis** — compare predictions
    - **Model Comparison** — measured quality and efficiency
    - **Batch Analysis** — classify CSV files
    - **Research Lab** — methodology and limitations
    """)
    
    ready = (OUT / "student_distilled.pt").exists()
    if not ready:
        st.info("Train the models using the command on the About Project page.")
    else:
        st.success("Models trained and ready!")

# Architecture overview
st.markdown("### System Architecture")
st.markdown("""
```mermaid
graph LR
    A[IMDb Dataset] --> B[Tokenization]
    B --> C[Teacher Model]
    C --> D[Soft Targets]
    D --> E[Student Training]
    B --> E
    E --> F[Evaluation]
    F --> G[Web Application]
```
""")

st.markdown("---")
st.caption("SentimentLab AI · Knowledge Distillation Research Studio")
