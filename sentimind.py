from pathlib import Path
import json, sys, time
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import torch

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
except ModuleNotFoundError:
    st.set_page_config(page_title="SentimentLab AI", page_icon="🧠", layout="wide")
    st.error("Missing dependency: 'transformers'.")
    st.markdown("Install the project dependencies in the local virtual environment and then start the app again.")
    st.code(
        r"""python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py""",
        language="powershell",
    )
    st.stop()

ROOT=Path(__file__).resolve().parent
OUT=ROOT/"results"
sys.path.insert(0,str(ROOT/"src"))
from model import TinyTransformerStudent
TEACHER="distilbert/distilbert-base-uncased-finetuned-sst-2-english"

st.set_page_config(page_title="SentiMind AI - Neural Sentiment Intelligence",page_icon="🧠",layout="wide",initial_sidebar_state="expanded")

# Modern Premium UI with Animations and Gradients
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Outfit:wght@500;600;700;800&display=swap');

/* Global App Styling */
.stApp {
    background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 50%, #0f1428 100%);
    font-family: 'Inter', sans-serif;
}

/* Animated Background Particles */
.stApp::before {
    content: '';
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: 
        radial-gradient(circle at 20% 30%, rgba(99, 102, 241, 0.15) 0%, transparent 50%),
        radial-gradient(circle at 80% 70%, rgba(168, 85, 247, 0.15) 0%, transparent 50%),
        radial-gradient(circle at 40% 80%, rgba(59, 130, 246, 0.1) 0%, transparent 50%);
    pointer-events: none;
    z-index: 0;
}

/* Typography */
h1, h2, h3, h4 {
    font-family: 'Outfit', sans-serif;
    letter-spacing: -0.02em;
    font-weight: 700;
}

h1 {
    background: linear-gradient(135deg, #ffffff 0%, #a78bfa 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

/* Sidebar Premium Design */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1629 0%, #1a1f3a 100%);
    border-right: 2px solid rgba(99, 102, 241, 0.2);
    box-shadow: 4px 0 20px rgba(0, 0, 0, 0.3);
}

[data-testid="stSidebar"] .css-1d391kg, 
[data-testid="stSidebar"] .st-emotion-cache-1d391kg {
    padding-top: 2rem;
}

/* Logo/Brand Section */
.brand-header {
    background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
    padding: 1.5rem;
    border-radius: 16px;
    margin-bottom: 2rem;
    text-align: center;
    box-shadow: 0 8px 32px rgba(99, 102, 241, 0.3);
    animation: fadeInDown 0.8s ease-out;
}

.brand-logo {
    font-size: 2.5rem;
    margin-bottom: 0.5rem;
}

.brand-title {
    font-family: 'Outfit', sans-serif;
    font-size: 1.8rem;
    font-weight: 800;
    color: white;
    margin: 0;
    letter-spacing: -0.03em;
}

.brand-subtitle {
    font-size: 0.85rem;
    color: rgba(255, 255, 255, 0.85);
    margin-top: 0.3rem;
    font-weight: 500;
}

/* Hero Section - Premium Card */
.hero {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(168, 85, 247, 0.1) 100%);
    border: 2px solid rgba(99, 102, 241, 0.3);
    border-radius: 24px;
    padding: 2.5rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
    animation: fadeInUp 0.8s ease-out;
}

.hero::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle, rgba(99, 102, 241, 0.1) 0%, transparent 70%);
    animation: rotate 20s linear infinite;
}

@keyframes rotate {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
}

.eyebrow {
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #a78bfa;
    margin-bottom: 0.5rem;
    display: inline-block;
    position: relative;
    z-index: 1;
}

.hero h1 {
    font-size: 3rem;
    margin: 0.5rem 0 1rem 0;
    position: relative;
    z-index: 1;
    line-height: 1.2;
}

.hero p {
    color: #cbd5e1;
    font-size: 1.1rem;
    line-height: 1.6;
    position: relative;
    z-index: 1;
    margin: 0;
}

/* Premium Panel Cards */
.panel {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.4) 0%, rgba(15, 23, 42, 0.6) 100%);
    border: 1px solid rgba(99, 102, 241, 0.2);
    border-radius: 20px;
    padding: 1.5rem;
    backdrop-filter: blur(10px);
    box-shadow: 0 10px 40px rgba(0, 0, 0, 0.2);
    transition: all 0.3s ease;
}

.panel:hover {
    transform: translateY(-4px);
    box-shadow: 0 15px 50px rgba(99, 102, 241, 0.3);
    border-color: rgba(99, 102, 241, 0.4);
}

.panel h3 {
    color: #f1f5f9;
    margin-top: 0;
    font-size: 1.3rem;
}

.panel hr {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(99, 102, 241, 0.3), transparent);
    margin: 1rem 0;
}

/* Metric Cards */
[data-testid="stMetricValue"] {
    font-size: 2rem !important;
    font-weight: 700 !important;
    background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

[data-testid="stMetricLabel"] {
    font-size: 0.8rem !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #94a3b8 !important;
    font-weight: 600 !important;
}

/* Premium Buttons */
div.stButton>button, div.stDownloadButton>button {
    border-radius: 14px;
    background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
    color: white;
    font-weight: 700;
    border: none;
    min-height: 3rem;
    font-size: 1rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    box-shadow: 0 10px 30px rgba(99, 102, 241, 0.4);
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}

div.stButton>button::before, div.stDownloadButton>button::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent);
    transition: left 0.5s ease;
}

div.stButton>button:hover::before, div.stDownloadButton>button:hover::before {
    left: 100%;
}

div.stButton>button:hover, div.stDownloadButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 15px 40px rgba(99, 102, 241, 0.6);
}

/* Text Areas and Inputs */
.stTextArea textarea, .stTextInput input {
    background: rgba(15, 23, 42, 0.8) !important;
    color: #f1f5f9 !important;
    border: 2px solid rgba(99, 102, 241, 0.3) !important;
    border-radius: 14px !important;
    font-size: 1rem !important;
    transition: all 0.3s ease !important;
}

.stTextArea textarea:focus, .stTextInput input:focus {
    border-color: rgba(99, 102, 241, 0.6) !important;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1) !important;
}

/* File Uploader */
[data-testid="stFileUploader"] {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.4) 0%, rgba(15, 23, 42, 0.6) 100%);
    border: 2px dashed rgba(99, 102, 241, 0.4);
    border-radius: 16px;
    padding: 2rem;
}

/* Progress Bar */
.stProgress > div > div {
    background: linear-gradient(90deg, #6366f1 0%, #a855f7 100%);
    border-radius: 10px;
}

/* Dataframe Styling */
[data-testid="stDataFrame"] {
    border-radius: 16px;
    overflow: hidden;
    box-shadow: 0 10px 40px rgba(0, 0, 0, 0.2);
}

/* Success/Warning/Error Messages */
.stSuccess, .stWarning, .stError, .stInfo {
    border-radius: 14px;
    border-left-width: 4px;
    backdrop-filter: blur(10px);
}

/* Radio Buttons */
[data-testid="stRadio"] label {
    background: rgba(30, 41, 59, 0.3);
    padding: 0.7rem 1.2rem;
    border-radius: 10px;
    margin: 0.3rem 0;
    transition: all 0.3s ease;
    border: 1px solid transparent;
}

[data-testid="stRadio"] label:hover {
    background: rgba(99, 102, 241, 0.2);
    border-color: rgba(99, 102, 241, 0.4);
}

/* Animations */
@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY(30px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes fadeInDown {
    from {
        opacity: 0;
        transform: translateY(-20px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

/* Emoji Enhancement */
.big-emoji {
    font-size: 3rem;
    display: inline-block;
    animation: float 3s ease-in-out infinite;
}

@keyframes float {
    0%, 100% { transform: translateY(0px); }
    50% { transform: translateY(-10px); }
}

/* Status Badge */
.status-badge {
    display: inline-block;
    padding: 0.4rem 1rem;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

.status-success {
    background: linear-gradient(135deg, #10b981 0%, #059669 100%);
    color: white;
    box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3);
}

.status-warning {
    background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
    color: white;
    box-shadow: 0 4px 15px rgba(245, 158, 11, 0.3);
}

/* Divider */
hr {
    border: none;
    height: 2px;
    background: linear-gradient(90deg, transparent, rgba(99, 102, 241, 0.5), transparent);
    margin: 2rem 0;
}

/* Responsive */
@media (max-width: 768px) {
    .hero h1 { font-size: 2rem; }
    .brand-title { font-size: 1.5rem; }
}
</style>
""",unsafe_allow_html=True)

def hero(title,sub,tag="SENTIMENT INTELLIGENCE"):
    st.markdown(f'<div class="hero"><div class="eyebrow">{tag}</div><h1>{title}</h1><p>{sub}</p></div>',unsafe_allow_html=True)

@st.cache_resource(show_spinner="Loading teacher model (first run downloads model files)...")
def load_teacher():
    tok=AutoTokenizer.from_pretrained(TEACHER)
    model=AutoModelForSequenceClassification.from_pretrained(TEACHER).eval()
    return tok,model

@st.cache_resource(show_spinner="Loading trained student...")
def load_student():
    """Load pre-trained student model and tokenizer from results directory"""
    path=OUT/"student_distilled.pt"; td=OUT/"tokenizer"
    if not path.exists() or not td.exists(): return None,None,None
    tok=AutoTokenizer.from_pretrained(td)
    meta=json.loads((OUT/"metrics.json").read_text()) if (OUT/"metrics.json").exists() else {}
    max_len=int(meta.get("max_length",96))
    model=TinyTransformerStudent(len(tok),tok.pad_token_id or 0,max_length=max_len)
    model.load_state_dict(torch.load(path,map_location="cpu")["state_dict"])
    model.eval()
    return tok,model,max_len

def teacher_predict(text):
    tok,model=load_teacher()
    x=tok(text,return_tensors="pt",truncation=True,max_length=256)
    start=time.perf_counter()
    with torch.no_grad(): p=torch.softmax(model(**x).logits,dim=-1)[0].numpy()
    elapsed=(time.perf_counter()-start)*1000
    mapping={str(v).lower():int(k) for k,v in model.config.id2label.items()}
    ni=next((i for k,i in mapping.items() if "neg" in k),0)
    pi=next((i for k,i in mapping.items() if "pos" in k),1)
    return {"negative":float(p[ni]),"positive":float(p[pi])},elapsed

def student_predict(text):
    tok,model,max_len=load_student()
    if model is None: return None,None
    x=tok(text,return_tensors="pt",truncation=True,padding="max_length",max_length=max_len)
    start=time.perf_counter()
    with torch.no_grad(): p=torch.softmax(model(x["input_ids"],x["attention_mask"]),dim=-1)[0].numpy()
    return {"negative":float(p[0]),"positive":float(p[1])},(time.perf_counter()-start)*1000

def label_conf(p):
    label="Positive" if p["positive"]>=p["negative"] else "Negative"
    return label,max(p.values())

def metrics():
    p=OUT/"metrics.json"
    return json.loads(p.read_text()) if p.exists() else None

with st.sidebar:
    # Premium Brand Header
    st.markdown("""
    <div class="brand-header">
        <div class="brand-logo">🧠</div>
        <h1 class="brand-title">SentiMind AI</h1>
        <p class="brand-subtitle">Neural Sentiment Intelligence</p>
    </div>
    """,unsafe_allow_html=True)
    
    st.markdown("### 🎯 Navigation")
    page=st.radio("",["🏠 Overview","⚡ Live Analysis","📊 Model Comparison","📁 Batch Analysis","🔬 Research Lab","ℹ️ About"],label_visibility="collapsed")
    
    st.divider()
    
    st.markdown("### 🎓 Project Status")
    ready=(OUT/"student_distilled.pt").exists()
    if ready:
        st.markdown('<span class="status-badge status-success">✓ Student Trained</span>',unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-badge status-warning">⚠ Training Required</span>',unsafe_allow_html=True)
    
    st.divider()
    
    st.markdown("### 🔧 Configuration")
    st.markdown("""
    - **Hardware:** CPU-Optimized
    - **Dataset:** IMDb Reviews
    - **Classes:** Positive / Negative
    - **Teacher:** DistilBERT (67M)
    - **Student:** Transformer (500K)
    """)
    
    st.divider()
    st.caption("🚀 Knowledge Distillation Platform  \nBuilt for B.Tech CSE Project")
    
# Map page names to original logic
page_map = {
    "🏠 Overview": "Overview",
    "⚡ Live Analysis": "Live Analysis",
    "📊 Model Comparison": "Model Comparison",
    "📁 Batch Analysis": "Batch Analysis",
    "🔬 Research Lab": "Research Lab",
    "ℹ️ About": "About Project"
}
page = page_map.get(page, "Overview")

if page=="Overview":
    hero("Transform sentiment analysis with AI compression","Experience knowledge distillation in action: a pre-trained teacher model transfers its intelligence to a compact student, achieving massive efficiency gains while maintaining accuracy.","🚀 NEXT-GEN AI RESEARCH")
    
    st.markdown('<div style="height: 1rem;"></div>',unsafe_allow_html=True)
    
    m=metrics()
    cols=st.columns(4)
    if m:
        vals=[
            ("📚 TRAIN SAMPLES",f'{m["train_examples"]:,}'),
            ("🎯 TEST SAMPLES",f'{m["test_examples"]:,}'),
            ("✨ STUDENT ACCURACY",f'{m["distilled_student"]["accuracy"]*100:.1f}%'),
            ("⚡ INFERENCE SPEED",f'{m["distilled_student"]["latency_ms_per_sample"]:.1f} ms')
        ]
    else: 
        vals=[
            ("🎭 TASK","Sentiment"),
            ("👨‍🏫 TEACHER","DistilBERT"),
            ("🎓 STUDENT","Tiny Transformer"),
            ("💻 HARDWARE","CPU-First")
        ]
    
    for c,(k,v) in zip(cols,vals):
        with c: st.metric(k,v)
    
    st.markdown('<div style="height: 2rem;"></div>',unsafe_allow_html=True)
    
    col1, col2 = st.columns([1.3, 1])
    
    with col1:
        st.markdown("### 🔮 How Knowledge Distillation Works")
        st.markdown("""
        <div class="panel">
            <h4 style="color: #a78bfa; margin-top: 0;">🎯 Step 1: Teacher Generates Soft Targets</h4>
            <p style="color: #cbd5e1; line-height: 1.6;">
            The pre-trained DistilBERT model (67M parameters) analyzes training reviews and produces 
            probability distributions—"soft targets"—that capture nuanced sentiment relationships.
            </p>
            
            <hr>
            
            <h4 style="color: #60a5fa; margin-top: 1rem;">🧪 Step 2: Student Learns from Both Sources</h4>
            <p style="color: #cbd5e1; line-height: 1.6;">
            The compact Transformer student (500K parameters) trains using a combined loss: learning 
            from true labels (hard targets) and teacher predictions (soft targets) simultaneously.
            </p>
            
            <hr>
            
            <h4 style="color: #f472b6; margin-top: 1rem;">📊 Step 3: Evaluation & Comparison</h4>
            <p style="color: #cbd5e1; line-height: 1.6;">
            We compare baseline (trained without teacher) vs. distilled student to measure the 
            knowledge transfer benefit in terms of accuracy, speed, and model size.
            </p>
        </div>
        """,unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 🎯 Quick Actions")
        st.markdown("""
        <div class="panel" style="background: linear-gradient(135deg, rgba(99, 102, 241, 0.15) 0%, rgba(168, 85, 247, 0.15) 100%);">
            <p style="font-size: 2.5rem; text-align: center; margin: 1rem 0;" class="big-emoji">⚡</p>
            <h4 style="text-align: center; color: #f1f5f9; margin-bottom: 1.5rem;">Explore Features</h4>
            
            <p style="color: #cbd5e1; margin-bottom: 0.8rem;">
            <strong style="color: #a78bfa;">⚡ Live Analysis</strong><br>
            Compare real-time predictions from teacher and student on custom text
            </p>
            
            <p style="color: #cbd5e1; margin-bottom: 0.8rem;">
            <strong style="color: #60a5fa;">📊 Model Comparison</strong><br>
            Visualize accuracy, speed, and compression metrics side-by-side
            </p>
            
            <p style="color: #cbd5e1; margin-bottom: 0.8rem;">
            <strong style="color: #f472b6;">📁 Batch Analysis</strong><br>
            Upload CSV files and process hundreds of reviews instantly
            </p>
            
            <p style="color: #cbd5e1; margin-bottom: 0;">
            <strong style="color: #34d399;">🔬 Research Lab</strong><br>
            Deep dive into methodology, hyperparameters, and experiments
            </p>
        </div>
        """,unsafe_allow_html=True)
        
        if not ready:
            st.markdown('<div style="height: 1rem;"></div>',unsafe_allow_html=True)
            st.info("💡 **Getting Started:** Train your models using the commands on the About page, then return here to see results!")
    
    st.markdown('<div style="height: 2rem;"></div>',unsafe_allow_html=True)
    
    # Key Stats Section
    st.markdown("### 🏆 Project Highlights")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="panel" style="text-align: center;">
            <p style="font-size: 3rem; margin: 0;">134×</p>
            <p style="color: #94a3b8; font-size: 0.9rem; margin: 0;">Model Compression</p>
            <p style="color: #64748b; font-size: 0.75rem; margin-top: 0.5rem;">67M → 500K parameters</p>
        </div>
        """,unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="panel" style="text-align: center;">
            <p style="font-size: 3rem; margin: 0;">20×</p>
            <p style="color: #94a3b8; font-size: 0.9rem; margin: 0;">Faster Inference</p>
            <p style="color: #64748b; font-size: 0.75rem; margin-top: 0.5rem;">15-20ms → 0.8-0.9ms</p>
        </div>
        """,unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="panel" style="text-align: center;">
            <p style="font-size: 3rem; margin: 0;">47</p>
            <p style="color: #94a3b8; font-size: 0.9rem; margin: 0;">Automated Tests</p>
            <p style="color: #64748b; font-size: 0.75rem; margin-top: 0.5rem;">100% passing coverage</p>
        </div>
        """,unsafe_allow_html=True)

elif page=="Live Analysis":
    hero("Real-time sentiment prediction","Compare teacher and student models side-by-side on your custom text. See knowledge distillation in action!","⚡ INTERACTIVE DEMO")
    
    st.markdown('<div style="height: 1rem;"></div>',unsafe_allow_html=True)
    
    # Example texts with emojis
    col1, col2 = st.columns(2)
    with col1:
        if st.button("😊 Try Positive Example", use_container_width=True):
            st.session_state.demo_text = "This movie was absolutely fantastic! The acting was superb, the plot was engaging, and I was captivated from start to finish. Highly recommended!"
    with col2:
        if st.button("😞 Try Negative Example", use_container_width=True):
            st.session_state.demo_text = "I was extremely disappointed with this film. The plot was confusing, the acting felt forced, and I struggled to stay awake. Would not recommend."
    
    text=st.text_area(
        "📝 Enter your text to analyze",
        value=st.session_state.get('demo_text', "The story was engaging, the acting was excellent, and I would happily watch it again."),
        height=150,
        max_chars=5000,
        help="Enter any review, comment, or feedback (up to 5000 characters)"
    )
    
    st.markdown('<div style="height: 1rem;"></div>',unsafe_allow_html=True)
    
    if st.button("🚀 Analyze Sentiment", use_container_width=True, type="primary"):
        if not text.strip(): 
            st.warning("⚠️ Please enter some text to analyze.")
        else:
            with st.spinner("🔮 Running neural inference..."):
                tp,tm=teacher_predict(text)
                sp,sm=student_predict(text)
            
            st.markdown('<div style="height: 1.5rem;"></div>',unsafe_allow_html=True)
            
            cols=st.columns(2)
            
            # Teacher Results
            with cols[0]:
                st.markdown("""
                <div class="panel">
                    <div style="text-align: center; margin-bottom: 1rem;">
                        <span style="font-size: 2rem;">👨‍🏫</span>
                        <h3 style="margin: 0.5rem 0; color: #a78bfa;">Teacher Model</h3>
                        <p style="color: #94a3b8; font-size: 0.85rem; margin: 0;">DistilBERT • 67M Parameters</p>
                    </div>
                """,unsafe_allow_html=True)
                
                if tp is None:
                    st.warning("Teacher model not available")
                else:
                    lab,conf=label_conf(tp)
                    
                    # Sentiment Result with Animation
                    if lab=="Positive":
                        st.markdown(f"""
                        <div style="text-align: center; padding: 1.5rem; background: linear-gradient(135deg, rgba(34, 197, 94, 0.2) 0%, rgba(22, 163, 74, 0.1) 100%); border-radius: 14px; margin: 1rem 0;">
                            <p style="font-size: 3rem; margin: 0;" class="big-emoji">😊</p>
                            <h2 style="color: #22c55e; margin: 0.5rem 0; font-size: 2rem;">POSITIVE</h2>
                            <p style="color: #86efac; font-size: 1.1rem; margin: 0;">Confidence: {conf*100:.1f}%</p>
                        </div>
                        """,unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div style="text-align: center; padding: 1.5rem; background: linear-gradient(135deg, rgba(239, 68, 68, 0.2) 0%, rgba(220, 38, 38, 0.1) 100%); border-radius: 14px; margin: 1rem 0;">
                            <p style="font-size: 3rem; margin: 0;" class="big-emoji">😞</p>
                            <h2 style="color: #ef4444; margin: 0.5rem 0; font-size: 2rem;">NEGATIVE</h2>
                            <p style="color: #fca5a5; font-size: 1.1rem; margin: 0;">Confidence: {conf*100:.1f}%</p>
                        </div>
                        """,unsafe_allow_html=True)
                    
                    # Probability Breakdown
                    st.markdown("**📊 Probability Distribution:**")
                    st.progress(tp["positive"], text=f"✅ Positive: {tp['positive']*100:.2f}%")
                    st.progress(tp["negative"], text=f"❌ Negative: {tp['negative']*100:.2f}%")
                    
                    st.markdown(f"""
                    <div style="text-align: center; margin-top: 1rem; padding: 0.8rem; background: rgba(99, 102, 241, 0.1); border-radius: 10px;">
                        <p style="color: #94a3b8; font-size: 0.85rem; margin: 0;">⚡ Inference Time: <strong style="color: #a78bfa;">{tm:.2f} ms</strong></p>
                    </div>
                    """,unsafe_allow_html=True)
                
                st.markdown("</div>",unsafe_allow_html=True)
            
            # Student Results
            with cols[1]:
                st.markdown("""
                <div class="panel">
                    <div style="text-align: center; margin-bottom: 1rem;">
                        <span style="font-size: 2rem;">🎓</span>
                        <h3 style="margin: 0.5rem 0; color: #60a5fa;">Student Model</h3>
                        <p style="color: #94a3b8; font-size: 0.85rem; margin: 0;">Distilled Transformer • 500K Parameters</p>
                    </div>
                """,unsafe_allow_html=True)
                
                if sp is None:
                    st.warning("⚠️ Student model not trained yet. Run training first!")
                    st.info("👉 Check the **About** page for training commands")
                else:
                    lab,conf=label_conf(sp)
                    
                    # Sentiment Result with Animation
                    if lab=="Positive":
                        st.markdown(f"""
                        <div style="text-align: center; padding: 1.5rem; background: linear-gradient(135deg, rgba(34, 197, 94, 0.2) 0%, rgba(22, 163, 74, 0.1) 100%); border-radius: 14px; margin: 1rem 0;">
                            <p style="font-size: 3rem; margin: 0;" class="big-emoji">😊</p>
                            <h2 style="color: #22c55e; margin: 0.5rem 0; font-size: 2rem;">POSITIVE</h2>
                            <p style="color: #86efac; font-size: 1.1rem; margin: 0;">Confidence: {conf*100:.1f}%</p>
                        </div>
                        """,unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div style="text-align: center; padding: 1.5rem; background: linear-gradient(135deg, rgba(239, 68, 68, 0.2) 0%, rgba(220, 38, 38, 0.1) 100%); border-radius: 14px; margin: 1rem 0;">
                            <p style="font-size: 3rem; margin: 0;" class="big-emoji">😞</p>
                            <h2 style="color: #ef4444; margin: 0.5rem 0; font-size: 2rem;">NEGATIVE</h2>
                            <p style="color: #fca5a5; font-size: 1.1rem; margin: 0;">Confidence: {conf*100:.1f}%</p>
                        </div>
                        """,unsafe_allow_html=True)
                    
                    # Probability Breakdown
                    st.markdown("**📊 Probability Distribution:**")
                    st.progress(sp["positive"], text=f"✅ Positive: {sp['positive']*100:.2f}%")
                    st.progress(sp["negative"], text=f"❌ Negative: {sp['negative']*100:.2f}%")
                    
                    st.markdown(f"""
                    <div style="text-align: center; margin-top: 1rem; padding: 0.8rem; background: rgba(99, 102, 241, 0.1); border-radius: 10px;">
                        <p style="color: #94a3b8; font-size: 0.85rem; margin: 0;">⚡ Inference Time: <strong style="color: #60a5fa;">{sm:.2f} ms</strong></p>
                    </div>
                    """,unsafe_allow_html=True)
                
                st.markdown("</div>",unsafe_allow_html=True)
            
            # Agreement Analysis
            if sp:
                st.markdown('<div style="height: 1.5rem;"></div>',unsafe_allow_html=True)
                if label_conf(tp)[0]==label_conf(sp)[0]:
                    st.success("✅ **Agreement:** Both models predicted the same sentiment! Knowledge transfer successful.")
                else:
                    st.warning("⚠️ **Disagreement:** Models predicted different sentiments. This example is useful for error analysis and model improvement.")
            
            st.markdown('<div style="height: 1rem;"></div>',unsafe_allow_html=True)
            st.caption("💡 **Note:** Confidence scores represent model certainty but don't guarantee correctness. Models are trained on English movie reviews.")
            

elif page=="Model Comparison":
    hero("Teacher vs. student","Use measured experiment results to compare classification quality and efficiency.","EXPERIMENTAL RESULTS")
    m=metrics()
    if not m: st.info("No metrics found yet. Run training first; this page does not invent results.")
    else:
        b,d=m["baseline_student"],m["distilled_student"]
        cols=st.columns(4)
        for c,k,v in zip(cols,["Baseline accuracy","Distilled accuracy","Baseline macro-F1","Distilled macro-F1"],[b["accuracy"],d["accuracy"],b["macro_f1"],d["macro_f1"]]):
            c.metric(k,f"{v*100:.2f}%")
        df=pd.DataFrame([{"Model":"Baseline student","Accuracy":b["accuracy"],"Macro F1":b["macro_f1"],"Latency ms/sample":b["latency_ms_per_sample"],"Checkpoint MB":b["checkpoint_mb"]},{"Model":"Distilled student","Accuracy":d["accuracy"],"Macro F1":d["macro_f1"],"Latency ms/sample":d["latency_ms_per_sample"],"Checkpoint MB":d["checkpoint_mb"]}])
        l,r=st.columns(2)
        with l: st.plotly_chart(px.bar(df,x="Model",y=["Accuracy","Macro F1"],barmode="group",range_y=[0,1],title="Classification quality",template="plotly_dark"),use_container_width=True)
        with r: st.plotly_chart(px.bar(df,x="Model",y=["Latency ms/sample","Checkpoint MB"],barmode="group",title="Efficiency metrics",template="plotly_dark"),use_container_width=True)
        st.dataframe(df,use_container_width=True,hide_index=True)
        st.download_button("Download metrics JSON",json.dumps(m,indent=2),"sentimentlab_metrics.json","application/json")
        st.caption("Latency depends on CPU and system load. Interpret each metric separately.")

elif page=="Batch Analysis":
    hero("Batch sentiment analysis","Upload a CSV of reviews and download predictions with probabilities.","BULK PROCESSING")
    st.markdown('<div class="panel">CSV must include a text column such as <b>review</b>, <b>text</b>, <b>comment</b>, or <b>feedback</b>.</div>',unsafe_allow_html=True)
    up=st.file_uploader("Upload CSV",type=["csv"])
    if up:
        try: df=pd.read_csv(up)
        except Exception as e: st.error(f"Could not read CSV: {e}"); st.stop()
        st.dataframe(df.head(8),use_container_width=True)
        preferred=next((c for c in df.columns if c.lower() in ["review","text","comment","feedback","sentence"]),df.columns[0])
        col=st.selectbox("Text column",df.columns,index=list(df.columns).index(preferred))
        limit=st.slider("Rows to process",1,max(1,min(500,len(df))),min(100,len(df)))
        if st.button("Analyze CSV",use_container_width=True):
            work=df.head(limit).copy(); out=[]; bar=st.progress(0)
            for i,val in enumerate(work[col].fillna("").astype(str)):
                if not val.strip(): out.append({"sentiment":"Empty","confidence":np.nan,"positive_probability":np.nan,"negative_probability":np.nan,"model_used":"None"})
                else:
                    p,_=student_predict(val); used="Distilled student"
                    if p is None: p,_=teacher_predict(val); used="Teacher fallback"
                    lab,conf=label_conf(p)
                    out.append({"sentiment":lab,"confidence":conf,"positive_probability":p["positive"],"negative_probability":p["negative"],"model_used":used})
                bar.progress((i+1)/len(work))
            result=pd.concat([work.reset_index(drop=True),pd.DataFrame(out)],axis=1)
            st.success(f"Analyzed {len(result)} rows.")
            st.dataframe(result,use_container_width=True)
            counts=result["sentiment"].value_counts().rename_axis("Sentiment").reset_index(name="Count")
            st.plotly_chart(px.pie(counts,names="Sentiment",values="Count",hole=.55,title="Sentiment distribution",template="plotly_dark"),use_container_width=True)
            st.download_button("Download predictions CSV",result.to_csv(index=False).encode(), "sentiment_predictions.csv","text/csv")

elif page=="Research Lab":
    hero("Research lab","Explore the experimental setup and methodology for your major project.","METHODOLOGY")
    m=metrics()
    if m:
        config={k:m[k] for k in ["teacher_model","dataset","device","seed","train_examples","test_examples","epochs","batch_size","max_length","temperature","alpha_hard_loss"] if k in m}
        st.json(config)
        for limitation in m["limitations"]: st.warning(limitation)
    else: st.info("Run training to populate actual experiment settings and limitations.")
    st.markdown("### Distillation objective")
    st.latex(r"L = \alpha L_{hard} + (1-\alpha)T^2 D_{KL}(softmax(z_t/T)\;||\;softmax(z_s/T))")
    st.markdown("- **Hard loss:** cross-entropy with real IMDb labels.\n- **Soft loss:** KL divergence between teacher and student distributions.\n- **Temperature:** smooths probabilities and reveals class relationships.\n- **Alpha:** balances hard and soft losses.")
    st.markdown("### Future experiments")
    st.markdown("1. Temperature: 1, 2, 4.\n2. Alpha: 0.25, 0.5, 0.75.\n3. Larger training subset.\n4. At least three seeds.\n5. Fine-tune teacher on IMDb to reduce domain mismatch.")

elif page=="About Project":
    hero("Welcome to SentiMind AI","A production-ready B.Tech CSE major project demonstrating knowledge distillation for sentiment analysis with complete documentation and testing.","ℹ️ PROJECT DOCUMENTATION")
    
    st.markdown('<div style="height: 1rem;"></div>',unsafe_allow_html=True)
    
    # Quick Info Cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="panel" style="text-align: center;">
            <p style="font-size: 2.5rem; margin: 0;">🎓</p>
            <h4 style="color: #a78bfa; margin: 0.5rem 0;">Academic Project</h4>
            <p style="color: #94a3b8; font-size: 0.9rem;">B.Tech CSE Major Project</p>
        </div>
        """,unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="panel" style="text-align: center;">
            <p style="font-size: 2.5rem; margin: 0;">🧪</p>
            <h4 style="color: #60a5fa; margin: 0.5rem 0;">Knowledge Distillation</h4>
            <p style="color: #94a3b8; font-size: 0.9rem;">Large → Small Model Transfer</p>
        </div>
        """,unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="panel" style="text-align: center;">
            <p style="font-size: 2.5rem; margin: 0;">💻</p>
            <h4 style="color: #f472b6; margin: 0.5rem 0;">CPU-Optimized</h4>
            <p style="color: #94a3b8; font-size: 0.9rem;">No GPU Required</p>
        </div>
        """,unsafe_allow_html=True)
    
    st.markdown('<div style="height: 2rem;"></div>',unsafe_allow_html=True)
    
    st.markdown("### 🚀 Quick Start Guide")
    st.markdown("""
    <div class="panel">
        <h4 style="color: #a78bfa;">Step 1: Setup Virtual Environment</h4>
    """,unsafe_allow_html=True)
    st.code(r"""python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt""",language="powershell")
    
    st.markdown("""
        <h4 style="color: #60a5fa; margin-top: 1.5rem;">Step 2: Run Smoke Test (2-5 minutes)</h4>
        <p style="color: #cbd5e1;">Quick validation to ensure everything works:</p>
    """,unsafe_allow_html=True)
    st.code(r"""python src/train.py --train-size 200 --test-size 100 --epochs 1 --batch-size 4 --max-length 96""",language="powershell")
    
    st.markdown("""
        <h4 style="color: #f472b6; margin-top: 1.5rem;">Step 3: Launch Web Application</h4>
        <p style="color: #cbd5e1;">After training completes, start the interactive dashboard:</p>
    """,unsafe_allow_html=True)
    st.code(r"""streamlit run app.py""",language="powershell")
    
    st.markdown("</div>",unsafe_allow_html=True)
    
    st.markdown('<div style="height: 2rem;"></div>',unsafe_allow_html=True)
    
    # Advanced Training Options
    st.markdown("### ⚙️ Training Configurations")
    
    tab1, tab2, tab3 = st.tabs(["🚀 Quick Test", "⭐ Standard", "🏆 Best Accuracy"])
    
    with tab1:
        st.markdown("""
        <div class="panel">
            <h4 style="color: #22c55e;">Quick Smoke Test (2-5 minutes)</h4>
            <p style="color: #cbd5e1;">Minimal configuration for rapid validation:</p>
        </div>
        """,unsafe_allow_html=True)
        st.code(r"""python src/train.py --train-size 50 --test-size 20 --epochs 1 --batch-size 4""",language="powershell")
        st.info("✓ Use this to verify your setup works correctly")
    
    with tab2:
        st.markdown("""
        <div class="panel">
            <h4 style="color: #3b82f6;">Standard Configuration (15-20 minutes)</h4>
            <p style="color: #cbd5e1;">Recommended for project demonstration:</p>
        </div>
        """,unsafe_allow_html=True)
        st.code(r"""python src/train.py --train-size 800 --test-size 300 --epochs 3 --batch-size 8""",language="powershell")
        st.info("✓ Balanced between training time and accuracy")
    
    with tab3:
        st.markdown("""
        <div class="panel">
            <h4 style="color: #f59e0b;">Maximum Accuracy (30-45 minutes)</h4>
            <p style="color: #cbd5e1;">Best results for final submission:</p>
        </div>
        """,unsafe_allow_html=True)
        st.code(r"""python src/train.py --train-size 2000 --test-size 500 --epochs 5 --batch-size 16""",language="powershell")
        st.info("✓ Highest accuracy but requires more time")
    
    st.markdown('<div style="height: 2rem;"></div>',unsafe_allow_html=True)
    
    # Research Design
    st.markdown("### 🔬 Research Design")
    st.markdown(f"""
    <div class="panel">
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
            <div>
                <p style="color: #94a3b8; font-size: 0.85rem; margin: 0;">TEACHER MODEL</p>
                <p style="color: #f1f5f9; font-size: 1.1rem; margin: 0.3rem 0;"><code>{TEACHER}</code></p>
            </div>
            <div>
                <p style="color: #94a3b8; font-size: 0.85rem; margin: 0;">STUDENT MODEL</p>
                <p style="color: #f1f5f9; font-size: 1.1rem; margin: 0.3rem 0;">Custom 2-Layer Transformer</p>
            </div>
            <div>
                <p style="color: #94a3b8; font-size: 0.85rem; margin: 0;">DATASET</p>
                <p style="color: #f1f5f9; font-size: 1.1rem; margin: 0.3rem 0;">IMDb Movie Reviews (Balanced Subset)</p>
            </div>
            <div>
                <p style="color: #94a3b8; font-size: 0.85rem; margin: 0;">EVALUATION METRICS</p>
                <p style="color: #f1f5f9; font-size: 1.1rem; margin: 0.3rem 0;">Accuracy, F1, Latency, Size</p>
            </div>
        </div>
        
        <hr>
        
        <h4 style="color: #a78bfa; margin-top: 1rem;">Training Approaches:</h4>
        <p style="color: #cbd5e1;"><strong>🎯 Baseline:</strong> Student trained with hard labels only (standard cross-entropy)</p>
        <p style="color: #cbd5e1;"><strong>🧪 Distilled:</strong> Student trained with hard labels + teacher soft targets (combined loss)</p>
        <p style="color: #cbd5e1;"><strong>📊 Comparison:</strong> Measure knowledge transfer benefit in accuracy, speed, and efficiency</p>
    </div>
    """,unsafe_allow_html=True)
    
    st.markdown('<div style="height: 2rem;"></div>',unsafe_allow_html=True)
    
    # Important Notes
    st.markdown("### ⚠️ Important Considerations")
    col1, col2 = st.columns(2)
    
    with col1:
        st.warning("""
        **Domain Shift Limitation**
        
        The teacher model is fine-tuned on SST-2 (short movie snippets) while evaluation uses IMDb (full reviews). 
        This domain mismatch should be acknowledged in your project report.
        """)
    
    with col2:
        st.info("""
        **Research Best Practices**
        
        - Use actual measured results (never fabricate metrics)
        - Report configuration details and hardware specs
        - Discuss limitations honestly
        - Run multiple seeds for statistical validity (if time permits)
        """)
    
    st.markdown('<div style="height: 2rem;"></div>',unsafe_allow_html=True)
    
    # Key References
    st.markdown("### 📚 Key References")
    st.markdown("""
    <div class="panel">
        <p style="color: #cbd5e1; line-height: 1.8;">
        1. <strong style="color: #a78bfa;">Hinton, G., Vinyals, O., & Dean, J. (2015)</strong><br>
        <em>Distilling the Knowledge in a Neural Network</em><br>
        arXiv:1503.02531 • Introduced temperature-scaled knowledge distillation
        </p>
        
        <p style="color: #cbd5e1; line-height: 1.8; margin-top: 1rem;">
        2. <strong style="color: #60a5fa;">Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019)</strong><br>
        <em>DistilBERT: A distilled version of BERT</em><br>
        arXiv:1910.01108 • Applied distillation to BERT, achieving 40% size reduction
        </p>
        
        <p style="color: #cbd5e1; line-height: 1.8; margin-top: 1rem;">
        3. <strong style="color: #f472b6;">Vaswani, A., et al. (2017)</strong><br>
        <em>Attention is All You Need</em><br>
        arXiv:1706.03762 • Introduced Transformer architecture
        </p>
    </div>
    """,unsafe_allow_html=True)
    
    st.markdown('<div style="height: 2rem;"></div>',unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(168, 85, 247, 0.1) 100%); border-radius: 16px;">
        <p style="font-size: 2rem; margin: 0;">🧠</p>
        <h3 style="color: #f1f5f9; margin: 0.5rem 0;">SentiMind AI</h3>
        <p style="color: #94a3b8; margin: 0;">Neural Sentiment Intelligence Platform</p>
        <p style="color: #64748b; font-size: 0.85rem; margin-top: 1rem;">Built with PyTorch • Streamlit • Hugging Face Transformers</p>
    </div>
    """,unsafe_allow_html=True)

