"""Live Sentiment Analysis Page"""

from pathlib import Path
import json
import sys
import time
import streamlit as st
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
sys.path.insert(0, str(ROOT / "src"))

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from model import TinyTransformerStudent
except ImportError:
    st.error("Missing dependencies. Install requirements.txt first.")
    st.stop()

TEACHER = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"

st.set_page_config(page_title="Live Analysis - SentimentLab AI", page_icon="🎯", layout="wide")

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
.panel{background:#111b30;border:1px solid #2a3b5a;padding:1.1rem;border-radius:17px;margin-bottom:1rem}
div.stButton>button{border-radius:11px;background:linear-gradient(135deg,#65e6c4,#81c9ff);color:#07121f;font-weight:700;border:0;min-height:2.7rem}
.stTextArea textarea{background:#0d1729!important;color:#f4f7ff!important;border:1px solid #2a3b5a!important;border-radius:12px!important}
</style>
""", unsafe_allow_html=True)

def hero(title, sub, tag):
    st.markdown(
        f'<div class="hero"><div class="eyebrow">{tag}</div><h1>{title}</h1><p>{sub}</p></div>',
        unsafe_allow_html=True
    )

@st.cache_resource(show_spinner="Loading teacher model (first run downloads model files)...")
def load_teacher():
    tok = AutoTokenizer.from_pretrained(TEACHER)
    model = AutoModelForSequenceClassification.from_pretrained(TEACHER).eval()
    return tok, model

@st.cache_resource(show_spinner="Loading trained student...")
def load_student():
    path = OUT / "student_distilled.pt"
    td = OUT / "tokenizer"
    if not path.exists() or not td.exists():
        return None, None, None
    
    tok = AutoTokenizer.from_pretrained(td)
    meta_path = OUT / "metrics.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    max_len = int(meta.get("max_length", 96))
    
    model = TinyTransformerStudent(len(tok), tok.pad_token_id or 0, max_length=max_len)
    model.load_state_dict(torch.load(path, map_location="cpu")["state_dict"])
    model.eval()
    return tok, model, max_len

def teacher_predict(text):
    tok, model = load_teacher()
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=256)
    
    start = time.perf_counter()
    with torch.no_grad():
        probs = torch.softmax(model(**inputs).logits, dim=-1)[0].numpy()
    elapsed = (time.perf_counter() - start) * 1000
    
    # Map model output to negative/positive
    mapping = {str(v).lower(): int(k) for k, v in model.config.id2label.items()}
    neg_idx = next((i for k, i in mapping.items() if "neg" in k), 0)
    pos_idx = next((i for k, i in mapping.items() if "pos" in k), 1)
    
    return {"negative": float(probs[neg_idx]), "positive": float(probs[pos_idx])}, elapsed

def student_predict(text):
    tok, model, max_len = load_student()
    if model is None:
        return None, None
    
    inputs = tok(text, return_tensors="pt", truncation=True, padding="max_length", max_length=max_len)
    
    start = time.perf_counter()
    with torch.no_grad():
        probs = torch.softmax(model(inputs["input_ids"], inputs["attention_mask"]), dim=-1)[0].numpy()
    elapsed = (time.perf_counter() - start) * 1000
    
    return {"negative": float(probs[0]), "positive": float(probs[1])}, elapsed

def label_conf(probs):
    label = "Positive" if probs["positive"] >= probs["negative"] else "Negative"
    return label, max(probs.values())

# Hero
hero(
    "Live sentiment analysis",
    "Enter a review, comment, or feedback to compare teacher and student predictions.",
    "INTERACTIVE PREDICTION"
)

# Example texts
examples = [
    "The story was engaging, the acting was excellent, and I would happily watch it again.",
    "This movie was a complete waste of time. Boring plot and terrible dialogue.",
    "The cinematography was beautiful but the pacing was too slow for my taste.",
    "Not the best film I've seen, but definitely not the worst either.",
]

with st.expander("📝 Click to use example sentences"):
    for i, ex in enumerate(examples):
        if st.button(f"Example {i+1}", key=f"ex_{i}", use_container_width=True):
            st.session_state["analysis_text"] = ex

# Text input
text = st.text_area(
    "TEXT TO ANALYZE",
    value=st.session_state.get("analysis_text", examples[0]),
    height=150,
    max_chars=5000,
    help="Enter any text up to 5000 characters"
)

# Analyze button
if st.button("🚀 Analyze Sentiment", use_container_width=True):
    if not text.strip():
        st.warning("⚠️ Please enter some text to analyze.")
    else:
        with st.spinner("Running CPU inference..."):
            teacher_probs, teacher_ms = teacher_predict(text)
            student_probs, student_ms = student_predict(text)
        
        # Display results side by side
        col1, col2 = st.columns(2)
        
        models = [
            (col1, "Teacher · DistilBERT", teacher_probs, teacher_ms),
            (col2, "Student · Distilled Transformer", student_probs, student_ms)
        ]
        
        for col, name, probs, ms in models:
            with col:
                st.markdown(f'<div class="panel"><h3>{name}</h3>', unsafe_allow_html=True)
                
                if probs is None:
                    st.warning("Student not trained. Run the training command on About Project page.")
                else:
                    label, conf = label_conf(probs)
                    emoji = "🟢" if label == "Positive" else "🔴"
                    
                    st.subheader(f"{emoji} {label}")
                    st.progress(conf, text=f"Predicted-class confidence: {conf*100:.1f}%")
                    st.write(f'Positive: **{probs["positive"]*100:.2f}%**')
                    st.write(f'Negative: **{probs["negative"]*100:.2f}%**')
                    st.caption(f"Single-request inference: {ms:.2f} ms (may vary)")
                
                st.markdown("</div>", unsafe_allow_html=True)
        
        # Agreement check
        if student_probs:
            teacher_label, _ = label_conf(teacher_probs)
            student_label, _ = label_conf(student_probs)
            
            if teacher_label == student_label:
                st.success("✅ Teacher and student agree on the sentiment.")
            else:
                st.warning("⚠️ Teacher and student disagree — interesting case for error analysis!")

st.info("💡 **Note:** Confidence is not a guarantee of correctness. This model is trained for English sentiment analysis on movie reviews.")

# Text statistics
st.markdown("---")
col1, col2, col3 = st.columns(3)
col1.metric("Characters", len(text))
col2.metric("Words", len(text.split()))
col3.metric("Lines", text.count('\n') + 1)
