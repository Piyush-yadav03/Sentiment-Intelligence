"""Batch Sentiment Analysis Page"""

from pathlib import Path
import sys
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
sys.path.insert(0, str(ROOT / "src"))

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from model import TinyTransformerStudent
    import json
except ImportError:
    st.error("Missing dependencies. Install requirements.txt first.")
    st.stop()

TEACHER = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"

st.set_page_config(page_title="Batch Analysis - SentimentLab AI", page_icon="📁", layout="wide")

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
div.stButton>button,div.stDownloadButton>button{border-radius:11px;background:linear-gradient(135deg,#65e6c4,#81c9ff);color:#07121f;font-weight:700;border:0;min-height:2.7rem}
</style>
""", unsafe_allow_html=True)

def hero(title, sub, tag):
    st.markdown(
        f'<div class="hero"><div class="eyebrow">{tag}</div><h1>{title}</h1><p>{sub}</p></div>',
        unsafe_allow_html=True
    )

@st.cache_resource
def load_teacher():
    tok = AutoTokenizer.from_pretrained(TEACHER)
    model = AutoModelForSequenceClassification.from_pretrained(TEACHER).eval()
    return tok, model

@st.cache_resource
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

def student_predict(text):
    tok, model, max_len = load_student()
    if model is None:
        return None
    
    inputs = tok(text, return_tensors="pt", truncation=True, padding="max_length", max_length=max_len)
    with torch.no_grad():
        probs = torch.softmax(model(inputs["input_ids"], inputs["attention_mask"]), dim=-1)[0].numpy()
    return {"negative": float(probs[0]), "positive": float(probs[1])}

def teacher_predict(text):
    tok, model = load_teacher()
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        probs = torch.softmax(model(**inputs).logits, dim=-1)[0].numpy()
    
    mapping = {str(v).lower(): int(k) for k, v in model.config.id2label.items()}
    neg_idx = next((i for k, i in mapping.items() if "neg" in k), 0)
    pos_idx = next((i for k, i in mapping.items() if "pos" in k), 1)
    
    return {"negative": float(probs[neg_idx]), "positive": float(probs[pos_idx])}

def label_conf(probs):
    label = "Positive" if probs["positive"] >= probs["negative"] else "Negative"
    return label, max(probs.values())

# Hero
hero(
    "Batch sentiment analysis",
    "Upload a CSV of reviews and download predictions with probabilities.",
    "BULK PROCESSING"
)

st.markdown("""
<div class="panel">
📋 <b>CSV Requirements:</b> Your file must include a text column such as <b>review</b>, <b>text</b>, 
<b>comment</b>, <b>feedback</b>, or <b>sentence</b>. The analysis will preserve all original columns.
</div>
""", unsafe_allow_html=True)

# File uploader
uploaded_file = st.file_uploader(
    "📤 Upload CSV File",
    type=["csv"],
    help="Upload a CSV file containing text data to analyze"
)

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ Successfully loaded CSV with {len(df)} rows and {len(df.columns)} columns.")
        
        # Preview
        with st.expander("👀 Preview uploaded data", expanded=True):
            st.dataframe(df.head(10), use_container_width=True)
        
        # Column selection
        preferred = next((c for c in df.columns if c.lower() in ["review", "text", "comment", "feedback", "sentence"]), df.columns[0])
        text_column = st.selectbox(
            "📝 Select text column to analyze",
            df.columns,
            index=list(df.columns).index(preferred)
        )
        
        # Processing limit
        max_rows = min(1000, len(df))
        num_rows = st.slider(
            "🔢 Number of rows to process",
            min_value=1,
            max_value=max_rows,
            value=min(100, max_rows),
            help="Processing many rows may take time on CPU"
        )
        
        # Analyze button
        if st.button("🚀 Analyze CSV", use_container_width=True):
            work_df = df.head(num_rows).copy()
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, text_value in enumerate(work_df[text_column].fillna("").astype(str)):
                status_text.text(f"Processing row {i+1}/{len(work_df)}...")
                
                if not text_value.strip():
                    results.append({
                        "sentiment": "Empty",
                        "confidence": np.nan,
                        "positive_probability": np.nan,
                        "negative_probability": np.nan,
                        "model_used": "None"
                    })
                else:
                    # Try student first, fall back to teacher
                    probs = student_predict(text_value)
                    model_used = "Distilled Student"
                    
                    if probs is None:
                        probs = teacher_predict(text_value)
                        model_used = "Teacher Fallback"
                    
                    label, confidence = label_conf(probs)
                    results.append({
                        "sentiment": label,
                        "confidence": confidence,
                        "positive_probability": probs["positive"],
                        "negative_probability": probs["negative"],
                        "model_used": model_used
                    })
                
                progress_bar.progress((i + 1) / len(work_df))
            
            status_text.empty()
            progress_bar.empty()
            
            # Combine results
            result_df = pd.concat([
                work_df.reset_index(drop=True),
                pd.DataFrame(results)
            ], axis=1)
            
            st.success(f"✅ Analyzed {len(result_df)} rows successfully!")
            
            # Display results
            st.markdown("### 📊 Results")
            st.dataframe(result_df, use_container_width=True)
            
            # Sentiment distribution
            sentiment_counts = result_df["sentiment"].value_counts().rename_axis("Sentiment").reset_index(name="Count")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.pie(
                    sentiment_counts,
                    names="Sentiment",
                    values="Count",
                    hole=0.55,
                    title="Sentiment Distribution",
                    template="plotly_dark",
                    color_discrete_sequence=["#65e6c4", "#ff6b9d", "#aebcda"]
                )
                fig.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="DM Sans", size=12, color="#aebcda"),
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("#### Summary Statistics")
                total = len(result_df[result_df["sentiment"] != "Empty"])
                if total > 0:
                    pos_count = len(result_df[result_df["sentiment"] == "Positive"])
                    neg_count = len(result_df[result_df["sentiment"] == "Negative"])
                    
                    st.metric("Total Analyzed", total)
                    st.metric("Positive", f"{pos_count} ({pos_count/total*100:.1f}%)")
                    st.metric("Negative", f"{neg_count} ({neg_count/total*100:.1f}%)")
            
            # Download button
            st.markdown("---")
            csv_data = result_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download Predictions CSV",
                csv_data,
                "sentiment_predictions.csv",
                "text/csv",
                use_container_width=True
            )
    
    except Exception as e:
        st.error(f"❌ Error processing CSV: {e}")
        st.info("Make sure your CSV is properly formatted and contains valid text data.")

else:
    st.info("👆 Upload a CSV file to get started with batch sentiment analysis.")
    
    # Show example CSV format
    st.markdown("### 📝 Example CSV Format")
    example_df = pd.DataFrame({
        "review_id": [1, 2, 3],
        "text": [
            "This product exceeded my expectations!",
            "Disappointed with the quality.",
            "Average product, nothing special."
        ],
        "date": ["2024-01-01", "2024-01-02", "2024-01-03"]
    })
    st.dataframe(example_df, use_container_width=True, hide_index=True)
