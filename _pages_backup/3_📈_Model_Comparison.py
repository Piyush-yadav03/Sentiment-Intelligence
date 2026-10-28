"""Model Comparison Page"""

from pathlib import Path
import json
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"

st.set_page_config(page_title="Model Comparison - SentimentLab AI", page_icon="📈", layout="wide")

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
    "Teacher vs. student",
    "Use measured experiment results to compare classification quality and efficiency.",
    "EXPERIMENTAL RESULTS"
)

metrics = load_metrics()

if not metrics:
    st.info("📊 No metrics found yet. Run training first; this page does not invent results.")
    st.code("python src/train.py --train-size 800 --test-size 300 --epochs 3", language="powershell")
else:
    baseline = metrics["baseline_student"]
    distilled = metrics["distilled_student"]
    
    # Key metrics overview
    cols = st.columns(4)
    metrics_to_show = [
        ("Baseline Accuracy", f'{baseline["accuracy"]*100:.2f}%'),
        ("Distilled Accuracy", f'{distilled["accuracy"]*100:.2f}%'),
        ("Baseline Macro-F1", f'{baseline["macro_f1"]*100:.2f}%'),
        ("Distilled Macro-F1", f'{distilled["macro_f1"]*100:.2f}%'),
    ]
    
    for col, (label, value) in zip(cols, metrics_to_show):
        col.metric(label, value)
    
    st.markdown("---")
    
    # Prepare comparison dataframe
    df = pd.DataFrame([
        {
            "Model": "Baseline Student",
            "Accuracy": baseline["accuracy"],
            "Macro F1": baseline["macro_f1"],
            "Latency (ms/sample)": baseline["latency_ms_per_sample"],
            "Checkpoint (MB)": baseline["checkpoint_mb"],
        },
        {
            "Model": "Distilled Student",
            "Accuracy": distilled["accuracy"],
            "Macro F1": distilled["macro_f1"],
            "Latency (ms/sample)": distilled["latency_ms_per_sample"],
            "Checkpoint (MB)": distilled["checkpoint_mb"],
        }
    ])
    
    # Quality comparison
    col1, col2 = st.columns(2)
    
    with col1:
        fig_quality = px.bar(
            df,
            x="Model",
            y=["Accuracy", "Macro F1"],
            barmode="group",
            title="Classification Quality",
            range_y=[0, 1],
            template="plotly_dark",
            color_discrete_sequence=["#65e6c4", "#81c9ff"]
        )
        fig_quality.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="DM Sans", size=12, color="#aebcda"),
            title_font=dict(size=16, color="#f4f7ff"),
        )
        st.plotly_chart(fig_quality, use_container_width=True)
    
    with col2:
        fig_efficiency = px.bar(
            df,
            x="Model",
            y=["Latency (ms/sample)", "Checkpoint (MB)"],
            barmode="group",
            title="Efficiency Metrics",
            template="plotly_dark",
            color_discrete_sequence=["#c77dff", "#ff6b9d"]
        )
        fig_efficiency.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="DM Sans", size=12, color="#aebcda"),
            title_font=dict(size=16, color="#f4f7ff"),
        )
        st.plotly_chart(fig_efficiency, use_container_width=True)
    
    # Detailed comparison table
    st.markdown("### 📋 Detailed Metrics")
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Performance difference
    st.markdown("### 🔄 Performance Delta")
    col1, col2, col3 = st.columns(3)
    
    accuracy_diff = (distilled["accuracy"] - baseline["accuracy"]) * 100
    f1_diff = (distilled["macro_f1"] - baseline["macro_f1"]) * 100
    latency_diff = distilled["latency_ms_per_sample"] - baseline["latency_ms_per_sample"]
    
    col1.metric(
        "Accuracy Difference",
        f"{accuracy_diff:+.2f}%",
        delta=f"{accuracy_diff:+.2f}%",
        delta_color="normal"
    )
    col2.metric(
        "Macro-F1 Difference",
        f"{f1_diff:+.2f}%",
        delta=f"{f1_diff:+.2f}%",
        delta_color="normal"
    )
    col3.metric(
        "Latency Difference",
        f"{latency_diff:+.2f} ms",
        delta=f"{latency_diff:+.2f} ms",
        delta_color="inverse"
    )
    
    # Download button
    st.markdown("---")
    st.download_button(
        "📥 Download Metrics JSON",
        json.dumps(metrics, indent=2),
        "sentimentlab_metrics.json",
        "application/json",
        use_container_width=True
    )
    
    st.caption("""
    📝 **Note:** Latency depends on CPU and system load. Interpret each metric separately. 
    These results are from a subset of IMDb reviews and should not be generalized without further validation.
    """)
