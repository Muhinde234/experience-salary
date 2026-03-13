import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ARTIFACT_DIR = Path("artifacts")
LATEST_PATH = ARTIFACT_DIR / "latest_model.json"
DEFAULT_MODEL_PATH = ARTIFACT_DIR / "salary_model.joblib"
DEFAULT_METADATA_PATH = ARTIFACT_DIR / "model_metadata.json"


@st.cache_resource
def load_model_and_metadata() -> tuple[object, dict]:
    model_path = DEFAULT_MODEL_PATH
    metadata_path = DEFAULT_METADATA_PATH

    if LATEST_PATH.exists():
        latest = json.loads(LATEST_PATH.read_text(encoding="utf-8"))
        model_path = Path(latest["model_path"])
        metadata_path = Path(latest["metadata_path"])

    if not model_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            "Model artifacts not found. Run training first: python scripts/train.py"
        )

    model = joblib.load(model_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return model, metadata


# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Salary Predictor · AIMS",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ---- Global ---- */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(145deg, #0d1117 0%, #161b27 60%, #1a1f35 100%);
    color: #e6edf3;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #161b27 0%, #0d1117 100%);
    border-right: 1px solid #30363d;
}
/* ---- Hero banner ---- */
.hero {
    background: linear-gradient(135deg, #1f6feb 0%, #8957e5 50%, #bc8cff 100%);
    border-radius: 16px;
    padding: 2.4rem 2.8rem;
    margin-bottom: 2rem;
    box-shadow: 0 8px 32px rgba(31,111,235,0.35);
}
.hero h1 { font-size: 2.4rem; font-weight: 800; color: #ffffff; margin: 0 0 0.4rem; }
.hero p  { font-size: 1.05rem; color: rgba(255,255,255,0.82); margin: 0; }
/* ---- Cards ---- */
.card {
    background: #161b27;
    border: 1px solid #30363d;
    border-radius: 14px;
    padding: 1.6rem 1.8rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 4px 18px rgba(0,0,0,0.4);
}
.metric-card {
    background: linear-gradient(135deg, #1f2d45 0%, #1a2236 100%);
    border: 1px solid #1f6feb55;
    border-radius: 14px;
    padding: 1.5rem 1.8rem;
    text-align: center;
    box-shadow: 0 4px 20px rgba(31,111,235,0.2);
}
.metric-card .label {
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #8b949e;
    margin-bottom: 0.5rem;
}
.metric-card .value {
    font-size: 2.1rem;
    font-weight: 800;
    color: #58a6ff;
    line-height: 1;
}
.metric-card .sub {
    font-size: 0.88rem;
    color: #8b949e;
    margin-top: 0.3rem;
}
/* Result spotlight */
.result-card {
    background: linear-gradient(135deg, #1a2d1a 0%, #162816 100%);
    border: 1px solid #3fb95055;
    border-radius: 16px;
    padding: 2rem 2.2rem;
    text-align: center;
    box-shadow: 0 6px 30px rgba(63,185,80,0.2);
    margin-top: 1.2rem;
}
.result-card .result-label {
    font-size: 0.8rem; font-weight: 700; letter-spacing: 0.12em;
    text-transform: uppercase; color: #3fb950; margin-bottom: 0.6rem;
}
.result-card .result-value {
    font-size: 3rem; font-weight: 900; color: #56d364; line-height: 1;
}
.result-card .result-sub {
    font-size: 1rem; color: #8b949e; margin-top: 0.5rem;
}
/* Badge */
.badge {
    display: inline-block;
    background: linear-gradient(90deg,#1f6feb,#8957e5);
    color: #fff;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    padding: 0.22rem 0.7rem;
    border-radius: 20px;
    margin-left: 0.5rem;
    vertical-align: middle;
}
/* Leaderboard table */
.lb-row {
    display: flex; align-items: center;
    padding: 0.7rem 1rem;
    border-radius: 10px;
    margin-bottom: 0.5rem;
    background: #1c2030;
    border: 1px solid #30363d;
}
.lb-row.winner { border-color: #3fb95066; background: #192819; }
.lb-name  { flex: 2; font-weight: 600; font-size: 0.93rem; color: #e6edf3; }
.lb-val   { flex: 1; text-align: right; font-size: 0.88rem; color: #8b949e; }
.lb-val strong { color: #58a6ff; }
/* Sidebar labels */
.sidebar-section {
    font-size: 0.72rem; font-weight: 700; letter-spacing: 0.1em;
    text-transform: uppercase; color: #8b949e; margin: 1.2rem 0 0.5rem;
}
</style>
""", unsafe_allow_html=True)

# ── Load artifacts ─────────────────────────────────────────────────────────────
try:
    model, metadata = load_model_and_metadata()
except Exception as exc:
    st.error(str(exc))
    st.stop()

lb       = metadata["leaderboard"]
best     = metadata["best_model"]
version  = metadata["model_version"]
trained  = metadata["trained_at_utc"][:10]
cr       = metadata["cleaning_report"]

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-section">Model Info</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="card" style="padding:1rem 1.2rem">
        <div style="font-size:0.8rem;color:#8b949e">Version</div>
        <div style="font-size:1.3rem;font-weight:800;color:#58a6ff">{version}</div>
        <div style="font-size:0.8rem;color:#8b949e;margin-top:.6rem">Algorithm</div>
        <div style="font-size:1rem;font-weight:700;color:#e6edf3">{best.replace('_',' ').title()}</div>
        <div style="font-size:0.8rem;color:#8b949e;margin-top:.6rem">Trained</div>
        <div style="font-size:0.9rem;color:#e6edf3">{trained}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">Model Equation</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card" style="padding:1rem 1.2rem">
        <div style="font-size:0.82rem;color:#8b949e;margin-bottom:.4rem">In thousands:</div>
        <div style="font-family:monospace;font-size:0.95rem;color:#bc8cff">
            ŷ = 4.937 + 0.831 · x
        </div>
        <div style="font-size:0.78rem;color:#8b949e;margin-top:.6rem">
            x = experience (months)<br>
            ŷ = predicted salary (× 1 000)
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">Data Cleaning</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="card" style="padding:1rem 1.2rem;font-size:0.85rem">
        <div style="display:flex;justify-content:space-between;margin-bottom:.3rem">
            <span style="color:#8b949e">Original rows</span>
            <span style="color:#e6edf3;font-weight:700">{cr['original_rows']:,}</span>
        </div>
        <div style="display:flex;justify-content:space-between;margin-bottom:.3rem">
            <span style="color:#8b949e">After cleaning</span>
            <span style="color:#3fb950;font-weight:700">{cr['cleaned_rows']:,}</span>
        </div>
        <div style="display:flex;justify-content:space-between">
            <span style="color:#8b949e">Removed</span>
            <span style="color:#f85149;font-weight:700">{cr['rows_removed']}</span>
        </div>
        <div style="font-size:0.75rem;color:#8b949e;margin-top:.5rem">{cr['rule']}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">Test Performance</div>', unsafe_allow_html=True)
    best_m = lb[best]
    st.markdown(f"""
    <div class="card" style="padding:1rem 1.2rem;font-size:0.85rem">
        <div style="display:flex;justify-content:space-between;margin-bottom:.3rem">
            <span style="color:#8b949e">RMSE</span>
            <span style="color:#58a6ff;font-weight:700">{best_m['rmse']:.3f}k</span>
        </div>
        <div style="display:flex;justify-content:space-between;margin-bottom:.3rem">
            <span style="color:#8b949e">MAE</span>
            <span style="color:#58a6ff;font-weight:700">{best_m['mae']:.3f}k</span>
        </div>
        <div style="display:flex;justify-content:space-between">
            <span style="color:#8b949e">R²</span>
            <span style="color:#3fb950;font-weight:700">{best_m['r2']:.4f}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero">
    <h1>💼 Experience → Salary Predictor</h1>
    <p>Production ML · {version} &nbsp;·&nbsp; {best.replace('_',' ').title()}
       &nbsp;·&nbsp; Trained {trained}</p>
</div>
""", unsafe_allow_html=True)

# ── Input + Live Equation ──────────────────────────────────────────────────────
left, right = st.columns([1.6, 1], gap="large")

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### Enter Experience")
    months = st.slider(
        "Experience (months)", min_value=0.0, max_value=120.0,
        value=24.0, step=0.5, label_visibility="collapsed"
    )
    years = months / 12
    c1, c2 = st.columns(2)
    c1.metric("Months", f"{months:.1f}")
    c2.metric("Years", f"{years:.2f}")
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    # Live prediction (always visible, no button needed)
    features = pd.DataFrame(
        [{"experience_months": float(months)}], columns=["experience_months"]
    )
    predicted_salary_thousands = float(model.predict(features)[0])
    predicted_salary = predicted_salary_thousands * 1000

    st.markdown(f"""
    <div class="result-card">
        <div class="result-label">Predicted Salary</div>
        <div class="result-value">{predicted_salary_thousands:,.2f}k</div>
        <div class="result-sub">≈ {predicted_salary:,.0f} in full units</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="card" style="margin-top:1rem;padding:1rem 1.2rem;text-align:center">
        <div style="font-size:0.78rem;color:#8b949e;margin-bottom:.3rem">Live Equation</div>
        <div style="font-family:monospace;font-size:1rem;color:#bc8cff">
            ŷ = 4.937 + 0.831 × {months:.1f}
        </div>
        <div style="font-family:monospace;font-size:1.1rem;font-weight:800;color:#58a6ff;margin-top:.3rem">
            = {predicted_salary_thousands:.3f}k
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Regression Chart ───────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("#### Salary vs Experience — Regression Line")

x_range = np.linspace(0, 120, 300)
feat_range = pd.DataFrame({"experience_months": x_range})
y_range = model.predict(feat_range)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=x_range, y=y_range,
    mode="lines",
    name="Fitted line",
    line=dict(color="#58a6ff", width=2.5),
))
fig.add_trace(go.Scatter(
    x=[months], y=[predicted_salary_thousands],
    mode="markers+text",
    name="Your prediction",
    marker=dict(size=14, color="#3fb950", symbol="circle",
                line=dict(color="#56d364", width=2)),
    text=[f"  {predicted_salary_thousands:.2f}k"],
    textposition="middle right",
    textfont=dict(color="#56d364", size=13, family="monospace"),
))
fig.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(22,27,39,0.9)",
    font=dict(color="#8b949e", family="Inter, sans-serif"),
    xaxis=dict(
        title="Experience (months)", gridcolor="#21262d",
        zerolinecolor="#30363d", color="#8b949e",
    ),
    yaxis=dict(
        title="Predicted Salary (thousands)", gridcolor="#21262d",
        zerolinecolor="#30363d", color="#8b949e",
    ),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#e6edf3")),
    margin=dict(l=10, r=10, t=20, b=10),
    height=360,
)
st.plotly_chart(fig, use_container_width=True)

# ── Leaderboard ────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("#### Model Leaderboard")

lb_cols = st.columns(len(lb))
model_display = {"linear_regression": "Linear Regression", "random_forest": "Random Forest"}
icons = {"linear_regression": "📐", "random_forest": "🌲"}

for col, (mname, mval) in zip(lb_cols, lb.items()):
    is_best = mname == best
    badge = " 🏆 Best" if is_best else ""
    border_color = "#3fb95066" if is_best else "#30363d"
    bg = "linear-gradient(135deg,#192819,#162816)" if is_best else "linear-gradient(135deg,#1c2030,#161b27)"
    col.markdown(f"""
    <div style="background:{bg};border:1px solid {border_color};border-radius:14px;
                padding:1.3rem 1.4rem;text-align:center;box-shadow:0 4px 18px rgba(0,0,0,.35)">
        <div style="font-size:1.6rem;margin-bottom:.3rem">{icons.get(mname,'🤖')}</div>
        <div style="font-size:0.92rem;font-weight:700;color:#e6edf3">
            {model_display.get(mname, mname)}{badge}
        </div>
        <div style="margin-top:.8rem;font-size:0.8rem;color:#8b949e">RMSE</div>
        <div style="font-size:1.35rem;font-weight:800;color:#58a6ff">{mval['rmse']:.3f}k</div>
        <div style="margin-top:.5rem;font-size:0.8rem;color:#8b949e">MAE</div>
        <div style="font-size:1.1rem;font-weight:700;color:#58a6ff">{mval['mae']:.3f}k</div>
        <div style="margin-top:.5rem;font-size:0.8rem;color:#8b949e">R²</div>
        <div style="font-size:1.1rem;font-weight:700;color:#3fb950">{mval['r2']:.4f}</div>
    </div>
    """, unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center;font-size:0.78rem;color:#484f58;padding:1rem 0 .5rem">
    Experience-Salary Predictor · AIMS Project · Linear Regression Pipeline
</div>
""", unsafe_allow_html=True)

