import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
import os, pickle, datetime

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Isolation Forest · Anomaly Detection",
                   page_icon="🛡️", layout="wide",
                   initial_sidebar_state="expanded")

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Syne:wght@400;600;700;800&display=swap');

html, body, [class*="css"] { font-family:'Syne',sans-serif; background:#07090f; color:#dde1f0; }
.main .block-container { background:#07090f; padding:2rem 3rem; max-width:1400px; }
[data-testid="stSidebar"] { background:#0b0d18 !important; border-right:1px solid #1a1f35; }
[data-testid="stSidebar"] .block-container { padding:1.4rem 1rem; }

.hero {
    background:linear-gradient(135deg,#090b18 0%,#0d1025 60%,#07090f 100%);
    border:1px solid #1f1a40; border-radius:14px;
    padding:2rem 2.5rem; margin-bottom:1.8rem; position:relative; overflow:hidden;
}
.hero::before {
    content:''; position:absolute; top:-60px; right:-60px;
    width:260px; height:260px;
    background:radial-gradient(circle,rgba(239,68,68,.08) 0%,transparent 70%);
    border-radius:50%;
}
.hero::after {
    content:''; position:absolute; bottom:-80px; left:-40px;
    width:300px; height:300px;
    background:radial-gradient(circle,rgba(251,146,60,.06) 0%,transparent 70%);
    border-radius:50%;
}
.hero h1 {
    font-size:2.3rem; font-weight:800;
    background:linear-gradient(90deg,#f87171 0%,#fb923c 45%,#fbbf24 100%);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    margin:0 0 .3rem 0; font-family:'JetBrains Mono',monospace; letter-spacing:-1px;
}
.hero p { color:#4a5070; font-size:.9rem; margin:0; }

.ct {
    font-size:.68rem; font-weight:700; letter-spacing:.14em;
    text-transform:uppercase; color:#f87171; margin-bottom:.5rem;
    font-family:'JetBrains Mono',monospace;
}

.metric-row { display:flex; gap:.9rem; flex-wrap:wrap; margin-bottom:1.4rem; }
.metric-box {
    flex:1; min-width:120px; background:#0b0d18; border:1px solid #1a1f35;
    border-radius:10px; padding:1rem 1.2rem; text-align:center;
}
.metric-box .val {
    font-size:1.8rem; font-weight:700; font-family:'JetBrains Mono',monospace;
    background:linear-gradient(90deg,#f87171,#fb923c);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
}
.metric-box .val.safe {
    background:linear-gradient(90deg,#34d399,#38bdf8);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
}
.metric-box .lbl { font-size:.68rem; color:#3a4060; text-transform:uppercase;
                   letter-spacing:.09em; margin-top:.2rem; }

.train-wrap {
    background:#0b0d18; border:1px solid #1f1a35; border-radius:10px;
    padding:1.1rem 1.6rem; margin:1.1rem 0; display:flex;
    align-items:center; justify-content:space-between; gap:1rem; flex-wrap:wrap;
}
.train-info { color:#4a5070; font-size:.84rem; }
.train-info strong { color:#8892b0; }

.pred-badge {
    display:inline-block; padding:.35rem 1.1rem; border-radius:6px;
    font-family:'JetBrains Mono',monospace; font-size:1rem; font-weight:600;
    color:#fff; margin-top:.5rem;
}
.pred-badge.anomaly { background:linear-gradient(135deg,#991b1b,#c2410c); }
.pred-badge.normal  { background:linear-gradient(135deg,#065f46,#0369a1); }
.pred-card {
    background:#0b0d18; border:1px solid #1f1a35; border-radius:10px;
    padding:1.4rem 1.8rem; margin-top:1rem;
}

.info-box {
    background:#0d0f1f; border:1px solid #1f1a35; border-radius:8px;
    padding:.8rem 1.2rem; margin:.6rem 0; font-size:.82rem; color:#4a5070;
}
.info-box strong { color:#f87171; }

.stButton > button {
    background:linear-gradient(135deg,#991b1b,#c2410c) !important;
    color:#fff !important; border:none !important; border-radius:8px !important;
    font-family:'JetBrains Mono',monospace !important; font-size:.82rem !important;
    font-weight:600 !important; letter-spacing:.05em !important;
    padding:.55rem 1.5rem !important; transition:opacity .2s !important;
}
.stButton > button:hover { opacity:.82 !important; }

[data-testid="stSlider"] > div > div > div { background:#dc2626 !important; }
[data-testid="stSelectbox"] div[data-baseweb="select"] > div,
[data-testid="stMultiSelect"] div[data-baseweb="select"] > div {
    background-color:#0b0d18 !important; border-color:#1a1f35 !important; color:#dde1f0 !important;
}
[data-testid="stFileUploader"] section {
    background:#0b0d18 !important; border:1px dashed #1f1a35 !important; border-radius:10px !important;
}
[data-testid="stDataFrame"] { border:1px solid #1a1f35; border-radius:8px; }
hr { border-color:#1a1f35 !important; }

.saved-banner {
    background:#150a0a; border:1px solid #4a1515; border-radius:10px;
    padding:1rem 1.5rem; margin-top:1.5rem;
    font-family:'JetBrains Mono',monospace; font-size:.82rem; color:#f87171;
}
.saved-banner span { color:#3a4060; }

.score-bar-wrap { margin:.3rem 0; }
.score-bar-bg {
    background:#1a1f35; border-radius:4px; height:8px; width:100%; overflow:hidden;
}
.score-bar-fill {
    height:8px; border-radius:4px;
    background:linear-gradient(90deg,#34d399,#fbbf24,#f87171);
    transition:width .4s ease;
}

.js-plotly-plot .plotly .modebar { background:transparent !important; }
</style>
""", unsafe_allow_html=True)

# ── Dirs ──────────────────────────────────────────────────────────────────────
RAW_DIR, RESULT_DIR, MODEL_DIR = "data/raw", "data/anomaly", "models"
for d in [RAW_DIR, RESULT_DIR, MODEL_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Session state ─────────────────────────────────────────────────────────────
for key in ["if_labels","if_scores","if_X","if_feature_cols","if_df_result",
            "if_contamination","if_n_estimators","if_max_features",
            "if_scale","if_model_path","if_out_csv",
            "if_n_anomaly","if_n_normal","if_pred_result","if_pred_vals",
            "if_model"]:
    if key not in st.session_state:
        st.session_state[key] = None

# ── Helpers ───────────────────────────────────────────────────────────────────
ANOMALY_COLOR = "#f87171"
NORMAL_COLOR  = "#34d399"
PALETTE       = [NORMAL_COLOR, ANOMALY_COLOR,
                 "#38bdf8","#818cf8","#fb923c","#fbbf24","#a78bfa",
                 "#f472b6","#4ade80","#60a5fa"]

def mpl_dark(figsize=(8,5)):
    fig, ax = plt.subplots(figsize=figsize, facecolor="#07090f")
    ax.set_facecolor("#0b0d18")
    for sp in ax.spines.values(): sp.set_edgecolor("#1a1f35")
    ax.tick_params(colors="#4a5070", labelsize=8)
    ax.xaxis.label.set_color("#4a5070"); ax.yaxis.label.set_color("#4a5070")
    ax.title.set_color("#c0c8e0")
    return fig, ax

PLOTLY_LAYOUT = dict(
    paper_bgcolor="#07090f", plot_bgcolor="#0b0d18",
    font=dict(family="JetBrains Mono", color="#8892b0", size=11),
    margin=dict(l=10, r=10, t=40, b=10),
    scene=dict(
        bgcolor="#0b0d18",
        xaxis=dict(backgroundcolor="#0b0d18", gridcolor="#1a1f35",
                   showbackground=True, tickfont=dict(color="#4a5070")),
        yaxis=dict(backgroundcolor="#0b0d18", gridcolor="#1a1f35",
                   showbackground=True, tickfont=dict(color="#4a5070")),
        zaxis=dict(backgroundcolor="#0b0d18", gridcolor="#1a1f35",
                   showbackground=True, tickfont=dict(color="#4a5070")),
    ),
    legend=dict(bgcolor="#0b0d18", bordercolor="#1a1f35", borderwidth=1,
                font=dict(color="#8892b0", size=10)),
)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🛡️ Isolation Forest</h1>
  <p>Anomaly Detection · Anomaly Scores · Feature Importance · Interactive 3-D · Predict</p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.divider()
    st.markdown("**Dataset**")
    uploaded = st.file_uploader("Upload CSV / Excel", type=["csv","xlsx","xls"],
                                label_visibility="collapsed")
    st.divider()
    st.markdown("**Model Parameters**")
    contamination = st.slider("Contamination (expected anomaly fraction)",
                              0.01, 0.50, 0.05, step=0.01,
                              help="Proportion of anomalies expected in the dataset")
    n_estimators  = st.slider("n_estimators (number of trees)", 50, 500, 100, step=10)
    max_features  = st.slider("max_features (features per tree)", 0.1, 1.0, 1.0, step=0.05,
                              help="Fraction of features used to build each tree")
    max_samples   = st.selectbox("max_samples", ["auto","256","512","1024"],
                                 help="Samples per tree; 'auto' = min(256, n_samples)")
    scale_data    = st.checkbox("Standardise features", value=True)
    random_state  = st.number_input("Random seed", value=42, step=1)
    st.divider()
    st.markdown("**Threshold Override**")
    use_custom_thresh = st.checkbox("Override score threshold", value=False)
    custom_thresh     = st.slider("Anomaly score threshold", -1.0, 0.0, -0.1, step=0.01,
                                  disabled=not use_custom_thresh,
                                  help="Scores below this → anomaly (default uses contamination)")

# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
df_raw = None
if uploaded:
    raw_path = os.path.join(RAW_DIR, uploaded.name)
    with open(raw_path,"wb") as f: f.write(uploaded.getbuffer())
    df_raw = pd.read_csv(raw_path) if uploaded.name.endswith(".csv") \
             else pd.read_excel(raw_path)
    st.success(f"✔ **{uploaded.name}** — {df_raw.shape[0]:,} rows × {df_raw.shape[1]} cols")

if df_raw is None:
    st.info("⬆ Upload a CSV or Excel file from the sidebar to get started.")
    st.stop()

num_cols = df_raw.select_dtypes(include=np.number).columns.tolist()
if len(num_cols) < 1:
    st.error("Need at least 1 numeric column."); st.stop()

with st.expander("🔍 Preview data", expanded=False):
    st.dataframe(df_raw.head(200), use_container_width=True)

feature_cols = st.multiselect("", options=num_cols, default=num_cols,
                              placeholder="Select feature columns…",
                              label_visibility="collapsed")
if len(feature_cols) < 1:
    st.warning("Select at least 1 feature column."); st.stop()

X_raw  = df_raw[feature_cols].dropna().values
sc_tmp = StandardScaler()
X      = sc_tmp.fit_transform(X_raw) if scale_data else X_raw.astype(float)

ms_arg = "auto" if max_samples == "auto" else int(max_samples)

# ── Train button ──────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="train-wrap">
  <div class="train-info">
    <strong>contamination={contamination}</strong> &nbsp;·&nbsp;
    <strong>trees={n_estimators}</strong> &nbsp;·&nbsp;
    <strong>max_features={max_features}</strong> &nbsp;·&nbsp;
    <strong>{len(X):,}</strong> samples &nbsp;·&nbsp;
    scaled={scale_data}
  </div>
""", unsafe_allow_html=True)
train_btn = st.button("🛡️ Fit Isolation Forest", use_container_width=False)
st.markdown("</div>", unsafe_allow_html=True)

# ── Fit ───────────────────────────────────────────────────────────────────────
if train_btn:
    with st.spinner("Growing isolation trees …"):
        clf = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            max_features=max_features,
            max_samples=ms_arg,
            random_state=int(random_state)
        )
        clf.fit(X)

    raw_scores = clf.decision_function(X)   # higher = more normal
    pred_raw   = clf.predict(X)             # 1 = normal, -1 = anomaly

    if use_custom_thresh:
        labels = np.where(raw_scores < custom_thresh, -1, 1)
    else:
        labels = pred_raw

    n_anomaly = int(np.sum(labels == -1))
    n_normal  = int(np.sum(labels == 1))

    # Normalise scores to [0,1] for display; 1 = most anomalous
    score_min, score_max = raw_scores.min(), raw_scores.max()
    rng_sc = score_max - score_min if score_max != score_min else 1.0
    anomaly_score_norm = 1.0 - (raw_scores - score_min) / rng_sc  # invert

    model_path = os.path.join(MODEL_DIR,
        f"isolation_forest_c{contamination}_t{n_estimators}.pkl")
    with open(model_path,"wb") as f:
        pickle.dump({"model": clf, "labels": labels, "scaler": sc_tmp,
                     "features": feature_cols, "scaled": scale_data,
                     "contamination": contamination,
                     "n_estimators": n_estimators,
                     "max_features": max_features}, f)

    valid_idx  = df_raw[feature_cols].dropna().index
    df_result  = df_raw.loc[valid_idx].copy()
    df_result["anomaly"]       = labels
    df_result["anomaly_score"] = np.round(anomaly_score_norm, 5)
    df_result["is_anomaly"]    = (labels == -1).astype(int)
    out_csv = os.path.join(RESULT_DIR,
        f"isolation_forest_c{contamination}.csv")
    df_result.to_csv(out_csv, index=False)

    st.session_state.if_labels        = labels
    st.session_state.if_scores        = anomaly_score_norm
    st.session_state.if_X             = X
    st.session_state.if_feature_cols  = feature_cols
    st.session_state.if_df_result     = df_result
    st.session_state.if_contamination = contamination
    st.session_state.if_n_estimators  = n_estimators
    st.session_state.if_max_features  = max_features
    st.session_state.if_scale         = scale_data
    st.session_state.if_model_path    = model_path
    st.session_state.if_out_csv       = out_csv
    st.session_state.if_n_anomaly     = n_anomaly
    st.session_state.if_n_normal      = n_normal
    st.session_state.if_pred_result   = None
    st.session_state.if_model         = clf

# ── Guard ─────────────────────────────────────────────────────────────────────
if st.session_state.if_labels is None:
    st.stop()

labels       = st.session_state.if_labels
scores       = st.session_state.if_scores
X_ss         = st.session_state.if_X
feature_cols = st.session_state.if_feature_cols
df_result    = st.session_state.if_df_result
contamination_ss = st.session_state.if_contamination
n_anomaly    = st.session_state.if_n_anomaly
n_normal     = st.session_state.if_n_normal
model_path   = st.session_state.if_model_path
out_csv      = st.session_state.if_out_csv
clf          = st.session_state.if_model

anomaly_pct = 100.0 * n_anomaly / len(labels) if len(labels) > 0 else 0

st.divider()
st.markdown("### 📊 Results")

# ── Metrics ───────────────────────────────────────────────────────────────────
mean_score_anom = float(scores[labels == -1].mean()) if n_anomaly > 0 else 0.0
mean_score_norm = float(scores[labels == 1].mean())  if n_normal  > 0 else 0.0

st.markdown(f"""
<div class="metric-row">
  <div class="metric-box"><div class="val">{n_anomaly}</div><div class="lbl">Anomalies</div></div>
  <div class="metric-box"><div class="val safe">{n_normal}</div><div class="lbl">Normal</div></div>
  <div class="metric-box"><div class="val">{anomaly_pct:.1f}%</div><div class="lbl">Anomaly %</div></div>
  <div class="metric-box"><div class="val">{len(X_ss):,}</div><div class="lbl">Samples</div></div>
  <div class="metric-box"><div class="val">{mean_score_anom:.3f}</div><div class="lbl">Avg Anom Score</div></div>
  <div class="metric-box"><div class="val">{len(feature_cols)}</div><div class="lbl">Features</div></div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# ANOMALY SCORE DISTRIBUTION + 2-D SCATTER
# ══════════════════════════════════════════════════════════════════════════════
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="ct">Anomaly Score Distribution</div>', unsafe_allow_html=True)
    fig, ax = mpl_dark((5, 3.5))
    ax.hist(scores[labels == 1],  bins=50, color=NORMAL_COLOR,
            alpha=0.7, label="Normal",  density=True)
    ax.hist(scores[labels == -1], bins=50, color=ANOMALY_COLOR,
            alpha=0.8, label="Anomaly", density=True)
    ax.set_xlabel("Anomaly Score (0=normal, 1=anomalous)")
    ax.set_ylabel("Density")
    ax.set_title("Score Distributions", fontsize=10)
    ax.legend(fontsize=7, facecolor="#0b0d18", edgecolor="#1a1f35", labelcolor="#c0c8e0")
    plt.tight_layout(); st.pyplot(fig); plt.close()

with col2:
    st.markdown('<div class="ct">2-D Scatter — Anomalies Highlighted (PCA)</div>',
                unsafe_allow_html=True)
    pca2 = PCA(n_components=2, random_state=42)
    X2   = pca2.fit_transform(X_ss) if X_ss.shape[1] > 2 else X_ss[:,:2]
    xl2  = "PC1" if X_ss.shape[1] > 2 else feature_cols[0]
    yl2  = "PC2" if X_ss.shape[1] > 2 else feature_cols[1]
    fig, ax = mpl_dark((5, 3.8))
    mask_n = labels == 1;  mask_a = labels == -1
    ax.scatter(X2[mask_n,0], X2[mask_n,1], s=12, alpha=.45,
               color=NORMAL_COLOR, label="Normal", linewidths=0)
    ax.scatter(X2[mask_a,0], X2[mask_a,1], s=28, alpha=.85,
               color=ANOMALY_COLOR, label="Anomaly",
               linewidths=0.5, edgecolors="#ff0000")
    ax.set_xlabel(xl2); ax.set_ylabel(yl2)
    ax.set_title("Normal vs Anomaly", fontsize=10)
    ax.legend(fontsize=7, facecolor="#0b0d18", edgecolor="#1a1f35",
              labelcolor="#c0c8e0", loc="best")
    plt.tight_layout(); st.pyplot(fig); plt.close()

# ── Score heatmap + box plots ─────────────────────────────────────────────────
col3, col4 = st.columns(2)

with col3:
    st.markdown('<div class="ct">Anomaly Score Heatmap (PCA 2-D)</div>', unsafe_allow_html=True)
    fig_h, ax_h = mpl_dark((5, 3.8))
    sc = ax_h.scatter(X2[:,0], X2[:,1], c=scores, cmap="RdYlGn_r",
                      s=8, alpha=0.7, vmin=0, vmax=1, linewidths=0)
    plt.colorbar(sc, ax=ax_h, label="Anomaly Score")
    ax_h.set_xlabel(xl2); ax_h.set_ylabel(yl2)
    ax_h.set_title("Score Gradient (red = anomalous)", fontsize=10)
    plt.tight_layout(); st.pyplot(fig_h); plt.close()

with col4:
    st.markdown('<div class="ct">Feature Distributions · Normal vs Anomaly</div>',
                unsafe_allow_html=True)
    show_feat = feature_cols[:min(6, len(feature_cols))]
    fig_b, ax_b = mpl_dark((5.5, 3.8))
    positions_n = np.arange(len(show_feat)) - 0.2
    positions_a = np.arange(len(show_feat)) + 0.2
    data_n = [X_ss[mask_n, i] for i in range(len(show_feat))]
    data_a = [X_ss[mask_a, i] for i in range(len(show_feat))]
    bp_n = ax_b.boxplot(data_n, positions=positions_n, widths=0.32,
                        patch_artist=True, showfliers=False,
                        medianprops=dict(color="#07090f", linewidth=1.5))
    bp_a = ax_b.boxplot(data_a, positions=positions_a, widths=0.32,
                        patch_artist=True, showfliers=False,
                        medianprops=dict(color="#07090f", linewidth=1.5))
    for patch in bp_n["boxes"]: patch.set_facecolor(NORMAL_COLOR);  patch.set_alpha(0.7)
    for patch in bp_a["boxes"]: patch.set_facecolor(ANOMALY_COLOR); patch.set_alpha(0.7)
    for el in bp_n["whiskers"] + bp_n["caps"]: el.set_color(NORMAL_COLOR)
    for el in bp_a["whiskers"] + bp_a["caps"]: el.set_color(ANOMALY_COLOR)
    ax_b.set_xticks(np.arange(len(show_feat)))
    ax_b.set_xticklabels(show_feat, rotation=30, ha="right", fontsize=7.5, color="#c0c8e0")
    ax_b.set_title("Feature Box Plots (scaled)", fontsize=10)
    from matplotlib.patches import Patch
    ax_b.legend(handles=[Patch(facecolor=NORMAL_COLOR, label="Normal"),
                          Patch(facecolor=ANOMALY_COLOR, label="Anomaly")],
                fontsize=7, facecolor="#0b0d18", edgecolor="#1a1f35", labelcolor="#c0c8e0")
    plt.tight_layout(); st.pyplot(fig_b); plt.close()

# ── Feature importance (mean score delta) ─────────────────────────────────────
st.markdown('<div class="ct">Feature Anomaly Contribution (mean |score| per feature)</div>',
            unsafe_allow_html=True)
st.markdown("""
<div class="info-box">
  Each feature's <strong>anomaly contribution</strong> is estimated by the mean absolute
  value of that feature dimension for anomalous points vs normal points. Larger difference
  → stronger signal for that feature.
</div>
""", unsafe_allow_html=True)

mean_anom_feat = np.abs(X_ss[mask_a]).mean(axis=0) if n_anomaly > 0 else np.zeros(len(feature_cols))
mean_norm_feat = np.abs(X_ss[mask_n]).mean(axis=0) if n_normal  > 0 else np.zeros(len(feature_cols))
contribution   = mean_anom_feat - mean_norm_feat
sort_idx       = np.argsort(contribution)[::-1]

fig_fi, ax_fi = mpl_dark((12, 3.2))
colors_fi = [ANOMALY_COLOR if contribution[i] >= 0 else NORMAL_COLOR for i in sort_idx]
ax_fi.bar(np.array(feature_cols)[sort_idx], contribution[sort_idx],
          color=colors_fi, alpha=0.85, width=0.6)
ax_fi.axhline(0, color="#1a1f35", lw=1)
ax_fi.set_ylabel("Δ mean |value| (anomaly − normal)")
ax_fi.set_title("Feature Contribution to Anomaly", fontsize=10)
ax_fi.set_xticklabels(np.array(feature_cols)[sort_idx],
                      rotation=35, ha="right", fontsize=8, color="#c0c8e0")
plt.tight_layout(); st.pyplot(fig_fi); plt.close()

# ── Cumulative anomaly scores (sorted) ────────────────────────────────────────
col5, col6 = st.columns(2)
with col5:
    st.markdown('<div class="ct">Sorted Anomaly Scores</div>', unsafe_allow_html=True)
    sorted_scores = np.sort(scores)[::-1]
    fig_s, ax_s   = mpl_dark((5, 3.2))
    cmap_colors   = plt.cm.RdYlGn_r(sorted_scores)
    ax_s.bar(range(len(sorted_scores)), sorted_scores, color=cmap_colors, width=1.0)
    ax_s.set_xlabel("Sample rank"); ax_s.set_ylabel("Anomaly Score")
    ax_s.set_title("All Scores (sorted desc)", fontsize=10)
    if contamination_ss and len(sorted_scores) > 0:
        cut_idx = int(contamination_ss * len(sorted_scores))
        ax_s.axvline(cut_idx, color="#f472b6", lw=1.2, linestyle="--",
                     label=f"contamination cut @ {cut_idx}")
        ax_s.legend(fontsize=7, facecolor="#0b0d18", edgecolor="#1a1f35", labelcolor="#c0c8e0")
    plt.tight_layout(); st.pyplot(fig_s); plt.close()

with col6:
    st.markdown('<div class="ct">Normal vs Anomaly Pie</div>', unsafe_allow_html=True)
    fig_p, ax_p = mpl_dark((5, 3.2))
    wedges, texts, autotexts = ax_p.pie(
        [n_normal, n_anomaly],
        labels=["Normal","Anomaly"],
        colors=[NORMAL_COLOR, ANOMALY_COLOR],
        autopct="%1.1f%%", pctdistance=0.82,
        wedgeprops=dict(linewidth=1.5, edgecolor="#07090f")
    )
    for t in texts: t.set_color("#8892b0"); t.set_fontsize(9)
    for a in autotexts: a.set_color("#dde1f0"); a.set_fontsize(8)
    ax_p.set_title("Dataset Composition", fontsize=10)
    plt.tight_layout(); st.pyplot(fig_p); plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# TOP ANOMALIES TABLE
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
st.markdown("### 🚨 Top Anomalies")
st.markdown('<div class="ct">Highest anomaly score rows</div>', unsafe_allow_html=True)

top_n = st.slider("Show top N anomalies", 5, min(200, n_anomaly if n_anomaly > 0 else 5),
                  min(20, n_anomaly if n_anomaly > 0 else 5), key="top_n_slider")
df_anomalies = df_result[df_result["is_anomaly"] == 1]\
    .sort_values("anomaly_score", ascending=False).head(top_n)

def color_anomaly(val):
    try:
        v = float(val)
        intensity = int(v * 180)
        return f"color: rgb(248,{255-intensity},{113-intensity//2}); font-weight:700; font-family:'JetBrains Mono',monospace"
    except:
        return ""

st.dataframe(
    df_result.style
        .map(color_label, subset=["anomaly"])
        .map(color_anomaly, subset=["anomaly_score"]),
    use_container_width=True,
    height=320
)

# ══════════════════════════════════════════════════════════════════════════════
# FULL RESULTS TABLE
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("### 🏷️ Full Results")
st.markdown('<div class="ct">Dataset with anomaly labels and scores</div>', unsafe_allow_html=True)

def color_label(val):
    try:
        v = int(val)
        if v == -1:
            return f"color:{ANOMALY_COLOR}; font-weight:700; font-family:'JetBrains Mono',monospace"
        return f"color:{NORMAL_COLOR}; font-weight:700; font-family:'JetBrains Mono',monospace"
    except:
        return ""

st.dataframe(
    df_result.style
        .applymap(color_label, subset=["anomaly"])
        .applymap(color_anomaly, subset=["anomaly_score"]),
    use_container_width=True, height=320
)

# ══════════════════════════════════════════════════════════════════════════════
# SINGLE-ROW PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
st.markdown('<div class="ct">Predict anomaly for a new data point</div>', unsafe_allow_html=True)
with st.expander("🔮 Enter feature values", expanded=True):
    col_inputs = st.columns(min(len(feature_cols), 4))
    new_vals   = {}
    for i, col in enumerate(feature_cols):
        col_min  = float(df_raw[col].min())
        col_max  = float(df_raw[col].max())
        col_mean = float(df_raw[col].mean())
        prev     = (st.session_state.if_pred_vals or {}).get(col, round(col_mean, 4))
        new_vals[col] = col_inputs[i % len(col_inputs)].number_input(
            col, value=prev, min_value=col_min, max_value=col_max,
            step=round((col_max - col_min) / 100, 6), format="%.4f",
            key=f"if_pred_{col}")

    if st.button("🔍 Predict", key="if_predict"):
        new_row    = np.array([[new_vals[c] for c in feature_cols]])
        new_scaled = sc_tmp.transform(new_row) if st.session_state.if_scale else new_row
        raw_score  = float(clf.decision_function(new_scaled)[0])
        pred_label = int(clf.predict(new_scaled)[0])

        score_min_ss = scores.min(); score_max_ss = scores.max()
        rng_ss = score_max_ss - score_min_ss if score_max_ss != score_min_ss else 1.0
        raw_score_norm = 1.0 - (raw_score - (1.0 - score_max_ss / rng_ss + score_min_ss / rng_ss))
        # simple normalisation against training range
        norm_sc = float(np.clip(1.0 - (raw_score - (1 - scores.mean())) , 0, 1))

        st.session_state.if_pred_result = {
            "label": pred_label, "raw": raw_score, "norm": norm_sc}
        st.session_state.if_pred_vals   = new_vals

    if st.session_state.if_pred_result is not None:
        res        = st.session_state.if_pred_result
        is_anomaly = res["label"] == -1
        badge_cls  = "pred-badge anomaly" if is_anomaly else "pred-badge normal"
        badge_lbl  = "🚨 ANOMALY" if is_anomaly else "✅ Normal"
        bar_pct    = int(res["norm"] * 100)
        st.markdown(f"""
        <div class="pred-card">
          <div style="color:#3a4060;font-size:.78rem;font-family:'JetBrains Mono',monospace;
                      text-transform:uppercase;letter-spacing:.1em;">Prediction result</div>
          <div class="{badge_cls}">{badge_lbl}</div>
          <div style="margin-top:1rem;">
            <div style="color:#3a4060;font-size:.75rem;font-family:'JetBrains Mono',monospace;
                        text-transform:uppercase;letter-spacing:.09em;margin-bottom:.4rem;">
              Anomaly Score &nbsp;<strong style="color:#f87171">{res["norm"]:.4f}</strong>
            </div>
            <div class="score-bar-bg">
              <div class="score-bar-fill" style="width:{bar_pct}%;"></div>
            </div>
          </div>
          <div style="margin-top:.7rem;color:#3a4060;font-size:.75rem;
                      font-family:'JetBrains Mono',monospace;">
            Raw decision function score: <span style="color:#8892b0">{res["raw"]:.5f}</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# INTERACTIVE 3-D VISUALISATIONS
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
st.markdown("### 🧊 Interactive 3-D Visualisations")
st.caption("Drag to rotate · Scroll to zoom · Double-click to reset")

pca3 = PCA(n_components=min(3, X_ss.shape[1]), random_state=42)
Xr   = pca3.fit_transform(X_ss)
if Xr.shape[1] == 2:
    Xr = np.hstack([Xr, np.zeros((len(Xr),1))])
xl3 = "PC1" if X_ss.shape[1]>2 else feature_cols[0]
yl3 = "PC2" if X_ss.shape[1]>2 else feature_cols[1]
zl3 = "PC3" if X_ss.shape[1]>3 else (feature_cols[2] if X_ss.shape[1]==3 else "—")

# ── 3-D Scatter (Normal vs Anomaly) ──────────────────────────────────────────
st.markdown('<div class="ct">3-D Anomaly Scatter (rotate me!)</div>', unsafe_allow_html=True)
fig_3d = go.Figure()
fig_3d.add_trace(go.Scatter3d(
    x=Xr[mask_n,0], y=Xr[mask_n,1], z=Xr[mask_n,2],
    mode="markers",
    marker=dict(size=2.8, color=NORMAL_COLOR, opacity=0.45, line=dict(width=0)),
    name="Normal"
))
fig_3d.add_trace(go.Scatter3d(
    x=Xr[mask_a,0], y=Xr[mask_a,1], z=Xr[mask_a,2],
    mode="markers",
    marker=dict(size=5, color=ANOMALY_COLOR, opacity=0.9,
                line=dict(color="#ff0000", width=0.5)),
    name="Anomaly"
))
fig_3d.update_layout(
    paper_bgcolor=PLOTLY_LAYOUT["paper_bgcolor"],
    plot_bgcolor=PLOTLY_LAYOUT["plot_bgcolor"],
    font=PLOTLY_LAYOUT["font"],
    height=560,
    title=dict(text="3-D Normal vs Anomaly Scatter",
               font=dict(color="#8892b0", size=13)),
    scene=dict(**PLOTLY_LAYOUT["scene"],
               xaxis_title=xl3, yaxis_title=yl3, zaxis_title=zl3),
    legend=PLOTLY_LAYOUT["legend"]
)
st.plotly_chart(fig_3d, use_container_width=True)

# ── 3-D score gradient + feature landscape ───────────────────────────────────
col4a, col4b = st.columns(2)

with col4a:
    st.markdown('<div class="ct">3-D Anomaly Score Gradient (rotate me!)</div>',
                unsafe_allow_html=True)
    fig_sg = go.Figure(data=[go.Scatter3d(
        x=Xr[:,0], y=Xr[:,1], z=Xr[:,2],
        mode="markers",
        marker=dict(size=3, color=scores, colorscale="RdYlGn_r",
                    cmin=0, cmax=1, opacity=0.75, line=dict(width=0),
                    colorbar=dict(title="Score", thickness=10,
                                  tickfont=dict(color="#4a5070", size=9))),
        text=[f"score: {s:.3f}" for s in scores],
        hoverinfo="text", name="Score"
    )])
    fig_sg.update_layout(
        **{k:v for k,v in PLOTLY_LAYOUT.items() if k != "scene"},
        scene=dict(**PLOTLY_LAYOUT["scene"],
                   xaxis_title=xl3, yaxis_title=yl3, zaxis_title=zl3),
        height=420,
        title=dict(text="Anomaly Score Gradient",
                   font=dict(color="#8892b0", size=12))
    )
    st.plotly_chart(fig_sg, use_container_width=True)

with col4b:
    st.markdown('<div class="ct">3-D Feature Mean Landscape (rotate me!)</div>',
                unsafe_allow_html=True)
    mean_n_feat = X_ss[mask_n].mean(axis=0) if n_normal  > 0 else np.zeros(len(feature_cols))
    mean_a_feat = X_ss[mask_a].mean(axis=0) if n_anomaly > 0 else np.zeros(len(feature_cols))
    z_surf   = np.vstack([mean_n_feat, mean_a_feat])
    feat_idx = list(range(len(feature_cols)))
    fig_surf = go.Figure(data=[go.Surface(
        x=feat_idx, y=[0,1], z=z_surf,
        colorscale="RdYlGn_r", opacity=0.88,
        contours=dict(
            x=dict(show=True, color="#1a1f35", width=1),
            y=dict(show=True, color="#1a1f35", width=1),
        )
    )])
    fig_surf.update_layout(
        paper_bgcolor=PLOTLY_LAYOUT["paper_bgcolor"],
        plot_bgcolor=PLOTLY_LAYOUT["plot_bgcolor"],
        font=PLOTLY_LAYOUT["font"],
        scene=dict(
            xaxis=dict(tickvals=feat_idx, ticktext=feature_cols, title="Feature"),
            yaxis=dict(tickvals=[0,1], ticktext=["Normal","Anomaly"], title="Class"),
            zaxis=dict(title="Mean (scaled)")
        ),
        height=420,
        title=dict(text="Normal vs Anomaly Feature Means",
                   font=dict(color="#8892b0", size=12))
    )
    st.plotly_chart(fig_surf, use_container_width=True)

# ── 3-D score surface over PC1/PC2 ───────────────────────────────────────────
st.markdown('<div class="ct">3-D Anomaly Score Surface over PCA space (rotate me!)</div>',
            unsafe_allow_html=True)
fig_vol = go.Figure(data=[go.Scatter3d(
    x=Xr[:,0], y=Xr[:,1], z=scores,
    mode="markers",
    marker=dict(
        size=3, color=scores, colorscale="Hot",
        cmin=0, cmax=1, opacity=0.70, line=dict(width=0),
        colorbar=dict(title="Anomaly Score", thickness=10,
                      tickfont=dict(color="#4a5070", size=9))
    ),
    text=[f"score:{s:.3f}" for s in scores],
    hoverinfo="text", name="Score surface"
)])
fig_vol.update_layout(
    **{k:v for k,v in PLOTLY_LAYOUT.items() if k != "scene"},
    scene=dict(**PLOTLY_LAYOUT["scene"],
               xaxis_title=xl3, yaxis_title=yl3,
               zaxis_title="Anomaly Score"),
    height=520,
    title=dict(text="Score Elevation Surface (PC1 × PC2 × Score)",
               font=dict(color="#8892b0", size=13))
)
st.plotly_chart(fig_vol, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# DOWNLOADS + SAVED BANNER
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
c1, c2 = st.columns(2)
with c1:
    st.download_button("⬇ Download Results CSV",
                       df_result.to_csv(index=False).encode(),
                       file_name=f"isolation_forest_c{contamination_ss}.csv",
                       mime="text/csv", use_container_width=True)
with c2:
    with open(model_path,"rb") as f:
        st.download_button("⬇ Download Model (.pkl)", f.read(),
                           file_name=os.path.basename(model_path),
                           mime="application/octet-stream",
                           use_container_width=True)

abs_model = os.path.abspath(model_path)
abs_csv   = os.path.abspath(out_csv)
ts        = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
st.markdown(f"""
<div class="saved-banner">
  ✔ Model saved to &nbsp;<strong>{abs_model}</strong><br>
  ✔ Results saved to &nbsp;<strong>{abs_csv}</strong><br>
  <span>Saved at {ts} &nbsp;·&nbsp; contamination={contamination_ss}
  &nbsp;·&nbsp; trees={st.session_state.if_n_estimators}
  &nbsp;·&nbsp; anomalies={n_anomaly} ({anomaly_pct:.1f}%)
  &nbsp;·&nbsp; normal={n_normal}</span>
</div>
""", unsafe_allow_html=True)