


"""
app.py  v4
----------
Strawberry Yield Harvest Advisor — Streamlit app.

Key changes from v3:
  - Loads pre-trained models from deployment/ folder (no training at runtime)
  - Stage 2 uses Rule B (Method B with velocity), not Rule A
  - coarsen_n=7 fixed to match training grid
  - Raw harvest data loaded from parquet (real kg, not feature-normalised values)
  - Yield maps and total yield shown in real kg

Run locally:
    streamlit run app.py

Folder structure:
    app.py
    harvest_advisor.py
    feature_engineering.py
    models.py
    data_pipeline.py
    deployment/
        model_sm_7x7.pkl
        model_sal_7x7.pkl
        thresholds_sm.pkl
        thresholds_sal.pkl
        df_raw_sm.parquet
        df_raw_sal.parquet
        weather_SantaMaria.csv
        weather_Salinas.csv
        deploy_config.json
"""

import sys
import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Harvest Advisor",
    page_icon="🍓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# # ── CSS ───────────────────────────────────────────────────────────────────────
# st.markdown("""
# <style>
# @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

# html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
# h1, h2, h3 { font-family: 'DM Serif Display', serif; }

# section[data-testid="stSidebar"] { background: #0f1f14; }
# section[data-testid="stSidebar"] * { color: #d4e8d0 !important; }
# section[data-testid="stSidebar"] .stSelectbox label,
# section[data-testid="stSidebar"] .stSlider label,
# section[data-testid="stSidebar"] .stFileUploader label {
#     color: #7fbf7f !important;
#     font-size: 0.82rem;
#     text-transform: uppercase;
#     letter-spacing: 0.05em;
# }

# div[data-testid="metric-container"] {
#     background: #f7f5f0;
#     border: 1px solid #e0dbd0;
#     border-radius: 10px;
#     padding: 1rem 1.2rem;
# }
# div[data-testid="metric-container"] label {
#     color: #6b6558 !important;
#     font-size: 0.78rem !important;
#     text-transform: uppercase;
#     letter-spacing: 0.05em;
# }
# div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
#     font-family: 'DM Serif Display', serif;
#     font-size: 2rem !important;
#     color: #1a2e1f !important;
# }

# .rec-card {
#     background: linear-gradient(135deg, #1a3320 0%, #2d5a3d 100%);
#     border-radius: 14px;
#     padding: 1.6rem 2rem;
#     color: white;
#     margin-bottom: 1rem;
# }
# .rec-title {
#     font-family: 'DM Serif Display', serif;
#     font-size: 1.1rem;
#     color: #7fbf7f;
#     margin-bottom: 0.3rem;
#     text-transform: uppercase;
#     letter-spacing: 0.08em;
# }
# .rec-date {
#     font-family: 'DM Serif Display', serif;
#     font-size: 2.8rem;
#     color: white;
#     line-height: 1.1;
# }
# .rec-sub { font-size: 0.9rem; color: #a8d5a8; margin-top: 0.4rem; }

# .rule-badge {
#     display: inline-block;
#     background: #2d6a3f;
#     color: white;
#     border-radius: 6px;
#     padding: 0.2rem 0.6rem;
#     font-size: 0.78rem;
#     font-weight: 600;
#     letter-spacing: 0.04em;
#     margin-top: 0.5rem;
# }
# .threshold-card {
#     background: #f0f7f0;
#     border-left: 4px solid #2d6a3f;
#     border-radius: 0 8px 8px 0;
#     padding: 0.8rem 1rem;
#     font-size: 0.88rem;
#     color: #1a3320;
#     margin-top: 0.8rem;
#     font-family: monospace;
# }
# .info-banner {
#     background: #f0f7ff;
#     border-left: 4px solid #3b82f6;
#     border-radius: 0 8px 8px 0;
#     padding: 0.8rem 1rem;
#     font-size: 0.88rem;
#     color: #1e3a5f;
#     margin-bottom: 1rem;
# }
# </style>
# """, unsafe_allow_html=True)




# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1, h2, h3 { font-family: 'DM Serif Display', serif; }

/* Sidebar */
section[data-testid="stSidebar"] { background: #3a1024; }
section[data-testid="stSidebar"] * { color: #ffe4ef !important; }
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stFileUploader label {
    color: #ff9fbd !important;
    font-size: 0.82rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* Metric cards */
div[data-testid="metric-container"] {
    background: #fff6f8;
    border: 1px solid #f2c6d3;
    border-radius: 10px;
    padding: 1rem 1.2rem;
}
div[data-testid="metric-container"] label {
    color: #8a5a68 !important;
    font-size: 0.78rem !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
    font-family: 'DM Serif Display', serif;
    font-size: 2rem !important;
    color: #4a1329 !important;
}

/* Recommendation card */
.rec-card {
    background: linear-gradient(135deg, #5a1633 0%, #c44569 100%);
    border-radius: 14px;
    padding: 1.6rem 2rem;
    color: white;
    margin-bottom: 1rem;
}
.rec-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.1rem;
    color: #ffc2d6;
    margin-bottom: 0.3rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
.rec-date {
    font-family: 'DM Serif Display', serif;
    font-size: 2.8rem;
    color: white;
    line-height: 1.1;
}
.rec-sub {
    font-size: 0.9rem;
    color: #ffe0eb;
    margin-top: 0.4rem;
}

/* Rule badge */
.rule-badge {
    display: inline-block;
    background: #d6336c;
    color: white;
    border-radius: 6px;
    padding: 0.2rem 0.6rem;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    margin-top: 0.5rem;
}

/* Threshold card */
.threshold-card {
    background: #fff0f5;
    border-left: 4px solid #d6336c;
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1rem;
    font-size: 0.88rem;
    color: #4a1329;
    margin-top: 0.8rem;
    font-family: monospace;
}

/* Info banner */
.info-banner {
    background: #fff4f7;
    border-left: 4px solid #ff7aa2;
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1rem;
    font-size: 0.88rem;
    color: #5a1633;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)



# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE       = os.path.dirname(os.path.abspath(__file__))
_DEPLOY_DIR = os.path.join(_HERE, "deployment")
_SRC        = os.path.join(_HERE, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import harvest_advisor as ha

# ── Load deploy config ────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_deploy_config():
    path = os.path.join(_DEPLOY_DIR, "deploy_config.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)

deploy_config = load_deploy_config()
if deploy_config is None:
    st.error("deployment/deploy_config.json not found. "
             "Run the Colab save cell first.")
    st.stop()

# ── Cached loaders ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model...")
def load_model(site: str):
    fname = deploy_config[site]['model_file']
    path  = os.path.join(_DEPLOY_DIR, fname)
    with open(path, 'rb') as f:
        return pickle.load(f)

@st.cache_resource(show_spinner="Loading thresholds...")
def load_thresholds(site: str):
    fname = deploy_config[site]['thresholds_file']
    path  = os.path.join(_DEPLOY_DIR, fname)
    with open(path, 'rb') as f:
        return pickle.load(f)

@st.cache_data(show_spinner="Loading harvest history...")
def load_raw_data(site: str) -> pd.DataFrame:
    fname = deploy_config[site]['raw_data_file']
    path  = os.path.join(_DEPLOY_DIR, fname)
    df    = pd.read_parquet(path)
    df['harvest_date'] = pd.to_datetime(df['harvest_date'])
    return df

@st.cache_data(show_spinner="Loading weather data...")
def load_weather(site: str) -> pd.DataFrame:
    fname = deploy_config[site]['weather_file']
    path  = os.path.join(_DEPLOY_DIR, fname)
    return pd.read_csv(path, index_col=0, parse_dates=True)

# ── CSV parser for uploaded harvest file ──────────────────────────────────────
def parse_uploaded_csv(uploaded_file):
    df = pd.read_csv(uploaded_file, header=None)
    if df.shape[1] == 6:
        df.columns = ["index","field_x","field_y","weight_kg","easting","northing"]
        df = df.drop(columns=["index"])
    elif df.shape[1] == 5:
        df.columns = ["field_x","field_y","weight_kg","easting","northing"]
    else:
        st.error(f"Unexpected column count: {df.shape[1]}. Expected 5 or 6.")
        return None
    for col in ["field_x","field_y","weight_kg","easting","northing"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["field_x","field_y","weight_kg"])

# ── Visualisation helpers ─────────────────────────────────────────────────────
def build_yield_map_fig(df_day: pd.DataFrame, title: str,
                         cmap: str = "YlOrRd") -> plt.Figure:
    x_vals = sorted(df_day["field_x"].unique())
    y_vals = sorted(df_day["field_y"].unique())
    x2i    = {v: i for i, v in enumerate(x_vals)}
    y2i    = {v: i for i, v in enumerate(y_vals)}
    grid   = np.zeros((len(y_vals), len(x_vals)))
    for _, row in df_day.iterrows():
        grid[y2i[row["field_y"]], x2i[row["field_x"]]] = row["weight_kg"]
    vmax = float(np.quantile(grid[grid>0], 0.99)) if (grid>0).any() else 1.0
    fig, ax = plt.subplots(figsize=(7, 4.5))
    im = ax.imshow(grid, cmap=cmap, aspect="auto", vmin=0, vmax=vmax)
    plt.colorbar(im, ax=ax, label="Yield (kg)", shrink=0.85)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("field_x index", fontsize=9)
    ax.set_ylabel("field_y index", fontsize=9)
    ax.text(0.02, 0.97, f"Total: {df_day['weight_kg'].sum():,.0f} kg",
            transform=ax.transAxes, fontsize=9, va="top", color="white",
            bbox=dict(boxstyle="round,pad=0.3", fc="#333", alpha=0.65))
    plt.tight_layout()
    return fig


def build_predicted_map_fig(advice: dict, opt_days: int) -> plt.Figure:
    """Yield map for the optimal harvest day (real kg)."""
    inf_df = advice["inference_dfs"][opt_days]
    pred   = advice["yield_maps"][opt_days]     # real kg (no normalisation)
    x_vals = sorted(inf_df["field_x"].unique())
    y_vals = sorted(inf_df["field_y"].unique())
    x2i = {v: i for i, v in enumerate(x_vals)}
    y2i = {v: i for i, v in enumerate(y_vals)}
    grid = np.zeros((len(y_vals), len(x_vals)))
    for (_, row), p in zip(inf_df.iterrows(), pred):
        grid[y2i[row["field_y"]], x2i[row["field_x"]]] = p
    vmax = float(np.quantile(grid[grid>0], 0.99)) if (grid>0).any() else 1.0
    cdate = advice["last_harvest_date"] + timedelta(days=opt_days)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    im = ax.imshow(grid, cmap="YlOrRd", aspect="auto", vmin=0, vmax=vmax)
    plt.colorbar(im, ax=ax, label="kg", shrink=0.85)
    ax.set_title(
        f"Predicted Yield Map\n{cdate.date()}  (+{opt_days} days)  "
        f"{pred.sum():,.0f} kg",
        fontsize=10, fontweight="bold", color="#2d6a3f")
    ax.set_xlabel("field_x", fontsize=8)
    ax.set_ylabel("field_y", fontsize=8)
    plt.tight_layout()
    return fig


def build_growth_rate_fig(thresholds: dict,
                           gr_pred: float,
                           velocity: float,
                           opt_days: int) -> plt.Figure:
    """Single-point growth rate chart for Method B."""
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.axhline(thresholds["t_high"], color="#E07B39", ls="--", lw=1.5,
               label=f"t_high = {thresholds['t_high']:.3f}  (wait longer)")
    ax.axhline(thresholds["t_low"],  color="#c0392b", ls="--", lw=1.5,
               label=f"t_low  = {thresholds['t_low']:.3f}  (harvest soon)")
    ax.fill_between([0, 1], thresholds["t_low"], thresholds["t_high"],
                    alpha=0.08, color="#5B8DB8", label="Stable zone")
    color = ("#2d6a3f" if gr_pred >= thresholds["t_high"] else
             "#c0392b" if gr_pred < thresholds["t_low"] else "#5B8DB8")
    ax.scatter([0.5], [gr_pred], s=200, color=color, zorder=5,
               label=f"gr_pred = {gr_pred:.3f}")
    ax.annotate(f"gr = {gr_pred:.3f}\nvelocity = {velocity:+.3f}\n→ wait {opt_days}d",
                (0.5, gr_pred), textcoords="offset points", xytext=(20, 8),
                fontsize=9, color=color, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    ax.set_xlim(0, 1); ax.set_xticks([])
    ax.set_ylabel("Growth rate (predicted / last actual)", fontsize=9)
    ax.set_title("Stage 2 — Method B Decision", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    return fig

# ════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════

with st.sidebar:
    st.markdown("##  Harvest Advisor")
    st.markdown("---")

    site = st.selectbox("Site", ["SantaMaria", "Salinas"])

    cfg = deploy_config[site]
    st.markdown(f"""
    <small>
    Model: <b>{cfg['best_model']}</b><br>
    Grid:  <b>7×7</b><br>
    Stage 2: <b>{cfg['stage2_rule']}</b>
    </small>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**Upload latest harvest CSV**")
    st.markdown(
        "<small>Format: field_x, field_y, weight_kg, easting, northing<br>"
        "(or with index column as first column)</small>",
        unsafe_allow_html=True)
    uploaded    = st.file_uploader("", type=["csv"], label_visibility="collapsed")
    upload_date = st.date_input("Harvest date of uploaded file",
                                 value=pd.Timestamp("2024-07-09"))

    st.markdown("---")
    run_btn = st.button("Run Analysis", type="primary", use_container_width=True)


# ════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════

st.markdown(f"# Strawberry Yield — {site}")
st.markdown(
    '<div class="info-banner">Upload the latest harvest CSV on the left, '
    'then click <strong>Run Analysis</strong> to get your harvest recommendation.</div>',
    unsafe_allow_html=True)

if not run_btn:
    c1, c2, c3 = st.columns(3)
    c1.metric("Site", site)
    c2.metric("Grid", "7×7")
    c3.metric("Stage 2 Rule", "Method B")
    st.info("Upload a harvest CSV and press **Run Analysis** to begin.")
    st.stop()

# ── Load pre-trained artifacts ────────────────────────────────────────────────
with st.spinner("Loading pre-trained model and data..."):
    model_results = load_model(site)
    thresholds    = load_thresholds(site)
    df_raw        = load_raw_data(site)
    weather       = load_weather(site)

coarsen_n = deploy_config[site]['coarsen_n']   # 7

# ── Append uploaded harvest if provided ───────────────────────────────────────
if uploaded is not None:
    df_upload = parse_uploaded_csv(uploaded)
    if df_upload is None:
        st.stop()
    df_upload["harvest_date"] = pd.Timestamp(upload_date)
    df_upload["site"]         = site
    last_date = pd.Timestamp(upload_date)
    df_raw = pd.concat([
        df_raw,
        df_upload[["harvest_date","field_x","field_y",
                   "weight_kg","easting","northing"]]
    ], ignore_index=True)
else:
    last_date = df_raw["harvest_date"].max()
    st.info(f"No CSV uploaded — demo mode (last harvest: {last_date.date()})")
# Ensure site column exists (required by coarsen_grid)
if "site" not in df_raw.columns:
    df_raw["site"] = site
# ── Last harvest stat cards ───────────────────────────────────────────────────
df_last  = df_raw[df_raw["harvest_date"] == last_date]
total_kg = df_last["weight_kg"].sum()
active   = (df_last["weight_kg"] > 0).sum()
mean_kg  = df_last.loc[df_last["weight_kg"] > 0, "weight_kg"].mean()
pct_zero = (df_last["weight_kg"] == 0).mean() * 100

st.markdown("### Last harvest summary")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total yield",       f"{total_kg:,.0f} kg")
c2.metric("Active cells",      f"{active:,}")
c3.metric("Mean yield / cell", f"{mean_kg:.3f} kg")
c4.metric("Zero cells",        f"{pct_zero:.1f}%")

# ── Run Stage 1 + Stage 2 (Method B) ─────────────────────────────────────────
with st.spinner("Running Stage 1 prediction and Stage 2 rule..."):

    # Stage 1: predict next harvest yield
    from harvest_advisor import _build_inference_row, _predict_yield
    inf_df     = _build_inference_row(
        df_raw, last_date,
        last_date + timedelta(days=1),
        weather, lag_depth=3, coarsen_n=coarsen_n)
    pred       = _predict_yield(inf_df, model_results)
    pred_total = float(pred.sum())

    # Stage 1 history totals for velocity calculation
    all_dates      = sorted(df_raw["harvest_date"].unique())
    past_dates     = [d for d in all_dates if pd.Timestamp(d) <= last_date]
    actual_yields  = df_raw.groupby("harvest_date")["weight_kg"].sum()

    if len(past_dates) >= 3:
        history_totals = [float(actual_yields.get(pd.Timestamp(d), 0))
                          for d in past_dates[-3:]]
    else:
        # Fallback: use last actual for simple growth rate only
        history_totals = [total_kg, total_kg, total_kg]

    # Stage 2: Method B decision
    rule     = ha.apply_rule_method_b(pred_total, history_totals, thresholds)
    opt_days = rule['rec_days']
    opt_date = last_date + timedelta(days=opt_days)
    gr_pred  = rule['gr_pred']
    velocity = rule['velocity_raw']

# ── Recommendation card ───────────────────────────────────────────────────────
st.markdown("---")
col_left, col_right = st.columns([1, 1.6])

with col_left:
    trend = ("rising 📈" if gr_pred >= thresholds['t_high'] else
             "declining 📉" if gr_pred < thresholds['t_low'] else
             "stable ➡")
    velocity_sign = "+" if velocity >= 0 else ""

    st.markdown(f"""
    <div class="rec-card">
        <div class="rec-title">Optimal harvest date</div>
        <div class="rec-date">{opt_date.strftime('%b %d, %Y')}</div>
        <div class="rec-sub">
            +{opt_days} days from last harvest ({last_date.date()})<br>
            Expected yield: <strong>{pred_total:,.0f} kg</strong><br>
            Growth rate: <strong>{gr_pred:.3f}</strong> — {trend}<br>
            Velocity: <strong>{velocity_sign}{velocity:.3f}</strong>
        </div>
        <div class="rule-badge">Stage 2: Method B (velocity-aware)</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="threshold-card">
        Stage 2 thresholds (Rule B from full training data)<br>
        t_high = {thresholds['t_high']:.3f} → wait {thresholds['days_map']['long']}d<br>
        t_low  = {thresholds['t_low']:.3f} → wait {thresholds['days_map']['short']}d<br>
        gr_pred = {gr_pred:.3f}  |  velocity = {velocity_sign}{velocity:.3f}
    </div>
    """, unsafe_allow_html=True)

    # Growth rate chart
    st.pyplot(build_growth_rate_fig(thresholds, gr_pred, velocity, opt_days),
              use_container_width=True)

with col_right:
    st.markdown("**Predicted yield map — optimal day (real kg)**")
    advice_for_map = {
        "inference_dfs":    {opt_days: inf_df},
        "yield_maps":       {opt_days: pred},
        "last_harvest_date": last_date,
        "optimal_days":     opt_days,
    }
    st.pyplot(build_predicted_map_fig(advice_for_map, opt_days),
              use_container_width=True)

# ── Last harvest yield map ────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### Last harvest yield map (actual)")
if len(df_last) > 0:
    st.pyplot(
        build_yield_map_fig(df_last,
                            title=f"{site}  —  {last_date.date()}  "
                                  f"(last harvest, actual yield)"),
        use_container_width=True)
else:
    st.warning("No grid data found for the last harvest date.")

# ── Forecast detail table ─────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### Forecast detail")
detail = pd.DataFrame([{
    "Last harvest date": str(last_date.date()),
    "Recommended date":  str(opt_date.date()),
    "Days to wait":      opt_days,
    "Predicted yield (kg)": f"{pred_total:,.0f}",
    "Growth rate (gr_pred)": f"{gr_pred:.4f}",
    "Velocity (raw)":    f"{velocity:+.4f}",
    "Trend":             trend,
    "t_low":             f"{thresholds['t_low']:.3f}",
    "t_high":            f"{thresholds['t_high']:.3f}",
}]).T.rename(columns={0: "Value"})
st.dataframe(detail, use_container_width=True)

# ── Historical harvest timeline ───────────────────────────────────────────────
st.markdown("---")
st.markdown("### Historical total yield per harvest")
hist = (df_raw.groupby("harvest_date")["weight_kg"]
        .sum().reset_index().sort_values("harvest_date"))
hist.columns = ["Harvest date", "Total yield (kg)"]

fig_hist, ax_hist = plt.subplots(figsize=(12, 3.5))
ax_hist.bar(range(len(hist)), hist["Total yield (kg)"],
            color="#5B8DB8", edgecolor="white", linewidth=0.5)
ax_hist.axvline(len(hist) - 1, color="#2d6a3f", lw=2, ls="--",
                label="Last harvest")
ax_hist.set_xticks(range(len(hist)))
ax_hist.set_xticklabels([str(d.date()) for d in hist["Harvest date"]],
                          rotation=30, ha="right", fontsize=8)
ax_hist.set_ylabel("Total yield (kg)")
ax_hist.set_title(f"{site} — Historical yield per harvest", fontweight="bold")
ax_hist.legend(fontsize=9)
ax_hist.grid(axis="y", alpha=0.3)
plt.tight_layout()
st.pyplot(fig_hist, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<small style='color:#999'>Strawberry Yield Harvest Advisor · "
    "Two-Stage Pipeline v4 · Grid 7×7 · Stage 2 Method B · 2024</small>",
    unsafe_allow_html=True)