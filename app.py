"""
app.py  v3
----------
Streamlit app for the Strawberry Yield Harvest Advisor.

Run locally:
    streamlit run app.py

Two-stage pipeline:
    Stage 1 : LightGBM predicts yield map (weight_kg per cell)
    Stage 2 : Growth-rate rules → optimal harvest interval (days)

Structure:
    Sidebar  : upload CSV, select site, set forecast window, run button
    Main     : stat cards → yield map → harvest recommendation → DQ info
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta
import streamlit as st

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Harvest Advisor",
    page_icon="🍓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1, h2, h3 { font-family: 'DM Serif Display', serif; }

section[data-testid="stSidebar"] { background: #0f1f14; }
section[data-testid="stSidebar"] * { color: #d4e8d0 !important; }
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stFileUploader label {
    color: #7fbf7f !important;
    font-size: 0.82rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

div[data-testid="metric-container"] {
    background: #f7f5f0;
    border: 1px solid #e0dbd0;
    border-radius: 10px;
    padding: 1rem 1.2rem;
}
div[data-testid="metric-container"] label {
    color: #6b6558 !important;
    font-size: 0.78rem !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
    font-family: 'DM Serif Display', serif;
    font-size: 2rem !important;
    color: #1a2e1f !important;
}

.rec-card {
    background: linear-gradient(135deg, #1a3320 0%, #2d5a3d 100%);
    border-radius: 14px;
    padding: 1.6rem 2rem;
    color: white;
    margin-bottom: 1rem;
}
.rec-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.1rem;
    color: #7fbf7f;
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
.rec-sub { font-size: 0.9rem; color: #a8d5a8; margin-top: 0.4rem; }

.threshold-card {
    background: #f0f7f0;
    border-left: 4px solid #2d6a3f;
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1rem;
    font-size: 0.88rem;
    color: #1a3320;
    margin-top: 0.8rem;
    font-family: monospace;
}
.info-banner {
    background: #f0f7ff;
    border-left: 4px solid #3b82f6;
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1rem;
    font-size: 0.88rem;
    color: #1e3a5f;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ── src path ──────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC  = os.path.join(_HERE, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import data_pipeline
import feature_engineering as fe
import models as m
import harvest_advisor as ha


# ── cached loaders ────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading weather data...")
def load_weather(site: str) -> pd.DataFrame:
    cache_path = os.path.join(_HERE, "outputs", "processed_data",
                              f"weather_{site}.csv")
    if os.path.exists(cache_path):
        return pd.read_csv(cache_path, index_col=0, parse_dates=True)
    return fe.fetch_weather(site)


@st.cache_data(show_spinner="Loading features and training model...")
def load_model_and_splits(site: str, feature_config: str):
    """
    Load v3 features, split data, train Stage 1 model.
    Returns (model_results_df, splits, thresholds) or (None, None, None).
    """
    feat_path = os.path.join(_HERE, "outputs", "processed_data",
                             f"features_{site}_v3.csv")
    if not os.path.exists(feat_path):
        return None, None, None

    df_feat = pd.read_csv(feat_path, parse_dates=["harvest_date"])
    df_feat = df_feat.loc[:, ~df_feat.columns.duplicated()]

    splits   = fe.split_data(df_feat, site)
    features = m.ABLATION_CONFIGS[feature_config]

    # Stage 1: model comparison → returns DataFrame
    model_results = m.run_model_comparison(splits, site, features)

    # Stage 2: derive thresholds from training data
    thresholds = ha.derive_thresholds(splits["train"], site=site)

    return model_results, splits, thresholds


# ── helpers ───────────────────────────────────────────────────────────────────

def parse_uploaded_csv(uploaded_file) -> pd.DataFrame:
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


def build_forecast_bar_fig(summary: pd.DataFrame,
                            opt_days: int) -> plt.Figure:
    """Bar chart of predicted total yield vs candidate days."""
    colours = ["#2d6a3f" if d == opt_days else "#cbd5c0"
               for d in summary["days_ahead"]]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    bars = ax.bar(
        [f"+{d}d" for d in summary["days_ahead"]],
        summary["pred_total"],           # v3 uses pred_total
        color=colours, edgecolor="white", linewidth=0.5
    )
    ax.set_ylabel("Predicted total yield (kg)", fontsize=9)
    ax.set_title("Stage 1: Predicted yield per candidate day",
                 fontsize=10, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    for bar, (_, row) in zip(bars, summary.iterrows()):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + summary["pred_total"].max()*0.015,
                f"{row['pred_total']:,.0f}\ngr={row['growth_rate']:.2f}",
                ha="center", va="bottom", fontsize=7.5,
                color="#2d6a3f" if row["days_ahead"]==opt_days else "#666")
    plt.tight_layout()
    return fig


def build_growth_rate_fig(summary: pd.DataFrame,
                           thresholds: dict,
                           opt_days: int) -> plt.Figure:
    """Line chart of growth rate vs candidate days with threshold lines."""
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.plot(summary["days_ahead"], summary["growth_rate"],
            "o-", color="#5B8DB8", linewidth=2, markersize=8)
    ax.axhline(thresholds["t_high"], color="#E07B39", linestyle="--",
               linewidth=1.5,
               label=f"t_high={thresholds['t_high']:.3f} (wait longer)")
    ax.axhline(thresholds["t_low"], color="#c0392b", linestyle="--",
               linewidth=1.5,
               label=f"t_low={thresholds['t_low']:.3f} (harvest soon)")
    ax.axvline(opt_days, color="#2d6a3f", linewidth=2, alpha=0.4)
    ax.fill_between(summary["days_ahead"],
                    thresholds["t_low"], thresholds["t_high"],
                    alpha=0.08, color="#5B8DB8", label="Stable zone")
    ax.set_xlabel("Days ahead", fontsize=9)
    ax.set_ylabel("Growth rate", fontsize=9)
    ax.set_title("Stage 2: Growth rate rule",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_xticks(summary["days_ahead"])
    plt.tight_layout()
    return fig


def build_predicted_map_fig(advice: dict, k: int) -> plt.Figure:
    """Yield map for one candidate day."""
    inf_df = advice["inference_dfs"][k]
    pred   = advice["yield_maps"][k]
    x_vals = sorted(inf_df["field_x"].unique())
    y_vals = sorted(inf_df["field_y"].unique())
    x2i = {v: i for i, v in enumerate(x_vals)}
    y2i = {v: i for i, v in enumerate(y_vals)}
    grid = np.zeros((len(y_vals), len(x_vals)))
    for (_, row), p in zip(inf_df.iterrows(), pred):
        grid[y2i[row["field_y"]], x2i[row["field_x"]]] = p
    vmax = float(np.quantile(grid[grid>0], 0.99)) if (grid>0).any() else 1.0
    cdate = advice["last_harvest_date"] + timedelta(days=k)

    # pred_total from summary table
    row_summary = advice["summary_table"][
        advice["summary_table"]["days_ahead"] == k
    ]
    pred_total = row_summary["pred_total"].values[0] if len(row_summary) > 0 else 0

    fig, ax = plt.subplots(figsize=(5.5, 4))
    im = ax.imshow(grid, cmap="YlOrRd", aspect="auto", vmin=0, vmax=vmax)
    plt.colorbar(im, ax=ax, label="kg", shrink=0.8)
    marker = " ★ OPTIMAL" if k == advice["optimal_days"] else ""
    ax.set_title(
        f"+{k} days  ({cdate.date()}){marker}\n{pred_total:,.0f} kg",
        fontsize=9,
        fontweight="bold" if k == advice["optimal_days"] else "normal",
        color="#2d6a3f" if k == advice["optimal_days"] else "black"
    )
    ax.set_xlabel("field_x", fontsize=8)
    ax.set_ylabel("field_y", fontsize=8)
    ax.tick_params(labelsize=7)
    plt.tight_layout()
    return fig


# ════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🍓 Harvest Advisor")
    st.markdown("---")

    site = st.selectbox("Site", ["SantaMaria", "Salinas"])

    feat_config = st.selectbox(
        "Feature config",
        ["A4", "A2", "A5", "A6"],
        help="A4 = spatio-temporal (recommended). A6 = all 20 features."
    )

    st.markdown("**Forecast window**")
    min_days = st.slider("Min days ahead", 1, 5, 3)
    max_days = st.slider("Max days ahead", min_days+1, 14, 7)
    candidate_days = list(range(min_days, max_days+1))

    st.markdown("---")
    st.markdown("**Upload latest harvest CSV**")
    st.markdown(
        "<small>Format: index, field_x, field_y, weight_kg, easting, northing</small>",
        unsafe_allow_html=True
    )
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
    unsafe_allow_html=True
)

if not run_btn:
    c1, c2, c3 = st.columns(3)
    c1.metric("Site", site)
    c2.metric("Forecast window", f"+{min_days} to +{max_days} days")
    c3.metric("Feature config", feat_config)
    st.info("Upload a harvest CSV and press **Run Analysis** to begin.")
    st.stop()


# ── Load model ────────────────────────────────────────────────────────────────

with st.spinner("Loading model and weather data..."):
    weather = load_weather(site)
    model_results, splits, thresholds = load_model_and_splits(site, feat_config)

if model_results is None:
    st.error(
        f"Feature file not found for {site}. "
        f"Make sure `outputs/processed_data/features_{site}_v3.csv` exists."
    )
    st.stop()


# ── Load raw data proxy ───────────────────────────────────────────────────────

feat_path = os.path.join(_HERE, "outputs", "processed_data",
                          f"features_{site}_v3.csv")
df_feat_base = pd.read_csv(feat_path, parse_dates=["harvest_date"])
df_feat_base = df_feat_base.loc[:, ~df_feat_base.columns.duplicated()]
df_raw_proxy = df_feat_base[[
    "harvest_date","field_x","field_y","weight_kg","easting","northing"
]].copy()

if uploaded is not None:
    df_upload = parse_uploaded_csv(uploaded)
    if df_upload is None:
        st.stop()
    df_upload["harvest_date"] = pd.Timestamp(upload_date)
    df_upload["site"]         = site
    last_date = pd.Timestamp(upload_date)
    df_raw_proxy = pd.concat([
        df_raw_proxy,
        df_upload[["harvest_date","field_x","field_y",
                   "weight_kg","easting","northing"]]
    ], ignore_index=True)
else:
    last_date = df_feat_base["harvest_date"].max()
    st.info(f"No CSV uploaded — demo mode (last harvest: {last_date.date()})")


# ── Stat cards ────────────────────────────────────────────────────────────────

df_last  = df_raw_proxy[df_raw_proxy["harvest_date"] == last_date]
total_kg = df_last["weight_kg"].sum()
active   = (df_last["weight_kg"] > 0).sum()
mean_kg  = df_last.loc[df_last["weight_kg"]>0, "weight_kg"].mean()
pct_zero = (df_last["weight_kg"] == 0).mean() * 100

st.markdown("### Last harvest summary")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total yield",       f"{total_kg:,.0f} kg")
c2.metric("Active cells",      f"{active:,}")
c3.metric("Mean yield / cell", f"{mean_kg:.3f} kg")
c4.metric("Zero cells",        f"{pct_zero:.1f}%")


# ── Run harvest advisor ───────────────────────────────────────────────────────

with st.spinner("Running two-stage harvest advisor..."):
    advice = ha.recommend_harvest(
        model_results     = model_results,
        df_raw            = df_raw_proxy,
        weather           = weather,
        site              = site,
        last_harvest_date = last_date,
        thresholds        = thresholds,       # Stage 2 rule thresholds
        candidate_days    = candidate_days,
    )

opt_days  = advice["optimal_days"]            # v3 key name
opt_date  = advice["optimal_date"]
summary   = advice["summary_table"]
opt_row   = summary[summary["days_ahead"] == opt_days].iloc[0]


# ── Recommendation + maps ─────────────────────────────────────────────────────

st.markdown("---")
col_left, col_right = st.columns([1, 1.6])

with col_left:
    st.markdown(f"""
    <div class="rec-card">
        <div class="rec-title">Optimal harvest date</div>
        <div class="rec-date">{opt_date.strftime('%b %d, %Y')}</div>
        <div class="rec-sub">
            +{opt_days} days from last harvest<br>
            Expected yield: <strong>{opt_row['pred_total']:,.0f} kg</strong><br>
            Growth rate: <strong>{opt_row['growth_rate']:.3f}</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Threshold info card
    st.markdown(f"""
    <div class="threshold-card">
        Stage 2 thresholds (from training data)<br>
        t_high = {thresholds['t_high']:.3f} → wait longer<br>
        t_low  = {thresholds['t_low']:.3f} → harvest soon<br>
        growth_rate = {opt_row['growth_rate']:.3f}
    </div>
    """, unsafe_allow_html=True)

    # Forecast bar chart
    st.pyplot(build_forecast_bar_fig(summary, opt_days),
              use_container_width=True)

    # Growth rate chart
    st.pyplot(build_growth_rate_fig(summary, thresholds, opt_days),
              use_container_width=True)

with col_right:
    st.markdown("**Predicted yield map — optimal day**")
    st.pyplot(build_predicted_map_fig(advice, opt_days),
              use_container_width=True)


# ── Last harvest map ──────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("### Last harvest yield map")
if len(df_last) > 0:
    st.pyplot(
        build_yield_map_fig(df_last,
                            title=f"{site}  —  {last_date.date()}  (last harvest)"),
        use_container_width=True
    )
else:
    st.warning("No grid data for the last harvest date.")


# ── All candidate maps ────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("### All candidate harvest day maps")
map_cols = st.columns(len(candidate_days))
for col, k in zip(map_cols, candidate_days):
    with col:
        st.pyplot(build_predicted_map_fig(advice, k), use_container_width=True)


# ── Forecast table ────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("### Forecast detail")
tbl = summary.copy()
tbl["date"] = tbl["date"].dt.strftime("%Y-%m-%d")
tbl = tbl.rename(columns={
    "days_ahead":  "Days ahead",
    "date":        "Date",
    "pred_total":  "Predicted yield (kg)",
    "growth_rate": "Growth rate",
    "stage2_days": "Stage 2 recommendation",
})
st.dataframe(tbl.set_index("Days ahead"), use_container_width=True)


# ── Footer ────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown(
    "<small style='color:#999'>Strawberry Yield Harvest Advisor · "
    "Two-Stage Pipeline v3 · Spatio-Temporal Modeling · 2024</small>",
    unsafe_allow_html=True
)