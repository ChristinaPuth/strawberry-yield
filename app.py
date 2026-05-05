# cd "/Users/pusi/Library/CloudStorage/GoogleDrive-tnzhang@ucdavis.edu/我的云端硬盘/Spatio-Temporal Modeling of Grid-Level Crop Yield Using Weather and Historical Yield Data/Codes"



"""
app.py
------
Streamlit app for the Strawberry Yield Harvest Advisor.

Run locally:
    streamlit run app.py

Structure:
    Sidebar  : upload CSV, select site, set forecast window, run button
    Main     : stat cards → yield map → harvest recommendation → season trend
"""

import sys
import os
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import timedelta
import streamlit as st

# ── page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Harvest Advisor",
    page_icon="🍓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── inject custom CSS ─────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
h1, h2, h3 { font-family: 'DM Serif Display', serif; }

/* sidebar */
section[data-testid="stSidebar"] {
    background: #0f1f14;
}
section[data-testid="stSidebar"] * {
    color: #d4e8d0 !important;
}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stFileUploader label {
    color: #7fbf7f !important;
    font-size: 0.82rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* metric cards */
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

/* recommendation card */
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
.rec-sub {
    font-size: 0.9rem;
    color: #a8d5a8;
    margin-top: 0.4rem;
}
.warn-card {
    background: #fff8e6;
    border-left: 4px solid #e8a020;
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1rem;
    font-size: 0.88rem;
    color: #7a4f00;
    margin-top: 0.8rem;
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


# ── src path (works both locally and in Colab) ────────────────────────────────
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
    """Fetch or load cached weather."""
    cache_path = os.path.join(_HERE, "outputs", "processed_data",
                              f"weather_{site}.csv")
    if os.path.exists(cache_path):
        return pd.read_csv(cache_path, index_col=0, parse_dates=True)
    return fe.fetch_weather(site)


@st.cache_data(show_spinner="Training model...")
def load_model_results(site: str, feature_config: str):
    """Load pre-saved features and train model. Cached by site + config."""
    feat_path = os.path.join(_HERE, "outputs", "processed_data",
                             f"features_{site}.csv")
    if not os.path.exists(feat_path):
        return None, None
    df_feat = pd.read_csv(feat_path, parse_dates=["harvest_date"])
    df_feat = df_feat.loc[:, ~df_feat.columns.duplicated()]
    splits  = fe.split_data(df_feat, site)
    features = m.ABLATION_CONFIGS[feature_config]
    results  = m.run_model_comparison(splits, site, features)
    return results, splits


# ── helpers ───────────────────────────────────────────────────────────────────

def parse_uploaded_csv(uploaded_file) -> pd.DataFrame:
    """Parse a user-uploaded per-day yield CSV (no header)."""
    df = pd.read_csv(uploaded_file, header=None)
    if df.shape[1] == 6:
        df.columns = ["index", "field_x", "field_y",
                      "weight_kg", "easting", "northing"]
        df = df.drop(columns=["index"])
    elif df.shape[1] == 5:
        df.columns = ["field_x", "field_y",
                      "weight_kg", "easting", "northing"]
    else:
        st.error(f"Unexpected column count: {df.shape[1]}. Expected 5 or 6.")
        return None
    for col in ["field_x", "field_y", "weight_kg", "easting", "northing"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["field_x", "field_y", "weight_kg"])


def build_yield_map_fig(df_day: pd.DataFrame,
                         title: str,
                         cmap: str = "YlOrRd") -> plt.Figure:
    """Render a grid-index yield heatmap."""
    x_vals = sorted(df_day["field_x"].unique())
    y_vals = sorted(df_day["field_y"].unique())
    x2i    = {v: i for i, v in enumerate(x_vals)}
    y2i    = {v: i for i, v in enumerate(y_vals)}
    grid   = np.zeros((len(y_vals), len(x_vals)))
    for _, row in df_day.iterrows():
        grid[y2i[row["field_y"]], x2i[row["field_x"]]] = row["weight_kg"]

    vmax = float(np.quantile(grid[grid > 0], 0.99)) if (grid > 0).any() else 1.0
    fig, ax = plt.subplots(figsize=(7, 4.5))
    im = ax.imshow(grid, cmap=cmap, aspect="auto", vmin=0, vmax=vmax)
    plt.colorbar(im, ax=ax, label="Yield (kg)", shrink=0.85)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("field_x index", fontsize=9)
    ax.set_ylabel("field_y index", fontsize=9)
    total = df_day["weight_kg"].sum()
    ax.text(0.02, 0.97, f"Total: {total:,.0f} kg",
            transform=ax.transAxes, fontsize=9, va="top", color="white",
            bbox=dict(boxstyle="round,pad=0.3", fc="#333", alpha=0.65))
    plt.tight_layout()
    return fig


def build_forecast_bar_fig(summary: pd.DataFrame,
                            opt_day: int) -> plt.Figure:
    """Bar chart of adjusted yield vs candidate days."""
    colours = ["#2d6a3f" if d == opt_day else "#cbd5c0"
               for d in summary["days_ahead"]]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    bars = ax.bar(
        [f"+{d}d" for d in summary["days_ahead"]],
        summary["adj_yield"],
        color=colours, edgecolor="white", linewidth=0.5
    )
    ax.set_ylabel("Adj. predicted yield (kg)", fontsize=9)
    ax.set_title("Forecast comparison — candidate harvest days",
                 fontsize=10, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    for bar, (_, row) in zip(bars, summary.iterrows()):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + summary["adj_yield"].max() * 0.015,
                f"{row['adj_yield']:,.0f}",
                ha="center", va="bottom", fontsize=8.5,
                color="#2d6a3f" if row["days_ahead"] == opt_day else "#666")
    plt.tight_layout()
    return fig


def build_predicted_map_fig(advice: dict, k: int) -> plt.Figure:
    """Yield map for one candidate day from harvest_advisor output."""
    inf_df = advice["inference_dfs"][k]
    pred   = advice["yield_maps"][k]

    x_vals = sorted(inf_df["field_x"].unique())
    y_vals = sorted(inf_df["field_y"].unique())
    x2i    = {v: i for i, v in enumerate(x_vals)}
    y2i    = {v: i for i, v in enumerate(y_vals)}
    grid   = np.zeros((len(y_vals), len(x_vals)))
    for (_, row), p in zip(inf_df.iterrows(), pred):
        grid[y2i[row["field_y"]], x2i[row["field_x"]]] = p

    vmax = float(np.quantile(grid[grid > 0], 0.99)) if (grid > 0).any() else 1.0
    cdate = advice["last_harvest_date"] + timedelta(days=k)
    adj   = advice["summary_table"].loc[
        advice["summary_table"]["days_ahead"] == k, "adj_yield"
    ].values[0]

    fig, ax = plt.subplots(figsize=(5.5, 4))
    im = ax.imshow(grid, cmap="YlOrRd", aspect="auto", vmin=0, vmax=vmax)
    plt.colorbar(im, ax=ax, label="kg", shrink=0.8)
    marker = " ★" if k == advice["optimal_day"] else ""
    ax.set_title(f"+{k} days  ({cdate.date()}){marker}\n{adj:,.0f} kg",
                 fontsize=9,
                 fontweight="bold" if k == advice["optimal_day"] else "normal",
                 color="#2d6a3f" if k == advice["optimal_day"] else "black")
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

    # site selector
    site = st.selectbox("Site", ["SantaMaria", "Salinas"])

    # feature config
    feat_config = st.selectbox(
        "Feature config",
        ["A4", "A2", "A5", "A7"],
        help="A4 = spatio-temporal (recommended). A7 = all features."
    )

    # forecast window
    st.markdown("**Forecast window**")
    min_days = st.slider("Min days ahead", 1, 5, 3)
    max_days = st.slider("Max days ahead", min_days + 1, 14, 7)
    candidate_days = list(range(min_days, max_days + 1))

    # penalty
    penalty = st.slider("Over-ripening penalty", 0.0, 0.2, 0.05, 0.01,
                         help="Higher = more conservative, wait less")

    st.markdown("---")

    # CSV uploader
    st.markdown("**Upload latest harvest CSV**")
    st.markdown(
        "<small>Format: index, field_x, field_y, weight_kg, easting, northing"
        "</small>", unsafe_allow_html=True
    )
    uploaded = st.file_uploader("", type=["csv"], label_visibility="collapsed")

    # harvest date for uploaded file
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
    # landing state
    c1, c2, c3 = st.columns(3)
    c1.metric("Site", site)
    c2.metric("Forecast window", f"+{min_days} to +{max_days} days")
    c3.metric("Feature config", feat_config)
    st.info("Upload a harvest CSV and press **Run Analysis** to begin.")
    st.stop()

# ── on run ────────────────────────────────────────────────────────────────────

with st.spinner("Loading model and weather data..."):
    weather       = load_weather(site)
    model_results, splits = load_model_results(site, feat_config)

if model_results is None:
    st.error(
        f"Pre-trained feature file not found for {site}. "
        "Make sure `outputs/processed_data/features_{site}.csv` exists."
    )
    st.stop()

# load uploaded CSV or fall back to last test harvest
if uploaded is not None:
    df_upload = parse_uploaded_csv(uploaded)
    if df_upload is None:
        st.stop()
    df_upload["harvest_date"] = pd.Timestamp(upload_date)
    df_upload["site"]         = site
    last_date = pd.Timestamp(upload_date)

    # merge uploaded row into raw data for lag construction
    feat_path = os.path.join(_HERE, "outputs", "processed_data",
                              f"features_{site}.csv")
    df_feat_base = pd.read_csv(feat_path, parse_dates=["harvest_date"])
    df_feat_base = df_feat_base.loc[:, ~df_feat_base.columns.duplicated()]

    # reconstruct raw data from features file (use lag1 of first date as proxy)
    # For simplicity, use the underlying raw site data cached in outputs
    raw_path = os.path.join(_HERE, "outputs", "processed_data",
                             f"features_{site}.csv")
    df_raw_proxy = pd.read_csv(raw_path, parse_dates=["harvest_date"])[[
        "harvest_date","field_x","field_y","weight_kg","easting","northing"
    ]]
    # append uploaded harvest
    df_raw_proxy = pd.concat([df_raw_proxy, df_upload[
        ["harvest_date","field_x","field_y","weight_kg","easting","northing"]
    ]], ignore_index=True)
else:
    # demo mode: use last harvest in the dataset
    feat_path = os.path.join(_HERE, "outputs", "processed_data",
                              f"features_{site}.csv")
    df_feat_base = pd.read_csv(feat_path, parse_dates=["harvest_date"])
    df_feat_base = df_feat_base.loc[:, ~df_feat_base.columns.duplicated()]
    df_raw_proxy = df_feat_base[[
        "harvest_date","field_x","field_y","weight_kg","easting","northing"
    ]].copy()
    last_date = df_feat_base["harvest_date"].max()
    st.info(f"No CSV uploaded — using demo mode (last harvest: {last_date.date()})")


# ── compute latest yield stats ────────────────────────────────────────────────
df_last = df_raw_proxy[df_raw_proxy["harvest_date"] == last_date]

total_kg   = df_last["weight_kg"].sum()
active     = (df_last["weight_kg"] > 0).sum()
mean_kg    = df_last.loc[df_last["weight_kg"] > 0, "weight_kg"].mean()
pct_zero   = (df_last["weight_kg"] == 0).mean() * 100

# ── run harvest advisor ───────────────────────────────────────────────────────
with st.spinner("Computing harvest forecast..."):
    advice = ha.recommend_harvest(
        model_results     = model_results,
        df_raw            = df_raw_proxy,
        weather           = weather,
        site              = site,
        last_harvest_date = last_date,
        candidate_days    = candidate_days,
        penalty_weight    = penalty,
    )

opt_date = advice["optimal_date"]
opt_day  = advice["optimal_day"]
opt_row  = advice["summary_table"][
    advice["summary_table"]["days_ahead"] == opt_day
].iloc[0]


# ════════════════════════════════════════════
# STAT CARDS
# ════════════════════════════════════════════

st.markdown("### Last harvest summary")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total yield",      f"{total_kg:,.0f} kg")
c2.metric("Active cells",     f"{active:,}")
c3.metric("Mean yield / cell",f"{mean_kg:.3f} kg")
c4.metric("Zero cells",       f"{pct_zero:.1f}%")


# ════════════════════════════════════════════
# RECOMMENDATION + YIELD MAP
# ════════════════════════════════════════════

st.markdown("---")
col_left, col_right = st.columns([1, 1.6])

with col_left:
    st.markdown(f"""
    <div class="rec-card">
        <div class="rec-title">Optimal harvest date</div>
        <div class="rec-date">{opt_date.strftime('%b %d, %Y')}</div>
        <div class="rec-sub">
            +{opt_day} days from last harvest<br>
            Expected yield: <strong>{opt_row['total_yield']:,.0f} kg</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if opt_row["risk_cells"] > 0:
        st.markdown(f"""
        <div class="warn-card">
            ⚠️ <strong>{opt_row['risk_cells']:,} cells</strong> at over-ripening
            risk — consider harvesting sooner if field conditions allow.
        </div>
        """, unsafe_allow_html=True)

    # forecast bar chart
    st.pyplot(build_forecast_bar_fig(advice["summary_table"], opt_day),
              use_container_width=True)

with col_right:
    st.markdown("**Predicted yield map — optimal day**")
    st.pyplot(build_predicted_map_fig(advice, opt_day),
              use_container_width=True)


# ════════════════════════════════════════════
# LAST HARVEST MAP
# ════════════════════════════════════════════

st.markdown("---")
st.markdown("### Last harvest yield map")
if len(df_last) > 0:
    fig_last = build_yield_map_fig(
        df_last,
        title=f"{site}  —  {last_date.date()}  (last harvest)"
    )
    st.pyplot(fig_last, use_container_width=True)
else:
    st.warning("No grid data available for the last harvest date.")


# ════════════════════════════════════════════
# ALL CANDIDATE MAPS
# ════════════════════════════════════════════

st.markdown("---")
st.markdown("### All candidate harvest day maps")

map_cols = st.columns(len(candidate_days))
for col, k in zip(map_cols, candidate_days):
    with col:
        fig = build_predicted_map_fig(advice, k)
        st.pyplot(fig, use_container_width=True)


# ════════════════════════════════════════════
# FORECAST TABLE
# ════════════════════════════════════════════

st.markdown("---")
st.markdown("### Forecast detail")
tbl = advice["summary_table"].copy()
tbl["date"] = tbl["date"].dt.strftime("%Y-%m-%d")
tbl.columns = ["Days ahead", "Date", "Total yield (kg)",
                "Adj. yield (kg)", "Risk cells",
                "Mean yield/cell", "Max yield/cell"]
st.dataframe(tbl.set_index("Days ahead"), use_container_width=True)


# ════════════════════════════════════════════
# FOOTER
# ════════════════════════════════════════════

st.markdown("---")
st.markdown(
    "<small style='color:#999'>Strawberry Yield Harvest Advisor · "
    "Spatio-Temporal Modeling · 2024</small>",
    unsafe_allow_html=True
)