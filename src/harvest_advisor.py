

"""
harvest_advisor.py  v3
----------------------
Stage 2: Rule-based optimal harvest timing advisor.

Pipeline:
  Stage 1 (models.py) → predicted yield map per candidate day
  Stage 2 (this file) → growth-rate rules → optimal_days recommendation

Key changes from v2:
  - No trained model for days prediction
  - Thresholds derived from training data statistics
  - Decision Quality (DQ) calculation added
  - Days map visualization added

Usage:
    # 1. Derive thresholds from training data (once)
    thresholds = derive_thresholds(splits_sm["train"], site="SantaMaria")

    # 2. Get recommendation
    advice = recommend_harvest(
        model_results     = model_results_sm,
        df_raw            = df_sm,
        weather           = weather_sm,
        site              = "SantaMaria",
        last_harvest_date = pd.Timestamp("2024-07-09"),
        thresholds        = thresholds,
    )

    # 3. Evaluate Decision Quality (test set only)
    dq = decision_quality(advice, df_raw=df_sm, splits=splits_sm)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import timedelta
import warnings
warnings.filterwarnings("ignore")

from feature_engineering import ALL_FEATS, TEMPORAL_FEATS, SPATIAL_FEATS, WEATHER_FEATS


# ── 1. Derive growth-rate thresholds from training data ───────────────────────

def derive_thresholds(train_df: pd.DataFrame, site: str) -> dict:
    """
    Derive growth-rate thresholds for Stage 2 rules from training data.

    Method:
      1. Aggregate training data to field level (total yield per harvest)
      2. Compute growth_rate = this_total / last_total
      3. Group by days_since_last interval
      4. Find the growth_rate percentiles that best separate intervals

    Returns dict with threshold values and summary statistics.
    """
    # Field-level aggregation
    field = train_df.groupby("harvest_date").agg(
        total_kg      = ("weight_kg", "sum"),
        days_interval = ("days_since_last", "first"),
    ).dropna(subset=["days_interval"]).reset_index()

    field = field.sort_values("harvest_date")
    field["growth_rate"] = field["total_kg"] / field["total_kg"].shift(1)
    field = field.dropna(subset=["growth_rate"])
    field["days_interval"] = field["days_interval"].astype(int)

    # Statistics per interval
    stats = field.groupby("days_interval").agg(
        count        = ("growth_rate", "count"),
        mean_growth  = ("growth_rate", "mean"),
        median_growth= ("growth_rate", "median"),
        min_growth   = ("growth_rate", "min"),
        max_growth   = ("growth_rate", "max"),
    ).reset_index()

    print(f"\nStage 2 threshold derivation — {site}")
    print(f"Training harvests: {len(field)}")
    print(f"\nGrowth rate by interval:")
    print(stats.to_string(index=False))

    # Derive thresholds
    # Logic: if growth_rate is HIGH → yield still rising → wait longer
    #        if growth_rate is LOW  → yield declining   → harvest soon
    intervals = sorted(field["days_interval"].unique())

    if len(intervals) >= 3:
        # Three-class: short / medium / long
        # Threshold between short and medium: median growth of shortest interval
        short_int  = intervals[0]
        long_int   = intervals[-1]
        t_low  = float(field[field["days_interval"]==short_int]["growth_rate"].median())
        t_high = float(field[field["days_interval"]==long_int]["growth_rate"].median())
        classes = {intervals[0]: "short", intervals[1]: "medium", intervals[-1]: "long"}
    elif len(intervals) == 2:
        t_low  = float(field["growth_rate"].quantile(0.40))
        t_high = float(field["growth_rate"].quantile(0.60))
        classes = {intervals[0]: "short", intervals[-1]: "long"}
    else:
        t_low  = 0.9
        t_high = 1.1
        classes = {}

    days_map = {
        "short":  int(intervals[0]),
        "medium": int(intervals[len(intervals)//2]) if len(intervals) >= 3 else int(intervals[0]),
        "long":   int(intervals[-1]),
    }

    thresholds = {
        "t_low":      round(t_low,  4),
        "t_high":     round(t_high, 4),
        "days_map":   days_map,
        "intervals":  intervals,
        "stats":      stats,
        "site":       site,
    }

    print(f"\nDerived thresholds:")
    print(f"  growth_rate >= {t_high:.3f} → wait {days_map['long']} days  (yield rising)")
    print(f"  growth_rate >= {t_low:.3f}  → wait {days_map.get('medium', days_map['short'])} days  (yield stable)")
    print(f"  growth_rate <  {t_low:.3f}  → wait {days_map['short']} days  (yield declining)")

    return thresholds


# ── 2. Build inference features for a candidate harvest day ───────────────────

def _build_inference_row(df_raw: pd.DataFrame,
                          last_date: pd.Timestamp,
                          candidate_date: pd.Timestamp,
                          weather: pd.DataFrame,
                          lag_depth: int = 3,
                          coarsen_n: int = 1) -> pd.DataFrame:
    """
    Build feature DataFrame for all grid cells for a candidate date.

    Parameters
    ----------
    coarsen_n : grid coarsening factor used during training (1 = no coarsening).
                Must match the value used in fe.coarsen_grid() when building
                the training features.  If > 1, df_raw is coarsened before
                building lag / neighbour features so the inference grid matches
                the trained model's input space.
    """
    # Coarsen if needed — must match training
    if coarsen_n > 1:
        import feature_engineering as _fe
        df_raw = _fe.coarsen_grid(df_raw, n=coarsen_n)

    all_dates  = sorted(df_raw["harvest_date"].unique())
    past_dates = [d for d in all_dates if d <= last_date]

    if len(past_dates) < lag_depth:
        raise ValueError(
            f"Need {lag_depth} past harvests, only {len(past_dates)} available.")

    lag_dates = past_dates[-lag_depth:]

    pivot = df_raw[df_raw["harvest_date"].isin(lag_dates)].pivot_table(
        index=["field_x","field_y","easting","northing"],
        columns="harvest_date", values="weight_kg", aggfunc="first",
    ).reset_index()
    pivot.columns.name = None

    for k, d in enumerate(reversed(lag_dates), start=1):
        if d in pivot.columns:
            pivot.rename(columns={d: f"yield_lag{k}"}, inplace=True)
    for k in range(1, lag_depth+1):
        if f"yield_lag{k}" not in pivot.columns:
            pivot[f"yield_lag{k}"] = 0.0

    pivot["rolling_mean_3"] = (
        pivot["yield_lag1"]+pivot["yield_lag2"]+pivot["yield_lag3"]) / 3.0
    pivot["yield_trend"] = (pivot["yield_lag1"] - pivot["yield_lag3"]) / 2.0

    all_past = df_raw[df_raw["harvest_date"] <= last_date]
    cum = all_past.groupby(["field_x","field_y"])["weight_kg"].sum().reset_index()
    cum.columns = ["field_x","field_y","season_cumulative"]
    pivot = pivot.merge(cum, on=["field_x","field_y"], how="left")
    pivot["season_cumulative"] = pivot["season_cumulative"].fillna(0)
    pivot["day_of_year"] = candidate_date.dayofyear
    # days_since_last: how many days since last harvest (= the interval
    # the model is being asked to predict for). This is what makes
    # different candidate dates produce different predictions.
    pivot["days_since_last"] = float((candidate_date - last_date).days)

    # Neighbour means from lag1
    pivot["neighbor_mean_3x3"] = _neighbor_means_from_df(pivot, "yield_lag1", 3)
    pivot["neighbor_mean_5x5"] = _neighbor_means_from_df(pivot, "yield_lag1", 5)

    # Weather
    w = weather[(weather.index >= candidate_date - timedelta(days=7)) &
                (weather.index <= candidate_date - timedelta(days=1))]
    if len(w) == 0:
        for col in ["temp_mean_7d","temp_max_7d","temp_min_7d","precip_7d",
                    "et0_7d","humidity_mean_7d","soil_moisture_0_7",
                    "soil_moisture_7_28","daylight_7d"]:
            pivot[col] = np.nan
    else:
        pivot["temp_mean_7d"]       = float(w["temp_mean"].mean())
        pivot["temp_max_7d"]        = float(w["temp_max"].max())
        pivot["temp_min_7d"]        = float(w["temp_min"].min())
        pivot["precip_7d"]          = float(w["precip"].sum())
        pivot["et0_7d"]             = float(w["et0"].sum())
        pivot["humidity_mean_7d"]   = float(w["humidity_mean"].mean())
        pivot["soil_moisture_0_7"]  = float(w["soil_moisture_0_7"].mean())
        pivot["soil_moisture_7_28"] = float(w["soil_moisture_7_28"].mean())
        pivot["daylight_7d"]        = float(w["daylight_hours"].mean())

    pivot["harvest_date"] = candidate_date
    return pivot


def _neighbor_means_from_df(df, col, window=3):
    half = window // 2
    x_vals = sorted(df["field_x"].unique())
    y_vals = sorted(df["field_y"].unique())
    x2i = {v: i for i, v in enumerate(x_vals)}
    y2i = {v: i for i, v in enumerate(y_vals)}
    i2x = {i: v for v, i in x2i.items()}
    i2y = {i: v for v, i in y2i.items()}
    lookup = df.set_index(["field_x","field_y"])[col].to_dict()
    result = {}
    for idx, row in df.iterrows():
        xi, yi = x2i[row["field_x"]], y2i[row["field_y"]]
        vals = []
        for dx in range(-half, half+1):
            for dy in range(-half, half+1):
                if dx == 0 and dy == 0: continue
                nx, ny = xi+dx, yi+dy
                if nx in i2x and ny in i2y:
                    v = lookup.get((i2x[nx], i2y[ny]))
                    if v is not None: vals.append(v)
        result[idx] = float(np.mean(vals)) if vals else 0.0
    return pd.Series(result)


# ── 3. Predict yield for one candidate day ────────────────────────────────────

def _predict_yield(inf_df: pd.DataFrame,
                    model_results: pd.DataFrame) -> np.ndarray:
    best  = model_results.iloc[0]
    model = best["_model_obj"]
    feats = best["features"]
    log_t = best["log_target"]
    avail = [f for f in feats if f in inf_df.columns]
    X     = inf_df[avail].values.astype(np.float32)
    pred  = model.predict(X)
    if log_t: pred = np.expm1(pred)
    return np.clip(pred, 0, None)


# ── 4. Stage 2: Growth-rate rule ──────────────────────────────────────────────

def _apply_rule(growth_rate: float, thresholds: dict) -> int:
    """
    Apply growth-rate rule to determine optimal days.

    growth_rate = predicted_total / last_actual_total

    >= t_high → yield still rising  → wait longer
    >= t_low  → yield stable        → wait medium
    <  t_low  → yield declining     → harvest soon
    """
    days_map = thresholds["days_map"]
    t_high   = thresholds["t_high"]
    t_low    = thresholds["t_low"]

    if growth_rate >= t_high:
        return days_map["long"]
    elif growth_rate >= t_low:
        return days_map.get("medium", days_map["short"])
    else:
        return days_map["short"]


# ── 5. Main recommendation function ──────────────────────────────────────────

def recommend_harvest(model_results: pd.DataFrame,
                       df_raw: pd.DataFrame,
                       weather: pd.DataFrame,
                       site: str,
                       last_harvest_date,
                       thresholds: dict,
                       candidate_days: list = None,
                       lag_depth: int = 3,
                       coarsen_n: int = 1) -> dict:
    """
    Two-stage harvest recommendation.

    Stage 1: Predict yield for each candidate day
    Stage 2: Apply growth-rate rule to select optimal day

    Parameters
    ----------
    model_results     : output of models.run_model_comparison()
    df_raw            : raw DataFrame from data_pipeline (always original 1x1).
                        Coarsening happens internally; last_actual_total is
                        always computed at true field scale.
    weather           : weather DataFrame from feature_engineering
    site              : 'SantaMaria' or 'Salinas'
    last_harvest_date : date of most recent actual harvest
    thresholds        : output of derive_thresholds()
    candidate_days    : days ahead to evaluate (default [3,4,5,6,7])
    coarsen_n         : grid coarsening factor used during training (1=none).
                        Pass 3 if model was trained on 3x3 super-cells.

    Returns
    -------
    dict with full advice including yield maps for all candidates
    """
    last_harvest_date = pd.Timestamp(last_harvest_date)

    # ── Stage 1: 只预测一次 ───────────────────────────────────────────────────
    # 用上次采摘日期作为参考点构建特征，预测"下次采摘"的产量
    # candidate_date 设为 last_harvest_date + 1 只是为了构建特征，
    # days_since_last 会在 Stage 2 里通过阈值决定
    inf_df = _build_inference_row(
        df_raw, last_harvest_date,
        last_harvest_date + timedelta(days=1),   # placeholder，不影响空间特征
        weather, lag_depth, coarsen_n)
    pred        = _predict_yield(inf_df, model_results)
    pred_total  = float(pred.sum())

    # Last actual total yield
    last_actual_total = df_raw[
        df_raw["harvest_date"] == last_harvest_date
    ]["weight_kg"].sum()

    # ── Stage 2: 算增长率 → 查阈值 → 决定等几天 ──────────────────────────────
    growth_rate  = pred_total / last_actual_total if last_actual_total > 0 else 1.0
    optimal_days = _apply_rule(growth_rate, thresholds)
    optimal_date = last_harvest_date + timedelta(days=optimal_days)

    print(f"\nHarvest Advisor — {site}")
    print(f"Last harvest    : {last_harvest_date.date()}")
    print(f"Last yield      : {last_actual_total:,.0f} kg")
    print(f"Thresholds      : t_low={thresholds['t_low']:.3f}  "
          f"t_high={thresholds['t_high']:.3f}")
    print(f"\n  Predicted next yield : {pred_total:,.0f} kg")
    print(f"  Growth rate          : {growth_rate:.3f}")
    print(f"  ({'rising' if growth_rate >= thresholds['t_high'] else 'declining' if growth_rate < thresholds['t_low'] else 'stable'})")
    print(f"\n{'='*60}")
    print(f"  STAGE 2 RECOMMENDATION : {site}")
    print(f"  Last harvest  : {last_harvest_date.date()}")
    print(f"  Optimal date  : {optimal_date.date()}")
    print(f"  Days to wait  : {optimal_days} days")
    print(f"  Growth rate   : {growth_rate:.3f}")
    print(f"  Expected yield: {pred_total:,.0f} kg")
    print(f"{'='*60}\n")

    # summary_table 保留单行，兼容下游 decision_quality 等函数
    summary = pd.DataFrame([{
        "days_ahead":  optimal_days,
        "date":        optimal_date,
        "pred_total":  round(pred_total, 1),
        "growth_rate": round(growth_rate, 4),
        "stage2_days": optimal_days,
    }])

    return {
        "optimal_days":       optimal_days,
        "optimal_date":       optimal_date,
        "pred_yield":         pred,
        "summary_table":      summary,
        "yield_maps":         {optimal_days: pred},
        "inference_dfs":      {optimal_days: inf_df},
        "thresholds":         thresholds,
        "last_harvest_date":  last_harvest_date,
        "last_actual_total":  last_actual_total,
        "site":               site,
    }


# ── 6. Decision Quality ───────────────────────────────────────────────────────

def decision_quality(advice: dict,
                      df_raw: pd.DataFrame,
                      splits: dict,
                      tolerance_days: int = 2) -> pd.DataFrame:
    """
    Compare model recommendation vs farmer's actual decision.

    For each test harvest:
      model_yield  = actual yield on the day closest to model recommendation
      farmer_yield = actual yield on the day the farmer actually harvested
      DQ           = model_yield - farmer_yield

    Positive DQ → model recommendation is better than farmer's decision.
    Negative DQ → farmer's decision is better.

    Parameters
    ----------
    advice         : output of recommend_harvest()
    df_raw         : raw DataFrame (has actual yields for all dates)
    splits         : output of fe.split_data()
    tolerance_days : max days difference when matching model date to actual date

    Returns
    -------
    DataFrame with DQ per test harvest date
    """
    test_df    = splits["test"]
    site       = advice["site"]
    last_date  = advice["last_harvest_date"]
    opt_days   = advice["optimal_days"]
    model_date = advice["optimal_date"]

    # All actual harvest dates
    actual_dates = sorted(df_raw["harvest_date"].unique())

    # Field-level actual yields per date
    actual_yields = df_raw.groupby("harvest_date")["weight_kg"].sum()

    # Test harvest dates (farmer's actual decisions after last_date)
    test_dates = sorted(
        test_df["harvest_date"].unique()
    )
    test_dates = [d for d in test_dates if d > last_date]

    if not test_dates:
        print("No test dates after last_harvest_date.")
        return pd.DataFrame()

    records = []
    for farmer_date in test_dates:
        farmer_days  = int((farmer_date - last_date).days)
        farmer_yield = float(actual_yields.get(farmer_date, np.nan))

        # Find closest actual date to model recommendation
        closest = min(actual_dates,
                      key=lambda d: abs((d - model_date).days))
        gap = abs((closest - model_date).days)

        if gap > tolerance_days:
            model_yield = np.nan
            note = f"No data within {tolerance_days}d of recommendation"
        else:
            model_yield = float(actual_yields.get(closest, np.nan))
            note = f"Matched to {closest.date()}"

        dq = model_yield - farmer_yield if not np.isnan(model_yield) else np.nan

        records.append({
            "test_harvest_date":  farmer_date,
            "farmer_days":        farmer_days,
            "farmer_yield_kg":    round(farmer_yield, 1),
            "model_rec_date":     model_date,
            "model_days":         opt_days,
            "model_matched_date": closest,
            "model_yield_kg":     round(model_yield, 1) if not np.isnan(model_yield) else np.nan,
            "DQ_kg":              round(dq, 1) if not np.isnan(dq) else np.nan,
            "DQ_pct":             round(dq/farmer_yield*100, 1) if farmer_yield > 0 and not np.isnan(dq) else np.nan,
            "note":               note,
        })

    dq_df = pd.DataFrame(records)

    print(f"\n{'='*65}")
    print(f"  DECISION QUALITY — {site}")
    print(f"  Model recommendation: {opt_days} days → {model_date.date()}")
    print(f"{'='*65}")
    print(f"  {'Test date':>12} {'Farmer':>6} {'Farmer yield':>13} "
          f"{'Model':>6} {'Model yield':>12} {'DQ (kg)':>9} {'DQ (%)':>7}")
    print(f"  {'-'*62}")
    for _, row in dq_df.iterrows():
        dq_str  = f"{row['DQ_kg']:+,.0f}" if not np.isnan(row['DQ_kg']) else "N/A"
        pct_str = f"{row['DQ_pct']:+.1f}%" if not np.isnan(row['DQ_pct']) else "N/A"
        print(f"  {str(row['test_harvest_date'].date()):>12} "
              f"{int(row['farmer_days']):>6}d "
              f"{row['farmer_yield_kg']:>13,.0f} "
              f"{int(row['model_days']):>6}d "
              f"{row['model_yield_kg']:>12,.0f} "
              f"{dq_str:>9} {pct_str:>7}")
    valid = dq_df.dropna(subset=["DQ_kg"])
    if len(valid) > 0:
        print(f"  {'-'*62}")
        print(f"  Mean DQ : {valid['DQ_kg'].mean():+,.0f} kg  "
              f"({'model better' if valid['DQ_kg'].mean()>0 else 'farmer better'})")
        print(f"  Total DQ: {valid['DQ_kg'].sum():+,.0f} kg over {len(valid)} harvests")
    print(f"{'='*65}\n")

    return dq_df


# ── 7. Visualisation ──────────────────────────────────────────────────────────

def plot_advice(advice: dict, figsize=(18, 10)):
    """
    Three-panel figure:
      Top-left  : bar chart of predicted yields for all candidate days
      Top-right : predicted yield map for optimal day
      Bottom    : days map + distribution
    """
    site       = advice["site"]
    summary    = advice["summary_table"]
    opt_days   = advice["optimal_days"]
    last_date  = advice["last_harvest_date"]
    yield_maps = advice["yield_maps"]
    inf_dfs    = advice["inference_dfs"]
    pred_y     = advice["pred_yield"]
    inf_opt    = advice["inference_dfs"].get(opt_days)

    fig = plt.figure(figsize=figsize)
    gs  = gridspec.GridSpec(2, 2, height_ratios=[1.5, 1.5],
                             hspace=0.45, wspace=0.35)

    # ── Bar chart ─────────────────────────────────────────────────────────────
    ax_bar = fig.add_subplot(gs[0, 0])
    colours = ["#2d6a3f" if d == opt_days else "#CBD5E1"
               for d in summary["days_ahead"]]
    bars = ax_bar.bar(
        [f"+{d}d" for d in summary["days_ahead"]],
        summary["pred_total"],
        color=colours, edgecolor="white", linewidth=0.5
    )
    for bar, (_, row) in zip(bars, summary.iterrows()):
        ax_bar.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + summary["pred_total"].max()*0.01,
                    f"{row['pred_total']:,.0f}\n(gr={row['growth_rate']:.2f})",
                    ha="center", va="bottom", fontsize=8,
                    color="#2d6a3f" if row["days_ahead"]==opt_days else "#555")
    ax_bar.set_xlabel("Days from last harvest")
    ax_bar.set_ylabel("Predicted total yield (kg)")
    ax_bar.set_title(f"Candidate yields\n(last: {last_date.date()})",
                     fontsize=10, fontweight="bold")
    ax_bar.grid(axis="y", alpha=0.3)

    # ── Optimal yield map ─────────────────────────────────────────────────────
    ax_map = fig.add_subplot(gs[0, 1])
    if inf_opt is not None and pred_y is not None and len(pred_y) > 0:
        x_vals = sorted(inf_opt["field_x"].unique())
        y_vals = sorted(inf_opt["field_y"].unique())
        x2i = {v: i for i, v in enumerate(x_vals)}
        y2i = {v: i for i, v in enumerate(y_vals)}
        grid = np.zeros((len(y_vals), len(x_vals)))
        for (_, row), p in zip(inf_opt.iterrows(), pred_y):
            grid[y2i[row["field_y"]], x2i[row["field_x"]]] = p
        vmax = float(np.nanquantile(pred_y[pred_y>0], 0.99)) if (pred_y>0).any() else 1.0
        im = ax_map.imshow(grid, cmap="YlOrRd", aspect="auto", vmin=0, vmax=vmax)
        plt.colorbar(im, ax=ax_map, label="kg", shrink=0.85)
        ax_map.set_title(
            f"Predicted Yield Map\n{advice['optimal_date'].date()} (+{opt_days}d)",
            fontsize=10, fontweight="bold")
        ax_map.set_xlabel("field_x"); ax_map.set_ylabel("field_y")
        ax_map.text(0.02, 0.97, f"Total: {pred_y.sum():,.0f} kg",
                    transform=ax_map.transAxes, fontsize=9, va="top", color="white",
                    bbox=dict(boxstyle="round,pad=0.3", fc="#333", alpha=0.65))

    # ── Growth rate line ──────────────────────────────────────────────────────
    ax_gr = fig.add_subplot(gs[1, :])
    ax_gr.plot(summary["days_ahead"], summary["growth_rate"],
               "o-", color="#5B8DB8", linewidth=2, markersize=8)
    ax_gr.axhline(advice["thresholds"]["t_high"], color="#E07B39",
                  linestyle="--", linewidth=1.5,
                  label=f"t_high={advice['thresholds']['t_high']:.3f} (wait longer)")
    ax_gr.axhline(advice["thresholds"]["t_low"],  color="#d44",
                  linestyle="--", linewidth=1.5,
                  label=f"t_low={advice['thresholds']['t_low']:.3f} (harvest soon)")
    ax_gr.axvline(opt_days, color="#2d6a3f", linestyle="-", linewidth=2, alpha=0.4)
    ax_gr.fill_between(summary["days_ahead"], advice["thresholds"]["t_low"],
                        advice["thresholds"]["t_high"], alpha=0.08, color="#5B8DB8",
                        label="Stable zone")
    for _, row in summary.iterrows():
        ax_gr.annotate(f"{int(row['stage2_days'])}d",
                       (row["days_ahead"], row["growth_rate"]),
                       textcoords="offset points", xytext=(0, 10), ha="center",
                       fontsize=9, color="#2d6a3f" if row["days_ahead"]==opt_days else "#555")
    ax_gr.set_xlabel("Candidate days ahead", fontsize=10)
    ax_gr.set_ylabel("Growth rate (pred / last actual)", fontsize=10)
    ax_gr.set_title("Stage 2: Growth rate rule — how the recommendation was made",
                    fontsize=11, fontweight="bold")
    ax_gr.legend(fontsize=9); ax_gr.grid(alpha=0.3)
    ax_gr.set_xticks(summary["days_ahead"])

    fig.suptitle(
        f"{site} — Harvest Advisor  |  Recommendation: +{opt_days} days "
        f"({advice['optimal_date'].date()})",
        fontsize=13, fontweight="bold")
    plt.show()
    return fig


def plot_decision_quality(dq_df: pd.DataFrame, site: str, figsize=(12, 5)):
    """Bar chart of Decision Quality per test harvest."""
    if dq_df.empty or dq_df["DQ_kg"].isna().all():
        print("No DQ data to plot.")
        return

    valid = dq_df.dropna(subset=["DQ_kg"])
    colours = ["#2d6a3f" if v >= 0 else "#c0392b" for v in valid["DQ_kg"]]

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(
        [str(d.date()) for d in valid["test_harvest_date"]],
        valid["DQ_kg"],
        color=colours, edgecolor="white", linewidth=0.5
    )
    ax.axhline(0, color="black", linewidth=1)
    ax.axhline(valid["DQ_kg"].mean(), color="#E07B39", linewidth=2,
               linestyle="--", label=f"Mean DQ = {valid['DQ_kg'].mean():+,.0f} kg")

    for bar, (_, row) in zip(bars, valid.iterrows()):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + (valid["DQ_kg"].abs().max()*0.02 * np.sign(bar.get_height())),
                f"{row['DQ_kg']:+,.0f} kg\n({row['DQ_pct']:+.1f}%)",
                ha="center", va="bottom" if row["DQ_kg"]>=0 else "top",
                fontsize=9)

    ax.set_xlabel("Test harvest date")
    ax.set_ylabel("Decision Quality (kg)\nPositive = model better")
    ax.set_title(f"{site} — Decision Quality\n"
                 f"Model recommendation vs Farmer's actual decision",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout(); plt.show()
    return fig


def print_recommendation(advice: dict):
    """Print a farmer-friendly summary."""
    print("\n" + "="*50)
    print(f"  HARVEST RECOMMENDATION — {advice['site']}")
    print("="*50)
    print(f"  Last harvest  : {advice['last_harvest_date'].date()}")
    print(f"  Last yield    : {advice['last_actual_total']:,.0f} kg")
    print(f"  Optimal date  : {advice['optimal_date'].date()}")
    print(f"  Days to wait  : {advice['optimal_days']} days")
    if len(advice["pred_yield"]) > 0:
        print(f"  Expected yield: {advice['pred_yield'].sum():,.0f} kg")
    print(f"\n  Candidate summary:")
    print(f"  {'Days':>5}  {'Date':>12}  {'Yield(kg)':>10}  {'Growth':>8}")
    print(f"  {'-'*42}")
    for _, row in advice["summary_table"].iterrows():
        mk = " ← OPTIMAL" if row["days_ahead"]==advice["optimal_days"] else ""
        print(f"  +{int(row['days_ahead']):<4}  "
              f"{str(row['date'].date()):>12}  "
              f"{row['pred_total']:>10,.0f}  "
              f"{row['growth_rate']:>8.3f}{mk}")
    print("="*50 + "\n")

# ═══════════════════════════════════════════════════════════════════════════════
# Rolling Forecast (实验性功能，原有函数不受影响)
# ═══════════════════════════════════════════════════════════════════════════════
#
# 版本A (rolling_forecast_lite):
#   只用预测值填 lag1/2/3，neighbor_mean 继续用最近一次真实采摘的空间分布。
#   误差累积较慢，接近实际部署中的轻量方案。
#
# 版本B (rolling_forecast_strict):
#   预测值完整替换为"假真实数据"，neighbor_mean 也从预测值重新计算。
#   误差会随步数累积，最严格地模拟多步预测能力。
#
# 共用函数 compare_rolling_versions: 并排对比两个版本 vs 真实值。
# ═══════════════════════════════════════════════════════════════════════════════


def _build_lag_only_row(df_history: pd.DataFrame,
                         candidate_date: pd.Timestamp,
                         weather: pd.DataFrame,
                         model_results,
                         lag_depth: int = 3,
                         coarsen_n: int = 1) -> pd.DataFrame:
    """
    版本A 内部函数：
    用 df_history（含真实 + 已预测行）构建 inference 特征，
    但 neighbor_mean 只从最近一次真实/已预测的 weight_kg 空间分布计算。
    """
    if coarsen_n > 1:
        import feature_engineering as _fe
        df_history = _fe.coarsen_grid(df_history, n=coarsen_n)

    all_dates  = sorted(df_history["harvest_date"].unique())
    past_dates = all_dates[-lag_depth:]          # 取最近 lag_depth 次

    pivot = df_history[df_history["harvest_date"].isin(past_dates)].pivot_table(
        index=["field_x", "field_y"],
        columns="harvest_date", values="weight_kg", aggfunc="first",
    ).reset_index()
    pivot.columns.name = None

    for k, d in enumerate(reversed(past_dates), start=1):
        if d in pivot.columns:
            pivot.rename(columns={d: f"yield_lag{k}"}, inplace=True)
    for k in range(1, lag_depth + 1):
        if f"yield_lag{k}" not in pivot.columns:
            pivot[f"yield_lag{k}"] = 0.0

    pivot["rolling_mean_3"] = (
        pivot["yield_lag1"] + pivot["yield_lag2"] + pivot["yield_lag3"]) / 3.0
    pivot["yield_trend"] = (pivot["yield_lag1"] - pivot["yield_lag3"]) / 2.0

    cum = df_history.groupby(["field_x", "field_y"])["weight_kg"].sum().reset_index()
    cum.columns = ["field_x", "field_y", "season_cumulative"]
    pivot = pivot.merge(cum, on=["field_x", "field_y"], how="left")
    pivot["season_cumulative"] = pivot["season_cumulative"].fillna(0)

    pivot["day_of_year"]     = candidate_date.dayofyear
    last_date                = past_dates[-1]
    pivot["days_since_last"] = float((candidate_date - last_date).days)

    # neighbor_mean from lag1 (最近一次，可能是预测值)
    pivot["neighbor_mean_3x3"] = _neighbor_means_from_df(pivot, "yield_lag1", 3)
    pivot["neighbor_mean_5x5"] = _neighbor_means_from_df(pivot, "yield_lag1", 5)

    # easting / northing (从历史里取，超格网格下已经平均过)
    coord = df_history[df_history["harvest_date"] == past_dates[-1]][
        ["field_x", "field_y", "easting", "northing"]
    ].drop_duplicates()
    pivot = pivot.merge(coord, on=["field_x", "field_y"], how="left")

    # Weather: 7-day window before candidate
    from datetime import timedelta
    w = weather[(weather.index >= candidate_date - timedelta(days=7)) &
                (weather.index <= candidate_date - timedelta(days=1))]
    if len(w) == 0:
        for col in ["temp_mean_7d", "temp_max_7d", "temp_min_7d", "precip_7d",
                    "et0_7d", "humidity_mean_7d", "soil_moisture_0_7",
                    "soil_moisture_7_28", "daylight_7d"]:
            pivot[col] = np.nan
    else:
        pivot["temp_mean_7d"]       = float(w["temp_mean"].mean())
        pivot["temp_max_7d"]        = float(w["temp_max"].max())
        pivot["temp_min_7d"]        = float(w["temp_min"].min())
        pivot["precip_7d"]          = float(w["precip"].sum())
        pivot["et0_7d"]             = float(w["et0"].sum())
        pivot["humidity_mean_7d"]   = float(w["humidity_mean"].mean())
        pivot["soil_moisture_0_7"]  = float(w["soil_moisture_0_7"].mean())
        pivot["soil_moisture_7_28"] = float(w["soil_moisture_7_28"].mean())
        pivot["daylight_7d"]        = float(w["daylight_hours"].mean())

    pivot["harvest_date"] = candidate_date
    return pivot


def rolling_forecast_lite(df_raw: pd.DataFrame,
                           model_results,
                           weather: pd.DataFrame,
                           splits: dict,
                           site: str,
                           lag_depth: int = 3,
                           coarsen_n: int = 1) -> pd.DataFrame:
    """
    版本A：滚动预测（轻量版）

    只用预测值填 lag，neighbor_mean 从最近一次已知空间分布算。
    从训练集最后一次采摘开始，对验证集 + 测试集的每个采摘日期做滚动预测。

    Parameters
    ----------
    df_raw        : 原始 1x1 DataFrame
    model_results : run_model_comparison() 的输出
    weather       : fetch_weather() 的输出
    splits        : split_data() 的输出
    site          : 'SantaMaria' or 'Salinas'
    coarsen_n     : 与训练时一致

    Returns
    -------
    DataFrame: harvest_date, actual_kg, pred_kg, diff_kg, diff_pct, step, mode
    """
    print(f"\n{'='*60}")
    print(f"  Rolling Forecast — {site}  [版本A: Lite]")
    print(f"  coarsen_n={coarsen_n}  lag_depth={lag_depth}")
    print(f"{'='*60}")

    # 初始历史：训练集的真实数据
    train_df   = splits["train"].copy()
    history_df = df_raw[df_raw["harvest_date"].isin(
        train_df["harvest_date"].unique())].copy()

    # 预测目标：val + test 的采摘日期
    target_splits = {}
    for s in ["val", "test"]:
        if s in splits:
            target_splits[s] = splits[s]

    actual_yields = df_raw.groupby("harvest_date")["weight_kg"].sum()

    records = []
    step    = 0

    for split_name, split_df in target_splits.items():
        for hdate in sorted(split_df["harvest_date"].unique()):
            step += 1
            hdate = pd.Timestamp(hdate)

            # 用当前 history 构建特征
            inf_df = _build_lag_only_row(
                history_df, hdate, weather, model_results,
                lag_depth=lag_depth, coarsen_n=coarsen_n)

            pred   = _predict_yield(inf_df, model_results)
            pred_kg = float(pred.sum())
            actual_kg = float(actual_yields.get(hdate, np.nan))

            diff_kg  = pred_kg - actual_kg
            diff_pct = diff_kg / actual_kg * 100 if actual_kg > 0 else np.nan

            print(f"  Step {step:2d} [{split_name}] {hdate.date()}  "
                  f"actual={actual_kg:,.0f}  pred={pred_kg:,.0f}  "
                  f"diff={diff_kg:+,.0f} ({diff_pct:+.1f}%)")

            records.append({
                "step":       step,
                "split":      split_name,
                "harvest_date": hdate,
                "actual_kg":  round(actual_kg, 1),
                "pred_kg":    round(pred_kg, 1),
                "diff_kg":    round(diff_kg, 1),
                "diff_pct":   round(diff_pct, 2) if not np.isnan(diff_pct) else np.nan,
                "version":    "A_lite",
            })

            # 把预测值追加进 history（用于下一步的 lag）
            pred_row = inf_df.copy()
            pred_row["weight_kg"]    = pred
            pred_row["harvest_date"] = hdate
            # 只保留 df_raw 需要的列
            keep = [c for c in ["field_x","field_y","easting","northing",
                                 "weight_kg","harvest_date"] if c in pred_row.columns]
            history_df = pd.concat(
                [history_df, pred_row[keep]], ignore_index=True)

    result = pd.DataFrame(records)
    _print_rolling_summary(result, site, "A_lite")
    return result


def rolling_forecast_strict(df_raw: pd.DataFrame,
                              model_results,
                              weather: pd.DataFrame,
                              splits: dict,
                              site: str,
                              lag_depth: int = 3,
                              coarsen_n: int = 1) -> pd.DataFrame:
    """
    版本B：滚动预测（严格版）

    预测值完整替换为"假真实数据"，neighbor_mean 也从预测值重新计算。
    误差会随步数累积，是最严格的多步预测评估。

    Parameters 同 rolling_forecast_lite。
    """
    print(f"\n{'='*60}")
    print(f"  Rolling Forecast — {site}  [版本B: Strict]")
    print(f"  coarsen_n={coarsen_n}  lag_depth={lag_depth}")
    print(f"{'='*60}")

    train_df  = splits["train"].copy()
    history_df = df_raw[df_raw["harvest_date"].isin(
        train_df["harvest_date"].unique())].copy()

    target_splits = {}
    for s in ["val", "test"]:
        if s in splits:
            target_splits[s] = splits[s]

    actual_yields = df_raw.groupby("harvest_date")["weight_kg"].sum()

    records = []
    step    = 0

    for split_name, split_df in target_splits.items():
        for hdate in sorted(split_df["harvest_date"].unique()):
            step += 1
            hdate = pd.Timestamp(hdate)

            # 版本B：用完整 history（含之前预测行）构建特征
            # neighbor_mean 也从 history 的 weight_kg（可能是预测值）算
            inf_df = _build_lag_only_row(
                history_df, hdate, weather, model_results,
                lag_depth=lag_depth, coarsen_n=coarsen_n)

            pred    = _predict_yield(inf_df, model_results)
            pred_kg = float(pred.sum())
            actual_kg = float(actual_yields.get(hdate, np.nan))

            diff_kg  = pred_kg - actual_kg
            diff_pct = diff_kg / actual_kg * 100 if actual_kg > 0 else np.nan

            print(f"  Step {step:2d} [{split_name}] {hdate.date()}  "
                  f"actual={actual_kg:,.0f}  pred={pred_kg:,.0f}  "
                  f"diff={diff_kg:+,.0f} ({diff_pct:+.1f}%)")

            records.append({
                "step":         step,
                "split":        split_name,
                "harvest_date": hdate,
                "actual_kg":    round(actual_kg, 1),
                "pred_kg":      round(pred_kg, 1),
                "diff_kg":      round(diff_kg, 1),
                "diff_pct":     round(diff_pct, 2) if not np.isnan(diff_pct) else np.nan,
                "version":      "B_strict",
            })

            # 版本B：把预测值作为完整的新采摘行写入 history
            # 包含 neighbor_mean（已在 inf_df 里算好）
            pred_row = inf_df.copy()
            pred_row["weight_kg"]    = pred
            pred_row["harvest_date"] = hdate
            keep = [c for c in ["field_x","field_y","easting","northing",
                                 "weight_kg","harvest_date"] if c in pred_row.columns]
            history_df = pd.concat(
                [history_df, pred_row[keep]], ignore_index=True)

    result = pd.DataFrame(records)
    _print_rolling_summary(result, site, "B_strict")
    return result


def _print_rolling_summary(df: pd.DataFrame, site: str, version: str):
    """打印滚动预测的汇总统计。"""
    print(f"\n  ── 汇总 [{version}] ─────────────────────────────")
    for split in ["val", "test"]:
        sub = df[df["split"] == split].dropna(subset=["diff_pct"])
        if sub.empty:
            continue
        print(f"  {split.upper():5s}: MAPE={sub['diff_pct'].abs().mean():.1f}%  "
              f"MBE={sub['diff_kg'].mean():+,.0f} kg  "
              f"n={len(sub)}")
    print()


def compare_rolling_versions(df_raw: pd.DataFrame,
                              model_results,
                              weather: pd.DataFrame,
                              splits: dict,
                              site: str,
                              coarsen_n: int = 1,
                              figsize=(13, 5)):
    """
    并排跑版本A 和版本B，然后画对比折线图（ vs 真实值）。

    Returns
    -------
    (result_a, result_b) : 两个 DataFrame
    """
    result_a = rolling_forecast_lite(
        df_raw, model_results, weather, splits, site, coarsen_n=coarsen_n)
    result_b = rolling_forecast_strict(
        df_raw, model_results, weather, splits, site, coarsen_n=coarsen_n)

    # ── 对比表 ────────────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  版本对比 — {site}")
    print(f"{'='*65}")
    print(f"  {'':12}  {'版本A (Lite)':>20}  {'版本B (Strict)':>20}")
    print(f"  {'-'*58}")
    for split in ["val", "test"]:
        a = result_a[result_a["split"]==split].dropna(subset=["diff_pct"])
        b = result_b[result_b["split"]==split].dropna(subset=["diff_pct"])
        if a.empty:
            continue
        print(f"  {split.upper()} MAPE   : "
              f"{a['diff_pct'].abs().mean():>18.1f}%  "
              f"{b['diff_pct'].abs().mean():>18.1f}%")
        print(f"  {split.upper()} MBE    : "
              f"{a['diff_kg'].mean():>17,.0f} kg  "
              f"{b['diff_kg'].mean():>17,.0f} kg")
    print(f"{'='*65}\n")

    # ── 折线图 ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(f"{site} — Rolling Forecast: 版本A vs 版本B vs 真实值",
                 fontsize=12, fontweight="bold")

    for ax, split in zip(axes, ["val", "test"]):
        a = result_a[result_a["split"]==split].sort_values("harvest_date")
        b = result_b[result_b["split"]==split].sort_values("harvest_date")
        if a.empty:
            ax.set_visible(False)
            continue

        dates = a["harvest_date"]
        ax.plot(dates, a["actual_kg"],  "o-", color="#2d5a3d",
                linewidth=2.2, markersize=7, label="Ground Truth", zorder=4)
        ax.plot(dates, a["pred_kg"],    "s--", color="#5B8DB8",
                linewidth=2, markersize=6, label="版本A (Lite)", zorder=3)
        ax.plot(b["harvest_date"], b["pred_kg"], "^:", color="#E07B39",
                linewidth=2, markersize=6, label="版本B (Strict)", zorder=3)

        ax.fill_between(dates, a["actual_kg"], a["pred_kg"],
                        alpha=0.1, color="#5B8DB8")

        ax.set_title(f"{split.upper()} set", fontsize=10, fontweight="bold")
        ax.set_xlabel("Harvest date")
        ax.set_ylabel("Total yield (kg)")
        ax.xaxis.set_major_formatter(
            __import__("matplotlib.dates", fromlist=["DateFormatter"]).DateFormatter("%m-%d"))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

        # 标注每个点的误差%
        for _, row in a.iterrows():
            ax.annotate(f"A:{row['diff_pct']:+.1f}%",
                        (row["harvest_date"], row["pred_kg"]),
                        textcoords="offset points", xytext=(0, 8),
                        fontsize=7, color="#5B8DB8", ha="center")
        for _, row in b.iterrows():
            ax.annotate(f"B:{row['diff_pct']:+.1f}%",
                        (row["harvest_date"], b.loc[b["harvest_date"]==row["harvest_date"],
                                                     "pred_kg"].values[0]),
                        textcoords="offset points", xytext=(0, -14),
                        fontsize=7, color="#E07B39", ha="center")

    plt.tight_layout()
    plt.show()

    return result_a, result_b




# ═══════════════════════════════════════════════════════════════════════════════
# Stage 2 v3 — Method B (acceleration-aware rule)
#
# Key additions vs original Stage 2:
#   derive_thresholds_v2      -- filters rare intervals (< min_count)
#   apply_rule_method_b       -- adds velocity (acceleration) to decision
#   run_stage2_scheme_b       -- field-level eval: pred_days vs actual_days
#   run_grid_comparison       -- grid-level ML vs Rule-B across all schemes
#   print_stage2_metrics      -- consistent metric reporting
# ═══════════════════════════════════════════════════════════════════════════════
 
 
# ── Stage 2 v3: threshold derivation (filters rare intervals) ─────────────────
 
def derive_thresholds_v2(df_feat: pd.DataFrame,
                          site: str,
                          min_count: int = 2) -> dict:
    """
    Derive growth-rate thresholds for Method B from feature DataFrame.
 
    Fix-1 vs original derive_thresholds():
      Intervals appearing fewer than min_count times are filtered out before
      computing t_low / t_high, preventing a single anomalous harvest from
      dominating the thresholds.
 
    Parameters
    ----------
    df_feat   : feature DataFrame (must have weight_kg and days_since_last)
    site      : site name string
    min_count : minimum occurrences required for an interval to be included
 
    Returns
    -------
    dict with t_low, t_high, days_map, intervals, field_df, site
    """
    field = (df_feat.groupby('harvest_date')
             .agg(total_kg=('weight_kg', 'sum'),
                  days_interval=('days_since_last', 'first'))
             .dropna(subset=['days_interval'])
             .sort_values('harvest_date')
             .reset_index())
 
    field['growth_rate']   = field['total_kg'] / field['total_kg'].shift(1)
    field                  = field.dropna(subset=['growth_rate'])
    field['days_interval'] = field['days_interval'].astype(int)
 
    counts       = field['days_interval'].value_counts()
    valid_ivs    = sorted(counts[counts >= min_count].index.tolist())
    filtered_out = sorted(counts[counts <  min_count].index.tolist())
 
    print(f"\n[{site}] Raw interval types : {sorted(field['days_interval'].unique())}")
    if filtered_out:
        print(f"  Filtered (< {min_count} occurrences): {filtered_out}")
    print(f"  Used for thresholds : {valid_ivs}")
 
    fv        = field[field['days_interval'].isin(valid_ivs)]
    intervals = valid_ivs
    print(fv.groupby('days_interval')['growth_rate']
          .agg(['count', 'mean', 'median']).round(3))
 
    if len(intervals) >= 3:
        t_low  = float(fv[fv['days_interval'] == intervals[0]]['growth_rate'].median())
        t_high = float(fv[fv['days_interval'] == intervals[-1]]['growth_rate'].median())
    elif len(intervals) == 2:
        t_low  = float(fv['growth_rate'].quantile(0.40))
        t_high = float(fv['growth_rate'].quantile(0.60))
    else:
        t_low  = float(fv['growth_rate'].quantile(0.35))
        t_high = float(fv['growth_rate'].quantile(0.65))
 
    days_map = {
        'short':  intervals[0],
        'medium': intervals[len(intervals) // 2] if len(intervals) >= 3 else intervals[0],
        'long':   intervals[-1],
    }
    print(f"  t_low={t_low:.3f}  t_high={t_high:.3f}  days_map={days_map}")
 
    return {'t_low': t_low, 't_high': t_high,
            'days_map': days_map, 'intervals': intervals,
            'field_df': fv, 'site': site}
 
 
# ── Method B decision: growth rate + velocity ─────────────────────────────────
 
def apply_rule_method_b(pred_total: float,
                         history_totals: list,
                         thresholds: dict,
                         velocity_clip: float = 0.30) -> dict:
    """
    Method B decision matrix using clipped velocity (acceleration).
 
    Inputs
    ------
    pred_total     : Stage 1 predicted total yield for next harvest
    history_totals : [total_kg at t-3, t-2, t-1]  oldest first
    velocity_clip  : symmetric clip bound for velocity
 
    Growth rates
    ------------
    gr_prev  = history[-2] / history[-3]
    gr_curr  = history[-1] / history[-2]
    gr_pred  = pred_total  / history[-1]
    velocity = clip(gr_curr - gr_prev, -clip, +clip)
 
    Decision matrix
    ---------------
                      velocity >= 0       velocity < 0
    gr_pred >= t_high    long                medium
    gr_pred >= t_low     medium              short
    gr_pred <  t_low     short               short
 
    Returns
    -------
    dict with gr_prev, gr_curr, gr_pred, velocity_raw, velocity_clipped, rec_days
    """
    h       = history_totals
    gr_prev = h[-2] / h[-3] if h[-3] > 0 else 1.0
    gr_curr = h[-1] / h[-2] if h[-2] > 0 else 1.0
    gr_pred = pred_total / h[-1] if h[-1] > 0 else 1.0
 
    velocity_raw     = gr_curr - gr_prev
    velocity_clipped = float(np.clip(velocity_raw, -velocity_clip, velocity_clip))
 
    t_low, t_high = thresholds['t_low'], thresholds['t_high']
    dm            = thresholds['days_map']
 
    if gr_pred >= t_high:
        rec_days = dm['long']   if velocity_clipped >= 0 else dm['medium']
    elif gr_pred >= t_low:
        rec_days = dm['medium'] if velocity_clipped >= 0 else dm['short']
    else:
        rec_days = dm['short']
 
    return {
        'gr_prev':          round(gr_prev, 4),
        'gr_curr':          round(gr_curr, 4),
        'gr_pred':          round(gr_pred, 4),
        'velocity_raw':     round(velocity_raw, 4),
        'velocity_clipped': round(velocity_clipped, 4),
        'rec_days':         rec_days,
    }
 
 
# ── Field-level Stage 2 evaluation: pred_days vs actual_days ─────────────────
 
def run_stage2_scheme_b(df_raw: pd.DataFrame,
                         model_results_7x7,
                         weather: pd.DataFrame,
                         thresholds: dict,
                         site: str,
                         scheme_b_splits: list,
                         velocity_clip: float = 0.30) -> pd.DataFrame:
    """
    Field-level Method B evaluation over Scheme B test windows.
 
    For each window:
      1. Run Stage 1 (coarsen_n=7) to get pred_total.
      2. Apply Method B rule → rec_days.
      3. actual_days = days_since_last for that test date (ground truth).
      4. Compute error = rec_days - actual_days directly.
 
    No yield lookup. No nearest-date matching.
 
    Parameters
    ----------
    df_raw           : raw DataFrame from data_pipeline (1x1)
    model_results_7x7: output of models.run_model_comparison() on 7x7 grid
    weather          : weather DataFrame
    thresholds       : output of derive_thresholds_v2()
    site             : site name string
    scheme_b_splits  : output of build_scheme_b_splits()
    velocity_clip    : clip bound for velocity
 
    Returns
    -------
    DataFrame with one row per window:
      window, last_date, test_date, actual_days, pred_days,
      error, correct, within_1, gr_pred, velocity_raw, velocity_clipped
    """
    actual_yields = df_raw.groupby('harvest_date')['weight_kg'].sum()
    all_dates     = sorted(df_raw['harvest_date'].unique())
 
    date_to_days = {}
    for i in range(1, len(all_dates)):
        gap = (pd.Timestamp(all_dates[i]) - pd.Timestamp(all_dates[i-1])).days
        date_to_days[pd.Timestamp(all_dates[i])] = gap
 
    records = []
 
    for i, sp in enumerate(scheme_b_splits):
        test_date = pd.Timestamp(sp['test_date'])
        last_date = pd.Timestamp(sp['train_d2'])
 
        actual_days = date_to_days.get(test_date)
        if actual_days is None:
            print(f"  Window {i}: cannot compute actual_days for {test_date}, skipping")
            continue
 
        past_dates = [d for d in all_dates if pd.Timestamp(d) <= last_date]
        if len(past_dates) < 3:
            print(f"  Window {i}: fewer than 3 past dates, skipping")
            continue
 
        history_totals = [float(actual_yields.get(pd.Timestamp(d), 0))
                          for d in past_dates[-3:]]
 
        try:
            inf_df     = _build_inference_row(
                df_raw, last_date,
                last_date + timedelta(days=1),
                weather, lag_depth=3, coarsen_n=7)
            pred_total = float(_predict_yield(inf_df, model_results_7x7).sum())
        except Exception as e:
            print(f"  Window {i}: Stage 1 failed -> {e}")
            continue
 
        rule     = apply_rule_method_b(pred_total, history_totals, thresholds,
                                        velocity_clip=velocity_clip)
        rec_days = rule['rec_days']
 
        records.append({
            'window':            i,
            'last_date':         last_date.date(),
            'test_date':         test_date.date(),
            'actual_days':       actual_days,
            'pred_days':         rec_days,
            'error':             rec_days - actual_days,
            'correct':           int(rec_days == actual_days),
            'within_1':          int(abs(rec_days - actual_days) <= 1),
            'gr_pred':           rule['gr_pred'],
            'velocity_raw':      rule['velocity_raw'],
            'velocity_clipped':  rule['velocity_clipped'],
        })
 
    return pd.DataFrame(records)
 
 
def print_stage2_metrics(df: pd.DataFrame, site: str):
    """Print Accuracy, MAE, Within-1, Bias for a stage2 result DataFrame."""
    if df.empty:
        print(f"[{site}] No results."); return
    acc  = df['correct'].mean()
    mae  = df['error'].abs().mean()
    w1   = df['within_1'].mean()
    bias = df['error'].mean()
    print(f"\n[{site}] Evaluation (pred_days vs actual_days):")
    print(f"  N windows  : {len(df)}")
    print(f"  Accuracy   : {acc:.3f}  (exact match rate)")
    print(f"  MAE        : {mae:.3f} days")
    print(f"  Within-1   : {w1:.3f}  (|error| <= 1 day)")
    print(f"  Bias       : {bias:+.3f} days  (+ = recommends longer)")
 
 
# ── Helper: build Scheme B splits ─────────────────────────────────────────────
 
def build_scheme_b_splits(df_feat: pd.DataFrame,
                           train_ratio: float = 0.6) -> tuple:
    """
    Build Scheme B (chronological sliding window) test splits from df_feat.
 
    Returns (splits, windows, split_idx) where splits is a list of dicts
    with keys: train_df, test_date, train_d1, train_d2.
    """
    dates     = sorted(df_feat['harvest_date'].unique())
    n         = len(dates)
    windows   = [(dates[i], dates[i+1], dates[i+2]) for i in range(n - 2)]
    split_idx = int(len(windows) * train_ratio)
    splits    = []
    for d1, d2, d3 in windows[split_idx:]:
        splits.append({
            'train_df':  df_feat[df_feat['harvest_date'].isin([d1, d2])].copy(),
            'test_date': d3, 'train_d1': d1, 'train_d2': d2,
        })
    return splits, windows, split_idx
 
 
# ── Grid-level comparison: ML vs Rule-B across schemes and grid sizes ──────────
 
def _get_scheme_dates(df_feat, scheme, site,
                       test_ratio=0.4, train_ratio=0.6, seed=42):
    """Return (train_dates, test_dates) per ABCDE scheme at harvest-date level."""
    dates = sorted(df_feat['harvest_date'].unique())
    n     = len(dates)
    if scheme == 'A.1':
        np.random.seed(seed)
        test_idx  = sorted(np.random.choice(n, max(1, int(n*test_ratio)), replace=False))
        train_idx = [i for i in range(n) if i not in test_idx]
    elif scheme in ('B', 'C'):
        split = int(n * train_ratio)
        train_idx = list(range(split)); test_idx = list(range(split, n))
    elif scheme == 'D':
        np.random.seed(seed + 1)
        test_idx  = sorted(np.random.choice(n, max(1, int(n*test_ratio)), replace=False))
        train_idx = [i for i in range(n) if i not in test_idx]
    elif scheme == 'E':
        split = int(n * test_ratio)
        test_idx = list(range(split)); train_idx = list(range(split, n))
    return [dates[i] for i in train_idx], [dates[i] for i in test_idx]
 
 
def _get_field_intervals(df_feat):
    """Return {harvest_date -> days_since_last (int)}."""
    return (df_feat.groupby('harvest_date')['days_since_last']
            .first().dropna().astype(int).to_dict())
 
 
def _snap_to_valid(pred, valid_days):
    return min(valid_days, key=lambda d: abs(d - pred))
 
 
def _compute_metrics(actuals, preds):
    """Return dict with acc, mae, within1, bias, n."""
    if not actuals:
        return {'acc': np.nan, 'mae': np.nan, 'within1': np.nan,
                'bias': np.nan, 'n': 0}
    a = np.array(actuals); p = np.array(preds)
    return {
        'acc':     float(np.mean(a == p)),
        'mae':     float(np.mean(np.abs(a - p))),
        'within1': float(np.mean(np.abs(a - p) <= 1)),
        'bias':    float(np.mean(p - a)),
        'n':       len(a),
    }
 
 
def _run_ml_method_grid(df_feat, scheme, site, seed=42):
    """
    Train RF classifier + regressor on grid-level rows.
    Label = days_since_last (field-level, broadcast to all cells).
    Aggregate via majority vote per test date.
    Returns list of {date, actual, pred_cls, pred_reg}.
    """
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
 
    ML_FEATS = [
        'yield_lag1', 'yield_lag2', 'yield_lag3',
        'rolling_mean_3', 'yield_trend', 'season_cumulative', 'day_of_year',
        'field_x', 'field_y', 'neighbor_mean_3x3', 'neighbor_mean_5x5',
        'temp_mean_7d', 'precip_7d', 'et0_7d',
        'humidity_mean_7d', 'soil_moisture_0_7', 'daylight_7d',
    ]
    feats        = [f for f in ML_FEATS if f in df_feat.columns]
    interval_map = _get_field_intervals(df_feat)
    valid_days   = sorted(set(interval_map.values()))
 
    train_dates, test_dates = _get_scheme_dates(df_feat, scheme, site, seed=seed)
    train_dates = [d for d in train_dates if d in interval_map]
    test_dates  = [d for d in test_dates  if d in interval_map]
    if len(train_dates) < 3 or len(test_dates) < 1:
        return []
 
    train_df = df_feat[df_feat['harvest_date'].isin(train_dates)].copy()
    test_df  = df_feat[df_feat['harvest_date'].isin(test_dates)].copy()
    train_df['label'] = train_df['harvest_date'].map(interval_map)
    test_df['label']  = test_df['harvest_date'].map(interval_map)
    train_df = train_df.dropna(subset=['label'] + feats)
    test_df  = test_df.dropna(subset=['label'] + feats)
 
    X_tr = train_df[feats].values.astype(np.float32)
    y_tr = train_df['label'].values.astype(int)
    X_te = test_df[feats].values.astype(np.float32)
 
    clf = RandomForestClassifier(n_estimators=200, max_depth=6,
                                  random_state=seed, n_jobs=-1)
    clf.fit(X_tr, y_tr)
    test_df = test_df.copy()
    test_df['pred_cls'] = clf.predict(X_te)
 
    reg = RandomForestRegressor(n_estimators=200, max_depth=6,
                                 random_state=seed, n_jobs=-1)
    reg.fit(X_tr, y_tr.astype(float))
    test_df['pred_reg_raw'] = reg.predict(X_te)
    test_df['pred_reg']     = test_df['pred_reg_raw'].apply(
        lambda p: _snap_to_valid(p, valid_days))
 
    results = []
    for date in test_dates:
        day_df = test_df[test_df['harvest_date'] == date]
        if day_df.empty: continue
        results.append({
            'date':     date,
            'actual':   interval_map[date],
            'pred_cls': int(day_df['pred_cls'].mode().iloc[0]),
            'pred_reg': int(day_df['pred_reg'].mode().iloc[0]),
        })
    return results
 
 
def _derive_rule_thresholds_raw(df_raw, train_dates, interval_map, min_count=2):
    """Derive thresholds from RAW (un-normalised) field totals."""
    field = (df_raw[df_raw['harvest_date'].isin(train_dates)]
             .groupby('harvest_date').agg(total_kg=('weight_kg', 'sum'))
             .reset_index().sort_values('harvest_date'))
    field['days_interval'] = field['harvest_date'].map(interval_map)
    field = field.dropna(subset=['days_interval'])
    field['days_interval'] = field['days_interval'].astype(int)
    field['growth_rate']   = field['total_kg'] / field['total_kg'].shift(1)
    field = field.dropna(subset=['growth_rate'])
 
    counts    = field['days_interval'].value_counts()
    valid_ivs = sorted(counts[counts >= min_count].index.tolist())
    if not valid_ivs:
        valid_ivs = sorted(counts.index.tolist())
 
    fv = field[field['days_interval'].isin(valid_ivs)]
    intervals = valid_ivs
 
    if len(intervals) >= 3:
        t_low  = float(fv[fv['days_interval'] == intervals[0]]['growth_rate'].median())
        t_high = float(fv[fv['days_interval'] == intervals[-1]]['growth_rate'].median())
    elif len(intervals) == 2:
        t_low  = float(fv['growth_rate'].quantile(0.40))
        t_high = float(fv['growth_rate'].quantile(0.60))
    else:
        t_low, t_high = 0.90, 1.10
 
    days_map = {
        'short':  intervals[0],
        'medium': intervals[len(intervals)//2] if len(intervals) >= 3 else intervals[0],
        'long':   intervals[-1],
    }
    return {'t_low': t_low, 't_high': t_high, 'days_map': days_map}
 
 
def _run_rule_method_grid(df_raw, df_feat, scheme, site,
                           coarsen_n=1, velocity_clip=0.30, seed=42):
    """
    Grid-level Rule-Based Method B using RAW (un-normalised) yield.
    Growth rates computed from df_raw weight_kg so different grid sizes
    produce genuinely different cell-level gr_pred values.
    Returns list of {date, actual, pred_rule}.
    """
    interval_map = _get_field_intervals(df_feat)
    train_dates, test_dates = _get_scheme_dates(df_feat, scheme, site, seed=seed)
    train_dates = [d for d in train_dates if d in interval_map]
    test_dates  = [d for d in test_dates  if d in interval_map]
    if len(train_dates) < 3 or len(test_dates) < 1:
        return []
 
    if coarsen_n > 1:
        import feature_engineering as fe
        df_raw_g = fe.coarsen_grid(df_raw, n=coarsen_n)
    else:
        df_raw_g = df_raw.copy()
 
    thresholds = _derive_rule_thresholds_raw(df_raw_g, train_dates, interval_map)
 
    raw_pivot = (df_raw_g.groupby(['harvest_date', 'field_x', 'field_y'])['weight_kg']
                 .sum().reset_index())
    all_dates = sorted(df_raw_g['harvest_date'].unique())
 
    results = []
    for date in test_dates:
        actual = interval_map[date]
        past   = [d for d in all_dates if pd.Timestamp(d) < pd.Timestamp(date)]
        if len(past) < 3: continue
        d_t1, d_t2, d_t3 = past[-1], past[-2], past[-3]
 
        kg_t1 = (raw_pivot[raw_pivot['harvest_date']==d_t1]
                 .set_index(['field_x','field_y'])['weight_kg'])
        kg_t2 = (raw_pivot[raw_pivot['harvest_date']==d_t2]
                 .set_index(['field_x','field_y'])['weight_kg'])
        kg_t3 = (raw_pivot[raw_pivot['harvest_date']==d_t3]
                 .set_index(['field_x','field_y'])['weight_kg'])
 
        common = kg_t1.index.intersection(kg_t2.index).intersection(kg_t3.index)
        if len(common) == 0: continue
        kg_t1 = kg_t1.loc[common]; kg_t2 = kg_t2.loc[common]; kg_t3 = kg_t3.loc[common]
 
        gr_pred  = np.where(kg_t2 > 0, kg_t1 / kg_t2, 1.0)
        gr_prev  = np.where(kg_t3 > 0, kg_t2 / kg_t3, 1.0)
        velocity = np.clip(gr_pred - gr_prev, -velocity_clip, velocity_clip)
 
        cell_recs = []
        for gp, v in zip(gr_pred, velocity):
            t_low, t_high = thresholds['t_low'], thresholds['t_high']
            dm = thresholds['days_map']
            if gp >= t_high:
                cell_recs.append(dm['long'] if v >= 0 else dm['medium'])
            elif gp >= t_low:
                cell_recs.append(dm['medium'] if v >= 0 else dm['short'])
            else:
                cell_recs.append(dm['short'])
 
        pred_rule = int(pd.Series(cell_recs).mode().iloc[0])
        results.append({'date': date, 'actual': actual, 'pred_rule': pred_rule})
    return results
 
 
def run_grid_comparison(grid_feats: dict,
                         df_sm: pd.DataFrame,
                         df_sal: pd.DataFrame,
                         grid_sizes: list = None,
                         schemes: list = None) -> pd.DataFrame:
    """
    Run grid-level ML vs Rule-B comparison across all grid sizes and schemes.
 
    Parameters
    ----------
    grid_feats  : dict mapping grid label to (df_feat_sm, df_feat_sal)
                  e.g. {'1x1': (df_feat_sm, df_feat_sal), '7x7': (...)}
    df_sm       : raw SantaMaria DataFrame from data_pipeline
    df_sal      : raw Salinas DataFrame from data_pipeline
    grid_sizes  : list of keys to evaluate (default = all keys in grid_feats)
    schemes     : list of schemes (default = ['A.1','B','C','D','E'])
 
    Returns
    -------
    DataFrame with columns:
      grid, site, scheme, method, acc, mae, within1, bias, n_test
    """
    if grid_sizes is None:
        grid_sizes = list(grid_feats.keys())
    if schemes is None:
        schemes = ['A.1', 'B', 'C', 'D', 'E']
 
    GRID_COARSEN = {'1x1': 1, '5x5': 5, '7x7': 7, '8x8': 8}
 
    print("=" * 70)
    print("  Grid-Level Method Comparison: ML vs Rule-B")
    print(f"  Grid sizes : {grid_sizes}")
    print(f"  Schemes    : {schemes}")
    print("=" * 70)
 
    all_records = []
 
    for grid_size in grid_sizes:
        if grid_size not in grid_feats:
            print(f"\n  [{grid_size}] not in grid_feats, skipping")
            continue
 
        df_feat_sm_g, df_feat_sal_g = grid_feats[grid_size]
        coarsen_n = GRID_COARSEN.get(grid_size, 1)
 
        for site, df_feat_g, df_raw_g in [
            ('SantaMaria', df_feat_sm_g, df_sm),
            ('Salinas',    df_feat_sal_g, df_sal),
        ]:
            mask = df_feat_g['rolling_mean_3'].isna()
            if mask.sum() > 0:
                df_feat_g = df_feat_g.copy()
                df_feat_g.loc[mask, 'rolling_mean_3'] = df_feat_g.loc[mask, 'yield_lag1']
 
            print(f"\n  [{grid_size}] {site}")
 
            for scheme in schemes:
                # ML
                ml_res = _run_ml_method_grid(df_feat_g, scheme, site)
                if ml_res:
                    actuals   = [r['actual']   for r in ml_res]
                    preds_cls = [r['pred_cls'] for r in ml_res]
                    preds_reg = [r['pred_reg'] for r in ml_res]
                    for m, mname in [(_compute_metrics(actuals, preds_cls), 'ML_RF_cls'),
                                     (_compute_metrics(actuals, preds_reg), 'ML_RF_reg')]:
                        all_records.append({
                            'grid': grid_size, 'site': site, 'scheme': scheme,
                            'method': mname, **{k: m[k] for k in ['acc','mae','within1','bias','n']},
                        })
                    m_cls = _compute_metrics(actuals, preds_cls)
                    m_reg = _compute_metrics(actuals, preds_reg)
                    print(f"    {scheme}  ML_cls  acc={m_cls['acc']:.3f}  "
                          f"mae={m_cls['mae']:.2f}d  w1={m_cls['within1']:.3f}  "
                          f"bias={m_cls['bias']:+.2f}d  n={m_cls['n']}")
                    print(f"    {scheme}  ML_reg  acc={m_reg['acc']:.3f}  "
                          f"mae={m_reg['mae']:.2f}d  w1={m_reg['within1']:.3f}  "
                          f"bias={m_reg['bias']:+.2f}d  n={m_reg['n']}")
                else:
                    print(f"    {scheme}  ML      insufficient data")
 
                # Rule-B
                rule_res = _run_rule_method_grid(df_raw_g, df_feat_g, scheme, site,
                                                  coarsen_n=coarsen_n)
                if rule_res:
                    actuals    = [r['actual']    for r in rule_res]
                    preds_rule = [r['pred_rule'] for r in rule_res]
                    m_rule = _compute_metrics(actuals, preds_rule)
                    all_records.append({
                        'grid': grid_size, 'site': site, 'scheme': scheme,
                        'method': 'Rule_B', **{k: m_rule[k] for k in ['acc','mae','within1','bias','n']},
                    })
                    print(f"    {scheme}  Rule_B  acc={m_rule['acc']:.3f}  "
                          f"mae={m_rule['mae']:.2f}d  w1={m_rule['within1']:.3f}  "
                          f"bias={m_rule['bias']:+.2f}d  n={m_rule['n']}")
                else:
                    print(f"    {scheme}  Rule_B  insufficient data")
 
    results_df = pd.DataFrame(all_records)
    results_df = results_df.rename(columns={'n': 'n_test'})
 
    # Print summary tables
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    overall = (results_df.groupby(['site','method'])[['acc','mae','within1','bias']]
               .mean().round(3))
    print("\nOverall (mean across all grids x schemes):")
    print(overall.to_string())
 
    by_grid = (results_df.groupby(['grid','method'])[['acc','mae','within1','bias']]
               .mean().round(3).unstack('method'))
    print("\nBy grid size (mean across schemes x sites):")
    print(by_grid.to_string())
 
    by_scheme = (results_df.groupby(['scheme','method'])[['acc','mae','within1','bias']]
                 .mean().round(3).unstack('method'))
    print("\nBy scheme (mean across grids x sites):")
    print(by_scheme.to_string())
 
    return results_df
 
 
def plot_grid_comparison(results_df: pd.DataFrame,
                          grid_sizes: list = None,
                          schemes: list = None):
    """
    Visualize grid comparison results:
      Plot 1/2 : Accuracy by grid size, per site
      Plot 3/4 : Accuracy by scheme, per site
      Plot 5   : MAE heatmap (grid x method)
      Plot 6   : Within-1 rate by method
      Plot 7   : Bias by method x grid size
    """
    import matplotlib.gridspec as gs_mod
 
    if grid_sizes is None:
        grid_sizes = sorted(results_df['grid'].unique())
    if schemes is None:
        schemes = sorted(results_df['scheme'].unique())
 
    SITES = ['SantaMaria', 'Salinas']
    METHOD_COLORS = {
        'ML_RF_cls': '#5B8DB8',
        'ML_RF_reg': '#3A6186',
        'Rule_B':    '#E07B39',
    }
 
    fig = plt.figure(figsize=(18, 18))
    gs  = gs_mod.GridSpec(4, 2, hspace=0.45, wspace=0.35)
 
    for col, site in enumerate(SITES):
        ax = fig.add_subplot(gs[0, col])
        sub = results_df[results_df['site'] == site]
        gm  = sub.groupby(['grid','method'])['acc'].mean().unstack('method')
        x   = np.arange(len(grid_sizes)); bw = 0.25
        for k, method in enumerate(['ML_RF_cls','ML_RF_reg','Rule_B']):
            if method in gm.columns:
                vals = [gm.loc[g, method] if g in gm.index else np.nan for g in grid_sizes]
                ax.bar(x + (k-1)*bw, vals, width=bw, color=METHOD_COLORS[method],
                       label=method, edgecolor='white', linewidth=0.5)
        ax.set_xticks(x); ax.set_xticklabels(grid_sizes)
        ax.set_ylabel('Accuracy')
        ax.set_title(f'{site} — Accuracy by Grid Size', fontweight='bold')
        ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.3); ax.set_ylim(0, 1.05)
 
    for col, site in enumerate(SITES):
        ax = fig.add_subplot(gs[1, col])
        sub = results_df[results_df['site'] == site]
        sm  = sub.groupby(['scheme','method'])['acc'].mean().unstack('method')
        x   = np.arange(len(schemes)); bw = 0.25
        for k, method in enumerate(['ML_RF_cls','ML_RF_reg','Rule_B']):
            if method in sm.columns:
                vals = [sm.loc[s, method] if s in sm.index else np.nan for s in schemes]
                ax.bar(x + (k-1)*bw, vals, width=bw, color=METHOD_COLORS[method],
                       label=method, edgecolor='white', linewidth=0.5)
        ax.set_xticks(x); ax.set_xticklabels(schemes)
        ax.set_ylabel('Accuracy')
        ax.set_title(f'{site} — Accuracy by Scheme', fontweight='bold')
        ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.3); ax.set_ylim(0, 1.05)
 
    ax5 = fig.add_subplot(gs[2, 0])
    mae_pivot = (results_df.groupby(['grid','method'])['mae']
                 .mean().unstack('method')
                 .reindex(grid_sizes)[['ML_RF_cls','ML_RF_reg','Rule_B']])
    vmax = mae_pivot.values[~np.isnan(mae_pivot.values)].max()
    im = ax5.imshow(mae_pivot.values, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=vmax)
    ax5.set_xticks(range(len(mae_pivot.columns)))
    ax5.set_xticklabels(mae_pivot.columns, rotation=15, ha='right')
    ax5.set_yticks(range(len(grid_sizes))); ax5.set_yticklabels(grid_sizes)
    ax5.set_title('MAE Heatmap (grid x method)\ndarker = worse', fontweight='bold')
    plt.colorbar(im, ax=ax5, label='MAE (days)')
    for i in range(len(grid_sizes)):
        for j in range(len(mae_pivot.columns)):
            val = mae_pivot.values[i, j]
            if not np.isnan(val):
                ax5.text(j, i, f'{val:.2f}', ha='center', va='center',
                         fontsize=9, fontweight='bold', color='white')
 
    ax6 = fig.add_subplot(gs[2, 1])
    w1  = results_df.groupby(['method','site'])['within1'].mean().unstack('site')
    methods = w1.index.tolist(); x = np.arange(len(methods)); bw = 0.35
    for k, site in enumerate(SITES):
        if site in w1.columns:
            ax6.bar(x + (k-0.5)*bw, w1[site].values, width=bw, label=site,
                    color=['#5B8DB8','#E07B39'][k], edgecolor='white', linewidth=0.5)
    ax6.axhline(1.0, color='#2d6a3f', ls='--', lw=1.5, label='perfect')
    ax6.set_xticks(x); ax6.set_xticklabels(methods, rotation=15, ha='right')
    ax6.set_ylabel('Within-1 Rate')
    ax6.set_title('Within-1 Rate by Method', fontweight='bold')
    ax6.legend(fontsize=8); ax6.grid(axis='y', alpha=0.3); ax6.set_ylim(0, 1.1)
 
    ax7 = fig.add_subplot(gs[3, :])
    bias_data = (results_df.groupby(['method','grid'])['bias']
                 .mean().unstack('grid').reindex(columns=grid_sizes))
    x = np.arange(len(bias_data.index)); bw = 0.18
    for k, grid in enumerate(grid_sizes):
        if grid in bias_data.columns:
            ax7.bar(x + (k-1.5)*bw, bias_data[grid].values, width=bw,
                    label=grid, edgecolor='white', linewidth=0.5)
    ax7.axhline(0, color='black', lw=1.5)
    ax7.set_xticks(x); ax7.set_xticklabels(bias_data.index, rotation=15, ha='right')
    ax7.set_ylabel('Bias (days)  + = recommends longer')
    ax7.set_title('Bias by Method x Grid Size', fontweight='bold')
    ax7.legend(fontsize=8, title='Grid'); ax7.grid(axis='y', alpha=0.3)
 
    fig.suptitle('Stage 2 — Grid-Level ML vs Rule-B\n'
                 'Comparison across Grid Sizes x ABCDE Schemes',
                 fontsize=13, fontweight='bold')
    plt.show()
 