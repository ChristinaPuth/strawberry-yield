# """
# harvest_advisor.py
# ------------------
# Optimal harvest timing advisor for strawberry yield prediction.

# Given the most recent harvest data, this module:
# 1. Predicts yield maps for candidate harvest days (t+3 to t+7)
# 2. Computes total predicted yield for each candidate day
# 3. Applies an over-ripening penalty for cells that have been
#    high-yield for multiple consecutive harvests
# 4. Returns the optimal harvest window and predicted yield map

# Usage in Colab:
#     import importlib, harvest_advisor as ha
#     importlib.reload(ha)

#     advice = ha.recommend_harvest(
#         model_results = model_results_sm,
#         df_feat       = df_feat_sm,
#         df_raw        = df_sm,
#         weather       = weather_sm,
#         site          = "SantaMaria",
#         last_harvest_date = pd.Timestamp("2024-07-09"),
#         candidate_days    = [3, 4, 5, 6, 7],
#     )
#     ha.plot_advice(advice, site="SantaMaria")
# """

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
# from datetime import timedelta
# import warnings
# warnings.filterwarnings("ignore")


# # ── Feature groups (must match models.py) ────────────────────────────────────

# TEMPORAL_FEATS = [
#     "yield_lag1", "yield_lag2", "yield_lag3",
#     "rolling_mean_3", "yield_trend",
#     "season_cumulative", "days_since_last", "day_of_year",
# ]
# SPATIAL_FEATS = [
#     "field_x", "field_y",
#     "neighbor_mean_3x3", "neighbor_mean_5x5",
# ]
# WEATHER_FEATS = [
#     "temp_mean_7d", "temp_max_7d", "temp_min_7d",
#     "precip_7d", "et0_7d", "humidity_mean_7d",
#     "soil_moisture_0_7", "soil_moisture_7_28", "daylight_7d",
# ]


# # ── 1. Build inference features for a candidate harvest day ──────────────────

# def _build_inference_row(df_raw: pd.DataFrame,
#                           last_date: pd.Timestamp,
#                           candidate_date: pd.Timestamp,
#                           weather: pd.DataFrame,
#                           lag_depth: int = 3) -> pd.DataFrame:
#     """
#     Build a feature DataFrame for all grid cells assuming harvest
#     happens on `candidate_date`, given that the last actual harvest
#     was on `last_date`.

#     Uses the 3 most recent actual harvests for lag features.
#     Weather features are computed from the 7 days before candidate_date.
#     """
#     # get the last lag_depth harvest dates up to and including last_date
#     all_dates = sorted(df_raw["harvest_date"].unique())
#     past_dates = [d for d in all_dates if d <= last_date]

#     if len(past_dates) < lag_depth:
#         raise ValueError(
#             f"Need at least {lag_depth} past harvests. "
#             f"Only {len(past_dates)} available before {last_date.date()}."
#         )

#     # use the most recent lag_depth dates
#     lag_dates = past_dates[-lag_depth:]   # [t-2, t-1, t]

#     # pivot: one row per cell, columns = each harvest date's weight
#     pivot = df_raw[df_raw["harvest_date"].isin(lag_dates)].pivot_table(
#         index=["field_x", "field_y", "easting", "northing"],
#         columns="harvest_date",
#         values="weight_kg",
#         aggfunc="first",
#     ).reset_index()
#     pivot.columns.name = None

#     # rename date columns to lag1/2/3
#     # lag1 = most recent (last_date), lag2 = one before, lag3 = two before
#     for k, d in enumerate(reversed(lag_dates), start=1):
#         if d in pivot.columns:
#             pivot.rename(columns={d: f"yield_lag{k}"}, inplace=True)

#     # fill any missing lag columns with 0
#     for k in range(1, lag_depth + 1):
#         if f"yield_lag{k}" not in pivot.columns:
#             pivot[f"yield_lag{k}"] = 0.0

#     # derived temporal features
#     pivot["rolling_mean_3"] = (
#         pivot["yield_lag1"] + pivot["yield_lag2"] + pivot["yield_lag3"]
#     ) / 3.0
#     pivot["yield_trend"] = (pivot["yield_lag1"] - pivot["yield_lag3"]) / 2.0

#     # season_cumulative: sum of all yields up to and including lag1
#     all_past = df_raw[df_raw["harvest_date"] <= last_date]
#     cum = all_past.groupby(["field_x", "field_y"])["weight_kg"].sum().reset_index()
#     cum.columns = ["field_x", "field_y", "season_cumulative"]
#     pivot = pivot.merge(cum, on=["field_x", "field_y"], how="left")
#     pivot["season_cumulative"] = pivot["season_cumulative"].fillna(0)

#     # days_since_last: days from last actual harvest to candidate date
#     pivot["days_since_last"] = (candidate_date - last_date).days

#     # day_of_year for candidate date
#     pivot["day_of_year"] = candidate_date.dayofyear

#     # neighbour means (3x3 and 5x5) using lag1 as the "current" yield
#     pivot["neighbor_mean_3x3"] = _neighbor_means_from_df(pivot, "yield_lag1", 3)
#     pivot["neighbor_mean_5x5"] = _neighbor_means_from_df(pivot, "yield_lag1", 5)

#     # weather: 7-day aggregate before candidate_date
#     end_w   = candidate_date - timedelta(days=1)
#     start_w = candidate_date - timedelta(days=7)
#     w = weather[(weather.index >= start_w) & (weather.index <= end_w)]

#     if len(w) == 0:
#         for col in WEATHER_FEATS:
#             pivot[col] = np.nan
#     else:
#         pivot["temp_mean_7d"]      = float(w["temp_mean"].mean())
#         pivot["temp_max_7d"]       = float(w["temp_max"].max())
#         pivot["temp_min_7d"]       = float(w["temp_min"].min())
#         pivot["precip_7d"]         = float(w["precip"].sum())
#         pivot["et0_7d"]            = float(w["et0"].sum())
#         pivot["humidity_mean_7d"]  = float(w["humidity_mean"].mean())
#         pivot["soil_moisture_0_7"] = float(w["soil_moisture_0_7"].mean())
#         pivot["soil_moisture_7_28"]= float(w["soil_moisture_7_28"].mean())
#         pivot["daylight_7d"]       = float(w["daylight_hours"].mean())

#     pivot["harvest_date"] = candidate_date
#     return pivot


# def _neighbor_means_from_df(df: pd.DataFrame,
#                               col: str,
#                               window: int = 3) -> pd.Series:
#     """Compute neighbourhood means for a given column in df."""
#     half   = window // 2
#     x_vals = sorted(df["field_x"].unique())
#     y_vals = sorted(df["field_y"].unique())
#     x2i    = {v: i for i, v in enumerate(x_vals)}
#     y2i    = {v: i for i, v in enumerate(y_vals)}
#     i2x    = {i: v for v, i in x2i.items()}
#     i2y    = {i: v for v, i in y2i.items()}
#     lookup = df.set_index(["field_x", "field_y"])[col].to_dict()

#     result = {}
#     for idx, row in df.iterrows():
#         xi   = x2i[row["field_x"]]
#         yi   = y2i[row["field_y"]]
#         vals = []
#         for dx in range(-half, half + 1):
#             for dy in range(-half, half + 1):
#                 if dx == 0 and dy == 0:
#                     continue
#                 nx, ny = xi + dx, yi + dy
#                 if nx in i2x and ny in i2y:
#                     v = lookup.get((i2x[nx], i2y[ny]))
#                     if v is not None:
#                         vals.append(v)
#         result[idx] = float(np.mean(vals)) if vals else 0.0
#     return pd.Series(result)


# # ── 2. Predict for one candidate day ─────────────────────────────────────────

# def _predict_one_day(inference_df: pd.DataFrame,
#                       model_results: pd.DataFrame) -> np.ndarray:
#     """
#     Run the best model on an inference DataFrame.
#     Returns predicted yield array (one value per grid cell).
#     """
#     best_row = model_results.iloc[0]
#     model    = best_row["_model_obj"]
#     features = best_row["features"]
#     log_t    = best_row["log_target"]

#     available = [f for f in features if f in inference_df.columns]
#     X = inference_df[available].values.astype(np.float32)

#     pred = model.predict(X)
#     if log_t:
#         pred = np.expm1(pred)
#     return np.clip(pred, 0, None)


# # ── 3. Over-ripening penalty ──────────────────────────────────────────────────

# def _overripe_penalty(inference_df: pd.DataFrame,
#                        penalty_weight: float = 0.05) -> np.ndarray:
#     """
#     Compute a per-cell over-ripening penalty.

#     Logic: cells that have been consistently high-yield for multiple
#     consecutive harvests are at higher risk of over-ripening if we wait.
#     Penalty = penalty_weight × (yield_lag1 × (yield_lag1 > 75th pct))

#     Returns a penalty array (same length as inference_df).
#     Subtract from predicted yield to get risk-adjusted yield.
#     """
#     lag1 = inference_df["yield_lag1"].values
#     threshold = np.percentile(lag1[lag1 > 0], 75) if (lag1 > 0).any() else 1.0
#     high_yield_mask = lag1 > threshold
#     penalty = penalty_weight * lag1 * high_yield_mask
#     return penalty


# # ── 4. Main recommendation function ──────────────────────────────────────────

# def recommend_harvest(model_results: pd.DataFrame,
#                        df_raw: pd.DataFrame,
#                        weather: pd.DataFrame,
#                        site: str,
#                        last_harvest_date,
#                        candidate_days: list = None,
#                        penalty_weight: float = 0.05,
#                        lag_depth: int = 3) -> dict:
#     """
#     Recommend the optimal harvest day from a set of candidates.

#     Parameters
#     ----------
#     model_results     : output of models.run_model_comparison()
#     df_raw            : raw DataFrame from data_pipeline.load_site()
#                         (used to get lag features from actual history)
#     weather           : DataFrame from feature_engineering.fetch_weather()
#     site              : 'SantaMaria' or 'Salinas'
#     last_harvest_date : date of the most recent actual harvest
#     candidate_days    : days ahead to evaluate, default [3,4,5,6,7]
#     penalty_weight    : over-ripening penalty strength (0 = no penalty)
#     lag_depth         : number of lag features to use

#     Returns
#     -------
#     dict with keys:
#         optimal_day        : int (days from last harvest)
#         optimal_date       : pd.Timestamp
#         summary_table      : DataFrame with one row per candidate day
#         yield_maps         : dict {days_ahead: predicted yield array}
#         inference_dfs      : dict {days_ahead: inference DataFrame}
#         site               : str
#         last_harvest_date  : pd.Timestamp
#     """
#     if candidate_days is None:
#         candidate_days = [3, 4, 5, 6, 7]

#     last_harvest_date = pd.Timestamp(last_harvest_date)
#     print(f"\nHarvest advisor — {site}")
#     print(f"Last harvest    : {last_harvest_date.date()}")
#     print(f"Candidates      : +{candidate_days} days")
#     print(f"Penalty weight  : {penalty_weight}")
#     print("-" * 55)
#     print(f"  {'Days':>5}  {'Date':>12}  "
#           f"{'Raw yield':>11}  {'Adjusted':>10}  {'Risk cells':>11}")
#     print(f"  {'-'*52}")

#     records      = []
#     yield_maps   = {}
#     inference_dfs = {}

#     for k in candidate_days:
#         candidate_date = last_harvest_date + timedelta(days=k)

#         # build inference features
#         inf_df = _build_inference_row(
#             df_raw, last_harvest_date, candidate_date, weather, lag_depth
#         )

#         # predict
#         pred = _predict_one_day(inf_df, model_results)

#         # penalty
#         penalty  = _overripe_penalty(inf_df, penalty_weight)
#         adjusted = pred - penalty

#         total_raw  = float(pred.sum())
#         total_adj  = float(adjusted.sum())
#         risk_cells = int((penalty > 0).sum())

#         print(f"  +{k:<4d}  {str(candidate_date.date()):>12}  "
#               f"{total_raw:>11.1f}  {total_adj:>10.1f}  {risk_cells:>11,}")

#         records.append({
#             "days_ahead":   k,
#             "date":         candidate_date,
#             "total_yield":  round(total_raw,  1),
#             "adj_yield":    round(total_adj,  1),
#             "risk_cells":   risk_cells,
#             "mean_yield":   round(float(pred.mean()), 4),
#             "max_yield":    round(float(pred.max()),  4),
#         })
#         yield_maps[k]    = pred
#         inference_dfs[k] = inf_df

#     summary = pd.DataFrame(records)
#     best_idx  = summary["adj_yield"].idxmax()
#     best_row  = summary.loc[best_idx]

#     print(f"\n{'='*55}")
#     print(f"  RECOMMENDATION:  Harvest on {best_row['date'].date()}")
#     print(f"  Days from now :  +{int(best_row['days_ahead'])} days")
#     print(f"  Expected yield:  {best_row['total_yield']:,.1f} kg (raw)")
#     print(f"  Adjusted yield:  {best_row['adj_yield']:,.1f} kg")
#     print(f"  Risk cells    :  {int(best_row['risk_cells']):,}")
#     print(f"{'='*55}\n")

#     return {
#         "optimal_day":       int(best_row["days_ahead"]),
#         "optimal_date":      best_row["date"],
#         "summary_table":     summary,
#         "yield_maps":        yield_maps,
#         "inference_dfs":     inference_dfs,
#         "site":              site,
#         "last_harvest_date": last_harvest_date,
#     }


# # ── 5. Visualisation ──────────────────────────────────────────────────────────

# def plot_advice(advice: dict, figsize=(18, 10)):
#     """
#     Two-panel visualisation:
#     - Top: bar chart of predicted total yield vs candidate days,
#            with optimal day highlighted and risk cells shown
#     - Bottom: predicted yield maps for each candidate day
#     """
#     site      = advice["site"]
#     summary   = advice["summary_table"]
#     maps      = advice["yield_maps"]
#     opt_day   = advice["optimal_day"]
#     last_date = advice["last_harvest_date"]
#     inf_dfs   = advice["inference_dfs"]

#     n_cands = len(summary)
#     fig = plt.figure(figsize=figsize)
#     gs  = gridspec.GridSpec(2, n_cands, height_ratios=[1.4, 2], hspace=0.45)

#     # ── top panel: bar chart spanning all columns ──
#     ax_bar = fig.add_subplot(gs[0, :])

#     colours = ["#E07B39" if d == opt_day else "#CBD5E1"
#                for d in summary["days_ahead"]]
#     bars = ax_bar.bar(
#         summary["days_ahead"].astype(str),
#         summary["adj_yield"],
#         color=colours, edgecolor="white", linewidth=0.5, zorder=2
#     )
#     ax_bar.set_xlabel("Days from last harvest", fontsize=11)
#     ax_bar.set_ylabel("Adj. predicted yield (kg)", fontsize=11)
#     ax_bar.set_title(
#         f"{site}  —  Harvest timing recommendation  "
#         f"(last harvest: {last_date.date()})",
#         fontsize=13, fontweight="bold"
#     )
#     ax_bar.grid(axis="y", alpha=0.3, zorder=1)

#     # annotate bars
#     for bar, (_, row) in zip(bars, summary.iterrows()):
#         ax_bar.text(
#             bar.get_x() + bar.get_width() / 2,
#             bar.get_height() + summary["adj_yield"].max() * 0.01,
#             f"{row['adj_yield']:,.0f} kg\n({row['risk_cells']:,} risk)",
#             ha="center", va="bottom", fontsize=8.5,
#             color="#E07B39" if row["days_ahead"] == opt_day else "#64748B"
#         )

#     # optimal marker
#     opt_bar = bars[summary["days_ahead"].tolist().index(opt_day)]
#     ax_bar.annotate(
#         "OPTIMAL",
#         xy=(opt_bar.get_x() + opt_bar.get_width() / 2,
#             opt_bar.get_height()),
#         xytext=(0, 28), textcoords="offset points",
#         ha="center", fontsize=9, fontweight="bold", color="#E07B39",
#         arrowprops=dict(arrowstyle="->", color="#E07B39", lw=1.5)
#     )

#     # ── bottom panel: yield maps ──
#     vmax = max(pred.max() for pred in maps.values())
#     vmax = float(np.quantile(
#         np.concatenate([p for p in maps.values()]), 0.99
#     ))

#     for col_idx, (k, pred) in enumerate(sorted(maps.items())):
#         ax = fig.add_subplot(gs[1, col_idx])
#         inf_df = inf_dfs[k]
#         cdate  = advice["last_harvest_date"] + timedelta(days=k)

#         # build grid
#         x_vals = sorted(inf_df["field_x"].unique())
#         y_vals = sorted(inf_df["field_y"].unique())
#         x2i = {v: i for i, v in enumerate(x_vals)}
#         y2i = {v: i for i, v in enumerate(y_vals)}
#         grid = np.zeros((len(y_vals), len(x_vals)))
#         for (_, row), p in zip(inf_df.iterrows(), pred):
#             grid[y2i[row["field_y"]], x2i[row["field_x"]]] = p

#         im = ax.imshow(grid, cmap="YlOrRd", aspect="auto",
#                        vmin=0, vmax=vmax)
#         plt.colorbar(im, ax=ax, label="kg", shrink=0.8)

#         marker = " ★ OPTIMAL" if k == opt_day else ""
#         adj    = summary.loc[summary["days_ahead"] == k, "adj_yield"].values[0]
#         ax.set_title(
#             f"+{k} days  ({cdate.date()}){marker}\n{adj:,.0f} kg",
#             fontsize=9,
#             fontweight="bold" if k == opt_day else "normal",
#             color="#E07B39" if k == opt_day else "black"
#         )
#         ax.set_xlabel("field_x", fontsize=8)
#         ax.set_ylabel("field_y", fontsize=8)
#         ax.tick_params(labelsize=7)

#     plt.suptitle(
#         f"Predicted yield maps for candidate harvest days",
#         fontsize=11, y=0.98
#     )
#     plt.show()
#     return fig


# # ── 6. Quick summary text ─────────────────────────────────────────────────────

# def print_recommendation(advice: dict):
#     """Print a farmer-friendly recommendation summary."""
#     s   = advice["summary_table"]
#     opt = s[s["days_ahead"] == advice["optimal_day"]].iloc[0]

#     print("\n" + "=" * 50)
#     print(f"  HARVEST RECOMMENDATION — {advice['site']}")
#     print("=" * 50)
#     print(f"  Last harvest   : {advice['last_harvest_date'].date()}")
#     print(f"  Optimal date   : {advice['optimal_date'].date()}")
#     print(f"  Days to wait   : {advice['optimal_day']} days")
#     print(f"  Expected yield : {opt['total_yield']:,.1f} kg")
#     print(f"  Overripe risk  : {opt['risk_cells']:,} cells at risk")
#     print()
#     print("  Full forecast:")
#     print(f"  {'Date':>12}  {'Yield (kg)':>11}  {'Risk cells':>11}")
#     print(f"  {'-'*38}")
#     for _, row in s.iterrows():
#         marker = " <-- optimal" if row["days_ahead"] == advice["optimal_day"] else ""
#         print(f"  {str(row['date'].date()):>12}  "
#               f"{row['total_yield']:>11.1f}  "
#               f"{row['risk_cells']:>11,}{marker}")
#     print("=" * 50 + "\n")


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
                          lag_depth: int = 3) -> pd.DataFrame:
    """Build feature DataFrame for all grid cells for a candidate date."""
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
                       lag_depth: int = 3) -> dict:
    """
    Two-stage harvest recommendation.

    Stage 1: Predict yield for each candidate day
    Stage 2: Apply growth-rate rule to select optimal day

    Parameters
    ----------
    model_results     : output of models.run_model_comparison()
    df_raw            : raw DataFrame from data_pipeline
    weather           : weather DataFrame from feature_engineering
    site              : 'SantaMaria' or 'Salinas'
    last_harvest_date : date of most recent actual harvest
    thresholds        : output of derive_thresholds()
    candidate_days    : days ahead to evaluate (default [3,4,5,6,7])

    Returns
    -------
    dict with full advice including yield maps for all candidates
    """
    if candidate_days is None:
        candidate_days = [3, 4, 5, 6, 7]

    last_harvest_date = pd.Timestamp(last_harvest_date)

    # Last actual total yield (for growth rate calculation)
    last_actual_total = df_raw[
        df_raw["harvest_date"] == last_harvest_date
    ]["weight_kg"].sum()

    print(f"\nHarvest Advisor — {site}")
    print(f"Last harvest    : {last_harvest_date.date()}")
    print(f"Last yield      : {last_actual_total:,.0f} kg")
    print(f"Candidates      : +{candidate_days} days")
    print(f"Thresholds      : t_low={thresholds['t_low']:.3f}  "
          f"t_high={thresholds['t_high']:.3f}")
    print("-" * 60)
    print(f"  {'Days':>5}  {'Date':>12}  {'Pred yield':>11}  "
          f"{'Growth rate':>12}  {'Stage 2 rule':>12}")
    print(f"  {'-'*57}")

    records      = []
    yield_maps   = {}
    inference_dfs = {}

    for k in candidate_days:
        candidate_date = last_harvest_date + timedelta(days=k)
        inf_df = _build_inference_row(
            df_raw, last_harvest_date, candidate_date, weather, lag_depth)
        pred   = _predict_yield(inf_df, model_results)

        pred_total   = float(pred.sum())
        growth_rate  = pred_total / last_actual_total if last_actual_total > 0 else 1.0
        stage2_days  = _apply_rule(growth_rate, thresholds)
        is_selected  = (k == stage2_days)

        marker = " ← SELECTED" if is_selected else ""
        print(f"  +{k:<4d}  {str(candidate_date.date()):>12}  "
              f"{pred_total:>11,.0f}  "
              f"{growth_rate:>12.3f}  "
              f"{stage2_days:>12d}{marker}")

        records.append({
            "days_ahead":  k,
            "date":        candidate_date,
            "pred_total":  round(pred_total, 1),
            "growth_rate": round(growth_rate, 4),
            "stage2_days": stage2_days,
        })
        yield_maps[k]     = pred
        inference_dfs[k]  = inf_df

    summary = pd.DataFrame(records)

    # Stage 2 decision: mode of stage2_days recommendations
    from collections import Counter
    stage2_votes  = Counter(summary["stage2_days"].tolist())
    optimal_days  = stage2_votes.most_common(1)[0][0]
    optimal_date  = last_harvest_date + timedelta(days=optimal_days)
    optimal_yield = summary[summary["days_ahead"]==optimal_days]["pred_total"].values[0]

    print(f"\n{'='*60}")
    print(f"  STAGE 2 RECOMMENDATION : {site}")
    print(f"  Last harvest  : {last_harvest_date.date()}")
    print(f"  Optimal date  : {optimal_date.date()}")
    print(f"  Days to wait  : {optimal_days} days")
    print(f"  Growth rate   : {summary[summary['days_ahead']==optimal_days]['growth_rate'].values[0]:.3f}")
    print(f"  Expected yield: {optimal_yield:,.0f} kg")
    print(f"{'='*60}\n")

    return {
        "optimal_days":       optimal_days,
        "optimal_date":       optimal_date,
        "pred_yield":         yield_maps.get(optimal_days, np.array([])),
        "summary_table":      summary,
        "yield_maps":         yield_maps,
        "inference_dfs":      inference_dfs,
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