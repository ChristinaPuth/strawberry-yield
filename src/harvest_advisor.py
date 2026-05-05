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
harvest_advisor.py
------------------
Optimal harvest timing advisor — v2.

Key changes vs v1:
  - No more enumeration (+3 to +7 days)
  - Model 2 directly predicts optimal_days for each grid cell
  - Field-level recommendation = median of all cell predictions
  - days_since_last removed from inference features (it is now an output)
  - New: plot_days_map() shows spatial distribution of predicted days
  - New: plot_ground_truth_vs_pred() for validation dates

Usage:
    advice = recommend_harvest(
        model_results = model_results_sm,   # dict with 'yield' and 'days'
        df_raw        = df_sm,
        weather       = weather_sm,
        site          = "SantaMaria",
        last_harvest_date = pd.Timestamp("2024-07-09"),
    )
    plot_advice(advice, site="SantaMaria")
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import timedelta
import warnings
warnings.filterwarnings("ignore")

# ── Feature groups (days_since_last removed) ─────────────────────────────────

FEAT_COLS = [
    # Temporal (7)
    "yield_lag1", "yield_lag2", "yield_lag3",
    "rolling_mean_3", "yield_trend", "season_cumulative", "day_of_year",
    # Spatial (4)
    "field_x", "field_y", "neighbor_mean_3x3", "neighbor_mean_5x5",
    # Weather (9)
    "temp_mean_7d", "temp_max_7d", "temp_min_7d",
    "precip_7d", "et0_7d", "humidity_mean_7d",
    "soil_moisture_0_7", "soil_moisture_7_28", "daylight_7d",
]


# ── 1. Build inference features for candidate harvest day ────────────────────

def _build_inference_row(df_raw: pd.DataFrame,
                          last_date: pd.Timestamp,
                          candidate_date: pd.Timestamp,
                          weather: pd.DataFrame,
                          lag_depth: int = 3) -> pd.DataFrame:
    """
    Build feature DataFrame for all grid cells for a candidate harvest date.
    days_since_last is NOT included in features (it is a model output now).
    day_of_year of the candidate date IS included (no leakage).
    """
    all_dates  = sorted(df_raw["harvest_date"].unique())
    past_dates = [d for d in all_dates if d <= last_date]

    if len(past_dates) < lag_depth:
        raise ValueError(
            f"Need at least {lag_depth} past harvests. "
            f"Only {len(past_dates)} available before {last_date.date()}."
        )

    lag_dates = past_dates[-lag_depth:]

    pivot = df_raw[df_raw["harvest_date"].isin(lag_dates)].pivot_table(
        index=["field_x","field_y","easting","northing"],
        columns="harvest_date",
        values="weight_kg",
        aggfunc="first",
    ).reset_index()
    pivot.columns.name = None

    for k, d in enumerate(reversed(lag_dates), start=1):
        if d in pivot.columns:
            pivot.rename(columns={d: f"yield_lag{k}"}, inplace=True)

    for k in range(1, lag_depth + 1):
        if f"yield_lag{k}" not in pivot.columns:
            pivot[f"yield_lag{k}"] = 0.0

    pivot["rolling_mean_3"] = (
        pivot["yield_lag1"] + pivot["yield_lag2"] + pivot["yield_lag3"]
    ) / 3.0
    pivot["yield_trend"] = (pivot["yield_lag1"] - pivot["yield_lag3"]) / 2.0

    all_past = df_raw[df_raw["harvest_date"] <= last_date]
    cum = all_past.groupby(["field_x","field_y"])["weight_kg"].sum().reset_index()
    cum.columns = ["field_x","field_y","season_cumulative"]
    pivot = pivot.merge(cum, on=["field_x","field_y"], how="left")
    pivot["season_cumulative"] = pivot["season_cumulative"].fillna(0)

    # day_of_year of candidate date (feature, not leakage)
    pivot["day_of_year"] = candidate_date.dayofyear

    # NOTE: days_since_last is NOT added here — it is now a model output

    # Neighbour means from lag1
    pivot["neighbor_mean_3x3"] = _neighbor_means_from_df(pivot, "yield_lag1", 3)
    pivot["neighbor_mean_5x5"] = _neighbor_means_from_df(pivot, "yield_lag1", 5)

    # Weather: 7 days before candidate date
    end_w   = candidate_date - timedelta(days=1)
    start_w = candidate_date - timedelta(days=7)
    w = weather[(weather.index >= start_w) & (weather.index <= end_w)]

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
    half   = window // 2
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
                if dx == 0 and dy == 0:
                    continue
                nx, ny = xi+dx, yi+dy
                if nx in i2x and ny in i2y:
                    v = lookup.get((i2x[nx], i2y[ny]))
                    if v is not None:
                        vals.append(v)
        result[idx] = float(np.mean(vals)) if vals else 0.0
    return pd.Series(result)


# ── 2. Run both models on inference DataFrame ─────────────────────────────────

def _predict_both(inf_df: pd.DataFrame,
                   model_results: dict) -> tuple:
    """
    Returns:
        pred_yield : np.ndarray  — per-cell predicted weight_kg
        pred_days  : np.ndarray  — per-cell predicted optimal_days
    """
    # — yield ——————————————————————————————————————————————————————————————————
    best_y   = model_results["yield"].iloc[0]
    m_y      = best_y["_model_obj"]
    feats    = best_y["features"]
    log_t    = best_y["log_target"]
    avail    = [f for f in feats if f in inf_df.columns]
    X        = inf_df[avail].values.astype(np.float32)
    p_yield  = m_y.predict(X)
    if log_t:
        p_yield = np.expm1(p_yield)
    p_yield = np.clip(p_yield, 0, None)

    # — days ———————————————————————————————————————————————————————————————————
    best_d  = model_results["days"].iloc[0]
    m_d     = best_d["_model_obj"]
    p_days  = np.clip(m_d.predict(X), 1, 14)

    return p_yield, p_days


# ── 3. Main recommendation function ──────────────────────────────────────────

def recommend_harvest(model_results: dict,
                       df_raw: pd.DataFrame,
                       weather: pd.DataFrame,
                       site: str,
                       last_harvest_date,
                       lag_depth: int = 3) -> dict:
    """
    Directly predict optimal harvest interval using Model 2.

    No enumeration — model predicts days for each cell,
    field-level recommendation = median across all cells.

    Parameters
    ----------
    model_results     : dict with 'yield' and 'days' DataFrames
                        (output of models.run_model_comparison)
    df_raw            : raw DataFrame from data_pipeline
    weather           : DataFrame from feature_engineering.fetch_weather
    site              : 'SantaMaria' or 'Salinas'
    last_harvest_date : date of most recent actual harvest

    Returns
    -------
    dict with keys:
        optimal_days        : int (field-level recommendation)
        optimal_date        : pd.Timestamp
        pred_yield          : np.ndarray (per-cell predicted yield)
        pred_days           : np.ndarray (per-cell predicted days)
        inference_df        : pd.DataFrame (feature rows used)
        days_distribution   : pd.Series (value_counts of pred_days)
        site, last_harvest_date
    """
    last_harvest_date = pd.Timestamp(last_harvest_date)

    # Build inference features for the "next harvest" scenario
    # We don't know the exact date yet — use last_date + median predicted days
    # So we first predict with a placeholder date, get predicted days,
    # then refine the candidate date

    # Step 1: placeholder candidate = last + 5 days (neutral starting point)
    placeholder_date = last_harvest_date + timedelta(days=5)
    inf_df = _build_inference_row(
        df_raw, last_harvest_date, placeholder_date, weather, lag_depth
    )

    # Step 2: model predicts both yield and days
    pred_yield, pred_days = _predict_both(inf_df, model_results)

    # Step 3: field-level days = median of all cell predictions
    optimal_days = int(np.round(np.median(pred_days)))
    optimal_date = last_harvest_date + timedelta(days=optimal_days)

    # Step 4: rebuild inference with the actual predicted date for final yield map
    inf_df_final = _build_inference_row(
        df_raw, last_harvest_date, optimal_date, weather, lag_depth
    )
    pred_yield_final, pred_days_final = _predict_both(inf_df_final, model_results)

    days_dist = pd.Series(
        np.round(pred_days_final).astype(int)
    ).value_counts().sort_index()

    total_yield = float(pred_yield_final.sum())
    mean_yield  = float(pred_yield_final.mean())

    print(f"\n{'='*55}")
    print(f"  HARVEST RECOMMENDATION — {site}")
    print(f"  Last harvest   : {last_harvest_date.date()}")
    print(f"  Optimal days   : {optimal_days} days")
    print(f"  Optimal date   : {optimal_date.date()}")
    print(f"  Expected yield : {total_yield:,.1f} kg (total field)")
    print(f"  Mean per cell  : {mean_yield:.4f} kg")
    print(f"  Days distribution (cells):")
    for d, cnt in days_dist.items():
        print(f"    {d} days: {cnt:,} cells")
    print(f"{'='*55}\n")

    return {
        "optimal_days":      optimal_days,
        "optimal_date":      optimal_date,
        "pred_yield":        pred_yield_final,
        "pred_days":         pred_days_final,
        "inference_df":      inf_df_final,
        "days_distribution": days_dist,
        "site":              site,
        "last_harvest_date": last_harvest_date,
    }


# ── 4. Plot yield map for optimal day ────────────────────────────────────────

def _make_grid(df, values, x_col="field_x", y_col="field_y"):
    x_vals = sorted(df[x_col].unique())
    y_vals = sorted(df[y_col].unique())
    x2i = {v: i for i, v in enumerate(x_vals)}
    y2i = {v: i for i, v in enumerate(y_vals)}
    grid = np.zeros((len(y_vals), len(x_vals)))
    for (_, row), v in zip(df.iterrows(), values):
        grid[y2i[row[y_col]], x2i[row[x_col]]] = v
    return grid


def plot_advice(advice: dict, figsize=(16, 10)):
    """
    Two-panel figure:
      Top-left  : predicted yield map for optimal day
      Top-right : days map (per-cell predicted interval)
      Bottom    : summary text + days distribution bar chart
    """
    site      = advice["site"]
    inf_df    = advice["inference_df"]
    pred_y    = advice["pred_yield"]
    pred_d    = advice["pred_days"]
    opt_days  = advice["optimal_days"]
    opt_date  = advice["optimal_date"]
    last_date = advice["last_harvest_date"]
    days_dist = advice["days_distribution"]

    grid_yield = _make_grid(inf_df, pred_y)
    grid_days  = _make_grid(inf_df, pred_d)

    vmax_y = float(np.nanquantile(pred_y[pred_y > 0], 0.99)) if (pred_y > 0).any() else 1.0

    fig = plt.figure(figsize=figsize)
    gs  = gridspec.GridSpec(2, 2, height_ratios=[2, 1], hspace=0.45, wspace=0.35)

    # ── Yield map ─────────────────────────────────────────────────────────────
    ax_y = fig.add_subplot(gs[0, 0])
    im_y = ax_y.imshow(grid_yield, cmap="YlOrRd", aspect="auto", vmin=0, vmax=vmax_y)
    plt.colorbar(im_y, ax=ax_y, label="kg", shrink=0.85)
    ax_y.set_title(f"Predicted Yield Map\n{opt_date.date()}  (+{opt_days} days)",
                   fontsize=11, fontweight="bold")
    ax_y.set_xlabel("field_x"); ax_y.set_ylabel("field_y")
    ax_y.text(0.02, 0.97, f"Total: {pred_y.sum():,.0f} kg",
              transform=ax_y.transAxes, fontsize=9, va="top", color="white",
              bbox=dict(boxstyle="round,pad=0.3", fc="#333", alpha=0.65))

    # ── Days map ──────────────────────────────────────────────────────────────
    ax_d = fig.add_subplot(gs[0, 1])
    im_d = ax_d.imshow(grid_days, cmap="RdYlGn_r", aspect="auto",
                        vmin=1, vmax=10)
    plt.colorbar(im_d, ax=ax_d, label="days", shrink=0.85)
    ax_d.set_title(f"Days Map (per-cell optimal interval)\nMedian = {opt_days} days",
                   fontsize=11, fontweight="bold")
    ax_d.set_xlabel("field_x"); ax_d.set_ylabel("field_y")

    # ── Days distribution bar ─────────────────────────────────────────────────
    ax_bar = fig.add_subplot(gs[1, 0])
    colours = ["#2d6a3f" if d == opt_days else "#cbd5c0" for d in days_dist.index]
    ax_bar.bar([str(d) for d in days_dist.index], days_dist.values,
               color=colours, edgecolor="white")
    ax_bar.set_xlabel("Predicted days ahead"); ax_bar.set_ylabel("Number of cells")
    ax_bar.set_title("Cell-level days distribution", fontsize=10)
    ax_bar.grid(axis="y", alpha=0.3)

    # ── Summary text ──────────────────────────────────────────────────────────
    ax_txt = fig.add_subplot(gs[1, 1])
    ax_txt.axis("off")
    summary = (
        f"HARVEST RECOMMENDATION\n"
        f"{'─'*30}\n"
        f"Site          : {site}\n"
        f"Last harvest  : {last_date.date()}\n"
        f"Optimal date  : {opt_date.date()}\n"
        f"Days to wait  : {opt_days} days\n"
        f"Expected yield: {pred_y.sum():,.0f} kg\n"
        f"Mean/cell     : {pred_y.mean():.4f} kg\n"
        f"Active cells  : {(pred_y > 0).sum():,}"
    )
    ax_txt.text(0.05, 0.95, summary, transform=ax_txt.transAxes,
                fontsize=10, va="top", fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.6", fc="#f0f7f0", alpha=0.9))

    fig.suptitle(f"{site} — Harvest Advisor  (last harvest: {last_date.date()})",
                 fontsize=13, fontweight="bold")
    plt.show()
    return fig


# ── 5. Ground truth vs predicted (for validation/test dates) ─────────────────

def plot_ground_truth_vs_pred(model_results: dict,
                               df_feat: pd.DataFrame,
                               site: str,
                               target: str = "yield"):
    """
    Line chart: ground truth vs predicted values across all test/val harvest dates.

    target = 'yield' → total field yield per harvest date
    target = 'days'  → actual vs predicted days_since_last per harvest date
    """
    import matplotlib.dates as mdates
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

    best_row = model_results[target].iloc[0]
    model    = best_row["_model_obj"]
    features = best_row["features"]
    log_t    = best_row.get("log_target", False) if target == "yield" else False
    t_col    = "weight_kg" if target == "yield" else "optimal_days"

    avail = [f for f in features if f in df_feat.columns]
    X     = df_feat[avail].values.astype(np.float32)
    y_true = df_feat[t_col].values.astype(np.float32)

    y_pred = model.predict(X)
    if log_t:
        y_pred = np.expm1(y_pred)
    if target == "days":
        y_pred = np.clip(y_pred, 1, 14)
    else:
        y_pred = np.clip(y_pred, 0, None)

    df_plot = df_feat[["harvest_date"]].copy()
    df_plot["y_true"] = y_true
    df_plot["y_pred"] = y_pred

    if target == "yield":
        # Aggregate to field level: total yield per harvest date
        agg = df_plot.groupby("harvest_date").agg(
            true_total=("y_true","sum"),
            pred_total=("y_pred","sum"),
        ).reset_index().sort_values("harvest_date")
        y_t = agg["true_total"].values
        y_p = agg["pred_total"].values
        dates = agg["harvest_date"]
        ylabel = "Total field yield (kg)"
        title  = f"{site} — Ground Truth vs Predicted Yield per Harvest"
    else:
        # Field level: one value per harvest date (all cells share same label)
        agg = df_plot.groupby("harvest_date").agg(
            true_days =("y_true","first"),
            pred_days =("y_pred","median"),
        ).reset_index().sort_values("harvest_date")
        y_t = agg["true_days"].values
        y_p = agg["pred_days"].values
        dates = agg["harvest_date"]
        ylabel = "Days since last harvest"
        title  = f"{site} — Ground Truth vs Predicted Optimal Days"

    rmse = float(np.sqrt(mean_squared_error(y_t, y_p)))
    mae  = float(mean_absolute_error(y_t, y_p))
    r2   = float(r2_score(y_t, y_p))

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(dates, y_t, "o-",  color="#2d5a3d", linewidth=2,
            markersize=6, label="Ground Truth", zorder=3)
    ax.plot(dates, y_p, "s--", color="#E07B39", linewidth=2,
            markersize=6, label="Predicted",    zorder=3)
    ax.fill_between(dates, y_t, y_p, alpha=0.12, color="#E07B39")

    ax.set_xlabel("Harvest date", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    plt.xticks(rotation=45)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    ax.text(0.02, 0.96,
            f"RMSE={rmse:.3f}   MAE={mae:.3f}   R²={r2:.3f}",
            transform=ax.transAxes, fontsize=10, va="top",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.85))

    plt.tight_layout()
    plt.show()
    return fig


# ── 6. Quick print ────────────────────────────────────────────────────────────

def print_recommendation(advice: dict):
    print("\n" + "="*50)
    print(f"  HARVEST RECOMMENDATION — {advice['site']}")
    print("="*50)
    print(f"  Last harvest  : {advice['last_harvest_date'].date()}")
    print(f"  Optimal date  : {advice['optimal_date'].date()}")
    print(f"  Days to wait  : {advice['optimal_days']} days")
    print(f"  Expected yield: {advice['pred_yield'].sum():,.1f} kg")
    print(f"  Mean/cell     : {advice['pred_yield'].mean():.4f} kg")
    print(f"\n  Days distribution:")
    for d, cnt in advice["days_distribution"].items():
        bar = "█" * (cnt // max(advice["days_distribution"].values // 20 + 1, 1))
        marker = " ← optimal" if d == advice["optimal_days"] else ""
        print(f"    {d:2d} days: {cnt:6,} cells  {bar}{marker}")
    print("="*50 + "\n")