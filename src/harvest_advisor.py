


# """
# harvest_advisor.py  v3
# ----------------------
# Stage 2: Rule-based optimal harvest timing advisor.

# Pipeline:
#   Stage 1 (models.py) → predicted yield map per candidate day
#   Stage 2 (this file) → growth-rate rules → optimal_days recommendation

# Key changes from v2:
#   - No trained model for days prediction
#   - Thresholds derived from training data statistics
#   - Decision Quality (DQ) calculation added
#   - Days map visualization added

# Usage:
#     # 1. Derive thresholds from training data (once)
#     thresholds = derive_thresholds(splits_sm["train"], site="SantaMaria")

#     # 2. Get recommendation
#     advice = recommend_harvest(
#         model_results     = model_results_sm,
#         df_raw            = df_sm,
#         weather           = weather_sm,
#         site              = "SantaMaria",
#         last_harvest_date = pd.Timestamp("2024-07-09"),
#         thresholds        = thresholds,
#     )

#     # 3. Evaluate Decision Quality (test set only)
#     dq = decision_quality(advice, df_raw=df_sm, splits=splits_sm)
# """

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
# from datetime import timedelta
# import warnings
# warnings.filterwarnings("ignore")

# from feature_engineering import ALL_FEATS, TEMPORAL_FEATS, SPATIAL_FEATS, WEATHER_FEATS


# # ── 1. Derive growth-rate thresholds from training data ───────────────────────

# def derive_thresholds(train_df: pd.DataFrame, site: str) -> dict:
#     """
#     Derive growth-rate thresholds for Stage 2 rules from training data.

#     Method:
#       1. Aggregate training data to field level (total yield per harvest)
#       2. Compute growth_rate = this_total / last_total
#       3. Group by days_since_last interval
#       4. Find the growth_rate percentiles that best separate intervals

#     Returns dict with threshold values and summary statistics.
#     """
#     # Field-level aggregation
#     field = train_df.groupby("harvest_date").agg(
#         total_kg      = ("weight_kg", "sum"),
#         days_interval = ("days_since_last", "first"),
#     ).dropna(subset=["days_interval"]).reset_index()

#     field = field.sort_values("harvest_date")
#     field["growth_rate"] = field["total_kg"] / field["total_kg"].shift(1)
#     field = field.dropna(subset=["growth_rate"])
#     field["days_interval"] = field["days_interval"].astype(int)

#     # Statistics per interval
#     stats = field.groupby("days_interval").agg(
#         count        = ("growth_rate", "count"),
#         mean_growth  = ("growth_rate", "mean"),
#         median_growth= ("growth_rate", "median"),
#         min_growth   = ("growth_rate", "min"),
#         max_growth   = ("growth_rate", "max"),
#     ).reset_index()

#     print(f"\nStage 2 threshold derivation — {site}")
#     print(f"Training harvests: {len(field)}")
#     print(f"\nGrowth rate by interval:")
#     print(stats.to_string(index=False))

#     # Derive thresholds
#     # Logic: if growth_rate is HIGH → yield still rising → wait longer
#     #        if growth_rate is LOW  → yield declining   → harvest soon
#     intervals = sorted(field["days_interval"].unique())

#     if len(intervals) >= 3:
#         # Three-class: short / medium / long
#         # Threshold between short and medium: median growth of shortest interval
#         short_int  = intervals[0]
#         long_int   = intervals[-1]
#         t_low  = float(field[field["days_interval"]==short_int]["growth_rate"].median())
#         t_high = float(field[field["days_interval"]==long_int]["growth_rate"].median())
#         classes = {intervals[0]: "short", intervals[1]: "medium", intervals[-1]: "long"}
#     elif len(intervals) == 2:
#         t_low  = float(field["growth_rate"].quantile(0.40))
#         t_high = float(field["growth_rate"].quantile(0.60))
#         classes = {intervals[0]: "short", intervals[-1]: "long"}
#     else:
#         t_low  = 0.9
#         t_high = 1.1
#         classes = {}

#     days_map = {
#         "short":  int(intervals[0]),
#         "medium": int(intervals[len(intervals)//2]) if len(intervals) >= 3 else int(intervals[0]),
#         "long":   int(intervals[-1]),
#     }

#     thresholds = {
#         "t_low":      round(t_low,  4),
#         "t_high":     round(t_high, 4),
#         "days_map":   days_map,
#         "intervals":  intervals,
#         "stats":      stats,
#         "site":       site,
#     }

#     print(f"\nDerived thresholds:")
#     print(f"  growth_rate >= {t_high:.3f} → wait {days_map['long']} days  (yield rising)")
#     print(f"  growth_rate >= {t_low:.3f}  → wait {days_map.get('medium', days_map['short'])} days  (yield stable)")
#     print(f"  growth_rate <  {t_low:.3f}  → wait {days_map['short']} days  (yield declining)")

#     return thresholds


# # ── 2. Build inference features for a candidate harvest day ───────────────────

# def _build_inference_row(df_raw: pd.DataFrame,
#                           last_date: pd.Timestamp,
#                           candidate_date: pd.Timestamp,
#                           weather: pd.DataFrame,
#                           lag_depth: int = 3,
#                           coarsen_n: int = 1) -> pd.DataFrame:
#     """
#     Build feature DataFrame for all grid cells for a candidate date.

#     Parameters
#     ----------
#     coarsen_n : grid coarsening factor used during training (1 = no coarsening).
#                 Must match the value used in fe.coarsen_grid() when building
#                 the training features.  If > 1, df_raw is coarsened before
#                 building lag / neighbour features so the inference grid matches
#                 the trained model's input space.
#     """
#     # Coarsen if needed — must match training
#     if coarsen_n > 1:
#         import feature_engineering as _fe
#         df_raw = _fe.coarsen_grid(df_raw, n=coarsen_n)

#     all_dates  = sorted(df_raw["harvest_date"].unique())
#     past_dates = [d for d in all_dates if d <= last_date]

#     if len(past_dates) < lag_depth:
#         raise ValueError(
#             f"Need {lag_depth} past harvests, only {len(past_dates)} available.")

#     lag_dates = past_dates[-lag_depth:]

#     pivot = df_raw[df_raw["harvest_date"].isin(lag_dates)].pivot_table(
#         index=["field_x","field_y","easting","northing"],
#         columns="harvest_date", values="weight_kg", aggfunc="first",
#     ).reset_index()
#     pivot.columns.name = None

#     for k, d in enumerate(reversed(lag_dates), start=1):
#         if d in pivot.columns:
#             pivot.rename(columns={d: f"yield_lag{k}"}, inplace=True)
#     for k in range(1, lag_depth+1):
#         if f"yield_lag{k}" not in pivot.columns:
#             pivot[f"yield_lag{k}"] = 0.0

#     pivot["rolling_mean_3"] = (
#         pivot["yield_lag1"]+pivot["yield_lag2"]+pivot["yield_lag3"]) / 3.0
#     pivot["yield_trend"] = (pivot["yield_lag1"] - pivot["yield_lag3"]) / 2.0

#     all_past = df_raw[df_raw["harvest_date"] <= last_date]
#     cum = all_past.groupby(["field_x","field_y"])["weight_kg"].sum().reset_index()
#     cum.columns = ["field_x","field_y","season_cumulative"]
#     pivot = pivot.merge(cum, on=["field_x","field_y"], how="left")
#     pivot["season_cumulative"] = pivot["season_cumulative"].fillna(0)
#     pivot["day_of_year"] = candidate_date.dayofyear

#     # Neighbour means from lag1
#     pivot["neighbor_mean_3x3"] = _neighbor_means_from_df(pivot, "yield_lag1", 3)
#     pivot["neighbor_mean_5x5"] = _neighbor_means_from_df(pivot, "yield_lag1", 5)

#     # Weather
#     w = weather[(weather.index >= candidate_date - timedelta(days=7)) &
#                 (weather.index <= candidate_date - timedelta(days=1))]
#     if len(w) == 0:
#         for col in ["temp_mean_7d","temp_max_7d","temp_min_7d","precip_7d",
#                     "et0_7d","humidity_mean_7d","soil_moisture_0_7",
#                     "soil_moisture_7_28","daylight_7d"]:
#             pivot[col] = np.nan
#     else:
#         pivot["temp_mean_7d"]       = float(w["temp_mean"].mean())
#         pivot["temp_max_7d"]        = float(w["temp_max"].max())
#         pivot["temp_min_7d"]        = float(w["temp_min"].min())
#         pivot["precip_7d"]          = float(w["precip"].sum())
#         pivot["et0_7d"]             = float(w["et0"].sum())
#         pivot["humidity_mean_7d"]   = float(w["humidity_mean"].mean())
#         pivot["soil_moisture_0_7"]  = float(w["soil_moisture_0_7"].mean())
#         pivot["soil_moisture_7_28"] = float(w["soil_moisture_7_28"].mean())
#         pivot["daylight_7d"]        = float(w["daylight_hours"].mean())

#     pivot["harvest_date"] = candidate_date
#     return pivot


# def _neighbor_means_from_df(df, col, window=3):
#     half = window // 2
#     x_vals = sorted(df["field_x"].unique())
#     y_vals = sorted(df["field_y"].unique())
#     x2i = {v: i for i, v in enumerate(x_vals)}
#     y2i = {v: i for i, v in enumerate(y_vals)}
#     i2x = {i: v for v, i in x2i.items()}
#     i2y = {i: v for v, i in y2i.items()}
#     lookup = df.set_index(["field_x","field_y"])[col].to_dict()
#     result = {}
#     for idx, row in df.iterrows():
#         xi, yi = x2i[row["field_x"]], y2i[row["field_y"]]
#         vals = []
#         for dx in range(-half, half+1):
#             for dy in range(-half, half+1):
#                 if dx == 0 and dy == 0: continue
#                 nx, ny = xi+dx, yi+dy
#                 if nx in i2x and ny in i2y:
#                     v = lookup.get((i2x[nx], i2y[ny]))
#                     if v is not None: vals.append(v)
#         result[idx] = float(np.mean(vals)) if vals else 0.0
#     return pd.Series(result)


# # ── 3. Predict yield for one candidate day ────────────────────────────────────

# def _predict_yield(inf_df: pd.DataFrame,
#                     model_results: pd.DataFrame) -> np.ndarray:
#     best  = model_results.iloc[0]
#     model = best["_model_obj"]
#     feats = best["features"]
#     log_t = best["log_target"]
#     avail = [f for f in feats if f in inf_df.columns]
#     X     = inf_df[avail].values.astype(np.float32)
#     pred  = model.predict(X)
#     if log_t: pred = np.expm1(pred)
#     return np.clip(pred, 0, None)


# # ── 4. Stage 2: Growth-rate rule ──────────────────────────────────────────────

# def _apply_rule(growth_rate: float, thresholds: dict) -> int:
#     """
#     Apply growth-rate rule to determine optimal days.

#     growth_rate = predicted_total / last_actual_total

#     >= t_high → yield still rising  → wait longer
#     >= t_low  → yield stable        → wait medium
#     <  t_low  → yield declining     → harvest soon
#     """
#     days_map = thresholds["days_map"]
#     t_high   = thresholds["t_high"]
#     t_low    = thresholds["t_low"]

#     if growth_rate >= t_high:
#         return days_map["long"]
#     elif growth_rate >= t_low:
#         return days_map.get("medium", days_map["short"])
#     else:
#         return days_map["short"]


# # ── 5. Main recommendation function ──────────────────────────────────────────

# def recommend_harvest(model_results: pd.DataFrame,
#                        df_raw: pd.DataFrame,
#                        weather: pd.DataFrame,
#                        site: str,
#                        last_harvest_date,
#                        thresholds: dict,
#                        candidate_days: list = None,
#                        lag_depth: int = 3,
#                        coarsen_n: int = 1) -> dict:
#     """
#     Two-stage harvest recommendation.

#     Stage 1: Predict yield for each candidate day
#     Stage 2: Apply growth-rate rule to select optimal day

#     Parameters
#     ----------
#     model_results     : output of models.run_model_comparison()
#     df_raw            : raw DataFrame from data_pipeline (always original 1x1).
#                         Coarsening happens internally; last_actual_total is
#                         always computed at true field scale.
#     weather           : weather DataFrame from feature_engineering
#     site              : 'SantaMaria' or 'Salinas'
#     last_harvest_date : date of most recent actual harvest
#     thresholds        : output of derive_thresholds()
#     candidate_days    : days ahead to evaluate (default [3,4,5,6,7])
#     coarsen_n         : grid coarsening factor used during training (1=none).
#                         Pass 3 if model was trained on 3x3 super-cells.

#     Returns
#     -------
#     dict with full advice including yield maps for all candidates
#     """
#     if candidate_days is None:
#         candidate_days = [3, 4, 5, 6, 7]

#     last_harvest_date = pd.Timestamp(last_harvest_date)

#     # Last actual total yield (for growth rate calculation)
#     last_actual_total = df_raw[
#         df_raw["harvest_date"] == last_harvest_date
#     ]["weight_kg"].sum()

#     print(f"\nHarvest Advisor — {site}")
#     print(f"Last harvest    : {last_harvest_date.date()}")
#     print(f"Last yield      : {last_actual_total:,.0f} kg")
#     print(f"Candidates      : +{candidate_days} days")
#     print(f"Thresholds      : t_low={thresholds['t_low']:.3f}  "
#           f"t_high={thresholds['t_high']:.3f}")
#     print("-" * 60)
#     print(f"  {'Days':>5}  {'Date':>12}  {'Pred yield':>11}  "
#           f"{'Growth rate':>12}  {'Stage 2 rule':>12}")
#     print(f"  {'-'*57}")

#     records      = []
#     yield_maps   = {}
#     inference_dfs = {}

#     for k in candidate_days:
#         candidate_date = last_harvest_date + timedelta(days=k)
#         inf_df = _build_inference_row(
#             df_raw, last_harvest_date, candidate_date, weather, lag_depth, coarsen_n)
#         pred   = _predict_yield(inf_df, model_results)

#         pred_total   = float(pred.sum())
#         growth_rate  = pred_total / last_actual_total if last_actual_total > 0 else 1.0
#         stage2_days  = _apply_rule(growth_rate, thresholds)
#         is_selected  = (k == stage2_days)

#         marker = " ← SELECTED" if is_selected else ""
#         print(f"  +{k:<4d}  {str(candidate_date.date()):>12}  "
#               f"{pred_total:>11,.0f}  "
#               f"{growth_rate:>12.3f}  "
#               f"{stage2_days:>12d}{marker}")

#         records.append({
#             "days_ahead":  k,
#             "date":        candidate_date,
#             "pred_total":  round(pred_total, 1),
#             "growth_rate": round(growth_rate, 4),
#             "stage2_days": stage2_days,
#         })
#         yield_maps[k]     = pred
#         inference_dfs[k]  = inf_df

#     summary = pd.DataFrame(records)

#     # Stage 2 decision: mode of stage2_days recommendations
#     from collections import Counter
#     stage2_votes  = Counter(summary["stage2_days"].tolist())
#     optimal_days  = stage2_votes.most_common(1)[0][0]
#     optimal_date  = last_harvest_date + timedelta(days=optimal_days)
#     optimal_yield = summary[summary["days_ahead"]==optimal_days]["pred_total"].values[0]

#     print(f"\n{'='*60}")
#     print(f"  STAGE 2 RECOMMENDATION : {site}")
#     print(f"  Last harvest  : {last_harvest_date.date()}")
#     print(f"  Optimal date  : {optimal_date.date()}")
#     print(f"  Days to wait  : {optimal_days} days")
#     print(f"  Growth rate   : {summary[summary['days_ahead']==optimal_days]['growth_rate'].values[0]:.3f}")
#     print(f"  Expected yield: {optimal_yield:,.0f} kg")
#     print(f"{'='*60}\n")

#     return {
#         "optimal_days":       optimal_days,
#         "optimal_date":       optimal_date,
#         "pred_yield":         yield_maps.get(optimal_days, np.array([])),
#         "summary_table":      summary,
#         "yield_maps":         yield_maps,
#         "inference_dfs":      inference_dfs,
#         "thresholds":         thresholds,
#         "last_harvest_date":  last_harvest_date,
#         "last_actual_total":  last_actual_total,
#         "site":               site,
#     }


# # ── 6. Decision Quality ───────────────────────────────────────────────────────

# def decision_quality(advice: dict,
#                       df_raw: pd.DataFrame,
#                       splits: dict,
#                       tolerance_days: int = 2) -> pd.DataFrame:
#     """
#     Compare model recommendation vs farmer's actual decision.

#     For each test harvest:
#       model_yield  = actual yield on the day closest to model recommendation
#       farmer_yield = actual yield on the day the farmer actually harvested
#       DQ           = model_yield - farmer_yield

#     Positive DQ → model recommendation is better than farmer's decision.
#     Negative DQ → farmer's decision is better.

#     Parameters
#     ----------
#     advice         : output of recommend_harvest()
#     df_raw         : raw DataFrame (has actual yields for all dates)
#     splits         : output of fe.split_data()
#     tolerance_days : max days difference when matching model date to actual date

#     Returns
#     -------
#     DataFrame with DQ per test harvest date
#     """
#     test_df    = splits["test"]
#     site       = advice["site"]
#     last_date  = advice["last_harvest_date"]
#     opt_days   = advice["optimal_days"]
#     model_date = advice["optimal_date"]

#     # All actual harvest dates
#     actual_dates = sorted(df_raw["harvest_date"].unique())

#     # Field-level actual yields per date
#     actual_yields = df_raw.groupby("harvest_date")["weight_kg"].sum()

#     # Test harvest dates (farmer's actual decisions after last_date)
#     test_dates = sorted(
#         test_df["harvest_date"].unique()
#     )
#     test_dates = [d for d in test_dates if d > last_date]

#     if not test_dates:
#         print("No test dates after last_harvest_date.")
#         return pd.DataFrame()

#     records = []
#     for farmer_date in test_dates:
#         farmer_days  = int((farmer_date - last_date).days)
#         farmer_yield = float(actual_yields.get(farmer_date, np.nan))

#         # Find closest actual date to model recommendation
#         closest = min(actual_dates,
#                       key=lambda d: abs((d - model_date).days))
#         gap = abs((closest - model_date).days)

#         if gap > tolerance_days:
#             model_yield = np.nan
#             note = f"No data within {tolerance_days}d of recommendation"
#         else:
#             model_yield = float(actual_yields.get(closest, np.nan))
#             note = f"Matched to {closest.date()}"

#         dq = model_yield - farmer_yield if not np.isnan(model_yield) else np.nan

#         records.append({
#             "test_harvest_date":  farmer_date,
#             "farmer_days":        farmer_days,
#             "farmer_yield_kg":    round(farmer_yield, 1),
#             "model_rec_date":     model_date,
#             "model_days":         opt_days,
#             "model_matched_date": closest,
#             "model_yield_kg":     round(model_yield, 1) if not np.isnan(model_yield) else np.nan,
#             "DQ_kg":              round(dq, 1) if not np.isnan(dq) else np.nan,
#             "DQ_pct":             round(dq/farmer_yield*100, 1) if farmer_yield > 0 and not np.isnan(dq) else np.nan,
#             "note":               note,
#         })

#     dq_df = pd.DataFrame(records)

#     print(f"\n{'='*65}")
#     print(f"  DECISION QUALITY — {site}")
#     print(f"  Model recommendation: {opt_days} days → {model_date.date()}")
#     print(f"{'='*65}")
#     print(f"  {'Test date':>12} {'Farmer':>6} {'Farmer yield':>13} "
#           f"{'Model':>6} {'Model yield':>12} {'DQ (kg)':>9} {'DQ (%)':>7}")
#     print(f"  {'-'*62}")
#     for _, row in dq_df.iterrows():
#         dq_str  = f"{row['DQ_kg']:+,.0f}" if not np.isnan(row['DQ_kg']) else "N/A"
#         pct_str = f"{row['DQ_pct']:+.1f}%" if not np.isnan(row['DQ_pct']) else "N/A"
#         print(f"  {str(row['test_harvest_date'].date()):>12} "
#               f"{int(row['farmer_days']):>6}d "
#               f"{row['farmer_yield_kg']:>13,.0f} "
#               f"{int(row['model_days']):>6}d "
#               f"{row['model_yield_kg']:>12,.0f} "
#               f"{dq_str:>9} {pct_str:>7}")
#     valid = dq_df.dropna(subset=["DQ_kg"])
#     if len(valid) > 0:
#         print(f"  {'-'*62}")
#         print(f"  Mean DQ : {valid['DQ_kg'].mean():+,.0f} kg  "
#               f"({'model better' if valid['DQ_kg'].mean()>0 else 'farmer better'})")
#         print(f"  Total DQ: {valid['DQ_kg'].sum():+,.0f} kg over {len(valid)} harvests")
#     print(f"{'='*65}\n")

#     return dq_df


# # ── 7. Visualisation ──────────────────────────────────────────────────────────

# def plot_advice(advice: dict, figsize=(18, 10)):
#     """
#     Three-panel figure:
#       Top-left  : bar chart of predicted yields for all candidate days
#       Top-right : predicted yield map for optimal day
#       Bottom    : days map + distribution
#     """
#     site       = advice["site"]
#     summary    = advice["summary_table"]
#     opt_days   = advice["optimal_days"]
#     last_date  = advice["last_harvest_date"]
#     yield_maps = advice["yield_maps"]
#     inf_dfs    = advice["inference_dfs"]
#     pred_y     = advice["pred_yield"]
#     inf_opt    = advice["inference_dfs"].get(opt_days)

#     fig = plt.figure(figsize=figsize)
#     gs  = gridspec.GridSpec(2, 2, height_ratios=[1.5, 1.5],
#                              hspace=0.45, wspace=0.35)

#     # ── Bar chart ─────────────────────────────────────────────────────────────
#     ax_bar = fig.add_subplot(gs[0, 0])
#     colours = ["#2d6a3f" if d == opt_days else "#CBD5E1"
#                for d in summary["days_ahead"]]
#     bars = ax_bar.bar(
#         [f"+{d}d" for d in summary["days_ahead"]],
#         summary["pred_total"],
#         color=colours, edgecolor="white", linewidth=0.5
#     )
#     for bar, (_, row) in zip(bars, summary.iterrows()):
#         ax_bar.text(bar.get_x() + bar.get_width()/2,
#                     bar.get_height() + summary["pred_total"].max()*0.01,
#                     f"{row['pred_total']:,.0f}\n(gr={row['growth_rate']:.2f})",
#                     ha="center", va="bottom", fontsize=8,
#                     color="#2d6a3f" if row["days_ahead"]==opt_days else "#555")
#     ax_bar.set_xlabel("Days from last harvest")
#     ax_bar.set_ylabel("Predicted total yield (kg)")
#     ax_bar.set_title(f"Candidate yields\n(last: {last_date.date()})",
#                      fontsize=10, fontweight="bold")
#     ax_bar.grid(axis="y", alpha=0.3)

#     # ── Optimal yield map ─────────────────────────────────────────────────────
#     ax_map = fig.add_subplot(gs[0, 1])
#     if inf_opt is not None and pred_y is not None and len(pred_y) > 0:
#         x_vals = sorted(inf_opt["field_x"].unique())
#         y_vals = sorted(inf_opt["field_y"].unique())
#         x2i = {v: i for i, v in enumerate(x_vals)}
#         y2i = {v: i for i, v in enumerate(y_vals)}
#         grid = np.zeros((len(y_vals), len(x_vals)))
#         for (_, row), p in zip(inf_opt.iterrows(), pred_y):
#             grid[y2i[row["field_y"]], x2i[row["field_x"]]] = p
#         vmax = float(np.nanquantile(pred_y[pred_y>0], 0.99)) if (pred_y>0).any() else 1.0
#         im = ax_map.imshow(grid, cmap="YlOrRd", aspect="auto", vmin=0, vmax=vmax)
#         plt.colorbar(im, ax=ax_map, label="kg", shrink=0.85)
#         ax_map.set_title(
#             f"Predicted Yield Map\n{advice['optimal_date'].date()} (+{opt_days}d)",
#             fontsize=10, fontweight="bold")
#         ax_map.set_xlabel("field_x"); ax_map.set_ylabel("field_y")
#         ax_map.text(0.02, 0.97, f"Total: {pred_y.sum():,.0f} kg",
#                     transform=ax_map.transAxes, fontsize=9, va="top", color="white",
#                     bbox=dict(boxstyle="round,pad=0.3", fc="#333", alpha=0.65))

#     # ── Growth rate line ──────────────────────────────────────────────────────
#     ax_gr = fig.add_subplot(gs[1, :])
#     ax_gr.plot(summary["days_ahead"], summary["growth_rate"],
#                "o-", color="#5B8DB8", linewidth=2, markersize=8)
#     ax_gr.axhline(advice["thresholds"]["t_high"], color="#E07B39",
#                   linestyle="--", linewidth=1.5,
#                   label=f"t_high={advice['thresholds']['t_high']:.3f} (wait longer)")
#     ax_gr.axhline(advice["thresholds"]["t_low"],  color="#d44",
#                   linestyle="--", linewidth=1.5,
#                   label=f"t_low={advice['thresholds']['t_low']:.3f} (harvest soon)")
#     ax_gr.axvline(opt_days, color="#2d6a3f", linestyle="-", linewidth=2, alpha=0.4)
#     ax_gr.fill_between(summary["days_ahead"], advice["thresholds"]["t_low"],
#                         advice["thresholds"]["t_high"], alpha=0.08, color="#5B8DB8",
#                         label="Stable zone")
#     for _, row in summary.iterrows():
#         ax_gr.annotate(f"{int(row['stage2_days'])}d",
#                        (row["days_ahead"], row["growth_rate"]),
#                        textcoords="offset points", xytext=(0, 10), ha="center",
#                        fontsize=9, color="#2d6a3f" if row["days_ahead"]==opt_days else "#555")
#     ax_gr.set_xlabel("Candidate days ahead", fontsize=10)
#     ax_gr.set_ylabel("Growth rate (pred / last actual)", fontsize=10)
#     ax_gr.set_title("Stage 2: Growth rate rule — how the recommendation was made",
#                     fontsize=11, fontweight="bold")
#     ax_gr.legend(fontsize=9); ax_gr.grid(alpha=0.3)
#     ax_gr.set_xticks(summary["days_ahead"])

#     fig.suptitle(
#         f"{site} — Harvest Advisor  |  Recommendation: +{opt_days} days "
#         f"({advice['optimal_date'].date()})",
#         fontsize=13, fontweight="bold")
#     plt.show()
#     return fig


# def plot_decision_quality(dq_df: pd.DataFrame, site: str, figsize=(12, 5)):
#     """Bar chart of Decision Quality per test harvest."""
#     if dq_df.empty or dq_df["DQ_kg"].isna().all():
#         print("No DQ data to plot.")
#         return

#     valid = dq_df.dropna(subset=["DQ_kg"])
#     colours = ["#2d6a3f" if v >= 0 else "#c0392b" for v in valid["DQ_kg"]]

#     fig, ax = plt.subplots(figsize=figsize)
#     bars = ax.bar(
#         [str(d.date()) for d in valid["test_harvest_date"]],
#         valid["DQ_kg"],
#         color=colours, edgecolor="white", linewidth=0.5
#     )
#     ax.axhline(0, color="black", linewidth=1)
#     ax.axhline(valid["DQ_kg"].mean(), color="#E07B39", linewidth=2,
#                linestyle="--", label=f"Mean DQ = {valid['DQ_kg'].mean():+,.0f} kg")

#     for bar, (_, row) in zip(bars, valid.iterrows()):
#         ax.text(bar.get_x() + bar.get_width()/2,
#                 bar.get_height() + (valid["DQ_kg"].abs().max()*0.02 * np.sign(bar.get_height())),
#                 f"{row['DQ_kg']:+,.0f} kg\n({row['DQ_pct']:+.1f}%)",
#                 ha="center", va="bottom" if row["DQ_kg"]>=0 else "top",
#                 fontsize=9)

#     ax.set_xlabel("Test harvest date")
#     ax.set_ylabel("Decision Quality (kg)\nPositive = model better")
#     ax.set_title(f"{site} — Decision Quality\n"
#                  f"Model recommendation vs Farmer's actual decision",
#                  fontsize=12, fontweight="bold")
#     ax.legend(fontsize=10); ax.grid(axis="y", alpha=0.3)
#     plt.tight_layout(); plt.show()
#     return fig


# def print_recommendation(advice: dict):
#     """Print a farmer-friendly summary."""
#     print("\n" + "="*50)
#     print(f"  HARVEST RECOMMENDATION — {advice['site']}")
#     print("="*50)
#     print(f"  Last harvest  : {advice['last_harvest_date'].date()}")
#     print(f"  Last yield    : {advice['last_actual_total']:,.0f} kg")
#     print(f"  Optimal date  : {advice['optimal_date'].date()}")
#     print(f"  Days to wait  : {advice['optimal_days']} days")
#     if len(advice["pred_yield"]) > 0:
#         print(f"  Expected yield: {advice['pred_yield'].sum():,.0f} kg")
#     print(f"\n  Candidate summary:")
#     print(f"  {'Days':>5}  {'Date':>12}  {'Yield(kg)':>10}  {'Growth':>8}")
#     print(f"  {'-'*42}")
#     for _, row in advice["summary_table"].iterrows():
#         mk = " ← OPTIMAL" if row["days_ahead"]==advice["optimal_days"] else ""
#         print(f"  +{int(row['days_ahead']):<4}  "
#               f"{str(row['date'].date()):>12}  "
#               f"{row['pred_total']:>10,.0f}  "
#               f"{row['growth_rate']:>8.3f}{mk}")
#     print("="*50 + "\n")



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
            df_raw, last_harvest_date, candidate_date, weather, lag_depth, coarsen_n)
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
    fig.suptitle(f"{site} — Rolling Forecast: VersionA vs VersionB vs Ground Truth",
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
                linewidth=2, markersize=6, label="VersionA (Lite)", zorder=3)
        ax.plot(b["harvest_date"], b["pred_kg"], "^:", color="#E07B39",
                linewidth=2, markersize=6, label="VersionB (Strict)", zorder=3)
 
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