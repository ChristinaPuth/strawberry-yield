# """
# models.py
# ---------
# Ablation study and model comparison for strawberry yield prediction.

# Ablation configurations (8):
#     A0: yield_lag1 only                         (minimum baseline)
#     A1: temporal yield features only            (lag1/2/3 + rolling + trend)
#     A2: full temporal block                     (A1 + cumulative + days_since + doy)
#     A3: spatial only                            (field_x/y + neighbors)
#     A4: spatio-temporal, no weather             (A2 + A3)
#     A5: A4 + 3 core weather vars               (temp_mean + precip + et0)
#     A6: A4 + all 8 weather vars                (full weather block)
#     A7: all 21 features                         (complete)

# Models compared (on best feature set):
#     Linear Regression, Random Forest, LightGBM, XGBoost, LightGBM + log(y+1)

# Usage in Colab:
#     import importlib, models as m
#     importlib.reload(m)

#     results = m.run_ablation(splits_sm, "SantaMaria")
#     best_feats = m.best_feature_set(results)
#     model_results = m.run_model_comparison(splits_sm, "SantaMaria", best_feats)
# """

# import numpy as np
# import pandas as pd
# from sklearn.metrics import mean_squared_error, r2_score
# from sklearn.linear_model import LinearRegression
# from sklearn.ensemble import RandomForestRegressor
# import lightgbm as lgb
# import xgboost as xgb
# import warnings
# warnings.filterwarnings("ignore")


# # ── Feature groups ────────────────────────────────────────────────────────────

# FEATURE_GROUPS = {
#     "temporal_yield": [
#         "yield_lag1", "yield_lag2", "yield_lag3",
#         "rolling_mean_3", "yield_trend",
#     ],
#     "temporal_full": [
#         "yield_lag1", "yield_lag2", "yield_lag3",
#         "rolling_mean_3", "yield_trend",
#         "season_cumulative", "days_since_last", "day_of_year",
#     ],
#     "spatial": [
#         "field_x", "field_y",
#         "neighbor_mean_3x3", "neighbor_mean_5x5",
#     ],
#     "weather_core": [
#         "temp_mean_7d", "precip_7d", "et0_7d",
#     ],
#     "weather_full": [
#         "temp_mean_7d", "temp_max_7d", "temp_min_7d",
#         "precip_7d", "et0_7d", "humidity_mean_7d",
#         "soil_moisture_0_7", "soil_moisture_7_28", "daylight_7d",
#     ],
# }

# # ── 8 ablation configurations ─────────────────────────────────────────────────

# ABLATION_CONFIGS = {
#     "A0": ["yield_lag1"],
#     "A1": FEATURE_GROUPS["temporal_yield"],
#     "A2": FEATURE_GROUPS["temporal_full"],
#     "A3": FEATURE_GROUPS["spatial"],
#     "A4": FEATURE_GROUPS["temporal_full"] + FEATURE_GROUPS["spatial"],
#     "A5": FEATURE_GROUPS["temporal_full"] + FEATURE_GROUPS["spatial"]
#           + FEATURE_GROUPS["weather_core"],
#     "A6": FEATURE_GROUPS["temporal_full"] + FEATURE_GROUPS["spatial"]
#           + FEATURE_GROUPS["weather_full"],
#     "A7": FEATURE_GROUPS["temporal_full"] + FEATURE_GROUPS["spatial"]
#           + FEATURE_GROUPS["weather_full"],   # same as A6 but explicit "all"
# }

# ABLATION_DESCRIPTIONS = {
#     "A0": "yield_lag1 only (minimum baseline)",
#     "A1": "Temporal yield only (5 features)",
#     "A2": "Full temporal block (8 features)",
#     "A3": "Spatial only — no yield history",
#     "A4": "Spatio-temporal, no weather (12 features)",
#     "A5": "A4 + 3 core weather vars (15 features)",
#     "A6": "A4 + all 8 weather vars (20 features)",
#     "A7": "All 21 features (complete set)",
# }

# TARGET = "weight_kg"


# # ── Helpers ───────────────────────────────────────────────────────────────────

# def _xy(df: pd.DataFrame, features: list):
#     """Return X (feature matrix) and y (target vector) as numpy arrays."""
#     available = [f for f in features if f in df.columns]
#     X = df[available].values.astype(np.float32)
#     y = df[TARGET].values.astype(np.float32)
#     return X, y, available


# def _metrics(y_true, y_pred):
#     """Compute RMSE and R² from arrays."""
#     rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
#     r2   = float(r2_score(y_true, y_pred))
#     return rmse, r2


# def _default_lgbm():
#     return lgb.LGBMRegressor(
#         n_estimators=200,
#         num_leaves=63,
#         learning_rate=0.05,
#         min_child_samples=20,
#         random_state=42,
#         verbose=-1,
#         n_jobs=-1,
#     )


# # ── 1. Ablation study ─────────────────────────────────────────────────────────

# def run_ablation(splits: dict, site: str) -> pd.DataFrame:
#     """
#     Run 8 ablation configurations with default-parameter LightGBM.
#     Trains on splits['train'], evaluates on splits['val'].

#     Parameters
#     ----------
#     splits : dict with keys 'train', 'val', 'test'
#              (output of feature_engineering.split_data)
#     site   : site name for display

#     Returns
#     -------
#     DataFrame with columns:
#         config, description, n_features, train_rmse, train_r2,
#         val_rmse, val_r2
#     Sorted by val_r2 descending.
#     """
#     train_df = splits["train"]
#     val_df   = splits["val"]

#     print(f"\nAblation study — {site}")
#     print(f"Train: {len(train_df):,} rows  |  Val: {len(val_df):,} rows")
#     print("=" * 72)
#     print(f"  {'Config':<4} {'#Feat':>5}  "
#           f"{'Train RMSE':>11} {'Train R²':>9}  "
#           f"{'Val RMSE':>10} {'Val R²':>8}  Description")
#     print(f"  {'-'*68}")

#     records = []
#     for cfg_id, features in ABLATION_CONFIGS.items():
#         X_tr, y_tr, used = _xy(train_df, features)
#         X_va, y_va, _    = _xy(val_df,   features)

#         model = _default_lgbm()
#         model.fit(X_tr, y_tr)

#         tr_rmse, tr_r2 = _metrics(y_tr, model.predict(X_tr))
#         va_rmse, va_r2 = _metrics(y_va, model.predict(X_va))

#         desc = ABLATION_DESCRIPTIONS[cfg_id]
#         print(f"  {cfg_id:<4} {len(used):>5}  "
#               f"{tr_rmse:>11.4f} {tr_r2:>9.4f}  "
#               f"{va_rmse:>10.4f} {va_r2:>8.4f}  {desc}")

#         records.append({
#             "config":      cfg_id,
#             "description": desc,
#             "n_features":  len(used),
#             "features":    used,
#             "train_rmse":  round(tr_rmse, 4),
#             "train_r2":    round(tr_r2,   4),
#             "val_rmse":    round(va_rmse, 4),
#             "val_r2":      round(va_r2,   4),
#         })

#     print("=" * 72)
#     results = pd.DataFrame(records).sort_values("val_r2", ascending=False)
#     best = results.iloc[0]
#     print(f"\nBest config: {best['config']}  "
#           f"(val R²={best['val_r2']:.4f}, RMSE={best['val_rmse']:.4f})")
#     print(f"Description: {best['description']}\n")
#     return results


# def best_feature_set(ablation_results: pd.DataFrame) -> list:
#     """Return the feature list of the best-performing ablation config."""
#     return ablation_results.iloc[0]["features"]


# # ── 2. Model comparison ───────────────────────────────────────────────────────

# def run_model_comparison(splits: dict,
#                           site: str,
#                           features: list,
#                           use_log_target: bool = False) -> pd.DataFrame:
#     """
#     Train and evaluate 5 models on the given feature set.

#     Models: LinearRegression, RandomForest, LightGBM, XGBoost,
#             LightGBM + log(y+1) transform

#     Parameters
#     ----------
#     splits   : dict with 'train', 'val', 'test'
#     site     : site name for display
#     features : list of feature column names (use best_feature_set())

#     Returns
#     -------
#     DataFrame with val metrics for each model, sorted by val_r2.
#     """
#     train_df = splits["train"]
#     val_df   = splits["val"]

#     X_tr, y_tr, used = _xy(train_df, features)
#     X_va, y_va, _    = _xy(val_df,   features)

#     print(f"\nModel comparison — {site}")
#     print(f"Features: {len(used)}  |  "
#           f"Train: {len(train_df):,}  |  Val: {len(val_df):,}")
#     print("=" * 65)
#     print(f"  {'Model':<28} {'Val RMSE':>10} {'Val R²':>8} {'Train R²':>9}")
#     print(f"  {'-'*60}")

#     # define models
#     model_defs = [
#         ("Linear Regression", LinearRegression(), False),
#         ("Random Forest",
#          RandomForestRegressor(n_estimators=100, max_depth=10,
#                                min_samples_leaf=20, random_state=42,
#                                n_jobs=-1),
#          False),
#         ("LightGBM",
#          lgb.LGBMRegressor(n_estimators=300, num_leaves=63,
#                             learning_rate=0.05, min_child_samples=20,
#                             random_state=42, verbose=-1, n_jobs=-1),
#          False),
#         ("XGBoost",
#          xgb.XGBRegressor(n_estimators=300, max_depth=6, eta=0.05,
#                            subsample=0.8, colsample_bytree=0.8,
#                            random_state=42, verbosity=0, n_jobs=-1),
#          False),
#         ("LightGBM + log(y+1)",
#          lgb.LGBMRegressor(n_estimators=300, num_leaves=63,
#                             learning_rate=0.05, min_child_samples=20,
#                             random_state=42, verbose=-1, n_jobs=-1),
#          True),   # True = use log transform
#     ]

#     records = []
#     for name, model, log_transform in model_defs:
#         if log_transform:
#             y_tr_fit = np.log1p(y_tr)
#         else:
#             y_tr_fit = y_tr

#         model.fit(X_tr, y_tr_fit)

#         pred_tr = model.predict(X_tr)
#         pred_va = model.predict(X_va)

#         if log_transform:
#             pred_tr = np.expm1(pred_tr)
#             pred_va = np.expm1(pred_va)

#         # clip negatives
#         pred_tr = np.clip(pred_tr, 0, None)
#         pred_va = np.clip(pred_va, 0, None)

#         tr_rmse, tr_r2 = _metrics(y_tr, pred_tr)
#         va_rmse, va_r2 = _metrics(y_va, pred_va)

#         print(f"  {name:<28} {va_rmse:>10.4f} {va_r2:>8.4f} {tr_r2:>9.4f}")

#         records.append({
#             "model":      name,
#             "val_rmse":   round(va_rmse, 4),
#             "val_r2":     round(va_r2,   4),
#             "train_r2":   round(tr_r2,   4),
#             "train_rmse": round(tr_rmse, 4),
#             "log_target": log_transform,
#             "n_features": len(used),
#             "features":   used,
#             "_model_obj": model,   # keep for test evaluation
#         })

#     print("=" * 65)
#     results = pd.DataFrame(records).sort_values("val_r2", ascending=False)
#     best = results.iloc[0]
#     print(f"\nBest model: {best['model']}  "
#           f"(val R²={best['val_r2']:.4f}, RMSE={best['val_rmse']:.4f})\n")
#     return results


# # ── 3. Test set evaluation (run ONCE at the very end) ─────────────────────────

# def evaluate_on_test(model_results: pd.DataFrame,
#                      splits: dict,
#                      site: str) -> dict:
#     """
#     Evaluate the best model on the held-out test set.
#     Call this ONLY once, after all design decisions are finalised.

#     Returns dict with test_rmse, test_r2, y_true, y_pred arrays.
#     """
#     best_row = model_results.iloc[0]
#     model    = best_row["_model_obj"]
#     features = best_row["features"]
#     log_t    = best_row["log_target"]

#     test_df        = splits["test"]
#     X_te, y_te, _  = _xy(test_df, features)

#     pred = model.predict(X_te)
#     if log_t:
#         pred = np.expm1(pred)
#     pred = np.clip(pred, 0, None)

#     rmse, r2 = _metrics(y_te, pred)

#     print(f"\n{'='*50}")
#     print(f"  TEST SET RESULTS — {site}")
#     print(f"  Model   : {best_row['model']}")
#     print(f"  Features: {len(features)}")
#     print(f"  RMSE    : {rmse:.4f} kg")
#     print(f"  R²      : {r2:.4f}")
#     print(f"{'='*50}\n")

#     return {
#         "model":     best_row["model"],
#         "test_rmse": round(rmse, 4),
#         "test_r2":   round(r2,   4),
#         "y_true":    y_te,
#         "y_pred":    pred,
#         "test_df":   test_df,
#     }


# # ── 4. Feature importance ─────────────────────────────────────────────────────

# def plot_feature_importance(model_results: pd.DataFrame,
#                              top_n: int = 21):
#     """
#     Plot feature importance from the best tree-based model.
#     Skips LinearRegression (no native importance).
#     """
#     import matplotlib.pyplot as plt

#     # find best tree model
#     tree_models = model_results[
#         model_results["model"].str.contains("LightGBM|XGBoost|Random Forest")
#     ]
#     if tree_models.empty:
#         print("No tree model found.")
#         return

#     best_row = tree_models.iloc[0]
#     model    = best_row["_model_obj"]
#     features = best_row["features"]

#     # get importances
#     if hasattr(model, "feature_importances_"):
#         importances = model.feature_importances_
#     else:
#         print("Model has no feature_importances_.")
#         return

#     imp_df = pd.DataFrame({
#         "feature":    features[:len(importances)],
#         "importance": importances,
#     }).sort_values("importance", ascending=True).tail(top_n)

#     # colour by group
#     def _colour(feat):
#         if any(feat.startswith(p) for p in ["yield_lag","rolling","yield_trend",
#                                               "season","days_since","day_of"]):
#             return "#E07B39"   # temporal
#         if any(feat.startswith(p) for p in ["field_","neighbor"]):
#             return "#5B8DB8"   # spatial
#         return "#6BBF8C"       # weather

#     colours = [_colour(f) for f in imp_df["feature"]]

#     fig, ax = plt.subplots(figsize=(9, 0.4 * len(imp_df) + 1.5))
#     bars = ax.barh(imp_df["feature"], imp_df["importance"],
#                    color=colours, edgecolor="white", linewidth=0.4)
#     ax.set_xlabel("Feature importance")
#     ax.set_title(f"Feature importance — {best_row['model']}", fontweight="bold")

#     # legend
#     from matplotlib.patches import Patch
#     legend = [
#         Patch(color="#E07B39", label="Temporal"),
#         Patch(color="#5B8DB8", label="Spatial"),
#         Patch(color="#6BBF8C", label="Weather"),
#     ]
#     ax.legend(handles=legend, loc="lower right", fontsize=9)
#     plt.tight_layout()
#     plt.show()
#     return fig


# # ── 5. Prediction yield map ───────────────────────────────────────────────────

# def predict_yield_map(model_results: pd.DataFrame,
#                       df_feat: pd.DataFrame,
#                       harvest_date,
#                       site: str):
#     """
#     Generate a predicted vs actual yield map for a given harvest date.
#     Uses the best model from model_results.
#     """
#     import matplotlib.pyplot as plt

#     best_row = model_results.iloc[0]
#     model    = best_row["_model_obj"]
#     features = best_row["features"]
#     log_t    = best_row["log_target"]

#     d = df_feat[df_feat["harvest_date"] == pd.Timestamp(harvest_date)].copy()
#     if d.empty:
#         print(f"No data for {harvest_date}")
#         return

#     X, y_true, _ = _xy(d, features)
#     y_pred = model.predict(X)
#     if log_t:
#         y_pred = np.expm1(y_pred)
#     y_pred = np.clip(y_pred, 0, None)

#     d = d.copy()
#     d["y_pred"] = y_pred

#     rmse, r2 = _metrics(y_true, y_pred)

#     # build grids
#     x_vals = sorted(d["field_x"].unique())
#     y_vals = sorted(d["field_y"].unique())
#     x2i = {v: i for i, v in enumerate(x_vals)}
#     y2i = {v: i for i, v in enumerate(y_vals)}

#     grid_true = np.zeros((len(y_vals), len(x_vals)))
#     grid_pred = np.zeros((len(y_vals), len(x_vals)))
#     grid_err  = np.full((len(y_vals), len(x_vals)), np.nan)

#     for _, row in d.iterrows():
#         xi = x2i[row["field_x"]]
#         yi = y2i[row["field_y"]]
#         grid_true[yi, xi] = row["weight_kg"]
#         grid_pred[yi, xi] = row["y_pred"]
#         grid_err [yi, xi] = row["y_pred"] - row["weight_kg"]

#     vmax = float(np.nanquantile(
#         np.concatenate([grid_true.ravel(), grid_pred.ravel()]), 0.99))

#     fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
#     fig.suptitle(
#         f"{site}  —  {str(harvest_date)[:10]}  |  "
#         f"RMSE={rmse:.3f} kg   R²={r2:.3f}",
#         fontsize=13, fontweight="bold"
#     )

#     for ax, grid, title, cmap in [
#         (axes[0], grid_true, "Actual yield",    "YlOrRd"),
#         (axes[1], grid_pred, "Predicted yield", "YlOrRd"),
#         (axes[2], grid_err,  "Error (pred-actual)", "RdBu_r"),
#     ]:
#         if title.startswith("Error"):
#             vm = float(np.nanquantile(np.abs(grid_err[~np.isnan(grid_err)]), 0.95))
#             im = ax.imshow(grid, cmap=cmap, aspect="auto",
#                            vmin=-vm, vmax=vm)
#         else:
#             im = ax.imshow(grid, cmap=cmap, aspect="auto",
#                            vmin=0, vmax=vmax)
#         plt.colorbar(im, ax=ax, label="kg", shrink=0.85)
#         ax.set_title(title, fontsize=11)
#         ax.set_xlabel("field_x index")
#         ax.set_ylabel("field_y index")

#     plt.tight_layout()
#     plt.show()
#     return fig


#     # error map/ ground truth map using %
#     #consider 
#     # day1 10kg day4   7
#     # day 7  8



"""
models.py  v3
-------------
Stage 1 ablation study and model comparison for weight_kg prediction.

Ablation configurations (9 + 2 baselines):

  Baselines (no model training):
    B0: Seasonal Mean     — per-cell expanding historical mean
    B1: Trend Extrap      — 2*lag1 - lag2

  Feature ablations (LightGBM):
    A0: yield_lag1 only              (minimum ML baseline)
    A1: Temporal yield (5)           (lag1/2/3 + rolling + trend)
    A2: Full temporal (7)            (A1 + cumulative + day_of_year)
    A3: Spatial only (4)             (field_x/y + neighbors)
    A4: Spatio-temporal (11)         (A2 + A3)
    A5: A4 + core weather (3)        (+temp_mean + precip + et0)
    A6: A4 + all weather (9)         (20 features, complete)
    A7: Weather only (9)             (weather alone, no yield history)
    A8: Spatial + weather (13)       (no temporal yield history)

Key question A7/A8 answer:
  "What if we had no harvest history?" (first season, new field)

Models compared (on best feature set):
  Linear Regression, Random Forest, LightGBM,
  XGBoost, LightGBM + log(y+1)
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

from feature_engineering import (
    TEMPORAL_FEATS, SPATIAL_FEATS, WEATHER_FEATS, ALL_FEATS
)

# ── Ablation feature sets ─────────────────────────────────────────────────────

ABLATION_CONFIGS = {
    # ML models
    "A0": ["yield_lag1"],
    "A1": TEMPORAL_FEATS[:5],                          # lag1/2/3 + rolling + trend
    "A2": TEMPORAL_FEATS,                              # full temporal (7)
    "A3": SPATIAL_FEATS,                               # spatial only (4)
    "A4": TEMPORAL_FEATS + SPATIAL_FEATS,              # spatio-temporal (11)
    "A5": TEMPORAL_FEATS + SPATIAL_FEATS               # A4 + core weather (14)
          + ["temp_mean_7d","precip_7d","et0_7d"],
    "A6": ALL_FEATS,                                   # all 20 features
    "A7": WEATHER_FEATS,                               # weather only (9)
    "A8": SPATIAL_FEATS + WEATHER_FEATS,               # spatial + weather (13)
}

ABLATION_DESCRIPTIONS = {
    "A0": "yield_lag1 only (minimum ML baseline)",
    "A1": "Temporal yield only — 5 features",
    "A2": "Full temporal block — 7 features",
    "A3": "Spatial only — 4 features (no yield history)",
    "A4": "Spatio-temporal — 11 features",
    "A5": "A4 + core weather — 14 features",
    "A6": "All 20 features (complete set)",
    "A7": "Weather only — 9 features (no yield history)",
    "A8": "Spatial + weather — 13 features (no yield history)",
}

TARGET = "weight_kg"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _xy(df, features, target=TARGET):
    avail = [f for f in features if f in df.columns]
    X = df[avail].values.astype(np.float32)
    y = df[target].values.astype(np.float32)
    return X, y, avail


def _metrics(y_true, y_pred):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))
    return rmse, mae, r2


def _default_lgbm():
    return lgb.LGBMRegressor(
        n_estimators=200, num_leaves=63, learning_rate=0.05,
        min_child_samples=20, random_state=42, verbose=-1, n_jobs=-1,
    )


# ── 1. Ablation study ─────────────────────────────────────────────────────────

def run_ablation(splits: dict, site: str) -> pd.DataFrame:
    """
    Run ablation study with two baselines + 9 feature configs.

    Baselines (B0, B1) use direct column values, no model training.
    Configs A0-A8 train default LightGBM.

    Returns DataFrame sorted by val_r2 descending.
    """
    train_df = splits["train"]
    val_df   = splits["val"]
    y_val    = val_df[TARGET].values.astype(np.float32)

    print(f"\nAblation study — {site}")
    print(f"Train: {len(train_df):,} rows  |  Val: {len(val_df):,} rows")
    print(f"Target: {TARGET}")
    print("=" * 80)
    print(f"  {'Cfg':<4} {'#F':>3}  {'Val R²':>8} {'Val RMSE':>10} "
          f"{'Val MAE':>9}  Description")
    print(f"  {'-'*75}")

    records = []

    # ── Baselines ─────────────────────────────────────────────────────────────
    baseline_defs = [
        ("B0", "Seasonal Mean (per-cell expanding mean)",  "seasonal_mean"),
        ("B1", "Trend Extrapolation (2*lag1 - lag2)",      "trend_extrap"),
    ]
    for cfg_id, desc, col in baseline_defs:
        if col not in val_df.columns:
            print(f"  {cfg_id:<4} {'—':>3}  {'—':>8} {'—':>10} {'—':>9}  "
                  f"{desc} [MISSING]")
            continue
        y_pred = val_df[col].fillna(0).values.astype(np.float32)
        rmse, mae, r2 = _metrics(y_val, y_pred)
        print(f"  {cfg_id:<4} {'—':>3}  {r2:>8.4f} {rmse:>10.4f} "
              f"{mae:>9.4f}  {desc}")
        records.append({
            "config": cfg_id, "description": desc,
            "n_features": 0, "features": [col],
            "is_baseline": True,
            "val_r2": round(r2,4), "val_rmse": round(rmse,4),
            "val_mae": round(mae,4),
            "train_r2": None, "_model_obj": None,
        })

    print(f"  {'─'*75}")

    # ── ML configs ────────────────────────────────────────────────────────────
    for cfg_id, features in ABLATION_CONFIGS.items():
        X_tr, y_tr, used = _xy(train_df, features)
        X_va, y_va, _    = _xy(val_df,   features)

        model = _default_lgbm()
        model.fit(X_tr, y_tr)

        _, _, tr_r2      = _metrics(y_tr, model.predict(X_tr))
        rmse, mae, va_r2 = _metrics(y_va, model.predict(X_va))

        desc = ABLATION_DESCRIPTIONS[cfg_id]
        print(f"  {cfg_id:<4} {len(used):>3}  {va_r2:>8.4f} {rmse:>10.4f} "
              f"{mae:>9.4f}  {desc}")

        records.append({
            "config": cfg_id, "description": desc,
            "n_features": len(used), "features": used,
            "is_baseline": False,
            "val_r2":   round(va_r2, 4),
            "val_rmse": round(rmse,  4),
            "val_mae":  round(mae,   4),
            "train_r2": round(tr_r2, 4),
            "_model_obj": model,
        })

    print("=" * 80)
    results = pd.DataFrame(records)

    # Best ML config (excluding baselines)
    ml_results = results[~results["is_baseline"]]
    best = ml_results.sort_values("val_r2", ascending=False).iloc[0]

    # Baseline reference
    b0 = results[results["config"]=="B0"]
    b1 = results[results["config"]=="B1"]
    b0_r2 = b0["val_r2"].values[0] if len(b0) else float("nan")
    b1_r2 = b1["val_r2"].values[0] if len(b1) else float("nan")

    print(f"\n  Baselines   : B0 R²={b0_r2:.4f}  B1 R²={b1_r2:.4f}")
    print(f"  Best ML     : {best['config']}  "
          f"(val R²={best['val_r2']:.4f}, RMSE={best['val_rmse']:.4f})")
    if best["val_r2"] > max(b0_r2, b1_r2):
        print(f"  ✅ Best ML config exceeds both baselines")
    else:
        print(f"  ⚠️  Best ML config does NOT exceed baselines")
    print()

    return results.sort_values("val_r2", ascending=False).reset_index(drop=True)


def best_feature_set(ablation_results: pd.DataFrame) -> list:
    """Return feature list of the best ML config (excluding baselines)."""
    ml = ablation_results[~ablation_results["is_baseline"]]
    return ml.sort_values("val_r2", ascending=False).iloc[0]["features"]


# ── 2. Model comparison ───────────────────────────────────────────────────────

def run_model_comparison(splits: dict,
                          site: str,
                          features: list) -> pd.DataFrame:
    """
    Train and evaluate 5 models on the given feature set.
    Returns DataFrame sorted by val_r2.
    """
    train_df = splits["train"]
    val_df   = splits["val"]
    X_tr, y_tr, used = _xy(train_df, features)
    X_va, y_va, _    = _xy(val_df,   features)

    print(f"\nModel comparison — {site}")
    print(f"Features: {len(used)}  |  Train: {len(train_df):,}  "
          f"|  Val: {len(val_df):,}")
    print("=" * 65)
    print(f"  {'Model':<28} {'Val RMSE':>10} {'Val R²':>8} {'Train R²':>9}")
    print(f"  {'-'*60}")

    model_defs = [
        ("Linear Regression",
         LinearRegression(), False),
        ("Random Forest",
         RandomForestRegressor(n_estimators=100, max_depth=10,
                               min_samples_leaf=20, random_state=42, n_jobs=-1),
         False),
        ("LightGBM",
         lgb.LGBMRegressor(n_estimators=300, num_leaves=63, learning_rate=0.05,
                            min_child_samples=20, random_state=42,
                            verbose=-1, n_jobs=-1),
         False),
        ("XGBoost",
         xgb.XGBRegressor(n_estimators=300, max_depth=6, eta=0.05,
                           subsample=0.8, colsample_bytree=0.8,
                           random_state=42, verbosity=0, n_jobs=-1),
         False),
        ("LightGBM + log(y+1)",
         lgb.LGBMRegressor(n_estimators=300, num_leaves=63, learning_rate=0.05,
                            min_child_samples=20, random_state=42,
                            verbose=-1, n_jobs=-1),
         True),
    ]

    records = []
    for name, model, log_t in model_defs:
        y_fit = np.log1p(y_tr) if log_t else y_tr
        model.fit(X_tr, y_fit)
        p_tr = np.clip(np.expm1(model.predict(X_tr)) if log_t
                       else model.predict(X_tr), 0, None)
        p_va = np.clip(np.expm1(model.predict(X_va)) if log_t
                       else model.predict(X_va), 0, None)
        _, _, tr_r2      = _metrics(y_tr, p_tr)
        va_rmse, _, va_r2 = _metrics(y_va, p_va)
        print(f"  {name:<28} {va_rmse:>10.4f} {va_r2:>8.4f} {tr_r2:>9.4f}")
        records.append({
            "model": name, "val_rmse": round(va_rmse,4),
            "val_r2": round(va_r2,4), "train_r2": round(tr_r2,4),
            "log_target": log_t, "features": used, "_model_obj": model,
        })

    print("=" * 65)
    results = pd.DataFrame(records).sort_values("val_r2", ascending=False)
    best = results.iloc[0]
    print(f"\nBest model: {best['model']}  "
          f"(val R²={best['val_r2']:.4f}, RMSE={best['val_rmse']:.4f})\n")
    return results


# ── 3. Test set evaluation ────────────────────────────────────────────────────

def evaluate_on_test(model_results: pd.DataFrame,
                     splits: dict,
                     site: str) -> dict:
    """Evaluate best model on held-out test set. Call ONCE only."""
    best  = model_results.iloc[0]
    model = best["_model_obj"]
    feats = best["features"]
    log_t = best["log_target"]

    test_df       = splits["test"]
    X_te, y_te, _ = _xy(test_df, feats)
    p = np.clip(np.expm1(model.predict(X_te)) if log_t
                else model.predict(X_te), 0, None)
    rmse, mae, r2 = _metrics(y_te, p)

    # Field-level: total predicted vs actual per harvest date
    df_res = test_df[["harvest_date","weight_kg"]].copy()
    df_res["y_pred"] = p
    field_level = df_res.groupby("harvest_date").agg(
        actual_total  = ("weight_kg","sum"),
        pred_total    = ("y_pred","sum"),
        actual_mean   = ("weight_kg","mean"),
        pred_mean     = ("y_pred","mean"),
    ).reset_index()
    field_level["diff_kg"]  = field_level["pred_total"] - field_level["actual_total"]
    field_level["diff_pct"] = (field_level["diff_kg"] /
                                field_level["actual_total"] * 100).round(1)

    print(f"\n{'='*55}")
    print(f"  TEST SET RESULTS — {site}")
    print(f"  Model  : {best['model']}")
    print(f"  RMSE   : {rmse:.4f} kg")
    print(f"  MAE    : {mae:.4f} kg")
    print(f"  R²     : {r2:.4f}")
    print(f"\n  Field-level (total yield per harvest):")
    print(f"  {'Date':>12} {'Actual':>10} {'Predicted':>11} "
          f"{'Diff(kg)':>10} {'Diff(%)':>8}")
    for _, row in field_level.iterrows():
        print(f"  {str(row['harvest_date'].date()):>12} "
              f"{row['actual_total']:>10,.0f} "
              f"{row['pred_total']:>11,.0f} "
              f"{row['diff_kg']:>+10,.0f} "
              f"{row['diff_pct']:>+7.1f}%")
    print(f"{'='*55}\n")

    return {
        "model":        best["model"],
        "test_rmse":    round(rmse, 4),
        "test_mae":     round(mae,  4),
        "test_r2":      round(r2,   4),
        "y_true":       y_te,
        "y_pred":       p,
        "field_level":  field_level,
        "test_df":      test_df.assign(y_pred=p),
        "log_target":   log_t,
        "features":     feats,
        "_model_obj":   model,
    }


# ── 4. Feature importance ─────────────────────────────────────────────────────

def plot_feature_importance(model_results: pd.DataFrame, top_n: int = 20):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    tree_models = model_results[
        model_results["model"].str.contains("LightGBM|XGBoost|Random Forest")
    ]
    if tree_models.empty:
        print("No tree model found."); return

    best_row = tree_models.iloc[0]
    model    = best_row["_model_obj"]
    features = best_row["features"]

    if not hasattr(model, "feature_importances_"):
        print("No feature_importances_."); return

    imp_df = pd.DataFrame({
        "feature":    features[:len(model.feature_importances_)],
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=True).tail(top_n)

    def _colour(f):
        if any(f.startswith(p) for p in
               ["yield_lag","rolling","yield_trend","season","day_of"]):
            return "#E07B39"
        if any(f.startswith(p) for p in ["field_","neighbor"]):
            return "#5B8DB8"
        return "#6BBF8C"

    colours = [_colour(f) for f in imp_df["feature"]]
    fig, ax = plt.subplots(figsize=(9, 0.4*len(imp_df)+1.5))
    ax.barh(imp_df["feature"], imp_df["importance"],
            color=colours, edgecolor="white", linewidth=0.4)
    ax.set_xlabel("Feature importance")
    ax.set_title(f"Feature importance — {best_row['model']} → weight_kg",
                 fontweight="bold")
    ax.legend(handles=[
        Patch(color="#E07B39", label="Temporal"),
        Patch(color="#5B8DB8", label="Spatial"),
        Patch(color="#6BBF8C", label="Weather"),
    ], loc="lower right", fontsize=9)
    plt.tight_layout(); plt.show()
    return fig


# ── 5. Predicted vs actual yield map ─────────────────────────────────────────

def predict_yield_map(model_results: pd.DataFrame,
                       df_feat: pd.DataFrame,
                       harvest_date,
                       site: str):
    import matplotlib.pyplot as plt

    best  = model_results.iloc[0]
    model = best["_model_obj"]
    feats = best["features"]
    log_t = best["log_target"]

    d = df_feat[df_feat["harvest_date"] == pd.Timestamp(harvest_date)].copy()
    if d.empty:
        print(f"No data for {harvest_date}"); return

    X, y_true, _ = _xy(d, feats)
    y_pred = np.clip(np.expm1(model.predict(X)) if log_t
                     else model.predict(X), 0, None)
    d["y_pred"] = y_pred
    rmse, mae, r2 = _metrics(y_true, y_pred)

    x_vals = sorted(d["field_x"].unique())
    y_vals = sorted(d["field_y"].unique())
    x2i = {v: i for i, v in enumerate(x_vals)}
    y2i = {v: i for i, v in enumerate(y_vals)}
    grid_true = np.zeros((len(y_vals), len(x_vals)))
    grid_pred = np.zeros((len(y_vals), len(x_vals)))
    grid_err  = np.full((len(y_vals), len(x_vals)), np.nan)
    for _, row in d.iterrows():
        xi = x2i[row["field_x"]]; yi = y2i[row["field_y"]]
        grid_true[yi,xi] = row["weight_kg"]
        grid_pred[yi,xi] = row["y_pred"]
        grid_err [yi,xi] = row["y_pred"] - row["weight_kg"]

    vmax = float(np.nanquantile(
        np.concatenate([grid_true.ravel(), grid_pred.ravel()]), 0.99))
    vm_e = float(np.nanquantile(np.abs(grid_err[~np.isnan(grid_err)]), 0.95))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle(
        f"{site}  —  {str(harvest_date)[:10]}  |  "
        f"RMSE={rmse:.3f}  MAE={mae:.3f}  R²={r2:.3f}",
        fontsize=13, fontweight="bold")
    for ax, grid, title, cmap, vmin, vmx in [
        (axes[0], grid_true, "Ground Truth (actual)", "YlOrRd", 0,    vmax),
        (axes[1], grid_pred, "Predicted yield",       "YlOrRd", 0,    vmax),
        (axes[2], grid_err,  "Error (pred−actual)",   "RdBu_r", -vm_e, vm_e),
    ]:
        im = ax.imshow(grid, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmx)
        plt.colorbar(im, ax=ax, label="kg", shrink=0.85)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("field_x index"); ax.set_ylabel("field_y index")
    axes[0].text(0.02,0.97, f'Total: {y_true.sum():,.0f} kg',
                 transform=axes[0].transAxes, fontsize=9, va='top', color='white',
                 bbox=dict(boxstyle='round,pad=0.3', fc='#333', alpha=0.65))
    axes[1].text(0.02,0.97, f'Total: {y_pred.sum():,.0f} kg',
                 transform=axes[1].transAxes, fontsize=9, va='top', color='white',
                 bbox=dict(boxstyle='round,pad=0.3', fc='#333', alpha=0.65))
    plt.tight_layout(); plt.show()
    return fig