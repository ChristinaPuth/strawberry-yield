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
models.py
---------
Dual-target ablation study and model comparison.

Two models trained simultaneously:
    Model 1 (yield):  predicts weight_kg       (grid-cell level)
    Model 2 (days):   predicts optimal_days    (grid-cell level → field median)

Ablation configurations (8) — same feature sets, evaluated on BOTH targets:
    A0: yield_lag1 only
    A1: temporal yield features (lag1/2/3 + rolling + trend)
    A2: full temporal block (A1 + cumulative + day_of_year)
         NOTE: days_since_last removed from all configs vs v1
    A3: spatial only (field_x/y + neighbors)
    A4: spatio-temporal, no weather (A2 + A3)
    A5: A4 + 3 core weather vars
    A6: A4 + all 9 weather vars
    A7: all 20 features (complete)

Models compared: Linear Regression, Random Forest, LightGBM, XGBoost,
                 LightGBM + log(y+1)
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


# ── Feature groups (days_since_last removed everywhere) ──────────────────────

FEATURE_GROUPS = {
    "temporal_yield": [
        "yield_lag1", "yield_lag2", "yield_lag3",
        "rolling_mean_3", "yield_trend",
    ],
    # days_since_last removed; day_of_year kept (no leakage)
    "temporal_full": [
        "yield_lag1", "yield_lag2", "yield_lag3",
        "rolling_mean_3", "yield_trend",
        "season_cumulative", "day_of_year",
    ],
    "spatial": [
        "field_x", "field_y",
        "neighbor_mean_3x3", "neighbor_mean_5x5",
    ],
    "weather_core": [
        "temp_mean_7d", "precip_7d", "et0_7d",
    ],
    "weather_full": [
        "temp_mean_7d", "temp_max_7d", "temp_min_7d",
        "precip_7d", "et0_7d", "humidity_mean_7d",
        "soil_moisture_0_7", "soil_moisture_7_28", "daylight_7d",
    ],
}

ABLATION_CONFIGS = {
    "A0": ["yield_lag1"],
    "A1": FEATURE_GROUPS["temporal_yield"],
    "A2": FEATURE_GROUPS["temporal_full"],
    "A3": FEATURE_GROUPS["spatial"],
    "A4": FEATURE_GROUPS["temporal_full"] + FEATURE_GROUPS["spatial"],
    "A5": FEATURE_GROUPS["temporal_full"] + FEATURE_GROUPS["spatial"]
          + FEATURE_GROUPS["weather_core"],
    "A6": FEATURE_GROUPS["temporal_full"] + FEATURE_GROUPS["spatial"]
          + FEATURE_GROUPS["weather_full"],
    "A7": FEATURE_GROUPS["temporal_full"] + FEATURE_GROUPS["spatial"]
          + FEATURE_GROUPS["weather_full"],
}

ABLATION_DESCRIPTIONS = {
    "A0": "yield_lag1 only (minimum baseline)",
    "A1": "Temporal yield only (5 features)",
    "A2": "Full temporal block (7 features, no days_since_last)",
    "A3": "Spatial only — no yield history",
    "A4": "Spatio-temporal, no weather (11 features)",
    "A5": "A4 + 3 core weather vars (14 features)",
    "A6": "A4 + all 9 weather vars (20 features)",
    "A7": "All 20 features (complete set)",
}

TARGET_YIELD = "weight_kg"
TARGET_DAYS  = "optimal_days"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _xy(df, features, target):
    available = [f for f in features if f in df.columns]
    X = df[available].values.astype(np.float32)
    y = df[target].values.astype(np.float32)
    return X, y, available


def _metrics(y_true, y_pred):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))
    return rmse, mae, r2


def _default_lgbm(target="yield"):
    """Slightly different hyperparams for days vs yield target."""
    if target == "days":
        return lgb.LGBMRegressor(
            n_estimators=200, num_leaves=31,
            learning_rate=0.05, min_child_samples=50,
            random_state=42, verbose=-1, n_jobs=-1,
        )
    return lgb.LGBMRegressor(
        n_estimators=200, num_leaves=63,
        learning_rate=0.05, min_child_samples=20,
        random_state=42, verbose=-1, n_jobs=-1,
    )


# ── 1. Joint ablation study ───────────────────────────────────────────────────

def run_ablation(splits: dict, site: str) -> pd.DataFrame:
    """
    Joint ablation: same 8 feature configs evaluated on BOTH targets.

    Trains two LightGBM models per config:
      - Model 1: predicts weight_kg
      - Model 2: predicts optimal_days

    Returns DataFrame with both sets of val metrics per config.
    """
    train_df = splits["train"]
    val_df   = splits["val"]

    print(f"\nJoint ablation study — {site}")
    print(f"Train: {len(train_df):,} rows  |  Val: {len(val_df):,} rows")
    print(f"Two targets: weight_kg  +  optimal_days")
    print("=" * 90)
    print(f"  {'Cfg':<4} {'#F':>3}  "
          f"{'Yield val R²':>13} {'Yield RMSE':>11}  "
          f"{'Days val R²':>12} {'Days MAE':>9}  Description")
    print(f"  {'-'*85}")

    records = []
    for cfg_id, features in ABLATION_CONFIGS.items():

        # — Model 1: yield ————————————————————————————————————————————————————
        X_tr_y, y_tr_y, used = _xy(train_df, features, TARGET_YIELD)
        X_va_y, y_va_y, _    = _xy(val_df,   features, TARGET_YIELD)
        m_yield = _default_lgbm("yield")
        m_yield.fit(X_tr_y, y_tr_y)
        _, _, tr_r2_y  = _metrics(y_tr_y, m_yield.predict(X_tr_y))
        va_rmse_y, _, va_r2_y = _metrics(y_va_y, m_yield.predict(X_va_y))

        # — Model 2: optimal_days ─────────────────────────────────────────────
        X_tr_d, y_tr_d, _ = _xy(train_df, features, TARGET_DAYS)
        X_va_d, y_va_d, _ = _xy(val_df,   features, TARGET_DAYS)
        m_days = _default_lgbm("days")
        m_days.fit(X_tr_d, y_tr_d)
        _, _, tr_r2_d       = _metrics(y_tr_d, m_days.predict(X_tr_d))
        va_rmse_d, va_mae_d, va_r2_d = _metrics(y_va_d, m_days.predict(X_va_d))

        desc = ABLATION_DESCRIPTIONS[cfg_id]
        print(f"  {cfg_id:<4} {len(used):>3}  "
              f"{va_r2_y:>13.4f} {va_rmse_y:>11.4f}  "
              f"{va_r2_d:>12.4f} {va_mae_d:>9.4f}  {desc}")

        records.append({
            "config":       cfg_id,
            "description":  desc,
            "n_features":   len(used),
            "features":     used,
            # yield metrics
            "yield_train_r2":  round(tr_r2_y,   4),
            "yield_val_r2":    round(va_r2_y,   4),
            "yield_val_rmse":  round(va_rmse_y, 4),
            # days metrics
            "days_train_r2":   round(tr_r2_d,   4),
            "days_val_r2":     round(va_r2_d,   4),
            "days_val_mae":    round(va_mae_d,  4),
            "days_val_rmse":   round(va_rmse_d, 4),
        })

    print("=" * 90)
    results = pd.DataFrame(records)

    best_y = results.sort_values("yield_val_r2", ascending=False).iloc[0]
    best_d = results.sort_values("days_val_r2",  ascending=False).iloc[0]
    print(f"\nBest for yield : {best_y['config']}  "
          f"(val R²={best_y['yield_val_r2']:.4f}, RMSE={best_y['yield_val_rmse']:.4f})")
    print(f"Best for days  : {best_d['config']}  "
          f"(val R²={best_d['days_val_r2']:.4f}, MAE={best_d['days_val_mae']:.4f} days)\n")
    return results


def best_feature_set(ablation_results: pd.DataFrame,
                     target: str = "yield") -> list:
    """Return feature list of the best config for a given target."""
    col = f"{target}_val_r2"
    return ablation_results.sort_values(col, ascending=False).iloc[0]["features"]


# ── 2. Model comparison ───────────────────────────────────────────────────────

def run_model_comparison(splits: dict,
                          site: str,
                          features: list) -> dict:
    """
    Train and evaluate 5 models on both targets simultaneously.

    Returns dict with keys 'yield' and 'days', each a DataFrame of results.
    """
    train_df = splits["train"]
    val_df   = splits["val"]

    X_tr_y, y_tr_y, used = _xy(train_df, features, TARGET_YIELD)
    X_va_y, y_va_y, _    = _xy(val_df,   features, TARGET_YIELD)
    X_tr_d, y_tr_d, _    = _xy(train_df, features, TARGET_DAYS)
    X_va_d, y_va_d, _    = _xy(val_df,   features, TARGET_DAYS)

    print(f"\nModel comparison — {site}")
    print(f"Features: {len(used)}  |  Train: {len(train_df):,}  |  Val: {len(val_df):,}")

    model_defs = [
        ("Linear Regression",
         LinearRegression(), LinearRegression(), False),
        ("Random Forest",
         RandomForestRegressor(n_estimators=100, max_depth=10,
                               min_samples_leaf=20, random_state=42, n_jobs=-1),
         RandomForestRegressor(n_estimators=100, max_depth=6,
                               min_samples_leaf=50, random_state=42, n_jobs=-1),
         False),
        ("LightGBM",
         lgb.LGBMRegressor(n_estimators=300, num_leaves=63, learning_rate=0.05,
                            min_child_samples=20, random_state=42, verbose=-1, n_jobs=-1),
         lgb.LGBMRegressor(n_estimators=300, num_leaves=31, learning_rate=0.05,
                            min_child_samples=50, random_state=42, verbose=-1, n_jobs=-1),
         False),
        ("XGBoost",
         xgb.XGBRegressor(n_estimators=300, max_depth=6, eta=0.05,
                           subsample=0.8, colsample_bytree=0.8,
                           random_state=42, verbosity=0, n_jobs=-1),
         xgb.XGBRegressor(n_estimators=300, max_depth=4, eta=0.05,
                           subsample=0.8, colsample_bytree=0.8,
                           random_state=42, verbosity=0, n_jobs=-1),
         False),
        ("LightGBM + log(y+1)",
         lgb.LGBMRegressor(n_estimators=300, num_leaves=63, learning_rate=0.05,
                            min_child_samples=20, random_state=42, verbose=-1, n_jobs=-1),
         lgb.LGBMRegressor(n_estimators=300, num_leaves=31, learning_rate=0.05,
                            min_child_samples=50, random_state=42, verbose=-1, n_jobs=-1),
         True),
    ]

    # ── Yield model comparison ────────────────────────────────────────────────
    print(f"\n  {'—'*60}")
    print(f"  TARGET 1: weight_kg")
    print(f"  {'Model':<28} {'Val RMSE':>10} {'Val R²':>8} {'Train R²':>9}")
    print(f"  {'-'*58}")

    yield_records = []
    for name, m_y, _, log_t in model_defs:
        y_fit = np.log1p(y_tr_y) if log_t else y_tr_y
        m_y.fit(X_tr_y, y_fit)
        p_tr = np.clip(np.expm1(m_y.predict(X_tr_y)) if log_t else m_y.predict(X_tr_y), 0, None)
        p_va = np.clip(np.expm1(m_y.predict(X_va_y)) if log_t else m_y.predict(X_va_y), 0, None)
        _, _, tr_r2  = _metrics(y_tr_y, p_tr)
        va_rmse, _, va_r2 = _metrics(y_va_y, p_va)
        print(f"  {name:<28} {va_rmse:>10.4f} {va_r2:>8.4f} {tr_r2:>9.4f}")
        yield_records.append({
            "model": name, "val_rmse": round(va_rmse,4),
            "val_r2": round(va_r2,4), "train_r2": round(tr_r2,4),
            "log_target": log_t, "features": used,
            "_model_obj": m_y,
        })

    # ── Days model comparison ─────────────────────────────────────────────────
    print(f"\n  {'—'*60}")
    print(f"  TARGET 2: optimal_days")
    print(f"  {'Model':<28} {'Val MAE(days)':>13} {'Val R²':>8} {'Train R²':>9}")
    print(f"  {'-'*58}")

    days_records = []
    for name, _, m_d, log_t in model_defs:
        # days are always positive integers — no log transform needed
        m_d.fit(X_tr_d, y_tr_d)
        p_tr_d = np.clip(m_d.predict(X_tr_d), 1, 14)
        p_va_d = np.clip(m_d.predict(X_va_d), 1, 14)
        _, tr_mae_d, tr_r2_d    = _metrics(y_tr_d, p_tr_d)
        va_rmse_d, va_mae_d, va_r2_d = _metrics(y_va_d, p_va_d)
        print(f"  {name:<28} {va_mae_d:>13.4f} {va_r2_d:>8.4f} {tr_r2_d:>9.4f}")
        days_records.append({
            "model": name, "val_mae": round(va_mae_d,4),
            "val_rmse": round(va_rmse_d,4),
            "val_r2": round(va_r2_d,4), "train_r2": round(tr_r2_d,4),
            "features": used, "_model_obj": m_d,
        })

    yield_df = pd.DataFrame(yield_records).sort_values("val_r2", ascending=False)
    days_df  = pd.DataFrame(days_records).sort_values("val_r2",  ascending=False)

    print(f"\n  Best yield model : {yield_df.iloc[0]['model']}  "
          f"(R²={yield_df.iloc[0]['val_r2']:.4f})")
    print(f"  Best days  model : {days_df.iloc[0]['model']}   "
          f"(MAE={days_df.iloc[0]['val_mae']:.4f} days)\n")

    return {"yield": yield_df, "days": days_df}


# ── 3. Test set evaluation ────────────────────────────────────────────────────

def evaluate_on_test(model_results: dict,
                     splits: dict,
                     site: str) -> dict:
    """
    Evaluate best yield model and best days model on held-out test set.
    Call ONCE after all design decisions are final.
    """
    test_df = splits["test"]

    # — Yield ——————————————————————————————————————————————————————————————————
    best_y   = model_results["yield"].iloc[0]
    m_y      = best_y["_model_obj"]
    feats    = best_y["features"]
    log_t    = best_y["log_target"]
    X_te_y, y_te_y, _ = _xy(test_df, feats, TARGET_YIELD)
    p_y = np.clip(np.expm1(m_y.predict(X_te_y)) if log_t else m_y.predict(X_te_y), 0, None)
    te_rmse_y, te_mae_y, te_r2_y = _metrics(y_te_y, p_y)

    # — Days ———————————————————————————————————————————————————————————————————
    best_d   = model_results["days"].iloc[0]
    m_d      = best_d["_model_obj"]
    X_te_d, y_te_d, _ = _xy(test_df, feats, TARGET_DAYS)
    p_d = np.clip(m_d.predict(X_te_d), 1, 14)
    te_rmse_d, te_mae_d, te_r2_d = _metrics(y_te_d, p_d)

    # Field-level days: median of all cell predictions per harvest date
    test_df = test_df.copy()
    test_df["pred_days"] = p_d
    field_days = (test_df.groupby("harvest_date")
                         .agg(pred_days_median=("pred_days","median"),
                              actual_days=("optimal_days","first"))
                         .reset_index())

    print(f"\n{'='*55}")
    print(f"  TEST SET RESULTS — {site}")
    print(f"  {'—'*50}")
    print(f"  Model (yield) : {best_y['model']}")
    print(f"  RMSE          : {te_rmse_y:.4f} kg")
    print(f"  MAE           : {te_mae_y:.4f} kg")
    print(f"  R²            : {te_r2_y:.4f}")
    print(f"  {'—'*50}")
    print(f"  Model (days)  : {best_d['model']}")
    print(f"  RMSE          : {te_rmse_d:.4f} days")
    print(f"  MAE           : {te_mae_d:.4f} days")
    print(f"  R²            : {te_r2_d:.4f}")
    print(f"  {'—'*50}")
    print(f"  Field-level days prediction vs actual:")
    print(field_days.to_string(index=False))
    print(f"{'='*55}\n")

    return {
        # yield
        "yield_model":   best_y["model"],
        "test_rmse_y":   round(te_rmse_y, 4),
        "test_mae_y":    round(te_mae_y,  4),
        "test_r2_y":     round(te_r2_y,   4),
        "y_true_yield":  y_te_y,
        "y_pred_yield":  p_y,
        # days
        "days_model":    best_d["model"],
        "test_rmse_d":   round(te_rmse_d, 4),
        "test_mae_d":    round(te_mae_d,  4),
        "test_r2_d":     round(te_r2_d,   4),
        "y_true_days":   y_te_d,
        "y_pred_days":   p_d,
        "field_days":    field_days,
        "test_df":       test_df,
    }


# ── 4. Feature importance ─────────────────────────────────────────────────────

def plot_feature_importance(model_results: dict, target: str = "yield", top_n: int = 20):
    import matplotlib.pyplot as plt

    df_res = model_results[target]
    tree_models = df_res[df_res["model"].str.contains("LightGBM|XGBoost|Random Forest")]
    if tree_models.empty:
        print("No tree model found.")
        return

    best_row = tree_models.iloc[0]
    model    = best_row["_model_obj"]
    features = best_row["features"]

    if not hasattr(model, "feature_importances_"):
        print("Model has no feature_importances_.")
        return

    imp_df = pd.DataFrame({
        "feature":    features[:len(model.feature_importances_)],
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=True).tail(top_n)

    def _colour(feat):
        if any(feat.startswith(p) for p in ["yield_lag","rolling","yield_trend",
                                              "season","day_of"]):
            return "#E07B39"
        if any(feat.startswith(p) for p in ["field_","neighbor"]):
            return "#5B8DB8"
        return "#6BBF8C"

    colours = [_colour(f) for f in imp_df["feature"]]
    fig, ax = plt.subplots(figsize=(9, 0.4 * len(imp_df) + 1.5))
    ax.barh(imp_df["feature"], imp_df["importance"],
            color=colours, edgecolor="white", linewidth=0.4)
    ax.set_xlabel("Feature importance")
    target_label = "weight_kg" if target == "yield" else "optimal_days"
    ax.set_title(f"Feature importance — {best_row['model']} → {target_label}",
                 fontweight="bold")
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color="#E07B39", label="Temporal"),
        Patch(color="#5B8DB8", label="Spatial"),
        Patch(color="#6BBF8C", label="Weather"),
    ], loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.show()
    return fig


# ── 5. Prediction yield map (with ground truth) ───────────────────────────────

def predict_yield_map(model_results: dict,
                      df_feat: pd.DataFrame,
                      harvest_date,
                      site: str):
    """
    Predicted vs actual yield map for a given harvest date.
    Shows: actual map, predicted map, error map.
    """
    import matplotlib.pyplot as plt

    best_row = model_results["yield"].iloc[0]
    model    = best_row["_model_obj"]
    features = best_row["features"]
    log_t    = best_row["log_target"]

    d = df_feat[df_feat["harvest_date"] == pd.Timestamp(harvest_date)].copy()
    if d.empty:
        print(f"No data for {harvest_date}")
        return

    X, y_true, _ = _xy(d, features, TARGET_YIELD)
    y_pred = np.clip(
        np.expm1(model.predict(X)) if log_t else model.predict(X), 0, None
    )
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
        grid_true[yi, xi] = row["weight_kg"]
        grid_pred[yi, xi] = row["y_pred"]
        grid_err [yi, xi] = row["y_pred"] - row["weight_kg"]

    vmax = float(np.nanquantile(
        np.concatenate([grid_true.ravel(), grid_pred.ravel()]), 0.99))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle(
        f"{site}  —  {str(harvest_date)[:10]}  |  "
        f"RMSE={rmse:.3f} kg   MAE={mae:.3f} kg   R²={r2:.3f}",
        fontsize=13, fontweight="bold"
    )
    for ax, grid, title, cmap in [
        (axes[0], grid_true, "Ground Truth (actual yield)", "YlOrRd"),
        (axes[1], grid_pred, "Predicted yield",             "YlOrRd"),
        (axes[2], grid_err,  "Error (pred − actual)",       "RdBu_r"),
    ]:
        if title.startswith("Error"):
            vm = float(np.nanquantile(np.abs(grid_err[~np.isnan(grid_err)]), 0.95))
            im = ax.imshow(grid, cmap=cmap, aspect="auto", vmin=-vm, vmax=vm)
        else:
            im = ax.imshow(grid, cmap=cmap, aspect="auto", vmin=0, vmax=vmax)
        plt.colorbar(im, ax=ax, label="kg", shrink=0.85)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("field_x index"); ax.set_ylabel("field_y index")

    plt.tight_layout()
    plt.show()
    return fig