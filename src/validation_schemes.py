"""
validation_schemes.py
---------------------
Sliding-window cross-validation schemes for strawberry yield prediction.

Schemes
-------
A.1 : Random 40% window selection, 2-harvest train, 14 features (no temporal)
B   : Chronological split, last 40% as test, 2-harvest train, 14 features
C   : Chronological split, last 40% as test, 5-harvest train, 20 features
D   : Random 40% window selection, 5-harvest train, 20 features
E   : Chronological split, first 40% as test (early season), 2-harvest train, 14 features

Usage
-----
    import validation_schemes as vs

    # Build GRID_FEATS dict first
    GRID_FEATS = {
        '1x1': (df_feat_sm, df_feat_sal),
        '7x7': (feat_sm_7x7, feat_sal_7x7),
        ...
    }

    # Run all schemes across all grids
    cross_df = vs.run_cross_experiment(GRID_FEATS)

    # Print summary table
    vs.print_cross_summary(cross_df)

    # Plot heatmap
    vs.plot_cross_heatmap(cross_df)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import lightgbm as lgb

warnings.filterwarnings('ignore')


# ── Feature sets ──────────────────────────────────────────────────────────────

FEATS_14 = [
    'field_x', 'field_y',
    'neighbor_mean_3x3', 'neighbor_mean_5x5',
    'day_of_year',
    'temp_mean_7d', 'temp_max_7d', 'temp_min_7d',
    'precip_7d', 'et0_7d', 'humidity_mean_7d',
    'soil_moisture_0_7', 'soil_moisture_7_28',
    'daylight_7d',
]

FEATS_20 = FEATS_14 + [
    'yield_lag1', 'yield_lag2', 'yield_lag3',
    'rolling_mean_3', 'yield_trend', 'season_cumulative',
]


# ── Helper: print windows ─────────────────────────────────────────────────────

def _print_windows(val_windows, test_windows, scheme_type='AB'):
    """
    Print train/val/test windows for any scheme.

    scheme_type:
        'AB' — 2-harvest windows (A.1 and B)
        'E'  — 2-harvest windows, test is early season
        'CD' — 5-harvest windows (C and D)
    """
    def fmt_2(windows, label):
        print(f"  {label}:")
        for i, (d1, d2, d3) in enumerate(windows):
            print(f"    [{i:2d}] train: {pd.Timestamp(d1).date()} + "
                  f"{pd.Timestamp(d2).date()}  "
                  f"-> predict: {pd.Timestamp(d3).date()}")

    def fmt_5(windows, label):
        print(f"  {label}:")
        for i, (train_dates, test_date) in enumerate(windows):
            dates_str = ' + '.join(
                [str(pd.Timestamp(d).date()) for d in train_dates])
            print(f"    [{i:2d}] train: {dates_str}  "
                  f"-> predict: {pd.Timestamp(test_date).date()}")

    if scheme_type == 'AB':
        fmt_2(val_windows,  'val windows')
        fmt_2(test_windows, 'test windows')
    elif scheme_type == 'E':
        fmt_2(test_windows, 'test windows (early season)')
        fmt_2(val_windows,  'val windows (late season)')
    elif scheme_type == 'CD':
        fmt_5(val_windows,  'val windows')
        fmt_5(test_windows, 'test windows')


# ── Core training and evaluation ──────────────────────────────────────────────

def train_and_eval_v2(train_df, test_df, features, site, scheme_name):
    """
    Train LightGBM and return cell-level and field-level metrics.
    Includes cell_mape (excludes cells where actual weight_kg == 0).
    """
    avail_feats = [f for f in features if f in train_df.columns]

    X_train = train_df[avail_feats].fillna(0).values.astype(np.float32)
    y_train = train_df['weight_kg'].values
    X_test  = test_df[avail_feats].fillna(0).values.astype(np.float32)
    y_test  = test_df['weight_kg'].values

    model = lgb.LGBMRegressor(
        n_estimators=300, learning_rate=0.05,
        num_leaves=63, min_child_samples=20,
        random_state=42, verbose=-1
    )
    model.fit(X_train, y_train)
    y_pred = np.clip(model.predict(X_test), 0, None)

    # Cell-level metrics
    cell_r2   = r2_score(y_test, y_pred)
    cell_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    cell_mae  = mean_absolute_error(y_test, y_pred)

    # Cell MAPE: exclude zero-yield cells
    nonzero = y_test > 0
    cell_mape = (np.abs(y_test[nonzero] - y_pred[nonzero]) /
                 y_test[nonzero]).mean() * 100 if nonzero.sum() > 0 else np.nan

    # Field-level metrics (total yield per harvest date)
    field = test_df[['harvest_date', 'weight_kg']].copy()
    field['pred'] = y_pred
    field = field.groupby('harvest_date').agg(
        actual=('weight_kg', 'sum'),
        predicted=('pred', 'sum')
    ).reset_index()
    field_r2   = r2_score(field['actual'], field['predicted'])
    field_rmse = np.sqrt(mean_squared_error(field['actual'], field['predicted']))
    field_mape = (np.abs(field['actual'] - field['predicted']) /
                  field['actual'].replace(0, np.nan)).mean() * 100

    return {
        'scheme':     scheme_name,
        'site':       site,
        'n_train':    len(train_df),
        'n_test':     len(test_df),
        'cell_r2':    round(cell_r2,   4),
        'cell_rmse':  round(cell_rmse, 4),
        'cell_mae':   round(cell_mae,  4),
        'cell_mape':  round(cell_mape, 2),
        'field_r2':   round(field_r2,  4),
        'field_rmse': round(field_rmse,2),
        'field_mape': round(field_mape,2),
    }


# ── Temporal feature rebuild (for C and D) ────────────────────────────────────

def rebuild_temporal_feats(df_feat, group_dates):
    """
    Recompute temporal features using only dates within the group.
    Pretend data outside the group does not exist.
    Early-harvest NaN values are filled with the group median.
    """
    group_dates = sorted(group_dates)
    df = df_feat[df_feat['harvest_date'].isin(group_dates)].copy()
    df = df.sort_values(['field_x', 'field_y', 'harvest_date'])

    grp = df.groupby(['field_x', 'field_y'])

    df['yield_lag1'] = grp['weight_kg'].shift(1)
    df['yield_lag2'] = grp['weight_kg'].shift(2)
    df['yield_lag3'] = grp['weight_kg'].shift(3)
    df['rolling_mean_3'] = grp['weight_kg'].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    df['yield_trend'] = (df['yield_lag1'] - df['yield_lag3']) / 2.0
    df['season_cumulative'] = grp['weight_kg'].transform(
        lambda x: x.shift(1).expanding().sum().fillna(0))

    for col in ['yield_lag1', 'yield_lag2', 'yield_lag3',
                'rolling_mean_3', 'yield_trend', 'season_cumulative']:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val if pd.notna(median_val) else 0)

    return df


# ── NaN fix helper ────────────────────────────────────────────────────────────

def _fix_rolling_nan(df_feat):
    """Fix rolling_mean_3 NaN at harvest_idx=2 by filling with yield_lag1."""
    df = df_feat.copy()
    mask = df['rolling_mean_3'].isna()
    if mask.sum() > 0:
        df.loc[mask, 'rolling_mean_3'] = df.loc[mask, 'yield_lag1']
    return df


# ── Scheme A.1 ────────────────────────────────────────────────────────────────

def run_scheme_A1(df_feat, site, features=None, test_ratio=0.4, seed=42):
    """
    A.1: Randomly selected window groups.
    Window size: 2 train + 1 predict.
    Test: random 40% of windows.
    Features: 14 (no temporal).
    """
    if features is None:
        features = FEATS_14
    np.random.seed(seed)
    dates = sorted(df_feat['harvest_date'].unique())
    n = len(dates)
    windows = [(dates[i], dates[i+1], dates[i+2]) for i in range(n-2)]

    n_test   = int(len(windows) * test_ratio)
    test_idx = sorted(np.random.choice(len(windows), n_test, replace=False))
    val_idx  = [i for i in range(len(windows)) if i not in test_idx]

    val_windows  = [windows[i] for i in val_idx]
    test_windows = [windows[i] for i in test_idx]

    print(f"\n{site} A.1: total_windows={len(windows)}, "
          f"val={len(val_windows)}, test={len(test_windows)}")
    _print_windows(val_windows, test_windows, scheme_type='AB')

    results = []
    for split_name, split_windows in [('val', val_windows), ('test', test_windows)]:
        train_rows, test_rows = [], []
        for d1, d2, d3 in split_windows:
            train_rows.append(df_feat[df_feat['harvest_date'].isin([d1, d2])])
            test_rows.append(df_feat[df_feat['harvest_date'] == d3])
        train_df = pd.concat(train_rows, ignore_index=True)
        test_df  = pd.concat(test_rows,  ignore_index=True)
        res = train_and_eval_v2(train_df, test_df, features, site,
                                f'A.1_{split_name}')
        results.append(res)
        print(f"  {split_name}: cell_r2={res['cell_r2']:.4f}  "
              f"cell_mape={res['cell_mape']:.1f}%  "
              f"field_r2={res['field_r2']:.4f}  "
              f"field_mape={res['field_mape']:.1f}%")
    return results


# ── Scheme B ──────────────────────────────────────────────────────────────────

def run_scheme_B(df_feat, site, features=None, train_ratio=0.6):
    """
    B: Chronological sliding-window split.
    Window size: 2 train + 1 predict.
    Test: last 40% of windows (late season).
    Features: 14 (no temporal).
    """
    if features is None:
        features = FEATS_14
    dates = sorted(df_feat['harvest_date'].unique())
    n = len(dates)
    windows = [(dates[i], dates[i+1], dates[i+2]) for i in range(n-2)]

    split_idx    = int(len(windows) * train_ratio)
    val_windows  = windows[:split_idx]
    test_windows = windows[split_idx:]

    print(f"\n{site} B: total_windows={len(windows)}, "
          f"val={len(val_windows)} (first {split_idx}), "
          f"test={len(test_windows)} (last {len(windows)-split_idx})")
    _print_windows(val_windows, test_windows, scheme_type='AB')

    results = []
    for split_name, split_windows in [('val', val_windows), ('test', test_windows)]:
        train_rows, test_rows = [], []
        for d1, d2, d3 in split_windows:
            train_rows.append(df_feat[df_feat['harvest_date'].isin([d1, d2])])
            test_rows.append(df_feat[df_feat['harvest_date'] == d3])
        train_df = pd.concat(train_rows, ignore_index=True)
        test_df  = pd.concat(test_rows,  ignore_index=True)
        res = train_and_eval_v2(train_df, test_df, features, site,
                                f'B_{split_name}')
        results.append(res)
        print(f"  {split_name}: cell_r2={res['cell_r2']:.4f}  "
              f"cell_mape={res['cell_mape']:.1f}%  "
              f"field_r2={res['field_r2']:.4f}  "
              f"field_mape={res['field_mape']:.1f}%")
    return results


# ── Scheme C ──────────────────────────────────────────────────────────────────

def run_scheme_C(df_feat, site, features=None, train_ratio=0.6):
    """
    C: Chronological sliding-window split.
    Window size: 5 train + 1 predict.
    Test: last 40% of windows (late season).
    Features: 20 (full set, rebuilt within each window).
    """
    if features is None:
        features = FEATS_20
    dates = sorted(df_feat['harvest_date'].unique())
    n = len(dates)
    windows = [(dates[i:i+5], dates[i+5]) for i in range(n-5)]

    split_idx    = int(len(windows) * train_ratio)
    val_windows  = windows[:split_idx]
    test_windows = windows[split_idx:]

    print(f"\n{site} C: total_windows={len(windows)}, "
          f"val={len(val_windows)} (first {split_idx}), "
          f"test={len(test_windows)} (last {len(windows)-split_idx})")
    _print_windows(val_windows, test_windows, scheme_type='CD')

    results = []
    for split_name, split_windows in [('val', val_windows), ('test', test_windows)]:
        train_rows, test_rows = [], []
        for train_dates, test_date in split_windows:
            all_dates = list(train_dates) + [test_date]
            group_df  = rebuild_temporal_feats(df_feat, all_dates)
            train_rows.append(group_df[group_df['harvest_date'].isin(train_dates)])
            test_rows.append(group_df[group_df['harvest_date'] == test_date])
        train_df = pd.concat(train_rows, ignore_index=True)
        test_df  = pd.concat(test_rows,  ignore_index=True)
        res = train_and_eval_v2(train_df, test_df, features, site,
                                f'C_{split_name}')
        results.append(res)
        print(f"  {split_name}: cell_r2={res['cell_r2']:.4f}  "
              f"cell_mape={res['cell_mape']:.1f}%  "
              f"field_r2={res['field_r2']:.4f}  "
              f"field_mape={res['field_mape']:.1f}%")
    return results


# ── Scheme D ──────────────────────────────────────────────────────────────────

def run_scheme_D(df_feat, site, features=None, test_ratio=0.4, seed=42):
    """
    D: Randomly selected window groups.
    Window size: 5 train + 1 predict.
    Test: random 40% of windows.
    Features: 20 (full set, rebuilt within each window).
    """
    if features is None:
        features = FEATS_20
    np.random.seed(seed)
    dates = sorted(df_feat['harvest_date'].unique())
    n = len(dates)
    windows = [(dates[i:i+5], dates[i+5]) for i in range(n-5)]

    n_test   = int(len(windows) * test_ratio)
    test_idx = sorted(np.random.choice(len(windows), n_test, replace=False))
    val_idx  = [i for i in range(len(windows)) if i not in test_idx]

    val_windows  = [windows[i] for i in val_idx]
    test_windows = [windows[i] for i in test_idx]

    print(f"\n{site} D: total_windows={len(windows)}, "
          f"val={len(val_windows)}, test={len(test_windows)}")
    print(f"  test window indices: {test_idx}")
    _print_windows(val_windows, test_windows, scheme_type='CD')

    results = []
    for split_name, split_windows in [('val', val_windows), ('test', test_windows)]:
        train_rows, test_rows = [], []
        for train_dates, test_date in split_windows:
            all_dates = list(train_dates) + [test_date]
            group_df  = rebuild_temporal_feats(df_feat, all_dates)
            train_rows.append(group_df[group_df['harvest_date'].isin(train_dates)])
            test_rows.append(group_df[group_df['harvest_date'] == test_date])
        train_df = pd.concat(train_rows, ignore_index=True)
        test_df  = pd.concat(test_rows,  ignore_index=True)
        res = train_and_eval_v2(train_df, test_df, features, site,
                                f'D_{split_name}')
        results.append(res)
        print(f"  {split_name}: cell_r2={res['cell_r2']:.4f}  "
              f"cell_mape={res['cell_mape']:.1f}%  "
              f"field_r2={res['field_r2']:.4f}  "
              f"field_mape={res['field_mape']:.1f}%")
    return results


# ── Scheme E ──────────────────────────────────────────────────────────────────

def run_scheme_E(df_feat, site, features=None, test_ratio=0.4):
    """
    E: Chronological sliding-window split (reversed from B).
    Window size: 2 train + 1 predict.
    Test: first 40% of windows (early season).
    Val:  last 60% of windows (late season).
    Features: 14 (no temporal).
    Contrasts with B to reveal early vs late season difficulty.
    """
    if features is None:
        features = FEATS_14
    dates = sorted(df_feat['harvest_date'].unique())
    n = len(dates)
    windows = [(dates[i], dates[i+1], dates[i+2]) for i in range(n-2)]

    split_idx    = int(len(windows) * test_ratio)
    test_windows = windows[:split_idx]   # early season = test
    val_windows  = windows[split_idx:]   # late season  = val

    print(f"\n{site} E: total_windows={len(windows)}, "
          f"test={len(test_windows)} (first {split_idx}, early season), "
          f"val={len(val_windows)} (last {len(windows)-split_idx}, late season)")
    _print_windows(val_windows, test_windows, scheme_type='E')

    results = []
    for split_name, split_windows in [('val', val_windows), ('test', test_windows)]:
        train_rows, test_rows = [], []
        for d1, d2, d3 in split_windows:
            train_rows.append(df_feat[df_feat['harvest_date'].isin([d1, d2])])
            test_rows.append(df_feat[df_feat['harvest_date'] == d3])
        train_df = pd.concat(train_rows, ignore_index=True)
        test_df  = pd.concat(test_rows,  ignore_index=True)
        res = train_and_eval_v2(train_df, test_df, features, site,
                                f'E_{split_name}')
        results.append(res)
        print(f"  {split_name}: cell_r2={res['cell_r2']:.4f}  "
              f"cell_mape={res['cell_mape']:.1f}%  "
              f"field_r2={res['field_r2']:.4f}  "
              f"field_mape={res['field_mape']:.1f}%")
    return results


# ── Main cross experiment ─────────────────────────────────────────────────────

def run_cross_experiment(grid_feats, schemes='ABCDE', seed_a1=42, seed_d=42):
    """
    Run all schemes across all grid shapes.

    Parameters
    ----------
    grid_feats : dict
        {shape_label: (df_feat_sm, df_feat_sal)}
        e.g. {'1x1': (df_feat_sm, df_feat_sal), '7x7': (...), ...}
    schemes : str
        Which schemes to run, any subset of 'ABCDE'. Default: 'ABCDE'.
    seed_a1 : int
        Random seed for scheme A.1 window selection.
    seed_d : int
        Random seed for scheme D window selection.

    Returns
    -------
    pd.DataFrame with all results (val + test splits)
    """
    all_results = []

    for grid_shape, (df_sm_g, df_sal_g) in grid_feats.items():
        print(f"\n{'#'*60}")
        print(f"  Grid shape: {grid_shape}")
        print(f"{'#'*60}")

        for site, df_feat_g in [('SantaMaria', df_sm_g), ('Salinas', df_sal_g)]:
            print(f"\n  --- {site} ---")

            # Fix rolling_mean_3 NaN
            df_feat_g = _fix_rolling_nan(df_feat_g)

            if 'A' in schemes:
                for r in run_scheme_A1(df_feat_g, site, seed=seed_a1):
                    r['grid'] = grid_shape
                    all_results.append(r)

            if 'B' in schemes:
                for r in run_scheme_B(df_feat_g, site):
                    r['grid'] = grid_shape
                    all_results.append(r)

            if 'C' in schemes:
                for r in run_scheme_C(df_feat_g, site):
                    r['grid'] = grid_shape
                    all_results.append(r)

            if 'D' in schemes:
                for r in run_scheme_D(df_feat_g, site, seed=seed_d):
                    r['grid'] = grid_shape
                    all_results.append(r)

            if 'E' in schemes:
                for r in run_scheme_E(df_feat_g, site):
                    r['grid'] = grid_shape
                    all_results.append(r)

    print("\nAll experiments done!")
    return pd.DataFrame(all_results)


# ── Summary table ─────────────────────────────────────────────────────────────

def print_cross_summary(cross_df, grid_order=None, scheme_order=None):
    """
    Print the cross-experiment summary table (test split only).

    Parameters
    ----------
    cross_df    : output of run_cross_experiment()
    grid_order  : list of grid shape labels in display order
    scheme_order: list of scheme labels in display order
    """
    test_df = cross_df[cross_df['scheme'].str.contains('test')].copy()
    test_df['scheme'] = test_df['scheme'].str.replace('_test', '')

    if grid_order:
        test_df['grid'] = pd.Categorical(
            test_df['grid'], categories=grid_order, ordered=True)
    if scheme_order:
        test_df['scheme'] = pd.Categorical(
            test_df['scheme'], categories=scheme_order, ordered=True)

    test_df = test_df.sort_values(['grid', 'scheme'])

    cols = ['grid', 'scheme', 'cell_r2', 'cell_rmse', 'cell_mape',
            'field_r2', 'field_rmse', 'field_mape']
    # Only include columns that exist
    cols = [c for c in cols if c in test_df.columns]

    print('\n' + '='*110)
    print('  Grid Shape x Scheme Cross Comparison (test split)')
    print('='*110)
    for site in ['SantaMaria', 'Salinas']:
        print(f"\n  {site}:")
        sub = test_df[test_df['site'] == site]
        print(sub[cols].to_string(index=False))
    print('='*110)


# ── Heatmap visualisation ─────────────────────────────────────────────────────

def plot_cross_heatmap(cross_df, grid_order=None, scheme_order=None):
    """
    Plot heatmap of Grid Shape x Scheme for each site.

    Parameters
    ----------
    cross_df     : output of run_cross_experiment()
    grid_order   : list of grid shape labels (rows)
    scheme_order : list of scheme labels (columns)
    """
    test_df = cross_df[cross_df['scheme'].str.contains('test')].copy()
    test_df['scheme'] = test_df['scheme'].str.replace('_test', '')

    grids   = grid_order   or sorted(test_df['grid'].unique())
    schemes = scheme_order or sorted(test_df['scheme'].unique())

    metrics_info = [
        ('cell_r2',    'Cell R²',        True),
        ('cell_mape',  'Cell MAPE (%)',   False),
        ('field_r2',   'Field R²',        True),
        ('field_mape', 'Field MAPE (%)',  False),
    ]
    # Only plot metrics that exist
    metrics_info = [(m, t, h) for m, t, h in metrics_info
                    if m in test_df.columns]

    for site in ['SantaMaria', 'Salinas']:
        fig, axes = plt.subplots(1, len(metrics_info),
                                 figsize=(6 * len(metrics_info), 6))
        if len(metrics_info) == 1:
            axes = [axes]
        fig.suptitle(
            f'{site} — Grid Shape × Scheme '
            f'({len(grids)} grids × {len(schemes)} schemes)',
            fontsize=13, fontweight='bold')

        sub = test_df[test_df['site'] == site]

        for ax, (metric, title, higher_better) in zip(axes, metrics_info):
            matrix = np.full((len(grids), len(schemes)), np.nan)
            for i, grid in enumerate(grids):
                for j, scheme in enumerate(schemes):
                    row = sub[(sub['grid'] == grid) & (sub['scheme'] == scheme)]
                    if len(row) > 0:
                        matrix[i, j] = row[metric].values[0]

            cmap = 'RdYlGn' if higher_better else 'RdYlGn_r'
            im = ax.imshow(matrix, cmap=cmap, aspect='auto')
            plt.colorbar(im, ax=ax, shrink=0.8)

            for i in range(len(grids)):
                for j in range(len(schemes)):
                    if not np.isnan(matrix[i, j]):
                        ax.text(j, i, f'{matrix[i,j]:.2f}',
                                ha='center', va='center',
                                fontsize=8, fontweight='bold', color='black')

            ax.set_xticks(range(len(schemes)))
            ax.set_yticks(range(len(grids)))
            ax.set_xticklabels(schemes, fontsize=10)
            ax.set_yticklabels(grids, fontsize=10)
            ax.set_xlabel('Scheme')
            ax.set_ylabel('Grid Shape')
            ax.set_title(title, fontweight='bold')

        plt.tight_layout()
        plt.show()




# ──────────────────────────────────────────────────────────────────────────────
# Cross-site transfer experiment
# Add this to the END of validation_schemes.py
# ──────────────────────────────────────────────────────────────────────────────

import feature_engineering as fe


def run_transfer_experiment(df_source, df_target,
                             source_name, target_name,
                             features=None,
                             grid_shape='1x1',
                             verbose=True):
    """
    Train on source site, evaluate on target site (cross-site transfer).

    Uses Scheme B window structure on both sites:
      - Source: last 60% windows → training pool
      - Target: all windows      → test pool

    All features and target normalized using source site scaler,
    then applied to target site before prediction.
    Predictions denormalized back to kg for evaluation.

    Parameters
    ----------
    df_source   : feature DataFrame for training site
    df_target   : feature DataFrame for evaluation site
    source_name : e.g. 'SantaMaria'
    target_name : e.g. 'Salinas'
    features    : list of feature columns. Defaults to FEATS_14.
    grid_shape  : label for result tracking (e.g. '7x7')
    verbose     : print progress

    Returns
    -------
    list of result dicts (one per target window)
    """
    if features is None:
        features = FEATS_14

    # ── Fix rolling_mean_3 NaN ────────────────────────────────────────────────
    df_source = _fix_rolling_nan(df_source)
    df_target = _fix_rolling_nan(df_target)

    # ── Build source training pool (Scheme B: last 60% windows) ──────────────
    src_dates = sorted(df_source['harvest_date'].unique())
    n_src = len(src_dates)
    src_windows = [(src_dates[i], src_dates[i+1], src_dates[i+2])
                   for i in range(n_src - 2)]
    split_idx = int(len(src_windows) * 0.6)
    train_windows = src_windows[:split_idx]   # val windows from B
    # Use ALL windows as training pool for transfer (more data = better)
    train_rows = []
    for d1, d2, d3 in src_windows:
        train_rows.append(df_source[df_source['harvest_date'].isin([d1, d2])])
    train_df = pd.concat(train_rows, ignore_index=True).drop_duplicates()

    if verbose:
        print(f"\n  Transfer: {source_name} → {target_name}  "
              f"[grid={grid_shape}, features={len(features)}]")
        print(f"  Source train rows: {len(train_df):,}  "
              f"(from {len(src_windows)} windows, all used)")

    # ── Normalize using source scaler ─────────────────────────────────────────
    train_norm, scaler = fe.normalize_features(train_df, fit=True)

    # ── Build target test windows (all sliding windows) ───────────────────────
    tgt_dates = sorted(df_target['harvest_date'].unique())
    n_tgt = len(tgt_dates)
    tgt_windows = [(tgt_dates[i], tgt_dates[i+1], tgt_dates[i+2])
                   for i in range(n_tgt - 2)]

    if verbose:
        print(f"  Target test windows: {len(tgt_windows)}")

    # ── Train model on normalized source data ─────────────────────────────────
    avail_feats = [f for f in features if f in train_norm.columns]
    X_train = train_norm[avail_feats].fillna(0).values.astype(np.float32)
    y_train = train_norm['weight_kg'].values   # normalized target

    model = lgb.LGBMRegressor(
        n_estimators=300, learning_rate=0.05,
        num_leaves=63, min_child_samples=20,
        random_state=42, verbose=-1
    )
    model.fit(X_train, y_train)

    # ── Predict on each target window ─────────────────────────────────────────
    all_test_rows, all_pred_rows = [], []

    for d1, d2, d3 in tgt_windows:
        # Build test set for this target window
        test_df_win = df_target[df_target['harvest_date'] == d3].copy()

        # Normalize using SOURCE scaler (cross-site transfer)
        test_norm, _ = fe.normalize_features(
            test_df_win, scaler=scaler, fit=False)

        X_test = test_norm[avail_feats].fillna(0).values.astype(np.float32)
        y_pred_norm = np.clip(model.predict(X_test), 0, None)

        # Denormalize predictions back to kg
        y_pred_kg = fe.denormalize_predictions(y_pred_norm, scaler)
        y_true_kg = test_df_win['weight_kg'].values

        all_test_rows.append(
            test_df_win[['harvest_date', 'weight_kg']].assign(pred=y_pred_kg))

    # ── Aggregate all target windows ──────────────────────────────────────────
    all_test_df = pd.concat(all_test_rows, ignore_index=True)
    y_test = all_test_df['weight_kg'].values
    y_pred = all_test_df['pred'].values

    # Cell-level metrics
    cell_r2   = r2_score(y_test, y_pred)
    cell_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    cell_mae  = mean_absolute_error(y_test, y_pred)
    nonzero   = y_test > 0
    cell_mape = (np.abs(y_test[nonzero] - y_pred[nonzero]) /
                 y_test[nonzero]).mean() * 100 if nonzero.sum() > 0 else np.nan

    # Field-level metrics
    field = all_test_df.groupby('harvest_date').agg(
        actual=('weight_kg', 'sum'),
        predicted=('pred', 'sum')
    ).reset_index()
    field_r2   = r2_score(field['actual'], field['predicted'])
    field_rmse = np.sqrt(mean_squared_error(field['actual'], field['predicted']))
    field_mape = (np.abs(field['actual'] - field['predicted']) /
                  field['actual'].replace(0, np.nan)).mean() * 100

    result = {
        'direction':  f'{source_name}→{target_name}',
        'source':     source_name,
        'target':     target_name,
        'grid':       grid_shape,
        'n_features': len(avail_feats),
        'n_train':    len(train_df),
        'n_test':     len(all_test_df),
        'cell_r2':    round(cell_r2,   4),
        'cell_rmse':  round(cell_rmse, 4),
        'cell_mae':   round(cell_mae,  4),
        'cell_mape':  round(cell_mape, 2),
        'field_r2':   round(field_r2,  4),
        'field_rmse': round(field_rmse,2),
        'field_mape': round(field_mape,2),
    }

    if verbose:
        print(f"  cell_r2={cell_r2:.4f}  cell_mape={cell_mape:.1f}%  "
              f"field_r2={field_r2:.4f}  field_mape={field_mape:.1f}%")

    return result


def run_cross_site_experiment(grid_feats,
                               features=None,
                               verbose=True):
    """
    Run bidirectional cross-site transfer for all grid shapes.

    For each grid shape:
      - SM → Sal  (train on SantaMaria, test on Salinas)
      - Sal → SM  (train on Salinas, test on SantaMaria)

    Also runs within-site B scheme for comparison baseline.

    Parameters
    ----------
    grid_feats : dict {shape: (df_sm, df_sal)}
    features   : feature list. Defaults to FEATS_14.
    verbose    : print progress

    Returns
    -------
    pd.DataFrame with transfer results + within-site baseline
    """
    if features is None:
        features = FEATS_14

    results = []

    for grid_shape, (df_sm_g, df_sal_g) in grid_feats.items():
        print(f"\n{'='*60}")
        print(f"  Grid: {grid_shape}")
        print(f"{'='*60}")

        # SM → Sal
        res = run_transfer_experiment(
            df_sm_g, df_sal_g,
            'SantaMaria', 'Salinas',
            features=features,
            grid_shape=grid_shape,
            verbose=verbose
        )
        results.append(res)

        # Sal → SM
        res = run_transfer_experiment(
            df_sal_g, df_sm_g,
            'Salinas', 'SantaMaria',
            features=features,
            grid_shape=grid_shape,
            verbose=verbose
        )
        results.append(res)

    return pd.DataFrame(results)


def print_transfer_summary(transfer_df, within_df=None):
    """
    Print cross-site transfer results alongside within-site B baseline.

    Parameters
    ----------
    transfer_df : output of run_cross_site_experiment()
    within_df   : output of run_cross_experiment() for within-site comparison.
                  If provided, B scheme results are shown next to transfer results.
    """
    print('\n' + '='*100)
    print('  Cross-Site Transfer Results')
    print('='*100)

    cols = ['direction', 'grid', 'cell_r2', 'cell_mape',
            'field_r2', 'field_mape']
    print(transfer_df[cols].to_string(index=False))

    if within_df is not None:
        # Extract B scheme test results for comparison
        b_df = within_df[within_df['scheme'].str.contains('test')].copy()
        b_df = b_df[b_df['scheme'].str.startswith('B')].copy()
        b_df['scheme'] = 'B (within-site)'

        print('\n' + '-'*100)
        print('  Within-Site Baseline (Scheme B, for comparison)')
        print('-'*100)
        b_cols = ['site', 'grid', 'cell_r2', 'cell_mape',
                  'field_r2', 'field_mape']
        b_cols = [c for c in b_cols if c in b_df.columns]
        print(b_df[b_cols].to_string(index=False))

    print('='*100)


def plot_transfer_vs_within(transfer_df, within_df,
                             grid_order=None):
    """
    Bar chart comparing cross-site transfer vs within-site B for each grid.

    Parameters
    ----------
    transfer_df : output of run_cross_site_experiment()
    within_df   : output of run_cross_experiment()
    grid_order  : display order of grid shapes
    """
    grids = grid_order or sorted(transfer_df['grid'].unique())

    metrics = [
        ('cell_r2',    'Cell R²',       True),
        ('cell_mape',  'Cell MAPE (%)', False),
        ('field_r2',   'Field R²',      True),
        ('field_mape', 'Field MAPE (%)', False),
    ]

    # Within-site B results
    b_df = within_df[within_df['scheme'].str.contains('test')].copy()
    b_df = b_df[b_df['scheme'].str.startswith('B')].copy()
    b_df['scheme'] = b_df['scheme'].str.replace('_test', '')

    fig, axes = plt.subplots(1, len(metrics), figsize=(22, 6))
    fig.suptitle('Cross-Site Transfer vs Within-Site (Scheme B)',
                 fontsize=13, fontweight='bold')

    colors = {
        'SM→Sal':             '#E07B54',
        'Sal→SM':             '#5B8DB8',
        'B SM (within)':      '#2d6a3f',
        'B Sal (within)':     '#8DB85B',
    }

    x = np.arange(len(grids))
    width = 0.2

    for ax, (metric, title, higher_better) in zip(axes, metrics):
        # SM→Sal transfer
        sm2sal = [transfer_df[
            (transfer_df['grid']==g) &
            (transfer_df['direction']=='SantaMaria→Salinas')
        ][metric].values[0] if len(transfer_df[
            (transfer_df['grid']==g) &
            (transfer_df['direction']=='SantaMaria→Salinas')]) > 0
        else np.nan for g in grids]

        # Sal→SM transfer
        sal2sm = [transfer_df[
            (transfer_df['grid']==g) &
            (transfer_df['direction']=='Salinas→SantaMaria')
        ][metric].values[0] if len(transfer_df[
            (transfer_df['grid']==g) &
            (transfer_df['direction']=='Salinas→SantaMaria')]) > 0
        else np.nan for g in grids]

        # Within-site B (SantaMaria)
        b_sm = [b_df[
            (b_df['grid']==g) & (b_df['site']=='SantaMaria')
        ][metric].values[0] if len(b_df[
            (b_df['grid']==g) & (b_df['site']=='SantaMaria')]) > 0
        else np.nan for g in grids]

        # Within-site B (Salinas)
        b_sal = [b_df[
            (b_df['grid']==g) & (b_df['site']=='Salinas')
        ][metric].values[0] if len(b_df[
            (b_df['grid']==g) & (b_df['site']=='Salinas')]) > 0
        else np.nan for g in grids]

        for offset, vals, label in zip(
            [-1.5*width, -0.5*width, 0.5*width, 1.5*width],
            [sm2sal, sal2sm, b_sm, b_sal],
            ['SM→Sal', 'Sal→SM', 'B SM (within)', 'B Sal (within)']
        ):
            bars = ax.bar(x + offset, vals, width,
                          label=label, color=colors[label], alpha=0.85)
            for bar, v in zip(bars, vals):
                if not np.isnan(v):
                    ax.text(bar.get_x() + bar.get_width()/2,
                            bar.get_height() + 0.005,
                            f'{v:.2f}', ha='center',
                            fontsize=7, color=colors[label])

        ax.set_xticks(x)
        ax.set_xticklabels(grids, fontsize=9)
        ax.set_ylabel(title)
        ax.set_title(title, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.show()


def run_transfer_experiment_A1(df_source, df_target,
                                source_name, target_name,
                                features=None,
                                grid_shape='1x1',
                                test_ratio=0.4,
                                seed=42,
                                verbose=True):
    """
    Train on source site using Scheme A.1 window logic,
    evaluate on target site (cross-site transfer).

    A.1: randomly selected windows, 2-harvest train, 14 features.
    Source: random 60% windows -> training pool.
    Target: all windows        -> test pool.
    All features normalized using source scaler.
    """
    if features is None:
        features = FEATS_14

    np.random.seed(seed)

    # Fix rolling_mean_3 NaN
    df_source = _fix_rolling_nan(df_source)
    df_target = _fix_rolling_nan(df_target)

    # ── Build source training pool (A.1: random 60% windows) ─────────────────
    src_dates = sorted(df_source['harvest_date'].unique())
    n_src = len(src_dates)
    src_windows = [(src_dates[i], src_dates[i+1], src_dates[i+2])
                   for i in range(n_src - 2)]

    # Random split: 60% val (training pool), 40% test (unused for transfer)
    n_test   = int(len(src_windows) * test_ratio)
    test_idx = sorted(np.random.choice(len(src_windows), n_test, replace=False))
    val_idx  = [i for i in range(len(src_windows)) if i not in test_idx]
    train_windows = [src_windows[i] for i in val_idx]

    train_rows = []
    for d1, d2, d3 in train_windows:
        train_rows.append(df_source[df_source['harvest_date'].isin([d1, d2])])
    train_df = pd.concat(train_rows, ignore_index=True).drop_duplicates()

    if verbose:
        print(f"\n  Transfer A.1: {source_name} → {target_name}  "
              f"[grid={grid_shape}, features={len(features)}]")
        print(f"  Source train rows: {len(train_df):,}  "
              f"(from {len(train_windows)}/{len(src_windows)} windows, "
              f"random 60%)")

    # ── Normalize using source scaler ─────────────────────────────────────────
    train_norm, scaler = fe.normalize_features(train_df, fit=True)

    # ── Build target test windows (all sliding windows) ───────────────────────
    tgt_dates = sorted(df_target['harvest_date'].unique())
    n_tgt = len(tgt_dates)
    tgt_windows = [(tgt_dates[i], tgt_dates[i+1], tgt_dates[i+2])
                   for i in range(n_tgt - 2)]

    if verbose:
        print(f"  Target test windows: {len(tgt_windows)}")

    # ── Train model ───────────────────────────────────────────────────────────
    avail_feats = [f for f in features if f in train_norm.columns]
    X_train = train_norm[avail_feats].fillna(0).values.astype(np.float32)
    y_train = train_norm['weight_kg'].values

    model = lgb.LGBMRegressor(
        n_estimators=300, learning_rate=0.05,
        num_leaves=63, min_child_samples=20,
        random_state=42, verbose=-1
    )
    model.fit(X_train, y_train)

    # ── Predict on each target window ─────────────────────────────────────────
    all_test_rows = []
    for d1, d2, d3 in tgt_windows:
        test_df_win = df_target[df_target['harvest_date'] == d3].copy()
        test_norm, _ = fe.normalize_features(
            test_df_win, scaler=scaler, fit=False)
        X_test = test_norm[avail_feats].fillna(0).values.astype(np.float32)
        y_pred_norm = np.clip(model.predict(X_test), 0, None)
        y_pred_kg   = fe.denormalize_predictions(y_pred_norm, scaler)
        all_test_rows.append(
            test_df_win[['harvest_date', 'weight_kg']].assign(pred=y_pred_kg))

    # ── Metrics ───────────────────────────────────────────────────────────────
    all_test_df = pd.concat(all_test_rows, ignore_index=True)
    y_test = all_test_df['weight_kg'].values
    y_pred = all_test_df['pred'].values

    cell_r2   = r2_score(y_test, y_pred)
    cell_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    cell_mae  = mean_absolute_error(y_test, y_pred)
    nonzero   = y_test > 0
    cell_mape = (np.abs(y_test[nonzero] - y_pred[nonzero]) /
                 y_test[nonzero]).mean() * 100 if nonzero.sum() > 0 else np.nan

    field = all_test_df.groupby('harvest_date').agg(
        actual=('weight_kg', 'sum'),
        predicted=('pred', 'sum')
    ).reset_index()
    field_r2   = r2_score(field['actual'], field['predicted'])
    field_rmse = np.sqrt(mean_squared_error(field['actual'], field['predicted']))
    field_mape = (np.abs(field['actual'] - field['predicted']) /
                  field['actual'].replace(0, np.nan)).mean() * 100

    result = {
        'direction':  f'{source_name}→{target_name}',
        'scheme':     'A.1',
        'source':     source_name,
        'target':     target_name,
        'grid':       grid_shape,
        'n_features': len(avail_feats),
        'n_train':    len(train_df),
        'n_test':     len(all_test_df),
        'cell_r2':    round(cell_r2,   4),
        'cell_rmse':  round(cell_rmse, 4),
        'cell_mae':   round(cell_mae,  4),
        'cell_mape':  round(cell_mape, 2),
        'field_r2':   round(field_r2,  4),
        'field_rmse': round(field_rmse,2),
        'field_mape': round(field_mape,2),
    }

    if verbose:
        print(f"  cell_r2={cell_r2:.4f}  cell_mape={cell_mape:.1f}%  "
              f"field_r2={field_r2:.4f}  field_mape={field_mape:.1f}%")

    return result


def run_cross_site_A1(grid_feats, features=None, seed=42, verbose=True):
    """
    Run bidirectional A.1 cross-site transfer for all grid shapes.

    Parameters
    ----------
    grid_feats : dict {shape: (df_sm, df_sal)}
    features   : feature list. Defaults to FEATS_14.
    seed       : random seed for A.1 window selection.
    verbose    : print progress.

    Returns
    -------
    pd.DataFrame with transfer results
    """
    if features is None:
        features = FEATS_14

    results = []

    for grid_shape, (df_sm_g, df_sal_g) in grid_feats.items():
        print(f"\n{'='*60}")
        print(f"  Grid: {grid_shape}  [Scheme A.1]")
        print(f"{'='*60}")

        # SM → Sal
        res = run_transfer_experiment_A1(
            df_sm_g, df_sal_g,
            'SantaMaria', 'Salinas',
            features=features,
            grid_shape=grid_shape,
            seed=seed,
            verbose=verbose
        )
        results.append(res)

        # Sal → SM
        res = run_transfer_experiment_A1(
            df_sal_g, df_sm_g,
            'Salinas', 'SantaMaria',
            features=features,
            grid_shape=grid_shape,
            seed=seed,
            verbose=verbose
        )
        results.append(res)

    return pd.DataFrame(results)




    # ── All-schemes transfer experiment ──────────────────────────────────────────
 
def _transfer_metrics(all_test_df):
    """Compute cell and field metrics from concatenated test rows."""
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
 
    y_test = all_test_df['weight_kg'].values
    y_pred = all_test_df['pred'].values
 
    cell_r2   = r2_score(y_test, y_pred)
    cell_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    cell_mae  = mean_absolute_error(y_test, y_pred)
    nonzero   = y_test > 0
    cell_mape = (np.abs(y_test[nonzero] - y_pred[nonzero]) /
                 y_test[nonzero]).mean() * 100 if nonzero.sum() > 0 else np.nan
 
    field = all_test_df.groupby('harvest_date').agg(
        actual=('weight_kg', 'sum'),
        predicted=('pred', 'sum')
    ).reset_index()
    field_r2   = r2_score(field['actual'], field['predicted'])
    field_rmse = np.sqrt(mean_squared_error(field['actual'], field['predicted']))
    field_mape = (np.abs(field['actual'] - field['predicted']) /
                  field['actual'].replace(0, np.nan)).mean() * 100
 
    return {
        'cell_r2':    round(cell_r2,   4),
        'cell_rmse':  round(cell_rmse, 4),
        'cell_mae':   round(cell_mae,  4),
        'cell_mape':  round(cell_mape, 2),
        'field_r2':   round(field_r2,  4),
        'field_rmse': round(field_rmse,2),
        'field_mape': round(field_mape,2),
    }
 
 
def _train_and_predict(train_norm, test_dfs, avail_feats, scaler, seed=42):
    """Train LightGBM on normalized source data and predict on list of test DataFrames."""
    import feature_engineering as fe
 
    X_train = train_norm[avail_feats].fillna(0).values.astype(np.float32)
    y_train = train_norm['weight_kg'].values
 
    model = lgb.LGBMRegressor(
        n_estimators=300, learning_rate=0.05,
        num_leaves=63, min_child_samples=20,
        random_state=seed, verbose=-1
    )
    model.fit(X_train, y_train)
 
    all_rows = []
    for test_df_win in test_dfs:
        test_norm, _ = fe.normalize_features(
            test_df_win, scaler=scaler, fit=False)
        X_test = test_norm[avail_feats].fillna(0).values.astype(np.float32)
        y_pred_norm = np.clip(model.predict(X_test), 0, None)
        y_pred_kg   = fe.denormalize_predictions(y_pred_norm, scaler)
        all_rows.append(
            test_df_win[['harvest_date', 'weight_kg']].assign(pred=y_pred_kg))
 
    return pd.concat(all_rows, ignore_index=True)
 
 
def run_all_transfer_schemes(df_source, df_target,
                              source_name, target_name,
                              grid_shape, seed=42):
    """
    Run all transfer schemes (A.1/B/C/D/E/AllWindows) for one direction
    and one grid shape. All use 14 features (no temporal).
 
    Parameters
    ----------
    df_source   : feature DataFrame for training site
    df_target   : feature DataFrame for evaluation site
    source_name : e.g. 'SantaMaria'
    target_name : e.g. 'Salinas'
    grid_shape  : label for result tracking (e.g. '7x7')
    seed        : random seed
 
    Returns
    -------
    list of result dicts, one per scheme
    """
    import feature_engineering as fe
 
    np.random.seed(seed)
 
    df_source = _fix_rolling_nan(df_source)
    df_target = _fix_rolling_nan(df_target)
 
    src_dates = sorted(df_source['harvest_date'].unique())
    tgt_dates = sorted(df_target['harvest_date'].unique())
    n_src = len(src_dates)
    n_tgt = len(tgt_dates)
 
    features = FEATS_14
 
    # Target test windows (predict day only)
    tgt_windows_2 = [df_target[df_target['harvest_date'] == tgt_dates[i+2]].copy()
                     for i in range(n_tgt - 2)]
    tgt_windows_5 = [df_target[df_target['harvest_date'] == tgt_dates[i+5]].copy()
                     for i in range(n_tgt - 5)]
 
    # Source window date lists
    src_win2 = [(src_dates[i], src_dates[i+1]) for i in range(n_src - 2)]
    src_win5 = [src_dates[i:i+5] for i in range(n_src - 5)]
 
    def build_train(windows_dates):
        rows = []
        for dates in windows_dates:
            if isinstance(dates, tuple):
                dates = list(dates)
            rows.append(df_source[df_source['harvest_date'].isin(dates)])
        return pd.concat(rows, ignore_index=True).drop_duplicates()
 
    results = []
 
    # Scheme indices for 2-harvest windows
    n2 = len(src_win2)
    n_val2 = int(n2 * 0.6)
    n5 = len(src_win5)
    n_val5 = int(n5 * 0.6)
 
    scheme_configs = [
        # (name, train_dates_list, tgt_windows)
        ('A.1',        [src_win2[i] for i in
                        sorted(np.random.choice(n2, n_val2, replace=False))
                        if True],                          tgt_windows_2),
        ('B',          src_win2[:n_val2],                  tgt_windows_2),
        ('C',          src_win5[:n_val5],                  tgt_windows_5),
        ('D',          [src_win5[i] for i in
                        sorted(np.random.choice(n5, n_val5, replace=False))
                        if True],                          tgt_windows_5),
        ('E',          src_win2[int(n2*0.4):],             tgt_windows_2),
        ('AllWindows', src_win2,                           tgt_windows_2),
    ]
 
    # Rebuild A.1 and D with proper random selection
    np.random.seed(seed)
    idx_a1 = sorted(np.random.choice(n2, n_val2, replace=False))
    idx_d  = sorted(np.random.choice(n5, n_val5, replace=False))
 
    scheme_configs = [
        ('A.1',        [src_win2[i] for i in idx_a1],     tgt_windows_2),
        ('B',          src_win2[:n_val2],                  tgt_windows_2),
        ('C',          src_win5[:n_val5],                  tgt_windows_5),
        ('D',          [src_win5[i] for i in idx_d],       tgt_windows_5),
        ('E',          src_win2[int(n2*0.4):],             tgt_windows_2),
        ('AllWindows', src_win2,                           tgt_windows_2),
    ]
 
    for scheme_name, train_dates, tgt_wins in scheme_configs:
        train_df   = build_train(train_dates)
        train_norm, scaler = fe.normalize_features(train_df, fit=True)
        avail      = [f for f in features if f in train_norm.columns]
        all_test   = _train_and_predict(train_norm, tgt_wins, avail, scaler, seed)
 
        res = {
            'scheme':    scheme_name,
            'direction': f'{source_name}→{target_name}',
            'grid':      grid_shape,
            'n_train':   len(train_df),
            'n_test':    len(all_test),
        }
        res.update(_transfer_metrics(all_test))
        results.append(res)
        print(f"  {scheme_name:12s}: cell_r2={res['cell_r2']:.4f}  "
              f"field_r2={res['field_r2']:.4f}  "
              f"field_mape={res['field_mape']:.1f}%")
 
    return results
 
 
def run_full_transfer_experiment(grid_feats, seed=42, verbose=True):
    """
    Run all transfer schemes (A.1/B/C/D/E/AllWindows) bidirectionally
    for all grid shapes.
 
    Parameters
    ----------
    grid_feats : dict {shape: (df_sm, df_sal)}
    seed       : random seed
    verbose    : print progress
 
    Returns
    -------
    pd.DataFrame with all results
    """
    all_results = []
 
    for grid_shape, (df_sm_g, df_sal_g) in grid_feats.items():
        print(f"\n{'#'*60}")
        print(f"  Grid: {grid_shape}")
        print(f"{'#'*60}")
 
        print(f"\n  SantaMaria → Salinas")
        res = run_all_transfer_schemes(
            df_sm_g, df_sal_g,
            'SantaMaria', 'Salinas',
            grid_shape=grid_shape, seed=seed,
        )
        all_results.extend(res)
 
        print(f"\n  Salinas → SantaMaria")
        res = run_all_transfer_schemes(
            df_sal_g, df_sm_g,
            'Salinas', 'SantaMaria',
            grid_shape=grid_shape, seed=seed,
        )
        all_results.extend(res)
 
    print("\nAll transfer experiments done!")
    return pd.DataFrame(all_results)
 
 
def print_full_transfer_summary(full_transfer_df, within_df=None,
                                 grids=None):
    """
    Print full transfer experiment results with optional within-site baseline.
 
    Parameters
    ----------
    full_transfer_df : output of run_full_transfer_experiment()
    within_df        : output of run_cross_experiment() for within-site B baseline
    grids            : list of grid shapes to show
    """
    if grids is None:
        grids = sorted(full_transfer_df['grid'].unique())
 
    cols = ['direction', 'scheme', 'grid',
            'n_train', 'cell_r2', 'cell_mape',
            'field_r2', 'field_mape']
 
    print('\n' + '='*110)
    print('  Full Cross-Site Transfer — All Schemes (A.1/B/C/D/E/AllWindows)')
    print('='*110)
 
    for grid in grids:
        print(f"\n  Grid: {grid}")
        sub = full_transfer_df[full_transfer_df['grid'] == grid].sort_values(
            ['direction', 'scheme'])
        print(sub[cols].to_string(index=False))
 
    if within_df is not None:
        print('\n' + '='*110)
        print('  Within-Site Baseline (Scheme B, upper bound)')
        print('='*110)
        b = within_df[within_df['scheme'].str.contains('test')].copy()
        b = b[b['scheme'].str.startswith('B')].copy()
        b['scheme'] = b['scheme'].str.replace('_test', '')
        print(b[['site', 'grid', 'cell_r2', 'cell_mape',
                  'field_r2', 'field_mape']].to_string(index=False))
 
    print('='*110)
 
 
def plot_full_transfer(full_transfer_df, grids=None):
    """
    Bar chart comparing all transfer schemes for each direction.
 
    Parameters
    ----------
    full_transfer_df : output of run_full_transfer_experiment()
    grids            : list of grid shapes to show
    """
    import matplotlib.pyplot as plt
 
    if grids is None:
        grids = sorted(full_transfer_df['grid'].unique())
 
    schemes_order = ['A.1', 'B', 'C', 'D', 'E', 'AllWindows']
    directions    = ['SantaMaria→Salinas', 'Salinas→SantaMaria']
    metrics       = [
        ('cell_r2',    'Cell R²',        True),
        ('cell_mape',  'Cell MAPE (%)',  False),
        ('field_r2',   'Field R²',       True),
        ('field_mape', 'Field MAPE (%)', False),
    ]
    colors = {
        'A.1':        '#2d6a3f',
        'B':          '#5B8DB8',
        'C':          '#E07B54',
        'D':          '#9B59B6',
        'E':          '#F4D03F',
        'AllWindows': '#E74C3C',
    }
 
    for direction in directions:
        fig, axes = plt.subplots(1, 4, figsize=(24, 6))
        fig.suptitle(f'Transfer: {direction}  (14 features, no temporal)',
                     fontsize=13, fontweight='bold')
 
        sub = full_transfer_df[full_transfer_df['direction'] == direction]
        x = np.arange(len(grids))
        width = 0.13
 
        for ax, (metric, title, higher_better) in zip(axes, metrics):
            for k, scheme in enumerate(schemes_order):
                vals = []
                for grid in grids:
                    row = sub[(sub['grid'] == grid) & (sub['scheme'] == scheme)]
                    vals.append(row[metric].values[0]
                                if len(row) > 0 else np.nan)
                offset = (k - len(schemes_order)/2 + 0.5) * width
                bars = ax.bar(x + offset, vals, width,
                              label=scheme, color=colors[scheme], alpha=0.85)
                for bar, v in zip(bars, vals):
                    if not np.isnan(v):
                        ax.text(bar.get_x() + bar.get_width()/2,
                                bar.get_height() + 0.005,
                                f'{v:.2f}', ha='center',
                                fontsize=6, color=colors[scheme])
 
            ax.set_xticks(x)
            ax.set_xticklabels(grids)
            ax.set_ylabel(title)
            ax.set_title(title, fontweight='bold')
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3, axis='y')
 
        plt.tight_layout()
        plt.show()
 



 # ──────────────────────────────────────────────────────────────────────────────
# Cross-experiment average summary
# Add this to the END of validation_schemes.py
# ──────────────────────────────────────────────────────────────────────────────

def _prepare_cross_test_df(cross_df,
                           grid_order=None,
                           scheme_order=None,
                           test_only=True):
    """
    Prepare cross-experiment results for summary analysis.

    Parameters
    ----------
    cross_df     : output of run_cross_experiment()
    grid_order   : optional display order for grid shapes
    scheme_order : optional display order for schemes
    test_only    : if True, keep only rows whose scheme contains 'test'

    Returns
    -------
    pd.DataFrame with cleaned scheme names.
    """
    import pandas as pd

    df = cross_df.copy()

    if test_only:
        df = df[df["scheme"].astype(str).str.contains("test")].copy()

    # A.1_test -> A.1, B_test -> B
    df["scheme_clean"] = (
        df["scheme"]
        .astype(str)
        .str.replace("_test", "", regex=False)
        .str.replace("_val", "", regex=False)
    )

    if grid_order is not None and "grid" in df.columns:
        df["grid"] = pd.Categorical(
            df["grid"],
            categories=grid_order,
            ordered=True
        )

    if scheme_order is not None:
        df["scheme_clean"] = pd.Categorical(
            df["scheme_clean"],
            categories=scheme_order,
            ordered=True
        )

    return df


def summarize_cross_averages(cross_df,
                              grid_order=None,
                              scheme_order=None,
                              metrics=None,
                              test_only=True):
    """
    Compute average performance summaries for cross-experiment results.

    Returns
    -------
    dict with:
        overall_by_scheme : average across both sites and all grids
        site_by_scheme    : average by site and scheme
        grid_by_scheme    : average by grid and scheme
    """
    import pandas as pd

    if metrics is None:
        metrics = [
            "cell_r2", "cell_rmse", "cell_mape",
            "field_r2", "field_rmse", "field_mape"
        ]

    df = _prepare_cross_test_df(
        cross_df,
        grid_order=grid_order,
        scheme_order=scheme_order,
        test_only=test_only
    )

    # Only keep metrics that exist
    metrics = [m for m in metrics if m in df.columns]

    overall_by_scheme = (
        df.groupby("scheme_clean", observed=True)[metrics]
        .mean()
        .round(4)
        .reset_index()
        .rename(columns={"scheme_clean": "scheme"})
    )

    site_by_scheme = (
        df.groupby(["site", "scheme_clean"], observed=True)[metrics]
        .mean()
        .round(4)
        .reset_index()
        .rename(columns={"scheme_clean": "scheme"})
    )

    grid_by_scheme = (
        df.groupby(["grid", "scheme_clean"], observed=True)[metrics]
        .mean()
        .round(4)
        .reset_index()
        .rename(columns={"scheme_clean": "scheme"})
    )

    return {
        "overall_by_scheme": overall_by_scheme,
        "site_by_scheme": site_by_scheme,
        "grid_by_scheme": grid_by_scheme,
        "prepared_df": df,
    }


def compare_scheme_b(cross_df,
                     grid_order=None,
                     scheme_order=None,
                     group_col=None,
                     test_only=True):
    """
    Compare Scheme B against A.1 and E.

    Parameters
    ----------
    group_col : None, 'site', or 'grid'
        None  -> overall average comparison
        site  -> compare within each site
        grid  -> compare within each grid shape

    Returns
    -------
    pd.DataFrame with B improvements over A.1 and E.
    """
    import pandas as pd

    summaries = summarize_cross_averages(
        cross_df,
        grid_order=grid_order,
        scheme_order=scheme_order,
        test_only=test_only
    )

    if group_col is None:
        avg_df = summaries["overall_by_scheme"].copy()
        groups = [("Overall", avg_df)]
        output_group_col = "group"

    elif group_col == "site":
        avg_df = summaries["site_by_scheme"].copy()
        groups = list(avg_df.groupby("site", observed=True))
        output_group_col = "site"

    elif group_col == "grid":
        avg_df = summaries["grid_by_scheme"].copy()
        groups = list(avg_df.groupby("grid", observed=True))
        output_group_col = "grid"

    else:
        raise ValueError("group_col must be None, 'site', or 'grid'.")

    records = []

    for group_name, sub in groups:
        sub = sub.copy()

        if "scheme" not in sub.columns:
            continue

        sub = sub.set_index("scheme")

        if "B" not in sub.index:
            continue

        b = sub.loc["B"]

        for other in ["A.1", "E"]:
            if other not in sub.index:
                continue

            o = sub.loc[other]

            record = {
                output_group_col: group_name,
                "compare": f"B vs {other}",

                # Higher is better
                "cell_r2_gain": round(b["cell_r2"] - o["cell_r2"], 4),
                "field_r2_gain": round(b["field_r2"] - o["field_r2"], 4),

                # Lower is better
                "cell_mape_reduction_abs": round(o["cell_mape"] - b["cell_mape"], 4),
                "field_mape_reduction_abs": round(o["field_mape"] - b["field_mape"], 4),
            }

            if o["cell_mape"] != 0:
                record["cell_mape_reduction_pct"] = round(
                    (o["cell_mape"] - b["cell_mape"]) / o["cell_mape"] * 100, 2
                )
            else:
                record["cell_mape_reduction_pct"] = None

            if o["field_mape"] != 0:
                record["field_mape_reduction_pct"] = round(
                    (o["field_mape"] - b["field_mape"]) / o["field_mape"] * 100, 2
                )
            else:
                record["field_mape_reduction_pct"] = None

            records.append(record)

    return pd.DataFrame(records)


def print_cross_average_summary(cross_df,
                                grid_order=None,
                                scheme_order=None,
                                test_only=True,
                                display_tables=True):
    """
    Print and optionally display average cross-experiment summaries.

    Returns
    -------
    dict containing all summary tables.
    """
    summaries = summarize_cross_averages(
        cross_df,
        grid_order=grid_order,
        scheme_order=scheme_order,
        test_only=test_only
    )

    overall_b_compare = compare_scheme_b(
        cross_df,
        grid_order=grid_order,
        scheme_order=scheme_order,
        group_col=None,
        test_only=test_only
    )

    site_b_compare = compare_scheme_b(
        cross_df,
        grid_order=grid_order,
        scheme_order=scheme_order,
        group_col="site",
        test_only=test_only
    )

    grid_b_compare = compare_scheme_b(
        cross_df,
        grid_order=grid_order,
        scheme_order=scheme_order,
        group_col="grid",
        test_only=test_only
    )

    results = {
        **summaries,
        "overall_b_compare": overall_b_compare,
        "site_b_compare": site_b_compare,
        "grid_b_compare": grid_b_compare,
    }

    print("\n" + "=" * 90)
    print("Average Test Performance by Scheme")
    print("Averaged across both sites and all grid shapes")
    print("=" * 90)

    if display_tables:
        try:
            from IPython.display import display
            display(results["overall_by_scheme"])
        except Exception:
            print(results["overall_by_scheme"].to_string(index=False))
    else:
        print(results["overall_by_scheme"].to_string(index=False))

    print("\n" + "=" * 90)
    print("Average Test Performance by Site and Scheme")
    print("Averaged across all grid shapes")
    print("=" * 90)

    if display_tables:
        try:
            from IPython.display import display
            display(results["site_by_scheme"])
        except Exception:
            print(results["site_by_scheme"].to_string(index=False))
    else:
        print(results["site_by_scheme"].to_string(index=False))

    print("\n" + "=" * 90)
    print("How much better is Scheme B than A.1 and E? — Overall")
    print("=" * 90)

    if display_tables:
        try:
            from IPython.display import display
            display(results["overall_b_compare"])
        except Exception:
            print(results["overall_b_compare"].to_string(index=False))
    else:
        print(results["overall_b_compare"].to_string(index=False))

    print("\n" + "=" * 90)
    print("How much better is Scheme B than A.1 and E? — By Site")
    print("=" * 90)

    if display_tables:
        try:
            from IPython.display import display
            display(results["site_b_compare"])
        except Exception:
            print(results["site_b_compare"].to_string(index=False))
    else:
        print(results["site_b_compare"].to_string(index=False))

    return results


def reload_validation_schemes():
    """
    Reload validation_schemes.py inside Colab / Jupyter.

    Usage:
        vs = reload_validation_schemes()
    """
    import importlib
    import validation_schemes as vs
    importlib.reload(vs)
    return vs