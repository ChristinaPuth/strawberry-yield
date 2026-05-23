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