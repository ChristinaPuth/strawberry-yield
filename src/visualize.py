"""
visualize.py
------------
All visualisation functions for the strawberry yield prediction project.

Changes vs v1:
  - plot_yield_map_grid / plot_yield_map_utm : unchanged
  - plot_season_trend                        : unchanged
  - plot_multi_date                          : unchanged
  - plot_site_compare                        : unchanged
  - plot_distribution                        : unchanged
  - print_stats                              : unchanged

  NEW:
  - plot_ground_truth_map   : side-by-side actual vs predicted yield map
                              (single harvest date, grid index view)
  - plot_days_map           : spatial map of per-cell predicted optimal_days
  - plot_ground_truth_line  : line chart of actual vs predicted values
                              across all harvest dates (yield or days)
  - plot_prediction_scatter : scatter plot of y_true vs y_pred
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

CMAP       = 'YlOrRd'
COLOR_BAR  = '#E07B39'
COLOR_LINE = '#5B8DB8'
COLOR_GT   = '#2d5a3d'   # ground truth green

UTM_BOUNDS = {
    'SantaMaria': dict(e=(729000, 732000), n=(3864000, 3865000)),
    'Salinas':    dict(e=(630500, 631000), n=(4054300, 4054700)),
}


# ── Internal helpers ──────────────────────────────────────────────────────────

def _get_grid(df_day, value_col="weight_kg"):
    x_vals = sorted(df_day['field_x'].unique())
    y_vals = sorted(df_day['field_y'].unique())
    xi = {v: i for i, v in enumerate(x_vals)}
    yi = {v: i for i, v in enumerate(y_vals)}
    grid = np.zeros((len(y_vals), len(x_vals)))
    for _, row in df_day.iterrows():
        grid[yi[row['field_y']], xi[row['field_x']]] = row[value_col]
    return grid

def _vmax(series, q=0.99):
    v = series.quantile(q)
    return v if v > 0 else series.max()

def _filter_utm(df, site):
    b = UTM_BOUNDS[site]
    return df[df['easting'].between(*b['e']) & df['northing'].between(*b['n'])]

def _metrics_str(y_true, y_pred):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))
    return f"RMSE={rmse:.3f}   MAE={mae:.3f}   R²={r2:.3f}", rmse, mae, r2


# ── Existing functions (unchanged) ────────────────────────────────────────────

def plot_yield_map_grid(df, site, harvest_date, ax=None, title=None):
    d = df[df['harvest_date'] == harvest_date].copy()
    if d.empty:
        print(f"No data for {site} on {harvest_date}")
        return
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(10, 6))
    grid = _get_grid(d)
    vm = _vmax(d['weight_kg'])
    im = ax.imshow(grid, cmap=CMAP, aspect='auto', origin='upper', vmin=0, vmax=vm)
    plt.colorbar(im, ax=ax, label='Yield (kg)', shrink=0.85)
    ax.set_title(title or f'{site}  —  {str(harvest_date)[:10]}',
                 fontsize=11, fontweight='bold')
    ax.set_xlabel('field_x index')
    ax.set_ylabel('field_y index')
    ax.text(0.02, 0.97, f'Total: {d["weight_kg"].sum():,.0f} kg',
            transform=ax.transAxes, fontsize=9, va='top', color='white',
            bbox=dict(boxstyle='round,pad=0.3', fc='#333', alpha=0.6))
    if standalone:
        plt.tight_layout()
        plt.show()
    return ax


def plot_yield_map_utm(df, site, harvest_date, ax=None, title=None, s=3):
    d = _filter_utm(df[df['harvest_date'] == harvest_date].copy(), site)
    if d.empty:
        print(f"No valid UTM data for {site} on {harvest_date}")
        return
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(10, 6))
    vm = _vmax(d['weight_kg'])
    sc = ax.scatter(d['easting'], d['northing'], c=d['weight_kg'],
                    cmap=CMAP, s=s, vmin=0, vmax=vm)
    plt.colorbar(sc, ax=ax, label='Yield (kg)', shrink=0.85)
    ax.set_title(title or f'{site}  —  {str(harvest_date)[:10]}  (UTM)',
                 fontsize=11, fontweight='bold')
    ax.set_xlabel('Easting (m)')
    ax.set_ylabel('Northing (m)')
    ax.ticklabel_format(style='plain', axis='both')
    ax.text(0.02, 0.97, f'Total: {d["weight_kg"].sum():,.0f} kg',
            transform=ax.transAxes, fontsize=9, va='top', color='white',
            bbox=dict(boxstyle='round,pad=0.3', fc='#333', alpha=0.6))
    if standalone:
        plt.tight_layout()
        plt.show()
    return ax


def plot_season_trend(df, site, summary_df=None):
    if summary_df is None:
        import data_pipeline
        summary_df = data_pipeline.summary(df)
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True,
                              gridspec_kw={'height_ratios': [2, 1]})
    fig.suptitle(f'{site}  —  Season yield trend', fontsize=14, fontweight='bold')
    axes[0].bar(summary_df['harvest_date'], summary_df['total_kg'],
                color=COLOR_BAR, alpha=0.85, width=1.8, zorder=2)
    axes[0].set_ylabel('Total yield (kg)')
    axes[0].grid(axis='y', alpha=0.3, zorder=1)
    peak = summary_df.loc[summary_df['total_kg'].idxmax()]
    axes[0].annotate(f"Peak\n{peak['total_kg']:,.0f} kg",
                     xy=(peak['harvest_date'], peak['total_kg']),
                     xytext=(0, 8), textcoords='offset points',
                     ha='center', fontsize=8, color='#8B2500')
    axes[1].plot(summary_df['harvest_date'], summary_df['pct_zero'],
                 color=COLOR_LINE, marker='o', linewidth=1.8, markersize=4)
    axes[1].fill_between(summary_df['harvest_date'], summary_df['pct_zero'],
                          alpha=0.15, color=COLOR_LINE)
    axes[1].set_ylabel('% zero cells')
    axes[1].set_xlabel('Harvest date')
    axes[1].set_ylim(0, 100)
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    return fig


def plot_multi_date(df, site, n_dates=4, use_utm=False):
    import data_pipeline
    s = data_pipeline.summary(df)
    indices = np.linspace(0, len(s) - 1, n_dates, dtype=int)
    labels  = ['Early', 'Mid-early', 'Mid-late', 'Late'][:n_dates]
    fig, axes = plt.subplots(1, n_dates, figsize=(5 * n_dates, 5.5))
    if n_dates == 1:
        axes = [axes]
    fig.suptitle(f'{site}  —  Seasonal progression',
                 fontsize=14, fontweight='bold', y=1.01)
    plot_fn = plot_yield_map_utm if use_utm else plot_yield_map_grid
    for i, (ax, label) in enumerate(zip(axes, labels)):
        row = s.iloc[indices[i]]
        plot_fn(df, site, row['harvest_date'], ax=ax,
                title=f'{label}\n{str(row["harvest_date"])[:10]}\n'
                      f'Total: {row["total_kg"]:,.0f} kg')
    plt.tight_layout()
    plt.show()
    return fig


def plot_site_compare(df_sm, df_sal, use_utm=False):
    import data_pipeline
    peak_sm  = data_pipeline.summary(df_sm).sort_values('total_kg').iloc[-1]['harvest_date']
    peak_sal = data_pipeline.summary(df_sal).sort_values('total_kg').iloc[-1]['harvest_date']
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Peak harvest day comparison', fontsize=14, fontweight='bold')
    plot_fn = plot_yield_map_utm if use_utm else plot_yield_map_grid
    plot_fn(df_sm,  'SantaMaria', peak_sm,  ax=axes[0])
    plot_fn(df_sal, 'Salinas',    peak_sal, ax=axes[1])
    plt.tight_layout()
    plt.show()
    return fig


def plot_distribution(df, site, harvest_date=None):
    if harvest_date is not None:
        d   = df[df['harvest_date'] == harvest_date]
        sub = str(harvest_date)[:10]
    else:
        d   = df
        sub = 'All harvest dates'
    zeros   = d[d['weight_kg'] == 0]
    nonzero = d[d['weight_kg'] > 0]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'{site}  —  Yield distribution  ({sub})',
                 fontsize=13, fontweight='bold')
    axes[0].hist(d['weight_kg'], bins=60, color=COLOR_BAR, alpha=0.8,
                 edgecolor='white', linewidth=0.3)
    axes[0].set_xlabel('Yield per cell (kg)')
    axes[0].set_ylabel('Count')
    axes[0].set_title(f'All cells  (n={len(d):,})')
    axes[0].text(0.62, 0.93, f'Zero cells: {len(zeros)/len(d)*100:.1f}%',
                 transform=axes[0].transAxes, fontsize=10,
                 bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.8))
    axes[1].hist(nonzero['weight_kg'], bins=60, color='#B85030', alpha=0.85,
                 edgecolor='white', linewidth=0.3)
    axes[1].set_xlabel('Yield per cell (kg)')
    axes[1].set_ylabel('Count (log scale)')
    axes[1].set_yscale('log')
    axes[1].set_title(f'Non-zero cells  (n={len(nonzero):,})')
    axes[1].axvline(nonzero['weight_kg'].median(), color='navy', linestyle='--',
                    linewidth=1.5,
                    label=f'Median: {nonzero["weight_kg"].median():.3f} kg')
    axes[1].axvline(nonzero['weight_kg'].mean(), color='green', linestyle='--',
                    linewidth=1.5,
                    label=f'Mean: {nonzero["weight_kg"].mean():.3f} kg')
    axes[1].legend(fontsize=9)
    plt.tight_layout()
    plt.show()
    return fig


def print_stats(df, site):
    import data_pipeline
    s = data_pipeline.summary(df)
    print(f"\n{'='*55}")
    print(f"  {site}  —  Season Statistics")
    print(f"{'='*55}")
    print(f"  Harvest dates      : {s['harvest_date'].min().date()} -> {s['harvest_date'].max().date()}")
    print(f"  Total harvests     : {len(s)}")
    print(f"  Grid cells/harvest : {s['n_cells'].iloc[0]:,}")
    print(f"  Season total yield : {s['total_kg'].sum():,.0f} kg")
    peak = s.loc[s['total_kg'].idxmax()]
    low  = s.loc[s['total_kg'].idxmin()]
    print(f"  Peak harvest       : {peak['harvest_date'].date()} ({peak['total_kg']:,.0f} kg)")
    print(f"  Lowest harvest     : {low['harvest_date'].date()} ({low['total_kg']:,.0f} kg)")
    print(f"  Avg % zero cells   : {s['pct_zero'].mean():.1f}%")
    print(f"  Max cell yield     : {df['weight_kg'].max():.3f} kg")
    nz = df[df['weight_kg'] > 0]['weight_kg']
    print(f"  Mean cell yield    : {nz.mean():.3f} kg")
    print(f"{'='*55}\n")


# ── NEW: Ground truth vs predicted map ───────────────────────────────────────

def plot_ground_truth_map(df_feat: pd.DataFrame,
                           model_results: dict,
                           harvest_date,
                           site: str,
                           figsize=(18, 6)):
    """
    Three-panel map for one harvest date:
      Panel 1: Ground truth (actual yield)
      Panel 2: Predicted yield
      Panel 3: Error map (pred - actual)

    Parameters
    ----------
    df_feat       : feature DataFrame (output of fe.build_features)
    model_results : dict with 'yield' key (output of m.run_model_comparison)
    harvest_date  : pd.Timestamp or string
    site          : 'SantaMaria' or 'Salinas'
    """
    harvest_date = pd.Timestamp(harvest_date)
    d = df_feat[df_feat['harvest_date'] == harvest_date].copy()
    if d.empty:
        print(f"No data for {site} on {harvest_date.date()}")
        return

    # Get predictions
    best_row = model_results['yield'].iloc[0]
    model    = best_row['_model_obj']
    features = best_row['features']
    log_t    = best_row.get('log_target', False)

    avail = [f for f in features if f in d.columns]
    X     = d[avail].values.astype(np.float32)
    y_pred = model.predict(X)
    if log_t:
        y_pred = np.expm1(y_pred)
    y_pred = np.clip(y_pred, 0, None)
    y_true = d['weight_kg'].values

    d = d.copy()
    d['y_pred'] = y_pred
    d['error']  = y_pred - y_true

    metrics_str, rmse, mae, r2 = _metrics_str(y_true, y_pred)

    # Build grids
    x_vals = sorted(d['field_x'].unique())
    y_vals = sorted(d['field_y'].unique())
    x2i = {v: i for i, v in enumerate(x_vals)}
    y2i = {v: i for i, v in enumerate(y_vals)}

    grid_true = np.zeros((len(y_vals), len(x_vals)))
    grid_pred = np.zeros((len(y_vals), len(x_vals)))
    grid_err  = np.full((len(y_vals), len(x_vals)), np.nan)

    for _, row in d.iterrows():
        xi = x2i[row['field_x']]; yi = y2i[row['field_y']]
        grid_true[yi, xi] = row['weight_kg']
        grid_pred[yi, xi] = row['y_pred']
        grid_err [yi, xi] = row['error']

    vmax = float(np.nanquantile(
        np.concatenate([grid_true.ravel(), grid_pred.ravel()]), 0.99))
    vm_err = float(np.nanquantile(
        np.abs(grid_err[~np.isnan(grid_err)]), 0.95))

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle(
        f"{site}  —  {harvest_date.date()}  |  {metrics_str}",
        fontsize=12, fontweight='bold'
    )

    panels = [
        (axes[0], grid_true, 'Ground Truth (actual yield)', CMAP,    0,      vmax,   'kg'),
        (axes[1], grid_pred, 'Predicted yield',             CMAP,    0,      vmax,   'kg'),
        (axes[2], grid_err,  'Error  (pred − actual)',      'RdBu_r',-vm_err,vm_err, 'kg'),
    ]
    for ax, grid, title, cmap, vmin, vmax_p, label in panels:
        im = ax.imshow(grid, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax_p)
        plt.colorbar(im, ax=ax, label=label, shrink=0.85)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel('field_x index')
        ax.set_ylabel('field_y index')

    # Annotate totals on first two panels
    axes[0].text(0.02, 0.97, f'Total: {y_true.sum():,.0f} kg',
                 transform=axes[0].transAxes, fontsize=9, va='top', color='white',
                 bbox=dict(boxstyle='round,pad=0.3', fc='#333', alpha=0.65))
    axes[1].text(0.02, 0.97, f'Total: {y_pred.sum():,.0f} kg',
                 transform=axes[1].transAxes, fontsize=9, va='top', color='white',
                 bbox=dict(boxstyle='round,pad=0.3', fc='#333', alpha=0.65))

    plt.tight_layout()
    plt.show()
    return fig


# ── NEW: Days map ─────────────────────────────────────────────────────────────

def plot_days_map(inference_df: pd.DataFrame,
                  pred_days: np.ndarray,
                  site: str,
                  optimal_days: int = None,
                  figsize=(10, 6)):
    """
    Spatial map of per-cell predicted optimal_days.
    Green = short interval (harvest soon), Red = long interval (wait more).

    Parameters
    ----------
    inference_df  : DataFrame with field_x, field_y columns
                    (output of ha._build_inference_row)
    pred_days     : np.ndarray of per-cell predicted days
    site          : 'SantaMaria' or 'Salinas'
    optimal_days  : field-level recommendation (median), shown in title
    """
    df = inference_df.copy()
    df['pred_days'] = np.round(pred_days).astype(int)

    x_vals = sorted(df['field_x'].unique())
    y_vals = sorted(df['field_y'].unique())
    x2i = {v: i for i, v in enumerate(x_vals)}
    y2i = {v: i for i, v in enumerate(y_vals)}
    grid = np.zeros((len(y_vals), len(x_vals)))
    for (_, row), p in zip(df.iterrows(), pred_days):
        grid[y2i[row['field_y']], x2i[row['field_x']]] = p

    d_min = max(1, int(np.percentile(pred_days, 2)))
    d_max = int(np.percentile(pred_days, 98))

    fig, axes = plt.subplots(1, 2, figsize=figsize,
                              gridspec_kw={'width_ratios': [2.5, 1]})

    # Map
    im = axes[0].imshow(grid, cmap='RdYlGn_r', aspect='auto',
                         vmin=d_min, vmax=d_max)
    plt.colorbar(im, ax=axes[0], label='Predicted days', shrink=0.85)
    title = f'{site}  —  Days Map'
    if optimal_days is not None:
        title += f'  |  Field recommendation: {optimal_days} days'
    axes[0].set_title(title, fontsize=11, fontweight='bold')
    axes[0].set_xlabel('field_x index')
    axes[0].set_ylabel('field_y index')

    # Distribution bar
    dist = pd.Series(np.round(pred_days).astype(int)).value_counts().sort_index()
    colours = ['#2d6a3f' if d == optimal_days else '#CBD5E1'
               for d in dist.index]
    axes[1].barh([str(d) for d in dist.index], dist.values,
                  color=colours, edgecolor='white')
    axes[1].set_xlabel('Number of cells')
    axes[1].set_title('Cell distribution', fontsize=10)
    axes[1].grid(axis='x', alpha=0.3)
    if optimal_days is not None:
        axes[1].axhline(
            [str(d) for d in dist.index].index(str(optimal_days)),
            color='#2d6a3f', linewidth=2, linestyle='--'
        )

    plt.tight_layout()
    plt.show()
    return fig


# ── NEW: Ground truth vs predicted line chart ─────────────────────────────────

def plot_ground_truth_line(df_feat: pd.DataFrame,
                            model_results: dict,
                            site: str,
                            target: str = 'yield',
                            figsize=(13, 5)):
    """
    Line chart: ground truth vs predicted across all harvest dates.

    target = 'yield' → total field yield per harvest date
    target = 'days'  → actual vs predicted optimal_days per harvest date

    Shows: two lines + shaded gap + RMSE/MAE/R² annotation.
    """
    t_col    = 'weight_kg' if target == 'yield' else 'optimal_days'
    best_row = model_results[target].iloc[0]
    model    = best_row['_model_obj']
    features = best_row['features']
    log_t    = best_row.get('log_target', False) if target == 'yield' else False

    avail  = [f for f in features if f in df_feat.columns]
    X      = df_feat[avail].values.astype(np.float32)
    y_true = df_feat[t_col].values.astype(np.float32)

    y_pred = model.predict(X)
    if log_t:
        y_pred = np.expm1(y_pred)
    if target == 'days':
        y_pred = np.clip(y_pred, 1, 14)
    else:
        y_pred = np.clip(y_pred, 0, None)

    df_plot = df_feat[['harvest_date']].copy()
    df_plot['y_true'] = y_true
    df_plot['y_pred'] = y_pred

    if target == 'yield':
        agg = df_plot.groupby('harvest_date').agg(
            true_val=('y_true', 'sum'),
            pred_val=('y_pred', 'sum'),
        ).reset_index().sort_values('harvest_date')
        ylabel = 'Total field yield (kg)'
        title  = f'{site}  —  Ground Truth vs Predicted Yield per Harvest'
    else:
        agg = df_plot.groupby('harvest_date').agg(
            true_val=('y_true', 'first'),
            pred_val=('y_pred', 'median'),
        ).reset_index().sort_values('harvest_date')
        ylabel = 'Days since last harvest'
        title  = f'{site}  —  Ground Truth vs Predicted Optimal Days'

    y_t = agg['true_val'].values
    y_p = agg['pred_val'].values
    dates = agg['harvest_date']

    metrics_str, rmse, mae, r2 = _metrics_str(y_t, y_p)

    fig, ax = plt.subplots(figsize=figsize)

    # Ground truth line
    ax.plot(dates, y_t, 'o-', color=COLOR_GT, linewidth=2.2,
            markersize=7, label='Ground Truth', zorder=4)

    # Predicted line
    ax.plot(dates, y_p, 's--', color=COLOR_BAR, linewidth=2.2,
            markersize=7, label='Predicted', zorder=4)

    # Shaded gap between the two lines
    ax.fill_between(dates, y_t, y_p, alpha=0.13, color=COLOR_BAR)

    ax.set_xlabel('Harvest date', fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    plt.xticks(rotation=45)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    # Metrics box
    ax.text(0.02, 0.97, metrics_str,
            transform=ax.transAxes, fontsize=10, va='top',
            bbox=dict(boxstyle='round,pad=0.4', fc='white', alpha=0.88))

    plt.tight_layout()
    plt.show()
    return fig


# ── NEW: Scatter plot (y_true vs y_pred) ─────────────────────────────────────

def plot_prediction_scatter(df_feat: pd.DataFrame,
                             model_results: dict,
                             site: str,
                             target: str = 'yield',
                             figsize=(7, 6)):
    """
    Scatter plot of y_true vs y_pred (cell level for yield, date level for days).
    Diagonal = perfect prediction line.
    """
    t_col    = 'weight_kg' if target == 'yield' else 'optimal_days'
    best_row = model_results[target].iloc[0]
    model    = best_row['_model_obj']
    features = best_row['features']
    log_t    = best_row.get('log_target', False) if target == 'yield' else False

    avail  = [f for f in features if f in df_feat.columns]
    X      = df_feat[avail].values.astype(np.float32)
    y_true = df_feat[t_col].values.astype(np.float32)

    y_pred = model.predict(X)
    if log_t:
        y_pred = np.expm1(y_pred)
    if target == 'days':
        y_pred = np.clip(y_pred, 1, 14)
    else:
        y_pred = np.clip(y_pred, 0, None)

    metrics_str, rmse, mae, r2 = _metrics_str(y_true, y_pred)

    fig, ax = plt.subplots(figsize=figsize)

    # Scatter (subsample for yield to avoid overplotting)
    if target == 'yield' and len(y_true) > 20000:
        idx = np.random.choice(len(y_true), 20000, replace=False)
        yt, yp = y_true[idx], y_pred[idx]
    else:
        yt, yp = y_true, y_pred

    ax.scatter(yt, yp, alpha=0.15, s=4, color=COLOR_BAR, rasterized=True)

    # Perfect prediction diagonal
    vmin = min(yt.min(), yp.min())
    vmax = max(yt.max(), yp.max())
    ax.plot([vmin, vmax], [vmin, vmax], 'k--', linewidth=1.5,
            label='Perfect prediction', zorder=5)

    ax.set_xlabel('Actual', fontsize=11)
    ax.set_ylabel('Predicted', fontsize=11)
    target_label = 'weight_kg (kg)' if target == 'yield' else 'optimal_days'
    ax.set_title(f'{site}  —  Scatter: {target_label}\n{best_row["model"]}',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.text(0.04, 0.96, metrics_str,
            transform=ax.transAxes, fontsize=9, va='top',
            bbox=dict(boxstyle='round,pad=0.4', fc='white', alpha=0.88))

    plt.tight_layout()
    plt.show()
    return fig



# ── NEW: Ablation table heatmap ───────────────────────────────────────────────
 
def plot_ablation_table(ablation_sm: pd.DataFrame,
                         ablation_sal: pd.DataFrame,
                         figsize=(14, 6)):
    """
    Side-by-side heatmap of ablation results for both sites.
    Baselines shown separately with dashed separator.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle("Ablation Study — Val R²  (Stage 1: weight_kg prediction)",
                 fontsize=13, fontweight='bold')
 
    for ax, ablation, site in [(axes[0], ablation_sm, "SantaMaria"),
                                (axes[1], ablation_sal, "Salinas")]:
        # Separate baselines and ML configs
        baselines = ablation[ablation["is_baseline"]==True].copy()
        ml_cfgs   = ablation[ablation["is_baseline"]==False].copy()
        ml_cfgs   = ml_cfgs.sort_values("config")
 
        configs  = list(baselines["config"]) + ["—"] + list(ml_cfgs["config"])
        r2_vals  = (list(baselines["val_r2"]) + [None] + list(ml_cfgs["val_r2"]))
        rmse_vals= (list(baselines["val_rmse"]) + [None] + list(ml_cfgs["val_rmse"]))
 
        y_pos = np.arange(len(configs))
        colours = []
        for i, cfg in enumerate(configs):
            if cfg == "—":
                colours.append("white")
            elif cfg.startswith("B"):
                colours.append("#CBD5E1")   # baseline: grey
            else:
                r2 = r2_vals[i]
                colours.append("#2d6a3f" if r2 and r2 > 0.5 else
                                "#E07B39" if r2 and r2 > 0.3 else
                                "#c0392b" if r2 is not None else "white")
 
        valid_r2 = [v for v in r2_vals if v is not None]
        bars = ax.barh(y_pos, [v if v is not None else 0 for v in r2_vals],
                       color=colours, edgecolor='white', linewidth=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(configs, fontsize=10)
        ax.set_xlabel("Val R²", fontsize=10)
        ax.set_title(site, fontsize=11, fontweight='bold')
        ax.set_xlim(-0.5, 1.0)
        ax.axvline(0, color='black', linewidth=0.8)
 
        for bar, r2, rmse in zip(bars, r2_vals, rmse_vals):
            if r2 is not None:
                ax.text(max(bar.get_width(), 0) + 0.02, bar.get_y()+bar.get_height()/2,
                        f"R²={r2:.3f}  RMSE={rmse:.3f}",
                        va='center', fontsize=8)
 
        # Mark best ML
        if len(valid_r2) > 0:
            best_r2 = max(v for v in r2_vals if v is not None)
            best_idx = r2_vals.index(best_r2)
            ax.get_yticklabels()[best_idx].set_fontweight('bold')
            ax.get_yticklabels()[best_idx].set_color('#2d6a3f')
 
        ax.grid(axis='x', alpha=0.3)
        ax.invert_yaxis()
 
    plt.tight_layout(); plt.show()
    return fig
 
 
# ── NEW: Decision Quality summary ─────────────────────────────────────────────
 
def plot_dq_summary(dq_sm: pd.DataFrame,
                     dq_sal: pd.DataFrame,
                     figsize=(14, 5)):
    """
    Side-by-side Decision Quality comparison for both sites.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle("Decision Quality: Model Recommendation vs Farmer's Decision",
                 fontsize=13, fontweight='bold')
 
    for ax, dq_df, site in [(axes[0], dq_sm, "SantaMaria"),
                             (axes[1], dq_sal, "Salinas")]:
        valid = dq_df.dropna(subset=["DQ_kg"])
        if valid.empty:
            ax.text(0.5, 0.5, "No DQ data", ha='center', va='center',
                    transform=ax.transAxes, fontsize=12)
            ax.set_title(site); continue
 
        colours = ["#2d6a3f" if v >= 0 else "#c0392b" for v in valid["DQ_kg"]]
        ax.bar([str(d.date()) for d in valid["test_harvest_date"]],
               valid["DQ_kg"], color=colours, edgecolor='white', linewidth=0.5)
        ax.axhline(0, color='black', linewidth=1)
        mean_dq = valid["DQ_kg"].mean()
        ax.axhline(mean_dq, color='#E07B39', linewidth=2, linestyle='--',
                   label=f"Mean DQ = {mean_dq:+,.0f} kg")
 
        for i, (_, row) in enumerate(valid.iterrows()):
            ax.text(i, row["DQ_kg"] + (valid["DQ_kg"].abs().max()*0.03 *
                    np.sign(row["DQ_kg"])),
                    f"{row['DQ_kg']:+,.0f}\n({row['DQ_pct']:+.1f}%)",
                    ha='center',
                    va='bottom' if row['DQ_kg'] >= 0 else 'top',
                    fontsize=8)
 
        ax.set_xlabel("Test harvest date")
        ax.set_ylabel("DQ (kg)  [positive = model better]")
        ax.set_title(f"{site}\nMean DQ = {mean_dq:+,.0f} kg  "
                     f"({'✅ model better' if mean_dq>0 else '❌ farmer better'})",
                     fontsize=11, fontweight='bold')
        ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')
 
    plt.tight_layout(); plt.show()
    return fig