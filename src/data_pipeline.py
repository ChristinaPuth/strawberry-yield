"""
data_pipeline.py
----------------
Loads all harvest date folders for SantaMaria and Salinas,
parses dates, assigns column names, and returns a clean DataFrame.

Usage in Colab:
    import importlib, data_pipeline
    importlib.reload(data_pipeline)
    df = data_pipeline.load_site("SantaMaria")
"""

import os
import re
import pandas as pd
from datetime import datetime


# ── Site configuration ─────────────────────────────────────────────────────

SITE_CONFIG = {
    "SantaMaria": {
        "folder_name": "Santa Maria - ranch 31_corrected_YieldMap",
        "accumulative_csv": "SantaMaria_accumulative_yield_2024.csv",
        "grid_size_m": 1.6260306232976949,
        "field_corners_east": [731046.4683576762, 730843.8802561829,
                                731045.3242675813, 730843.1998210086],
        "field_corners_north": [3864744.42259181, 3864744.42259181,
                                 3864934.306089565, 3864934.306089565],
    },
    "Salinas": {
        "folder_name": "Salinas Harvest 2024_corrected_YieldMap",
        "accumulative_csv": "Salinas_accumulative_yield_2024.csv",
        "grid_size_m": 1.2189278136,
        "field_corners_east": [630792.465489, 630660.1149771864,
                                630730.51086, 630597.7690841864],
        "field_corners_north": [4054479.4539071457, 4054635.1658983775,
                                 4054427.582404032, 4054583.074041546],
    },
}

# Column names for the per-day yield CSV (no header in file)
YIELD_COLS = ["index", "field_x", "field_y", "weight_kg", "easting", "northing"]

# Column names for the accumulative CSV (no header in file)
ACCUM_COLS = ["field_x", "field_y", "weight_kg", "easting", "northing"]


# ── Date parsing ────────────────────────────────────────────────────────────

# def parse_folder_date(folder_name: str) -> datetime | None:
def parse_folder_date(folder_name: str):
    """
    Parse a folder name like '6-15-24' or '6-15-2024' into a datetime.
    Returns None if the folder name doesn't look like a date.
    """
    # Match patterns: M-D-YY or M-D-YYYY
    match = re.fullmatch(r"(\d{1,2})-(\d{1,2})-(\d{2,4})", folder_name.strip())
    if not match:
        return None
    month, day, year = int(match.group(1)), int(match.group(2)), int(match.group(3))
    if year < 100:
        year += 2000
    try:
        return datetime(year, month, day)
    except ValueError:
        return None


# ── Single date loader ───────────────────────────────────────────────────────
# 读取某一天的 yield CSV
# def load_one_date(date_folder_path: str, harvest_date: datetime, site: str) -> pd.DataFrame | None:
def load_one_date(date_folder_path: str, harvest_date: datetime, site: str):
    """
    Load the yield CSV from one dated folder.
    Returns a DataFrame with columns:
        site, harvest_date, field_x, field_y, weight_kg, easting, northing
    Returns None if no yield CSV found.
    """
    # Find the yield CSV (named like '6-15-24_yield.csv')
    yield_file = None
    for fname in os.listdir(date_folder_path):
        if fname.endswith("_yield.csv") and "trays" not in fname.lower():
            yield_file = os.path.join(date_folder_path, fname)
            break

    if yield_file is None:
        print(f"  [SKIP] No yield CSV in {date_folder_path}")
        return None

    try:
        df = pd.read_csv(yield_file, header=None)
    except Exception as e:
        print(f"  [ERROR] Could not read {yield_file}: {e}")
        return None

    # Assign column names based on number of columns
    if df.shape[1] == 6:
        df.columns = YIELD_COLS
        df = df.drop(columns=["index"])
    elif df.shape[1] == 5:
        # Some files may already have dropped the index column
        df.columns = ACCUM_COLS
    else:
        print(f"  [WARN] Unexpected column count ({df.shape[1]}) in {yield_file}")
        return None

    df["site"] = site
    df["harvest_date"] = harvest_date

    # Keep only the columns we need
    df = df[["site", "harvest_date", "field_x", "field_y",
             "weight_kg", "easting", "northing"]].copy()

    # Convert to correct types
    df["field_x"] = pd.to_numeric(df["field_x"], errors="coerce")
    df["field_y"] = pd.to_numeric(df["field_y"], errors="coerce")
    df["weight_kg"] = pd.to_numeric(df["weight_kg"], errors="coerce")
    df["easting"] = pd.to_numeric(df["easting"], errors="coerce")
    df["northing"] = pd.to_numeric(df["northing"], errors="coerce")

    # Drop rows with missing key fields
    df = df.dropna(subset=["field_x", "field_y", "weight_kg"])

    return df


# ── Site loader ──────────────────────────────────────────────────────────────
# 读取一个地块的所有日期
def load_site(site: str, base_path: str) -> pd.DataFrame:
    """
    Load all harvest dates for a given site.

    Parameters
    ----------
    site      : "SantaMaria" or "Salinas"
    base_path : path to the YieldMapHarvest_Original Data folder
                e.g. '/content/drive/MyDrive/Spatio-Temporal Modeling/Codes/YieldMapHarvest_Original Data'

    Returns
    -------
    DataFrame with all harvest dates combined, sorted by harvest_date.
    Columns: site, harvest_date, field_x, field_y, weight_kg, easting, northing
    """
    config = SITE_CONFIG[site]
    site_path = os.path.join(base_path, config["folder_name"])

    if not os.path.exists(site_path):
        raise FileNotFoundError(f"Site folder not found: {site_path}")

    all_dfs = []
    skipped = []

    entries = sorted(os.listdir(site_path))
    for entry in entries:
        full_path = os.path.join(site_path, entry)
        if not os.path.isdir(full_path):
            continue  # skip files like ReadMe.txt and accumulative CSV

        harvest_date = parse_folder_date(entry)
        if harvest_date is None:
            continue  # not a date folder

        df = load_one_date(full_path, harvest_date, site)
        if df is not None and len(df) > 0:
            all_dfs.append(df)
            print(f"  [OK] {entry:12s} → {len(df):6,} rows")
        else:
            skipped.append(entry)

    if not all_dfs:
        raise ValueError(f"No data loaded for {site}. Check base_path.")

    combined = pd.concat(all_dfs, ignore_index=True)
    combined = combined.sort_values(["harvest_date", "field_x", "field_y"]).reset_index(drop=True)

    # Add harvest index (1, 2, 3, ...) in chronological order
    date_order = {d: i + 1 for i, d in enumerate(sorted(combined["harvest_date"].unique()))}
    combined["harvest_idx"] = combined["harvest_date"].map(date_order)

    print(f"\n{'='*50}")
    print(f"  Site         : {site}")
    print(f"  Dates loaded : {combined['harvest_date'].nunique()}")
    print(f"  Dates skipped: {len(skipped)}")
    print(f"  Total rows   : {len(combined):,}")
    print(f"  Date range   : {combined['harvest_date'].min().date()} → {combined['harvest_date'].max().date()}")
    print(f"  Grid cells   : ~{combined.groupby('harvest_date').size().median():,.0f} per harvest")
    print(f"{'='*50}\n")

    return combined


# ── Load both sites ──────────────────────────────────────────────────────────

def load_all(base_path: str) -> dict[str, pd.DataFrame]:
    """
    Load both SantaMaria and Salinas.

    Returns
    -------
    dict with keys 'SantaMaria' and 'Salinas'
    """
    print("Loading SantaMaria...")
    sm = load_site("SantaMaria", base_path)

    print("Loading Salinas...")
    sal = load_site("Salinas", base_path)

    return {"SantaMaria": sm, "Salinas": sal}


# ── Load accumulative CSV ────────────────────────────────────────────────────

def load_accumulative(site: str, base_path: str) -> pd.DataFrame:
    """
    Load the season-long accumulative yield CSV.
    Columns: field_x, field_y, weight_kg, easting, northing
    """
    config = SITE_CONFIG[site]
    csv_path = os.path.join(base_path, config["folder_name"], config["accumulative_csv"])

    df = pd.read_csv(csv_path, header=None)
    df.columns = ACCUM_COLS
    df["site"] = site

    for col in ["field_x", "field_y", "weight_kg", "easting", "northing"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["field_x", "field_y", "weight_kg"])
    print(f"Accumulative {site}: {len(df):,} grid cells, "
          f"total {df['weight_kg'].sum():.1f} kg")
    return df


# ── Quick summary ────────────────────────────────────────────────────────────

def summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a per-harvest-date summary table.
    Columns: harvest_date, harvest_idx, n_cells, total_kg, mean_kg, max_kg
    """
    return (
        df.groupby(["harvest_date", "harvest_idx"])
        .agg(
            n_cells=("weight_kg", "count"),
            total_kg=("weight_kg", "sum"),
            mean_kg=("weight_kg", "mean"),
            median_kg=("weight_kg", "median"),
            max_kg=("weight_kg", "max"),
            pct_zero=("weight_kg", lambda x: (x == 0).mean() * 100),
        )
        .round(3)
        .reset_index()
        .sort_values("harvest_date")
    )

#  ── Weather cache ────────────────────────────────────────────────────────────
 
def load_weather_cached(site: str, outputs_path: str):
    """
    Load weather data from cache, or fetch from Open-Meteo and save.
 
    Parameters
    ----------
    site         : 'SantaMaria' or 'Salinas'
    outputs_path : path to outputs/processed_data folder
 
    Returns
    -------
    pd.DataFrame with daily weather indexed by date
    """
    import feature_engineering as fe
 
    path = os.path.join(outputs_path, f'weather_{site}.csv')
    if os.path.exists(path):
        print(f'Loading cached weather for {site}...')
        return pd.read_csv(path, index_col=0, parse_dates=True)
    print(f'Fetching weather for {site} from Open-Meteo...')
    w = fe.fetch_weather(site)
    w.to_csv(path)
    print(f'  Saved to {path}')
    return w
 
 
# ── Feature cache ─────────────────────────────────────────────────────────────
 
def load_features_cached(site: str, df_raw, weather,
                          outputs_path: str,
                          force_rebuild: bool = False,
                          backfill_lags: bool = True,
                          drop_anomaly: bool = True):
    """
    Load pre-built features from cache, or build and save them.
 
    Parameters
    ----------
    site          : 'SantaMaria' or 'Salinas'
    df_raw        : raw harvest DataFrame from load_site()
    weather       : weather DataFrame from load_weather_cached()
    outputs_path  : path to outputs/processed_data folder
    force_rebuild : if True, rebuild even if cache exists
    backfill_lags : passed to fe.build_features()
    drop_anomaly  : passed to fe.build_features()
 
    Returns
    -------
    pd.DataFrame with all features
    """
    import feature_engineering as fe
 
    path = os.path.join(outputs_path, f'features_{site}_v3.csv')
    if os.path.exists(path) and not force_rebuild:
        print(f'Loading cached features for {site}...')
        df = pd.read_csv(path, parse_dates=['harvest_date'])
        return df.loc[:, ~df.columns.duplicated()]
    print(f'Building features for {site}...')
    df = fe.build_features(df_raw, site, weather,
                           backfill_lags=backfill_lags,
                           drop_anomaly=drop_anomaly)
    df.to_csv(path, index=False)
    print(f'  Saved to {path}')
    return df
# ── Hello test ───────────────────────────────────────────────────────────────

def hello():
    return "data_pipeline ready"