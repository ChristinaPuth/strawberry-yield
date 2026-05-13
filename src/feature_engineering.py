

# """
# feature_engineering.py  v3
# --------------------------
# Stage 1 feature engineering for strawberry yield prediction.

# Changes from v2:
#   - optimal_days removed (Stage 2 is now rule-based)
#   - days_since_last kept as METADATA only (not a feature)
#   - Two baseline predictors added:
#       seasonal_mean : per-cell expanding historical mean (no leakage)
#       trend_extrap  : 2*lag1 - lag2 (linear extrapolation)
#   - Salinas 2024-07-30 dropped (89.6% zero rate)

# Feature groups (20 total for Stage 1):
#     Temporal (7): yield_lag1/2/3, rolling_mean_3, yield_trend,
#                   season_cumulative, day_of_year
#     Spatial  (4): field_x, field_y, neighbor_mean_3x3, neighbor_mean_5x5
#     Weather  (9): temp_mean_7d, temp_max_7d, temp_min_7d, precip_7d,
#                   et0_7d, humidity_mean_7d, soil_moisture_0_7,
#                   soil_moisture_7_28, daylight_7d
# """

# import numpy as np
# import pandas as pd
# from datetime import timedelta

# # ── Site coordinates ──────────────────────────────────────────────────────────
# SITE_COORDS = {
#     "SantaMaria": {"lat": 34.929, "lon": -120.432},
#     "Salinas":    {"lat": 36.643, "lon": -121.543},
# }

# DAILY_VARS = [
#     "temperature_2m_mean", "temperature_2m_max", "temperature_2m_min",
#     "precipitation_sum", "et0_fao_evapotranspiration",
#     "relative_humidity_2m_mean", "soil_moisture_0_to_7cm_mean",
#     "soil_moisture_7_to_28cm_mean", "daylight_duration",
# ]

# SALINAS_DROP_DATES = [pd.Timestamp("2024-07-30")]

# # ── Feature groups ────────────────────────────────────────────────────────────
# TEMPORAL_FEATS = [
#     "yield_lag1", "yield_lag2", "yield_lag3",
#     "rolling_mean_3", "yield_trend", "season_cumulative", "day_of_year",
# ]
# SPATIAL_FEATS = [
#     "field_x", "field_y", "neighbor_mean_3x3", "neighbor_mean_5x5",
# ]
# WEATHER_FEATS = [
#     "temp_mean_7d", "temp_max_7d", "temp_min_7d",
#     "precip_7d", "et0_7d", "humidity_mean_7d",
#     "soil_moisture_0_7", "soil_moisture_7_28", "daylight_7d",
# ]
# ALL_FEATS = TEMPORAL_FEATS + SPATIAL_FEATS + WEATHER_FEATS  # 20 total


# # ── 1. Weather fetcher ────────────────────────────────────────────────────────

# def fetch_weather(site: str,
#                   start_date: str = "2024-04-01",
#                   end_date:   str = "2024-08-31") -> pd.DataFrame:
#     import openmeteo_requests, requests_cache
#     from retry_requests import retry

#     coords = SITE_COORDS[site]
#     print(f"Fetching weather for {site} ...")
#     cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
#     retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
#     om = openmeteo_requests.Client(session=retry_session)

#     params = {
#         "latitude": coords["lat"], "longitude": coords["lon"],
#         "start_date": start_date, "end_date": end_date,
#         "daily": DAILY_VARS, "timezone": "America/Los_Angeles",
#     }
#     responses = om.weather_api(
#         "https://archive-api.open-meteo.com/v1/archive", params=params)
#     r = responses[0]; d = r.Daily()

#     dates = pd.date_range(
#         start=pd.to_datetime(d.Time() + r.UtcOffsetSeconds(), unit="s", utc=True),
#         end=pd.to_datetime(d.TimeEnd() + r.UtcOffsetSeconds(), unit="s", utc=True),
#         freq=pd.Timedelta(seconds=d.Interval()), inclusive="left",
#     ).tz_localize(None)

#     weather = pd.DataFrame({
#         "date": dates,
#         "temp_mean": d.Variables(0).ValuesAsNumpy(),
#         "temp_max":  d.Variables(1).ValuesAsNumpy(),
#         "temp_min":  d.Variables(2).ValuesAsNumpy(),
#         "precip":    d.Variables(3).ValuesAsNumpy(),
#         "et0":       d.Variables(4).ValuesAsNumpy(),
#         "humidity_mean":      d.Variables(5).ValuesAsNumpy(),
#         "soil_moisture_0_7":  d.Variables(6).ValuesAsNumpy(),
#         "soil_moisture_7_28": d.Variables(7).ValuesAsNumpy(),
#         "daylight_hours":     d.Variables(8).ValuesAsNumpy() / 3600.0,
#     })
#     weather["date"] = pd.to_datetime(weather["date"])
#     weather = weather.set_index("date").sort_index()
#     print(f"  {len(weather)} days loaded")
#     return weather


# # ── 2. Weather aggregator ─────────────────────────────────────────────────────

# def _weather_for_dates(harvest_dates, weather: pd.DataFrame) -> pd.DataFrame:
#     records = []
#     for hdate in sorted(set(harvest_dates)):
#         hdate = pd.Timestamp(hdate)
#         w = weather[(weather.index >= hdate - timedelta(days=7)) &
#                     (weather.index <= hdate - timedelta(days=1))]
#         if len(w) == 0:
#             rec = {"harvest_date": hdate}
#             for c in ["temp_mean_7d","temp_max_7d","temp_min_7d","precip_7d",
#                       "et0_7d","humidity_mean_7d","soil_moisture_0_7",
#                       "soil_moisture_7_28","daylight_7d"]:
#                 rec[c] = np.nan
#         else:
#             rec = {
#                 "harvest_date":       hdate,
#                 "temp_mean_7d":       round(float(w["temp_mean"].mean()), 4),
#                 "temp_max_7d":        round(float(w["temp_max"].max()),   4),
#                 "temp_min_7d":        round(float(w["temp_min"].min()),   4),
#                 "precip_7d":          round(float(w["precip"].sum()),     4),
#                 "et0_7d":             round(float(w["et0"].sum()),        4),
#                 "humidity_mean_7d":   round(float(w["humidity_mean"].mean()), 4),
#                 "soil_moisture_0_7":  round(float(w["soil_moisture_0_7"].mean()), 4),
#                 "soil_moisture_7_28": round(float(w["soil_moisture_7_28"].mean()), 4),
#                 "daylight_7d":        round(float(w["daylight_hours"].mean()), 4),
#             }
#         records.append(rec)
#     out = pd.DataFrame(records)
#     out["harvest_date"] = pd.to_datetime(out["harvest_date"])
#     return out


# # ── 3. Neighbour mean ─────────────────────────────────────────────────────────

# def _neighbor_means(df_day: pd.DataFrame, window: int = 3) -> pd.Series:
#     half = window // 2
#     x_vals = sorted(df_day["field_x"].unique())
#     y_vals = sorted(df_day["field_y"].unique())
#     x2i = {v: i for i, v in enumerate(x_vals)}
#     y2i = {v: i for i, v in enumerate(y_vals)}
#     i2x = {i: v for v, i in x2i.items()}
#     i2y = {i: v for v, i in y2i.items()}
#     lookup = df_day.set_index(["field_x","field_y"])["weight_kg"].to_dict()
#     result = {}
#     for idx, row in df_day.iterrows():
#         xi, yi = x2i[row["field_x"]], y2i[row["field_y"]]
#         vals = []
#         for dx in range(-half, half+1):
#             for dy in range(-half, half+1):
#                 if dx == 0 and dy == 0: continue
#                 nx, ny = xi+dx, yi+dy
#                 if nx in i2x and ny in i2y:
#                     v = lookup.get((i2x[nx], i2y[ny]))
#                     if v is not None: vals.append(v)
#         result[idx] = float(np.mean(vals)) if vals else np.nan
#     return pd.Series(result)


# # ── 4. Main feature builder ───────────────────────────────────────────────────

# def build_features(df: pd.DataFrame,
#                    site: str,
#                    weather: pd.DataFrame,
#                    lag_depth: int = 3,
#                    drop_anomaly: bool = True) -> pd.DataFrame:
#     """
#     Build all 20 Stage 1 features + two baseline predictors.

#     days_since_last is included as METADATA only
#     (for Stage 2 statistics and Decision Quality calculation).
#     It is NOT in ALL_FEATS and must NOT be used as a model input.
#     """
#     print(f"\nBuilding features for {site} ...")
#     df = df.copy().sort_values(["field_x","field_y","harvest_date"])

#     # Drop anomaly
#     if drop_anomaly and site == "Salinas":
#         before = len(df)
#         df = df[~df["harvest_date"].isin(SALINAS_DROP_DATES)].copy()
#         print(f"  [Salinas] Dropped {before-len(df):,} rows "
#               f"({[str(d.date()) for d in SALINAS_DROP_DATES]})")

#     harvest_dates = sorted(df["harvest_date"].unique())

#     # days_since_last — METADATA only
#     gaps = {harvest_dates[0]: np.nan}
#     for i in range(1, len(harvest_dates)):
#         gaps[harvest_dates[i]] = (harvest_dates[i] - harvest_dates[i-1]).days
#     df["days_since_last"] = df["harvest_date"].map(gaps)

#     # day_of_year
#     df["day_of_year"] = df["harvest_date"].dt.dayofyear

#     # Lag features
#     print("  Computing lag + rolling features ...")
#     grp = df.groupby(["field_x","field_y"])
#     for k in range(1, lag_depth+1):
#         df[f"yield_lag{k}"] = grp["weight_kg"].shift(k)
#     df["rolling_mean_3"] = grp["weight_kg"].transform(
#         lambda x: x.shift(1).rolling(3, min_periods=1).mean())
#     df["yield_trend"] = (df.get("yield_lag1",0) - df.get("yield_lag3",0)) / 2.0
#     df["season_cumulative"] = grp["weight_kg"].transform(
#         lambda x: x.shift(1).expanding().sum().fillna(0))

#     df = df.dropna(subset=[f"yield_lag{lag_depth}"]).copy()
#     print(f"  Rows after lag filter : {len(df):,}")

#     # Neighbour features
#     print("  Computing 3x3 neighbour means ...")
#     n3 = pd.concat([_neighbor_means(g, 3) for _, g in df.groupby("harvest_date")])
#     df["neighbor_mean_3x3"] = n3
#     print("  Computing 5x5 neighbour means ...")
#     n5 = pd.concat([_neighbor_means(g, 5) for _, g in df.groupby("harvest_date")])
#     df["neighbor_mean_5x5"] = n5

#     # Weather
#     print("  Merging 7-day weather features ...")
#     df = df.merge(_weather_for_dates(df["harvest_date"].unique(), weather),
#                   on="harvest_date", how="left")

#     # Baseline predictors (no leakage: use expanding mean of past harvests)
#     df["trend_extrap"]  = (2.0 * df["yield_lag1"] - df["yield_lag2"]).clip(lower=0)
#     df["seasonal_mean"] = grp["weight_kg"].transform(
#         lambda x: x.shift(1).expanding().mean())

#     # Final columns
#     id_cols       = ["site","harvest_date","harvest_idx",
#                      "field_x","field_y","easting","northing"]
#     meta_cols     = ["days_since_last"]
#     baseline_cols = ["seasonal_mean","trend_extrap"]
#     keep = [c for c in id_cols + meta_cols + ["weight_kg"]
#             + ALL_FEATS + baseline_cols if c in df.columns]
#     df = df[keep].reset_index(drop=True)

#     n_nan = df[ALL_FEATS].isna().sum().sum()
#     print(f"\n{'='*56}")
#     print(f"  Site             : {site}")
#     print(f"  Rows             : {len(df):,}")
#     print(f"  Stage 1 features : {len(ALL_FEATS)}")
#     print(f"  Baselines        : seasonal_mean, trend_extrap")
#     print(f"  Metadata         : days_since_last (Stage 2 only)")
#     print(f"  Target           : weight_kg")
#     print(f"  NaN in features  : {n_nan}")
#     print(f"  Harvest dates    : {df['harvest_date'].nunique()}")
#     dsl = df.dropna(subset=["days_since_last"])["days_since_last"]
#     print(f"  days_since_last  : {dsl.value_counts().sort_index().to_dict()}")
#     print(f"{'='*56}\n")
#     # 删除重复列（防止merge产生 field_x.1, field_y.1）
#     # 删除重复列（防止merge产生 field_x.1, field_y.1）
#     df.columns = [str(c).split('.')[0] if str(c).endswith('.1')
#                   else c for c in df.columns]
#     df = df.loc[:, ~df.columns.duplicated()]
   
#     return df


# # ── 5. Train / val / test split ───────────────────────────────────────────────

# def split_data(df: pd.DataFrame, site: str) -> dict:
#     """
#     Strictly chronological split. No random shuffle.

#     SantaMaria : train 1-20  val 21-23  test 24-27
#     Salinas    : train 1-15  val 16-18  test 19-20
#     """
#     config = {
#         "SantaMaria": {"train":(1,20), "val":(21,23), "test":(24,27)},
#         "Salinas":    {"train":(1,15), "val":(16,18), "test":(19,20)},
#     }
#     result = {}
#     for name, (lo, hi) in config[site].items():
#         subset = df[df["harvest_idx"].between(lo, hi)].reset_index(drop=True)
#         result[name] = subset
#         dsl = subset.dropna(subset=["days_since_last"])["days_since_last"]
#         print(f"  {site} {name:5s}: harvest {lo:2d}-{hi:2d} "
#               f"-> {len(subset):,} rows  "
#               f"days={dsl.value_counts().sort_index().to_dict()}")
#     return result


# # ── 6. Sanity check ───────────────────────────────────────────────────────────

# def check_features(df: pd.DataFrame, site: str):
#     print(f"\n{'='*66}")
#     print(f"  Feature check -- {site}  ({len(df):,} rows)")
#     print(f"{'='*66}")
#     print(f"  {'Feature':<26} {'Non-null':>9}  {'Mean':>9}  {'Std':>9}")
#     print(f"  {'-'*62}")
#     for col in ALL_FEATS:
#         col = str(col)
#         if col not in df.columns:
#             print(f"  {col:<26} MISSING"); continue
#         s = df[col].dropna()
#         warn = "  <-- NaN!" if len(s) < len(df)*0.95 else ""
#         print(f"  {col:<26} {len(s):>9,}  {s.mean():>9.3f}  {s.std():>9.3f}{warn}")
#     print(f"\n  Baselines:")
#     for col in ["seasonal_mean","trend_extrap"]:
#         col = str(col)
#         if col in df.columns:
#             s = df[col].dropna()
#             print(f"  {col:<26} {len(s):>9,}  {s.mean():>9.3f}  {s.std():>9.3f}")
#     print(f"\n  Target + Metadata:")
#     for col in ["weight_kg","days_since_last"]:
#         col = str(col)
#         if col in df.columns:
#             s = df[col].dropna()
#             print(f"  {col:<26} {len(s):>9,}  {s.mean():>9.3f}  {s.std():>9.3f}")
#     print(f"{'='*66}\n")


"""
feature_engineering.py  v4
--------------------------
Stage 1 feature engineering for strawberry yield prediction.

Changes from v3:
  - Grid coarsening: coarsen_grid() aggregates 1x1 cells into NxN super-cells
      - target weight_kg is SUMMED within each super-cell
      - neighbor_mean_3x3 / neighbor_mean_5x5 are computed on super-cell grid
      - edge cells use natural boundary (no data dropped)
  - Lag backfill: first 2 harvests no longer dropped
      - yield_lag3 missing on harvests 1&2  filled with yield_lag2 (earliest known)
      - yield_lag2 missing on harvest 1      filled with yield_lag1
      - NOT data leakage: no future info used, only "earliest proxy" assumption
  - build_features() new param: backfill_lags=True (set False to replicate v3)

Feature groups (20 total for Stage 1) — unchanged:
    Temporal (7): yield_lag1/2/3, rolling_mean_3, yield_trend,
                  season_cumulative, day_of_year
    Spatial  (4): field_x, field_y, neighbor_mean_3x3, neighbor_mean_5x5
    Weather  (9): temp_mean_7d, temp_max_7d, temp_min_7d, precip_7d,
                  et0_7d, humidity_mean_7d, soil_moisture_0_7,
                  soil_moisture_7_28, daylight_7d
"""

import numpy as np
import pandas as pd
from datetime import timedelta

# ── Site coordinates ──────────────────────────────────────────────────────────
SITE_COORDS = {
    "SantaMaria": {"lat": 34.929, "lon": -120.432},
    "Salinas":    {"lat": 36.643, "lon": -121.543},
}

DAILY_VARS = [
    "temperature_2m_mean", "temperature_2m_max", "temperature_2m_min",
    "precipitation_sum", "et0_fao_evapotranspiration",
    "relative_humidity_2m_mean", "soil_moisture_0_to_7cm_mean",
    "soil_moisture_7_to_28cm_mean", "daylight_duration",
]

SALINAS_DROP_DATES = [pd.Timestamp("2024-07-30")]

# ── Feature groups ────────────────────────────────────────────────────────────
TEMPORAL_FEATS = [
    "yield_lag1", "yield_lag2", "yield_lag3",
    "rolling_mean_3", "yield_trend", "season_cumulative", "day_of_year",
]
SPATIAL_FEATS = [
    "field_x", "field_y", "neighbor_mean_3x3", "neighbor_mean_5x5",
]
WEATHER_FEATS = [
    "temp_mean_7d", "temp_max_7d", "temp_min_7d",
    "precip_7d", "et0_7d", "humidity_mean_7d",
    "soil_moisture_0_7", "soil_moisture_7_28", "daylight_7d",
]
ALL_FEATS = TEMPORAL_FEATS + SPATIAL_FEATS + WEATHER_FEATS  # 20 total


# ── 0. Grid coarsening (NEW in v4) ────────────────────────────────────────────

def coarsen_grid(df: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    """
    Aggregate original 1x1 grid cells into n x n super-cells.

    Strategy
    --------
    - field_x and field_y are integer grid indices.
    - Assign each cell: super_x = field_x // n,  super_y = field_y // n
    - weight_kg is SUMMED within the super-cell (total yield of the patch).
    - easting / northing are averaged (centre of super-cell).
    - Edge patches that don't fill a complete n x n window are included
      as-is — no data is dropped, they just contribute fewer sub-cells.
    - The new field_x / field_y equal super_x / super_y (integer indices
      of the super-cell grid), so all downstream code sees the same schema.

    Parameters
    ----------
    df : raw DataFrame from data_pipeline.load_site()
         Required columns: site, harvest_date, harvest_idx,
                           field_x, field_y, easting, northing, weight_kg
    n  : coarsening factor  (1 = no change, 3 = 3x3, 5 = 5x5)

    Returns
    -------
    DataFrame with same schema as input, fewer rows per harvest date.
    weight_kg now represents the summed yield of an n x n patch.
    """
    if n == 1:
        return df.copy()

    print(f"\nCoarsening grid: {n}x{n} super-cells ...")
    original_rows = len(df)

    df = df.copy()
    df["_xi"] = df["field_x"].astype(int)
    df["_yi"] = df["field_y"].astype(int)
    df["_sx"] = df["_xi"] // n
    df["_sy"] = df["_yi"] // n

    agg = (
        df.groupby(
            ["site", "harvest_date", "harvest_idx", "_sx", "_sy"],
            sort=False,
        )
        .agg(
            weight_kg =("weight_kg", "sum"),
            easting   =("easting",   "mean"),
            northing  =("northing",  "mean"),
        )
        .reset_index()
        .rename(columns={"_sx": "field_x", "_sy": "field_y"})
    )

    agg = agg.sort_values(
        ["harvest_date", "field_x", "field_y"]
    ).reset_index(drop=True)

    print(f"  Original : {original_rows:,} rows")
    print(f"  Coarsened: {len(agg):,} rows  "
          f"(~{original_rows/len(agg):.1f}x reduction)")
    cph = agg.groupby("harvest_date").size()
    print(f"  Super-cells/harvest: min={cph.min()}  "
          f"median={cph.median():.0f}  max={cph.max()}")
    return agg


# ── 1. Weather fetcher ────────────────────────────────────────────────────────

def fetch_weather(site: str,
                  start_date: str = "2024-04-01",
                  end_date:   str = "2024-08-31") -> pd.DataFrame:
    import openmeteo_requests, requests_cache
    from retry_requests import retry

    coords = SITE_COORDS[site]
    print(f"Fetching weather for {site} ...")
    cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    om = openmeteo_requests.Client(session=retry_session)

    params = {
        "latitude": coords["lat"], "longitude": coords["lon"],
        "start_date": start_date, "end_date": end_date,
        "daily": DAILY_VARS, "timezone": "America/Los_Angeles",
    }
    responses = om.weather_api(
        "https://archive-api.open-meteo.com/v1/archive", params=params)
    r = responses[0]; d = r.Daily()

    dates = pd.date_range(
        start=pd.to_datetime(d.Time() + r.UtcOffsetSeconds(), unit="s", utc=True),
        end=pd.to_datetime(d.TimeEnd() + r.UtcOffsetSeconds(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=d.Interval()), inclusive="left",
    ).tz_localize(None)

    weather = pd.DataFrame({
        "date":               dates,
        "temp_mean":          d.Variables(0).ValuesAsNumpy(),
        "temp_max":           d.Variables(1).ValuesAsNumpy(),
        "temp_min":           d.Variables(2).ValuesAsNumpy(),
        "precip":             d.Variables(3).ValuesAsNumpy(),
        "et0":                d.Variables(4).ValuesAsNumpy(),
        "humidity_mean":      d.Variables(5).ValuesAsNumpy(),
        "soil_moisture_0_7":  d.Variables(6).ValuesAsNumpy(),
        "soil_moisture_7_28": d.Variables(7).ValuesAsNumpy(),
        "daylight_hours":     d.Variables(8).ValuesAsNumpy() / 3600.0,
    })
    weather["date"] = pd.to_datetime(weather["date"])
    weather = weather.set_index("date").sort_index()
    print(f"  {len(weather)} days loaded")
    return weather


# ── 2. Weather aggregator ─────────────────────────────────────────────────────

def _weather_for_dates(harvest_dates, weather: pd.DataFrame) -> pd.DataFrame:
    records = []
    for hdate in sorted(set(harvest_dates)):
        hdate = pd.Timestamp(hdate)
        w = weather[(weather.index >= hdate - timedelta(days=7)) &
                    (weather.index <= hdate - timedelta(days=1))]
        if len(w) == 0:
            rec = {"harvest_date": hdate}
            for c in ["temp_mean_7d","temp_max_7d","temp_min_7d","precip_7d",
                      "et0_7d","humidity_mean_7d","soil_moisture_0_7",
                      "soil_moisture_7_28","daylight_7d"]:
                rec[c] = np.nan
        else:
            rec = {
                "harvest_date":       hdate,
                "temp_mean_7d":       round(float(w["temp_mean"].mean()), 4),
                "temp_max_7d":        round(float(w["temp_max"].max()),   4),
                "temp_min_7d":        round(float(w["temp_min"].min()),   4),
                "precip_7d":          round(float(w["precip"].sum()),     4),
                "et0_7d":             round(float(w["et0"].sum()),        4),
                "humidity_mean_7d":   round(float(w["humidity_mean"].mean()), 4),
                "soil_moisture_0_7":  round(float(w["soil_moisture_0_7"].mean()), 4),
                "soil_moisture_7_28": round(float(w["soil_moisture_7_28"].mean()), 4),
                "daylight_7d":        round(float(w["daylight_hours"].mean()), 4),
            }
        records.append(rec)
    out = pd.DataFrame(records)
    out["harvest_date"] = pd.to_datetime(out["harvest_date"])
    return out


# ── 3. Neighbour mean ─────────────────────────────────────────────────────────

def _neighbor_means(df_day: pd.DataFrame, window: int = 3) -> pd.Series:
    """
    Compute neighbour mean for every cell in df_day.

    Works on both the original 1x1 grid and coarsened n x n super-cell grids.
    field_x / field_y are treated as integer grid indices.
    For a super-cell grid one "step" in the window = n original cells.
    """
    half = window // 2
    x_vals = sorted(df_day["field_x"].unique())
    y_vals = sorted(df_day["field_y"].unique())
    x2i = {v: i for i, v in enumerate(x_vals)}
    y2i = {v: i for i, v in enumerate(y_vals)}
    i2x = {i: v for v, i in x2i.items()}
    i2y = {i: v for v, i in y2i.items()}
    lookup = df_day.set_index(["field_x","field_y"])["weight_kg"].to_dict()
    result = {}
    for idx, row in df_day.iterrows():
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
        result[idx] = float(np.mean(vals)) if vals else np.nan
    return pd.Series(result)


# ── 4. Lag backfill helper (NEW in v4) ───────────────────────────────────────

def _backfill_lags(df: pd.DataFrame, lag_depth: int = 3) -> pd.DataFrame:
    """
    Fill lag2 / lag3 for early harvests where they are NaN.

    Rules (applied in this order):
        lag3 NaN, lag2 available  ->  lag3 = lag2
        lag3 NaN, lag1 available  ->  lag3 = lag1   (harvest 2 edge case)
        lag2 NaN, lag1 available  ->  lag2 = lag1

    This is NOT data leakage: we use only already-past observations.
    The assumption is "treat the earliest known harvest as a proxy
    for the unknown earlier ones."
    """
    df = df.copy()

    if lag_depth >= 3 and "yield_lag3" in df.columns:
        mask = df["yield_lag3"].isna() & df["yield_lag2"].notna()
        df.loc[mask, "yield_lag3"] = df.loc[mask, "yield_lag2"]

        mask = df["yield_lag3"].isna() & df["yield_lag1"].notna()
        df.loc[mask, "yield_lag3"] = df.loc[mask, "yield_lag1"]

    if lag_depth >= 2 and "yield_lag2" in df.columns:
        mask = df["yield_lag2"].isna() & df["yield_lag1"].notna()
        df.loc[mask, "yield_lag2"] = df.loc[mask, "yield_lag1"]

    return df


# ── 5. Main feature builder ───────────────────────────────────────────────────

def build_features(df: pd.DataFrame,
                   site: str,
                   weather: pd.DataFrame,
                   lag_depth: int = 3,
                   drop_anomaly: bool = True,
                   backfill_lags: bool = True) -> pd.DataFrame:
    """
    Build all 20 Stage 1 features + two baseline predictors.

    Parameters
    ----------
    df            : raw OR coarsened DataFrame (load_site() or coarsen_grid())
    site          : "SantaMaria" or "Salinas"
    weather       : output of fetch_weather()
    lag_depth     : number of lag features (default 3)
    drop_anomaly  : drop known anomaly dates (Salinas 2024-07-30)
    backfill_lags : v4 default True.  Fills lag2/lag3 for early harvests so
                    only the first harvest row is dropped (not the first 3).
                    Set False to replicate v3 behaviour exactly.

    days_since_last is METADATA only — not in ALL_FEATS, not a model input.
    """
    print(f"\nBuilding features for {site} ...")
    df = df.copy().sort_values(["field_x","field_y","harvest_date"])

    # Drop anomaly
    if drop_anomaly and site == "Salinas":
        before = len(df)
        df = df[~df["harvest_date"].isin(SALINAS_DROP_DATES)].copy()
        print(f"  [Salinas] Dropped {before-len(df):,} rows "
              f"({[str(d.date()) for d in SALINAS_DROP_DATES]})")

    harvest_dates = sorted(df["harvest_date"].unique())

    # days_since_last — METADATA only
    gaps = {harvest_dates[0]: np.nan}
    for i in range(1, len(harvest_dates)):
        gaps[harvest_dates[i]] = (harvest_dates[i] - harvest_dates[i-1]).days
    df["days_since_last"] = df["harvest_date"].map(gaps)

    # day_of_year
    df["day_of_year"] = df["harvest_date"].dt.dayofyear

    # Lag features
    print("  Computing lag + rolling features ...")
    grp = df.groupby(["field_x","field_y"])
    for k in range(1, lag_depth+1):
        df[f"yield_lag{k}"] = grp["weight_kg"].shift(k)

    # v4: backfill lag2/lag3 before dropping rows
    if backfill_lags:
        df = _backfill_lags(df, lag_depth)
        df = df.dropna(subset=["yield_lag1"]).copy()
        print(f"  Rows after lag1-only filter (backfill_lags=True): {len(df):,}")
    else:
        df = df.dropna(subset=[f"yield_lag{lag_depth}"]).copy()
        print(f"  Rows after lag{lag_depth} filter (backfill_lags=False): {len(df):,}")

    # Recompute rolling / trend / cumulative after potential row drop
    grp = df.groupby(["field_x","field_y"])
    # df["rolling_mean_3"] = grp["weight_kg"].transform(
    #     lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    df["rolling_mean_3"] = grp["weight_kg"].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    df["rolling_mean_3"] = df["rolling_mean_3"].fillna(df["yield_lag1"])
    df["yield_trend"] = (df["yield_lag1"] - df["yield_lag3"]) / 2.0
    df["season_cumulative"] = grp["weight_kg"].transform(
        lambda x: x.shift(1).expanding().sum().fillna(0))

    # Neighbour features (work on any grid resolution)
    print("  Computing 3x3 neighbour means ...")
    n3 = pd.concat([_neighbor_means(g, 3) for _, g in df.groupby("harvest_date")])
    df["neighbor_mean_3x3"] = n3
    print("  Computing 5x5 neighbour means ...")
    n5 = pd.concat([_neighbor_means(g, 5) for _, g in df.groupby("harvest_date")])
    df["neighbor_mean_5x5"] = n5

    # Weather
    print("  Merging 7-day weather features ...")
    df = df.merge(_weather_for_dates(df["harvest_date"].unique(), weather),
                  on="harvest_date", how="left")

    # Baseline predictors (no leakage)
    grp = df.groupby(["field_x","field_y"])
    df["trend_extrap"]  = (2.0 * df["yield_lag1"] - df["yield_lag2"]).clip(lower=0)
    df["seasonal_mean"] = grp["weight_kg"].transform(
        lambda x: x.shift(1).expanding().mean())

    # Final columns
    id_cols       = ["site","harvest_date","harvest_idx",
                     "field_x","field_y","easting","northing"]
    meta_cols     = ["days_since_last"]
    baseline_cols = ["seasonal_mean","trend_extrap"]
    keep = [c for c in id_cols + meta_cols + ["weight_kg"]
            + ALL_FEATS + baseline_cols if c in df.columns]
    df = df[keep].reset_index(drop=True)

    n_nan = df[ALL_FEATS].isna().sum().sum()
    print(f"\n{'='*56}")
    print(f"  Site             : {site}")
    print(f"  Rows             : {len(df):,}")
    print(f"  Stage 1 features : {len(ALL_FEATS)}")
    print(f"  Baselines        : seasonal_mean, trend_extrap")
    print(f"  Metadata         : days_since_last (Stage 2 only)")
    print(f"  Target           : weight_kg")
    print(f"  NaN in features  : {n_nan}")
    print(f"  Harvest dates    : {df['harvest_date'].nunique()}")
    dsl = df.dropna(subset=["days_since_last"])["days_since_last"]
    print(f"  days_since_last  : {dsl.value_counts().sort_index().to_dict()}")
    print(f"{'='*56}\n")

    # 删除重复列（防止merge产生 field_x.1, field_y.1）
    df.columns = [str(c).split('.')[0] if str(c).endswith('.1')
                  else c for c in df.columns]
    df = df.loc[:, ~df.columns.duplicated()]

    return df


# ── 6. Train / val / test split ───────────────────────────────────────────────

def split_data(df: pd.DataFrame, site: str) -> dict:
    """
    Strictly chronological split. No random shuffle.

    SantaMaria : train 1-20  val 21-23  test 24-27
    Salinas    : train 1-15  val 16-18  test 19-20

    With backfill_lags=True, harvest_idx=2 is now in the data (previously
    dropped).  Split boundaries are unchanged — it falls in training.
    """
    config = {
        "SantaMaria": {"train":(1,20), "val":(21,23), "test":(24,27)},
        "Salinas":    {"train":(1,15), "val":(16,18), "test":(19,20)},
    }
    result = {}
    for name, (lo, hi) in config[site].items():
        subset = df[df["harvest_idx"].between(lo, hi)].reset_index(drop=True)
        result[name] = subset
        dsl = subset.dropna(subset=["days_since_last"])["days_since_last"]
        print(f"  {site} {name:5s}: harvest {lo:2d}-{hi:2d} "
              f"-> {len(subset):,} rows  "
              f"days={dsl.value_counts().sort_index().to_dict()}")
    return result


# ── 7. Sanity check ───────────────────────────────────────────────────────────

def check_features(df: pd.DataFrame, site: str):
    print(f"\n{'='*66}")
    print(f"  Feature check -- {site}  ({len(df):,} rows)")
    print(f"{'='*66}")
    print(f"  {'Feature':<26} {'Non-null':>9}  {'Mean':>9}  {'Std':>9}")
    print(f"  {'-'*62}")
    for col in ALL_FEATS:
        col = str(col)
        if col not in df.columns:
            print(f"  {col:<26} MISSING"); continue
        s = df[col].dropna()
        warn = "  <-- NaN!" if len(s) < len(df)*0.95 else ""
        print(f"  {col:<26} {len(s):>9,}  {s.mean():>9.3f}  {s.std():>9.3f}{warn}")
    print(f"\n  Baselines:")
    for col in ["seasonal_mean","trend_extrap"]:
        col = str(col)
        if col in df.columns:
            s = df[col].dropna()
            print(f"  {col:<26} {len(s):>9,}  {s.mean():>9.3f}  {s.std():>9.3f}")
    print(f"\n  Target + Metadata:")
    for col in ["weight_kg","days_since_last"]:
        col = str(col)
        if col in df.columns:
            s = df[col].dropna()
            print(f"  {col:<26} {len(s):>9,}  {s.mean():>9.3f}  {s.std():>9.3f}")
    print(f"{'='*66}\n")