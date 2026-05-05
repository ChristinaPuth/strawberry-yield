# """
# feature_engineering.py
# -----------------------
# Builds all 21 predictive features for the strawberry yield prediction model.

# Install dependencies (run once in Colab):
#     !pip install openmeteo-requests requests-cache retry-requests -q

# Feature groups:
#     Temporal (8):  yield_lag1/2/3, rolling_mean_3, yield_trend,
#                    season_cumulative, days_since_last, day_of_year
#     Spatial  (4):  field_x, field_y, neighbor_mean_3x3, neighbor_mean_5x5
#     Weather  (9):  temp_mean_7d, temp_max_7d, temp_min_7d, precip_7d,
#                    et0_7d, humidity_mean_7d, soil_moisture_0_7,
#                    soil_moisture_7_28, daylight_7d

# Usage in Colab:
#     import importlib, feature_engineering as fe
#     importlib.reload(fe)

#     weather_sm  = fe.fetch_weather("SantaMaria")
#     weather_sal = fe.fetch_weather("Salinas")

#     df_feat_sm  = fe.build_features(df_sm,  "SantaMaria", weather_sm)
#     df_feat_sal = fe.build_features(df_sal, "Salinas",    weather_sal)
# """

# import numpy as np
# import pandas as pd
# from datetime import timedelta

# # ── Site coordinates (WGS84, converted from UTM Zone 10N) ───────────────────
# SITE_COORDS = {
#     "SantaMaria": {"lat": 34.929, "lon": -120.432},
#     "Salinas":    {"lat": 36.643, "lon": -121.543},
# }

# # ── Daily weather variables to fetch ────────────────────────────────────────
# DAILY_VARS = [
#     "temperature_2m_mean",            # index 0 -> temp_mean
#     "temperature_2m_max",             # index 1 -> temp_max
#     "temperature_2m_min",             # index 2 -> temp_min
#     "precipitation_sum",              # index 3 -> precip
#     "et0_fao_evapotranspiration",     # index 4 -> et0
#     "relative_humidity_2m_mean",      # index 5 -> humidity_mean
#     "soil_moisture_0_to_7cm_mean",    # index 6 -> soil_moisture_0_7
#     "soil_moisture_7_to_28cm_mean",   # index 7 -> soil_moisture_7_28
#     "daylight_duration",              # index 8 -> daylight_hours (s -> h)
# ]


# # ── 1. Weather fetcher ────────────────────────────────────────────────────────

# def fetch_weather(site: str,
#                   start_date: str = "2024-04-01",
#                   end_date:   str = "2024-08-31") -> pd.DataFrame:
#     """
#     Fetch daily weather from Open-Meteo historical archive.

#     Returns a DataFrame indexed by date with columns:
#         temp_mean, temp_max, temp_min, precip, et0,
#         humidity_mean, soil_moisture_0_7, soil_moisture_7_28, daylight_hours
#     """
#     import openmeteo_requests
#     import requests_cache
#     from retry_requests import retry

#     coords = SITE_COORDS[site]
#     print(f"Fetching weather for {site}  "
#           f"({coords['lat']}N, {coords['lon']}W) ...")

#     cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
#     retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
#     om = openmeteo_requests.Client(session=retry_session)

#     params = {
#         "latitude":   coords["lat"],
#         "longitude":  coords["lon"],
#         "start_date": start_date,
#         "end_date":   end_date,
#         "daily":      DAILY_VARS,
#         "timezone":   "America/Los_Angeles",
#     }

#     responses = om.weather_api(
#         "https://archive-api.open-meteo.com/v1/archive", params=params
#     )
#     r = responses[0]
#     d = r.Daily()

#     dates = pd.date_range(
#         start     = pd.to_datetime(
#                         d.Time() + r.UtcOffsetSeconds(), unit="s", utc=True),
#         end       = pd.to_datetime(
#                         d.TimeEnd() + r.UtcOffsetSeconds(), unit="s", utc=True),
#         freq      = pd.Timedelta(seconds=d.Interval()),
#         inclusive = "left",
#     ).tz_localize(None)

#     weather = pd.DataFrame({
#         "date":               dates,
#         "temp_mean":          d.Variables(0).ValuesAsNumpy(),
#         "temp_max":           d.Variables(1).ValuesAsNumpy(),
#         "temp_min":           d.Variables(2).ValuesAsNumpy(),
#         "precip":             d.Variables(3).ValuesAsNumpy(),
#         "et0":                d.Variables(4).ValuesAsNumpy(),
#         "humidity_mean":      d.Variables(5).ValuesAsNumpy(),
#         "soil_moisture_0_7":  d.Variables(6).ValuesAsNumpy(),
#         "soil_moisture_7_28": d.Variables(7).ValuesAsNumpy(),
#         "daylight_hours":     d.Variables(8).ValuesAsNumpy() / 3600.0,
#     })

#     weather["date"] = pd.to_datetime(weather["date"])
#     weather = weather.set_index("date").sort_index()

#     print(f"  Loaded {len(weather)} days  "
#           f"({weather.index.min().date()} -> {weather.index.max().date()})")
#     return weather


# # ── 2. 7-day weather aggregator ───────────────────────────────────────────────

# def _weather_for_dates(harvest_dates, weather: pd.DataFrame) -> pd.DataFrame:
#     """
#     For each harvest date, aggregate the 7 days immediately before it.
#     Does NOT include the harvest day itself (no data leakage).
#     """
#     records = []
#     for hdate in sorted(set(harvest_dates)):
#         hdate = pd.Timestamp(hdate)
#         end   = hdate - timedelta(days=1)
#         start = hdate - timedelta(days=7)
#         w = weather[(weather.index >= start) & (weather.index <= end)]

#         if len(w) == 0:
#             rec = {"harvest_date": hdate}
#             for col in ["temp_mean_7d","temp_max_7d","temp_min_7d","precip_7d",
#                         "et0_7d","humidity_mean_7d","soil_moisture_0_7",
#                         "soil_moisture_7_28","daylight_7d"]:
#                 rec[col] = np.nan
#         else:
#             rec = {
#                 "harvest_date":        hdate,
#                 "temp_mean_7d":        round(float(w["temp_mean"].mean()), 4),
#                 "temp_max_7d":         round(float(w["temp_max"].max()),  4),
#                 "temp_min_7d":         round(float(w["temp_min"].min()),  4),
#                 "precip_7d":           round(float(w["precip"].sum()),    4),
#                 "et0_7d":              round(float(w["et0"].sum()),       4),
#                 "humidity_mean_7d":    round(float(w["humidity_mean"].mean()), 4),
#                 "soil_moisture_0_7":   round(float(w["soil_moisture_0_7"].mean()), 4),
#                 "soil_moisture_7_28":  round(float(w["soil_moisture_7_28"].mean()), 4),
#                 "daylight_7d":         round(float(w["daylight_hours"].mean()), 4),
#             }
#         records.append(rec)

#     out = pd.DataFrame(records)
#     out["harvest_date"] = pd.to_datetime(out["harvest_date"])
#     return out


# # ── 3. Neighbour mean ─────────────────────────────────────────────────────────

# def _neighbor_means(df_day: pd.DataFrame, window: int = 3) -> pd.Series:
#     """
#     For each cell in a single harvest-date slice, return the mean yield
#     of its (window x window) neighbourhood (excluding the cell itself).
#     Uses grid indices so irregular spacing is handled correctly.
#     """
#     half   = window // 2
#     x_vals = sorted(df_day["field_x"].unique())
#     y_vals = sorted(df_day["field_y"].unique())
#     x2i    = {v: i for i, v in enumerate(x_vals)}
#     y2i    = {v: i for i, v in enumerate(y_vals)}
#     i2x    = {i: v for v, i in x2i.items()}
#     i2y    = {i: v for v, i in y2i.items()}
#     lookup = df_day.set_index(["field_x","field_y"])["weight_kg"].to_dict()

#     result = {}
#     for idx, row in df_day.iterrows():
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
#         result[idx] = float(np.mean(vals)) if vals else np.nan
#     return pd.Series(result)


# # ── 4. Main feature builder ───────────────────────────────────────────────────

# def build_features(df: pd.DataFrame,
#                    site: str,
#                    weather: pd.DataFrame,
#                    lag_depth: int = 3,
#                    flag_anomalies: bool = True) -> pd.DataFrame:
#     """
#     Build all 21 features. Returns a DataFrame ready for model training.

#     Target column : weight_kg
#     Feature cols  : yield_lag1-3, rolling_mean_3, yield_trend,
#                     season_cumulative, days_since_last, day_of_year,
#                     field_x, field_y, neighbor_mean_3x3, neighbor_mean_5x5,
#                     temp_mean_7d ... daylight_7d  (9 weather cols)
#     """
#     print(f"\nBuilding features for {site} ...")
#     df = df.copy().sort_values(["field_x","field_y","harvest_date"])

#     harvest_dates = sorted(df["harvest_date"].unique())

#     # days_since_last
#     gaps = {harvest_dates[0]: np.nan}
#     for i in range(1, len(harvest_dates)):
#         gaps[harvest_dates[i]] = (harvest_dates[i] - harvest_dates[i-1]).days
#     df["days_since_last"] = df["harvest_date"].map(gaps)

#     # day_of_year
#     df["day_of_year"] = df["harvest_date"].dt.dayofyear

#     # per-cell lag features
#     print("  Computing lag + rolling features ...")
#     grp = df.groupby(["field_x","field_y"])

#     for k in range(1, lag_depth + 1):
#         df[f"yield_lag{k}"] = grp["weight_kg"].shift(k)

#     df["rolling_mean_3"] = grp["weight_kg"].transform(
#         lambda x: x.shift(1).rolling(3, min_periods=1).mean()
#     )
#     df["yield_trend"] = (
#         df.get("yield_lag1", 0) - df.get("yield_lag3", 0)
#     ) / 2.0
#     df["season_cumulative"] = grp["weight_kg"].transform(
#         lambda x: x.shift(1).expanding().sum().fillna(0)
#     )

#     # drop rows without full lag history
#     df = df.dropna(subset=[f"yield_lag{lag_depth}"]).copy()
#     print(f"  Rows after lag filter : {len(df):,}")

#     # neighbour features
#     print("  Computing 3x3 neighbour means (may take 1-2 min) ...")
#     n3 = pd.concat([
#         _neighbor_means(grp_day, window=3)
#         for _, grp_day in df.groupby("harvest_date")
#     ])
#     df["neighbor_mean_3x3"] = n3

#     print("  Computing 5x5 neighbour means ...")
#     n5 = pd.concat([
#         _neighbor_means(grp_day, window=5)
#         for _, grp_day in df.groupby("harvest_date")
#     ])
#     df["neighbor_mean_5x5"] = n5

#     # weather features
#     print("  Merging 7-day weather features ...")
#     w_feats = _weather_for_dates(df["harvest_date"].unique(), weather)
#     df = df.merge(w_feats, on="harvest_date", how="left")

#     # anomaly flag
#     if flag_anomalies and site == "Salinas":
#         anomaly_dates = [pd.Timestamp("2024-07-29"), pd.Timestamp("2024-07-30")]
#         df["is_anomaly"] = df["harvest_date"].isin(anomaly_dates).astype(int)
#     else:
#         df["is_anomaly"] = 0

#     # final column order
#     feat_cols = [
#         "yield_lag1","yield_lag2","yield_lag3",
#         "rolling_mean_3","yield_trend","season_cumulative",
#         "days_since_last","day_of_year",
      
#         "neighbor_mean_3x3","neighbor_mean_5x5",
#         "temp_mean_7d","temp_max_7d","temp_min_7d",
#         "precip_7d","et0_7d","humidity_mean_7d",
#         "soil_moisture_0_7","soil_moisture_7_28","daylight_7d",
#     ]
#     id_cols = ["site","harvest_date","harvest_idx",
#                "field_x","field_y","easting","northing"]
#     keep = [c for c in id_cols + ["weight_kg"] + feat_cols + ["is_anomaly"]
#             if c in df.columns]
#     df = df[keep].reset_index(drop=True)

#     n_nan = df[feat_cols].isna().sum().sum()
#     print(f"\n{'='*52}")
#     print(f"  Site          : {site}")
#     print(f"  Training rows : {len(df):,}")
#     print(f"  Features      : {len(feat_cols)}")
#     print(f"  Total NaN     : {n_nan}")
#     print(f"  Harvest dates : {df['harvest_date'].nunique()}")
#     print(f"{'='*52}\n")
#     return df


# # ── 5. Train / val / test split ───────────────────────────────────────────────

# def split_data(df: pd.DataFrame, site: str) -> dict:
#     """
#     Strictly chronological split by harvest_idx.
#     Random shuffle is NEVER used (would cause data leakage).

#     SantaMaria: train 1-20  val 21-23  test 24-27
#     Salinas:    train 1-15  val 16-18  test 19-21
#     """
#     config = {
#         "SantaMaria": {"train":(1,20), "val":(21,23), "test":(24,27)},
#         "Salinas":    {"train":(1,15), "val":(16,18), "test":(19,21)},
#     }
#     result = {}
#     for name, (lo, hi) in config[site].items():
#         subset = df[df["harvest_idx"].between(lo, hi)].reset_index(drop=True)
#         result[name] = subset
#         print(f"  {site} {name:5s}: harvest {lo:2d}-{hi:2d}  "
#               f"-> {len(subset):,} rows")
#     return result


# # ── 6. Feature sanity check ───────────────────────────────────────────────────

# def check_features(df: pd.DataFrame, site: str):
#     """Print a column-by-column sanity check of all 21 features."""
#     feat_cols = [
#         "yield_lag1","yield_lag2","yield_lag3",
#         "rolling_mean_3","yield_trend","season_cumulative",
#         "days_since_last","day_of_year",
#         "field_x","field_y","neighbor_mean_3x3","neighbor_mean_5x5",
#         "temp_mean_7d","temp_max_7d","temp_min_7d",
#         "precip_7d","et0_7d","humidity_mean_7d",
#         "soil_moisture_0_7","soil_moisture_7_28","daylight_7d",
#     ]
#     existing = [c for c in feat_cols if c in df.columns]
#     print(f"\n{'='*64}")
#     print(f"  Feature check -- {site}  ({len(df):,} rows)")
#     print(f"{'='*64}")
#     print(f"  {'Feature':<26} {'Non-null':>9}  {'Mean':>9}  {'Std':>9}")
#     print(f"  {'-'*60}")
#     for col in existing:
#         s = df[col].dropna()
#         warn = "  <-- NaN!" if len(s) < len(df) * 0.95 else ""
#         print(f"  {col:<26} {len(s):>9,}  "
#               f"{s.mean():>9.3f}  {s.std():>9.3f}{warn}")
#     missing = [c for c in feat_cols if c not in df.columns]
#     if missing:
#         print(f"\n  MISSING columns: {missing}")
#     print(f"{'='*64}\n")





"""
feature_engineering.py
-----------------------
Builds all 20 predictive features for the strawberry yield prediction model.
(days_since_last removed from input features → now a prediction target)

Changes from v1:
  - days_since_last removed from feature input list
  - optimal_days column added as prediction label (= days_since_last)
  - Salinas 2024-07-30 removed (anomalous zero-rate)
  - Total input features: 20 (was 21)

Feature groups:
    Temporal (7):  yield_lag1/2/3, rolling_mean_3, yield_trend,
                   season_cumulative, day_of_year
    Spatial  (4):  field_x, field_y, neighbor_mean_3x3, neighbor_mean_5x5
    Weather  (9):  temp_mean_7d, temp_max_7d, temp_min_7d, precip_7d,
                   et0_7d, humidity_mean_7d, soil_moisture_0_7,
                   soil_moisture_7_28, daylight_7d
"""

import numpy as np
import pandas as pd
from datetime import timedelta

# ── Site coordinates (WGS84) ─────────────────────────────────────────────────
SITE_COORDS = {
    "SantaMaria": {"lat": 34.929, "lon": -120.432},
    "Salinas":    {"lat": 36.643, "lon": -121.543},
}

DAILY_VARS = [
    "temperature_2m_mean",
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "et0_fao_evapotranspiration",
    "relative_humidity_2m_mean",
    "soil_moisture_0_to_7cm_mean",
    "soil_moisture_7_to_28cm_mean",
    "daylight_duration",
]

# ── Salinas anomaly date to drop ─────────────────────────────────────────────
SALINAS_DROP_DATES = [pd.Timestamp("2024-07-30")]


# ── 1. Weather fetcher ────────────────────────────────────────────────────────

def fetch_weather(site: str,
                  start_date: str = "2024-04-01",
                  end_date:   str = "2024-08-31") -> pd.DataFrame:
    import openmeteo_requests
    import requests_cache
    from retry_requests import retry

    coords = SITE_COORDS[site]
    print(f"Fetching weather for {site} ({coords['lat']}N, {coords['lon']}W)...")

    cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    om = openmeteo_requests.Client(session=retry_session)

    params = {
        "latitude":   coords["lat"],
        "longitude":  coords["lon"],
        "start_date": start_date,
        "end_date":   end_date,
        "daily":      DAILY_VARS,
        "timezone":   "America/Los_Angeles",
    }

    responses = om.weather_api(
        "https://archive-api.open-meteo.com/v1/archive", params=params
    )
    r = responses[0]
    d = r.Daily()

    dates = pd.date_range(
        start     = pd.to_datetime(d.Time() + r.UtcOffsetSeconds(), unit="s", utc=True),
        end       = pd.to_datetime(d.TimeEnd() + r.UtcOffsetSeconds(), unit="s", utc=True),
        freq      = pd.Timedelta(seconds=d.Interval()),
        inclusive = "left",
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
    print(f"  Loaded {len(weather)} days ({weather.index.min().date()} -> {weather.index.max().date()})")
    return weather


# ── 2. 7-day weather aggregator ───────────────────────────────────────────────

def _weather_for_dates(harvest_dates, weather: pd.DataFrame) -> pd.DataFrame:
    records = []
    for hdate in sorted(set(harvest_dates)):
        hdate = pd.Timestamp(hdate)
        end   = hdate - timedelta(days=1)
        start = hdate - timedelta(days=7)
        w = weather[(weather.index >= start) & (weather.index <= end)]

        if len(w) == 0:
            rec = {"harvest_date": hdate}
            for col in ["temp_mean_7d","temp_max_7d","temp_min_7d","precip_7d",
                        "et0_7d","humidity_mean_7d","soil_moisture_0_7",
                        "soil_moisture_7_28","daylight_7d"]:
                rec[col] = np.nan
        else:
            rec = {
                "harvest_date":        hdate,
                "temp_mean_7d":        round(float(w["temp_mean"].mean()), 4),
                "temp_max_7d":         round(float(w["temp_max"].max()),  4),
                "temp_min_7d":         round(float(w["temp_min"].min()),  4),
                "precip_7d":           round(float(w["precip"].sum()),    4),
                "et0_7d":              round(float(w["et0"].sum()),       4),
                "humidity_mean_7d":    round(float(w["humidity_mean"].mean()), 4),
                "soil_moisture_0_7":   round(float(w["soil_moisture_0_7"].mean()), 4),
                "soil_moisture_7_28":  round(float(w["soil_moisture_7_28"].mean()), 4),
                "daylight_7d":         round(float(w["daylight_hours"].mean()), 4),
            }
        records.append(rec)

    out = pd.DataFrame(records)
    out["harvest_date"] = pd.to_datetime(out["harvest_date"])
    return out


# ── 3. Neighbour mean ─────────────────────────────────────────────────────────

def _neighbor_means(df_day: pd.DataFrame, window: int = 3) -> pd.Series:
    half   = window // 2
    x_vals = sorted(df_day["field_x"].unique())
    y_vals = sorted(df_day["field_y"].unique())
    x2i    = {v: i for i, v in enumerate(x_vals)}
    y2i    = {v: i for i, v in enumerate(y_vals)}
    i2x    = {i: v for v, i in x2i.items()}
    i2y    = {i: v for v, i in y2i.items()}
    lookup = df_day.set_index(["field_x","field_y"])["weight_kg"].to_dict()

    result = {}
    for idx, row in df_day.iterrows():
        xi   = x2i[row["field_x"]]
        yi   = y2i[row["field_y"]]
        vals = []
        for dx in range(-half, half + 1):
            for dy in range(-half, half + 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = xi + dx, yi + dy
                if nx in i2x and ny in i2y:
                    v = lookup.get((i2x[nx], i2y[ny]))
                    if v is not None:
                        vals.append(v)
        result[idx] = float(np.mean(vals)) if vals else np.nan
    return pd.Series(result)


# ── 4. Main feature builder ───────────────────────────────────────────────────

def build_features(df: pd.DataFrame,
                   site: str,
                   weather: pd.DataFrame,
                   lag_depth: int = 3,
                   drop_anomaly: bool = True) -> pd.DataFrame:
    """
    Build all 20 features + optimal_days label.

    Key changes vs v1:
      - days_since_last removed from feature list (now only a label)
      - optimal_days = days_since_last added as prediction target
      - Salinas 2024-07-30 dropped if drop_anomaly=True

    Target columns : weight_kg, optimal_days
    Feature cols   : yield_lag1-3, rolling_mean_3, yield_trend,
                     season_cumulative, day_of_year,          (7 temporal)
                     field_x, field_y,
                     neighbor_mean_3x3, neighbor_mean_5x5,    (4 spatial)
                     temp_mean_7d ... daylight_7d              (9 weather)
    """
    print(f"\nBuilding features for {site} ...")
    df = df.copy().sort_values(["field_x","field_y","harvest_date"])

    # ── Drop Salinas anomaly date ────────────────────────────────────────────
    if drop_anomaly and site == "Salinas":
        before = len(df)
        df = df[~df["harvest_date"].isin(SALINAS_DROP_DATES)].copy()
        dropped = before - len(df)
        print(f"  [Salinas] Dropped {dropped:,} rows for anomaly dates: "
              f"{[str(d.date()) for d in SALINAS_DROP_DATES]}")

    harvest_dates = sorted(df["harvest_date"].unique())

    # ── days_since_last: compute but USE AS LABEL ONLY ───────────────────────
    # (not included in feature columns below)
    gaps = {harvest_dates[0]: np.nan}
    for i in range(1, len(harvest_dates)):
        gaps[harvest_dates[i]] = (harvest_dates[i] - harvest_dates[i-1]).days
    df["days_since_last"] = df["harvest_date"].map(gaps)

    # optimal_days label = days_since_last (shared across all cells per harvest)
    df["optimal_days"] = df["days_since_last"]

    # day_of_year (kept as feature — encodes season position without leaking days)
    df["day_of_year"] = df["harvest_date"].dt.dayofyear

    # ── Lag features ─────────────────────────────────────────────────────────
    print("  Computing lag + rolling features ...")
    grp = df.groupby(["field_x","field_y"])

    for k in range(1, lag_depth + 1):
        df[f"yield_lag{k}"] = grp["weight_kg"].shift(k)

    df["rolling_mean_3"] = grp["weight_kg"].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    )
    df["yield_trend"] = (
        df.get("yield_lag1", 0) - df.get("yield_lag3", 0)
    ) / 2.0
    df["season_cumulative"] = grp["weight_kg"].transform(
        lambda x: x.shift(1).expanding().sum().fillna(0)
    )

    # Drop rows without full lag history
    # Also drop rows where optimal_days is NaN (first harvest — no label)
    df = df.dropna(subset=[f"yield_lag{lag_depth}", "optimal_days"]).copy()
    print(f"  Rows after lag + label filter : {len(df):,}")

    # ── Neighbour features ────────────────────────────────────────────────────
    print("  Computing 3x3 neighbour means ...")
    n3 = pd.concat([
        _neighbor_means(grp_day, window=3)
        for _, grp_day in df.groupby("harvest_date")
    ])
    df["neighbor_mean_3x3"] = n3

    print("  Computing 5x5 neighbour means ...")
    n5 = pd.concat([
        _neighbor_means(grp_day, window=5)
        for _, grp_day in df.groupby("harvest_date")
    ])
    df["neighbor_mean_5x5"] = n5

    # ── Weather features ──────────────────────────────────────────────────────
    print("  Merging 7-day weather features ...")
    w_feats = _weather_for_dates(df["harvest_date"].unique(), weather)
    df = df.merge(w_feats, on="harvest_date", how="left")

    # ── Final column order ────────────────────────────────────────────────────
    # NOTE: days_since_last is NOT in feat_cols — it is only a label
    feat_cols = [
        # Temporal (7) — days_since_last removed
        "yield_lag1","yield_lag2","yield_lag3",
        "rolling_mean_3","yield_trend","season_cumulative",
        "day_of_year",
        # Spatial (4)
        "field_x","field_y","neighbor_mean_3x3","neighbor_mean_5x5",
        # Weather (9)
        "temp_mean_7d","temp_max_7d","temp_min_7d",
        "precip_7d","et0_7d","humidity_mean_7d",
        "soil_moisture_0_7","soil_moisture_7_28","daylight_7d",
    ]
    id_cols  = ["site","harvest_date","harvest_idx",
                "field_x","field_y","easting","northing"]
    # Both targets included in output
    target_cols = ["weight_kg","optimal_days"]

    keep = [c for c in id_cols + target_cols + feat_cols if c in df.columns]
    df = df[keep].reset_index(drop=True)

    n_nan = df[feat_cols].isna().sum().sum()
    print(f"\n{'='*54}")
    print(f"  Site            : {site}")
    print(f"  Training rows   : {len(df):,}")
    print(f"  Input features  : {len(feat_cols)} (days_since_last removed)")
    print(f"  Targets         : weight_kg, optimal_days")
    print(f"  Total NaN       : {n_nan}")
    print(f"  Harvest dates   : {df['harvest_date'].nunique()}")
    print(f"  optimal_days    : {df['optimal_days'].value_counts().sort_index().to_dict()}")
    print(f"{'='*54}\n")
    return df


# ── 5. Train / val / test split ───────────────────────────────────────────────

def split_data(df: pd.DataFrame, site: str) -> dict:
    """
    Strictly chronological split by harvest_idx.
    SantaMaria: train 1-20  val 21-23  test 24-27
    Salinas:    train 1-15  val 16-18  test 19-20
                (harvest 21 = Jul-30 dropped)
    """
    config = {
        "SantaMaria": {"train":(1,20), "val":(21,23), "test":(24,27)},
        "Salinas":    {"train":(1,15), "val":(16,18), "test":(19,20)},
    }
    result = {}
    for name, (lo, hi) in config[site].items():
        subset = df[df["harvest_idx"].between(lo, hi)].reset_index(drop=True)
        result[name] = subset
        print(f"  {site} {name:5s}: harvest {lo:2d}-{hi:2d}  -> {len(subset):,} rows")
    return result


# ── 6. Feature sanity check ───────────────────────────────────────────────────

def check_features(df: pd.DataFrame, site: str):
    feat_cols = [
        "yield_lag1","yield_lag2","yield_lag3",
        "rolling_mean_3","yield_trend","season_cumulative","day_of_year",
        "field_x","field_y","neighbor_mean_3x3","neighbor_mean_5x5",
        "temp_mean_7d","temp_max_7d","temp_min_7d",
        "precip_7d","et0_7d","humidity_mean_7d",
        "soil_moisture_0_7","soil_moisture_7_28","daylight_7d",
    ]
    existing = [c for c in feat_cols if c in df.columns]
    print(f"\n{'='*64}")
    print(f"  Feature check -- {site}  ({len(df):,} rows)")
    print(f"  Input features : {len(existing)}  (days_since_last is label only)")
    print(f"{'='*64}")
    print(f"  {'Feature':<26} {'Non-null':>9}  {'Mean':>9}  {'Std':>9}")
    print(f"  {'-'*60}")
    for col in existing:
        s = df[col].dropna()
        warn = "  <-- NaN!" if len(s) < len(df) * 0.95 else ""
        print(f"  {col:<26} {len(s):>9,}  {s.mean():>9.3f}  {s.std():>9.3f}{warn}")

    print(f"\n  Targets:")
    for col in ["weight_kg","optimal_days"]:
        if col in df.columns:
            s = df[col].dropna()
            print(f"  {col:<26} {len(s):>9,}  {s.mean():>9.3f}  {s.std():>9.3f}")

    missing = [c for c in feat_cols if c not in df.columns]
    if missing:
        print(f"\n  MISSING columns: {missing}")
    print(f"{'='*64}\n")