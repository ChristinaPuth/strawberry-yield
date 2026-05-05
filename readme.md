# Strawberry Yield Prediction & Harvest Advisor

**Spatio-Temporal Modeling of Grid-Level Crop Yield Using Weather and Historical Yield Data**

Sites: SantaMaria (Ranch 31) · Salinas | Season: 2024

---

## Project Overview

This project builds an end-to-end pipeline that answers two core questions for strawberry farm managers:

1. **When should I harvest?** — Given the most recent harvest data, which day in the next 3–7 days maximises total yield while minimising over-ripening risk?
2. **What will the yield map look like?** — A spatial prediction of per-cell yield (~1.2 m resolution) for each candidate harvest day.

The system uses only historical yield maps and publicly available weather data — no drone imagery or remote sensing required.

---

## Repository Structure

```
Codes/
├── app.py                          # Streamlit web application
├── src/
│   ├── data_pipeline.py            # Load and clean raw CSV data
│   ├── feature_engineering.py      # Build 21 predictive features + weather fetch
│   ├── models.py                   # Ablation study + model comparison
│   ├── harvest_advisor.py          # Optimal harvest timing decision layer
│   └── visualize.py                # Yield maps, season trends, distributions
├── notebooks/
│   ├── 02_data_visualization.ipynb # Data exploration and visualisation
│   ├── 03_feature_engineering.ipynb# Feature construction and validation
│   └── 04_modeling.ipynb           # Ablation, model comparison, harvest advice
├── outputs/
│   └── processed_data/
│       ├── features_SantaMaria.csv # Pre-built features (351,024 rows)
│       ├── features_Salinas.csv    # Pre-built features (205,650 rows)
│       ├── weather_SantaMaria.csv  # Cached daily weather
│       └── weather_Salinas.csv
└── YieldMapHarvest_Original Data/  # Raw data (read-only, never modified)
    ├── Santa Maria - ranch 31_corrected_YieldMap/
    └── Salinas Harvest 2024_corrected_YieldMap/
```

---

## System Architecture — Four Layers

### Layer 1: Data Pipeline (`data_pipeline.py`)

Reads all dated harvest folders, assigns column names (raw CSVs have no headers), parses dates, and merges into a single clean DataFrame.

**Input:** Raw CSV folders  
`{date}/{date}_yield.csv` — columns: index, field_x, field_y, weight_kg, easting, northing

**Output:** Clean DataFrame with columns:
`site | harvest_date | harvest_idx | field_x | field_y | weight_kg | easting | northing`

Key design decisions:
- **Harvest-event indexing**: each harvest is numbered 1, 2, 3… in chronological order. This index is used for all lag features and train/val/test splits.
- **UTM outlier filtering**: a small number of rows have corrupt GPS coordinates (~3, 4) and are removed.
- No data is ever written back to the original data folders.

---

### Layer 2: Feature Engineering (`feature_engineering.py`)

Constructs 21 predictive features across three groups.

#### Temporal features (8)
| Feature | Description |
|---|---|
| `yield_lag1` | Yield at the immediately preceding harvest |
| `yield_lag2` | Yield two harvests back |
| `yield_lag3` | Yield three harvests back |
| `rolling_mean_3` | Mean of lag1/2/3 — smooths noise |
| `yield_trend` | (lag1 - lag3) / 2 — directional momentum |
| `season_cumulative` | Running sum of all past yields for this cell |
| `days_since_last` | Calendar days between the last two harvests |
| `day_of_year` | Day of year — encodes phenological season position |

> **Why harvest-event lags, not fixed calendar lags?**  
> Strawberry harvest intervals vary from 3 to 7 days. "7 days ago" is meaningless when the interval is 3 days — the slot is simply empty. Harvest-event lags always refer to the immediately preceding harvest event, and `days_since_last` separately encodes how many calendar days elapsed. The model learns that a 3-day gap produces less regrowth than a 7-day gap.

#### Spatial features (4)
| Feature | Description |
|---|---|
| `field_x` | Grid column index |
| `field_y` | Grid row index |
| `neighbor_mean_3x3` | Mean yield of the 3×3 neighbourhood at lag1 |
| `neighbor_mean_5x5` | Mean yield of the 5×5 neighbourhood at lag1 |

Neighbourhood features capture the strong row-level spatial autocorrelation visible in the yield maps (vertical stripe patterns). Edge cells use only available neighbours — no zero-padding.

#### Weather features (9)
All aggregated over the 7 calendar days immediately before each harvest date (the harvest day itself is excluded to prevent data leakage). Fetched from the Open-Meteo historical archive API.

| Feature | Variable | Aggregation |
|---|---|---|
| `temp_mean_7d` | Air temperature 2m | 7-day mean |
| `temp_max_7d` | Max temperature | 7-day max |
| `temp_min_7d` | Min temperature | 7-day min |
| `precip_7d` | Precipitation | 7-day sum |
| `et0_7d` | Reference evapotranspiration | 7-day sum |
| `humidity_mean_7d` | Relative humidity | 7-day mean |
| `soil_moisture_0_7` | Soil moisture 0–7 cm | 7-day mean |
| `soil_moisture_7_28` | Soil moisture 7–28 cm | 7-day mean |
| `daylight_7d` | Daylight duration | 7-day mean (hours) |

> **Why field-level uniform weather?**  
> Each field is under 200 m wide, while Open-Meteo data has ~1 km resolution. All cells within a field share identical weather values on any given day. Spatial variation in yield is explained by spatial features, not weather.

**Train / Val / Test split** (strictly chronological — random shuffle is never used):

| Site | Train | Validation | Test |
|---|---|---|---|
| SantaMaria | Harvests 1–20 (Apr 10 – Jun 14) | 21–23 (Jun 18–28) | 24–27 (Jul 2–16) |
| Salinas | Harvests 1–15 (Jun 3 – Jul 29) | 16–18 (Jul 30 – Aug 6) | 19–21 (Aug 10–19) |

---

### Layer 3: Prediction Model (`models.py`)

#### Ablation study (8 configurations)
Before model selection, a systematic ablation identifies which feature groups contribute predictive signal. All configurations use default-parameter LightGBM.

| Config | Features | SantaMaria val R² | Salinas val R² |
|---|---|---|---|
| A0 | yield_lag1 only | -0.04 | -1.93 |
| A1 | Temporal yield (5) | -0.18 | -0.61 |
| A2 | Full temporal (8) | 0.17 | 0.01 |
| A3 | Spatial only (4) | **0.68** | 0.54 |
| A4 | Spatio-temporal (12) | 0.66 | **0.55** |
| A5 | A4 + 3 core weather | 0.65 | 0.55 |
| A6 | A4 + all weather | 0.65 | 0.55 |
| A7 | All 21 features | 0.65 | 0.55 |

**Key findings:**
- Spatial features dominate — field position alone explains ~68% of yield variance in SantaMaria.
- Adding weather features does not improve performance (consistent with prior work). The 7-day weather window is too coarse to add signal beyond what spatial and temporal features already encode.
- A4 (spatio-temporal, no weather) is used as the operational feature set because it includes time-series information needed for seasonal tracking.

#### Model comparison (on A4 feature set)

| Model | SantaMaria val R² | Salinas val R² |
|---|---|---|
| Linear Regression | -2.44 | -0.22 |
| Random Forest | 0.65 | 0.52 |
| LightGBM | 0.65 | 0.55 |
| XGBoost | 0.66 | 0.55 |
| LightGBM + log(y+1) | **0.67** | **0.55** |

LightGBM with log(y+1) target transform is selected. The log transform addresses the highly right-skewed yield distribution (63.9% zero cells in SantaMaria) and reduces systematic underestimation of high-yield cells.

---

### Layer 4: Harvest Advisor (`harvest_advisor.py`)

Given the most recent actual harvest, predicts yield for each candidate harvest day (default: +3 to +7 days) and recommends the optimal timing.

**Algorithm:**
1. For each candidate day `t+k`:
   - Build inference features using the 3 most recent actual harvests as lag1/2/3
   - Fetch 7-day weather ending the day before `t+k`
   - Compute neighbourhood means from lag1
   - Run the trained model → predicted yield per cell
   - Apply over-ripening penalty to cells that have been consistently high-yield
2. Compare adjusted total yield across all candidate days
3. Return the day with the highest adjusted yield as the recommendation

**Over-ripening penalty:**  
Cells with `yield_lag1 > 75th percentile` are flagged as at risk. Their predicted yield is reduced by `penalty_weight × lag1` to discourage waiting too long. Default penalty weight = 0.05.

---

## Data Summary

| Site | Harvests | Grid cells/harvest | Total rows | Season total | Peak harvest | Avg % zero cells |
|---|---|---|---|---|---|---|
| SantaMaria | 27 (Apr 10 – Jul 16) | 14,626 | 394,902 | 101,454 kg | May 29 (8,345 kg) | 63.9% |
| Salinas | 21 (Jun 3 – Aug 19) | 11,425 | 239,925 | 46,854 kg | Jun 24 (4,575 kg) | 24.1% |

> **Salinas anomaly:** The Jul 29 / Jul 30 pair has only a 1-day interval (operationally unusual). These rows are flagged with `is_anomaly=1` and should be examined separately in residual analysis.

---

## Running the App

### Prerequisites

Python 3.9+ with a virtual environment:

```bash
cd "path/to/Codes"
python3 -m venv venv
source venv/bin/activate
pip install streamlit lightgbm xgboost openmeteo-requests requests-cache \
    retry-requests matplotlib pandas numpy scikit-learn
```

### Start the app

```bash
source venv/bin/activate
python -m streamlit run app.py
```

Browser opens at `http://localhost:8501`.

### Using the app

1. Select **Site** (SantaMaria or Salinas) in the sidebar
2. Set **Forecast window** (default: +3 to +7 days)
3. Upload the latest harvest CSV (`{date}_yield.csv`)
4. Set the **Harvest date** of the uploaded file
5. Click **Run Analysis**

The app outputs:
- Stat cards: total yield, active cells, mean yield per cell, % zero cells
- Recommendation card: optimal harvest date + expected yield
- Predicted yield map for the optimal day
- Forecast bar chart comparing all candidate days
- All candidate day yield maps

---

## Running in Google Colab

All notebooks are designed to run in Colab with Google Drive mounted.

**Standard setup cell (add to top of every notebook):**

```python
from google.colab import drive
drive.mount('/content/drive')

import sys
sys.path.insert(0, '/content/drive/MyDrive/Spatio-Temporal Modeling of '
                   'Grid-Level Crop Yield Using Weather and Historical '
                   'Yield Data/Codes/src')

BASE = ('/content/drive/MyDrive/Spatio-Temporal Modeling of '
        'Grid-Level Crop Yield Using Weather and Historical '
        'Yield Data/Codes/YieldMapHarvest_Original Data')

OUTPUTS = ('/content/drive/MyDrive/Spatio-Temporal Modeling of '
           'Grid-Level Crop Yield Using Weather and Historical '
           'Yield Data/Codes/outputs/processed_data')
```

**Reload modules after editing src files:**

```python
import importlib, data_pipeline, feature_engineering as fe, models as m
importlib.reload(data_pipeline)
importlib.reload(fe)
importlib.reload(m)
```

---

## Notebook Guide

| Notebook | Purpose |
|---|---|
| `02_data_visualization.ipynb` | Load raw data, plot yield maps, season trends, distributions. Confirms data quality. |
| `03_feature_engineering.ipynb` | Fetch weather, build 21 features, save to `processed_data/`. Run once — results are cached. |
| `04_modeling.ipynb` | Ablation study (8 configs), model comparison (5 models), harvest advisor demo. |

---

## Key Design Decisions

| Decision | Choice | Reason |
|---|---|---|
| Lag type | Harvest-event lags | Irregular intervals (3–7 days) make calendar-day lags meaningless |
| Data split | Strictly chronological | Random split leaks future yield into training set |
| Weather granularity | Field-level uniform | Field < 200 m, weather data at ~1 km resolution |
| Prediction target | Next harvest event (t+1) | Only harvest days have yield labels; calendar dates do not |
| days_since_last | Explicit input feature | Model learns regrowth as a function of elapsed time |
| Spatial position | field_x, field_y (not UTM) | Grid indices are sufficient; UTM adds no predictive value |
| Target transform | log(y+1) | Addresses extreme right skew (median 0.54 kg vs mean 0.71 kg) |

---

## Open Questions

- **lag1 availability**: Does the farm enter harvest data the same evening, or the next morning? If same evening: lag1 (today's yield) is available as a feature (+R² ~0.05–0.10). If next morning: only lag2 onward can be used. Both scenarios can be supported.
- **days_since_last at inference**: Is this provided by the farmer (planned harvest date), or should the system use the historical mean (~4 days)?
- **Salinas Jul 29/30 anomaly**: Data entry error or true double-harvest event? Affects whether these samples are kept, dropped, or flagged only.
- **Confidence intervals**: Point estimates only, or also quantile regression intervals for uncertainty quantification?

---

## Dependencies

```
pandas >= 1.3
numpy >= 1.21
scikit-learn >= 1.0
lightgbm >= 3.3
xgboost >= 1.6
matplotlib >= 3.5
streamlit >= 1.20
openmeteo-requests
requests-cache
retry-requests
```