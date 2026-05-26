# Strawberry Yield Prediction & Harvest Advisor

**Spatio-Temporal Modeling of Grid-Level Crop Yield Using Weather and Historical Yield Data**

Sites: SantaMaria (Ranch 31) · Salinas | Season: 2024

---

## Overview

A two-stage machine learning pipeline that answers two core questions for strawberry farming:

1. **Stage 1 — Yield Prediction**: Given historical grid-level yield maps and weather data, predict the spatial yield distribution of the next harvest.
2. **Stage 2 — Harvest Timing**: Given the yield prediction and recent yield trajectory, recommend the optimal number of days to wait before harvesting.

The system is deployed as an interactive Streamlit web app where farmers can upload the latest harvest CSV and receive a data-driven recommendation.

---



## Project Structure

```
strawberry-yield/
├── app.py                      # Streamlit web application (v4)
├── requirements.txt            # Python dependencies for Streamlit Cloud
│
├── src/                        # Core pipeline modules
│   ├── data_pipeline.py        # Data loading, weather/feature caching
│   ├── feature_engineering.py  # Feature construction, normalisation, splits
│   ├── models.py               # Ablation study, model comparison, evaluation
│   ├── harvest_advisor.py      # Stage 2 rule engine + deployment API
│   ├── validation_schemes.py   # ABCDE cross-validation schemes
│   └── visualize.py            # Plotting utilities
│
├── notebooks/
│   └── 04_modeling_clean.ipynb # Main experiment notebook (full pipeline)
│
├── deployment/                 # Pre-trained artifacts (loaded by app.py)
│   ├── model_sm_7x7.pkl        # Stage 1 model — SantaMaria (LightGBM, A3, 7×7)
│   ├── model_sal_7x7.pkl       # Stage 1 model — Salinas (LightGBM+log, A4, 7×7)
│   ├── thresholds_sm.pkl       # Stage 2 Rule B thresholds — SantaMaria
│   ├── thresholds_sal.pkl      # Stage 2 Rule B thresholds — Salinas
│   ├── df_raw_sm.parquet       # Historical harvest data — SantaMaria
│   ├── df_raw_sal.parquet      # Historical harvest data — Salinas
│   ├── weather_SantaMaria.csv  # Weather data — SantaMaria
│   ├── weather_Salinas.csv     # Weather data — Salinas
│   └── deploy_config.json      # App configuration
│
└── outputs/
    └── processed_data/         # Cached features and weather (generated locally)
```

---

## Two-Stage Pipeline

### Stage 1 — Yield Prediction (LightGBM)

Predicts `weight_kg` for every grid cell at the next harvest.

**Input features (per grid cell):**

| Group | Features | Count |
|---|---|---|
| Temporal | `yield_lag1/2/3`, `rolling_mean_3`, `yield_trend`, `season_cumulative`, `day_of_year` | 7 |
| Spatial | `field_x`, `field_y`, `neighbor_mean_3x3`, `neighbor_mean_5x5` | 4 |
| Weather | `temp_mean/max/min_7d`, `precip_7d`, `et0_7d`, `humidity_mean_7d`, `soil_moisture_0_7/7_28`, `daylight_7d` | 9 |

**Best configurations (from Ablation Study):**

| Site | Feature Set | Model | Val R² |
|---|---|---|---|
| SantaMaria | A3 — Spatial only (4 features) | LightGBM | 0.913 |
| Salinas | A4 — Spatio-temporal (11 features) | LightGBM + log(y+1) | 0.735 |

**Grid resolution:** 7×7 super-cells (aggregated from original 1×1 cells)

---

### Stage 2 — Harvest Timing (Method B, Rule-Based)

Recommends how many days to wait before the next harvest using a growth-rate rule with velocity correction.

**Growth rates:**
```
gr_prev  = yield(t-2) / yield(t-3)   ← momentum of previous interval
gr_curr  = yield(t-1) / yield(t-2)   ← most recent momentum
gr_pred  = predicted_yield / yield(t-1)  ← Stage 1 prediction vs last actual
velocity = clip(gr_curr - gr_prev, -0.3, +0.3)  ← acceleration
```

**Decision matrix:**

|  | velocity ≥ 0 (accelerating) | velocity < 0 (decelerating) |
|---|---|---|
| `gr_pred ≥ t_high` | wait **long** | wait **medium** |
| `gr_pred ≥ t_low` | wait **medium** | wait **short** |
| `gr_pred < t_low` | wait **short** | wait **short** |

Thresholds (`t_low`, `t_high`) are derived from the training data distribution of harvest intervals.

---

## Experiment Design

### Validation Schemes (ABCDE)

Five cross-validation schemes to evaluate model robustness across different temporal scenarios:

| Scheme | Description | Test set |
|---|---|---|
| A.1 | Random window selection | Random 40% |
| B | Chronological sliding window (2 train + 1 predict) | Last 40% |
| C | Expanding window | Last 40% |
| D | Random split (different seed) | Random 40% |
| E | Early-season test | First 40% |

### Grid Sizes Evaluated

`1×1`, `5×5`, `7×7`, `8×8` super-cells

### Cross-Site Transfer

Train on SantaMaria → evaluate on Salinas (and vice versa) to test spatial generalisation of features across farms.

--

## Running Locally

### 1. Clone and set up environment

```bash
git clone https://github.com/ChristinaPuth/strawberry-yield.git
cd strawberry-yield
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run the app

```bash
streamlit run app.py
```

The app loads pre-trained models from `deployment/` — no training required.

### 3. Using the app

1. Select **Site** (SantaMaria or Salinas) in the sidebar
2. Upload the latest harvest CSV (format: `field_x, field_y, weight_kg, easting, northing`)
3. Set the **harvest date** of the uploaded file
4. Click **Run Analysis**
5. View the recommendation, predicted yield map, and historical timeline

---

## Reproducing the Experiments

Open `notebooks/04_modeling_clean.ipynb` in Google Colab or Kaggle.

The notebook is organised into sections:

| Section | Content |
|---|---|
| 0 — Setup | Install packages, mount Drive, import modules |
| 1 — Data Loading | Load raw harvest CSVs, fetch weather |
| 2 — Baseline Model | Feature engineering, ablation study, model comparison, test evaluation |
| 2.5 — Grid Scan | Compare 1×1 to 8×8 grid sizes on the original split |
| 3 — Cross Experiment | ABCDE × all grid sizes |
| 4 — Transfer | Cross-site generalisation |
| Stage 2 | Method B evaluation, ML vs Rule-B comparison |
| Deployment | Save artifacts to `deployment/` |

---

## Generating Deployment Artifacts

After running the full notebook, execute the **Save for Deployment** cell at the end. It reads pre-trained models and thresholds from memory and saves them to `outputs/deployment/`.

Prerequisites in memory: `df_sm`, `df_sal`, `weather_sm`, `weather_sal`, `df_feat_sm_7x7`, `df_feat_sal_7x7`, `model_results_sm_7x7`, `model_results_sal_7x7`, `thresholds_sm_7x7`, `thresholds_sal_7x7`.

---

## Data Format

Each harvest CSV file has no header and the following columns:

```
field_x, field_y, weight_kg, easting, northing
```

Or with an index column as the first column:

```
index, field_x, field_y, weight_kg, easting, northing
```

Raw data is not included in this repository. Contact the authors for access.

---

## Dependencies

```
streamlit
lightgbm
pandas
numpy
matplotlib
pyarrow
scikit-learn
xgboost
openmeteo-requests
requests-cache
retry-requests
```

---

## Citation

If you use this work, please cite:

```
Zhang, T. (2024). Spatio-Temporal Modeling of Grid-Level Crop Yield
Using Weather and Historical Yield Data. UC Davis.
```

---

## License

For academic and research use only. Contact the authors for commercial licensing.
