# CLAUDE.md — Marbl Energy DS Lab: New Group Context & Code Task Guide

This file is intended for **Claude Code** operating directly inside my fork of the original repository
`github.com/builtbymaxim/DsLab25W_marbl.energy`.

It provides (1) full project context, (2) a precise map of known bugs and their locations,
and (3) a prioritized task list for the new group's improvements.

---

## 1. Repository Layout (Expected)

```
DsLab25W_marbl.energy/
├── notebooks/
│   ├── 03_analysis/
│   │   └── DK1_pattern_detection.ipynb   ← clustering code lives here
│   │   └── ES_pattern_detection.ipynb   ← clustering code lives here
│   │   └── NO2_pattern_detection.ipynb   ← clustering code lives here
│   └── 04_predictions/
│       └── Predictions.ipynb             ← two-layer XGBoost lives here
├── data/
│   ├── raw/                              ← ENTSO-E + ERA5 downloads
│   └── processed/                        ← masterset CSVs per zone
├── dashboard/                            ← Streamlit app (MVC structure)
│   ├── app.py                            ← entry point / controller
│   ├── pages/                            ← historical, clusters, forecast pages
│   └── utils/                            ← inference engine, data loaders
└── requirements.txt
```

---

## 2. Project Context

**Industry partner:** Marbl Energy (`marbl.energy`)

**Previous group** (Dieringer, Körbel, Gomez Valverde, Klaric) built:
- A parallel ETL pipeline pulling day-ahead prices from ENTSO-E and ERA5 weather data
  from Copernicus for three bidding zones: **DK1** (wind), **ES** (solar), **NO2** (hydro).
- Ward linkage hierarchical clustering on 24h daily price vectors → regime labels.
- Two-layer XGBoost: Layer 1 classifies which cluster a day belongs to;
  Layer 2 has one regression model per cluster predicting hourly prices.
- A Streamlit dashboard with historical analysis, cluster viz, and day-ahead forecasts.

**New group** (Gradev, Skakala, Cvijanovic, Milosavljevic) goal:
> "We took the previous group's pipeline, diagnosed why it fails, and built a better one."

This file guides Claude Code to find the specific problems in this repo and implement the fixes.

---

## 3. Known Bugs — Exact Locations & What to Look For

### BUG 1 — Data Leakage in Clustering (CRITICAL)

**File:** `notebooks/03_analysis/DK1_pattern_detection.ipynb`
(and equivalent notebooks for ES and NO2 if they exist separately, or the same notebook
with zone-switching logic)

**What to look for:**
```python
# The leaking pattern looks like this:
clustering = AgglomerativeClustering(n_clusters=k, linkage='ward')
labels = clustering.fit_predict(price_matrix)   # price_matrix contains ALL 3 years
```

The Ward clustering call is made on the **full 3-year price matrix** — including whatever
rows later become the test set. The resulting `labels` array is then used as target variable
for the Layer 1 classifier and as the stratification key for Layer 2 regression models.
Because test days participated in forming cluster centroids, the labels are contaminated.

**Fix to implement:**
```python
from sklearn.cluster import AgglomerativeClustering
import numpy as np

# 1. Identify your train/test split index
#    (The previous group used a single split — find where test_mask or test_idx is defined)
train_mask = ...   # boolean array or index slice for training days
test_mask  = ...

# 2. Fit clustering on train only
clustering = AgglomerativeClustering(n_clusters=k, linkage='ward')
train_labels = clustering.fit_predict(price_matrix[train_mask])

# 3. Compute train centroids
centroids = np.array([
    price_matrix[train_mask][train_labels == c].mean(axis=0)
    for c in range(k)
])

# 4. Assign test days to nearest centroid (no re-fitting)
def assign_to_nearest_centroid(vectors, centroids):
    dists = np.linalg.norm(vectors[:, None, :] - centroids[None, :, :], axis=2)
    return dists.argmin(axis=1)

test_labels = assign_to_nearest_centroid(price_matrix[test_mask], centroids)

# 5. Reconstruct full label array for downstream use
all_labels = np.empty(len(price_matrix), dtype=int)
all_labels[train_mask] = train_labels
all_labels[test_mask]  = test_labels
```

**Verification:** After fixing, re-run the Layer 1 training and confirm that the cluster
label arrays fed to `XGBClassifier.fit(...)` contain **no** rows that overlap with the
rows passed to `XGBClassifier`'s evaluation/test data.

---

### BUG 2 — Data Leakage in Layer 2 Regression (CRITICAL, independent of Bug 1)

**File:** `notebooks/04_predictions/Predictions.ipynb`

**What to look for:** The Layer 2 models are trained using the *same* contaminated cluster
labels described in Bug 1. Even after fixing Bug 1, verify that the data slicing for
Layer 2 training only uses training-period rows:

```python
# Leaking pattern — cluster_labels is derived from full window:
for cluster_id in range(k):
    mask = cluster_labels == cluster_id      # full 3-year mask
    X_cluster = X_all[mask]                  # includes test rows
    y_cluster = y_all[mask]
    model.fit(X_cluster, y_cluster)          # trains on test data
```

**Fix:** After applying the Bug 1 fix, filter Layer 2 training data to `train_mask` only:

```python
for cluster_id in range(k):
    train_cluster_mask = (all_labels == cluster_id) & train_mask
    X_cluster = X_all[train_cluster_mask]
    y_cluster = y_all[train_cluster_mask]
    model.fit(X_cluster, y_cluster)
```

---

### BUG 3 — No Walk-Forward (Rolling Origin) Cross-Validation

**File:** `notebooks/04_predictions/Predictions.ipynb`

**What to look for:** A single `train_test_split` or manual index cut — something like:

```python
split_idx = int(len(df) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]
```

There is no cross-validation loop, no expanding window, no rolling origin.
The single reported WAPE numbers are therefore not robust estimates.

**Fix to implement (rolling origin CV):**

```python
from sklearn.model_selection import TimeSeriesSplit

n_splits = 5  # or choose based on dataset size
tscv = TimeSeriesSplit(n_splits=n_splits)

wape_scores = []
for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
    X_tr, X_val = X[train_idx], X[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]

    # Re-fit clustering on X_tr only (Bug 1 fix applies per fold)
    # Train Layer 1 and Layer 2 on X_tr / y_tr
    # Evaluate on X_val / y_val

    wape = compute_wape(y_val, y_pred)
    wape_scores.append(wape)
    print(f"Fold {fold}: WAPE = {wape:.2f}%")

print(f"Mean WAPE: {np.mean(wape_scores):.2f}% ± {np.std(wape_scores):.2f}%")
```

**Key constraint:** Clustering must be re-fit inside each fold's training window.
Do not fit the clusterer once globally and reuse it across folds.

Reference: Hyndman & Athanasopoulos, *Forecasting: Principles and Practice* (3rd ed.), §5.10.
URL: https://otexts.com/fpp3/tscv.html

---

### BUG 4 — Feature Poverty (No Fundamental Drivers)

**File:** `data/processed/` masterset CSVs and the feature-building cells in both notebooks.

**What to look for:** The masterset columns are currently only:
```
price_eur_mwh, temperature_2m, wind_speed_10m, precipitation_mm, solar_radiation_W
```
(5 columns total, 4 predictors)

**What to add (priority order):**

| Feature | Source | API / Dataset | Notes |
|---|---|---|---|
| Day-ahead load forecast | ENTSO-E | `entsoe-py` → `query_load_forecast()` | Highest-value addition per literature |
| Wind generation forecast | ENTSO-E | `query_wind_and_solar_forecast()` | Supply-side signal |
| Solar generation forecast | ENTSO-E | same | Supply-side signal |
| Cross-border flows | ENTSO-E | `query_crossborder_flows()` | Critical for NO2 |
| Natural gas price (TTF) | EEX / free APIs | Use D-2 settlement price | Primary marginal cost setter |
| CO₂ price (EUA) | Ember / Sandbag | Daily → broadcast to 24h | Adds directly to fossil cost |
| Hydro reservoir levels | NVE (Norway) | NVE API | Dominant driver for NO2 |

**Implementation pattern for daily → hourly broadcast:**
```python
# Daily commodity price assigned uniformly across all 24 hours of delivery day
# Use D-2 settlement price (last known before auction closes at 12:00 CET D-1)
# Handle Monday: use Friday (D-3) price since commodity markets close on weekends

def get_commodity_lag(date, settlement_prices):
    """Return the most recent settlement price available before the DA auction."""
    auction_day = date - pd.Timedelta(days=1)  # auction is on D-1
    lag = 2
    while True:
        candidate = auction_day - pd.Timedelta(days=lag - 1)
        if candidate in settlement_prices.index:
            return settlement_prices.loc[candidate]
        lag += 1  # skip weekends / holidays
```

**Test with ablation studies** — add feature groups one at a time and record WAPE delta
to produce a feature importance ladder for the final report.

---

### BUG 5 — No Autoregressive Features in Layer 2

**File:** `notebooks/04_predictions/Predictions.ipynb` — Layer 2 feature construction cell.

**What to look for:** Layer 2 feature matrix likely contains only weather variables:
```python
features_layer2 = ['temperature', 'wind_speed', 'irradiance', 'precip_rolling_Xd']
```
No lagged prices are included despite strong hourly price autocorrelation.

**Fix:** Add lagged price features — at minimum lag-24h (yesterday same hour) and lag-48h.
Be careful about look-ahead: at inference time for day D, you only know prices up to
end of day D-1. So lag-24 is safe, lag-1 (previous hour) is **not** safe for day-ahead.

```python
df['price_lag_24h'] = df['price_eur_mwh'].shift(24)
df['price_lag_48h'] = df['price_eur_mwh'].shift(48)
df['price_lag_168h'] = df['price_eur_mwh'].shift(168)  # same hour last week
```

---

### BUG 6 — No Hyperparameter Tuning

**File:** `notebooks/04_predictions/Predictions.ipynb` — wherever `XGBClassifier()`
and `XGBRegressor()` are instantiated.

**What to look for:**
```python
model = XGBClassifier()         # default params
model = XGBRegressor()          # default params
```

**Fix:** Use cross-validated tuning within the training fold only.
Recommended approach: `optuna` or `sklearn`'s `RandomizedSearchCV` with `TimeSeriesSplit`.

```python
import optuna
from xgboost import XGBRegressor

def objective(trial):
    params = {
        'n_estimators':  trial.suggest_int('n_estimators', 100, 1000),
        'max_depth':     trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample':     trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha':     trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda':    trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
    }
    # evaluate using inner TimeSeriesSplit on training data only
    ...
    return mean_wape

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)
```

---

### BUG 7 — Cluster 2 in DK1 Has Only 45 Days (Too Small for Reliable Regression)

**File:** `notebooks/04_predictions/Predictions.ipynb`

**What to look for:** After clustering, check cluster sizes before training Layer 2 models:

```python
cluster_sizes = pd.Series(all_labels).value_counts().sort_index()
print(cluster_sizes)
# DK1: cluster 2 → 45 days → only ~45 * 24 = 1080 hourly observations
```

45 days is far too small to reliably train an XGBoost regression with ~10 features.
This is a structural consequence of DK1 having k=6 clusters.

**Fix options:**
1. Merge small clusters (< 60 days) with the nearest centroid cluster.
2. Use Gaussian HMM (see Section 5) — regime size is an output of the model, not a
   hard split, and small regimes simply get lower posterior probability.
3. If keeping Ward, reduce k for DK1 from 6 to 5 and re-evaluate.

**Detection code to add as an assertion:**

```python
MIN_CLUSTER_DAYS = 60
for zone, labels in zone_labels.items():
    sizes = pd.Series(labels).value_counts()
    small = sizes[sizes < MIN_CLUSTER_DAYS]
    if not small.empty:
        print(f"WARNING [{zone}]: clusters {small.index.tolist()} have < {MIN_CLUSTER_DAYS} days")
```

---

### BUG 8 — Static Clustering: No Temporal Penalty for Regime Fragmentation

**File:** `notebooks/03_analysis/DK1_pattern_detection.ipynb`

**What to look for:** Ward clustering operates on the full price matrix with no temporal
ordering constraint. Two consecutive days can be assigned to completely different clusters
with no penalty, even though electricity market regimes exhibit strong persistence
(a "high wind" day is very likely to be followed by another "high wind" day).

The calendar heatmaps in the report visually confirm this: cluster assignments appear
noisy within a week, with frequent one-day excursions into a different cluster.

**Fix:** Replace Ward clustering with a Gaussian HMM (see Section 5 below).
The Markov transition matrix in an HMM explicitly encodes regime persistence:
a day is unlikely to switch regimes from one day to the next unless the price profile
shifts substantially. This produces temporally smoother regime sequences that are more
physically interpretable.

---

## 4. WAPE Computation — Reference Implementation

The previous group's WAPE metric is defined as:

```
WAPE = sum(|y_true - y_pred|) / sum(|y_true|) * 100
```

Reference implementation to use consistently:

```python
import numpy as np

def compute_wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Weighted Absolute Percentage Error.
    Volume-weighted, robust to near-zero prices unlike MAPE.
    Returns a percentage (e.g. 31.45 not 0.3145).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    assert len(y_true) == len(y_pred), "Length mismatch"
    denom = np.sum(np.abs(y_true))
    if denom == 0:
        raise ValueError("Sum of |y_true| is zero — WAPE undefined")
    return np.sum(np.abs(y_true - y_pred)) / denom * 100.0
```

**Baseline benchmarks to always re-report alongside any new model:**

| Baseline | Implementation |
|---|---|
| Naive-1 (yesterday) | `y_pred[t] = y_true[t - 24]` |
| Naive-2 (weather-only XGBoost, no clusters) | single XGBRegressor on all data with weather features only |

A new model result is only meaningful if it beats **both** baselines on the clean
(leakage-free, walk-forward CV) evaluation.

---

## 5. Improvement Tasks for the New Group

The following tasks are in priority order. Each task has a clear entry point in the code.

### Task 1 — Fix Leakage & Establish Clean Baseline (Weeks 1–3)
**Status: Must complete first. Everything else builds on this.**

Steps:
1. In `DK1_pattern_detection.ipynb` (and equivalents): implement the Bug 1 fix.
2. In `Predictions.ipynb`: implement the Bug 2 fix (Layer 2 train-only filtering).
3. Implement Bug 3 fix: replace single split with 5-fold `TimeSeriesSplit` CV.
4. Recompute WAPE for all three zones. These are the **true baseline numbers**.
5. Expect them to be worse than the originally reported numbers (31.45 / 33.84 / 23.28).
   This is the honest starting point — document it as such.

### Task 2 — Failure Mode Diagnostics (Weeks 4–7)
**Deliverable: Regime-level evaluation report. This is Marbl's explicit ask #1.**

After Task 1 is complete, compute WAPE broken down along every axis:
```python
# WAPE by cluster
for c in range(k):
    mask = (test_labels == c)
    wape_c = compute_wape(y_true[mask], y_pred[mask])
    print(f"Cluster {c} (n={mask.sum()}): WAPE = {wape_c:.2f}%")

# WAPE by season
for season in ['spring', 'summer', 'autumn', 'winter']:
    mask = (test_df['season'] == season)
    wape_s = compute_wape(y_true[mask], y_pred[mask])
    print(f"{season}: WAPE = {wape_s:.2f}%")

# WAPE by weekday vs weekend
for is_weekend in [False, True]:
    mask = (test_df['is_weekend'] == is_weekend)
    label = "Weekend" if is_weekend else "Weekday"
    print(f"{label}: WAPE = {compute_wape(y_true[mask], y_pred[mask]):.2f}%")

# WAPE on extreme price days (top/bottom 5% of daily mean prices)
p95 = np.percentile(test_df['daily_mean_price'], 95)
p05 = np.percentile(test_df['daily_mean_price'], 5)
mask_high = test_df['daily_mean_price'] >= p95
mask_low  = test_df['daily_mean_price'] <= p05
print(f"High-price days: WAPE = {compute_wape(y_true[mask_high], y_pred[mask_high]):.2f}%")
print(f"Low-price days:  WAPE = {compute_wape(y_true[mask_low],  y_pred[mask_low]):.2f}%")
```

### Task 3a — Feature Enrichment (Weeks 8–11, highest priority)

See Bug 4 above for the feature list and implementation pattern.

Run ablation studies: start from the clean Bug 1/2/3-fixed baseline, then add feature
groups one at a time. After each addition, re-run walk-forward CV and record WAPE delta.
This produces a feature importance ladder for the final report.

Suggested order:
1. Add day-ahead load forecast → re-evaluate
2. Add wind + solar generation forecast → re-evaluate
3. Add lagged prices within Layer 2 (Bug 5 fix) → re-evaluate
4. Add cross-border flows → re-evaluate
5. Add gas + CO₂ prices (D-2 settlement) → re-evaluate
6. Add hydro reservoir levels for NO2 → re-evaluate

### Task 3b — Gaussian HMM Clustering

**Library:** `hmmlearn` (`pip install hmmlearn`)

```python
from hmmlearn import hmm
import numpy as np

# price_matrix_train: shape (n_train_days, 24)
# Each row is the 24-hour price profile for one day

n_regimes = 5  # tune via BIC/AIC or elbow on log-likelihood
model = hmm.GaussianHMM(
    n_components=n_regimes,
    covariance_type='full',   # full covariance per regime
    n_iter=200,
    tol=1e-4,
    random_state=42
)

# hmmlearn expects (n_samples_total, n_features) with lengths array
lengths = [1] * len(price_matrix_train)   # one sequence of length 1 per day
                                            # OR treat the 3 years as one sequence
# Treating all days as a single time series (recommended for temporal continuity):
model.fit(price_matrix_train)

# Assign train days
train_regimes = model.predict(price_matrix_train)

# Assign test days via forward algorithm (no leakage)
test_regimes    = model.predict(price_matrix_test)
test_posteriors = model.predict_proba(price_matrix_test)  # shape (n_test, n_regimes)
# test_posteriors replaces Layer 1 XGBoost output directly
```

**Covariance regularization:** With 24-dimensional vectors and possibly small regimes,
`full` covariance may become singular. If `hmmlearn` raises `LinAlgError`, try:

```python
model = hmm.GaussianHMM(n_components=n_regimes, covariance_type='diag', ...)
# or add covariance_prior / min_covar parameter:
model.min_covar = 1e-3
```

**Model selection:** Fit models for `n_components` in `[3, 4, 5, 6, 7, 8]` and select
the best by BIC: `BIC = -2 * log_likelihood + n_params * log(n_samples)`

```python
from hmmlearn import hmm

def hmm_bic(model, X):
    ll = model.score(X)
    n = len(X)
    # number of free parameters: transition matrix + emission params
    k = model.n_components
    d = X.shape[1]
    if model.covariance_type == 'full':
        n_cov_params = k * d * (d + 1) / 2
    elif model.covariance_type == 'diag':
        n_cov_params = k * d
    n_params = k * (k - 1) + k * d + n_cov_params  # transition + means + covs
    return -2 * ll * n + n_params * np.log(n)

bic_scores = {}
for k in range(3, 9):
    m = hmm.GaussianHMM(n_components=k, covariance_type='full', n_iter=200, random_state=42)
    m.fit(price_matrix_train)
    bic_scores[k] = hmm_bic(m, price_matrix_train)
    print(f"k={k}: BIC={bic_scores[k]:.1f}, log-likelihood={m.score(price_matrix_train):.1f}")

best_k = min(bic_scores, key=bic_scores.get)
print(f"Best k by BIC: {best_k}")
```

### Task 4 — Hyperparameter Tuning (Bug 6 fix)

After Tasks 1–3a are complete. Tune XGBoost Layer 1 and Layer 2 separately using optuna
within the training fold only. Expected WAPE reduction: 5–10%.

### Task 5 — Benchmarking (Weeks 12–14)

Once the improved pipeline is stable, benchmark against LSTM as an alternative to XGBoost
Layer 2. Use the same walk-forward CV framework. Note: feature set matters more than
model architecture at this stage — prioritize features first.

---

## 6. Key Invariants — Never Violate These

1. **No future data in training.** At prediction time for day D, only features available
   before 12:00 CET on day D-1 (the auction close) are allowed.
2. **Clustering must be re-fit inside each CV fold** on that fold's training data only.
3. **Test set rows must never influence any fitted object** (clusterer, scaler, imputer,
   feature selector, or model). Use `fit_transform` only on train, `transform` only on test.
4. **WAPE is always reported alongside both naive baselines.** A result without baselines
   is not a result.
5. **Walk-forward CV only.** Never use k-fold or random splits on time series data.

---

## 7. Data Sources & Credentials

| Source | Access method | Notes |
|---|---|---|
| ENTSO-E | `entsoe-py` library + API key | Key stored in env var `ENTSOE_API_KEY` |
| ERA5 | `cdsapi` library + CDS account | Key in `~/.cdsapirc` |
| WeatherAPI.com | REST API + key | Key in env var `WEATHER_API_KEY` |
| NVE (Norway hydro) | REST API, no auth needed | `https://hydapi.nve.no/api/v1/` |
| Gas/CO₂ prices | TBD with Armin | Use D-2 settlement; daily → broadcast to 24h |

---

## 8. Literature References

| Paper | Cite for |
|---|---|
| Hyndman & Athanasopoulos, FPP3 §5.10 (https://otexts.com/fpp3/tscv.html) | Walk-forward CV methodology |
| Tschora et al. (2022), Applied Energy | Feature importance with SHAP for electricity forecasting |
| Lago et al. (2021), Applied Energy | Day-ahead forecasting review; load forecast as top feature |
| Mosquera-López et al. (2024), Energy Economics | Weather + fundamental drivers; merit-order justification |
| Janczura & Weron (2012), AStA Advances in Statistical Analysis | HMM for electricity spot prices |
| Das et al. (2025/2026), arXiv / Energy Economics | DS-HDP-HMM; regime-aware forecasting (stretch goal) |
| Sensfuß et al. (2008), Energy Policy | Merit-order effect; CO₂ and gas as marginal cost drivers |
| Roberts & Brown (2020), Energy Reports | Ward clustering baseline (what we are replacing) |

---

## 9. Quick Sanity Checks

Run these after any code change to catch regressions early:

```python
# Check 1: no test-period rows in any training set
assert not np.any(train_mask & test_mask), "Overlap between train and test masks"

# Check 2: cluster labels derived only from training rows
unique_test_labels = set(all_labels[test_mask])
unique_train_labels = set(all_labels[train_mask])
# (having the same label values is fine — they're just integers)
# What must NOT happen: test rows were used in clustering.fit()

# Check 3: cluster sizes
for c in range(n_clusters):
    n_train_days = (all_labels[train_mask] == c).sum()
    if n_train_days < 60:
        print(f"WARNING: cluster {c} has only {n_train_days} training days")

# Check 4: WAPE sanity
assert 0 < compute_wape(y_true, y_naive1) < 100, "Naive-1 WAPE out of expected range"
assert compute_wape(y_true, y_true) == 0.0,      "Perfect prediction should be 0% WAPE"

# Check 5: HMM posteriors sum to 1
posteriors = model.predict_proba(price_matrix_test)
np.testing.assert_allclose(posteriors.sum(axis=1), 1.0, atol=1e-6)
```

---

## 10. Contact & Deliverables

**Previous repo:** `github.com/builtbymaxim/DsLab25W_marbl.energy`
**Key delivery dates (new group):**
- Refined Project Plan: 29.3.2026
- Data Integration complete: 7.4.2026
- Enhanced Cluster Analysis: 15.4.2026
- Intermediate Report: 21.4.2026
- Trained & Evaluated Models: 10.5.2026
- Dashboard improvements: 1.6.2026
- Final report & slides: 18.6.2026
