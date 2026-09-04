"""
BPDB Dhaka Substation Load Forecasting - 6-Model Comparison (leakage-safe v2)
=============================================================================

WHAT CHANGED vs v1 (all changes are methodological, for thesis defensibility)
-----------------------------------------------------------------------------
1. NO PREPROCESSING LEAKAGE.
   v1 imputed missing loads, filled weather and winsorised outliers over the
   WHOLE 2019-2026 series inside load_and_clean(), and only afterwards split
   into train (<=2024) / test (>=2025). Statistics from the test years could
   therefore influence the training data.

   v2 pipeline is strictly:
       raw CSV
         -> rule-based cleaning only (no statistics: clock-time artifacts and
            physically impossible values -> NaN)
         -> split by date
         -> fit_preprocessor() on TRAIN ONLY  (group medians, clip bounds,
            day-of-year weather climatology)
         -> transform_train()  (bidirectional interpolation allowed INSIDE
            the training block only)
         -> transform_test()   (CAUSAL ONLY: forward fill + train-derived
            climatology / medians. No future test value is ever used to fill
            an earlier test value, and no test value ever touches train.)
         -> lag / rolling features (all shift(1)-based, i.e. causal)
         -> MinMaxScaler fit on train only (unchanged from v1, was already OK)

2. IMPUTED TEST DAYS ARE EXCLUDED FROM THE METRICS.
   Imputed load values are not observations, so scoring against them is
   meaningless. They are still filled (the recurrent models need a continuous
   input series) but flagged, and R2 / RMSE / MAE are computed on real
   observations only. Counts are reported (n_eval, n_imputed).

3. THE TEST SET IS NO LONGER USED FOR MODEL SELECTION.
   v1 picked the "best" model per substation by highest TEST R2 and ranked the
   6 models by mean TEST R2. That is selection on the held-out set and biases
   the reported numbers upward. v2 carves a chronological VALIDATION block out
   of the end of the training period (last 15%), selects and ranks on
   VALIDATION R2, and reports TEST R2 purely as an untouched held-out score.

4. THE TARGET DAY'S EXOGENOUS VARIABLES ARE NOW VISIBLE TO THE MODEL.
   In v1 the input window was days t-14..t-1 of [load + exog] and the target
   was day t, so the model never saw day t's own temperature / holiday flag -
   the thing an operational load forecast is normally conditioned on.
   v2 shifts the exogenous block by -1 ("target alignment"): row t carries
   day t's load and day t+1's calendar+weather. The last row of the input
   window then carries the target day's exogenous values. This is legitimate
   because calendar variables are known in advance and weather comes from a
   forecast. Use --exog-alignment lagged to reproduce the v1 behaviour for an
   ablation.

5. REAL FUTURE WEATHER INSTEAD OF "TOMORROW = TODAY".
   v1's future forecast persisted the last observed weather row for all 14
   days. v2 has three modes (--weather-mode):
       forecast     (default) : pulls the daily forecast from the Open-Meteo
                                forecast API - the same provider the dataset
                                builder uses for the historical archive - for
                                as far ahead as it is available (~16 days),
                                and falls back to climatology for any day or
                                variable the API does not return.
       climatology            : smoothed day-of-year normals from history.
       persistence            : the v1 behaviour, kept for comparison.
   The weather source actually used is recorded per forecast day in the output
   CSV, and a cool-wet / hot-dry sensitivity band (+/- 1 climatological sigma)
   is produced alongside the central forecast.

6. HONEST MULTI-STEP EVALUATION (rolling-origin recursive backtest).
   The standard test panel is a ONE-DAY-AHEAD evaluation: every prediction is
   made from real past loads. The 14-day product is recursive, so its errors
   compound. v2 adds a rolling-origin backtest across the test period that
   forecasts FORECAST_DAYS ahead recursively from each origin and reports
   error by horizon step (day+1 ... day+14). Disable with --no-backtest.

7. NAIVE BASELINES.
   Persistence (load_t = load_{t-1}) and seasonal naive (load_t = load_{t-7})
   are scored on the same masked test set. A deep model that does not beat
   these is not a result.

8. REPORTING THE SPREAD, NOT JUST THE MEAN.
   test_r2_distribution.png shows a boxplot of held-out Test R2 per
   architecture across substations (every substation drawn as a point) next to
   a sorted per-substation bar chart of the validation-selected model, weakest
   first. test_r2_distribution_stats.csv / .txt give mean, median, std, IQR and
   range, and name the weakest substations explicitly. Quote the mean AND this
   spread in the results chapter - it shows nothing is being hidden.

9. SEED SENSITIVITY, HANDLED CHEAPLY.
   The full experiment stays at ONE deterministic seed (--runs 1, the default)
   because it is reproducible and 18 x 6 x 3 = 324 trainings is a lot of
   runtime to buy one footnote. Instead, after the run the script flags any
   substation x model cell whose Test R2 falls more than 0.15 below the median
   of the other five architectures at that same substation - the signature of
   a bad initialisation rather than a real architectural difference - and
   prints the exact command to investigate it:

       python load_prediction.py --csv ... --probe-only --probe "Agargaon:CNN+BiLSTM+BiGRU"

   That probe re-trains only that one cell across 5 seeds (5 trainings, not
   324) and writes a CSV, a seed-sensitivity bar chart and a plain-language
   verdict: optimisation instability, consistently low (a real result), or
   stable. Add --auto-probe to do it in the same run. --runs N is still there
   if you do want every cell averaged over N seeds, in which case the ranking
   also prints a CAUTION when the top two models are closer together than the
   seed-to-seed noise.

10. The hand-tuned festive blend (overwriting the input window with
   0.4*window + 0.6*historical festive mean) is OFF by default. It is an
   ad-hoc, unlearned adjustment and Holiday_type_encoded is already a feature.
   Re-enable with --festive-blend; when enabled, the festive mean is computed
   from TRAINING data only.

Models: BiLSTM, BiGRU, BiLSTM+BiGRU, BiGRU+BiLSTM, CNN+BiLSTM+BiGRU,
        CNN+BiGRU+BiLSTM.

Split:  Train  Year <= 2024  (last 15% chronologically = validation)
        Test   Year >= 2025

Install:
    pip install tensorflow pandas numpy scikit-learn matplotlib

Run:
    python load_prediction.py --csv BPDB_Dhaka_City_Substations_Page3.csv
    python load_prediction.py --csv ... --substations Agargaon Mirpur
    python load_prediction.py --csv ... --weather-mode climatology --no-backtest
    python load_prediction.py --csv ... --exog-alignment lagged   # v1 ablation
"""

from __future__ import annotations

import argparse
import json
import os
import random
import ssl
import urllib.parse
import urllib.request
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# -- Reproducibility -----------------------------------------------------------
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
random.seed(SEED)
np.random.seed(SEED)

import tensorflow as tf
tf.random.set_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Input, Model, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (
    GRU, LSTM, Bidirectional, Conv1D, Dense, Dropout, MaxPooling1D,
)
from tensorflow.keras.optimizers import Adam

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -- Configuration -------------------------------------------------------------
USABLE_SUBSTATIONS = [
    "Agargaon", "Bangabhaban", "Bashundhara", "Dhanmondi", "Gulshan",
    "Hasnabad", "Kalyanpur", "Lalbag", "Madartek", "Maniknagar", "Matual",
    "Mirpur", "Narinda", "Savar", "Shyampur",
    "Ullon", "Uttara", "Uttara New",
]

# Deterministic, known-in-advance calendar features
CALENDAR_FEATURES = [
    "Holiday_type_encoded",
    "Is_Weekend",
    "Season_Encoded",
    "Month_sin", "Month_cos", "DOW_sin", "DOW_cos",
]

# Measured / forecastable weather features
WEATHER_FEATURES = [
    "Temp_Max_C", "Temp_Min_C", "Temp_Mean_C",
    "Humidity_Mean_Pct", "Precip_Sum_mm", "Rain_Sum_mm", "Wind_Max_kmh",
]

# Everything the model sees besides the load history and its lag features.
EXOG_FEATURES = CALENDAR_FEATURES + WEATHER_FEATURES

# Load-derived autoregressive features (always causal, always aligned to day t)
LAG_FEATURES = ["lag_1", "lag_7", "roll_mean_7", "roll_std_7"]

FESTIVE_HOLIDAY_TYPES = {"Eid", "Durga Puja"}
HOLIDAY_TYPE_MAP = {"No Holiday": 0, "Public Holiday": 1, "Eid": 2, "Durga Puja": 3}
SEASON_MAP = {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 2, 10: 3, 11: 3}

WINDOW          = 14
FORECAST_DAYS   = 14
TRAIN_END_YEAR  = 2024
TEST_START_YEAR = 2025
VAL_FRACTION    = 0.15      # chronological tail of the training block
EPOCHS          = 150
BATCH_SIZE      = 16
PATIENCE        = 15
OUTPUT_DIR      = Path("forecast_outputs")
MAX_PLAUSIBLE_LOAD_MW = 450

# Runtime options (set from CLI in main())
EXOG_ALIGNMENT  = "target"   # "target" | "lagged"
WEATHER_MODE    = "forecast" # "forecast" | "climatology" | "persistence"
FESTIVE_BLEND   = False
RUN_BACKTEST    = True
BACKTEST_STRIDE = 14
N_RUNS          = 1   # seeds per architecture; >=3 recommended for the thesis

# Open-Meteo (same provider as dataset_from_website.py, forecast endpoint)
WEATHER_LATITUDE  = 23.8103
WEATHER_LONGITUDE = 90.4125
WEATHER_TIMEZONE  = "Asia/Dhaka"
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
WEATHER_CACHE_FILE = Path("future_weather_cache.json")

# our column name -> Open-Meteo daily variable
OPEN_METEO_VARS = {
    "Temp_Max_C":        "temperature_2m_max",
    "Temp_Min_C":        "temperature_2m_min",
    "Temp_Mean_C":       "temperature_2m_mean",
    "Humidity_Mean_Pct": "relative_humidity_2m_mean",
    "Precip_Sum_mm":     "precipitation_sum",
    "Rain_Sum_mm":       "rain_sum",
    "Wind_Max_kmh":      "wind_speed_10m_max",
}

MODEL_NAMES = [
    "BiLSTM",
    "BiGRU",
    "BiLSTM+BiGRU",
    "BiGRU+BiLSTM",
    "CNN+BiLSTM+BiGRU",
    "CNN+BiGRU+BiLSTM",
]

BASELINE_NAMES = ["Naive-Persistence", "Naive-Seasonal7"]


# =============================================================================
# Step 1: Rule-based cleaning only. NO statistics, so this is split-agnostic.
# =============================================================================

def is_clocktime_artifact(load_val, peak_time_val) -> bool:
    """A load cell that is really a HHMM clock time that slipped one column."""
    if pd.isna(load_val):
        return False
    if pd.notna(peak_time_val):
        return False
    if load_val != int(load_val):
        return False
    v = int(load_val)
    if v < 100:
        return False
    h, m = divmod(v, 100)
    return 0 <= h <= 23 and m in (0, 30)


def is_implausible_load_outlier(load_val) -> bool:
    """Physical plausibility bound - a fixed prior, not a data statistic."""
    if pd.isna(load_val):
        return False
    return load_val > MAX_PLAUSIBLE_LOAD_MW


def load_raw(csv_path: str) -> pd.DataFrame:
    """
    Read the CSV, put it on a complete daily index, derive purely deterministic
    calendar features, and blank out obvious parsing artifacts.

    Deliberately does NOT impute loads or weather and does NOT winsorise:
    every one of those steps needs a statistic, and statistics must be fitted
    on the training block only (see fit_preprocessor).
    """
    df = pd.read_csv(csv_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    for sub in USABLE_SUBSTATIONS:
        load_col, time_col = f"{sub}_Load_MW", f"{sub}_Peak_Time"
        if load_col not in df.columns:
            continue
        if time_col in df.columns:
            clock_mask = df.apply(
                lambda r, lc=load_col, tc=time_col: is_clocktime_artifact(r[lc], r[tc]),
                axis=1,
            )
            df.loc[clock_mask, load_col] = np.nan
        outlier_mask = df[load_col].apply(is_implausible_load_outlier)
        df.loc[outlier_mask, load_col] = np.nan

    full_range = pd.date_range(df["Date"].min(), df["Date"].max(), freq="D")
    df = df.set_index("Date").reindex(full_range)
    df.index.name = "Date"

    df["Year"]  = df.index.year
    df["Month"] = df.index.month
    dow_num = df.index.dayofweek
    df["DOW_num"] = dow_num
    df["Is_Weekend"] = pd.Series(dow_num.isin([4, 5]).astype(int), index=df.index)
    df["Season_Encoded"] = df["Month"].map(SEASON_MAP)

    # Holiday type: forward fill only (causal). Missing -> "No Holiday".
    # A backward fill here would drag a 2025 holiday label into 2024.
    if "Holiday_type" in df.columns:
        df["Holiday_type"] = df["Holiday_type"].ffill(limit=3).fillna("No Holiday")
    else:
        df["Holiday_type"] = "No Holiday"
    df["Holiday_type_encoded"] = (
        df["Holiday_type"].map(HOLIDAY_TYPE_MAP).fillna(0).astype(int)
    )

    # Weather columns are left with their NaNs on purpose.
    for c in WEATHER_FEATURES:
        if c not in df.columns:
            df[c] = np.nan

    df["Month_sin"] = np.sin(2 * np.pi * df["Month"] / 12)
    df["Month_cos"] = np.cos(2 * np.pi * df["Month"] / 12)
    df["DOW_sin"]   = np.sin(2 * np.pi * dow_num / 7)
    df["DOW_cos"]   = np.cos(2 * np.pi * dow_num / 7)

    return df.reset_index()


# =============================================================================
# Step 2: Preprocessor fitted on the training block only
# =============================================================================

def _smooth_circular(src: pd.Series, smooth_window: int = 15) -> pd.Series:
    """Put a day-of-year aggregate on a full 1..366 index and smooth it
    circularly, so that 31 Dec and 1 Jan are treated as neighbours."""
    idx = np.arange(1, 367)
    full = pd.Series(index=idx, dtype=float)
    for k, v in src.items():
        k = int(k)
        if 1 <= k <= 366:
            full.loc[k] = float(v) if pd.notna(v) else np.nan
    if full.notna().sum() == 0:
        return full.fillna(0.0)
    full = full.interpolate(limit_direction="both")
    tiled = pd.concat([full, full, full], ignore_index=True)
    sm = tiled.rolling(smooth_window, center=True, min_periods=1).mean()
    return pd.Series(sm.iloc[366:732].values, index=idx)


def _circular_doy_normals(dates: pd.Series, values: pd.Series, smooth_window: int = 15):
    """Day-of-year mean and std as two Series indexed 1..366."""
    doy = pd.to_datetime(dates).dt.dayofyear.values
    s = pd.Series(np.asarray(values, dtype=float))
    g = s.groupby(doy)
    return (_smooth_circular(g.mean(), smooth_window),
            _smooth_circular(g.std(), smooth_window))


@dataclass
class Preprocessor:
    """All imputation/winsorisation parameters, fitted on one block of data."""
    load_col: str
    load_group_median: dict = field(default_factory=dict)   # (month, dow) -> MW
    load_overall_median: float = 0.0
    clip_lo: float = 0.0
    clip_hi: float = MAX_PLAUSIBLE_LOAD_MW
    weather_mean_by_doy: dict = field(default_factory=dict)  # col -> Series(1..366)
    weather_std_by_doy: dict = field(default_factory=dict)
    weather_median: dict = field(default_factory=dict)
    fitted_on_dates: tuple = ()

    def climatology(self, ts: pd.Timestamp, col: str, sigma: float = 0.0) -> float:
        """Climatological value for a date, optionally shifted by k sigma."""
        doy = int(pd.Timestamp(ts).dayofyear)
        mean_s = self.weather_mean_by_doy.get(col)
        if mean_s is None or doy not in mean_s.index or pd.isna(mean_s.loc[doy]):
            return float(self.weather_median.get(col, 0.0))
        val = float(mean_s.loc[doy])
        if sigma:
            std_s = self.weather_std_by_doy.get(col)
            if std_s is not None and doy in std_s.index and pd.notna(std_s.loc[doy]):
                val += sigma * float(std_s.loc[doy])
        if col in ("Precip_Sum_mm", "Rain_Sum_mm", "Humidity_Mean_Pct", "Wind_Max_kmh"):
            val = max(val, 0.0)
        if col == "Humidity_Mean_Pct":
            val = min(val, 100.0)
        return val


def fit_preprocessor(block: pd.DataFrame, load_col: str) -> Preprocessor:
    """
    Fit every data-dependent preprocessing parameter on `block`.
    In the main pipeline `block` is the TRAINING data only.
    """
    pp = Preprocessor(load_col=load_col)
    load = block[load_col]
    observed = load.dropna()

    tmp = pd.DataFrame({
        "load": load.values,
        "Month": block["Month"].values,
        "DOW_num": block["DOW_num"].values,
    }).dropna()
    if len(tmp):
        pp.load_group_median = (
            tmp.groupby(["Month", "DOW_num"])["load"].median().to_dict()
        )
    pp.load_overall_median = float(observed.median()) if len(observed) else 0.0

    if len(observed) > 20:
        lo, hi = observed.quantile([0.005, 0.995])
        pp.clip_lo, pp.clip_hi = float(lo), float(hi)
    else:
        pp.clip_lo, pp.clip_hi = 0.0, MAX_PLAUSIBLE_LOAD_MW

    for c in WEATHER_FEATURES:
        col = block[c]
        if col.notna().sum() >= 10:
            mean_s, std_s = _circular_doy_normals(block["Date"], col)
            pp.weather_mean_by_doy[c] = mean_s
            pp.weather_std_by_doy[c] = std_s
            pp.weather_median[c] = float(col.median())
        else:
            pp.weather_median[c] = 0.0

    if len(block):
        pp.fitted_on_dates = (block["Date"].min(), block["Date"].max())
    return pp


def transform_train(block: pd.DataFrame, pp: Preprocessor) -> pd.DataFrame:
    """
    Impute the training block. Bidirectional interpolation is allowed here
    because both neighbours are inside the training period - no test
    information can enter.
    """
    out = block.copy()
    lc = pp.load_col

    out["_load_imputed"] = out[lc].isna()

    s = out.set_index("Date")[lc]
    s = s.interpolate(method="time", limit=7, limit_direction="both")
    s = s.reset_index(drop=True)
    still_na = s.isna()
    if still_na.any():
        for i in s.index[still_na]:
            key = (out["Month"].iloc[i], out["DOW_num"].iloc[i])
            s.iloc[i] = pp.load_group_median.get(key, pp.load_overall_median)
    # Winsorise the TRAINING target only (a modelling choice about what the
    # network is asked to fit; test ground truth is left untouched).
    out[lc] = s.clip(pp.clip_lo, pp.clip_hi).values

    for c in WEATHER_FEATURES:
        col = out.set_index("Date")[c].interpolate(
            method="time", limit=7, limit_direction="both"
        ).reset_index(drop=True)
        if col.isna().any():
            fill = pd.Series(
                [pp.climatology(d, c) for d in out["Date"]], index=out.index
            )
            col = col.fillna(fill)
        out[c] = col.values

    return out


def transform_test(block: pd.DataFrame, pp: Preprocessor,
                   history_tail: pd.DataFrame) -> pd.DataFrame:
    """
    Impute the test block CAUSALLY, using only:
      - parameters fitted on the training block (pp), and
      - values from strictly earlier days (the training tail + already-seen
        test days).
    No backward fill, no bidirectional interpolation: a gap on 2025-06-10 is
    never filled using 2025-06-11.
    """
    out = block.copy()
    lc = pp.load_col
    out["_load_imputed"] = out[lc].isna()

    n_hist = len(history_tail)
    cols_needed = ["Date", "Month", "DOW_num", lc] + WEATHER_FEATURES
    hist = history_tail[cols_needed].copy()
    work = pd.concat([hist, out[cols_needed]], ignore_index=True)

    # Load: forward persistence for short gaps, then train group median.
    s = work[lc].ffill(limit=7)
    if s.isna().any():
        for i in s.index[s.isna()]:
            key = (work["Month"].iloc[i], work["DOW_num"].iloc[i])
            s.iloc[i] = pp.load_group_median.get(key, pp.load_overall_median)
    work[lc] = s

    # Weather: forward fill short gaps, then train-fitted climatology.
    for c in WEATHER_FEATURES:
        col = work[c].ffill(limit=3)
        if col.isna().any():
            fill = pd.Series([pp.climatology(d, c) for d in work["Date"]],
                             index=work.index)
            col = col.fillna(fill)
        work[c] = col

    tail = work.iloc[n_hist:].reset_index(drop=True)
    out[lc] = tail[lc].values           # NOT clipped: this is ground truth
    for c in WEATHER_FEATURES:
        out[c] = tail[c].values
    return out


# =============================================================================
# Step 3: Per-substation frame assembly (split-aware)
# =============================================================================

def build_substation_frame(df: pd.DataFrame, substation: str) -> pd.DataFrame:
    """Raw, uncleaned-by-statistics frame for one substation."""
    load_col = f"{substation}_Load_MW"
    keep = ["Date", "Month", "DOW_num", "Holiday_type", load_col] + EXOG_FEATURES
    keep = [c for c in keep if c in df.columns]
    sub_df = df[keep].copy()

    first_valid = sub_df[load_col].first_valid_index()
    if first_valid is None:
        raise ValueError(f"{substation}: no real load data found.")
    return sub_df.loc[first_valid:].reset_index(drop=True)


def process_substation_frame(raw_sub: pd.DataFrame, substation: str):
    """
    Split by date, fit preprocessing on train only, transform both blocks,
    then build the causal lag/rolling features and (optionally) align the
    exogenous block to the target day.

    Returns (processed_df, preprocessor, feature_cols, train_end_date).
    `train_end_date` is the exact last day of the training block, so the later
    array split cannot drift by even one row relative to the block the
    preprocessor was fitted on.
    """
    load_col = f"{substation}_Load_MW"
    years = raw_sub["Date"].dt.year
    train_mask = years <= TRAIN_END_YEAR
    test_mask = years >= TEST_START_YEAR

    if train_mask.sum() < WINDOW * 4 or test_mask.sum() == 0:
        # Substation started too late for the calendar split: fall back to a
        # chronological 80/20 split (still strictly time-ordered).
        split_idx = int(len(raw_sub) * 0.8)
        if split_idx < WINDOW + 8:
            raise ValueError(f"{substation}: not enough data for windowing.")
        train_mask = pd.Series(False, index=raw_sub.index)
        train_mask.iloc[:split_idx] = True
        test_mask = ~train_mask
        print(f"  NOTE: {substation} uses a chronological 80/20 split "
              f"({train_mask.sum()} train, {test_mask.sum()} test days).")

    train_raw = raw_sub.loc[train_mask].reset_index(drop=True)
    test_raw = raw_sub.loc[test_mask].reset_index(drop=True)
    train_end_date = pd.Timestamp(train_raw["Date"].max())

    pp = fit_preprocessor(train_raw, load_col)
    train_t = transform_train(train_raw, pp)
    test_t = transform_test(test_raw, pp, train_t.tail(WINDOW))

    full = pd.concat([train_t, test_t], ignore_index=True).sort_values("Date")
    full = full.reset_index(drop=True)

    # Keep an unshifted copy of rainfall for the diagnostic plot.
    full["Rain_actual_mm"] = full["Rain_Sum_mm"].values

    # Causal autoregressive features (shift(1) => strictly past information).
    full["lag_1"]       = full[load_col].shift(1)
    full["lag_7"]       = full[load_col].shift(7)
    full["roll_mean_7"] = full[load_col].shift(1).rolling(7).mean()
    full["roll_std_7"]  = full[load_col].shift(1).rolling(7).std()

    # Drop the warm-up rows instead of back-filling them (v1 used bfill(),
    # which invented the first week's lags from future values).
    full = full.iloc[7:].reset_index(drop=True)

    if EXOG_ALIGNMENT == "target":
        # Row t carries day t's load/lags and day t+1's calendar+weather, so
        # the last row of an input window holds the TARGET day's exogenous
        # values. Legitimate: calendar is known, weather comes from a forecast.
        for c in EXOG_FEATURES:
            full[c] = full[c].shift(-1)
        # The final row has no t+1 exogenous data yet; it is filled from the
        # weather forecast at prediction time. Seed it with climatology so the
        # frame is never NaN.
        last = full.index[-1]
        next_date = full["Date"].iloc[-1] + pd.Timedelta(days=1)
        for c, v in calendar_features_for_date(next_date, "No Holiday").items():
            if c in full.columns:
                full.loc[last, c] = v
        for c in WEATHER_FEATURES:
            full.loc[last, c] = pp.climatology(next_date, c)

    feature_cols = [load_col] + [c for c in EXOG_FEATURES if c in full.columns] + LAG_FEATURES
    return full, pp, feature_cols, train_end_date


# =============================================================================
# Step 4: Scaling + windowing + chronological validation split
# =============================================================================

def make_sequences(values: np.ndarray, window: int):
    X, y = [], []
    for i in range(len(values) - window):
        X.append(values[i: i + window])
        y.append(values[i + window, 0])
    return np.array(X), np.array(y)


@dataclass
class Arrays:
    X_fit: np.ndarray
    y_fit: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    fit_eval_mask: np.ndarray
    val_eval_mask: np.ndarray
    test_eval_mask: np.ndarray
    fit_dates: np.ndarray
    val_dates: np.ndarray
    test_dates: np.ndarray
    scaler: MinMaxScaler
    n_features: int


def prepare_arrays(full: pd.DataFrame, load_col: str, feature_cols: list,
                   train_end_date: pd.Timestamp) -> Arrays:
    train_mask = pd.to_datetime(full["Date"]) <= pd.Timestamp(train_end_date)
    test_mask = ~train_mask

    train_df = full.loc[train_mask]
    test_df = full.loc[test_mask]

    scaler = MinMaxScaler()
    train_values = train_df[feature_cols].values
    fit_values = train_values
    if EXOG_ALIGNMENT == "target" and len(train_values) > 1:
        # Under target alignment the LAST training row carries the FIRST test
        # day's calendar+weather. That row is never inside a training input
        # window (the last training target is the final train day, whose window
        # ends one row earlier), but it would still widen the scaler's range
        # with test-period values. Exclude it from the fit; it is still used,
        # legitimately, as context for the first test sequence.
        fit_values = train_values[:-1]
    scaler.fit(fit_values)
    train_scaled = scaler.transform(train_values)
    test_scaled = scaler.transform(test_df[feature_cols].values)

    X_train, y_train = make_sequences(train_scaled, WINDOW)
    # target of sequence i is train row i+WINDOW
    train_target_pos = np.arange(WINDOW, len(train_df))
    train_imputed = train_df["_load_imputed"].values[train_target_pos]
    train_dates = train_df["Date"].values[train_target_pos]

    context = train_scaled[-WINDOW:]
    extended_test = np.concatenate([context, test_scaled], axis=0)
    X_test, y_test = make_sequences(extended_test, WINDOW)
    test_imputed = test_df["_load_imputed"].values
    test_dates = test_df["Date"].values

    n_seq = len(y_train)
    n_val = max(WINDOW, int(round(n_seq * VAL_FRACTION)))
    n_val = min(n_val, max(1, n_seq // 3))
    split = n_seq - n_val

    return Arrays(
        X_fit=X_train[:split], y_fit=y_train[:split],
        X_val=X_train[split:], y_val=y_train[split:],
        X_test=X_test, y_test=y_test,
        fit_eval_mask=~train_imputed[:split],
        val_eval_mask=~train_imputed[split:],
        test_eval_mask=~test_imputed,
        fit_dates=train_dates[:split],
        val_dates=train_dates[split:],
        test_dates=test_dates,
        scaler=scaler,
        n_features=len(feature_cols),
    )


def inverse_target(scaler, scaled_target: np.ndarray, n_features: int) -> np.ndarray:
    dummy = np.zeros((len(scaled_target), n_features))
    dummy[:, 0] = scaled_target
    return scaler.inverse_transform(dummy)[:, 0]


def masked_metrics(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray, prefix: str) -> dict:
    """Score on observed days only; imputed days are not observations."""
    if mask is None:
        mask = np.ones(len(y_true), dtype=bool)
    mask = np.asarray(mask, dtype=bool)
    if mask.sum() < 2:
        return {f"{prefix}_r2": np.nan, f"{prefix}_rmse": np.nan,
                f"{prefix}_mae": np.nan, f"{prefix}_n": int(mask.sum())}
    yt, yp = y_true[mask], y_pred[mask]
    return {
        f"{prefix}_r2": float(r2_score(yt, yp)),
        f"{prefix}_rmse": float(np.sqrt(mean_squared_error(yt, yp))),
        f"{prefix}_mae": float(mean_absolute_error(yt, yp)),
        f"{prefix}_n": int(mask.sum()),
    }


# =============================================================================
# Step 5: Future weather (replaces "tomorrow's weather = today's weather")
# =============================================================================

def _load_weather_cache() -> dict:
    if WEATHER_CACHE_FILE.exists():
        try:
            return json.loads(WEATHER_CACHE_FILE.read_text())
        except Exception:
            return {}
    return {}


def _save_weather_cache(cache: dict) -> None:
    try:
        WEATHER_CACHE_FILE.write_text(json.dumps(cache, indent=1))
    except Exception as exc:
        print(f"  (could not write weather cache: {exc})")


def fetch_open_meteo_forecast(start: pd.Timestamp, end: pd.Timestamp) -> dict:
    """
    Pull the daily forecast from Open-Meteo for [start, end].
    Returns {iso_date: {col: value}}. Returns {} on any failure - the caller
    then falls back to climatology, so the script still runs offline.
    """
    params = {
        "latitude": WEATHER_LATITUDE,
        "longitude": WEATHER_LONGITUDE,
        "daily": ",".join(OPEN_METEO_VARS.values()),
        "timezone": WEATHER_TIMEZONE,
        "start_date": pd.Timestamp(start).date().isoformat(),
        "end_date": pd.Timestamp(end).date().isoformat(),
    }
    url = OPEN_METEO_FORECAST_URL + "?" + urllib.parse.urlencode(params)
    try:
        ctx = ssl.create_default_context()
        with urllib.request.urlopen(url, timeout=30, context=ctx) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        print(f"  Weather API unavailable ({exc}); falling back to climatology.")
        return {}

    daily = payload.get("daily", {})
    times = daily.get("time", [])
    out = {}
    for i, day in enumerate(times):
        rec = {}
        for col, var in OPEN_METEO_VARS.items():
            series = daily.get(var)
            if series is not None and i < len(series) and series[i] is not None:
                rec[col] = float(series[i])
        out[day] = rec
    return out


def build_future_weather(dates: list, pp_operational: Preprocessor,
                         last_known_row: pd.Series, mode: str,
                         sigma: float = 0.0) -> pd.DataFrame:
    """
    Weather for the days we are about to forecast.

    mode="forecast"    : Open-Meteo daily forecast, climatology for anything
                         the API does not cover (beyond ~16 days, or a
                         variable it does not return).
    mode="climatology" : smoothed day-of-year normals, optionally shifted by
                         `sigma` standard deviations (used for the hot-dry /
                         cool-wet sensitivity band).
    mode="persistence" : the v1 behaviour - repeat the last observed day.
    """
    dates = [pd.Timestamp(d) for d in dates]
    rows = []

    api = {}
    if mode == "forecast" and dates:
        cache = _load_weather_cache()
        missing = [d.date().isoformat() for d in dates
                   if d.date().isoformat() not in cache]
        if missing:
            fetched = fetch_open_meteo_forecast(min(dates), max(dates))
            cache.update(fetched)
            if fetched:
                _save_weather_cache(cache)
        api = cache

    for d in dates:
        key = d.date().isoformat()
        rec = {"Date": d}
        source = []
        for c in WEATHER_FEATURES:
            val = None
            if mode == "forecast":
                val = api.get(key, {}).get(c)
                if val is not None:
                    source.append("api")
            elif mode == "persistence":
                val = float(last_known_row.get(c, 0.0))
                source.append("persistence")
            if val is None:
                val = pp_operational.climatology(d, c, sigma=sigma)
                source.append("climatology")
            rec[c] = float(val)
        rec["weather_source"] = "api" if "api" in source else source[0]
        rows.append(rec)

    return pd.DataFrame(rows).set_index("Date")


# =============================================================================
# Step 6: Calendar / holiday resolution for future dates
# =============================================================================

def calendar_features_for_date(d: pd.Timestamp, holiday_type: str) -> dict:
    d = pd.Timestamp(d)
    dow = d.dayofweek
    return {
        "Holiday_type_encoded": HOLIDAY_TYPE_MAP.get(holiday_type, 0),
        "Is_Weekend": int(dow in (4, 5)),
        "Season_Encoded": SEASON_MAP[d.month],
        "Month_sin": float(np.sin(2 * np.pi * d.month / 12)),
        "Month_cos": float(np.cos(2 * np.pi * d.month / 12)),
        "DOW_sin": float(np.sin(2 * np.pi * dow / 7)),
        "DOW_cos": float(np.cos(2 * np.pi * dow / 7)),
    }


def _bd_holiday_lookup(d: pd.Timestamp):
    """Optional: use the `holidays` package if it supports Bangladesh."""
    try:
        import holidays as _hol
        cal = _hol.country_holidays("BD", years=[d.year])
        name = cal.get(d.date())
        if not name:
            return None
        low = name.lower()
        if "eid" in low:
            return "Eid"
        if "durga" in low or "puja" in low:
            return "Durga Puja"
        return "Public Holiday"
    except Exception:
        return None


def resolve_future_holiday(df_full: pd.DataFrame, next_date: pd.Timestamp) -> str:
    """
    Holiday label for a future date. Order of preference:
      1. the dataset itself (if the date is covered),
      2. the `holidays` package,
      3. "No Holiday".
    v1's "if this month had >=3 festive days in ANY year, assume festive"
    heuristic is removed - it silently marked ordinary future days as Eid.
    """
    if "Holiday_type" in df_full.columns:
        exact = df_full.loc[df_full["Date"] == next_date, "Holiday_type"]
        if len(exact) and pd.notna(exact.iloc[0]):
            return str(exact.iloc[0])
    lib = _bd_holiday_lookup(next_date)
    if lib:
        return lib
    return "No Holiday"


# =============================================================================
# Step 7: Recursive forecasting
# =============================================================================

def _recursive_predict(work: pd.DataFrame, load_col: str, feature_cols: list,
                       scaler, model, n_features: int, n_days: int,
                       exog_provider, train_festive_mean: dict,
                       df_full: pd.DataFrame) -> list[dict]:
    """
    Core recursive loop, shared by the operational forecast and the backtest.

    `exog_provider(date)` returns a dict of exogenous features for that date
    (calendar + weather). With EXOG_ALIGNMENT == "target" the exogenous block
    of the LAST row of the input window must describe the day being predicted,
    so we overwrite it before every prediction.
    """
    work = work.copy().reset_index(drop=True)
    results = []

    for _ in range(n_days):
        last_row = work.iloc[-1]
        next_date = pd.Timestamp(last_row["Date"]) + pd.Timedelta(days=1)
        exog = exog_provider(next_date)
        holiday_type = exog.pop("_holiday_type", "No Holiday")
        weather_source = exog.pop("_weather_source", "n/a")

        if EXOG_ALIGNMENT == "target":
            for c, v in exog.items():
                if c in work.columns:
                    work.iat[len(work) - 1, work.columns.get_loc(c)] = v

        window_raw = work[feature_cols].values[-WINDOW:].astype(float).copy()

        desc = f"Standard {WINDOW}-day window ending {pd.Timestamp(last_row['Date']).date()}"
        if FESTIVE_BLEND and holiday_type in FESTIVE_HOLIDAY_TYPES:
            fm = train_festive_mean.get(holiday_type)
            if fm is not None and np.isfinite(fm):
                window_raw[:, 0] = np.clip(
                    window_raw[:, 0] * 0.4 + fm * 0.6, 0, MAX_PLAUSIBLE_LOAD_MW
                )
                desc = (f"Festive context ({holiday_type}): blended with the "
                        f"TRAIN-period mean for this holiday ({fm:.1f} MW)")
        elif holiday_type in FESTIVE_HOLIDAY_TYPES:
            desc += f" | {holiday_type} flagged via Holiday_type_encoded"

        window_scaled = scaler.transform(window_raw)
        pred_scaled = model.predict(window_scaled[np.newaxis, ...], verbose=0).ravel()
        pred_mw = float(inverse_target(scaler, pred_scaled, n_features)[0])
        pred_mw = float(np.clip(pred_mw, 0.0, MAX_PLAUSIBLE_LOAD_MW))

        results.append({
            "date": next_date,
            "pred_mw": pred_mw,
            "holiday_type": holiday_type,
            "weather_source": weather_source,
            "window_desc": desc,
        })

        # Append the newly predicted day so the lags roll forward.
        new_row = {c: last_row.get(c, 0.0) for c in work.columns}
        new_row["Date"] = next_date
        new_row[load_col] = pred_mw
        new_row["lag_1"] = float(work[load_col].iloc[-1])
        new_row["lag_7"] = float(work[load_col].iloc[-7]) if len(work) >= 7 else pred_mw
        recent7 = work[load_col].iloc[-7:]
        new_row["roll_mean_7"] = float(recent7.mean())
        new_row["roll_std_7"] = float(recent7.std()) if len(recent7) > 1 else 0.0
        if EXOG_ALIGNMENT == "target":
            # row t must carry day t+1's exogenous block
            nxt = exog_provider(next_date + pd.Timedelta(days=1))
            nxt.pop("_holiday_type", None)
            nxt.pop("_weather_source", None)
            new_row.update({c: v for c, v in nxt.items() if c in work.columns})
        else:
            new_row.update({c: v for c, v in exog.items() if c in work.columns})
        work = pd.concat([work, pd.DataFrame([new_row])], ignore_index=True)

    return results


def forecast_next_n_days(df_full: pd.DataFrame, full_sub: pd.DataFrame,
                         load_col: str, feature_cols: list, scaler, model,
                         n_features: int, pp_operational: Preprocessor,
                         train_festive_mean: dict, n_days: int = FORECAST_DAYS,
                         weather_mode: str | None = None,
                         sigma: float = 0.0) -> list[dict]:
    """Operational forecast beyond the end of the data."""
    weather_mode = weather_mode or WEATHER_MODE
    last_date = pd.Timestamp(full_sub["Date"].iloc[-1])
    future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, n_days + 2)]
    # Under target alignment the final row's weather block is a placeholder for
    # day+1, so the last genuinely observed weather sits one row earlier.
    last_known_row = full_sub.iloc[-2] if (EXOG_ALIGNMENT == "target" and len(full_sub) > 1) \
        else full_sub.iloc[-1]

    wx = build_future_weather(future_dates, pp_operational, last_known_row,
                              weather_mode, sigma=sigma)

    def exog_provider(d: pd.Timestamp) -> dict:
        d = pd.Timestamp(d)
        htype = resolve_future_holiday(df_full, d)
        feats = calendar_features_for_date(d, htype)
        if d in wx.index:
            row = wx.loc[d]
            for c in WEATHER_FEATURES:
                feats[c] = float(row[c])
            src = str(row["weather_source"])
        else:
            for c in WEATHER_FEATURES:
                feats[c] = pp_operational.climatology(d, c, sigma=sigma)
            src = "climatology"
        feats["_holiday_type"] = htype
        feats["_weather_source"] = src
        return feats

    return _recursive_predict(full_sub, load_col, feature_cols, scaler, model,
                              n_features, n_days, exog_provider,
                              train_festive_mean, df_full)


def rolling_origin_backtest(full_sub: pd.DataFrame, load_col: str,
                            feature_cols: list, scaler, model, n_features: int,
                            test_start_date: pd.Timestamp, horizon: int,
                            stride: int, train_festive_mean: dict,
                            df_full: pd.DataFrame) -> pd.DataFrame:
    """
    Honest multi-step evaluation over the test period.

    From each origin we recursively forecast `horizon` days using ONLY
    predicted loads (no actual load is fed back), while the exogenous
    variables are the recorded ones - i.e. this measures the error of the
    recursive scheme itself, under perfect weather information. Combined with
    the weather-scenario band it brackets the real operational error.
    """
    full_sub = full_sub.reset_index(drop=True)
    dates = pd.to_datetime(full_sub["Date"])
    exog_lookup = full_sub.set_index("Date")

    first_origin = int((dates < test_start_date).sum()) - 1
    if first_origin < WINDOW:
        return pd.DataFrame()

    records = []
    origin = first_origin
    while origin + horizon < len(full_sub):
        history = full_sub.iloc[: origin + 1]

        def exog_provider(d: pd.Timestamp) -> dict:
            """Recorded exogenous values DESCRIBING day d.

            Under target alignment the exogenous block was shifted by -1, so
            the values describing day d live in the row dated d-1. Holiday_type
            was never shifted, so it is read from row d itself.
            """
            d = pd.Timestamp(d)
            src_date = d - pd.Timedelta(days=1) if EXOG_ALIGNMENT == "target" else d
            if src_date in exog_lookup.index:
                row = exog_lookup.loc[src_date]
                feats = {c: float(row[c]) for c in EXOG_FEATURES if c in exog_lookup.columns}
                htype = "No Holiday"
                if "Holiday_type" in exog_lookup.columns and d in exog_lookup.index:
                    htype = str(exog_lookup.loc[d, "Holiday_type"])
                feats["_holiday_type"] = htype
                feats["_weather_source"] = "recorded"
                return feats
            htype = resolve_future_holiday(df_full, d)
            feats = calendar_features_for_date(d, htype)
            feats["_holiday_type"] = htype
            feats["_weather_source"] = "n/a"
            return feats

        preds = _recursive_predict(history, load_col, feature_cols, scaler,
                                   model, n_features, horizon, exog_provider,
                                   train_festive_mean, df_full)
        for h, r in enumerate(preds, start=1):
            pos = origin + h
            if pos >= len(full_sub):
                break
            if bool(full_sub["_load_imputed"].iloc[pos]):
                continue           # never score against an imputed value
            records.append({
                "origin_date": dates.iloc[origin],
                "target_date": dates.iloc[pos],
                "horizon": h,
                "actual_mw": float(full_sub[load_col].iloc[pos]),
                "pred_mw": r["pred_mw"],
            })
        origin += stride

    return pd.DataFrame(records)


def summarise_backtest(bt: pd.DataFrame) -> pd.DataFrame:
    if bt.empty:
        return pd.DataFrame()
    rows = []
    for h, grp in bt.groupby("horizon"):
        if len(grp) < 2:
            continue
        rows.append({
            "horizon": int(h),
            "n": len(grp),
            "r2": float(r2_score(grp["actual_mw"], grp["pred_mw"])),
            "rmse": float(np.sqrt(mean_squared_error(grp["actual_mw"], grp["pred_mw"]))),
            "mae": float(mean_absolute_error(grp["actual_mw"], grp["pred_mw"])),
            "mape_pct": float(np.mean(
                np.abs((grp["actual_mw"] - grp["pred_mw"]) /
                       grp["actual_mw"].replace(0, np.nan)).dropna()) * 100),
        })
    return pd.DataFrame(rows)


# =============================================================================
# Step 8: Model architectures
# =============================================================================

def build_model(model_name: str, window: int, n_features: int) -> Model:
    l2 = regularizers.l2(1e-4)
    inp = Input(shape=(window, n_features))

    if model_name == "BiLSTM":
        x = Bidirectional(LSTM(32, return_sequences=True,  kernel_regularizer=l2, recurrent_regularizer=l2))(inp)
        x = Dropout(0.3)(x)
        x = Bidirectional(LSTM(16, return_sequences=False, kernel_regularizer=l2, recurrent_regularizer=l2))(x)
        x = Dropout(0.3)(x)

    elif model_name == "BiGRU":
        x = Bidirectional(GRU(32, return_sequences=True,  kernel_regularizer=l2, recurrent_regularizer=l2))(inp)
        x = Dropout(0.3)(x)
        x = Bidirectional(GRU(16, return_sequences=False, kernel_regularizer=l2, recurrent_regularizer=l2))(x)
        x = Dropout(0.3)(x)

    elif model_name == "BiLSTM+BiGRU":
        x = Bidirectional(LSTM(32, return_sequences=True,  kernel_regularizer=l2, recurrent_regularizer=l2))(inp)
        x = Dropout(0.3)(x)
        x = Bidirectional(GRU(16,  return_sequences=False, kernel_regularizer=l2, recurrent_regularizer=l2))(x)
        x = Dropout(0.3)(x)

    elif model_name == "BiGRU+BiLSTM":
        x = Bidirectional(GRU(32,  return_sequences=True,  kernel_regularizer=l2, recurrent_regularizer=l2))(inp)
        x = Dropout(0.3)(x)
        x = Bidirectional(LSTM(16, return_sequences=False, kernel_regularizer=l2, recurrent_regularizer=l2))(x)
        x = Dropout(0.3)(x)

    elif model_name == "CNN+BiLSTM+BiGRU":
        x = Conv1D(32, kernel_size=3, activation="relu", padding="same", kernel_regularizer=l2)(inp)
        x = MaxPooling1D(pool_size=2)(x)
        x = Bidirectional(LSTM(32, return_sequences=True,  kernel_regularizer=l2, recurrent_regularizer=l2))(x)
        x = Dropout(0.3)(x)
        x = Bidirectional(GRU(16,  return_sequences=False, kernel_regularizer=l2, recurrent_regularizer=l2))(x)
        x = Dropout(0.3)(x)

    elif model_name == "CNN+BiGRU+BiLSTM":
        x = Conv1D(32, kernel_size=3, activation="relu", padding="same", kernel_regularizer=l2)(inp)
        x = MaxPooling1D(pool_size=2)(x)
        x = Bidirectional(GRU(32,  return_sequences=True,  kernel_regularizer=l2, recurrent_regularizer=l2))(x)
        x = Dropout(0.3)(x)
        x = Bidirectional(LSTM(16, return_sequences=False, kernel_regularizer=l2, recurrent_regularizer=l2))(x)
        x = Dropout(0.3)(x)

    else:
        raise ValueError(f"Unknown model: {model_name}")

    x = Dense(16, activation="relu", kernel_regularizer=l2)(x)
    x = Dropout(0.2)(x)
    out = Dense(1, activation="linear")(x)
    m = Model(inp, out)
    m.compile(optimizer=Adam(learning_rate=1e-3), loss="mse", metrics=["mae"])
    return m


# =============================================================================
# Step 9: Train one model (explicit chronological validation set)
# =============================================================================

def train_one_model(model_name: str, arr: Arrays, seed: int = SEED):
    """Train one architecture with one seed. `seed` is exposed so the caller
    can repeat the run and report mean +/- std instead of a single number that
    might be a bad initialisation."""
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model = build_model(model_name, arr.X_fit.shape[1], arr.n_features)
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=7, min_lr=1e-6),
    ]
    model.fit(
        arr.X_fit, arr.y_fit,
        validation_data=(arr.X_val, arr.y_val),   # explicit, chronological
        epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=False,
        callbacks=callbacks, verbose=0,
    )

    inv = lambda a: inverse_target(arr.scaler, a, arr.n_features)
    y_fit_act,  y_fit_pred  = inv(arr.y_fit),  inv(model.predict(arr.X_fit,  verbose=0).ravel())
    y_val_act,  y_val_pred  = inv(arr.y_val),  inv(model.predict(arr.X_val,  verbose=0).ravel())
    y_test_act, y_test_pred = inv(arr.y_test), inv(model.predict(arr.X_test, verbose=0).ravel())

    metrics = {"model": model_name, "seed": seed}
    metrics.update(masked_metrics(y_fit_act,  y_fit_pred,  arr.fit_eval_mask,  "train"))
    metrics.update(masked_metrics(y_val_act,  y_val_pred,  arr.val_eval_mask,  "val"))
    metrics.update(masked_metrics(y_test_act, y_test_pred, arr.test_eval_mask, "test"))

    preds = {
        "train": (y_fit_act, y_fit_pred),
        "val":   (y_val_act, y_val_pred),
        "test":  (y_test_act, y_test_pred),
    }
    return metrics, preds, model


def baseline_metrics(full: pd.DataFrame, load_col: str, arr: Arrays) -> list[dict]:
    """Persistence and seasonal-naive scored on exactly the same masked days."""
    test_dates = pd.to_datetime(arr.test_dates)
    s = full.set_index("Date")[load_col]
    actual = s.reindex(test_dates).values
    out = []
    for name, lag in (("Naive-Persistence", 1), ("Naive-Seasonal7", 7)):
        pred = s.reindex(test_dates - pd.Timedelta(days=lag)).values
        ok = arr.test_eval_mask & np.isfinite(pred) & np.isfinite(actual)
        m = {"model": name}
        m.update(masked_metrics(np.nan_to_num(actual), np.nan_to_num(pred), ok, "test"))
        m.update({"train_r2": np.nan, "val_r2": np.nan})
        out.append(m)
    return out


# =============================================================================
# Step 10: Plots
# =============================================================================

def plot_train_val_test(substation, model_name, metrics, arr, preds, output_dir):
    """Three panels: TRAIN(fit) / VALIDATION / TEST, all one-day-ahead."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    panels = [
        ("train", arr.fit_dates,  "TRAIN (fit block)"),
        ("val",   arr.val_dates,  "VALIDATION (chronological tail of train)"),
        ("test",  arr.test_dates, f"TEST (>= {TEST_START_YEAR}), one-day-ahead"),
    ]
    for ax, (key, dts, label) in zip(axes, panels):
        act, pred = preds[key]
        dt = pd.to_datetime(dts)
        n = min(len(dt), len(act))
        ax.plot(dt[:n], act[:n],  label="Actual",    linewidth=1.2, color="steelblue")
        ax.plot(dt[:n], pred[:n], label="Predicted", linewidth=1.2, color="tomato",
                linestyle="--", alpha=0.85)
        ax.set_title(
            f"{substation} | {model_name} - {label}   "
            f"(R2={metrics[f'{key}_r2']:.4f}, RMSE={metrics[f'{key}_rmse']:.2f}, "
            f"MAE={metrics[f'{key}_mae']:.2f}, n={metrics[f'{key}_n']})",
            fontsize=10,
        )
        ax.set_ylabel("Load (MW)")
        ax.legend(fontsize=9)
        ax.tick_params(axis="x", rotation=20, labelsize=8)
    axes[-1].set_xlabel("Date")
    fig.suptitle(f"{substation} - {model_name}: Actual vs Predicted",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    safe = model_name.replace("+", "_").replace(" ", "_")
    fig.savefig(output_dir / f"{substation.replace(' ', '_')}_{safe}_train_val_test.png", dpi=120)
    plt.close(fig)


def plot_r2_comparison(substation, all_metrics, output_dir):
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(MODEL_NAMES))
    width = 0.27
    tr = [m["train_r2"] for m in all_metrics]
    va = [m["val_r2"] for m in all_metrics]
    te = [m["test_r2"] for m in all_metrics]

    te_err = [m.get("test_r2_std", 0.0) for m in all_metrics]
    va_err = [m.get("val_r2_std", 0.0) for m in all_metrics]
    b1 = ax.bar(x - width, tr, width, label="Train R2", color="steelblue", alpha=0.85)
    b2 = ax.bar(x,         va, width, label="Val R2",   color="seagreen",  alpha=0.85,
                yerr=va_err, capsize=3, error_kw={"linewidth": 0.8})
    b3 = ax.bar(x + width, te, width, label="Test R2",  color="tomato",    alpha=0.85,
                yerr=te_err, capsize=3, error_kw={"linewidth": 0.8})

    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_NAMES, rotation=18, ha="right", fontsize=9)
    ax.set_ylabel("R2")
    lo = min([v for v in tr + va + te if np.isfinite(v)] + [0])
    ax.set_ylim(min(0, lo - 0.05), 1.05)
    ax.axhline(0, color="black", linewidth=0.6, linestyle="--")
    ax.set_title(f"{substation} - R2 by split (model selection uses VAL only)")
    ax.legend()
    for bars in (b1, b2, b3):
        for bar in bars:
            h = bar.get_height()
            if np.isfinite(h):
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                        f"{h:.3f}", ha="center", va="bottom", fontsize=6.5)
    fig.tight_layout()
    fig.savefig(output_dir / f"{substation.replace(' ', '_')}_r2_comparison.png", dpi=120)
    plt.close(fig)


def plot_rain_r2(full, substation, arr, all_preds, output_dir):
    """Rolling-window rainfall vs rolling R2 over the test period."""
    if "Rain_actual_mm" not in full.columns:
        return
    te_dt = pd.to_datetime(arr.test_dates)
    rain_series = full.set_index("Date")["Rain_actual_mm"].reindex(te_dt)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(MODEL_NAMES)))
    for i, model_name in enumerate(MODEL_NAMES):
        act, pred = all_preds[model_name]["test"]
        r2_vals, rain_vals = [], []
        for j in range(WINDOW, len(te_dt)):
            seg_mask = arr.test_eval_mask[j - WINDOW:j]
            seg_act = act[j - WINDOW:j][seg_mask]
            seg_pred = pred[j - WINDOW:j][seg_mask]
            if len(seg_act) >= 3 and seg_act.std() > 0:
                r2_vals.append(r2_score(seg_act, seg_pred))
                rv = rain_series.iloc[j - WINDOW:j].mean()
                rain_vals.append(float(rv) if pd.notna(rv) else 0.0)
        if rain_vals:
            ax.scatter(rain_vals, r2_vals, label=model_name, alpha=0.55, s=18, color=colors[i])
            z = np.polyfit(rain_vals, r2_vals, 1)
            xline = np.linspace(min(rain_vals), max(rain_vals), 100)
            ax.plot(xline, np.poly1d(z)(xline), color=colors[i], linewidth=1.2, alpha=0.85)

    ax.set_xlabel(f"Mean rainfall in {WINDOW}-day window (mm)")
    ax.set_ylabel(f"Rolling {WINDOW}-day R2")
    ax.set_title(f"{substation} - Rain vs R2 (all 6 models, test period)")
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / f"{substation.replace(' ', '_')}_rain_r2.png", dpi=120)
    plt.close(fig)


def plot_summary_table(substation, all_metrics, baselines, best_name, output_dir):
    col_labels = ["Model", "Train R2", "Val R2", "Test R2", "Test RMSE", "Test MAE", "Test n"]
    rows, source = [], list(all_metrics) + list(baselines)
    for m in source:
        fmt = lambda v: f"{v:.4f}" if isinstance(v, float) and np.isfinite(v) else "-"
        rows.append([
            m["model"], fmt(m.get("train_r2", np.nan)), fmt(m.get("val_r2", np.nan)),
            fmt(m.get("test_r2", np.nan)),
            f"{m['test_rmse']:.2f}" if np.isfinite(m.get("test_rmse", np.nan)) else "-",
            f"{m['test_mae']:.2f}" if np.isfinite(m.get("test_mae", np.nan)) else "-",
            str(m.get("test_n", "-")),
        ])

    cell_colors = []
    for i, m in enumerate(source):
        if m["model"] == best_name:
            cell_colors.append(["#c6efce"] * len(col_labels))
        elif m["model"] in BASELINE_NAMES:
            cell_colors.append(["#fff2cc"] * len(col_labels))
        else:
            cell_colors.append(["#f7f7f7" if i % 2 == 0 else "white"] * len(col_labels))

    fig, ax = plt.subplots(figsize=(13, 0.55 * len(rows) + 1.8))
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=col_labels, cellColours=cell_colors,
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)
    tbl.scale(1, 1.55)
    for j in range(len(col_labels)):
        tbl[(0, j)].set_facecolor("#4472C4")
        tbl[(0, j)].get_text().set_color("white")
        tbl[(0, j)].get_text().set_fontweight("bold")
    ax.set_title(
        f"{substation} - metrics summary\n"
        f"[green = selected on VALIDATION R2; yellow = naive baselines; "
        f"test days with imputed load are excluded]",
        fontsize=11, fontweight="bold", pad=14,
    )
    fig.tight_layout()
    fig.savefig(output_dir / f"{substation.replace(' ', '_')}_summary_table.png",
                dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_backtest(substation, bt_summary, output_dir):
    if bt_summary.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    axes[0].plot(bt_summary["horizon"], bt_summary["r2"], marker="o", color="tomato")
    axes[0].set_xlabel("Forecast horizon (days ahead)")
    axes[0].set_ylabel("R2")
    axes[0].axhline(0, color="black", linewidth=0.5, linestyle="--")
    axes[0].set_title("Recursive forecast skill vs horizon")
    axes[1].plot(bt_summary["horizon"], bt_summary["rmse"], marker="o", color="steelblue", label="RMSE")
    axes[1].plot(bt_summary["horizon"], bt_summary["mae"], marker="s", color="seagreen", label="MAE")
    axes[1].set_xlabel("Forecast horizon (days ahead)")
    axes[1].set_ylabel("MW")
    axes[1].set_title("Error growth vs horizon")
    axes[1].legend()
    fig.suptitle(f"{substation} - rolling-origin recursive backtest (test period)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(output_dir / f"{substation.replace(' ', '_')}_recursive_backtest.png", dpi=120)
    plt.close(fig)


def plot_forecast_band(substation, best_name, central, low, high, full_sub,
                       load_col, output_dir):
    hist = full_sub.tail(60)
    dates = [r["date"] for r in central]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(pd.to_datetime(hist["Date"]), hist[load_col], color="steelblue",
            linewidth=1.3, label="Observed (last 60 d)")
    ax.plot(dates, [r["pred_mw"] for r in central], color="tomato", marker="o",
            markersize=3, linewidth=1.5, label="Forecast (central weather)")
    if low and high:
        lo = [min(a["pred_mw"], b["pred_mw"]) for a, b in zip(low, high)]
        hi = [max(a["pred_mw"], b["pred_mw"]) for a, b in zip(low, high)]
        ax.fill_between(dates, lo, hi, color="tomato", alpha=0.18,
                        label="Weather sensitivity (+/-1 sigma climatology)")
    ax.axvline(pd.to_datetime(full_sub["Date"].iloc[-1]), color="grey",
               linestyle=":", linewidth=1)
    ax.set_ylabel("Load (MW)")
    ax.set_xlabel("Date")
    ax.set_title(f"{substation} - {len(dates)}-day recursive forecast ({best_name}), "
                 f"weather mode = {WEATHER_MODE}")
    ax.legend(fontsize=9)
    ax.tick_params(axis="x", rotation=20, labelsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / f"{substation.replace(' ', '_')}_forecast_band.png", dpi=120)
    plt.close(fig)


# =============================================================================
# Step 11: Run all models for one substation
# =============================================================================

def run_substation(df_raw: pd.DataFrame, substation: str, output_dir: Path) -> dict:
    print(f"\n{'=' * 70}\nSubstation: {substation}\n{'=' * 70}")
    load_col = f"{substation}_Load_MW"
    if load_col not in df_raw.columns:
        print(f"  SKIPPING: {load_col} not found.")
        return {"metrics": [], "baselines": []}

    raw_sub = build_substation_frame(df_raw, substation)
    full, pp_train, feature_cols, train_end_date = process_substation_frame(raw_sub, substation)
    arr = prepare_arrays(full, load_col, feature_cols, train_end_date)

    print(f"  Preprocessing fitted on {pd.Timestamp(pp_train.fitted_on_dates[0]).date()} "
          f"-> {pd.Timestamp(pp_train.fitted_on_dates[1]).date()} (train only)")
    print(f"  Fit {arr.X_fit.shape} | Val {arr.X_val.shape} | Test {arr.X_test.shape}")
    print(f"  Test days scored: {int(arr.test_eval_mask.sum())} of {len(arr.test_eval_mask)} "
          f"({len(arr.test_eval_mask) - int(arr.test_eval_mask.sum())} imputed, excluded)")

    all_metrics, all_preds, all_models = [], {}, {}
    run_records = []
    seeds = [SEED + i for i in range(N_RUNS)]

    for model_name in MODEL_NAMES:
        print(f"  [{model_name}] training {N_RUNS} run(s)...", end=" ", flush=True)
        runs, best_seed_val, keep = [], -np.inf, None
        for seed in seeds:
            m, preds, model = train_one_model(model_name, arr, seed=seed)
            m["substation"] = substation
            runs.append(m)
            run_records.append(dict(m))
            v = m["val_r2"] if np.isfinite(m["val_r2"]) else -np.inf
            if v > best_seed_val:
                best_seed_val, keep = v, (preds, model, seed)

        # Aggregate across seeds. The reported figure is the MEAN over seeds;
        # the std tells the reader whether a difference between two
        # architectures is real or just initialisation noise.
        agg = {"model": model_name, "substation": substation, "n_runs": N_RUNS}
        for key in ("train_r2", "val_r2", "test_r2", "test_rmse", "test_mae",
                    "train_rmse", "train_mae", "val_rmse", "val_mae"):
            vals = np.array([r.get(key, np.nan) for r in runs], dtype=float)
            agg[key] = float(np.nanmean(vals)) if np.isfinite(vals).any() else np.nan
            agg[f"{key}_std"] = float(np.nanstd(vals)) if N_RUNS > 1 else 0.0
        for key in ("train_n", "val_n", "test_n"):
            agg[key] = runs[0].get(key, 0)
        agg["test_r2_min"] = float(np.nanmin([r["test_r2"] for r in runs]))
        agg["test_r2_max"] = float(np.nanmax([r["test_r2"] for r in runs]))
        agg["best_seed"] = keep[2]

        all_metrics.append(agg)
        all_preds[model_name] = keep[0]     # predictions of the best-val seed
        all_models[model_name] = keep[1]
        spread = f" +/-{agg['test_r2_std']:.3f}" if N_RUNS > 1 else ""
        print(f"Val R2={agg['val_r2']:.3f}  Test R2={agg['test_r2']:.3f}{spread}")
        if N_RUNS > 1 and agg["test_r2_max"] - agg["test_r2_min"] > 0.15:
            print(f"      ! unstable across seeds: Test R2 ranged "
                  f"{agg['test_r2_min']:.3f} to {agg['test_r2_max']:.3f} "
                  f"- report the mean, not a single run")
        plot_train_val_test(substation, model_name, agg, arr, keep[0], output_dir)

    bl = baseline_metrics(full, load_col, arr)
    for b in bl:
        b["substation"] = substation
    print("  Baselines:  " + "  ".join(
        f"{b['model']} Test R2={b['test_r2']:.3f}" for b in bl))

    # ---- model selection on VALIDATION, never on test ----
    val_scores = [m["val_r2"] if np.isfinite(m["val_r2"]) else -np.inf for m in all_metrics]
    best_idx = int(np.argmax(val_scores))
    best_name = all_metrics[best_idx]["model"]
    best_model = all_models[best_name]
    print(f"  Selected by validation R2: {best_name} "
          f"(Val R2={all_metrics[best_idx]['val_r2']:.4f}, "
          f"held-out Test R2={all_metrics[best_idx]['test_r2']:.4f})")

    plot_r2_comparison(substation, all_metrics, output_dir)
    plot_rain_r2(full, substation, arr, all_preds, output_dir)
    plot_summary_table(substation, all_metrics, bl, best_name, output_dir)

    # ---- recursive multi-step backtest ----
    bt_summary = pd.DataFrame()
    if RUN_BACKTEST:
        print(f"  Rolling-origin backtest (horizon={FORECAST_DAYS}, stride={BACKTEST_STRIDE})...",
              end=" ", flush=True)
        test_start = pd.to_datetime(arr.test_dates[0])
        train_festive_mean = _train_festive_means(full, load_col, arr)
        bt = rolling_origin_backtest(full, load_col, feature_cols, arr.scaler,
                                     best_model, arr.n_features, test_start,
                                     FORECAST_DAYS, BACKTEST_STRIDE,
                                     train_festive_mean, df_raw)
        bt_summary = summarise_backtest(bt)
        if not bt_summary.empty:
            bt_summary.insert(0, "substation", substation)
            bt_summary.insert(1, "model", best_name)
            bt_summary.to_csv(
                output_dir / f"{substation.replace(' ', '_')}_recursive_backtest.csv",
                index=False)
            plot_backtest(substation, bt_summary, output_dir)
            d1 = bt_summary.iloc[0]
            dn = bt_summary.iloc[-1]
            print(f"day+1 R2={d1['r2']:.3f} RMSE={d1['rmse']:.2f} -> "
                  f"day+{int(dn['horizon'])} R2={dn['r2']:.3f} RMSE={dn['rmse']:.2f}")
        else:
            print("not enough test data.")

    # ---- operational forecast with real future weather ----
    # Fitted on the RAW, unshifted history: at forecast time every observed day
    # is genuinely in the past, so using all of it is not leakage.
    pp_operational = fit_preprocessor(raw_sub, load_col)
    train_festive_mean = _train_festive_means(full, load_col, arr)

    central = forecast_next_n_days(df_raw, full, load_col, feature_cols, arr.scaler,
                                   best_model, arr.n_features, pp_operational,
                                   train_festive_mean, FORECAST_DAYS)
    low = forecast_next_n_days(df_raw, full, load_col, feature_cols, arr.scaler,
                               best_model, arr.n_features, pp_operational,
                               train_festive_mean, FORECAST_DAYS,
                               weather_mode="climatology", sigma=-1.0)
    high = forecast_next_n_days(df_raw, full, load_col, feature_cols, arr.scaler,
                                best_model, arr.n_features, pp_operational,
                                train_festive_mean, FORECAST_DAYS,
                                weather_mode="climatology", sigma=+1.0)

    forecast_df = pd.DataFrame([
        {
            "substation": substation,
            "model": best_name,
            "date": r["date"].date(),
            "day_ahead": i + 1,
            "predicted_mw": round(r["pred_mw"], 2),
            "predicted_mw_cool_wet": round(low[i]["pred_mw"], 2),
            "predicted_mw_hot_dry": round(high[i]["pred_mw"], 2),
            "weather_source": r["weather_source"],
            "holiday_type": r["holiday_type"],
            "context": r["window_desc"],
        }
        for i, r in enumerate(central)
    ])
    forecast_df.to_csv(
        output_dir / f"{substation.replace(' ', '_')}_{FORECAST_DAYS}day_forecast.csv",
        index=False)
    plot_forecast_band(substation, best_name, central, low, high, full, load_col, output_dir)

    last_row = full.iloc[-1]
    lines = [
        f"{FORECAST_DAYS}-day forecast for {substation} "
        f"({central[0]['date'].date()} -> {central[-1]['date'].date()})",
        f"Model selected on VALIDATION R2: {best_name} "
        f"(Val R2={all_metrics[best_idx]['val_r2']:.4f}, "
        f"held-out Test R2={all_metrics[best_idx]['test_r2']:.4f})",
        f"Weather mode: {WEATHER_MODE}   Exog alignment: {EXOG_ALIGNMENT}",
        f"Last known 7-day mean: {last_row.get('roll_mean_7', float('nan')):.1f} MW "
        f"| lag-1: {last_row.get('lag_1', float('nan')):.1f} MW",
        "",
    ]
    if not bt_summary.empty:
        lines.append("Expected recursive error at this horizon (from the backtest):")
        for _, r in bt_summary.iterrows():
            lines.append(f"    day+{int(r['horizon']):>2}: RMSE={r['rmse']:.1f} MW, "
                         f"MAE={r['mae']:.1f} MW, R2={r['r2']:.3f}")
        lines.append("")
    for i, r in enumerate(central):
        lines.append(
            f"  Day {i + 1:>2} ({r['date'].date()}): {r['pred_mw']:.1f} MW "
            f"[{low[i]['pred_mw']:.1f} - {high[i]['pred_mw']:.1f}] "
            f"weather={r['weather_source']}, holiday={r['holiday_type']}"
        )
    text = "\n".join(lines)
    print("\n" + text)
    (output_dir / f"{substation.replace(' ', '_')}_{FORECAST_DAYS}day_forecast.txt"
     ).write_text(text, encoding="utf-8")

    return {"metrics": all_metrics, "baselines": bl, "runs": run_records,
            "backtest": bt_summary if not bt_summary.empty else None}


def _train_festive_means(full: pd.DataFrame, load_col: str, arr: Arrays) -> dict:
    """Mean load per festive holiday type, computed on TRAINING days only."""
    if "Holiday_type" not in full.columns:
        return {}
    train_end = pd.to_datetime(arr.val_dates[-1]) if len(arr.val_dates) else None
    block = full if train_end is None else full[pd.to_datetime(full["Date"]) <= train_end]
    if "_load_imputed" in block.columns:
        block = block.loc[~block["_load_imputed"].astype(bool)]
    out = {}
    for h in FESTIVE_HOLIDAY_TYPES:
        vals = block.loc[block["Holiday_type"] == h, load_col].dropna()
        if len(vals):
            out[h] = float(vals.mean())
    return out


# =============================================================================
# Step 12: Global summaries
# =============================================================================

def plot_global_summary_table(records: list[dict], output_dir: Path):
    df_sum = pd.DataFrame(records)
    df_sum = df_sum[df_sum["model"].isin(MODEL_NAMES)]
    pivot = df_sum.pivot(index="substation", columns="model", values="test_r2")
    pivot = pivot[[m for m in MODEL_NAMES if m in pivot.columns]].sort_index()

    n_rows, n_cols = pivot.shape
    fig_h = max(6, 0.42 * n_rows + 2.0)
    fig, ax = plt.subplots(figsize=(14, fig_h))
    ax.axis("off")

    cell_text = [[f"{v:.4f}" if pd.notna(v) else "-" for v in row] for row in pivot.values]
    vmin, vmax = np.nanmin(pivot.values), np.nanmax(pivot.values)
    cmap = plt.cm.RdYlGn
    cell_colors = []
    for row in pivot.values:
        cc = []
        for v in row:
            if pd.isna(v):
                cc.append("#e0e0e0")
            else:
                cc.append(cmap((v - vmin) / max(vmax - vmin, 1e-9)))
        cell_colors.append(cc)

    tbl = ax.table(cellText=cell_text, rowLabels=list(pivot.index),
                   colLabels=list(pivot.columns), cellColours=cell_colors,
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 1.5)
    for j in range(n_cols):
        tbl[(0, j)].set_facecolor("#4472C4")
        tbl[(0, j)].get_text().set_color("white")
        tbl[(0, j)].get_text().set_fontweight("bold")
    ax.set_title("Global summary - held-out Test R2 by substation x model\n"
                 "(green = higher; selection was done on validation, not on these numbers)",
                 fontsize=12, fontweight="bold", pad=16)
    fig.tight_layout()
    fig.savefig(output_dir / "global_summary_test_r2.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("Saved global_summary_test_r2.png")


def plot_r2_distribution(summary: pd.DataFrame, output_dir: Path):
    """
    Transparent reporting of the spread, not just the headline mean.

    Left  : boxplot of held-out Test R2 per architecture across substations,
            with every substation drawn as a point on top.
    Right : per-substation Test R2 of the selected (validation-chosen) model,
            sorted, with the mean marked - so weak substations are visible
            rather than buried inside an average.
    """
    dl = summary[summary["model"].isin(MODEL_NAMES)]
    if dl.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6),
                             gridspec_kw={"width_ratios": [1.05, 1.0]})

    # -- Left: distribution per model ----------------------------------------
    ax = axes[0]
    data = [dl.loc[dl["model"] == m, "test_r2"].dropna().values for m in MODEL_NAMES]
    # matplotlib renamed boxplot's `labels` to `tick_labels` in 3.9 and removed
    # the old name in 3.11, so try the new spelling first and fall back.
    box_kw = dict(showmeans=True, patch_artist=True,
                  medianprops={"color": "black", "linewidth": 1.4},
                  meanprops={"marker": "D", "markerfacecolor": "white",
                             "markeredgecolor": "black", "markersize": 5})
    try:
        bp = ax.boxplot(data, tick_labels=MODEL_NAMES, **box_kw)
    except TypeError:
        bp = ax.boxplot(data, labels=MODEL_NAMES, **box_kw)
    for patch in bp["boxes"]:
        patch.set_facecolor("#cfe2f3")
        patch.set_alpha(0.9)
    rng = np.random.default_rng(SEED)
    for i, vals in enumerate(data, start=1):
        if len(vals):
            ax.scatter(np.full(len(vals), i) + rng.uniform(-0.11, 0.11, len(vals)),
                       vals, s=16, color="tomato", alpha=0.65, zorder=3)
    ax.set_xticklabels(MODEL_NAMES, rotation=18, ha="right", fontsize=8.5)
    ax.set_ylabel("Held-out Test R2")
    ax.axhline(dl["test_r2"].mean(), color="grey", linestyle="--", linewidth=1,
               label=f"overall mean = {dl['test_r2'].mean():.3f}")
    ax.set_title("Distribution of Test R2 across substations\n"
                 "(box = IQR, diamond = mean, dots = individual substations)",
                 fontsize=10.5)
    ax.legend(fontsize=8, loc="lower left")

    # -- Right: per-substation, selected model -------------------------------
    ax = axes[1]
    rows = []
    for sub, grp in dl.groupby("substation"):
        best = grp.loc[grp["val_r2"].idxmax()]
        rows.append({"substation": sub, "model": best["model"],
                     "test_r2": best["test_r2"],
                     "err": best.get("test_r2_std", 0.0)})
    per_sub = pd.DataFrame(rows).sort_values("test_r2")
    ypos = np.arange(len(per_sub))
    # Absolute colour scale (0 -> 1), not relative to this run's min/max, so a
    # substation at 0.85 is never coloured "red" just for being the weakest.
    colors = plt.cm.RdYlGn(np.clip(per_sub["test_r2"].values, 0.0, 1.0))
    ax.barh(ypos, per_sub["test_r2"], color=colors, edgecolor="grey", linewidth=0.4,
            xerr=per_sub["err"], capsize=2, error_kw={"linewidth": 0.7})
    ax.set_yticks(ypos)
    ax.set_yticklabels(
        [f"{r.substation}  ({r.model})" for r in per_sub.itertuples()], fontsize=8)
    mean_r2 = per_sub["test_r2"].mean()
    ax.axvline(mean_r2, color="black", linestyle="--", linewidth=1,
               label=f"mean = {mean_r2:.3f}")
    ax.set_xlabel("Held-out Test R2 of the validation-selected model")
    ax.set_title("Per-substation performance, weakest first", fontsize=10.5)
    ax.legend(fontsize=8, loc="lower right")

    fig.suptitle("Test R2: central tendency AND spread across the substation fleet",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output_dir / "test_r2_distribution.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("Saved test_r2_distribution.png")


def save_distribution_stats(summary: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """
    The numbers to quote in the results chapter alongside the mean: median,
    std, IQR, min/max and which substation sits at each extreme.
    """
    dl = summary[summary["model"].isin(MODEL_NAMES)]
    if dl.empty:
        return pd.DataFrame()

    rows = []
    for m in MODEL_NAMES:
        v = dl.loc[dl["model"] == m, "test_r2"].dropna()
        if v.empty:
            continue
        rows.append({
            "model": m, "n_substations": len(v),
            "mean": v.mean(), "median": v.median(), "std": v.std(),
            "min": v.min(), "q1": v.quantile(0.25),
            "q3": v.quantile(0.75), "max": v.max(),
            "iqr": v.quantile(0.75) - v.quantile(0.25),
        })
    stats = pd.DataFrame(rows)

    sel = []
    for sub, grp in dl.groupby("substation"):
        best = grp.loc[grp["val_r2"].idxmax()]
        sel.append({"substation": sub, "model": best["model"],
                    "val_r2": best["val_r2"], "test_r2": best["test_r2"]})
    sel = pd.DataFrame(sel).sort_values("test_r2")
    sel.to_csv(output_dir / "selected_model_per_substation.csv", index=False)
    stats.to_csv(output_dir / "test_r2_distribution_stats.csv", index=False)

    s = sel["test_r2"]
    lines = [
        "",
        "=" * 96,
        "DISTRIBUTION OF HELD-OUT TEST R2 ACROSS SUBSTATIONS "
        "(validation-selected model per substation)",
        "=" * 96,
        f"  n substations : {len(s)}",
        f"  mean          : {s.mean():.4f}",
        f"  median        : {s.median():.4f}",
        f"  std           : {s.std():.4f}",
        f"  IQR           : {s.quantile(0.25):.4f} - {s.quantile(0.75):.4f}",
        f"  range         : {s.min():.4f} ({sel.iloc[0]['substation']}) "
        f"to {s.max():.4f} ({sel.iloc[-1]['substation']})",
        "",
        "  Weakest three substations (report these, do not drop them):",
    ]
    for _, r in sel.head(3).iterrows():
        lines.append(f"    {r['substation']:<26} {r['model']:<20} Test R2={r['test_r2']:.4f}")
    lines.append("")
    lines.append("  Per-model spread across substations:")
    for _, r in stats.iterrows():
        lines.append(
            f"    {r['model']:<22} mean={r['mean']:.4f}  median={r['median']:.4f}  "
            f"std={r['std']:.4f}  min={r['min']:.4f}  max={r['max']:.4f}")
    text = "\n".join(lines)
    (output_dir / "test_r2_distribution.txt").write_text(text, encoding="utf-8")
    print(text)
    return stats


def flag_unstable_cells(summary: pd.DataFrame, drop_threshold: float = 0.15) -> pd.DataFrame:
    """
    Find substation x model cells whose held-out Test R2 sits far below the
    other five architectures at the SAME substation. A single cell at 0.17
    among five results around 0.75 is almost always a bad initialisation, not
    a property of the architecture - but that has to be demonstrated, not
    asserted. This only flags candidates; run_stability_probe() supplies the
    evidence.
    """
    dl = summary[summary["model"].isin(MODEL_NAMES)]
    rows = []
    for sub, grp in dl.groupby("substation"):
        med = grp["test_r2"].median()
        for _, r in grp.iterrows():
            drop = med - r["test_r2"]
            if drop > drop_threshold:
                rows.append({"substation": sub, "model": r["model"],
                             "test_r2": r["test_r2"], "substation_median": med,
                             "drop_below_median": drop})
    return pd.DataFrame(rows).sort_values("drop_below_median", ascending=False) \
        if rows else pd.DataFrame()


def run_stability_probe(df_raw: pd.DataFrame, substation: str, model_name: str,
                        seeds: list, output_dir: Path) -> pd.DataFrame:
    """
    Re-train ONE architecture at ONE substation across several seeds.

    This is the cheap, targeted alternative to running every seed for every
    cell: 18 x 6 x 3 = 324 trainings to make one anomaly explainable, versus
    5 trainings here. It answers the only question the anomaly actually
    raises - "is this optimisation instability or a real result?" - and
    produces a figure and a one-line verdict you can put in the thesis.
    """
    print(f"\n{'-' * 70}")
    print(f"STABILITY PROBE: {substation} / {model_name} over {len(seeds)} seeds")
    print(f"{'-' * 70}")

    load_col = f"{substation}_Load_MW"
    raw_sub = build_substation_frame(df_raw, substation)
    full, _, feature_cols, train_end_date = process_substation_frame(raw_sub, substation)
    arr = prepare_arrays(full, load_col, feature_cols, train_end_date)

    rows = []
    for seed in seeds:
        m, _, _ = train_one_model(model_name, arr, seed=seed)
        rows.append({"substation": substation, "model": model_name, "seed": seed,
                     "train_r2": m["train_r2"], "val_r2": m["val_r2"],
                     "test_r2": m["test_r2"], "test_rmse": m["test_rmse"],
                     "test_mae": m["test_mae"]})
        print(f"  seed {seed:<5} Val R2={m['val_r2']:.4f}  Test R2={m['test_r2']:.4f}  "
              f"RMSE={m['test_rmse']:.2f}")

    probe = pd.DataFrame(rows)
    te = probe["test_r2"]
    spread = float(te.max() - te.min())

    if spread > 0.15:
        verdict = (
            f"OPTIMISATION INSTABILITY. Across {len(seeds)} seeds the held-out "
            f"Test R2 ranged {te.min():.4f} to {te.max():.4f} (spread {spread:.4f}, "
            f"median {te.median():.4f}). The low value in the main run is a bad "
            f"initialisation, not a property of this architecture. Report the "
            f"median or note the instability; do not present the single low run "
            f"as this model's performance at {substation}.")
    elif te.max() < 0.55:
        verdict = (
            f"CONSISTENTLY LOW, NOT UNSTABLE. All {len(seeds)} seeds landed between "
            f"{te.min():.4f} and {te.max():.4f} (spread {spread:.4f}). This is a "
            f"genuine result for this architecture at {substation}, not a seed "
            f"artefact - discuss it as a real limitation.")
    else:
        verdict = (
            f"STABLE. Seed-to-seed spread is only {spread:.4f} "
            f"(range {te.min():.4f} to {te.max():.4f}, median {te.median():.4f}). "
            f"The single-seed number in the main experiment is trustworthy.")

    safe = model_name.replace("+", "_").replace(" ", "_")
    stem = f"{substation.replace(' ', '_')}_{safe}_stability_probe"
    probe.to_csv(output_dir / f"{stem}.csv", index=False)

    lines = [
        f"STABILITY PROBE - {substation} / {model_name}",
        f"seeds tested: {', '.join(str(s) for s in seeds)}",
        "",
        f"  Test R2   min={te.min():.4f}  median={te.median():.4f}  "
        f"max={te.max():.4f}  std={te.std():.4f}",
        f"  Val  R2   min={probe['val_r2'].min():.4f}  "
        f"median={probe['val_r2'].median():.4f}  max={probe['val_r2'].max():.4f}",
        "",
        "VERDICT: " + verdict,
    ]
    (output_dir / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar([str(s) for s in probe["seed"]], probe["test_r2"],
           color="steelblue", alpha=0.85, edgecolor="grey", linewidth=0.5)
    ax.axhline(float(te.median()), color="black", linestyle="--", linewidth=1,
               label=f"median = {te.median():.3f}")
    ax.set_xlabel("Random seed")
    ax.set_ylabel("Held-out Test R2")
    ax.set_ylim(min(0, te.min() - 0.05), max(1.0, te.max() + 0.05))
    ax.set_title(f"{substation} / {model_name} - seed sensitivity "
                 f"(spread = {spread:.3f})", fontsize=11)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / f"{stem}.png", dpi=120)
    plt.close(fig)

    print(f"\n  VERDICT: {verdict}")
    print(f"  Saved {stem}.csv / .txt / .png")
    return probe


def save_model_ranking(summary: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """
    Rank the 6 models by MEAN VALIDATION R2 across substations. Test metrics
    are reported alongside but are NOT used for ranking - that is the whole
    point of holding them out.
    """
    dl = summary[summary["model"].isin(MODEL_NAMES)]
    agg_cols = ["train_r2", "val_r2", "test_r2", "test_rmse", "test_mae"]
    grouped = dl.groupby("model")[agg_cols].mean()
    grouped["test_r2_sd_across_subs"] = dl.groupby("model")["test_r2"].std()
    if "test_r2_std" in dl.columns:
        grouped["test_r2_sd_across_seeds"] = dl.groupby("model")["test_r2_std"].mean()
    grouped = grouped.sort_values("val_r2", ascending=False).reset_index()
    grouped.insert(0, "rank", range(1, len(grouped) + 1))
    grouped.to_csv(output_dir / "model_ranking.csv", index=False)

    lines = [
        "=" * 96,
        "OVERALL MODEL RANKING - mean across substations, SORTED BY VALIDATION R2 "
        f"(n_substations={dl['substation'].nunique()})",
        "=" * 96,
    ]
    for _, r in grouped.iterrows():
        seed_sd = r.get("test_r2_sd_across_seeds", float("nan"))
        seed_txt = f"  [seed sd={seed_sd:.3f}]" if np.isfinite(seed_sd) and seed_sd > 0 else ""
        lines.append(
            f"  #{int(r['rank'])}  {r['model']:<22} "
            f"Val R2={r['val_r2']:.4f}  Train R2={r['train_r2']:.4f}  |  "
            f"held-out Test R2={r['test_r2']:.4f} "
            f"(sd across substations={r['test_r2_sd_across_subs']:.3f})  "
            f"RMSE={r['test_rmse']:.2f}  MAE={r['test_mae']:.2f}{seed_txt}"
        )
    bl = summary[summary["model"].isin(BASELINE_NAMES)]
    if not bl.empty:
        lines.append("")
        lines.append("Naive baselines (mean held-out Test metrics):")
        for name, grp in bl.groupby("model"):
            lines.append(f"    {name:<22} Test R2={grp['test_r2'].mean():.4f}  "
                         f"RMSE={grp['test_rmse'].mean():.2f}  MAE={grp['test_mae'].mean():.2f}")
    if len(grouped) > 1:
        gap = grouped.iloc[0]["val_r2"] - grouped.iloc[1]["val_r2"]
        noise = grouped.get("test_r2_sd_across_seeds", pd.Series([0.0])).max()
        if np.isfinite(noise) and gap < noise:
            lines.append("")
            lines.append(
                f"  CAUTION: the top two models differ by only {gap:.4f} in mean "
                f"validation R2, which is smaller than the seed-to-seed spread "
                f"({noise:.4f}). Treat them as tied rather than claiming a winner.")
    lines.append("")
    lines.append(f"BEST OVERALL MODEL (by validation): {grouped.iloc[0]['model']} "
                 f"- mean Val R2={grouped.iloc[0]['val_r2']:.4f}, "
                 f"mean held-out Test R2={grouped.iloc[0]['test_r2']:.4f}")
    text = "\n".join(lines)
    (output_dir / "model_ranking.txt").write_text(text, encoding="utf-8")
    print("\n" + text)
    return grouped


# =============================================================================
# Main
# =============================================================================

def write_global_reports(summary: pd.DataFrame, all_records, output_dir: Path) -> None:
    """Every fleet-level artefact: heatmap, distribution figure, ranking, stats.

    Kept separate from training so it can be re-run on its own via
    --report-only. A plotting library change should never cost you a retrain.
    """
    plot_global_summary_table(all_records, output_dir)
    plot_r2_distribution(summary, output_dir)
    save_model_ranking(summary, output_dir)
    save_distribution_stats(summary, output_dir)


def report_flagged_cells(summary: pd.DataFrame, output_dir: Path, csv_name: str,
                         suggest_commands: bool = True) -> pd.DataFrame:
    flagged = flag_unstable_cells(summary)
    if flagged.empty:
        return flagged
    flagged.to_csv(output_dir / "flagged_outlier_cells.csv", index=False)
    print("\n" + "=" * 96)
    print("POSSIBLE OPTIMISATION ANOMALIES (Test R2 far below the other "
          "architectures at the same substation)")
    print("=" * 96)
    for _, r in flagged.iterrows():
        print(f"  {r['substation']:<26} {r['model']:<20} "
              f"Test R2={r['test_r2']:.4f}  vs substation median "
              f"{r['substation_median']:.4f}  (drop {r['drop_below_median']:.4f})")
    if suggest_commands:
        print("\n  These are cheap to explain without re-running everything. "
              "For each one:")
        for _, r in flagged.head(3).iterrows():
            print(f"    python {Path(__file__).name} --csv {csv_name} "
                  f"--probe-only --probe \"{r['substation']}:{r['model']}\"")
        print("  (or add --auto-probe to do it in the same run)")
    return flagged


def main():
    global FORECAST_DAYS, EXOG_ALIGNMENT, WEATHER_MODE, FESTIVE_BLEND
    global RUN_BACKTEST, BACKTEST_STRIDE, EPOCHS, OUTPUT_DIR, N_RUNS

    p = argparse.ArgumentParser(description="BPDB 6-model load forecasting (leakage-safe)")
    p.add_argument("--csv", default="BPDB_Dhaka_City_Substations_Page3.csv")
    p.add_argument("--substations", nargs="*", default=None)
    p.add_argument("--forecast-days", type=int, default=FORECAST_DAYS)
    p.add_argument("--output-dir", default=str(OUTPUT_DIR))
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--exog-alignment", choices=["target", "lagged"], default=EXOG_ALIGNMENT,
                   help="'target': the input window carries the predicted day's "
                        "calendar+weather (recommended). 'lagged': v1 behaviour.")
    p.add_argument("--weather-mode", choices=["forecast", "climatology", "persistence"],
                   default=WEATHER_MODE,
                   help="How future weather is obtained for the operational forecast.")
    p.add_argument("--festive-blend", action="store_true",
                   help="Re-enable the hand-tuned Eid/Puja window blending (off by default).")
    p.add_argument("--no-backtest", action="store_true",
                   help="Skip the rolling-origin recursive backtest.")
    p.add_argument("--backtest-stride", type=int, default=BACKTEST_STRIDE)
    p.add_argument("--runs", type=int, default=N_RUNS,
                   help="Seeds per architecture for the FULL experiment. Leave at "
                        "1 (deterministic, reproducible) unless you have runtime "
                        "to spare - use --probe instead to investigate a single "
                        "suspicious cell at a fraction of the cost.")
    p.add_argument("--probe", nargs="*", default=None, metavar="SUBSTATION:MODEL",
                   help="Targeted multi-seed stability probe, e.g. "
                        "--probe 'Agargaon:CNN+BiLSTM+BiGRU'. Repeatable.")
    p.add_argument("--probe-seeds", type=int, default=5,
                   help="Seeds per probe (default 5).")
    p.add_argument("--report-only", action="store_true",
                   help="Do NOT train. Rebuild the fleet-level figures, ranking "
                        "and distribution stats from the metrics_summary.csv "
                        "already sitting in --output-dir. Use this when a "
                        "plotting or reporting step failed after training "
                        "finished - your per-substation outputs are untouched.")
    p.add_argument("--probe-only", action="store_true",
                   help="Run only the probes and skip the full experiment.")
    p.add_argument("--auto-probe", action="store_true",
                   help="After the full run, automatically probe any cell whose "
                        "Test R2 falls more than 0.15 below the median of the "
                        "other architectures at the same substation.")
    args = p.parse_args()

    FORECAST_DAYS   = args.forecast_days
    EXOG_ALIGNMENT  = args.exog_alignment
    WEATHER_MODE    = args.weather_mode
    FESTIVE_BLEND   = args.festive_blend
    RUN_BACKTEST    = not args.no_backtest
    BACKTEST_STRIDE = args.backtest_stride
    N_RUNS          = max(1, args.runs)
    EPOCHS          = args.epochs
    OUTPUT_DIR      = Path(args.output_dir)
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    print("Configuration")
    print(f"  window={WINDOW}  horizon={FORECAST_DAYS}  epochs={EPOCHS}")
    print(f"  split: train Year<={TRAIN_END_YEAR} (last {VAL_FRACTION:.0%} = validation), "
          f"test Year>={TEST_START_YEAR}")
    print(f"  exog alignment={EXOG_ALIGNMENT}  weather mode={WEATHER_MODE}  "
          f"festive blend={FESTIVE_BLEND}  backtest={RUN_BACKTEST}  runs/model={N_RUNS}")

    if args.report_only:
        summary_path = OUTPUT_DIR / "metrics_summary.csv"
        if not summary_path.exists():
            print(f"--report-only needs {summary_path}, which does not exist. "
                  f"Point --output-dir at the folder from your training run.")
            return
        summary = pd.read_csv(summary_path)
        print(f"\nRebuilding reports from {summary_path} "
              f"({summary['substation'].nunique()} substations, no training).")
        write_global_reports(summary, summary.to_dict("records"), OUTPUT_DIR)
        report_flagged_cells(summary, OUTPUT_DIR, args.csv, suggest_commands=True)
        print(f"\nReports written to: {OUTPUT_DIR.resolve()}")
        return

    print("\nLoading raw data (rule-based cleaning only, no statistics)...")
    df_raw = load_raw(args.csv)

    probe_seeds = [SEED + i for i in range(max(2, args.probe_seeds))]

    def _parse_probe(spec: str):
        if ":" not in spec:
            print(f"  Ignoring --probe '{spec}': expected SUBSTATION:MODEL.")
            return None
        sub, model = spec.split(":", 1)
        sub, model = sub.strip(), model.strip()
        if model not in MODEL_NAMES:
            print(f"  Ignoring --probe '{spec}': unknown model '{model}'. "
                  f"Choose from {MODEL_NAMES}.")
            return None
        return sub, model

    if args.probe_only:
        if not args.probe:
            print("--probe-only given but no --probe SUBSTATION:MODEL specified.")
            return
        for spec in args.probe:
            parsed = _parse_probe(spec)
            if parsed:
                run_stability_probe(df_raw, parsed[0], parsed[1], probe_seeds, OUTPUT_DIR)
        print(f"\nProbe outputs written to: {OUTPUT_DIR.resolve()}")
        return

    targets = args.substations if args.substations else USABLE_SUBSTATIONS
    all_records, all_backtests, all_runs = [], [], []

    for sub in targets:
        if sub not in USABLE_SUBSTATIONS:
            print(f"Skipping '{sub}': not in usable substation list.")
            continue
        try:
            res = run_substation(df_raw, sub, OUTPUT_DIR)
            all_records.extend(res["metrics"])
            all_records.extend(res["baselines"])
            all_runs.extend(res.get("runs", []))
            if res.get("backtest") is not None:
                all_backtests.append(res["backtest"])
        except Exception as exc:
            import traceback
            print(f"ERROR on {sub}: {exc}")
            traceback.print_exc()

    if not all_records:
        print("No results to save.")
        return

    summary = pd.DataFrame(all_records)
    summary.to_csv(OUTPUT_DIR / "metrics_summary.csv", index=False)

    if all_backtests:
        pd.concat(all_backtests, ignore_index=True).to_csv(
            OUTPUT_DIR / "recursive_backtest_summary.csv", index=False)

    if all_runs:
        pd.DataFrame(all_runs).to_csv(OUTPUT_DIR / "metrics_all_runs.csv", index=False)

    write_global_reports(summary, all_records, OUTPUT_DIR)

    print("\n" + "=" * 96)
    print("PER-SUBSTATION selected model (chosen on VALIDATION R2; test shown for reporting):")
    print("=" * 96)
    dl = summary[summary["model"].isin(MODEL_NAMES)]
    for sub, grp in dl.groupby("substation"):
        best = grp.loc[grp["val_r2"].idxmax()]
        bl = summary[(summary["substation"] == sub) &
                     (summary["model"] == "Naive-Persistence")]
        bl_r2 = bl["test_r2"].iloc[0] if len(bl) else float("nan")
        print(f"  {sub:<26} {best['model']:<20} Val R2={best['val_r2']:.4f}  "
              f"Test R2={best['test_r2']:.4f}  RMSE={best['test_rmse']:.2f}  "
              f"MAE={best['test_mae']:.2f}  (persistence Test R2={bl_r2:.4f})")

    # ---- anomaly detection: flag cells that look like bad initialisations --
    flagged = report_flagged_cells(summary, OUTPUT_DIR, args.csv,
                                   suggest_commands=not args.auto_probe)

    probes = list(args.probe or [])
    if args.auto_probe and not flagged.empty:
        probes += [f"{r['substation']}:{r['model']}" for _, r in flagged.iterrows()]
    seen = set()
    for spec in probes:
        if spec in seen:
            continue
        seen.add(spec)
        parsed = _parse_probe(spec)
        if parsed:
            try:
                run_stability_probe(df_raw, parsed[0], parsed[1], probe_seeds, OUTPUT_DIR)
            except Exception as exc:
                print(f"  Probe failed for {spec}: {exc}")

    print(f"\nAll outputs written to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()