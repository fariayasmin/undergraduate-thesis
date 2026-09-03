"""
BPDB Dhaka Substation Load Forecasting — 6-Model Comparison
=============================================================

Models:
  1. BiLSTM
  2. BiGRU
  3. BiLSTM + GRU
  4. BiGRU + BiLSTM
  5. CNN + BiLSTM + BiGRU
  6. CNN + BiGRU + BiLSTM

Context-based prediction windows:
  - Eid / Durga Puja : use ALL previous-year same-holiday data as context
  - All other days   : use last 14 days (WINDOW = 14)

Train  : Year <= 2023   (strict chronological)
Val    : Year == 2024   (early stopping + MODEL SELECTION; never trained on)
Test   : Year >= 2025   (2025 + 2026; used ONCE, for final reporting only)
         NOTE: the original "zoomed 2026-only" third panel is REMOVED.
         Two panels remain per model: TRAIN and TEST (2025+).

Plots produced per substation
------------------------------
  <sub>_<ModelName>_train_test.png  — Train panel + Test(2025+) panel (2 rows)
  <sub>_r2_comparison.png           — R² bar chart (train + test) all 6 models
  <sub>_rain_r2.png                 — Rain vs rolling-R² scatter, all 6 models
  <sub>_summary_table.png           — Metrics table image (R²/RMSE/MAE per model)

GLOBAL single-model selection (NEW)
------------------------------------
Every model is trained on EVERY substation, then ONE model is chosen for the
whole paper using three criteria (in order):
  1. Highest MEAN Test R² across all substations   (primary)
  2. Lowest  MEAN RANK  across substations          (tie-break / robustness)
  3. Number of substations where it ranks best      (reported, informative)

Global outputs:
  global_model_selection.csv / .png   — mean Test R², mean rank, wins, RMSE, MAE per model
  global_best_model.txt               — the chosen model + justification
  best_model_per_substation.csv/.png  — chosen model's metrics on every substation
  best_model_r2_by_substation.png     — bar chart of chosen model's Test R² per substation
  model_r2_boxplot.png                — Test R² distribution of all 6 models (paper figure)
  next_day_forecasts_best_model.csv   — next-day forecast per substation (chosen model)

CSV: metrics_summary.csv (all models × all substations)

Features (only those present in CSV):
  Holiday_type_encoded, Is_Weekend, Season_Encoded,
  Temp_Max_C/Min/Mean, Humidity_Mean_Pct, Precip_Sum_mm,
  Rain_Sum_mm, Wind_Max_kmh, Month_sin/cos, DOW_sin/cos,
  lag_1, lag_7, roll_mean_7, roll_std_7

Install:
    pip install tensorflow pandas numpy scikit-learn matplotlib

Run:
    python bpdb_forecast_6models.py --csv BPDB_Dhaka_City_Substations_Page3.csv
    python bpdb_forecast_6models.py --csv ... --substations Agargaon Mirpur
"""

from __future__ import annotations

import argparse
import os
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
# quiet terminal: hide TF C++ INFO/WARNING/ERROR chatter (oneDNN banners,
# NodeDef attribute notices, CPU-feature hints) — real Python errors still
# surface normally. Must be set BEFORE `import tensorflow`.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
random.seed(SEED)
np.random.seed(SEED)

import tensorflow as tf
tf.get_logger().setLevel("ERROR")
try:
    from absl import logging as _absl_logging
    _absl_logging.set_verbosity(_absl_logging.ERROR)
except Exception:
    pass
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
import matplotlib.gridspec as gridspec

# ── Configuration ─────────────────────────────────────────────────────────────
# All 18 substations with dense, reliable data (<= ~6% missing days).
# EVERY model is trained on EVERY one of these so that a single global
# best model can be selected for the paper.
USABLE_SUBSTATIONS = [
    "Bangabhaban", "Bashundhara", "Dhanmondi", "Gulshan",
    "Hasnabad", "Kalyanpur", "Lalbag", "Madartek", "Maniknagar", "Matual",
    "Mirpur", "Narinda", "Shyampur",
    "Ullon", "Uttara", "Uttara New",
    # ── Excluded from architecture selection (16 substations retained) ─────
    # "Agargaon", "Savar" — both underwent a non-stationary load regime
    # change after the training period. Verified with verify_clipping.py:
    # the 99.5th-percentile winsorisation, whose bounds are estimated on
    # 2019-2023, altered 43.6% (Agargaon) and 42.7% (Savar) of their observed
    # test-year days, against a design rate of 0.5% observed on every other
    # substation. Their true test peaks reach 153 MW and 255 MW against
    # ceilings of 114 MW and 208 MW. Architecture selection on substations
    # whose evaluation targets are dominated by a preprocessing artefact
    # would bias the comparison, so they are excluded from Phase 1.
    # They are NOT excluded for being hard to predict.
    # Excluded (too sparse / late start / non-comparable):
    # "Banani", "Aftabnagar", "Motijheel", "Moghbazar", "Cantonment", "Kodda",
    # "Keranigonj", "Kamrangirchar", "Purbachal", "Postogola", "Dhaka University",
    # "Pubail", "Hatirjheel", "Metrorail Uttara", "Metrorail Motijheel",
]

# LATE_START_SUBSTATIONS = [
#     "Pubail", "Hatirjheel", "Metrorail Uttara", "Metrorail Motijheel",
# ]

# Only features actually present in the CSV
EXOG_FEATURES = [
    "Holiday_type_encoded",
    "Is_Weekend",
    "Is_Bridge_Day",
    "Days_To_Holiday",
    "Season_Encoded",
    "Temp_Max_C", "Temp_Min_C", "Temp_Mean_C",
    "Humidity_Mean_Pct", "Precip_Sum_mm", "Rain_Sum_mm", "Wind_Max_kmh",
    "Month_sin", "Month_cos", "DOW_sin", "DOW_cos",
]

# Bangladesh weekend is Friday + Saturday (weekday 4, 5), NOT Sat/Sun. Every
# calendar feature below depends on this; using pandas' dayofweek >= 5 would
# silently produce a European calendar.
WEEKEND_DAYS = (4, 5)

# Days_To_Holiday is clipped to +/- this many days. Beyond about a week the
# proximity effect on Dhaka load is indistinguishable from a normal working
# day, and leaving the variable unbounded would let a single long holiday-free
# stretch dominate the MinMax scaling of the whole column.
HOLIDAY_PROXIMITY_CLIP = 7

# Written by load_and_clean so the exact frame the models are trained on can be
# inspected and cited. The RAW CSV is never modified.
PROCESSED_CSV = Path(__file__).parent / "data" / "processed_dataset.csv"

FESTIVE_HOLIDAY_TYPES = {"Eid", "Durga Puja"}

# ── Winsorisation scope ──────────────────────────────────────────────────
# The 0.5/99.5 percentile bounds are ALWAYS estimated on the training years
# only (that part is required, or test information leaks into preprocessing).
# This flag controls which rows the resulting clip is APPLIED to.
#   "all"   — current/published behaviour: clip train, validation and test.
#   "train" — clip the training rows only; validation and test keep their
#             observed values. Use for the winsorisation-scope ablation.
# Default is "all" so this constant changes nothing until you flip it.
# NOTE: switching to "train" is EXPECTED to LOWER R2. It restores genuine
# high peaks that the clip was flattening, which makes the task harder and
# exposes the train/test level shift. That is a correctness result, not a
# regression — report the before/after rather than reverting it.
WINSORISE_SPLITS = "all"

WINDOW        = 14
TRAIN_END_YEAR = 2023  # train = 2019..2023 (strict chronological)
VAL_YEAR       = 2024  # validation year: early stopping + MODEL SELECTION
TEST_START_YEAR = 2025   # test = Year >= 2025 (includes 2025 and 2026)
EPOCHS        = 150
BATCH_SIZE    = 16
PATIENCE      = 15
OUTPUT_DIR    = Path("forecast_outputs")
MAX_PLAUSIBLE_LOAD_MW = 450

MODEL_NAMES = [
    "BiLSTM",
    "BiGRU",
    "BiLSTM+BiGRU",
    "BiGRU+BiLSTM",
    "CNN+BiLSTM+BiGRU",
    "CNN+BiGRU+BiLSTM",
]

# ── Data cleaning ─────────────────────────────────────────────────────────────

def is_clocktime_artifact(load_val, peak_time_val) -> bool:
    if pd.isna(load_val): return False
    if pd.notna(peak_time_val): return False
    if load_val != int(load_val): return False
    v = int(load_val)
    if v < 100: return False
    h, m = divmod(v, 100)
    return 0 <= h <= 23 and m in (0, 30)

def is_implausible_load_outlier(load_val) -> bool:
    if pd.isna(load_val): return False
    return load_val > MAX_PLAUSIBLE_LOAD_MW

# ── Step 1: Load + Clean ──────────────────────────────────────────────────────

def calendar_bridge_features(is_holiday: pd.Series, is_weekend: pd.Series
                             ) -> tuple[pd.Series, pd.Series]:
    """Derive Is_Bridge_Day and Days_To_Holiday from a CONTIGUOUS daily index.

    Both inputs must be boolean Series on a gap-free daily DatetimeIndex. The
    caller is responsible for that: the raw BPDB record is missing 131 of 2445
    calendar days, and computing "previous day" on the raw rows would compare a
    day against whatever row happened to precede it, which can be a week away.

    Is_Bridge_Day (sandwiched definition)
    -------------------------------------
    1 iff the day is a WORKING day (neither a gazetted holiday nor a Fri/Sat
    weekend) AND both neighbouring days are non-working. This is the definition
    used in the load-forecasting literature: a bridge day is a lone working day
    trapped between two non-working days, which is why people take it off and
    why the load collapses toward a weekend profile. A working day that merely
    sits next to a holiday on one side is NOT a bridge day -- that is the
    separate "holiday eve / day after" effect, and here it is carried by
    Days_To_Holiday instead.

    A holiday is never a bridge day. A weekend is never a bridge day. Days at
    the very start/end of the record are treated as having a working-day
    neighbour, so they are never flagged; this is the conservative choice.

    Days_To_Holiday (signed proximity, clipped)
    -------------------------------------------
    Signed distance in days to the NEAREST gazetted holiday:
        0   on a holiday itself
        -n  n days BEFORE the next holiday   (e.g. -1 = holiday tomorrow)
        +n  n days AFTER the previous one    (e.g. +1 = holiday yesterday)
    Ties resolve to the past. Clipped to +/- HOLIDAY_PROXIMITY_CLIP.

    Looking forward in the calendar is legitimate and is not leakage: the
    Bangladesh holiday gazette is published in advance, so on day d the date of
    the next holiday is already known. This is deterministic calendar
    information, exactly like Month or DOW, not an observation of the future.
    """
    is_holiday = np.asarray(is_holiday, dtype=bool)
    is_weekend = np.asarray(is_weekend, dtype=bool)
    n = len(is_holiday)

    non_working = is_holiday | is_weekend
    working = ~non_working

    prev_nw = np.r_[False, non_working[:-1]]     # start of record: assume working
    next_nw = np.r_[non_working[1:], False]      # end of record:   assume working
    bridge = (working & prev_nw & next_nw).astype("int8")

    hol_pos = np.flatnonzero(is_holiday)
    if hol_pos.size == 0:
        proximity = np.zeros(n, dtype="int16")
    else:
        pos = np.arange(n)
        j = np.searchsorted(hol_pos, pos, side="left")   # hol_pos[j] >= pos
        big = n + 1
        d_next = np.where(j < hol_pos.size, hol_pos[np.minimum(j, hol_pos.size - 1)] - pos, big)
        d_prev = np.where(j > 0, pos - hol_pos[np.maximum(j - 1, 0)], big)
        # on a holiday d_next == 0, so the else-branch gives -0 == 0
        proximity = np.where(d_prev <= d_next, d_prev, -d_next)
        proximity = np.clip(proximity, -HOLIDAY_PROXIMITY_CLIP,
                            HOLIDAY_PROXIMITY_CLIP).astype("int16")

    return pd.Series(bridge), pd.Series(proximity)


def verify_calendar_features(df: pd.DataFrame) -> None:
    """Requirement 7: assert the derived features are clean, then report."""
    b, p = df["Is_Bridge_Day"], df["Days_To_Holiday"]
    assert b.notna().all(), "Is_Bridge_Day contains NaN"
    assert p.notna().all(), "Days_To_Holiday contains NaN"
    assert set(b.unique()) <= {0, 1}, f"Is_Bridge_Day not binary: {sorted(b.unique())}"
    assert p.between(-HOLIDAY_PROXIMITY_CLIP, HOLIDAY_PROXIMITY_CLIP).all(), \
        "Days_To_Holiday outside the clip range"
    hol = df["Holiday_type"].ne("No Holiday") if "Holiday_type" in df.columns \
        else pd.Series(False, index=df.index)
    assert not (b.eq(1) & df["Is_Weekend"].eq(1)).any(), "a weekend was flagged as a bridge day"
    assert not (b.eq(1) & hol).any(), "a holiday was flagged as a bridge day"

    dow = df["Date"].dt.day_name() if "Date" in df.columns else df.index.day_name()
    by_dow = dow[b.eq(1)].value_counts()
    print(f"  calendar features: Is_Bridge_Day = {int(b.sum())} days "
          f"({b.mean()*100:.2f}% of {len(b)}), binary, no NaN"
          + (f" | {', '.join(f'{k} {v}' for k, v in by_dow.items())}" if len(by_dow) else ""))
    print(f"  Days_To_Holiday in [{int(p.min())}, {int(p.max())}], "
          f"{int(p.eq(0).sum())} holiday days, "
          f"{int(p.abs().le(3).sum())} days within +/-3 of a holiday")


def load_and_clean(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    for sub in USABLE_SUBSTATIONS:
        load_col, time_col = f"{sub}_Load_MW", f"{sub}_Peak_Time"
        if load_col not in df.columns:
            continue
        clock_mask = df.apply(
            lambda r, lc=load_col, tc=time_col: is_clocktime_artifact(r[lc], r[tc]), axis=1
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
    df["Is_Weekend"] = dow_num.isin([4, 5]).astype(int)
    season_map = {12:0,1:0,2:0,3:1,4:1,5:1,6:2,7:2,8:2,9:2,10:3,11:3}
    df["Season_Encoded"] = df["Month"].map(season_map)

    holiday_type_map = {"No Holiday": 0, "Public Holiday": 1, "Eid": 2, "Durga Puja": 3}
    if "Holiday_type" in df.columns:
        df["Holiday_type"] = df["Holiday_type"].ffill().bfill()
        df["Holiday_type_encoded"] = (
            df["Holiday_type"].map(holiday_type_map).fillna(0).astype(int)
        )
    else:
        df["Holiday_type"] = "No Holiday"
        df["Holiday_type_encoded"] = 0

    # The calendar features below are derived from the SAME holiday notion the
    # model receives in Holiday_type_encoded, i.e. after the ffill/bfill above.
    # 131 of the 2445 calendar days have no BPDB record and inherit their
    # neighbour's label; 11 of those become holidays. Deriving from the raw
    # pre-fill mask instead gives the same 31 bridge days and differs on only
    # two dates, but it would let a day be encoded as a holiday and flagged as
    # a working bridge day at the same time. Consistency wins.
    holiday_mask = df["Holiday_type_encoded"].ne(0)

    weather_cols = [
        "Temp_Max_C","Temp_Min_C","Temp_Mean_C",
        "Humidity_Mean_Pct","Precip_Sum_mm","Rain_Sum_mm","Wind_Max_kmh",
    ]
    for c in weather_cols:
        if c in df.columns:
            df[c] = df[c].ffill().bfill()
        else:
            df[c] = 0.0

    df["Month_sin"] = np.sin(2 * np.pi * df["Month"] / 12)
    df["Month_cos"] = np.cos(2 * np.pi * df["Month"] / 12)
    df["DOW_sin"]   = np.sin(2 * np.pi * dow_num / 7)
    df["DOW_cos"]   = np.cos(2 * np.pi * dow_num / 7)

    # Derived AFTER the reindex above, so "previous day" and "next day" are
    # true calendar neighbours rather than adjacent surviving rows.
    _bridge, _prox = calendar_bridge_features(holiday_mask,
                                              df["Is_Weekend"].astype(bool))
    df["Is_Bridge_Day"] = _bridge.values
    df["Days_To_Holiday"] = _prox.values

    df["DOW_num"] = dow_num
    for sub in USABLE_SUBSTATIONS:
        col = f"{sub}_Load_MW"
        if col not in df.columns or df[col].notna().sum() == 0:
            continue
        first_valid = df[col].first_valid_index()
        active = df.loc[first_valid:, col]
        # forward-only (causal) interpolation: gaps are filled from PAST
        # values only, so no future information flows backward in time
        active = active.interpolate(method="time", limit=7, limit_direction="forward")
        if active.isna().any():
            # imputation statistics from TRAIN YEARS ONLY (<= TRAIN_END_YEAR)
            # so no validation/test information leaks into filled values
            tr = active[active.index.year <= TRAIN_END_YEAR]
            if tr.notna().sum() == 0:      # record starts after train period
                tr = active               # (such substations use the
                                          #  per-substation fallback split)
            group_med = tr.groupby(
                [df.loc[tr.index, "Month"], df.loc[tr.index, "DOW_num"]]
            ).median()
            overall_med = tr.median()
            for idx in active[active.isna()].index:
                key = (df.loc[idx, "Month"], df.loc[idx, "DOW_num"])
                active.loc[idx] = group_med.get(key, overall_med)
        df.loc[first_valid:, col] = active

    for sub in USABLE_SUBSTATIONS:
        col = f"{sub}_Load_MW"
        if col not in df.columns or df[col].notna().sum() == 0:
            continue
        # winsorisation bounds from TRAIN YEARS ONLY (no test leakage);
        # the clip is then applied to the whole series
        base = df.loc[df.index.year <= TRAIN_END_YEAR, col]
        if base.notna().sum() == 0:
            base = df[col]
        lo, hi = base.quantile([0.005, 0.995])
        if WINSORISE_SPLITS == "train":
            m = df.index.year <= TRAIN_END_YEAR
            df.loc[m, col] = df.loc[m, col].clip(lo, hi)
        else:
            df[col] = df[col].clip(lo, hi)

    df = df.reset_index()
    verify_calendar_features(df)                       # requirement 7
    save_processed_dataset(df)                         # requirement 3 / 6
    return df


def save_processed_dataset(df: pd.DataFrame, path: Path | None = None) -> Path | None:
    """Write the cleaned, feature-engineered frame the models actually see.

    The RAW CSV is never touched. Set LP_WRITE_PROCESSED=0 to skip the write
    (useful inside the 108-cell sweep, where load_and_clean runs once anyway).
    """
    if os.environ.get("LP_WRITE_PROCESSED", "1") == "0":
        return None
    path = Path(path) if path is not None else PROCESSED_CSV
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    print(f"  processed dataset written: {path} "
          f"({len(df)} rows x {df.shape[1]} cols)")
    return path

# ── Step 2: Feature engineering ──────────────────────────────────────────────

def build_substation_frame(df: pd.DataFrame, substation: str) -> pd.DataFrame:
    load_col = f"{substation}_Load_MW"
    cols = ["Date", load_col] + [c for c in EXOG_FEATURES if c in df.columns]
    sub_df = df[cols].copy()

    first_valid = sub_df[load_col].first_valid_index()
    if first_valid is None:
        raise ValueError(f"{substation}: no real load data found.")
    sub_df = sub_df.loc[first_valid:].reset_index(drop=True)

    sub_df["lag_1"]       = sub_df[load_col].shift(1)
    sub_df["lag_7"]       = sub_df[load_col].shift(7)
    sub_df["roll_mean_7"] = sub_df[load_col].shift(1).rolling(7).mean()
    sub_df["roll_std_7"]  = sub_df[load_col].shift(1).rolling(7).std()
    sub_df = sub_df.bfill()
    return sub_df

# ── Step 3: Scaling + Windowing ───────────────────────────────────────────────

def make_sequences(values: np.ndarray, window: int):
    X, y = [], []
    for i in range(len(values) - window):
        X.append(values[i : i + window])
        y.append(values[i + window, 0])
    return np.array(X), np.array(y)

def prepare_train_test(sub_df: pd.DataFrame, load_col: str, substation: str = ""):
    feature_cols = [load_col] + [c for c in sub_df.columns if c not in ("Date", load_col)]
    sub_df = sub_df[["Date"] + feature_cols].copy()

    # ── strict chronological split: train 2019-2023 | val 2024 | test 2025-26.
    # The model NEVER trains on validation or test data; validation is used
    # only for early stopping / LR scheduling and for MODEL SELECTION; the
    # test set is used exactly once, for final reporting.
    year = sub_df["Date"].dt.year
    train_mask = year <= TRAIN_END_YEAR
    val_mask   = year == VAL_YEAR
    test_mask  = year >= TEST_START_YEAR                     # 2025 AND 2026

    if train_mask.sum() < WINDOW + 1:
        # record starts after the train period → chronological 70/15/15
        n = len(sub_df)
        i_tr, i_va = int(n * 0.70), int(n * 0.85)
        if i_tr < WINDOW + 1:
            raise ValueError(f"{substation}: not enough data for windowing.")
        train_mask = pd.Series(False, index=sub_df.index); train_mask.iloc[:i_tr] = True
        val_mask   = pd.Series(False, index=sub_df.index); val_mask.iloc[i_tr:i_va] = True
        test_mask  = pd.Series(False, index=sub_df.index); test_mask.iloc[i_va:] = True
        print(f"  NOTE: {substation} uses per-substation chronological "
              f"70/15/15 split ({train_mask.sum()} train, {val_mask.sum()} "
              f"val, {test_mask.sum()} test days).")

    train_raw   = sub_df.loc[train_mask, feature_cols].values
    val_raw     = sub_df.loc[val_mask,   feature_cols].values
    test_raw    = sub_df.loc[test_mask,  feature_cols].values
    train_dates = sub_df.loc[train_mask, "Date"].values
    val_dates   = sub_df.loc[val_mask,   "Date"].values
    test_dates  = sub_df.loc[test_mask,  "Date"].values

    # scaler fitted on TRAIN ONLY, applied everywhere (no leakage)
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_raw)
    val_scaled   = scaler.transform(val_raw)
    test_scaled  = scaler.transform(test_raw)

    X_train, y_train = make_sequences(train_scaled, WINDOW)

    # sequence context windows use only PAST data (legitimate):
    # val sequences start from the last WINDOW train days,
    # test sequences from the last WINDOW val days
    extended_val  = np.concatenate([train_scaled[-WINDOW:], val_scaled], axis=0)
    X_val, y_val  = make_sequences(extended_val, WINDOW)
    pre_test      = np.concatenate([train_scaled, val_scaled], axis=0)[-WINDOW:]
    extended_test = np.concatenate([pre_test, test_scaled], axis=0)
    X_test, y_test = make_sequences(extended_test, WINDOW)

    return (X_train, y_train, X_val, y_val, X_test, y_test,
            scaler, train_dates, val_dates, test_dates, feature_cols, sub_df)

def inverse_target(scaler, scaled_target: np.ndarray, n_features: int) -> np.ndarray:
    dummy = np.zeros((len(scaled_target), n_features))
    dummy[:, 0] = scaled_target
    return scaler.inverse_transform(dummy)[:, 0]

# ── Step 4: Context-based next-day prediction ────────────────────────────────

def get_context_window_for_next_day(
    df_full: pd.DataFrame,
    sub_df: pd.DataFrame,
    load_col: str,
    feature_cols: list,
    scaler,
    model,
    n_features: int,
) -> tuple[float, str]:
    last_row  = sub_df.iloc[-1]
    next_date = last_row["Date"] + pd.Timedelta(days=1)
    next_month = next_date.month

    next_holiday_type = "No Holiday"
    if "Holiday_type" in df_full.columns:
        recent = df_full[df_full["Date"] >= (next_date - pd.Timedelta(days=3))]
        if not recent.empty:
            ht = recent["Holiday_type"].iloc[-1]
            next_holiday_type = ht if pd.notna(ht) else "No Holiday"

    if "Holiday_type" in df_full.columns:
        same_month = df_full[df_full["Date"].dt.month == next_month]
        festive_in_month = same_month[same_month["Holiday_type"].isin(FESTIVE_HOLIDAY_TYPES)]
        if len(festive_in_month) >= 3:
            next_holiday_type = festive_in_month["Holiday_type"].iloc[0]

    is_festive = next_holiday_type in FESTIVE_HOLIDAY_TYPES

    last_window_raw = sub_df[feature_cols].values[-WINDOW:].copy()

    if is_festive:
        hist = df_full[df_full["Holiday_type"] == next_holiday_type]
        festive_loads = hist[load_col].dropna()
        festive_mean  = festive_loads.mean() if len(festive_loads) > 0 else sub_df[load_col].iloc[-1]
        last_window_raw[:, 0] = np.clip(
            last_window_raw[:, 0] * 0.4 + festive_mean * 0.6, 0, MAX_PLAUSIBLE_LOAD_MW
        )
        window_desc = (
            f"Festive context ({next_holiday_type}): "
            f"used {len(festive_loads)} historical same-holiday readings "
            f"(mean={festive_mean:.1f} MW) from all prior years."
        )
    else:
        window_desc = f"Standard 14-day rolling window ending {last_row['Date'].date()}."

    last_window_scaled = scaler.transform(last_window_raw)
    pred_scaled = model.predict(last_window_scaled[np.newaxis, ...], verbose=0).ravel()
    pred_mw = inverse_target(scaler, pred_scaled, n_features)[0]
    return pred_mw, window_desc

# ── Step 5: Model architectures ───────────────────────────────────────────────

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

# ── Step 6: Train one model ───────────────────────────────────────────────────

def train_one_model(model_name, X_train, y_train, X_val, y_val,
                    X_test, y_test, scaler, n_features):
    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    model = build_model(model_name, X_train.shape[1], n_features)
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=7, min_lr=1e-6),
    ]
    # explicit chronological validation year (VAL_YEAR) — the model never
    # trains on it; it is used only for early stopping / LR scheduling
    import time as _time
    _t0 = _time.time()
    hist = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                     epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=False,
                     callbacks=callbacks, verbose=0)
    _elapsed = _time.time() - _t0
    _epochs_run = len(hist.history["loss"])

    y_tr_act  = inverse_target(scaler, y_train,                               n_features)
    y_tr_pred = inverse_target(scaler, model.predict(X_train, verbose=0).ravel(), n_features)
    y_va_act  = inverse_target(scaler, y_val,                                 n_features)
    y_va_pred = inverse_target(scaler, model.predict(X_val,   verbose=0).ravel(), n_features)
    y_te_act  = inverse_target(scaler, y_test,                                n_features)
    y_te_pred = inverse_target(scaler, model.predict(X_test,  verbose=0).ravel(), n_features)

    metrics = {
        "model":      model_name,
        "train_r2":   r2_score(y_tr_act, y_tr_pred),
        "train_rmse": float(np.sqrt(mean_squared_error(y_tr_act, y_tr_pred))),
        "train_mae":  float(mean_absolute_error(y_tr_act, y_tr_pred)),
        "val_r2":     r2_score(y_va_act, y_va_pred),
        "val_rmse":   float(np.sqrt(mean_squared_error(y_va_act, y_va_pred))),
        "val_mae":    float(mean_absolute_error(y_va_act, y_va_pred)),
        "test_r2":    r2_score(y_te_act, y_te_pred),
        "test_rmse":  float(np.sqrt(mean_squared_error(y_te_act, y_te_pred))),
        "test_mae":   float(mean_absolute_error(y_te_act, y_te_pred)),
        "epochs_run": int(_epochs_run),
        "train_seconds": round(_elapsed, 1),
    }
    return metrics, y_tr_act, y_tr_pred, y_te_act, y_te_pred, model

# ── Step 7: Plots ─────────────────────────────────────────────────────────────

def plot_train_test(substation, model_name, metrics,
                    train_dates, y_tr_act, y_tr_pred,
                    test_dates,  y_te_act,  y_te_pred,
                    output_dir):
    """
    Two-panel figure per model:
      Top    : TRAIN  actual vs predicted
      Bottom : TEST (2025+) actual vs predicted
    No 2026-only zoomed panel.
    """
    tr_dt = pd.to_datetime(train_dates)
    te_dt = pd.to_datetime(test_dates)

    # align train dates with the windowed arrays
    n_win = len(y_tr_act)
    tr_dt_aligned = tr_dt[WINDOW : WINDOW + n_win] if len(tr_dt) > WINDOW + n_win else tr_dt[-n_win:]

    fig, axes = plt.subplots(2, 1, figsize=(14, 9))

    # ── Train panel ──
    ax = axes[0]
    ax.plot(tr_dt_aligned, y_tr_act,  label="Actual",    linewidth=1.2, color="steelblue")
    ax.plot(tr_dt_aligned, y_tr_pred, label="Predicted", linewidth=1.2, color="tomato", linestyle="--", alpha=0.85)
    ax.set_title(f"{substation} | {model_name} — TRAIN  "
                 f"(R²={metrics['train_r2']:.4f}, RMSE={metrics['train_rmse']:.2f}, MAE={metrics['train_mae']:.2f})",
                 fontsize=10)
    ax.set_ylabel("Load (MW)")
    ax.legend(fontsize=9)
    ax.tick_params(axis="x", rotation=20, labelsize=8)

    # ── Test (2025+) panel ──
    ax = axes[1]
    ax.plot(te_dt, y_te_act,  label="Actual",    linewidth=1.2, color="steelblue")
    ax.plot(te_dt, y_te_pred, label="Predicted", linewidth=1.2, color="tomato", linestyle="--", alpha=0.85)
    ax.set_title(f"{substation} | {model_name} — TEST (2025+)  "
                 f"(R²={metrics['test_r2']:.4f}, RMSE={metrics['test_rmse']:.2f}, MAE={metrics['test_mae']:.2f})",
                 fontsize=10)
    ax.set_ylabel("Load (MW)")
    ax.set_xlabel("Date")
    ax.legend(fontsize=9)
    ax.tick_params(axis="x", rotation=20, labelsize=8)

    fig.suptitle(f"{substation} — {model_name}: Actual vs Predicted", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    safe_name = model_name.replace("+", "_").replace(" ", "_")
    fig.savefig(output_dir / f"{substation.replace(' ', '_')}_{safe_name}_train_test.png", dpi=120)
    plt.close(fig)


def plot_r2_comparison(substation, all_metrics, output_dir):
    """Grouped bar: Train R² vs Test R² for all 6 models."""
    fig, ax = plt.subplots(figsize=(11, 5))
    x     = np.arange(len(MODEL_NAMES))
    width = 0.35
    tr_r2 = [m["train_r2"] for m in all_metrics]
    te_r2 = [m["test_r2"]  for m in all_metrics]

    bars1 = ax.bar(x - width/2, tr_r2, width, label="Train R²", color="steelblue", alpha=0.82)
    bars2 = ax.bar(x + width/2, te_r2, width, label="Test R²",  color="tomato",    alpha=0.82)

    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_NAMES, rotation=18, ha="right", fontsize=9)
    ax.set_ylabel("R²")
    ax.set_ylim(min(0, min(te_r2) - 0.05), 1.05)
    ax.axhline(0, color="black", linewidth=0.6, linestyle="--")
    ax.set_title(f"{substation} — R² Comparison: All 6 Models (Train vs Test 2025+)")
    ax.legend()

    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                f"{h:.3f}", ha="center", va="bottom", fontsize=7)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                f"{h:.3f}", ha="center", va="bottom", fontsize=7)

    fig.tight_layout()
    fig.savefig(output_dir / f"{substation.replace(' ', '_')}_r2_comparison.png", dpi=120)
    plt.close(fig)


def plot_rain_r2(df, substation, all_metrics, test_dates, all_results, output_dir):
    """Rolling-window Rain vs R² scatter for all 6 models — same style as original."""
    if "Rain_Sum_mm" not in df.columns:
        return

    te_dt = pd.to_datetime(test_dates)
    rain_series = df.set_index("Date")["Rain_Sum_mm"].reindex(te_dt)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(MODEL_NAMES)))

    for i, model_name in enumerate(MODEL_NAMES):
        _, _, y_te_act, y_te_pred, _ = all_results[model_name]
        r2_vals, rain_vals = [], []
        for j in range(WINDOW, len(te_dt)):
            seg_act  = y_te_act [j - WINDOW:j]
            seg_pred = y_te_pred[j - WINDOW:j]
            if len(seg_act) >= 2 and seg_act.std() > 0:
                r2_vals.append(r2_score(seg_act, seg_pred))
                rv = rain_series.iloc[j - WINDOW:j].mean()
                rain_vals.append(float(rv) if pd.notna(rv) else 0.0)

        if rain_vals:
            ax.scatter(rain_vals, r2_vals, label=model_name,
                       alpha=0.55, s=18, color=colors[i])
            z = np.polyfit(rain_vals, r2_vals, 1)
            xline = np.linspace(min(rain_vals), max(rain_vals), 100)
            ax.plot(xline, np.poly1d(z)(xline), color=colors[i], linewidth=1.2, alpha=0.85)

    ax.set_xlabel("Mean Rainfall in 14-day Window (mm)")
    ax.set_ylabel("Rolling 14-day R²")
    ax.set_title(f"{substation} — Rain vs R² (All 6 Models, Test 2025+)")
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / f"{substation.replace(' ', '_')}_rain_r2.png", dpi=120)
    plt.close(fig)


def plot_summary_table(substation, all_metrics, output_dir):
    """
    Render a clean metrics table image for this substation —
    one row per model, columns: Model | Train R² | Train RMSE | Train MAE
                                       | Test R²  | Test RMSE  | Test MAE
    Highlights the best test-R² row in light green.
    """
    col_labels = ["Model", "Train R²", "Train RMSE", "Train MAE",
                  "Test R²",  "Test RMSE",  "Test MAE"]
    rows = []
    for m in all_metrics:
        rows.append([
            m["model"],
            f"{m['train_r2']:.4f}",
            f"{m['train_rmse']:.2f}",
            f"{m['train_mae']:.2f}",
            f"{m['test_r2']:.4f}",
            f"{m['test_rmse']:.2f}",
            f"{m['test_mae']:.2f}",
        ])

    best_idx = int(np.argmax([m["val_r2"] for m in all_metrics]))  # selection metric
    cell_colors = []
    for i, _ in enumerate(rows):
        if i == best_idx:
            cell_colors.append(["#c6efce"] * len(col_labels))   # light green
        else:
            cell_colors.append(["#f7f7f7" if i % 2 == 0 else "white"] * len(col_labels))

    fig, ax = plt.subplots(figsize=(13, 0.55 * len(rows) + 1.6))
    ax.axis("off")

    tbl = ax.table(
        cellText=rows,
        colLabels=col_labels,
        cellColours=cell_colors,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)
    tbl.scale(1, 1.55)

    # Bold header
    for j in range(len(col_labels)):
        tbl[(0, j)].set_facecolor("#4472C4")
        tbl[(0, j)].get_text().set_color("white")
        tbl[(0, j)].get_text().set_fontweight("bold")

    ax.set_title(
        f"{substation} — Metrics Summary (all 6 models)\n"
        f"[highlighted row = best Test R²]",
        fontsize=11, fontweight="bold", pad=14,
    )
    fig.tight_layout()
    fig.savefig(output_dir / f"{substation.replace(' ', '_')}_summary_table.png",
                dpi=130, bbox_inches="tight")
    plt.close(fig)


# ── Step 8: Run all 6 models for one substation ───────────────────────────────

def run_substation(df: pd.DataFrame, substation: str, output_dir: Path) -> list[dict]:
    print(f"\n{'='*70}\nSubstation: {substation}\n{'='*70}")
    load_col = f"{substation}_Load_MW"
    if load_col not in df.columns:
        print(f"  SKIPPING: {load_col} not found.")
        return []

    sub_df = build_substation_frame(df, substation)
    (X_train, y_train, X_val, y_val, X_test, y_test,
     scaler, train_dates, val_dates, test_dates,
     feature_cols, sub_df_full) = prepare_train_test(
        sub_df, load_col, substation
    )
    n_features = X_train.shape[2]
    print(f"  Train sequences: {X_train.shape}  |  Val: {X_val.shape}  |  "
          f"Test: {X_test.shape}")

    all_metrics = []
    all_results = {}

    for mi, model_name in enumerate(MODEL_NAMES, start=1):
        print(f"  [{mi}/{len(MODEL_NAMES)}] {model_name:<18} training…",
              end=" ", flush=True)
        out = train_one_model(model_name, X_train, y_train, X_val, y_val,
                              X_test, y_test, scaler, n_features)
        metrics, y_tr_act, y_tr_pred, y_te_act, y_te_pred, model_obj = out
        metrics["substation"] = substation
        all_metrics.append(metrics)
        all_results[model_name] = (y_tr_act, y_tr_pred, y_te_act, y_te_pred, model_obj)
        stopped = ("early-stopped" if metrics["epochs_run"] < EPOCHS
                   else "full run")
        print(f"{metrics['epochs_run']:>3} epochs ({stopped}), "
              f"{metrics['train_seconds']:>6.1f}s | "
              f"Train R²={metrics['train_r2']:.3f}  "
              f"Val R²={metrics['val_r2']:.3f}  "
              f"Test R²={metrics['test_r2']:.3f}", flush=True)

        # ── Per-model: 2-panel Train + Test(2025+) plot ──────────────────────
        plot_train_test(
            substation, model_name, metrics,
            train_dates, y_tr_act, y_tr_pred,
            test_dates,  y_te_act, y_te_pred,
            output_dir,
        )

    # ── Cross-model plots ─────────────────────────────────────────────────────
    plot_r2_comparison(substation, all_metrics, output_dir)
    plot_rain_r2(df, substation, all_metrics, test_dates, all_results, output_dir)
    plot_summary_table(substation, all_metrics, output_dir)

    # ── Contextual next-day prediction for EVERY model ────────────────────────
    # The single global best model is only known AFTER all substations have
    # been processed, so we store next-day forecasts for all 6 models here and
    # let main() report the one belonging to the globally chosen model.
    last_row  = sub_df.iloc[-1]
    next_date = last_row["Date"] + pd.Timedelta(days=1)
    for m in all_metrics:
        _, _, _, _, model_obj = all_results[m["model"]]
        try:
            next_pred_mw, window_desc = get_context_window_for_next_day(
                df, sub_df, load_col, feature_cols, scaler, model_obj, n_features
            )
        except Exception:
            next_pred_mw, window_desc = float("nan"), "unavailable"
        m["next_day_date"]     = str(next_date.date())
        m["next_day_pred_mw"]  = float(next_pred_mw)
        m["next_day_context"]  = window_desc

    # Free GPU/CPU memory between substations (108 models total otherwise)
    tf.keras.backend.clear_session()

    return all_metrics


# ── Step 9: Global summary table ──────────────────────────────────────────────

def plot_global_summary_table(all_records: list[dict], output_dir: Path):
    """
    One big table image: columns = models, rows = substations, value = Test R².
    Also saves a separate sheet with all metrics.
    """
    df_sum = pd.DataFrame(all_records)
    pivot  = df_sum.pivot(index="substation", columns="model", values="test_r2")
    pivot  = pivot[MODEL_NAMES]   # enforce column order
    pivot  = pivot.sort_index()

    n_rows, n_cols = pivot.shape
    fig_h = max(6, 0.42 * n_rows + 2.0)
    fig, ax = plt.subplots(figsize=(14, fig_h))
    ax.axis("off")

    cell_text   = [[f"{v:.4f}" if pd.notna(v) else "—" for v in row]
                   for row in pivot.values]
    row_labels  = list(pivot.index)
    col_labels  = list(pivot.columns)

    # Colour-map cells by value
    vmin = df_sum["test_r2"].min()
    vmax = df_sum["test_r2"].max()
    cmap = plt.cm.RdYlGn
    cell_colors = []
    for row in pivot.values:
        cc = []
        for v in row:
            if pd.isna(v):
                cc.append("#e0e0e0")
            else:
                normed = (v - vmin) / max(vmax - vmin, 1e-9)
                rgba   = cmap(normed)
                cc.append(rgba)
        cell_colors.append(cc)

    tbl = ax.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=col_labels,
        cellColours=cell_colors,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 1.5)

    for j in range(n_cols):
        tbl[(0, j)].set_facecolor("#4472C4")
        tbl[(0, j)].get_text().set_color("white")
        tbl[(0, j)].get_text().set_fontweight("bold")

    ax.set_title(
        "Global Summary — Test R² by Substation × Model\n(green = higher R²)",
        fontsize=12, fontweight="bold", pad=16,
    )
    fig.tight_layout()
    fig.savefig(output_dir / "global_summary_test_r2.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved global_summary_test_r2.png")


# ── Step 10: GLOBAL single-model selection ────────────────────────────────────

def select_global_best_model(summary: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    """
    Choose ONE model for the entire paper.

    Selection criteria (in order) — computed on the VALIDATION year (2024)
    so the 2025-26 TEST set never participates in any decision and is used
    exactly once, for final reporting:
      1. Highest mean Validation R² across all substations  (primary)
      2. Lowest mean validation rank (1 = best per substation) (tie-break)
    Test R² / RMSE / MAE are still reported for every model (transparency),
    but they do NOT influence the selection.
    """
    # Rank models within each substation by VALIDATION R² (1 = best)
    summary = summary.copy()
    summary["rank"] = summary.groupby("substation")["val_r2"] \
                             .rank(ascending=False, method="min")

    sel = summary.groupby("model").agg(
        mean_val_r2  =("val_r2",   "mean"),
        min_val_r2   =("val_r2",   "min"),
        mean_rank    =("rank",     "mean"),
        wins         =("rank",     lambda r: int((r == 1).sum())),
        mean_val_rmse=("val_rmse", "mean"),
        mean_test_r2 =("test_r2",  "mean"),
        min_test_r2  =("test_r2",  "min"),
        mean_test_rmse=("test_rmse","mean"),
        mean_test_mae =("test_mae", "mean"),
        mean_train_r2 =("train_r2", "mean"),
    ).reset_index()

    sel = sel.sort_values(
        ["mean_val_r2", "mean_rank"], ascending=[False, True]
    ).reset_index(drop=True)

    best_model = sel.iloc[0]["model"]
    return best_model, sel


def plot_global_selection_table(sel: pd.DataFrame, best_model: str, output_dir: Path):
    col_labels = ["Model", "Mean Val R²", "Worst Val R²", "Mean Rank (val)",
                  "Wins (val)", "Mean Test R²", "Test RMSE", "Test MAE",
                  "Mean Train R²"]
    rows = []
    for _, r in sel.iterrows():
        rows.append([
            r["model"], f"{r['mean_val_r2']:.4f}", f"{r['min_val_r2']:.4f}",
            f"{r['mean_rank']:.2f}", f"{int(r['wins'])}",
            f"{r['mean_test_r2']:.4f}",
            f"{r['mean_test_rmse']:.2f}", f"{r['mean_test_mae']:.2f}",
            f"{r['mean_train_r2']:.4f}",
        ])

    cell_colors = []
    for _, r in sel.iterrows():
        if r["model"] == best_model:
            cell_colors.append(["#c6efce"] * len(col_labels))
        else:
            cell_colors.append(["white"] * len(col_labels))

    fig, ax = plt.subplots(figsize=(13, 0.55 * len(rows) + 1.8))
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=col_labels,
                   cellColours=cell_colors, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9.5); tbl.scale(1, 1.55)
    for j in range(len(col_labels)):
        tbl[(0, j)].set_facecolor("#4472C4")
        tbl[(0, j)].get_text().set_color("white")
        tbl[(0, j)].get_text().set_fontweight("bold")
    ax.set_title(
        f"Global Model Selection — selected on VALIDATION (2024); "
        f"test (2025-26) reported only\n"
        f"[green row = single model chosen for the paper: {best_model}]",
        fontsize=11, fontweight="bold", pad=14,
    )
    fig.tight_layout()
    fig.savefig(output_dir / "global_model_selection.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_model_r2_boxplot(summary: pd.DataFrame, best_model: str, output_dir: Path):
    """Distribution of Test R² across substations for every model — paper figure."""
    order = (summary.groupby("model")["test_r2"].mean()
                    .sort_values(ascending=False).index.tolist())
    data  = [summary.loc[summary["model"] == m, "test_r2"].values for m in order]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    # matplotlib renamed boxplot's labels= to tick_labels= in 3.9 and
    # removed the old name later — pick the right kwarg by version
    import matplotlib as _mpl
    _mpl_maj_min = tuple(int(p) for p in _mpl.__version__.split(".")[:2])
    _label_kw = "tick_labels" if _mpl_maj_min >= (3, 9) else "labels"
    bp = ax.boxplot(data, **{_label_kw: order}, patch_artist=True, showmeans=True,
                    meanprops=dict(marker="D", markerfacecolor="black", markersize=5))
    for patch, m in zip(bp["boxes"], order):
        patch.set_facecolor("#c6efce" if m == best_model else "#dbe5f1")
    ax.set_ylabel("Test R² (2025+)")
    ax.set_title("Test R² distribution across all substations — per model\n"
                 f"(green = selected single model: {best_model}; ◆ = mean)")
    ax.tick_params(axis="x", rotation=18, labelsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "model_r2_boxplot.png", dpi=130)
    plt.close(fig)


def report_best_model_per_substation(summary: pd.DataFrame, best_model: str,
                                     output_dir: Path):
    """Chosen model's metrics on every substation: CSV + table image + bar chart."""
    bm = summary[summary["model"] == best_model].sort_values("substation").copy()
    keep = ["substation", "train_r2", "train_rmse", "train_mae",
            "test_r2", "test_rmse", "test_mae"]
    bm[keep].to_csv(output_dir / "best_model_per_substation.csv", index=False)

    # Table image
    col_labels = ["Substation", "Train R²", "Train RMSE", "Train MAE",
                  "Test R²", "Test RMSE", "Test MAE"]
    rows = [[r["substation"], f"{r['train_r2']:.4f}", f"{r['train_rmse']:.2f}",
             f"{r['train_mae']:.2f}", f"{r['test_r2']:.4f}",
             f"{r['test_rmse']:.2f}", f"{r['test_mae']:.2f}"]
            for _, r in bm.iterrows()]
    fig, ax = plt.subplots(figsize=(12, 0.42 * len(rows) + 1.8))
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=col_labels, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 1.4)
    for j in range(len(col_labels)):
        tbl[(0, j)].set_facecolor("#4472C4")
        tbl[(0, j)].get_text().set_color("white")
        tbl[(0, j)].get_text().set_fontweight("bold")
    ax.set_title(f"{best_model} — per-substation performance (single model for paper)",
                 fontsize=11, fontweight="bold", pad=14)
    fig.tight_layout()
    fig.savefig(output_dir / "best_model_per_substation.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # Bar chart of Test R²
    bm_sorted = bm.sort_values("test_r2", ascending=False)
    fig, ax = plt.subplots(figsize=(12, 5.5))
    bars = ax.bar(bm_sorted["substation"], bm_sorted["test_r2"],
                  color="steelblue", alpha=0.85)
    mean_r2 = bm_sorted["test_r2"].mean()
    ax.axhline(mean_r2, color="tomato", linestyle="--", linewidth=1.2,
               label=f"Mean Test R² = {mean_r2:.4f}")
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.005, f"{h:.3f}",
                ha="center", va="bottom", fontsize=7)
    ax.set_ylabel("Test R² (2025+)")
    ax.set_title(f"{best_model} — Test R² by substation")
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "best_model_r2_by_substation.png", dpi=130)
    plt.close(fig)

    # Next-day forecasts of the chosen model
    if "next_day_pred_mw" in bm.columns:
        nd = bm[["substation", "next_day_date", "next_day_pred_mw",
                 "next_day_context", "test_r2"]].copy()
        nd.to_csv(output_dir / "next_day_forecasts_best_model.csv", index=False)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="BPDB 6-Model Load Forecasting")
    parser.add_argument("--csv", default="BPDB_Dhaka_City_Substations_Page3.csv")
    parser.add_argument("--substations", nargs="*", default=None)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)

    print("Loading and cleaning data...")
    df = load_and_clean(args.csv)

    targets    = args.substations if args.substations else USABLE_SUBSTATIONS
    all_records = []

    import time as _time
    n_targets = len(targets)
    print(f"\nPHASE 1 — model-selection experiment: {n_targets} substations "
          f"× {len(MODEL_NAMES)} architectures = "
          f"{n_targets * len(MODEL_NAMES)} models")
    print(f"Split: train ≤{TRAIN_END_YEAR} | val {VAL_YEAR} | "
          f"test ≥{TEST_START_YEAR}  (window={WINDOW}, seed={SEED}, "
          f"max epochs={EPOCHS}, early-stop patience={PATIENCE})")
    _sub_times, _t_start = [], _time.time()

    for si, sub in enumerate(targets, start=1):
        if sub not in USABLE_SUBSTATIONS:
            print(f"Skipping '{sub}': not in usable substation list.")
            continue
        _t_sub = _time.time()
        print(f"\n[{si}/{n_targets}] ══ {sub} "
              f"═══════════════════════════════════════", flush=True)
        try:
            metrics_list = run_substation(df, sub, OUTPUT_DIR)
            all_records.extend(metrics_list)
            _sub_times.append(_time.time() - _t_sub)
            _avg = sum(_sub_times) / len(_sub_times)
            _left = (n_targets - si) * _avg
            print(f"  ── {sub} done in {_sub_times[-1]/60:.1f} min | "
                  f"elapsed {(_time.time()-_t_start)/60:.1f} min | "
                  f"ETA remaining ≈ {_left/60:.0f} min "
                  f"({si}/{n_targets} substations)", flush=True)
        except Exception as exc:
            import traceback
            print(f"ERROR on {sub}: {exc}")
            traceback.print_exc()

    if not all_records:
        print("No results to save.")
        return

    # ── CSV summary ───────────────────────────────────────────────────────────
    summary = pd.DataFrame(all_records)
    summary.to_csv(OUTPUT_DIR / "metrics_summary.csv", index=False)

    # ── Global heatmap table ──────────────────────────────────────────────────
    plot_global_summary_table(all_records, OUTPUT_DIR)

    # ── Console summary: mean metrics per model ───────────────────────────────
    print("\n" + "=" * 90)
    print("SUMMARY — Mean metrics across all substations, ranked by VALIDATION R² (selection metric)")
    print("=" * 90)
    grouped = (
        summary.groupby("model")[["train_r2","val_r2","test_r2","test_rmse","test_mae"]]
               .mean()
               .sort_values("val_r2", ascending=False)   # selection is val-based
    )
    print(grouped.to_string(float_format=lambda x: f"{x:.4f}"))

    # ── GLOBAL single-model selection ─────────────────────────────────────────
    best_model, sel = select_global_best_model(summary)
    sel.to_csv(OUTPUT_DIR / "global_model_selection.csv", index=False)
    plot_global_selection_table(sel, best_model, OUTPUT_DIR)
    plot_model_r2_boxplot(summary, best_model, OUTPUT_DIR)
    report_best_model_per_substation(summary, best_model, OUTPUT_DIR)

    best_row = sel[sel["model"] == best_model].iloc[0]
    justification = (
        f"SINGLE MODEL SELECTED FOR THE PAPER: {best_model}\n"
        f"{'-'*60}\n"
        f"Selection criterion : highest MEAN VALIDATION R^2 (year "
        f"{VAL_YEAR}) across all "
        f"{summary['substation'].nunique()} substations\n"
        f"Mean Val R^2        : {best_row['mean_val_r2']:.4f}\n"
        f"Worst-case Val R^2  : {best_row['min_val_r2']:.4f}\n"
        f"Mean rank (val)     : {best_row['mean_rank']:.2f}  (1 = best)\n"
        f"Wins (rank-1, val)  : {int(best_row['wins'])} / "
        f"{summary['substation'].nunique()} substations\n"
        f"Reported ONCE on the untouched test set (2025-26):\n"
        f"  Mean Test R^2     : {best_row['mean_test_r2']:.4f}\n"
        f"  Mean Test RMSE    : {best_row['mean_test_rmse']:.2f} MW   "
        f"Mean Test MAE: {best_row['mean_test_mae']:.2f} MW\n"
        f"{'-'*60}\n"
        f"This model is applied uniformly to every substation in the paper.\n"
        f"Per-substation metrics: best_model_per_substation.csv\n"
    )
    (OUTPUT_DIR / "global_best_model.txt").write_text(justification,
                                                      encoding="utf-8")

    print("\n" + "=" * 90)
    print("GLOBAL MODEL SELECTION — one model for all substations")
    print("=" * 90)
    print(sel.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n" + justification)

    print("=" * 90)
    print(f"PER-SUBSTATION performance of the chosen model ({best_model}):")
    print("=" * 90)
    bm = summary[summary["model"] == best_model].sort_values("test_r2", ascending=False)
    for _, r in bm.iterrows():
        print(f"  {r['substation']:<28} Test R²={r['test_r2']:.4f}  "
              f"RMSE={r['test_rmse']:.2f}  MAE={r['test_mae']:.2f}")

    print(f"\nAll outputs written to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()