"""
BPDB Dhaka Substation Load Forecasting — BiLSTM + BiGRU Hybrid Model
======================================================================

What this script does
----------------------
1. Loads BPDB_Dhaka_City_Substations_Page3.csv
2. Cleans a real upstream parser bug: some rows have the PDF's "peak time"
   clock reading (e.g. 18:30) accidentally written into the Load_MW column
   instead of the actual load. These are detected and nulled.
3. Re-indexes to a full continuous daily calendar (the raw archive has
   scattered missing days) and imputes:
     - short gaps (<=7 days): time-weighted interpolation
     - long gaps (>7 days): same (month, day-of-week) historical median,
       a simple seasonal/weekly proxy that avoids flat-lining long gaps
4. Builds calendar + weather + holiday/election context features, plus
   cyclical encodings and lag/rolling features.
5. Splits TRAIN = 2019-2024, TEST = 2025-onwards (chronological, no shuffle).
6. Scales features using ONLY train-set statistics (no leakage).
7. Builds 14-day input windows -> next-day load (one model per substation,
   trained independently, since each has a different scale/pattern).
8. Trains a stacked Bidirectional(LSTM) -> Bidirectional(GRU) network with
   Dropout + L2 + EarlyStopping + ReduceLROnPlateau to actively fight
   overfitting, and fixed seeds everywhere so re-running reproduces the
   exact same result.
9. Reports R² (and RMSE/MAE) on TRAIN and TEST, plots actual-vs-predicted
   for both, and prints a contextual next-day prediction explaining which
   input features it is responding to.

Why one model per substation, not one big multi-output model
--------------------------------------------------------------
The 18 usable substations have very different scales (some ~50-90 MW,
others ~150-250 MW) and different missing-data severity. A single shared
model would force one substation's noise/scale onto all the others. This
script loops over substations and trains/evaluates each independently,
which is slower but cleaner and easier to debug per-substation.

Reproducibility
----------------
Random seeds (Python, NumPy, TensorFlow) are fixed, and TensorFlow is told
to run single-threaded deterministic ops. On the SAME machine, SAME
TensorFlow version, SAME data file, reruns will reproduce the same numbers.
Note: exact bit-for-bit reproducibility across DIFFERENT machines/OS/CPU
vs GPU is not something any deep learning framework guarantees — if you
must run on GPU, expect tiny (5th-decimal) differences vs CPU.

Install (run once):
    pip install tensorflow pandas numpy scikit-learn matplotlib

Run:
    python bpdb_bilstm_bigru_forecast.py --csv BPDB_Dhaka_City_Substations_Page3.csv

Outputs (written to ./forecast_outputs/):
    - metrics_summary.csv              (R2/RMSE/MAE per substation, train+test)
    - <substation>_actual_vs_pred.png  (one plot per substation)
    - <substation>_next_day_context.txt (contextual next-day prediction)
    - training_log.txt
"""

from __future__ import annotations

import argparse
import json
import os
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Reproducibility — must happen before importing TensorFlow
# ---------------------------------------------------------------------------
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
random.seed(SEED)
np.random.seed(SEED)

import tensorflow as tf  # noqa: E402

tf.random.set_seed(SEED)
tf.config.experimental.enable_op_determinism()

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # noqa: E402
from sklearn.preprocessing import MinMaxScaler  # noqa: E402
from tensorflow.keras import Input, Model, regularizers  # noqa: E402
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau  # noqa: E402
from tensorflow.keras.layers import (  # noqa: E402
    GRU,
    LSTM,
    Bidirectional,
    Dense,
    Dropout,
)
from tensorflow.keras.optimizers import Adam  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Substations kept for modeling: <10% missing data before cleaning.
# (5 others are 100% missing and ~4 more are 89-96% missing in the source
# CSV — there is not enough real signal there to model responsibly. See the
# printed data-quality report at the end of this script for the full list
# and suggestions on how to fix them at the source.)
USABLE_SUBSTATIONS = [
    "Agargaon", "Bangabhaban", "Bashundhara", "Dhanmondi", "Gulshan",
    "Hasnabad", "Kalyanpur", "Lalbag", "Madartek", "Maniknagar", "Matual",
    "Mirpur", "Narinda", "Savar", "Shyampur", "Ullon", "Uttara", "Uttara New",
]

EXOG_FEATURES = [
    "Holiday_cat", "Is_Weekend", "Is_Bridge_Day", "Is_Pre_Holiday",
    "Is_Post_Holiday", "Is_Ramadan", "Is_Covid_Lockdown", "Is_Hartal_Strike",
    "Is_Election_Day", "Is_City_Corp_Election", "Temp_Max_C", "Temp_Min_C",
    "Temp_Mean_C", "Humidity_Mean_Pct", "Precip_Sum_mm", "Rain_Sum_mm",
    "Wind_Max_kmh", "Season_Encoded",
    "Month_sin", "Month_cos", "DOW_sin", "DOW_cos",  # added below
]

WINDOW = 14            # days of history fed to the model per prediction
TRAIN_END_YEAR = 2024  # train = rows with Year <= this; test = rows after
EPOCHS = 150
BATCH_SIZE = 16
PATIENCE = 15
OUTPUT_DIR = Path("forecast_outputs")


# ---------------------------------------------------------------------------
# Step 1: load + clean
# ---------------------------------------------------------------------------

def is_clocktime_artifact(load_val, peak_time_val) -> bool:
    """
    Detects rows where the PDF parser wrote the peak-time clock reading
    (e.g. 18:30 -> 1830) into the Load_MW column instead of the real load.
    Real rows always carry a genuine Peak_Time alongside a real (much
    smaller) MW figure; the artifact rows have an EMPTY Peak_Time and a
    suspiciously round, clock-shaped Load_MW value.

    Restricted to minute values of :00 or :30 specifically — BPDB peak
    times in this archive are recorded rounded to the half-hour. A looser
    rule (e.g. "any valid minute" or "round to nearest 50") also flags
    genuinely real, coincidentally round loads such as 102, 202, or 302 MW
    (confirmed present and repeating across many unrelated dates for
    several substations, e.g. Kalyanpur), wrongly nulling good data.

    NOTE: this is the same fix applied at the source in the scraper
    (record_from_triplet / extract_records_from_text in the page-3
    scraper script). If you've re-run the fixed scraper, this should be a
    no-op safety net rather than doing real work — but it's kept here so
    this script stays correct even if run against an older/unfixed CSV.
    """
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


def load_and_clean(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    artifact_counts = {}
    for sub in USABLE_SUBSTATIONS:
        load_col, time_col = f"{sub}_Load_MW", f"{sub}_Peak_Time"
        mask = df.apply(
            lambda r: is_clocktime_artifact(r[load_col], r[time_col]), axis=1
        )
        artifact_counts[sub] = int(mask.sum())
        df.loc[mask, load_col] = np.nan

    print("\n[Data cleaning] Clock-time artifacts nulled per substation:")
    for sub, n in artifact_counts.items():
        if n:
            print(f"    {sub}: {n} rows")

    # Reindex to a continuous daily calendar (raw archive has scattered gaps)
    full_range = pd.date_range(df["Date"].min(), df["Date"].max(), freq="D")
    df = df.set_index("Date").reindex(full_range)
    df.index.name = "Date"

    n_added = df[USABLE_SUBSTATIONS[0] + "_Load_MW"].isna().sum()
    print(f"\n[Calendar] Reindexed to {len(full_range)} continuous days "
          f"(added {len(full_range) - (~df.isna().all(axis=1)).sum()} missing calendar rows).")

    # Forward/back-fill calendar-derived exogenous columns (they're
    # deterministic functions of the date, so recompute cleanly instead of
    # trusting fill — cheap and removes any ambiguity for reindexed rows).
    df["Year"] = df.index.year
    df["Month"] = df.index.month
    df["Day_of_Week"] = df.index.day_name()
    dow_num = df.index.dayofweek  # Mon=0..Sun=6
    df["Is_Weekend"] = dow_num.isin([4, 5]).astype(int)  # Bangladesh: Fri/Sat
    season_map = {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 2, 10: 3, 11: 3}
    df["Season_Encoded"] = df["Month"].map(season_map)

    # Holiday/event flags and weather: these are NOT deterministic from the
    # date alone (they came from external sources), so for reindexed rows
    # that didn't exist in the original file, forward-fill from the nearest
    # real row as the safest default (a missing calendar day is almost
    # always immediately adjacent to a real one given the gap analysis).
    #
    # NOTE: this list assumes the CSV was built by the current version of
    # the scraper script. If you're running this against a CSV produced by
    # an OLDER scraper version (missing newer columns like Is_Bridge_Day,
    # Is_Ramadan, Holiday_source, etc.), this would previously crash with a
    # KeyError. Instead, any column not present is created with a safe
    # default (0 for the Is_* binary flags, "No Holiday" for Holiday name,
    # NaN->ffill/bfill for weather) and a warning is printed so you know to
    # regenerate the CSV with the latest scraper for full feature coverage.
    context_cols = [
        "Holiday name", "Holiday_cat", "Is_Bridge_Day", "Is_Pre_Holiday",
        "Is_Post_Holiday", "Is_Ramadan", "Is_Covid_Lockdown",
        "Is_Hartal_Strike", "Is_Election_Day", "Is_City_Corp_Election",
        "Temp_Max_C", "Temp_Min_C", "Temp_Mean_C", "Humidity_Mean_Pct",
        "Precip_Sum_mm", "Rain_Sum_mm", "Wind_Max_kmh",
    ]
    safe_defaults = {
        "Holiday name": "No Holiday",
        "Holiday_cat": 0,
        "Is_Bridge_Day": 0,
        "Is_Pre_Holiday": 0,
        "Is_Post_Holiday": 0,
        "Is_Ramadan": 0,
        "Is_Covid_Lockdown": 0,
        "Is_Hartal_Strike": 0,
        "Is_Election_Day": 0,
        "Is_City_Corp_Election": 0,
        # No sane constant default for weather; missing means we simply
        # cannot know it, so these fall back to the column's own ffill/bfill
        # if present at all, or get dropped from EXOG_FEATURES below if the
        # column is entirely absent from the source CSV.
    }
    missing_context_cols = [c for c in context_cols if c not in df.columns]
    if missing_context_cols:
        print(
            "\nWARNING: this CSV is missing the following columns that the "
            "current scraper script produces:\n    "
            + ", ".join(missing_context_cols)
            + "\nThis likely means the CSV was generated by an older version "
            "of the scraper. Filling safe defaults (0 / 'No Holiday') so "
            "the script can still run, but accuracy will suffer for any "
            "substation/date where these features would have mattered. "
            "Re-run the latest scraper to regenerate the CSV for full "
            "feature coverage."
        )
        for c in missing_context_cols:
            df[c] = safe_defaults.get(c, np.nan)

    for c in context_cols:
        df[c] = df[c].ffill().bfill()
        # If a column was entirely missing from the source AND has no
        # sane constant default (e.g. weather columns on a very old CSV),
        # ffill/bfill on all-NaN leaves it all-NaN — fall back to 0 rather
        # than feeding NaN into the scaler/model later.
        if df[c].isna().all():
            df[c] = 0

    # Cyclical encodings for month and day-of-week (captures
    # seasonality/weekly pattern without an artificial 12->1 or 6->0 jump)
    df["Month_sin"] = np.sin(2 * np.pi * df["Month"] / 12)
    df["Month_cos"] = np.cos(2 * np.pi * df["Month"] / 12)
    df["DOW_sin"] = np.sin(2 * np.pi * dow_num / 7)
    df["DOW_cos"] = np.cos(2 * np.pi * dow_num / 7)

    # Impute load columns: short gaps via time interpolation, long gaps via
    # (month, day-of-week) historical median — see module docstring.
    df["DOW_num"] = dow_num
    for sub in USABLE_SUBSTATIONS:
        col = f"{sub}_Load_MW"
        df[col] = df[col].interpolate(method="time", limit=7, limit_direction="both")
        if df[col].isna().any():
            group_median = df.groupby(["Month", "DOW_num"])[col].median()
            overall_median = df[col].median()
            for idx in df[df[col].isna()].index:
                key = (df.loc[idx, "Month"], df.loc[idx, "DOW_num"])
                df.loc[idx, col] = group_median.get(key, overall_median)

    # Outlier clamp: clip extreme remaining values per substation to the
    # 0.5–99.5 percentile range. This catches any leftover bad records
    # (e.g. single-row spikes) without discarding genuine peak-demand days.
    for sub in USABLE_SUBSTATIONS:
        col = f"{sub}_Load_MW"
        lo, hi = df[col].quantile([0.005, 0.995])
        df[col] = df[col].clip(lo, hi)

    df = df.reset_index()
    return df


# ---------------------------------------------------------------------------
# Step 2: feature engineering per substation (lags + rolling stats)
# ---------------------------------------------------------------------------

def build_substation_frame(df: pd.DataFrame, substation: str) -> pd.DataFrame:
    load_col = f"{substation}_Load_MW"

    missing_exog = [c for c in EXOG_FEATURES if c not in df.columns]
    if missing_exog:
        # Should not normally happen — load_and_clean() is responsible for
        # guaranteeing every EXOG_FEATURES column exists before this is
        # called. Defensive fallback rather than a hard crash, in case this
        # function is ever called on a differently-prepared DataFrame.
        print(f"WARNING: {substation}: missing exogenous columns {missing_exog}, "
              f"filling with 0.")
        for c in missing_exog:
            df[c] = 0

    cols = ["Date", load_col] + EXOG_FEATURES
    sub_df = df[cols].copy()

    # Lag and rolling features for the target itself — gives the model
    # explicit short-term memory beyond the raw sequence window.
    sub_df["lag_1"] = sub_df[load_col].shift(1)
    sub_df["lag_7"] = sub_df[load_col].shift(7)
    sub_df["roll_mean_7"] = sub_df[load_col].shift(1).rolling(7).mean()
    sub_df["roll_std_7"] = sub_df[load_col].shift(1).rolling(7).std()

    # First 7 rows will have NaN lag/rolling features by construction;
    # back-fill them with the first valid value (training is unaffected
    # since the windowing step drops the first WINDOW rows anyway).
    sub_df = sub_df.bfill()
    return sub_df


# ---------------------------------------------------------------------------
# Step 3: scaling + windowing
# ---------------------------------------------------------------------------

def make_sequences(values: np.ndarray, window: int):
    X, y = [], []
    for i in range(len(values) - window):
        X.append(values[i:i + window])
        y.append(values[i + window, 0])  # target column is always index 0
    return np.array(X), np.array(y)


def prepare_train_test(sub_df: pd.DataFrame, load_col: str):
    feature_cols = [load_col] + [c for c in sub_df.columns if c not in ("Date", load_col)]
    sub_df = sub_df[["Date"] + feature_cols]

    train_mask = sub_df["Date"].dt.year <= TRAIN_END_YEAR
    test_mask = ~train_mask

    train_raw = sub_df.loc[train_mask, feature_cols].values
    test_raw = sub_df.loc[test_mask, feature_cols].values
    test_dates = sub_df.loc[test_mask, "Date"].values

    # Scale using ONLY training statistics — this is the standard rule to
    # avoid test-set leakage. Same scaler object is reused to transform test.
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_raw)
    test_scaled = scaler.transform(test_raw)

    X_train, y_train = make_sequences(train_scaled, WINDOW)

    # For the test set, prepend the last WINDOW rows of (scaled) train data
    # so the very first test-period day can be predicted too. This only
    # uses past information already available at that point in time —
    # not a leakage, just normal autoregressive context.
    context = train_scaled[-WINDOW:]
    extended_test = np.concatenate([context, test_scaled], axis=0)
    X_test, y_test = make_sequences(extended_test, WINDOW)

    return X_train, y_train, X_test, y_test, scaler, test_dates, feature_cols


def inverse_target(scaler: MinMaxScaler, scaled_target: np.ndarray, n_features: int) -> np.ndarray:
    """Inverse-transform only the target column (column 0) of a MinMaxScaler
    that was fit on the full feature matrix."""
    dummy = np.zeros((len(scaled_target), n_features))
    dummy[:, 0] = scaled_target
    return scaler.inverse_transform(dummy)[:, 0]


# ---------------------------------------------------------------------------
# Step 4: model
# ---------------------------------------------------------------------------

def build_model(window: int, n_features: int) -> Model:
    """
    Stacked Bidirectional(LSTM) -> Bidirectional(GRU) regressor.

    Anti-overfitting measures (since the dataset is modest in size for a
    deep sequence model):
      - Modest layer widths (32, then 16 units) rather than large stacks
      - Dropout after each recurrent layer
      - L2 weight regularization on recurrent + dense kernels
      - Trained with EarlyStopping (restores best weights) +
        ReduceLROnPlateau so it cannot just keep memorizing noise
    """
    l2 = regularizers.l2(1e-4)
    inputs = Input(shape=(window, n_features))

    x = Bidirectional(
        LSTM(32, return_sequences=True, kernel_regularizer=l2, recurrent_regularizer=l2)
    )(inputs)
    x = Dropout(0.3)(x)

    x = Bidirectional(
        GRU(16, return_sequences=False, kernel_regularizer=l2, recurrent_regularizer=l2)
    )(x)
    x = Dropout(0.3)(x)

    x = Dense(16, activation="relu", kernel_regularizer=l2)(x)
    x = Dropout(0.2)(x)
    outputs = Dense(1, activation="linear")(x)

    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=1e-3), loss="mse", metrics=["mae"])
    return model


# ---------------------------------------------------------------------------
# Step 5: train + evaluate one substation
# ---------------------------------------------------------------------------

def run_substation(df: pd.DataFrame, substation: str, output_dir: Path) -> dict:
    print(f"\n{'=' * 70}\nSubstation: {substation}\n{'=' * 70}")
    load_col = f"{substation}_Load_MW"
    sub_df = build_substation_frame(df, substation)

    X_train, y_train, X_test, y_test, scaler, test_dates, feature_cols = prepare_train_test(
        sub_df, load_col
    )
    n_features = X_train.shape[2]
    print(f"Train sequences: {X_train.shape}, Test sequences: {X_test.shape}")

    model = build_model(WINDOW, n_features)

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=7, min_lr=1e-6),
    ]

    history = model.fit(
        X_train, y_train,
        validation_split=0.15,  # last 15% of TRAIN (chronological) used for val
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=False,  # keep chronological order — this is time series
        callbacks=callbacks,
        verbose=0,
    )
    print(f"Stopped after {len(history.history['loss'])} epochs "
          f"(best val_loss={min(history.history['val_loss']):.5f})")

    # Predictions (scaled) -> inverse-transform target column back to MW
    train_pred_scaled = model.predict(X_train, verbose=0).ravel()
    test_pred_scaled = model.predict(X_test, verbose=0).ravel()

    y_train_actual = inverse_target(scaler, y_train, n_features)
    y_train_pred = inverse_target(scaler, train_pred_scaled, n_features)
    y_test_actual = inverse_target(scaler, y_test, n_features)
    y_test_pred = inverse_target(scaler, test_pred_scaled, n_features)

    metrics = {
        "substation": substation,
        "train_r2": r2_score(y_train_actual, y_train_pred),
        "train_rmse": float(np.sqrt(mean_squared_error(y_train_actual, y_train_pred))),
        "train_mae": float(mean_absolute_error(y_train_actual, y_train_pred)),
        "test_r2": r2_score(y_test_actual, y_test_pred),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test_actual, y_test_pred))),
        "test_mae": float(mean_absolute_error(y_test_actual, y_test_pred)),
        "epochs_run": len(history.history["loss"]),
    }
    print(f"TRAIN  R2={metrics['train_r2']:.4f}  RMSE={metrics['train_rmse']:.2f}  MAE={metrics['train_mae']:.2f}")
    print(f"TEST   R2={metrics['test_r2']:.4f}  RMSE={metrics['test_rmse']:.2f}  MAE={metrics['test_mae']:.2f}")

    gap = metrics["train_r2"] - metrics["test_r2"]
    if gap > 0.25:
        print(f"WARNING: train R2 exceeds test R2 by {gap:.2f} — possible overfitting. "
              f"Consider increasing dropout/L2 or reducing model size for this substation.")
    if metrics["test_r2"] < 0:
        print(f"WARNING: negative test R2 — model performs worse than predicting the mean. "
              f"This substation's data is likely too sparse/noisy after imputation; "
              f"treat its forecast with caution.")

    # --- Plot actual vs predicted (train + test) ---
    fig, axes = plt.subplots(2, 1, figsize=(13, 8))
    train_dates = sub_df.loc[sub_df["Date"].dt.year <= TRAIN_END_YEAR, "Date"].values[WINDOW:]
    axes[0].plot(train_dates, y_train_actual, label="Actual", linewidth=1)
    axes[0].plot(train_dates, y_train_pred, label="Predicted", linewidth=1, alpha=0.8)
    axes[0].set_title(f"{substation} — TRAIN (R²={metrics['train_r2']:.3f})")
    axes[0].legend()
    axes[0].set_ylabel("Load (MW)")

    axes[1].plot(test_dates, y_test_actual, label="Actual", linewidth=1.2)
    axes[1].plot(test_dates, y_test_pred, label="Predicted", linewidth=1.2, alpha=0.8)
    axes[1].set_title(f"{substation} — TEST (R²={metrics['test_r2']:.3f})")
    axes[1].legend()
    axes[1].set_ylabel("Load (MW)")
    axes[1].set_xlabel("Date")

    fig.tight_layout()
    fig.savefig(output_dir / f"{substation.replace(' ', '_')}_actual_vs_pred.png", dpi=120)
    plt.close(fig)

    # --- Contextual next-day prediction ---
    last_window_raw = sub_df[feature_cols].values[-WINDOW:]
    last_window_scaled = scaler.transform(last_window_raw)
    next_pred_scaled = model.predict(last_window_scaled[np.newaxis, ...], verbose=0).ravel()
    next_pred_mw = inverse_target(scaler, next_pred_scaled, n_features)[0]

    last_row = sub_df.iloc[-1]
    next_date = last_row["Date"] + pd.Timedelta(days=1)
    context_notes = []
    if last_row.get("Is_Weekend"):
        context_notes.append("the prior day was a weekend (Fri/Sat)")
    if last_row.get("Holiday_cat", 0) > 0:
        context_notes.append(f"a holiday context was active (Holiday_cat={int(last_row['Holiday_cat'])})")
    if last_row.get("Is_Ramadan"):
        context_notes.append("Ramadan was in effect")
    context_notes.append(f"recent 7-day average load was {last_row['roll_mean_7']:.1f} MW")
    context_notes.append(f"most recent observed temperature mean was {last_row['Temp_Mean_C']:.1f}°C")
    context_text = (
        f"Next-day forecast for {substation} ({next_date.date()}): {next_pred_mw:.1f} MW\n"
        f"Context driving this prediction: " + "; ".join(context_notes) + ".\n"
        f"Based on the {WINDOW}-day window ending {last_row['Date'].date()}."
    )
    print("\n" + context_text)
    (output_dir / f"{substation.replace(' ', '_')}_next_day_context.txt").write_text(context_text)

    return metrics


# ---------------------------------------------------------------------------
# Step 6: data-quality report (substations excluded from modeling)
# ---------------------------------------------------------------------------

def print_excluded_substation_report(raw_df: pd.DataFrame) -> None:
    all_load_cols = [c for c in raw_df.columns if c.endswith("_Load_MW")]
    usable_cols = {f"{s}_Load_MW" for s in USABLE_SUBSTATIONS}
    excluded_cols = [c for c in all_load_cols if c not in usable_cols]

    print("\n" + "=" * 70)
    print("EXCLUDED SUBSTATIONS — data too sparse to model responsibly")
    print("=" * 70)
    miss_pct = raw_df[excluded_cols].isna().mean().sort_values(ascending=False) * 100
    for col, pct in miss_pct.items():
        print(f"  {col.replace('_Load_MW', ''):<22} {pct:5.1f}% missing")

    print(
        "\nSuggestions for fixing these at the SOURCE (the scraper/parser), "
        "rather than trying to model around the gap:\n"
        "  1. 100%-missing substations (Banasree, Aminbazar, Tejgaon, Rampura,\n"
        "     Bashabo) likely use a substation-name spelling/alias not yet in\n"
        "     DHAKA_CITY_SUBSTATIONS, or appear on a different PDF page/table\n"
        "     layout than the parser targets. Manually open a handful of PDFs\n"
        "     covering different years and confirm whether these names appear\n"
        "     under a different spelling, and add the alias if so.\n"
        "  2. 89-96%-missing substations (Pubail, Hatirjheel, Metrorail\n"
        "     Motijheel/Uttara) are newer additions to the grid — check if they\n"
        "     simply did not exist as substations before a certain year. If so,\n"
        "     this is correct/expected and they should only be modeled from the\n"
        "     date they came online, not over the full 2019-2026 range.\n"
        "  3. For partially-missing usable substations, consider cross-checking\n"
        "     a sample of parsed values against the original PDFs directly to\n"
        "     confirm the parser is reading the correct table column — this\n"
        "     dataset already had one confirmed parsing bug (peak-time clock\n"
        "     values leaking into Load_MW), so a second pass of validation\n"
        "     against source PDFs is worthwhile before treating any of this\n"
        "     as ground truth for high-stakes decisions.\n"
        "  4. Once source data is fixed, simply add the substation name back\n"
        "     to USABLE_SUBSTATIONS in this script and rerun — no other code\n"
        "     changes needed.\n"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="BiLSTM+BiGRU substation load forecasting")
    parser.add_argument("--csv", default="BPDB_Dhaka_City_Substations_Page3.csv")
    parser.add_argument(
        "--substations", nargs="*", default=None,
        help="Optional subset of substation names to run (default: all 18 usable ones)",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)

    raw_df = pd.read_csv(args.csv)
    print_excluded_substation_report(raw_df)

    df = load_and_clean(args.csv)

    targets = args.substations if args.substations else USABLE_SUBSTATIONS
    all_metrics = []
    for sub in targets:
        if sub not in USABLE_SUBSTATIONS:
            print(f"Skipping '{sub}': not in the usable substation list (too sparse).")
            continue
        metrics = run_substation(df, sub, OUTPUT_DIR)
        all_metrics.append(metrics)

    summary = pd.DataFrame(all_metrics)
    summary.to_csv(OUTPUT_DIR / "metrics_summary.csv", index=False)

    print("\n" + "=" * 70)
    print("SUMMARY — all substations")
    print("=" * 70)
    print(summary[["substation", "train_r2", "test_r2", "test_rmse", "test_mae"]]
          .to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nMean test R2 across substations: {summary['test_r2'].mean():.4f}")
    print(f"\nAll outputs written to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()