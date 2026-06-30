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

Train  : Year <= 2024
Test   : Year >= 2025  (2025 + 2026 actual vs predicted plotted)
         NOTE: the original "zoomed 2026-only" third panel is REMOVED.
         Two panels remain per model: TRAIN and TEST (2025+).

Plots produced per substation
------------------------------
  <sub>_<ModelName>_train_test.png  — Train panel + Test(2025+) panel (2 rows)
  <sub>_r2_comparison.png           — R² bar chart (train + test) all 6 models
  <sub>_rain_r2.png                 — Rain vs rolling-R² scatter, all 6 models
  <sub>_summary_table.png           — Metrics table image (R²/RMSE/MAE per model)

CSV: metrics_summary.csv

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
import matplotlib.gridspec as gridspec

# ── Configuration ─────────────────────────────────────────────────────────────
USABLE_SUBSTATIONS = [
    # "Agargaon", "Bangabhaban", "Bashundhara", "Dhanmondi", "Gulshan",
    # "Hasnabad", "Kalyanpur", "Lalbag", "Madartek", "Maniknagar", "Matual",
    # "Mirpur", "Narinda", "Savar", "Shyampur", 
    "Ullon", "Uttara", "Uttara New",
    # "Banani", "Aftabnagar", "Motijheel", "Moghbazar", "Cantonment",
    # "Pubail", "Hatirjheel", "Metrorail Uttara", "Metrorail Motijheel",
]

# LATE_START_SUBSTATIONS = [
#     "Pubail", "Hatirjheel", "Metrorail Uttara", "Metrorail Motijheel",
# ]

# Only features actually present in the CSV
EXOG_FEATURES = [
    "Holiday_type_encoded",
    "Is_Weekend",
    "Season_Encoded",
    "Temp_Max_C", "Temp_Min_C", "Temp_Mean_C",
    "Humidity_Mean_Pct", "Precip_Sum_mm", "Rain_Sum_mm", "Wind_Max_kmh",
    "Month_sin", "Month_cos", "DOW_sin", "DOW_cos",
]

FESTIVE_HOLIDAY_TYPES = {"Eid", "Durga Puja"}

WINDOW        = 14
TRAIN_END_YEAR = 2024
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

    df["DOW_num"] = dow_num
    for sub in USABLE_SUBSTATIONS:
        col = f"{sub}_Load_MW"
        if col not in df.columns or df[col].notna().sum() == 0:
            continue
        first_valid = df[col].first_valid_index()
        active = df.loc[first_valid:, col]
        active = active.interpolate(method="time", limit=7, limit_direction="both")
        if active.isna().any():
            group_med = active.groupby(
                [df.loc[active.index, "Month"], df.loc[active.index, "DOW_num"]]
            ).median()
            overall_med = active.median()
            for idx in active[active.isna()].index:
                key = (df.loc[idx, "Month"], df.loc[idx, "DOW_num"])
                active.loc[idx] = group_med.get(key, overall_med)
        df.loc[first_valid:, col] = active

    for sub in USABLE_SUBSTATIONS:
        col = f"{sub}_Load_MW"
        if col not in df.columns or df[col].notna().sum() == 0:
            continue
        lo, hi = df[col].quantile([0.005, 0.995])
        df[col] = df[col].clip(lo, hi)

    df = df.reset_index()
    return df

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

    train_mask = sub_df["Date"].dt.year <= TRAIN_END_YEAR
    test_mask  = sub_df["Date"].dt.year >= TEST_START_YEAR   # 2025 AND 2026

    if train_mask.sum() == 0:
        split_idx = int(len(sub_df) * 0.8)
        if split_idx < WINDOW + 1:
            raise ValueError(f"{substation}: not enough data for windowing.")
        train_mask = pd.Series(False, index=sub_df.index)
        train_mask.iloc[:split_idx] = True
        test_mask = ~train_mask
        print(f"  NOTE: {substation} uses per-substation 80/20 split "
              f"({train_mask.sum()} train, {test_mask.sum()} test days).")

    train_raw  = sub_df.loc[train_mask, feature_cols].values
    test_raw   = sub_df.loc[test_mask,  feature_cols].values
    test_dates = sub_df.loc[test_mask,  "Date"].values
    train_dates = sub_df.loc[train_mask, "Date"].values

    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_raw)
    test_scaled  = scaler.transform(test_raw)

    X_train, y_train = make_sequences(train_scaled, WINDOW)

    context      = train_scaled[-WINDOW:]
    extended_test = np.concatenate([context, test_scaled], axis=0)
    X_test, y_test = make_sequences(extended_test, WINDOW)

    return (X_train, y_train, X_test, y_test,
            scaler, train_dates, test_dates, feature_cols, sub_df)

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

def train_one_model(model_name, X_train, y_train, X_test, y_test, scaler, n_features):
    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    model = build_model(model_name, X_train.shape[1], n_features)
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=7, min_lr=1e-6),
    ]
    model.fit(X_train, y_train, validation_split=0.15, epochs=EPOCHS,
              batch_size=BATCH_SIZE, shuffle=False, callbacks=callbacks, verbose=0)

    y_tr_act  = inverse_target(scaler, y_train,                               n_features)
    y_tr_pred = inverse_target(scaler, model.predict(X_train, verbose=0).ravel(), n_features)
    y_te_act  = inverse_target(scaler, y_test,                                n_features)
    y_te_pred = inverse_target(scaler, model.predict(X_test,  verbose=0).ravel(), n_features)

    metrics = {
        "model":      model_name,
        "train_r2":   r2_score(y_tr_act, y_tr_pred),
        "train_rmse": float(np.sqrt(mean_squared_error(y_tr_act, y_tr_pred))),
        "train_mae":  float(mean_absolute_error(y_tr_act, y_tr_pred)),
        "test_r2":    r2_score(y_te_act, y_te_pred),
        "test_rmse":  float(np.sqrt(mean_squared_error(y_te_act, y_te_pred))),
        "test_mae":   float(mean_absolute_error(y_te_act, y_te_pred)),
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

    best_idx = int(np.argmax([m["test_r2"] for m in all_metrics]))
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
    (X_train, y_train, X_test, y_test,
     scaler, train_dates, test_dates, feature_cols, sub_df_full) = prepare_train_test(
        sub_df, load_col, substation
    )
    n_features = X_train.shape[2]
    print(f"  Train sequences: {X_train.shape}  |  Test sequences: {X_test.shape}")

    all_metrics = []
    all_results = {}

    for model_name in MODEL_NAMES:
        print(f"  [{model_name}] training...", end=" ", flush=True)
        out = train_one_model(model_name, X_train, y_train, X_test, y_test, scaler, n_features)
        metrics, y_tr_act, y_tr_pred, y_te_act, y_te_pred, model_obj = out
        metrics["substation"] = substation
        all_metrics.append(metrics)
        all_results[model_name] = (y_tr_act, y_tr_pred, y_te_act, y_te_pred, model_obj)
        print(f"Train R²={metrics['train_r2']:.3f}  Test R²={metrics['test_r2']:.3f}")

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

    # ── Contextual next-day prediction (best test-R² model) ──────────────────
    best_idx = int(np.argmax([m["test_r2"] for m in all_metrics]))
    best_name = all_metrics[best_idx]["model"]
    _, _, _, _, best_model = all_results[best_name]

    next_pred_mw, window_desc = get_context_window_for_next_day(
        df, sub_df, load_col, feature_cols, scaler, best_model, n_features
    )
    last_row  = sub_df.iloc[-1]
    next_date = last_row["Date"] + pd.Timedelta(days=1)
    context_text = (
        f"Next-day forecast for {substation} ({next_date.date()}): {next_pred_mw:.1f} MW\n"
        f"Best model: {best_name}  (Test R²={all_metrics[best_idx]['test_r2']:.4f})\n"
        f"Context window: {window_desc}\n"
        f"7-day rolling mean: {last_row.get('roll_mean_7', float('nan')):.1f} MW  "
        f"| Lag-1: {last_row.get('lag_1', float('nan')):.1f} MW\n"
        f"Temp mean: {last_row.get('Temp_Mean_C', float('nan')):.1f}°C  "
        f"| Is_Weekend: {int(last_row.get('Is_Weekend', 0))}  "
        f"| Season: {int(last_row.get('Season_Encoded', 0))}  "
        f"| Holiday_type_encoded: {last_row.get('Holiday_type_encoded', 0)}"
    )
    print(f"\n{context_text}")
    (output_dir / f"{substation.replace(' ', '_')}_next_day_context.txt").write_text(context_text)

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

    for sub in targets:
        if sub not in USABLE_SUBSTATIONS:
            print(f"Skipping '{sub}': not in usable substation list.")
            continue
        try:
            metrics_list = run_substation(df, sub, OUTPUT_DIR)
            all_records.extend(metrics_list)
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
    print("SUMMARY — Mean metrics across all substations, ranked by Test R²")
    print("=" * 90)
    grouped = (
        summary.groupby("model")[["train_r2","test_r2","test_rmse","test_mae"]]
               .mean()
               .sort_values("test_r2", ascending=False)
    )
    print(grouped.to_string(float_format=lambda x: f"{x:.4f}"))

    print("\n" + "=" * 90)
    print("PER-SUBSTATION best model (highest test R²):")
    print("=" * 90)
    for sub, grp in summary.groupby("substation"):
        best = grp.loc[grp["test_r2"].idxmax()]
        print(f"  {sub:<28} {best['model']:<22} Test R²={best['test_r2']:.4f}  "
              f"RMSE={best['test_rmse']:.2f}  MAE={best['test_mae']:.2f}")

    print(f"\nAll outputs written to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()