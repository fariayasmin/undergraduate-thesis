"""
multihorizon.py — direct K-step-ahead forecasting head for the substation
peak forecaster (Branch B).

Implements the parts of the mathematical model that the day-ahead code did not:

  Eq. 13  direct multi-horizon head, p̂_d = (p̂_{d+1|d}, …, p̂_{d+K|d}) ∈ R^K
  Eq. 14  horizon-weighted training objective with L2 regularisation
  Eq. 15  per-horizon residual scale σ_k, estimated on the validation year
  Eq. 16  robust planning peak p̃_{i,d+k} = p̂_{i,d+k|d} + z_β · σ_{i,k}

Design notes
------------
* DIRECT, not recursive.  A recursive rollout feeds its own output back in and
  compounds its own error; the direct head emits all K leads from one forward
  pass, so lead-k error does not inherit lead-(k-1) error.  This is what the
  model document specifies and it is what the reported skill curve measures.

* NO FUTURE COVARIATES.  A window ending on day d contains only days ≤ d, and
  the head maps that window to days d+1 … d+K.  Nothing dated after d enters
  the input, so the leak-free protocol of the day-ahead code is preserved
  unchanged at every horizon.

* σ_k is estimated on the VALIDATION year only (never on test), consistent
  with the rest of the protocol: the test set is touched once, for reporting.

This module deliberately re-uses load_prediction for cleaning, feature
engineering, scaling and the recurrent trunk, so the architecture is still
defined in exactly one place.

NOTE: this module has not been executed — the development environment has no
TensorFlow — so it should be run once on a single substation before the full
18-substation Phase-1 sweep.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Input, Model, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (
    GRU, LSTM, Bidirectional, Conv1D, Dense, Dropout, MaxPooling1D,
)
from tensorflow.keras.optimizers import Adam

import load_prediction as LP

HORIZON = 15                 # K, Eq. 13
Z_BETA = 1.2815515655446004  # z_0.90, Eq. 16


def horizon_weights(horizon: int = HORIZON, mode: str = "uniform"):
    """ω_k in Eq. 14.

    'uniform'  — every lead contributes equally.  This is the default because
                 the planning value of the horizon is the whole 15-day window
                 (battery pre-positioning and scheduled transfers use the deep
                 leads), not lead 1 alone.
    'decaying' — ω_k ∝ 1/k, if a run should prioritise near-term accuracy.
    """
    if mode == "uniform":
        w = np.ones(horizon, dtype="float32")
    elif mode == "decaying":
        w = 1.0 / np.arange(1, horizon + 1, dtype="float32")
    else:
        raise ValueError(f"unknown weighting mode: {mode}")
    return w / w.sum() * horizon          # mean weight 1 → loss stays comparable


def make_sequences_multi(values: np.ndarray, window: int, horizon: int):
    """Windows of `window` past days → the next `horizon` daily peaks.

    Returns X (n, window, n_features) and y (n, horizon).  The last usable
    window is the one that still has `horizon` days of target after it, so
    n = len(values) - window - horizon + 1.
    """
    n = len(values) - window - horizon + 1
    if n <= 0:
        raise ValueError(
            f"series too short: {len(values)} rows cannot yield a "
            f"{window}-day window plus a {horizon}-day target")
    X = np.empty((n, window, values.shape[1]), dtype="float32")
    y = np.empty((n, horizon), dtype="float32")
    for i in range(n):
        X[i] = values[i:i + window]
        y[i] = values[i + window:i + window + horizon, 0]
    return X, y


def build_model_multi(model_name: str, window: int, n_features: int,
                      horizon: int = HORIZON) -> Model:
    """Identical recurrent trunk to load_prediction.build_model; only the head
    differs (Dense(horizon) instead of Dense(1))."""
    l2 = regularizers.l2(1e-4)
    inp = Input(shape=(window, n_features))

    if model_name == "BiLSTM":
        x = Bidirectional(LSTM(32, return_sequences=True, kernel_regularizer=l2,
                               recurrent_regularizer=l2))(inp)
        x = Dropout(0.3)(x)
        x = Bidirectional(LSTM(16, return_sequences=False,
                               kernel_regularizer=l2,
                               recurrent_regularizer=l2))(x)
    elif model_name == "BiGRU":
        x = Bidirectional(GRU(32, return_sequences=True, kernel_regularizer=l2,
                              recurrent_regularizer=l2))(inp)
        x = Dropout(0.3)(x)
        x = Bidirectional(GRU(16, return_sequences=False, kernel_regularizer=l2,
                              recurrent_regularizer=l2))(x)
    elif model_name == "BiLSTM+BiGRU":
        x = Bidirectional(LSTM(32, return_sequences=True, kernel_regularizer=l2,
                               recurrent_regularizer=l2))(inp)
        x = Dropout(0.3)(x)
        x = Bidirectional(GRU(16, return_sequences=False, kernel_regularizer=l2,
                              recurrent_regularizer=l2))(x)
    elif model_name == "BiGRU+BiLSTM":
        x = Bidirectional(GRU(32, return_sequences=True, kernel_regularizer=l2,
                              recurrent_regularizer=l2))(inp)
        x = Dropout(0.3)(x)
        x = Bidirectional(LSTM(16, return_sequences=False,
                               kernel_regularizer=l2,
                               recurrent_regularizer=l2))(x)
    elif model_name == "CNN+BiLSTM+BiGRU":
        x = Conv1D(32, 3, activation="relu", padding="same",
                   kernel_regularizer=l2)(inp)
        x = MaxPooling1D(2)(x)
        x = Bidirectional(LSTM(32, return_sequences=True, kernel_regularizer=l2,
                               recurrent_regularizer=l2))(x)
        x = Dropout(0.3)(x)
        x = Bidirectional(GRU(16, return_sequences=False, kernel_regularizer=l2,
                              recurrent_regularizer=l2))(x)
    elif model_name == "CNN+BiGRU+BiLSTM":
        x = Conv1D(32, 3, activation="relu", padding="same",
                   kernel_regularizer=l2)(inp)
        x = MaxPooling1D(2)(x)
        x = Bidirectional(GRU(32, return_sequences=True, kernel_regularizer=l2,
                              recurrent_regularizer=l2))(x)
        x = Dropout(0.3)(x)
        x = Bidirectional(LSTM(16, return_sequences=False,
                               kernel_regularizer=l2,
                               recurrent_regularizer=l2))(x)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    x = Dropout(0.3)(x)
    x = Dense(16, activation="relu", kernel_regularizer=l2)(x)
    x = Dropout(0.2)(x)
    out = Dense(horizon, activation="linear")(x)      # Eq. 13

    m = Model(inp, out)
    w = horizon_weights(horizon)

    def weighted_mse(y_true, y_pred):                 # Eq. 14
        import tensorflow as tf
        se = tf.square(y_true - y_pred)
        return tf.reduce_mean(se * tf.constant(w), axis=-1)

    m.compile(optimizer=Adam(learning_rate=1e-3), loss=weighted_mse,
              metrics=["mae"])
    return m


def prepare_multi(sub_df, load_col: str, substation: str = "",
                  horizon: int = HORIZON):
    """Chronological split + scaling + multi-horizon windowing.

    Mirrors LP.prepare_train_test exactly, including the leak-free rules: the
    scaler is fitted on train only, and validation/test sequences are seeded
    from the last WINDOW days of the preceding split so no context window ever
    reaches forward in time.
    """
    feature_cols = [load_col] + [c for c in sub_df.columns
                                 if c not in ("Date", load_col)]
    sub_df = sub_df[["Date"] + feature_cols].copy()

    year = sub_df["Date"].dt.year
    train_mask = year <= LP.TRAIN_END_YEAR
    val_mask = year == LP.VAL_YEAR
    test_mask = year >= LP.TEST_START_YEAR

    if train_mask.sum() < LP.WINDOW + horizon + 1:
        raise ValueError(
            f"{substation}: not enough training data for a {LP.WINDOW}-day "
            f"window plus a {horizon}-day horizon")

    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(sub_df.loc[train_mask, feature_cols].values)
    val_scaled = scaler.transform(sub_df.loc[val_mask, feature_cols].values)
    test_scaled = scaler.transform(sub_df.loc[test_mask, feature_cols].values)

    X_tr, y_tr = make_sequences_multi(train_scaled, LP.WINDOW, horizon)
    ext_val = np.concatenate([train_scaled[-LP.WINDOW:], val_scaled], axis=0)
    X_va, y_va = make_sequences_multi(ext_val, LP.WINDOW, horizon)
    pre_test = np.concatenate([train_scaled, val_scaled], axis=0)[-LP.WINDOW:]
    ext_test = np.concatenate([pre_test, test_scaled], axis=0)
    X_te, y_te = make_sequences_multi(ext_test, LP.WINDOW, horizon)

    return (X_tr, y_tr, X_va, y_va, X_te, y_te, scaler, feature_cols, sub_df)


def inverse_multi(scaler, scaled: np.ndarray, n_features: int) -> np.ndarray:
    """Invert MinMax scaling on a (n, horizon) target block."""
    flat = scaled.reshape(-1)
    dummy = np.zeros((len(flat), n_features))
    dummy[:, 0] = flat
    return scaler.inverse_transform(dummy)[:, 0].reshape(scaled.shape)


def horizon_metrics(y_true_mw: np.ndarray, y_pred_mw: np.ndarray) -> list:
    """Per-lead R², RMSE, MAE — the §2.5 skill curve, reported not asserted."""
    out = []
    for k in range(y_true_mw.shape[1]):
        yt, yp = y_true_mw[:, k], y_pred_mw[:, k]
        out.append(dict(
            horizon_days=k + 1,
            r2=float(r2_score(yt, yp)),
            rmse=float(np.sqrt(mean_squared_error(yt, yp))),
            mae=float(mean_absolute_error(yt, yp)),
        ))
    return out


def sigma_by_horizon(y_val_mw: np.ndarray, y_val_pred_mw: np.ndarray) -> np.ndarray:
    """σ_k, Eq. 15 — sd of the validation residual at each lead.

    Estimated on the validation year only.  Empirically σ_1 ≤ … ≤ σ_K; the
    function does NOT enforce monotonicity, so that any violation shows up in
    the reported curve instead of being hidden.
    """
    return (y_val_mw - y_val_pred_mw).std(axis=0, ddof=1)


def robust_peak(pred_mw: np.ndarray, sigma_k: np.ndarray,
                z_beta: float = Z_BETA) -> np.ndarray:
    """p̃ = p̂ + z_β · σ_k, Eq. 16 — the uncertainty-inflated planning peak.

    Widening with k is the point: it is what justifies holding battery reserve
    rather than acting on a distant point forecast.
    """
    return pred_mw + z_beta * sigma_k[None, :]


def train_multi(model_name, X_tr, y_tr, X_va, y_va, scaler, n_features,
                horizon: int = HORIZON):
    """Train one multi-horizon model. Early stopping on validation loss, same
    patience and schedule as the day-ahead protocol."""
    import tensorflow as tf
    tf.random.set_seed(LP.SEED)
    np.random.seed(LP.SEED)

    model = build_model_multi(model_name, X_tr.shape[1], n_features, horizon)
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=LP.PATIENCE,
                      restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=7,
                          min_lr=1e-6),
    ]
    hist = model.fit(X_tr, y_tr, validation_data=(X_va, y_va),
                     epochs=LP.EPOCHS, batch_size=LP.BATCH_SIZE, shuffle=False,
                     callbacks=callbacks, verbose=0)
    return model, len(hist.history["loss"])