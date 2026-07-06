"""
05b_substation_forecast.py — PHASE 2: Operational day-ahead forecasting for
the Smart Grid pipeline case study.

Two-level research design (do not mix the levels)
--------------------------------------------------
LEVEL 1 — Research / model selection (OFFLINE, run once):
    load_prediction.py trains SIX architectures (BiLSTM, BiGRU, BiLSTM+BiGRU,
    BiGRU+BiLSTM, CNN+BiLSTM+BiGRU, CNN+BiGRU+BiLSTM) on ALL 18 BPDB
    substations, computes Test R2 / RMSE / MAE, ranks them, and writes the
    winner to forecast_outputs/global_model_selection.csv.
    Split: train 2019-2023 | validation 2024 (early stopping + SELECTION) |
    test 2025-26 (reported once, never used for any decision).
    That experiment is the thesis' model-comparison section and is NOT
    repeated here.

LEVEL 2 — Operational deployment (THIS MODULE, part of run_pipeline):
    Reads the winner from global_model_selection.csv (currently
    BiGRU+BiLSTM, mean Test R2 = 0.7456) and trains/deploys ONLY that
    architecture, ONLY for the two case-study substations of the Smart Grid
    pipeline (Kalyanpur, Dhanmondi). No model comparison happens here.
    The forecasts feed: MOLP (06) -> Priority Pool (07) -> Knowledge
    Graph (08) -> Twin Adapter / XAI.

If global_model_selection.csv is missing, this module STOPS with an
instructive error: Phase 1 must run first. (Override for experiments only:
FORCE_MODEL="BiGRU" python 05b_substation_forecast.py)

Engineering guarantees (from code review):
  * Single source of truth: architecture, cleaning, features, scaling,
    split and training loop are IMPORTED from load_prediction.py — the
    network is defined in exactly one place.
  * Persisted artefacts: trained model (.keras) + fitted scaler (.pkl) in
    models/, plus the daily forecast CSV consumed downstream.
  * Cache invalidation: fingerprint over (BPDB file, selection file, winner
    name, all hyper-parameters) in data/05b_forecast_meta.json — changing
    the data, re-running Phase 1, or changing config retrains automatically;
    otherwise this step is skipped.

Leakage note (same protocol as Phase 1): the deployed model trains on
2019-2023 only; 2024 is seen solely through early stopping (validation),
never through weight updates; 2025-2026 are fully out-of-sample. Within the
pipeline window 2024-2026, no day is a training day.

Outputs:
  models/<sub>_<model>.keras           trained checkpoint
  models/<sub>_<model>_scaler.pkl      fitted MinMaxScaler
  data/05b_<sub>_daily_forecast.csv    date, actual_mw, pred_mw
  data/05b_forecast_meta.json          fingerprint + winner + test metrics
"""

import hashlib
import json
import os

# quiet terminal: hide TF C++ chatter; must precede `import load_prediction`
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

from pathlib import Path

import numpy as np
import pandas as pd

import config as C
import load_prediction as LP          # single source of truth for the DL code

MODEL_DIR = C.BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)
META_PATH = C.DATA_DIR / "05b_forecast_meta.json"

KNOWN_MODELS = ["BiLSTM", "BiGRU", "BiLSTM+BiGRU", "BiGRU+BiLSTM",
                "CNN+BiLSTM+BiGRU", "CNN+BiGRU+BiLSTM"]

# Phase-1 output — searched in the usual locations
_SELECTION_CANDIDATES = [
    C.BASE_DIR / "forecast_outputs" / "global_model_selection.csv",
    Path("forecast_outputs/global_model_selection.csv"),
    C.BASE_DIR / "global_model_selection.csv",
    C.DATA_DIR / "global_model_selection.csv",
    Path("/mnt/user-data/uploads/global_model_selection.csv"),
]

# test/CI hook: FC_EPOCHS=3 python 05b_substation_forecast.py
if "FC_EPOCHS" in os.environ:
    LP.EPOCHS = int(os.environ["FC_EPOCHS"])


def read_selected_model():
    """Return (model_name, selection_csv_path, selection_row) from Phase 1.
    Hard-fails with instructions if the model-selection experiment has not
    been run — Phase 2 must never silently assume a winner."""
    forced = os.environ.get("FORCE_MODEL")
    sel_path = next((p for p in _SELECTION_CANDIDATES if p.exists()), None)

    if forced:
        if forced not in KNOWN_MODELS:
            raise SystemExit(f"FORCE_MODEL='{forced}' is not one of "
                             f"{KNOWN_MODELS}")
        print(f"05b: FORCE_MODEL override — deploying '{forced}' "
              f"(bypasses Phase-1 selection; for experiments only)")
        return forced, sel_path, None

    if sel_path is None:
        raise SystemExit(
            "05b: global_model_selection.csv not found.\n"
            "     Phase 1 (model selection) has not been run. Run:\n"
            "         python load_prediction.py\n"
            "     which trains all 6 architectures on all 18 BPDB "
            "substations and\n"
            "     writes forecast_outputs/global_model_selection.csv. "
            "Phase 2 deploys\n"
            "     ONLY the winner of that experiment.")

    sel = pd.read_csv(sel_path)
    if "mean_val_r2" not in sel.columns:
        raise SystemExit(
            f"05b: {sel_path} is from the OLD experimental protocol "
            f"(train <=2024, selection on TEST metrics) — it predates the "
            f"leak-free methodology and its winner must not be deployed.\n"
            f"     Re-run Phase 1 first:\n"
            f"         python Run_research.py     (or: python "
            f"load_prediction.py)\n"
            f"     which retrains all 6 architectures on all 18 substations "
            f"under the\n"
            f"     chronological split (train 2019-2023 | val 2024 | test "
            f"2025-26) and\n"
            f"     selects the winner by VALIDATION R2. Then run the "
            f"pipeline again.")
    row = sel.iloc[0]                       # Phase 1 sorts winner first
    model = str(row["model"])
    if model not in KNOWN_MODELS:
        raise SystemExit(f"05b: winner '{model}' in {sel_path} is not a "
                         f"known architecture {KNOWN_MODELS}")
    crit = ("mean Val R2" if "mean_val_r2" in row.index else "mean Test R2")
    crit_val = row.get("mean_val_r2", row.get("mean_test_r2", float("nan")))
    print(f"05b: Phase-1 winner from {sel_path.name}: {model} "
          f"({crit} = {crit_val:.4f}, mean rank = {row['mean_rank']:.2f}, "
          f"wins = {int(row['wins'])}, "
          f"reported mean Test R2 = {row.get('mean_test_r2', float('nan')):.4f})")
    return model, sel_path, row


def _slug(model_name: str) -> str:
    return model_name.lower().replace("+", "_")


def fingerprint(model_name: str, sel_path) -> str:
    """Hash of everything that should invalidate cached forecasts: the BPDB
    data file, the Phase-1 selection file (re-running the experiment with a
    new winner retriggers deployment), and every hyper-parameter."""
    h = hashlib.md5()
    h.update(str(os.path.getsize(C.BPDB_CSV)).encode())
    h.update(str(int(os.path.getmtime(C.BPDB_CSV))).encode())
    if sel_path is not None:
        h.update(str(os.path.getsize(sel_path)).encode())
        h.update(str(int(os.path.getmtime(sel_path))).encode())
    h.update(f"{model_name}|{LP.WINDOW}|{LP.EPOCHS}|{LP.SEED}|"
             f"{LP.TRAIN_END_YEAR}|{getattr(LP, 'VAL_YEAR', None)}|"
             f"{LP.BATCH_SIZE}|{LP.PATIENCE}".encode())
    return h.hexdigest()


def forecasts_up_to_date(fp: str) -> bool:
    if not META_PATH.exists():
        return False
    try:
        meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    except Exception:
        return False
    if meta.get("fingerprint") != fp:
        return False
    return all((C.DATA_DIR / f"05b_{s.lower()}_daily_forecast.csv").exists()
               for s in C.SUBSTATIONS)


def forecast_substation(df_clean: pd.DataFrame, substation: str,
                        model_name: str):
    """Train the SELECTED architecture (via load_prediction's own functions)
    and produce one-step-ahead daily peak forecasts over the substation's
    whole record."""
    load_col = f"{substation}_Load_MW"
    sub_df = LP.build_substation_frame(df_clean, substation)
    (X_tr, y_tr, X_va, y_va, X_te, y_te, scaler,
     _tr_dates, _va_dates, _te_dates, feature_cols, sub_df) = \
        LP.prepare_train_test(sub_df, load_col, substation)

    metrics, *_, model = LP.train_one_model(
        model_name, X_tr, y_tr, X_va, y_va, X_te, y_te,
        scaler, len(feature_cols))
    print(f"  {substation}: val R2={metrics['val_r2']:.3f}  "
          f"test R2={metrics['test_r2']:.3f}  "
          f"RMSE={metrics['test_rmse']:.2f}  MAE={metrics['test_mae']:.2f} MW")

    # one-step-ahead over the WHOLE record (actual lags as inputs; 2024 is
    # in-sample by construction, 2025+ out-of-sample)
    full_scaled = scaler.transform(sub_df[feature_cols].values)
    X_full, _ = LP.make_sequences(full_scaled, LP.WINDOW)
    pred_mw = LP.inverse_target(
        scaler, model.predict(X_full, verbose=0).ravel(), len(feature_cols))

    out = pd.DataFrame({
        "date": pd.to_datetime(sub_df["Date"].values[LP.WINDOW:]),
        "actual_mw": sub_df[load_col].values[LP.WINDOW:],
        "pred_mw": np.clip(pred_mw, 0, C.MAX_PLAUSIBLE_LOAD_MW).round(3),
    })

    slug = _slug(model_name)
    model.save(MODEL_DIR / f"{substation.lower()}_{slug}.keras")
    try:
        import joblib
        joblib.dump(scaler,
                    MODEL_DIR / f"{substation.lower()}_{slug}_scaler.pkl")
    except ImportError:
        print("  (joblib not installed — scaler not persisted; "
              "forecast CSV is unaffected)")
    return out, metrics


def main():
    model_name, sel_path, _row = read_selected_model()
    fp = fingerprint(model_name, sel_path)
    if forecasts_up_to_date(fp):
        print(f"05b: forecasts up to date (fingerprint {fp[:10]}…, "
              f"model {model_name}) — skipped. Delete "
              f"data/05b_forecast_meta.json to force retrain.")
        return

    print(f"05b: deploying {model_name} for the case-study substations "
          f"{C.SUBSTATIONS} (fingerprint {fp[:10]}…)")
    df_clean = LP.load_and_clean(str(C.BPDB_CSV))

    all_metrics = {}
    for sub in C.SUBSTATIONS:
        out, metrics = forecast_substation(df_clean, sub, model_name)
        path = C.DATA_DIR / f"05b_{sub.lower()}_daily_forecast.csv"
        out.to_csv(path, index=False, encoding="utf-8")
        print(f"  written: {path}  ({len(out)} days)")
        all_metrics[sub] = {k: round(float(v), 4)
                            for k, v in metrics.items()
                            if isinstance(v, (int, float, np.floating))}

    META_PATH.write_text(json.dumps({
        "fingerprint": fp,
        "phase1_selection_file": str(sel_path) if sel_path else None,
        "selected_model": model_name,
        "window": LP.WINDOW, "epochs": LP.EPOCHS, "seed": LP.SEED,
        "train_end_year": LP.TRAIN_END_YEAR,
        "deployment_test_metrics": all_metrics,
    }, indent=2), encoding="utf-8")
    print(f"05b: meta written: {META_PATH}")


if __name__ == "__main__":
    main()