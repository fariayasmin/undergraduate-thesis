"""
05b_substation_forecast.py — PHASE 2: operational 15-day rolling-horizon
substation peak forecasting.

CHANGE FROM THE DAY-AHEAD VERSION
---------------------------------
The previous module emitted a single next-day value (Dense(1)); the
mathematical model specifies a direct K=15 head (Eq. 13) with per-horizon
uncertainty (Eq. 15) and a robust planning peak (Eq. 16). This module now
deploys that head via multihorizon.py. Backwards compatibility is preserved:
`pred_mw` is still written and still means the lead-1 forecast, so
06_molp_optimizer.py and 08_knowledge_graph.py continue to work unchanged
while the deeper leads become available to the network layer.

Two-level research design (unchanged)
-------------------------------------
LEVEL 1 — model selection (offline, load_prediction.py / Run_research.py):
    six architectures x 18 substations, split train 2019-2023 | validation
    2024 (early stopping + SELECTION) | test 2025-26 (reported once).
LEVEL 2 — deployment (this module): trains and deploys ONLY the Phase-1
    winner, ONLY for the case-study substations.

IMPORTANT: Phase 1 must be re-run under the multi-horizon objective before its
winner is cited alongside 15-day results. A model selected on lead-1 skill is
not necessarily the best model over a 15-day window, and claiming otherwise
would be a selection/evaluation mismatch. Until then this module will deploy the
lead-1 winner and say so.

Outputs (data/):
  05b_<sub>_daily_forecast.csv   date, actual_mw, pred_mw (= lead 1),
                                 pred_mw_k1..k15, robust_mw_k1..k15
  05c_<sub>_horizon_skill.csv    per-lead R2 / RMSE / MAE / sigma_k
  05b_forecast_meta.json         fingerprint, winner, per-lead test metrics
  models/<sub>_<model>_k15.keras trained checkpoint
"""

import hashlib
import json
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

from pathlib import Path

import numpy as np
import pandas as pd

import config as C
import load_prediction as LP
import multihorizon as MH

MODEL_DIR = C.BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)
META_PATH = C.DATA_DIR / "05b_forecast_meta.json"

KNOWN_MODELS = ["BiLSTM", "BiGRU", "BiLSTM+BiGRU", "BiGRU+BiLSTM",
                "CNN+BiLSTM+BiGRU", "CNN+BiGRU+BiLSTM"]

_SELECTION_CANDIDATES = [
    C.BASE_DIR / "forecast_outputs" / "global_model_selection.csv",
    Path("forecast_outputs/global_model_selection.csv"),
    C.BASE_DIR / "global_model_selection.csv",
    C.DATA_DIR / "global_model_selection.csv",
]

if "FC_EPOCHS" in os.environ:
    LP.EPOCHS = int(os.environ["FC_EPOCHS"])


def read_selected_model():
    forced = os.environ.get("FORCE_MODEL")
    sel_path = next((p for p in _SELECTION_CANDIDATES if p.exists()), None)
    if forced:
        if forced not in KNOWN_MODELS:
            raise SystemExit(f"FORCE_MODEL='{forced}' not in {KNOWN_MODELS}")
        print(f"05b: FORCE_MODEL override — deploying '{forced}'")
        return forced, sel_path
    if sel_path is None:
        raise SystemExit(
            "05b: global_model_selection.csv not found. Run Phase 1 first:\n"
            "         python Run_research.py")
    sel = pd.read_csv(sel_path)
    if "mean_val_r2" not in sel.columns:
        raise SystemExit(
            f"05b: {sel_path} predates the leak-free protocol. Re-run "
            f"Run_research.py before deploying its winner.")
    row = sel.iloc[0]
    model = str(row["model"])
    if model not in KNOWN_MODELS:
        raise SystemExit(f"05b: winner '{model}' is not a known architecture")
    print(f"05b: Phase-1 winner: {model} "
          f"(mean Val R2 = {row['mean_val_r2']:.4f}, "
          f"wins = {int(row['wins'])})")
    if not sel.columns.str.contains("horizon").any():
        print("05b: NOTE — the selection file was produced under the lead-1 "
              "objective. Its winner is deployed here with a 15-day head, but "
              "Phase 1 should be re-run multi-horizon before 15-day skill is "
              "attributed to a 'selected' architecture.")
    return model, sel_path


def _slug(name: str) -> str:
    return name.lower().replace("+", "_")


def fingerprint(model_name: str, sel_path) -> str:
    h = hashlib.md5()
    h.update(str(os.path.getsize(C.BPDB_CSV)).encode())
    h.update(str(int(os.path.getmtime(C.BPDB_CSV))).encode())
    if sel_path is not None:
        h.update(str(os.path.getsize(sel_path)).encode())
        h.update(str(int(os.path.getmtime(sel_path))).encode())
    h.update(f"{model_name}|{LP.WINDOW}|{MH.HORIZON}|{LP.EPOCHS}|{LP.SEED}|"
             f"{LP.TRAIN_END_YEAR}|{LP.VAL_YEAR}|{LP.BATCH_SIZE}|"
             f"{LP.PATIENCE}|{MH.Z_BETA}".encode())
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
               and (C.DATA_DIR / f"05c_{s.lower()}_horizon_skill.csv").exists()
               for s in C.SUBSTATIONS)


def forecast_substation(df_clean: pd.DataFrame, substation: str,
                        model_name: str):
    load_col = f"{substation}_Load_MW"
    sub_df = LP.build_substation_frame(df_clean, substation)
    (X_tr, y_tr, X_va, y_va, X_te, y_te,
     scaler, feature_cols, sub_df) = MH.prepare_multi(sub_df, load_col,
                                                      substation)
    n_feat = len(feature_cols)

    model, epochs_run = MH.train_multi(model_name, X_tr, y_tr, X_va, y_va,
                                       scaler, n_feat)

    va_true = MH.inverse_multi(scaler, y_va, n_feat)
    va_pred = MH.inverse_multi(scaler, model.predict(X_va, verbose=0), n_feat)
    te_true = MH.inverse_multi(scaler, y_te, n_feat)
    te_pred = MH.inverse_multi(scaler, model.predict(X_te, verbose=0), n_feat)

    sigma_k = MH.sigma_by_horizon(va_true, va_pred)          # Eq. 15
    skill = pd.DataFrame(MH.horizon_metrics(te_true, te_pred))
    skill.insert(0, "substation", substation)
    skill["sigma_k_mw"] = np.round(sigma_k, 4)
    skill["z_beta_sigma_mw"] = np.round(MH.Z_BETA * sigma_k, 4)

    print(f"  {substation}: lead-1 test R2={skill.r2.iloc[0]:.3f}  "
          f"lead-7 R2={skill.r2.iloc[6]:.3f}  "
          f"lead-15 R2={skill.r2.iloc[-1]:.3f}  "
          f"| sigma_1={sigma_k[0]:.2f} → sigma_15={sigma_k[-1]:.2f} MW  "
          f"({epochs_run} epochs)")

    # forecasts issued on every day of the record
    full_scaled = scaler.transform(sub_df[feature_cols].values)
    Xf, _ = MH.make_sequences_multi(full_scaled, LP.WINDOW, MH.HORIZON)
    pred = np.clip(MH.inverse_multi(scaler, model.predict(Xf, verbose=0),
                                    n_feat), 0, C.MAX_PLAUSIBLE_LOAD_MW)
    robust = np.clip(MH.robust_peak(pred, sigma_k), 0, C.MAX_PLAUSIBLE_LOAD_MW)

    # issue day d = the last day inside each window
    issue = pd.to_datetime(sub_df["Date"].values[LP.WINDOW - 1:
                                                 LP.WINDOW - 1 + len(pred)])
    out = pd.DataFrame({"date": issue})
    # lead-1 target day == issue day + 1; kept under the original column name
    # so 06 and 08 need no change
    actual = sub_df[load_col].values
    out["actual_mw"] = actual[LP.WINDOW:LP.WINDOW + len(pred)]
    out["pred_mw"] = pred[:, 0].round(3)
    for k in range(MH.HORIZON):
        out[f"pred_mw_k{k+1}"] = pred[:, k].round(3)
        out[f"robust_mw_k{k+1}"] = robust[:, k].round(3)

    model.save(MODEL_DIR / f"{substation.lower()}_{_slug(model_name)}_k15.keras")
    try:
        import joblib
        joblib.dump(scaler, MODEL_DIR /
                    f"{substation.lower()}_{_slug(model_name)}_k15_scaler.pkl")
    except ImportError:
        print("  (joblib missing — scaler not persisted)")
    return out, skill


def main():
    model_name, sel_path = read_selected_model()
    fp = fingerprint(model_name, sel_path)
    if forecasts_up_to_date(fp):
        print(f"05b: forecasts up to date (fingerprint {fp[:10]}…) — skipped.")
        return

    print(f"05b: deploying {model_name} with K={MH.HORIZON} for "
          f"{C.SUBSTATIONS} (fingerprint {fp[:10]}…)")
    df_clean = LP.load_and_clean(str(C.BPDB_CSV))

    meta = {}
    for sub in C.SUBSTATIONS:
        out, skill = forecast_substation(df_clean, sub, model_name)
        out.to_csv(C.DATA_DIR / f"05b_{sub.lower()}_daily_forecast.csv",
                   index=False, encoding="utf-8")
        skill.to_csv(C.DATA_DIR / f"05c_{sub.lower()}_horizon_skill.csv",
                     index=False, encoding="utf-8")
        meta[sub] = {"lead1_test_r2": float(skill.r2.iloc[0]),
                     "lead15_test_r2": float(skill.r2.iloc[-1]),
                     "sigma_k_mw": [float(v) for v in skill.sigma_k_mw]}

    META_PATH.write_text(json.dumps({
        "fingerprint": fp,
        "phase1_selection_file": str(sel_path) if sel_path else None,
        "selected_model": model_name,
        "window": LP.WINDOW, "horizon": MH.HORIZON, "z_beta": MH.Z_BETA,
        "epochs": LP.EPOCHS, "seed": LP.SEED,
        "train_end_year": LP.TRAIN_END_YEAR,
        "per_substation": meta,
    }, indent=2), encoding="utf-8")
    print(f"05b: meta written: {META_PATH}")


if __name__ == "__main__":
    main()