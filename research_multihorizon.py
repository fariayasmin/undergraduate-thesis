"""
research_multihorizon.py — PHASE 1 under the MULTI-HORIZON objective.

WHAT CHANGED AND WHY
--------------------
The original Phase 1 (`load_prediction.py`) trained every architecture with a
scalar target, `Dense(1)`, and selected the winner on one-day-ahead validation
R2. The deployed forecaster now emits K = 15 leads (Eq. 13). A model that is
best at lead 1 is not necessarily best over a 15-day window, so selecting on
lead-1 skill and deploying a 15-day head is a selection/evaluation mismatch.

This module removes the mismatch. Every architecture is retrained under the
15-day objective of Eq. 14, and the winner is the model with the highest
R2 AVERAGED OVER ALL 15 LEADS, on the validation year.

The aggregation matches the loss weighting exactly (`MH.selection_score`): with
uniform omega_k the criterion is the plain mean over leads; with 1/k decay the
criterion decays identically. Changing WEIGHT_MODE changes both together, so
the selection can never drift away from what was optimised.

PROTOCOL (unchanged from Phase 1, and still leak-free)
------------------------------------------------------
  train      2019-2023
  validation 2024        early stopping AND model selection
  test       2025-2026   reported once, never used for any decision
Scalers are fitted on train only; validation/test windows seed from the last
WINDOW days of the preceding split; a window ending on day d contains only
days <= d at every lead.

COST
----
6 architectures x 18 substations = 108 trainings, roughly 6-8 hours on CPU.
The run is RESUMABLE: results are checkpointed after every model, and
re-running skips whatever is already done. A crash or a closed laptop costs
one model, not the whole sweep.

USAGE
-----
    python research_multihorizon.py                  # full sweep
    python Run_research.py                           # same, with banner

    # quick smoke test (~3 min) before committing to the full run:
    MH_EPOCHS=3 MH_SUBSTATIONS="Kalyanpur Dhanmondi" python research_multihorizon.py

    # sensitivity: does 1/k weighting change the winner?
    MH_WEIGHT_MODE=decaying MH_TAG=decaying python research_multihorizon.py

Environment overrides: MH_EPOCHS, MH_SUBSTATIONS, MH_MODELS, MH_WEIGHT_MODE,
MH_TAG, MH_FRESH (=1 to ignore existing checkpoints).

OUTPUTS (forecast_outputs/)
---------------------------
  global_model_selection.csv          THE selection artefact read by 05b.
                                      Same schema as before plus horizon
                                      columns, so Run_pipeline's pre-flight
                                      and 05b both keep working.
  global_model_selection_lead1.csv    the previous day-ahead file, preserved
  metrics_summary_multihorizon.csv    one row per (substation, model)
  horizon_skill_all.csv               one row per (substation, model, lead)
  best_model_per_substation_mh.csv    winner's per-substation test metrics
  horizon_skill_by_model.png          skill curves, one line per architecture
  multihorizon_model_selection.png    selection table
  multihorizon_r2_boxplot.png         distribution of the selection metric
  <sub>_<winner>_lead_forecast.png    actual vs predicted at leads 1 / 7 / 15
  mh_preds/<sub>__<model>.npz         stored test predictions, so the figures
                                      above can be redrawn without retraining
"""

from __future__ import annotations

import os
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import load_prediction as LP
import multihorizon as MH

OUTPUT_DIR = Path("forecast_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

WEIGHT_MODE = os.environ.get("MH_WEIGHT_MODE", "uniform")
TAG = os.environ.get("MH_TAG", "")
_SUF = f"_{TAG}" if TAG else ""
CKPT = OUTPUT_DIR / f"_multihorizon_partial{_SUF}.csv"
SKILL_CKPT = OUTPUT_DIR / f"_horizon_skill_partial{_SUF}.csv"

if "MH_EPOCHS" in os.environ:
    LP.EPOCHS = int(os.environ["MH_EPOCHS"])

MODELS = (os.environ["MH_MODELS"].split()
          if "MH_MODELS" in os.environ else list(LP.MODEL_NAMES))
SUBSTATIONS = (os.environ["MH_SUBSTATIONS"].split()
               if "MH_SUBSTATIONS" in os.environ
               else list(LP.USABLE_SUBSTATIONS))


PRED_DIR = OUTPUT_DIR / f"mh_preds{_SUF}"


def mape_by_horizon(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Per-lead MAPE (%), guarded against near-zero actuals.

    MH.horizon_metrics returns R2/RMSE/MAE only. MAPE is computed here rather
    than added there so multihorizon.py stays exactly as written.
    """
    out = np.empty(y_true.shape[1])
    for k in range(y_true.shape[1]):
        yt, yp = y_true[:, k], y_pred[:, k]
        m = np.abs(yt) > 1e-6
        out[k] = (np.mean(np.abs((yt[m] - yp[m]) / yt[m])) * 100.0
                  if m.any() else np.nan)
    return out


def test_first_target_dates(sub_df: pd.DataFrame) -> pd.DatetimeIndex:
    """Date of the lead-1 target of each TEST sequence.

    prepare_multi seeds the test block with the last WINDOW days of train+val,
    so test sequence i has its lead-1 target on the i-th test day; the lead-k
    target falls (k-1) days later because load_and_clean reindexes the record
    to a contiguous daily range.
    """
    year = sub_df["Date"].dt.year
    n_tr = int((year <= LP.TRAIN_END_YEAR).sum())
    n_va = int((year == LP.VAL_YEAR).sum())
    n_te = int((year >= LP.TEST_START_YEAR).sum())
    n_seq = n_te - MH.HORIZON + 1
    return pd.DatetimeIndex(
        sub_df["Date"].iloc[n_tr + n_va: n_tr + n_va + n_seq].values)


# ── one (substation, model) cell ────────────────────────────────────────────
def run_cell(df_clean: pd.DataFrame, substation: str, model_name: str):
    """Train one architecture on one substation under the 15-day objective."""
    import tensorflow as tf
    tf.keras.backend.clear_session()          # 108 models in one process

    load_col = f"{substation}_Load_MW"
    sub_df = LP.build_substation_frame(df_clean, substation)
    (X_tr, y_tr, X_va, y_va, X_te, y_te,
     scaler, feature_cols, sub_df) = MH.prepare_multi(sub_df, load_col,
                                                      substation)
    n_feat = len(feature_cols)

    t0 = time.time()
    model, epochs_run = MH.train_multi(model_name, X_tr, y_tr, X_va, y_va,
                                       scaler, n_feat,
                                       weight_mode=WEIGHT_MODE)
    train_seconds = time.time() - t0

    def ev(X, y):
        true = MH.inverse_multi(scaler, y, n_feat)
        pred = MH.inverse_multi(scaler, model.predict(X, verbose=0), n_feat)
        return true, pred

    tr_true, tr_pred = ev(X_tr, y_tr)
    va_true, va_pred = ev(X_va, y_va)
    te_true, te_pred = ev(X_te, y_te)

    tr_m = MH.horizon_metrics(tr_true, tr_pred)
    va_m = MH.horizon_metrics(va_true, va_pred)
    te_m = MH.horizon_metrics(te_true, te_pred)
    sigma_k = MH.sigma_by_horizon(va_true, va_pred)          # Eq. 15

    # per-lead detail (the §2.5 skill curve)
    skill = pd.DataFrame({
        "substation": substation, "model": model_name,
        "horizon_days": [m["horizon_days"] for m in te_m],
        "train_r2": [m["r2"] for m in tr_m],
        "val_r2":   [m["r2"] for m in va_m],
        "val_rmse": [m["rmse"] for m in va_m],
        "test_r2":  [m["r2"] for m in te_m],
        "test_rmse": [m["rmse"] for m in te_m],
        "test_mae": [m["mae"] for m in te_m],
        "val_mape": mape_by_horizon(va_true, va_pred),
        "test_mape": mape_by_horizon(te_true, te_pred),
        "sigma_k_mw": np.round(sigma_k, 4),
        "z_beta_sigma_mw": np.round(MH.Z_BETA * sigma_k, 4),
    })

    # Keep the test predictions. Without this the lead-1/7/15 figures below
    # cannot be produced without retraining all 108 models.
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        PRED_DIR / f"{substation.lower().replace(' ', '_')}__"
                   f"{model_name.lower().replace('+', '_')}.npz",
        first_target_date=test_first_target_dates(sub_df).values
                          .astype("datetime64[D]"),
        y_true=te_true.astype("float32"), y_pred=te_pred.astype("float32"))

    sc = MH.selection_score
    _w = MH.horizon_weights(MH.HORIZON, WEIGHT_MODE)

    def agg(per_lead_values):
        """Weighted mean over leads, using the SAME omega_k as the loss.

        Under WEIGHT_MODE='uniform' every weight is 1, so this is exactly the
        plain mean it replaces and the default run's numbers are unchanged.
        Under 'decaying' it stops val_rmse/val_mae from being aggregated on a
        different weighting than val_r2 and the training objective.
        """
        v = np.asarray(per_lead_values, dtype=float)
        return float(np.sum(v * _w) / np.sum(_w))

    row = {
        "substation": substation, "model": model_name,
        # aggregated over ALL leads — these are the selection quantities
        "train_r2": sc([m["r2"] for m in tr_m], WEIGHT_MODE),
        "val_r2":   sc([m["r2"] for m in va_m], WEIGHT_MODE),
        "val_rmse": agg([m["rmse"] for m in va_m]),
        "val_mae":  agg([m["mae"] for m in va_m]),
        "test_r2":  sc([m["r2"] for m in te_m], WEIGHT_MODE),
        "test_rmse": agg([m["rmse"] for m in te_m]),
        "test_mae":  agg([m["mae"] for m in te_m]),
        "val_mape":  agg(mape_by_horizon(va_true, va_pred)),
        "test_mape": agg(mape_by_horizon(te_true, te_pred)),
        # per-lead reference points, reported but never selected on
        "val_r2_lead1":   va_m[0]["r2"],
        "val_r2_lead7":   va_m[min(6, len(va_m) - 1)]["r2"],
        "val_r2_lead15":  va_m[-1]["r2"],
        "test_r2_lead1":  te_m[0]["r2"],
        "test_r2_lead7":  te_m[min(6, len(te_m) - 1)]["r2"],
        "test_r2_lead15": te_m[-1]["r2"],
        "sigma_1_mw":  float(sigma_k[0]),
        "sigma_15_mw": float(sigma_k[-1]),
        "horizon": MH.HORIZON,
        "weight_mode": WEIGHT_MODE,
        "epochs_run": epochs_run,
        "train_seconds": round(train_seconds, 1),
    }
    return row, skill


# ── selection ───────────────────────────────────────────────────────────────
def select_global_best_model(summary: pd.DataFrame):
    """Winner = highest mean-over-leads VALIDATION R2, averaged over
    substations. Tie-break: lowest mean validation rank. The test years take
    no part in the decision and are reported only for transparency.

    Schema is deliberately identical to the day-ahead selection file so that
    Run_pipeline's pre-flight check and 05b's reader keep working unchanged;
    the `horizon` and `weight_mode` columns are what tell 05b the selection was
    made under the multi-horizon objective.
    """
    summary = summary.copy()
    summary["rank"] = summary.groupby("substation")["val_r2"] \
                             .rank(ascending=False, method="min")

    sel = summary.groupby("model").agg(
        mean_val_r2=("val_r2", "mean"),
        min_val_r2=("val_r2", "min"),
        mean_rank=("rank", "mean"),
        wins=("rank", lambda r: int((r == 1).sum())),
        mean_val_rmse=("val_rmse", "mean"),
        mean_test_r2=("test_r2", "mean"),
        min_test_r2=("test_r2", "min"),
        mean_test_rmse=("test_rmse", "mean"),
        mean_test_mae=("test_mae", "mean"),
        mean_train_r2=("train_r2", "mean"),
        mean_val_r2_lead1=("val_r2_lead1", "mean"),
        mean_val_r2_lead15=("val_r2_lead15", "mean"),
        mean_test_r2_lead1=("test_r2_lead1", "mean"),
        mean_test_r2_lead15=("test_r2_lead15", "mean"),
        mean_sigma_1_mw=("sigma_1_mw", "mean"),
        mean_sigma_15_mw=("sigma_15_mw", "mean"),
        mean_epochs=("epochs_run", "mean"),
    ).reset_index()

    sel["horizon"] = MH.HORIZON
    sel["weight_mode"] = WEIGHT_MODE
    sel = sel.sort_values(["mean_val_r2", "mean_rank"],
                          ascending=[False, True]).reset_index(drop=True)
    return sel.iloc[0]["model"], sel


# ── figures ─────────────────────────────────────────────────────────────────
def plot_skill_curves(skill: pd.DataFrame, best: str, out: Path):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for name, g in skill.groupby("model"):
        m = g.groupby("horizon_days")["test_r2"].mean()
        is_best = name == best
        ax.plot(m.index, m.values, marker="o", ms=4,
                lw=2.6 if is_best else 1.3,
                alpha=1.0 if is_best else 0.55,
                label=f"{name} (selected)" if is_best else name,
                zorder=3 if is_best else 2)
    ax.set_xlabel("Forecast lead time $k$ (days)")
    ax.set_ylabel("Test $R^2$ (mean over substations)")
    ax.set_title(f"Horizon skill curve, $K$={MH.HORIZON}, "
                 f"$\\omega_k$={WEIGHT_MODE}\n"
                 f"selection on mean validation $R^2$ over all leads")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_xticks(range(1, MH.HORIZON + 1))
    fig.tight_layout()
    fig.savefig(out / f"horizon_skill_by_model{_SUF}.png", dpi=150)
    plt.close(fig)


def plot_selection_table(sel: pd.DataFrame, best: str, out: Path):
    cols = ["model", "mean_val_r2", "min_val_r2", "mean_rank", "wins",
            "mean_test_r2", "mean_test_r2_lead1", "mean_test_r2_lead15",
            "mean_sigma_15_mw"]
    labels = ["Model", "Mean Val R²\n(all leads)", "Worst Val R²", "Mean rank",
              "Wins", "Mean Test R²\n(all leads)", "Test R²\nlead 1",
              "Test R²\nlead 15", "σ₁₅ (MW)"]
    d = sel[cols].copy()
    for c in cols[1:]:
        d[c] = d[c].map(lambda v: f"{v:.4f}" if abs(v) < 100 else f"{v:.1f}")

    fig, ax = plt.subplots(figsize=(13, 1.0 + 0.45 * len(d)))
    ax.axis("off")
    t = ax.table(cellText=d.values, colLabels=labels, loc="center",
                 cellLoc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(8.5)
    t.scale(1, 1.7)
    for j in range(len(labels)):
        t[0, j].set_facecolor("#243b53")
        t[0, j].set_text_props(color="w", weight="bold")
    for i, name in enumerate(d["model"], start=1):
        if name == best:
            for j in range(len(labels)):
                t[i, j].set_facecolor("#d8f3dc")
                t[i, j].set_text_props(weight="bold")
    ax.set_title(f"Phase 1 multi-horizon model selection — winner: {best}\n"
                 f"criterion: mean validation R² across all {MH.HORIZON} leads "
                 f"(ω_k = {WEIGHT_MODE}); test years never used for selection",
                 pad=16, fontsize=11)
    fig.tight_layout()
    fig.savefig(out / f"multihorizon_model_selection{_SUF}.png", dpi=150)
    plt.close(fig)


def plot_boxplot(summary: pd.DataFrame, best: str, out: Path):
    order = (summary.groupby("model")["val_r2"].mean()
             .sort_values(ascending=False).index.tolist())
    data = [summary.loc[summary.model == m, "val_r2"].values for m in order]
    fig, ax = plt.subplots(figsize=(9, 5))
    # matplotlib renamed boxplot's labels= to tick_labels= in 3.9 and
    # removes the old name in 3.11 — same shim load_prediction.py uses
    import matplotlib as _mpl
    _mm = tuple(int(v) for v in _mpl.__version__.split(".")[:2])
    _label_kw = "tick_labels" if _mm >= (3, 9) else "labels"
    bp = ax.boxplot(data, **{_label_kw: order}, patch_artist=True)
    for patch, name in zip(bp["boxes"], order):
        patch.set_facecolor("#95d5b2" if name == best else "#e9ecef")
    ax.set_ylabel("Validation $R^2$, mean over all leads")
    ax.set_title(f"Selection metric across {summary.substation.nunique()} "
                 f"substations — winner: {best}")
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(out / f"multihorizon_r2_boxplot{_SUF}.png", dpi=150)
    plt.close(fig)


def plot_lead_forecasts(summary: pd.DataFrame, best: str, out: Path,
                        leads=(1, 7, MH.HORIZON)):
    """Actual vs predicted at lead 1 / 7 / 15 for the selected architecture.

    Priority 7 of the review: the skill curve shows how accuracy DECAYS with
    lead, but not what a 15-day-ahead forecast actually looks like against the
    truth. Drawn from the stored .npz, so no retraining.
    """
    leads = sorted({k for k in leads if 1 <= k <= MH.HORIZON})
    for sub in sorted(summary.loc[summary.model == best, "substation"].unique()):
        f = PRED_DIR / (f"{sub.lower().replace(' ', '_')}__"
                        f"{best.lower().replace('+', '_')}.npz")
        if not f.exists():
            continue
        with np.load(f) as z:
            d0 = pd.to_datetime(z["first_target_date"])
            y_true, y_pred = z["y_true"], z["y_pred"]

        fig, axes = plt.subplots(len(leads), 1, figsize=(13, 3.1 * len(leads)),
                                 sharex=True)
        axes = np.atleast_1d(axes)
        for ax, k in zip(axes, leads):
            yt, yp = y_true[:, k - 1], y_pred[:, k - 1]
            res = yt - yp
            r2 = 1 - np.sum(res ** 2) / np.sum((yt - yt.mean()) ** 2)
            ax.plot(d0 + pd.Timedelta(days=k - 1), yt, color="#1f3b57",
                    lw=1.1, label="Actual")
            ax.plot(d0 + pd.Timedelta(days=k - 1), yp, color="#d1495b",
                    lw=1.1, alpha=0.85, label=f"Predicted (lead {k})")
            ax.set_title(f"Lead {k} day{'s' if k > 1 else ''} ahead — "
                         f"$R^2$ = {r2:.3f}, "
                         f"RMSE = {np.sqrt(np.mean(res ** 2)):.2f} MW",
                         fontsize=10, fontweight="bold")
            ax.set_ylabel("Peak load (MW)")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8, loc="upper left")
        axes[-1].set_xlabel(f"Date (test years {LP.TEST_START_YEAR}+)")
        fig.suptitle(f"{sub} — {best}, direct {MH.HORIZON}-day forecast",
                     fontsize=12, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(out / f"{sub.lower()}_{best.lower().replace('+', '_')}"
                          f"_lead_forecast{_SUF}.png", dpi=150)
        plt.close(fig)
    print(f"lead-{'/'.join(str(k) for k in leads)} forecast figures written")


# ── main ────────────────────────────────────────────────────────────────────
def main():
    n_cells = len(SUBSTATIONS) * len(MODELS)
    print("=" * 74)
    print("PHASE 1 — MULTI-HORIZON MODEL SELECTION")
    print("=" * 74)
    print(f"  {len(MODELS)} architectures x {len(SUBSTATIONS)} substations "
          f"= {n_cells} trainings")
    print(f"  horizon K = {MH.HORIZON}, loss weights omega_k = {WEIGHT_MODE}")
    print(f"  split: train <= {LP.TRAIN_END_YEAR} | val {LP.VAL_YEAR} | "
          f"test >= {LP.TEST_START_YEAR}")
    print(f"  window {LP.WINDOW}, seed {LP.SEED}, max epochs {LP.EPOCHS}, "
          f"patience {LP.PATIENCE}")
    print("  SELECTION: highest mean validation R2 across ALL leads")
    print("             (matches the training objective; test never used)")
    print("=" * 74, flush=True)

    done_rows, done_skill = [], []
    if CKPT.exists() and os.environ.get("MH_FRESH") != "1":
        done_rows = pd.read_csv(CKPT).to_dict("records")
        if SKILL_CKPT.exists():
            done_skill = [pd.read_csv(SKILL_CKPT)]
        print(f"  resuming: {len(done_rows)} of {n_cells} cells already done "
              f"(delete {CKPT.name} or set MH_FRESH=1 to start over)\n")

    completed = {(r["substation"], r["model"]) for r in done_rows}
    import config as C
    df_clean = LP.load_and_clean(str(C.BPDB_CSV))

    t_start = time.time()
    times = []
    for si, sub in enumerate(SUBSTATIONS, start=1):
        for mi, mdl in enumerate(MODELS, start=1):
            if (sub, mdl) in completed:
                continue
            idx = len(completed) + 1
            print(f"[{idx}/{n_cells}] {sub:<12} {mdl:<18}", end="", flush=True)
            t0 = time.time()
            try:
                row, skill = run_cell(df_clean, sub, mdl)
            except Exception as exc:
                import traceback
                print(f"  ERROR: {exc}")
                traceback.print_exc()
                continue
            times.append(time.time() - t0)
            done_rows.append(row)
            done_skill.append(skill)
            completed.add((sub, mdl))

            pd.DataFrame(done_rows).to_csv(CKPT, index=False)
            pd.concat(done_skill, ignore_index=True).to_csv(SKILL_CKPT,
                                                            index=False)

            eta = (n_cells - len(completed)) * (sum(times) / len(times))
            print(f"  valR2(all leads)={row['val_r2']:.4f}  "
                  f"lead1={row['val_r2_lead1']:.3f} "
                  f"lead15={row['val_r2_lead15']:.3f}  "
                  f"{row['epochs_run']}ep {times[-1]:.0f}s  "
                  f"| ETA {eta/60:.0f} min", flush=True)

    if not done_rows:
        print("No results.")
        return

    summary = pd.DataFrame(done_rows)
    skill = pd.concat(done_skill, ignore_index=True) if done_skill \
        else pd.read_csv(SKILL_CKPT)

    # Restrict to the substations configured for THIS run. Without this, a
    # checkpoint written before a substation was removed from
    # LP.USABLE_SUBSTATIONS keeps feeding that substation into
    # select_global_best_model, so the excluded data still decides the winner.
    keep = set(SUBSTATIONS)
    dropped = sorted(set(summary["substation"]) - keep)
    if dropped:
        print(f"\nNOTE: dropping {len(dropped)} substation(s) present in the "
              f"checkpoint but not in this run's configuration: "
              f"{', '.join(dropped)}")
        summary = summary[summary["substation"].isin(keep)].reset_index(drop=True)
        skill = skill[skill["substation"].isin(keep)].reset_index(drop=True)
        if summary.empty:
            print("Nothing left after filtering — delete the checkpoint and re-run.")
            return
    summary.to_csv(OUTPUT_DIR / f"metrics_summary_multihorizon{_SUF}.csv",
                   index=False)
    skill.to_csv(OUTPUT_DIR / f"horizon_skill_all{_SUF}.csv", index=False)

    best, sel = select_global_best_model(summary)

    print("\n" + "=" * 74)
    print("SELECTION — mean over substations, ranked by validation R2 "
          "(all leads)")
    print("=" * 74)
    show = sel[["model", "mean_val_r2", "mean_rank", "wins",
                "mean_test_r2", "mean_test_r2_lead1", "mean_test_r2_lead15"]]
    print(show.to_string(index=False,
                         float_format=lambda x: f"{x:.4f}"))
    print(f"\nWINNER: {best}")

    # preserve the day-ahead artefact before overwriting
    target = OUTPUT_DIR / "global_model_selection.csv"
    if not TAG:
        if target.exists():
            prev = pd.read_csv(target)
            if "horizon" not in prev.columns:
                keep = OUTPUT_DIR / "global_model_selection_lead1.csv"
                prev.to_csv(keep, index=False)
                print(f"previous day-ahead selection preserved: {keep.name}")
        sel.to_csv(target, index=False)
        print(f"selection written: {target}")
    else:
        alt = OUTPUT_DIR / f"global_model_selection{_SUF}.csv"
        sel.to_csv(alt, index=False)
        print(f"tagged run — written to {alt.name}; "
              f"global_model_selection.csv left untouched")

    bm = summary[summary.model == best].copy()
    bm.to_csv(OUTPUT_DIR / f"best_model_per_substation_mh{_SUF}.csv",
              index=False)

    plot_skill_curves(skill, best, OUTPUT_DIR)
    plot_selection_table(sel, best, OUTPUT_DIR)
    plot_boxplot(summary, best, OUTPUT_DIR)
    plot_lead_forecasts(summary, best, OUTPUT_DIR)

    curve = (skill[skill.model == best].groupby("horizon_days")
             [["test_r2", "sigma_k_mw"]].mean())
    print(f"\nSkill curve of the winner ({best}), mean over substations:")
    print(f"  {'lead':>5} {'test R2':>9} {'sigma_k (MW)':>13}")
    for k, r in curve.iterrows():
        print(f"  {k:>5} {r.test_r2:>9.3f} {r.sigma_k_mw:>13.2f}")

    print(f"\nPhase 1 complete in {(time.time()-t_start)/60:.1f} min.")
    print("Next:  python Run_pipeline.py")


if __name__ == "__main__":
    main()