"""
report_per_substation.py — per-substation tables and figures for the
multi-horizon run, rebuilt from files Phase 1 already wrote.

    python report_per_substation.py
    python report_per_substation.py --tag decay
    python report_per_substation.py --substations Kalyanpur Dhanmondi

Trains nothing. Runs in seconds. Restores the per-substation reporting the
original load_prediction.py produced, adapted to a 15-lead forecaster.

READS
-----
  horizon_skill_all{tag}.csv        per (substation, model, lead) metrics
  global_model_selection{tag}.csv   the winner
  metrics_summary_multihorizon{tag}.csv   epochs_run
  mh_preds{tag}/<sub>__<model>.npz  stored TEST predictions, all 15 leads
  baselines_per_lead.csv            optional, drawn on the skill curves

WRITES  (into forecast_outputs/per_substation{tag}/)
  <sub>_summary_table.png     all 6 models x metrics, winner row highlighted
  <sub>_r2_comparison.png     lead-1 / mean / lead-15 test R2, 6 models
  <sub>_skill_curve.png       R2 vs lead, 6 models + naive baselines
  <sub>_<winner>_actual_vs_pred.png   test actual vs predicted, leads 1/7/15
  global_summary_test_r2.png  heatmap, substation x model
  best_model_r2_by_substation.png
  per_substation_metrics.csv  the table behind all of it

NOTE: only TEST predictions were stored during the sweep, so the original's
train-panel plot cannot be rebuilt here. Every metric-based figure can.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import load_prediction as LP
import multihorizon as MH

ROOT = Path(__file__).parent
OUT = ROOT / "forecast_outputs"
a = sys.argv[1:]
TAG = a[a.index("--tag") + 1] if "--tag" in a else ""
SUF = f"_{TAG}" if TAG else ""
SUBS = None
if "--substations" in a:
    i = a.index("--substations") + 1
    SUBS = [x for x in a[i:] if not x.startswith("--")]

DEST = OUT / f"per_substation{SUF}"
PRED = OUT / f"mh_preds{SUF}"
LEADS = [1, 7, MH.HORIZON]
BLUE, RED = "#1f3b57", "#d1495b"


def slug(s: str) -> str:
    return s.lower().replace("+", "_").replace(" ", "_")


def load_all():
    p = OUT / f"horizon_skill_all{SUF}.csv"
    if not p.exists():
        raise SystemExit(f"{p} not found — run Phase 1 first")
    sk = pd.read_csv(p).drop_duplicates(
        ["substation", "model", "horizon_days"], keep="last")
    selp = OUT / f"global_model_selection{SUF}.csv"
    winner = (pd.read_csv(selp)["model"].iloc[0] if selp.exists()
              else sk["model"].iloc[0])
    sump = OUT / f"metrics_summary_multihorizon{SUF}.csv"
    summ = pd.read_csv(sump) if sump.exists() else pd.DataFrame()
    basep = OUT / "baselines_per_lead.csv"
    base = pd.read_csv(basep) if basep.exists() else None
    return sk, winner, summ, base


def per_substation_table(sk: pd.DataFrame, summ: pd.DataFrame) -> pd.DataFrame:
    """Collapse the per-lead table to one row per (substation, model)."""
    g = sk.groupby(["substation", "model"])
    t = g.agg(mean_test_r2=("test_r2", "mean"),
              mean_test_rmse=("test_rmse", "mean"),
              mean_test_mae=("test_mae", "mean"),
              mean_val_r2=("val_r2", "mean")).reset_index()
    lead = lambda k, col: (sk[sk.horizon_days == k]
                           .set_index(["substation", "model"])[col])
    idx = pd.MultiIndex.from_frame(t[["substation", "model"]])
    for k, name in ((1, "lead1"), (MH.HORIZON, f"lead{MH.HORIZON}")):
        t[f"test_r2_{name}"] = lead(k, "test_r2").reindex(idx).values
        t[f"test_rmse_{name}"] = lead(k, "test_rmse").reindex(idx).values
    if "sigma_k_mw" in sk.columns:
        t["sigma_1_mw"] = lead(1, "sigma_k_mw").reindex(idx).values
        t[f"sigma_{MH.HORIZON}_mw"] = lead(MH.HORIZON, "sigma_k_mw").reindex(idx).values
    if len(summ) and "epochs_run" in summ.columns:
        t = t.merge(summ[["substation", "model", "epochs_run"]],
                    on=["substation", "model"], how="left")
    return t


def fig_summary_table(sub: str, t: pd.DataFrame, winner: str) -> None:
    d = t[t.substation == sub].copy()
    if d.empty:
        return
    d = d.sort_values("mean_val_r2", ascending=False)
    cols = [("model", "Model", "{}"),
            ("mean_val_r2", "Val R²\n(all leads)", "{:.4f}"),
            ("mean_test_r2", "Test R²\n(all leads)", "{:.4f}"),
            ("test_r2_lead1", "Test R²\nlead 1", "{:.4f}"),
            (f"test_r2_lead{MH.HORIZON}", f"Test R²\nlead {MH.HORIZON}", "{:.4f}"),
            ("mean_test_rmse", "Test RMSE\n(MW)", "{:.2f}"),
            ("mean_test_mae", "Test MAE\n(MW)", "{:.2f}")]
    cols = [c for c in cols if c[0] in d.columns]
    rows = [[f.format(r[k]) for k, _, f in cols] for _, r in d.iterrows()]
    colors = [["#c6efce"] * len(cols) if r["model"] == winner
              else ["#f7f7f7" if i % 2 else "white"] * len(cols)
              for i, (_, r) in enumerate(d.iterrows())]

    fig, ax = plt.subplots(figsize=(12, 0.55 * len(rows) + 1.9))
    ax.axis("off")
    tb = ax.table(cellText=rows, colLabels=[c[1] for c in cols],
                  cellColours=colors, loc="center", cellLoc="center")
    tb.auto_set_font_size(False); tb.set_fontsize(9.5); tb.scale(1, 1.7)
    for j in range(len(cols)):
        tb[(0, j)].set_facecolor("#4472C4")
        tb[(0, j)].get_text().set_color("white")
        tb[(0, j)].get_text().set_fontweight("bold")
    ax.set_title(f"{sub} — all 6 architectures, {MH.HORIZON}-day direct forecast\n"
                 f"[green = globally selected model: {winner}; "
                 f"ranked by validation R² over all leads]",
                 fontsize=11, fontweight="bold", pad=14)
    fig.tight_layout()
    fig.savefig(DEST / f"{slug(sub)}_summary_table.png", dpi=130,
                bbox_inches="tight")
    plt.close(fig)


def fig_r2_comparison(sub: str, t: pd.DataFrame, winner: str) -> None:
    d = t[t.substation == sub].copy()
    if d.empty:
        return
    d = d.sort_values("mean_val_r2", ascending=False)
    keys = [("test_r2_lead1", "Test R² lead 1"),
            ("mean_test_r2", f"Test R² mean (1-{MH.HORIZON})"),
            (f"test_r2_lead{MH.HORIZON}", f"Test R² lead {MH.HORIZON}")]
    keys = [k for k in keys if k[0] in d.columns]
    x = np.arange(len(d)); w = 0.8 / len(keys)
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, (col, lab) in enumerate(keys):
        bars = ax.bar(x + (i - (len(keys) - 1) / 2) * w, d[col], w, label=lab,
                      alpha=0.88)
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.006,
                    f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([m + ("\n(selected)" if m == winner else "")
                        for m in d.model], rotation=16, ha="right", fontsize=8.5)
    ax.axhline(0, color="black", lw=0.6, ls="--")
    ax.set_ylabel("Test R²"); ax.grid(axis="y", alpha=0.3)
    ax.set_title(f"{sub} — architecture comparison")
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    fig.savefig(DEST / f"{slug(sub)}_r2_comparison.png", dpi=130)
    plt.close(fig)


def fig_skill_curve(sub: str, sk: pd.DataFrame, winner: str, base) -> None:
    d = sk[sk.substation == sub]
    if d.empty:
        return
    fig, ax = plt.subplots(figsize=(11, 5.2))
    for m in sorted(d.model.unique()):
        s = d[d.model == m].sort_values("horizon_days")
        best = m == winner
        ax.plot(s.horizon_days, s.test_r2, marker="o", ms=3.5,
                lw=2.6 if best else 1.1, alpha=1 if best else 0.5,
                label=f"{m} (selected)" if best else m, zorder=3 if best else 2)
    if base is not None:
        b = base[base.substation == sub].sort_values("horizon_days")
        for col, style, lab in (("persistence_r2", ":", "persistence"),
                                ("ma28_r2", "-.", "28-day moving average")):
            if col in b.columns:
                ax.plot(b.horizon_days, b[col], style, color="black", lw=1.5,
                        alpha=0.75, label=f"{lab} (naive)")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xlabel("Forecast lead k (days ahead)"); ax.set_ylabel("Test R²")
    ax.set_xticks(range(1, MH.HORIZON + 1)); ax.grid(alpha=0.3)
    ax.set_title(f"{sub} — skill decay with horizon")
    ax.legend(fontsize=7.5)
    fig.tight_layout()
    fig.savefig(DEST / f"{slug(sub)}_skill_curve.png", dpi=130)
    plt.close(fig)


def fig_actual_vs_pred(sub: str, winner: str) -> None:
    f = PRED / f"{slug(sub)}__{slug(winner)}.npz"
    if not f.exists():
        return
    with np.load(f) as z:
        d0 = pd.to_datetime(z["first_target_date"])
        yt_all, yp_all = z["y_true"], z["y_pred"]
    leads = sorted({k for k in LEADS if 1 <= k <= yt_all.shape[1]})
    fig, axes = plt.subplots(len(leads), 1, figsize=(13, 3.1 * len(leads)),
                             sharex=True)
    axes = np.atleast_1d(axes)
    for ax, k in zip(axes, leads):
        yt, yp = yt_all[:, k - 1], yp_all[:, k - 1]
        res = yt - yp
        r2 = 1 - np.sum(res ** 2) / np.sum((yt - yt.mean()) ** 2)
        dates = d0 + pd.Timedelta(days=k - 1)
        ax.plot(dates, yt, color=BLUE, lw=1.1, label="Actual")
        ax.plot(dates, yp, color=RED, lw=1.1, alpha=0.85,
                label=f"Predicted (lead {k})")
        ax.set_title(f"Lead {k} day{'s' if k > 1 else ''} ahead — "
                     f"$R^2$={r2:.3f}, RMSE={np.sqrt((res**2).mean()):.2f} MW, "
                     f"MAE={np.abs(res).mean():.2f} MW",
                     fontsize=10, fontweight="bold")
        ax.set_ylabel("Peak load (MW)"); ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper left")
    axes[-1].set_xlabel(f"Date (test years {LP.TEST_START_YEAR}+)")
    fig.suptitle(f"{sub} — {winner}, direct {MH.HORIZON}-day forecast",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(DEST / f"{slug(sub)}_{slug(winner)}_actual_vs_pred.png", dpi=130)
    plt.close(fig)


def fig_global(t: pd.DataFrame, winner: str) -> None:
    piv = t.pivot(index="substation", columns="model",
                  values="mean_test_r2").sort_index()
    fig, ax = plt.subplots(figsize=(13, 0.45 * len(piv) + 2.2))
    ax.axis("off")
    vmin, vmax = np.nanmin(piv.values), np.nanmax(piv.values)
    cc = [[plt.cm.RdYlGn((v - vmin) / max(vmax - vmin, 1e-9))
           if pd.notna(v) else "#e0e0e0" for v in row] for row in piv.values]
    tb = ax.table(cellText=[[f"{v:.4f}" if pd.notna(v) else "—" for v in r]
                            for r in piv.values],
                  rowLabels=list(piv.index), colLabels=list(piv.columns),
                  cellColours=cc, loc="center", cellLoc="center")
    tb.auto_set_font_size(False); tb.set_fontsize(8.5); tb.scale(1, 1.5)
    for j in range(piv.shape[1]):
        tb[(0, j)].set_facecolor("#4472C4")
        tb[(0, j)].get_text().set_color("white")
        tb[(0, j)].get_text().set_fontweight("bold")
    ax.set_title(f"Mean Test R² (leads 1–{MH.HORIZON}) — substation × model\n"
                 "(green = higher)", fontsize=12, fontweight="bold", pad=16)
    fig.tight_layout()
    fig.savefig(DEST / "global_summary_test_r2.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    d = t[t.model == winner].sort_values("mean_test_r2", ascending=False)
    fig, ax = plt.subplots(figsize=(12, 5.5))
    bars = ax.bar(d.substation, d.mean_test_r2, color="steelblue", alpha=0.85)
    if "test_r2_lead1" in d.columns:
        ax.plot(d.substation, d.test_r2_lead1, "o--", color=RED, ms=5,
                lw=1.2, label="lead 1")
    m = d.mean_test_r2.mean()
    ax.axhline(m, color="tomato", ls="--", lw=1.2,
               label=f"mean = {m:.4f}")
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005,
                f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=7)
    ax.set_ylabel(f"Test R² (mean over leads 1–{MH.HORIZON})")
    ax.set_title(f"{winner} — performance by substation")
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.grid(axis="y", alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(DEST / "best_model_r2_by_substation.png", dpi=130)
    plt.close(fig)


def main() -> None:
    sk, winner, summ, base = load_all()
    DEST.mkdir(parents=True, exist_ok=True)
    t = per_substation_table(sk, summ)
    t.to_csv(DEST / "per_substation_metrics.csv", index=False)

    subs = SUBS if SUBS else sorted(t.substation.unique())
    print(f"run: {'tag ' + TAG if TAG else 'main'} | winner: {winner} | "
          f"{len(subs)} substation(s)")
    if base is None:
        print("note: baselines_per_lead.csv not found — run baselines.py to "
              "overlay naive methods on the skill curves")

    for sub in subs:
        fig_summary_table(sub, t, winner)
        fig_r2_comparison(sub, t, winner)
        fig_skill_curve(sub, sk, winner, base)
        fig_actual_vs_pred(sub, winner)
        d = t[(t.substation == sub) & (t.model == winner)]
        if len(d):
            r = d.iloc[0]
            print(f"  {sub:<14} mean R²={r.mean_test_r2:.3f}  "
                  f"lead1={r.get('test_r2_lead1', float('nan')):.3f}  "
                  f"lead{MH.HORIZON}={r.get(f'test_r2_lead{MH.HORIZON}', float('nan')):.3f}  "
                  f"RMSE={r.mean_test_rmse:.2f} MW")

    fig_global(t, winner)
    n = len(list(DEST.glob("*.png")))
    print(f"\n{n} figures + per_substation_metrics.csv -> "
          f"{DEST.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()