"""
baselines.py — naive forecasting baselines per horizon, and the skill table
your paper needs.

    python baselines.py                    # baselines only
    python baselines.py --compare          # merge with the trained model
    python baselines.py --compare --tag decay

Reads only. Trains nothing. Runs in seconds.

WHY THIS IS NOT OPTIONAL
------------------------
An R^2 of 0.604 means nothing on its own. It is only interpretable against
what a method with no model achieves on the same data. On this dataset
persistence scores 0.674 at lead 1 and 0.053 averaged over 15 leads, so the
same R^2 can be a failure at one horizon and a strong result at another.
Reviewers of load-forecasting papers expect the naive comparison.

BASELINES
---------
persistence      y(t+k) = y(t)                    "same as today"
ma7 / ma14 / ma28  y(t+k) = mean of the last 7/14/28 observed days
seasonal_naive   y(t+k) = y(t+k-7m), m = ceil(k/7) "same weekday, last week"
                 -- the right naive for daily load, which has strong weekly
                 seasonality. Never uses a day after t.
drift            persistence plus the average daily trend of the last 28 days

All are computed from observed values up to the issue day t only, on exactly
the substations in LP.USABLE_SUBSTATIONS and exactly the test years, so they
are directly comparable with the model's horizon_skill_all.csv.

SKILL SCORE
-----------
    skill_k = 1 - MSE_model(k) / MSE_baseline(k)
Positive means the model beats the baseline; 0 means it matches it; negative
means it is worse. This is the number to put in the paper, because it is
horizon-fair in a way raw R^2 is not.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

import load_prediction as LP
import multihorizon as MH

CSV = "BPDB_Dhaka_City_Substations_Page3.csv"
OUT = LP.OUTPUT_DIR
args = sys.argv[1:]
COMPARE = "--compare" in args
TAG = args[args.index("--tag") + 1] if "--tag" in args else ""
SUF = f"_{TAG}" if TAG else ""
MAXLAG = 28


def baselines_for(v: np.ndarray, yr: np.ndarray, k: int) -> dict | None:
    """All naive predictions of day t+k, issued on day t, for the test years."""
    i = np.arange(MAXLAG, len(v) - k)
    i = i[yr[i + k] >= LP.TEST_START_YEAR]
    if len(i) < 30:
        return None
    truth = v[i + k]
    m = int(np.ceil(k / 7))
    preds = {
        "persistence": v[i],
        "ma7": np.array([v[j - 6:j + 1].mean() for j in i]),
        "ma14": np.array([v[j - 13:j + 1].mean() for j in i]),
        "ma28": np.array([v[j - 27:j + 1].mean() for j in i]),
        "seasonal_naive": v[i + k - 7 * m],
        "drift": v[i] + k * np.array([(v[j] - v[j - 27]) / 27 for j in i]),
    }
    out = {}
    for name, p in preds.items():
        out[f"{name}_r2"] = r2_score(truth, p)
        out[f"{name}_mse"] = mean_squared_error(truth, p)
    return out


def main() -> None:
    df = LP.load_and_clean(CSV)
    rows = []
    for sub in LP.USABLE_SUBSTATIONS:
        col = f"{sub}_Load_MW"
        if col not in df.columns:
            continue
        s = LP.build_substation_frame(df, sub).reset_index(drop=True)
        v = s[col].to_numpy()
        yr = s["Date"].dt.year.to_numpy()
        for k in range(1, MH.HORIZON + 1):
            r = baselines_for(v, yr, k)
            if r:
                rows.append({"substation": sub, "horizon_days": k, **r})

    b = pd.DataFrame(rows)
    b.to_csv(OUT / "baselines_per_lead.csv", index=False)
    names = ["persistence", "ma7", "ma14", "ma28", "seasonal_naive", "drift"]
    piv = b.groupby("horizon_days")[[f"{n}_r2" for n in names]].mean()
    piv.columns = names

    print("=" * 88)
    print(f"NAIVE BASELINES — mean test R2 over "
          f"{b['substation'].nunique()} substations, test years "
          f"{LP.TEST_START_YEAR}+")
    print("=" * 88)
    print(piv.round(3).to_string())
    print(f"\nmean over all {MH.HORIZON} leads:")
    print(piv.mean().round(3).to_string())
    best = piv.mean().idxmax()
    print(f"\nstrongest naive overall: {best} ({piv.mean()[best]:.3f})")
    print(f"strongest naive at lead 1: {piv.loc[1].idxmax()} "
          f"({piv.loc[1].max():.3f})")

    if not COMPARE:
        print(f"\nWrote {OUT / 'baselines_per_lead.csv'}. "
              f"Re-run with --compare to build the skill table.")
        return

    skill_path = OUT / f"horizon_skill_all{SUF}.csv"
    sel_path = OUT / f"global_model_selection{SUF}.csv"
    if not skill_path.exists():
        raise SystemExit(f"{skill_path} not found — run Phase 1 first")
    sk = pd.read_csv(skill_path).drop_duplicates(
        ["substation", "model", "horizon_days"], keep="last")
    winner = (pd.read_csv(sel_path)["model"].iloc[0] if sel_path.exists()
              else sk["model"].iloc[0])
    sk = sk[sk["model"] == winner]

    # MSE per (substation, lead) from the model's RMSE, then pooled
    sk = sk.assign(model_mse=sk["test_rmse"] ** 2)
    merged = sk.merge(b, on=["substation", "horizon_days"], how="inner")
    agg = merged.groupby("horizon_days").agg(
        model_r2=("test_r2", "mean"), model_mse=("model_mse", "mean"),
        **{f"{n}_mse": (f"{n}_mse", "mean") for n in names},
        **{f"{n}_r2": (f"{n}_r2", "mean") for n in names})

    tbl = pd.DataFrame({"model_R2": agg.model_r2})
    for n in names:
        tbl[f"skill_vs_{n}"] = 1 - agg.model_mse / agg[f"{n}_mse"]
    tbl["best_naive_R2"] = agg[[f"{n}_r2" for n in names]].max(axis=1)
    tbl["R2_margin"] = tbl.model_R2 - tbl.best_naive_R2

    print("\n" + "=" * 88)
    print(f"SKILL OF '{winner}' AGAINST EACH BASELINE   "
          f"(1 - MSE_model / MSE_baseline)")
    print("=" * 88)
    print(tbl.round(3).to_string())
    tbl.to_csv(OUT / f"skill_vs_baselines{SUF}.csv")

    print("\n" + "-" * 88)
    neg = tbl[tbl.R2_margin < 0]
    if len(neg):
        print(f"LEADS WHERE A NAIVE METHOD WINS: {list(neg.index)}")
        for k, r in neg.iterrows():
            print(f"  lead {k}: model {r.model_R2:.3f} vs best naive "
                  f"{r.best_naive_R2:.3f}  ({r.R2_margin:+.3f})")
        print("\n  Report this honestly and explain it. A uniform 15-lead loss")
        print("  spends 14/15 of its gradient away from lead 1, and early")
        print("  stopping fires on a val_loss the deep leads dominate. Test it:")
        print("  MH_TAG=decay MH_WEIGHT_MODE=decaying python research_multihorizon.py")
    else:
        print("The model beats every naive baseline at every lead.")
    print(f"\nWrote {OUT / f'skill_vs_baselines{SUF}.csv'}")


if __name__ == "__main__":
    main()