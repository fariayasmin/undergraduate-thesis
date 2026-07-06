"""
05_validate_against_bpdb.py — verify that the bottom-up synthetic aggregate
matches the REAL BPDB daily peak series of each substation (2024-01-01 → 2026-06-28).

For every day: synthetic daily peak (max of the 30-min aggregate) vs the
recorded BPDB daily peak. Reports MAPE, correlation and monthly means, and
plots the overlay + scatter. No post-hoc scaling is applied — residuals are
reported as-is.

Outputs (data/):
  05_daily_peak_validation.csv
  05_validation_<sub>.png
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config as C


def validate(substation: str) -> pd.DataFrame:
    agg = pd.read_csv(C.DATA_DIR / f"04_{substation.lower()}_aggregate_30min.csv",
                      parse_dates=["timestamp"])
    syn = (agg.set_index("timestamp")["total_mw"]
              .resample("D").max().rename("synthetic_peak_mw"))

    bp = pd.read_csv(C.BPDB_CSV, usecols=["Date", f"{substation}_Load_MW"])
    bp["Date"] = pd.to_datetime(bp["Date"])
    bp = bp[(bp["Date"] >= C.START_DATE) & (bp["Date"] <= C.END_DATE)]
    bp = bp.set_index("Date")
    real = bp[f"{substation}_Load_MW"].where(bp[f"{substation}_Load_MW"] <= 450)
    real = real.rename("bpdb_peak_mw")

    df = pd.concat([syn, real], axis=1).dropna()
    df["error_mw"]  = df["synthetic_peak_mw"] - df["bpdb_peak_mw"]
    df["abs_pct_error"] = 100 * (df["error_mw"].abs() / df["bpdb_peak_mw"])
    df.insert(0, "substation", substation)

    mape = df["abs_pct_error"].mean()
    corr = df["synthetic_peak_mw"].corr(df["bpdb_peak_mw"])
    print(f"\n{substation}: n={len(df)} days  MAPE={mape:.2f}%  "
          f"corr={corr:.3f}  mean syn={df['synthetic_peak_mw'].mean():.1f} MW  "
          f"mean BPDB={df['bpdb_peak_mw'].mean():.1f} MW")

    # ── plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.6),
                             gridspec_kw={"width_ratios": [2.4, 1]})
    ax = axes[0]
    ax.plot(df.index, df["bpdb_peak_mw"], label="BPDB (real)",
            color="steelblue", lw=1.1)
    ax.plot(df.index, df["synthetic_peak_mw"], label="Synthetic bottom-up",
            color="tomato", lw=1.1, alpha=0.85)
    ax.set_title(f"{substation} — daily peak, 2024–2026  "
                 f"(MAPE={mape:.1f}%, r={corr:.3f})")
    ax.set_ylabel("Daily peak (MW)"); ax.legend(fontsize=9)

    ax = axes[1]
    ax.scatter(df["bpdb_peak_mw"], df["synthetic_peak_mw"], s=10, alpha=0.5)
    num = df[["bpdb_peak_mw", "synthetic_peak_mw"]]
    lims = [num.min().min() * 0.9, num.max().max() * 1.05]
    ax.plot(lims, lims, "k--", lw=0.8)
    ax.set_xlabel("BPDB peak (MW)"); ax.set_ylabel("Synthetic peak (MW)")
    ax.set_title("Scatter")
    fig.tight_layout()
    fig.savefig(C.DATA_DIR / f"05_validation_{substation.lower()}.png", dpi=130)
    plt.close(fig)

    yearly = df.groupby(df.index.year).apply(
        lambda g: pd.Series({
            "n_days": len(g),
            "syn_mean": g["synthetic_peak_mw"].mean(),
            "bpdb_mean": g["bpdb_peak_mw"].mean(),
            "mape_pct": g["abs_pct_error"].mean()})).round(2)
    print(yearly.to_string())
    return df


def main():
    frames = [validate(sub) for sub in C.SUBSTATIONS]
    out = C.DATA_DIR / "05_daily_peak_validation.csv"
    pd.concat(frames).to_csv(out, index_label="date")
    print(f"\nValidation table written: {out}")


if __name__ == "__main__":
    main()