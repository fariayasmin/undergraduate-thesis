#!/usr/bin/env python3
"""
Stage C/D runner - forecast to kappa.

    python 05c_forecast_to_kappa.py --build        # Eqs. 13-16, validate, cache
    python 05c_forecast_to_kappa.py --table        # per-horizon table
    python 05c_forecast_to_kappa.py --sweep        # z_beta propagation sweep
    python 05c_forecast_to_kappa.py --plot         # outputs/forecast_to_kappa.png
    python 05c_forecast_to_kappa.py --coverage PATH_TO_PER_RECORD_BACKTEST.csv

NOTATION, restated because it matters:
    kappa^scale_(i,d+k|d) = p~ / p_(i,d)   Eq. (16), the scaling factor
    kappa^cap_g                            Eq. (32), the curtailment cap
    z_beta = 1.645                         Eq. (14), the quantile multiplier
Eq. (14) is p~ = p_hat + z_beta*sigma. Writing kappa there would be circular.
"""

import argparse
import sys

import numpy as np
import pandas as pd

from gentwin import config as cfg
from gentwin import forecast as F


def _bpdb() -> pd.DataFrame:
    for p in (cfg.BPDB_CSV, cfg.DATA_DIR / "BPDB_Dhaka_City_Substations_Page3.csv"):
        if p.exists():
            return pd.read_csv(p, parse_dates=["Date"])
    raise FileNotFoundError(f"BPDB CSV not found; expected at {cfg.BPDB_CSV}")


def cmd_build(bpdb, subs) -> bool:
    ok = True
    for name in subs:
        fh = F.build_horizon(name, bpdb)
        path = F.write_cache(fh)
        d = fh.diagnostics
        r = fh.reference_day
        t = fh.p_str

        print(f"\n{name}")
        print(f"  reference day d      {r['date']}   p_(i,d) = {r['peak_kw']:.2f} kW"
              + (f", peak at {r['peak_clock']}" if r["peak_clock"] else ""))
        print(f"  sigma source         {fh.sigma_method}")
        print(f"                       {d['sigma_min_kw']:.2f} -> {d['sigma_max_kw']:.2f} kW "
              f"over k=1..{d['horizon_K']} ({d['sigma_growth_ratio']:.2f}x), "
              f"{d['sigma_raw_decreases']} raw decrease(s) repaired")
        print(f"  z_beta               {fh.z_beta} (beta = {fh.beta:.0%})")
        print(f"  P^str_i              {t['p_str_kw']:,.0f} kW  "
              f"(p{t['quantile']:.0%} of trailing {t['trailing_days']}d, "
              f"{t['window_start']} to {t['window_end']}, n={t['n_days_used']})")
        print(f"                       full-record value would be "
              f"{t['full_record_p_str_kw']:,.0f} kW vs trailing max "
              f"{t['recent_max_kw']:,.0f} kW")
        print(f"  P^max_i              {fh.p_max_kw:.1f} kW")
        print(f"  kappa^scale          {d['kappa_scale_min']:.4f} - "
              f"{d['kappa_scale_max']:.4f} (mean {d['kappa_scale_mean']:.4f})")
        print(f"  stress days          {d['n_stress_days']}/{d['horizon_K']}  "
              f"(cool-wet {d['n_stress_days_cool_wet']}, "
              f"hot-dry {d['n_stress_days_hot_dry']})")
        print(f"  deficit days         {d['n_deficit_days']}/{d['horizon_K']}")
        print(f"  horizon usage        k=1..{d['dispatch_days']} enter C1/C4 via "
              f"kappa^scale; k={d['dispatch_days'] + 1}..{d['horizon_K']} supply "
              f"s and Def for the regime only")
        print(f"  checks")
        for c in fh.checks:
            print(f"    [{'PASS' if c['pass'] else 'FAIL'}] {c['check']}")
            if c["detail"]:
                print(f"           {c['detail']}")
        print(f"  ALL PASS: {fh.all_pass}   cached -> {path}")
        ok &= fh.all_pass
    return ok


def cmd_table(bpdb, subs) -> None:
    for name in subs:
        rec = F.read_cache(name)
        if not rec:
            print(f"{name}: no cache; run --build first.")
            continue
        df = pd.DataFrame(rec["rows"])
        show = df[["horizon_k", "target_date", "in_dispatch_horizon", "p_hat_kw",
                   "sigma_kw", "uncertainty_margin_kw", "p_tilde_kw", "p_str_kw",
                   "s_stress", "deficit_kw", "kappa_scale",
                   "kappa_scale_cool_wet", "kappa_scale_hot_dry",
                   "weather_source"]].copy()
        show.columns = ["k", "date", "disp", "p_hat", "sigma", "z*sig", "p_tilde",
                        "P^str", "s", "Def", "kappa", "k_cool", "k_hot", "wx"]
        print(f"\n{name}   (p_ref = {rec['reference_day']['peak_kw']:.2f} kW "
              f"on {rec['reference_day']['date']})")
        print(show.to_string(index=False,
                             float_format=lambda v: f"{v:8.3f}"))
        df.to_csv(cfg.OUT_DIR / f"kappa_table_{name}.csv", index=False)
    print(f"\nCSVs written to {cfg.OUT_DIR}")


def cmd_sweep(bpdb, subs) -> None:
    print("How forecast uncertainty propagates into the MOLP.\n")
    print("Chain: sigma_(i,k) -> p~ (14) -> kappa^scale (16) -> C1 right-hand")
    print("side (27) -> shadow price pi_i(t) -> Proposition 2 threshold (35).")
    print("Raising z_beta inflates the demand the LP must serve at every slot.\n")
    for name in subs:
        sw = F.propagation_sweep(name, bpdb)
        print(f"{name}")
        print(sw.to_string(index=False, float_format=lambda v: f"{v:9.4f}"))
        sw.to_csv(cfg.OUT_DIR / f"kappa_zbeta_sweep_{name}.csv", index=False)
        base = sw[sw.z_beta == 1.645].iloc[0]
        print(f"  at the adopted z_beta=1.645: mean kappa^scale "
              f"{base['kappa_mean']:.4f}, i.e. C1 is asked to serve "
              f"{base['extra_demand_pct_vs_z0']:.1f}% more than the point "
              f"forecast implies, and {int(base['n_stress_days'])} of 14 days "
              f"are flagged stressed.\n")


def cmd_coverage(bpdb, subs, path) -> None:
    res = pd.read_csv(path)
    need = {"horizon", "actual_kw", "pred_kw"}
    if not need <= set(res.columns):
        print(f"Per-record backtest must have columns {sorted(need)}; "
              f"got {list(res.columns)}")
        return
    for name in subs:
        sub = res[res["substation"] == name] if "substation" in res.columns else res
        if sub.empty:
            continue
        fh = F.build_horizon(name, bpdb)
        cov = F.coverage_test(fh, sub)
        print(f"\n{name}: empirical coverage of p~ "
              f"(target {cov['target']:.0%}) -> {cov['overall_coverage']:.1%} "
              f"[{cov['verdict']}]")
        print(pd.DataFrame(cov["by_horizon"]).to_string(index=False))
        # Recompute sigma properly now that residuals exist.
        sg = F.sigma_from_residuals(sub)
        print(f"  bias by horizon: "
              f"{[round(b, 2) for b in sg['bias']]}")
        print(f"  RMSE overstates sigma by "
              f"{100 * (sg['rmse'].mean() / sg['sigma_raw'].mean() - 1):.1f}% on "
              f"average (the conservatism of using RMSE as sigma)")


def cmd_plot(bpdb, subs) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, len(subs), figsize=(7 * len(subs), 9),
                             squeeze=False)
    for j, name in enumerate(subs):
        rec = F.read_cache(name)
        if not rec:
            continue
        df = pd.DataFrame(rec["rows"])
        k = df["horizon_k"]

        ax = axes[0, j]
        ax.plot(k, df["p_hat_kw"], "o-", color="steelblue", label=r"$\hat{p}$ point forecast")
        ax.plot(k, df["p_tilde_kw"], "s-", color="tomato",
                label=r"$\tilde{p} = \hat{p} + z_\beta\sigma$  (14)")
        ax.fill_between(k, df["p_hat_kw"], df["p_tilde_kw"], color="tomato",
                        alpha=0.15, label=r"robustness margin $z_\beta\sigma$")
        ax.axhline(rec["p_str"]["p_str_kw"], color="darkorange", linestyle="--",
                   linewidth=1.2, label=f"$P^{{str}}_i$ = {rec['p_str']['p_str_kw']:,.0f} kW  (15)")
        ax.axhline(rec["p_max_kw"], color="black", linestyle=":", linewidth=1.2,
                   label=f"$P^{{max}}_i$ = {rec['p_max_kw']:.0f} kW")
        ax.axhline(rec["reference_day"]["peak_kw"], color="seagreen",
                   linestyle="-.", linewidth=1.2,
                   label=f"$p_{{i,d}}$ = {rec['reference_day']['peak_kw']:.1f} kW (ref day)")
        nd = rec["diagnostics"]["dispatch_days"]
        ax.axvspan(0.5, nd + 0.5, color="steelblue", alpha=0.08)
        ax.text(nd / 2 + 0.5, ax.get_ylim()[0], " dispatch", fontsize=7,
                va="bottom", ha="center", color="steelblue")
        ax.set_xlabel("horizon $k$ (days ahead)")
        ax.set_ylabel("kW")
        ax.set_title(f"{name} - robust peak", fontsize=11)
        ax.legend(fontsize=7.5, loc="best")

        ax = axes[1, j]
        ax.plot(k, df["kappa_scale"], "o-", color="purple",
                label=r"$\kappa^{scale}_{i,d+k|d} = \tilde{p}/p_{i,d}$  (16)")
        ax.fill_between(k, df["kappa_scale_cool_wet"], df["kappa_scale_hot_dry"],
                        color="purple", alpha=0.15,
                        label="climatology $\\pm1\\sigma$ weather scenarios")
        ax.axhline(1.0, color="grey", linestyle="--", linewidth=1,
                   label=r"$\kappa^{scale}=1$ (reference-day load)")
        for _, r in df[df["s_stress"] == 1].iterrows():
            ax.axvline(r["horizon_k"], color="darkorange", alpha=0.20, linewidth=6)
        ax.set_xlabel("horizon $k$ (days ahead)")
        ax.set_ylabel(r"$\kappa^{scale}$")
        ax.set_title(f"{name} - scaling factor into C1/C4 "
                     f"(shaded $k$: $s_{{i,d}}=1$)", fontsize=11)
        ax.legend(fontsize=7.5, loc="best")

    fig.suptitle("Stage C/D: forecast uncertainty into the optimisation "
                 r"($\tilde{p} = \hat{p} + z_\beta\sigma$, "
                 r"$\kappa^{scale} = \tilde{p}/p_{i,d}$)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = cfg.OUT_DIR / "forecast_to_kappa.png"
    fig.savefig(p, dpi=130, bbox_inches="tight")
    print(f"wrote {p}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    for f in ("build", "table", "sweep", "plot"):
        p.add_argument(f"--{f}", action="store_true")
    p.add_argument("--coverage", metavar="CSV", default=None)
    p.add_argument("--all", action="store_true")
    p.add_argument("--substations", nargs="*", default=list(cfg.SUBSTATIONS))
    a = p.parse_args()
    if not any((a.build, a.table, a.sweep, a.plot, a.coverage, a.all)):
        p.print_help()
        return 1

    bpdb = _bpdb()
    ok = True
    if a.build or a.all:
        print(f"\n{'=' * 76}\nBUILD  (Eqs. 13-16)\n{'=' * 76}")
        ok = cmd_build(bpdb, a.substations)
    if a.table or a.all:
        print(f"\n{'=' * 76}\nTABLE\n{'=' * 76}")
        cmd_table(bpdb, a.substations)
    if a.sweep or a.all:
        print(f"\n{'=' * 76}\nUNCERTAINTY PROPAGATION SWEEP\n{'=' * 76}")
        cmd_sweep(bpdb, a.substations)
    if a.plot or a.all:
        print(f"\n{'=' * 76}\nPLOT\n{'=' * 76}")
        cmd_plot(bpdb, a.substations)
    if a.coverage:
        print(f"\n{'=' * 76}\nCOVERAGE TEST\n{'=' * 76}")
        cmd_coverage(bpdb, a.substations, a.coverage)
    return 0 if ok else 3


if __name__ == "__main__":
    sys.exit(main())
