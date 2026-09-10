#!/usr/bin/env python3
"""
Stage 0b runner - adaptive local ToU activation window T^pk_i.

    python 00b_peak_window.py --derive        # derive, cache, report
    python 00b_peak_window.py --sensitivity   # window under 27 estimator settings
    python 00b_peak_window.py --compare       # adaptive vs national window
    python 00b_peak_window.py --plot          # density figure per substation

BERC supplies the peak/off-peak RATIOS (mu^pk = 1.20, mu^off = 0.90); this
stage supplies the ACTIVATION WINDOW, derived per substation from congestion
history. It is not a replacement tariff.
"""

import argparse
import sys

import numpy as np
import pandas as pd

from gentwin import config as cfg
from gentwin import peak_window as pw
from gentwin import tariff as T


def _load_bpdb() -> pd.DataFrame:
    for p in (cfg.BPDB_CSV,
              cfg.DATA_DIR / "BPDB_Dhaka_City_Substations_Page3.csv"):
        if p.exists():
            return pd.read_csv(p, parse_dates=["Date"])
    raise FileNotFoundError(
        f"BPDB CSV not found. Place it at {cfg.BPDB_CSV}")


def _report(w: pw.PeakWindow) -> None:
    nat = sorted(T.OFFICIAL_TOU_PEAK)
    print(f"\n{w.substation}")
    print(f"  T^pk_i (adaptive)   {pw.format_window(w.slots)}"
          f"   [{len(w.slots)} slots]")
    print(f"  national ToU peak   {cfg.slot_to_clock(nat[0])}-"
          f"{cfg.slot_to_clock(nat[-1] + 1)}   [{len(nat)} slots]")
    print(f"  overlap             {w.national_overlap_slots} of {len(w.slots)} "
          f"slots" + ("   << MISALIGNED" if w.national_overlap_slots == 0 else ""))
    print(f"  method              {w.method}")
    print(f"  population          {w.n_days_stress} congestion days "
          f"(peak >= {w.stress_threshold_mw:.1f} MW, p{int(w.stress_quantile * 100)}) "
          f"of {w.n_days_total} usable")
    print(f"  density mass in T^pk {w.mass_in_window:.3f}")
    b = w.bootstrap
    if b:
        print(f"  bootstrap ({b['n']})     start {cfg.slot_to_clock(int(b['start_median']))} "
              f"(IQR {b['start_iqr']:.0f} slots), "
              f"end {cfg.slot_to_clock(int(b['end_median']) + 1)} "
              f"(IQR {b['end_iqr']:.0f}), "
              f"width {b['width_median']:.0f} (IQR {b['width_iqr']:.0f})")
        print(f"  stability           "
              f"{'STABLE - safe to price against' if w.is_stable else 'UNSTABLE - indicative only'}")
    if w.artefact_share:
        print(f"  artefact rows       {w.artefact_share:.1%} dropped "
              f"(peak time recorded as {[cfg.slot_to_clock(s) for s in w.artefact_slots_dropped]})")
    mu = T.mu_profile("LT-A", peak_slots=w.slots)
    print(f"  mu_i(t)             {max(mu):.2f} on T^pk, {min(mu):.2f} elsewhere "
          f"(difference {max(mu) - min(mu):.2f}, the Eq. 34 shifting benefit)")
    for msg in w.warnings:
        print(f"  ! {msg}")


def cmd_derive(df: pd.DataFrame, subs: list) -> None:
    print("BERC supplies the RATIOS (mu^pk=1.20, mu^off=0.90).")
    print("GenTwin-SG supplies the WINDOW, derived below from congestion history.")
    print("This is an activation schedule over the regulator's multipliers,")
    print("not a replacement tariff.\n")
    for name in subs:
        w = pw.derive_peak_window(
            df, name,
            stress_quantile=cfg.PEAK_WINDOW_STRESS_QUANTILE,
            bandwidth=cfg.PEAK_WINDOW_BANDWIDTH_SLOTS,
            threshold_frac=cfg.PEAK_WINDOW_THRESHOLD_FRAC,
            n_bootstrap=cfg.PEAK_WINDOW_BOOTSTRAP)
        alt = pw.unweighted_comparison(df, name)
        path = pw.write_cache(w, {"alternative_populations": alt})
        _report(w)
        print(f"  alternative populations (reported, not used):")
        for k, v in alt.items():
            print(f"     {k:24} n={v['n']:5}  {v['window']}")
        print(f"  cached -> {path}")


def cmd_sensitivity(df: pd.DataFrame, subs: list) -> None:
    print("Window under 27 combinations of estimator settings. If the choices")
    print("were driving the answer rather than the data, it would show here.\n")
    for name in subs:
        s = pw.sensitivity(df, name)
        counts = s["window"].value_counts()
        print(f"{name}: {len(s)} settings -> {len(counts)} distinct windows")
        for win, n in counts.items():
            print(f"   {n:3d}/{len(s)}  {win}")
        modal = counts.index[0]
        print(f"   modal window {modal} in {counts.iloc[0] / len(s):.0%} of settings")
        widths = s["n_slots"]
        print(f"   width range {widths.min()}-{widths.max()} slots "
              f"({widths.min() / 2:.1f}-{widths.max() / 2:.1f} h)\n")
        s.to_csv(cfg.OUT_DIR / f"peak_window_sensitivity_{name}.csv", index=False)


def cmd_compare(df: pd.DataFrame, subs: list) -> None:
    print("Adaptive local vs national ToU window\n")
    nat = sorted(T.OFFICIAL_TOU_PEAK)
    rows = []
    for name in subs:
        rec = pw.read_cache(name)
        if not rec:
            print(f"{name}: no cache; run --derive first.")
            continue
        win = rec["slots"]
        dens = np.array(rec["density"])
        m_ad = float(dens[win].sum())
        m_na = float(dens[nat].sum())
        rows.append({
            "substation": name,
            "adaptive_window": pw.format_window(win),
            "adaptive_slots": len(win),
            "national_slots": len(nat),
            "overlap_slots": len(set(win) & set(nat)),
            "mass_adaptive": round(m_ad, 3),
            "mass_national": round(m_na, 4),
            "precision_adaptive": round(m_ad / len(win), 4),
            "precision_national": round(m_na / len(nat), 4),
            "precision_gain": round((m_ad / len(win)) / (m_na / len(nat)), 1)
            if m_na > 0 else float("inf"),
        })
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    out.to_csv(cfg.OUT_DIR / "tou_window_comparison.csv", index=False)
    print("\n'mass' is the share of stress-day peak-timing density inside the window.")
    print("Mass alone flatters a wide window: pricing all 48 slots captures 100%")
    print("and signals nothing. 'precision' = mass per priced slot is the fair")
    print("comparison - how much congestion each hour of peak pricing actually")
    print("reaches. 'precision_gain' is adaptive / national.")


def cmd_audit(df: pd.DataFrame, subs: list) -> None:
    from gentwin import peak_time_audit as pta
    print("Is `*_Peak_Time` a measurement or a clerical convention?")
    print("T2 in Stage B and T^pk_i here both depend on the answer.\n")
    fleet = pta.audit_fleet(df)
    print(fleet.to_string(index=False, float_format=lambda v: f"{v:8.3f}"))
    fleet.to_csv(cfg.OUT_DIR / "peak_time_audit_fleet.csv", index=False)
    print()
    for name in subs:
        a = pta.audit(df, name)
        print(f"{name}: {a['verdict']}  ({a['n_flags']} flag(s), n={a['n']})")
        print(f"   modal {a['modal_clock']} @ {a['modal_share']:.1%}, "
              f"entropy {a['entropy_bits']:.2f} bits, "
              f"{a['half_hour_share']:.1%} on :30, "
              f"{a['distinct_slots']}/47 distinct slots")
        for f in a["flags"]:
            print(f"   ! {f}")
        if a["seasonal"]:
            print(f"   {'season':14} {'n':>5} {'mode':>7} {'modal share':>12} "
                  f"{'median MW':>10}")
            for s_, v in sorted(a["seasonal"].items(),
                                key=lambda kv: -kv[1]["median_load_mw"]):
                print(f"   {s_:14} {v['n']:5d} {v['mode_clock']:>7} "
                      f"{v['modal_share_global_mode']:11.1%} "
                      f"{v['median_load_mw']:10.1f}")
        print(f"   -> T2 target usable: {a['t2_target_usable']}\n")


def _draw_peak_window_panel(ax, name: str, rec: dict) -> None:
    """Draw one substation's peak-timing density panel onto `ax`. Shared by
    the combined overview figure and the standalone per-substation PDF so
    the two can never drift apart."""
    nat = sorted(T.OFFICIAL_TOU_PEAK)
    dens = np.array(rec["density"])
    win = rec["slots"]
    x = np.arange(cfg.SLOTS_PER_DAY)
    ax.fill_between(x, dens, color="steelblue", alpha=0.35,
                    label="peak-timing density (congestion days)")
    ax.plot(x, dens, color="steelblue", linewidth=1.4)
    ax.axhline(rec["threshold_frac"] * dens.max(), color="grey",
               linestyle=":", linewidth=1,
               label=f"threshold = {rec['threshold_frac']:.2f} x max")
    ax.axvspan(min(win) - 0.5, max(win) + 0.5, color="tomato", alpha=0.20,
               label=f"adaptive T^pk_i  {pw.format_window(win)}")
    ax.axvspan(min(nat) - 0.5, max(nat) + 0.5, color="seagreen", alpha=0.14,
               label=f"national ToU  {cfg.slot_to_clock(nat[0])}-"
                     f"{cfg.slot_to_clock(nat[-1] + 1)}")
    ax.set_xticks(range(0, 48, 4))
    ax.set_xticklabels([cfg.slot_to_clock(s) for s in range(0, 48, 4)],
                       fontsize=8)
    ax.set_xlim(-0.5, 47.5)
    ax.set_ylabel("density")
    ax.set_xlabel("time of day (30-minute slots)")
    ax.set_title(f"{name} - when this substation peaks on its "
                 f"{rec['n_days_stress']} most congested days", fontsize=11)
    ax.legend(fontsize=8, loc="upper left")


def cmd_plot(df: pd.DataFrame, subs: list) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(subs), 1, figsize=(12, 4 * len(subs)), squeeze=False)
    recs = {}
    for ax, name in zip(axes[:, 0], subs):
        rec = pw.read_cache(name)
        if not rec:
            continue
        recs[name] = rec
        _draw_peak_window_panel(ax, name, rec)
        ax.set_xlabel("")
    axes[-1, 0].set_xlabel("time of day (30-minute slots)")
    fig.suptitle("Adaptive local ToU: BERC sets the ratios, the twin sets the window",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p = cfg.OUT_DIR / "adaptive_tou_windows.png"
    fig.savefig(p, dpi=130, bbox_inches="tight")
    print(f"wrote {p}")

    # One true vector PDF per substation, drawn fresh (not cropped from the
    # PNG above), so it stays sharp at any zoom or print size.
    for name, rec in recs.items():
        fig_i, ax_i = plt.subplots(figsize=(12, 4.2))
        _draw_peak_window_panel(ax_i, name, rec)
        fig_i.tight_layout()
        pdf_path = cfg.PDF_DIR / f"adaptive_tou_windows_{name.replace(' ', '_')}.pdf"
        fig_i.savefig(pdf_path, bbox_inches="tight")
        plt.close(fig_i)
        print(f"wrote {pdf_path}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    for f in ("derive", "sensitivity", "compare", "plot", "audit"):
        p.add_argument(f"--{f}", action="store_true")
    p.add_argument("--all", action="store_true")
    p.add_argument("--substations", nargs="*", default=list(cfg.SUBSTATIONS))
    a = p.parse_args()
    if not any((a.derive, a.sensitivity, a.compare, a.plot, a.audit, a.all)):
        p.print_help()
        return 1

    df = _load_bpdb()
    for name, fn in (("audit", cmd_audit), ("derive", cmd_derive),
                     ("sensitivity", cmd_sensitivity),
                     ("compare", cmd_compare), ("plot", cmd_plot)):
        if getattr(a, name) or a.all:
            print(f"\n{'=' * 74}\n{name.upper()}\n{'=' * 74}")
            fn(df, a.substations)
    return 0


if __name__ == "__main__":
    sys.exit(main())
