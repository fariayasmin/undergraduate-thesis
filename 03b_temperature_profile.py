#!/usr/bin/env python3
"""
Stage A/B runner - diurnal temperature Theta(t).

    python 03b_temperature_profile.py --date 2026-06-29
    python 03b_temperature_profile.py --date 2026-06-29 --plot

Theta(t) is RECONSTRUCTED from daily Tmin/Tmax/Tmean, not observed. It drives
f_a(Theta) in Eq. (2) and therefore substantially determines the afternoon
cooling load, so its provenance is stamped on every artefact.
"""
import argparse, json, sys
import numpy as np, pandas as pd
from gentwin import config as cfg, temperature as TM

def _bpdb():
    for p in (cfg.BPDB_CSV, cfg.DATA_DIR / "BPDB_Dhaka_City_Substations_Page3.csv"):
        if p.exists():
            return pd.read_csv(p, parse_dates=["Date"])
    raise FileNotFoundError(f"BPDB CSV not found at {cfg.BPDB_CSV}")

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", default="2026-06-29")
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args()

    b = _bpdb()
    p = TM.profile_for_date(b, a.date)
    th = np.array(p["theta_c"])

    print(f"Theta(t) for {p['date']}  ({p['day_of_week']}, {p['season']}, "
          f"holiday={p['holiday_type']})")
    print(f"  method       {p['method']}")
    print(f"  provenance   {p['provenance']}")
    print(f"  inputs       Tmin {p['t_min_c']} C, Tmax {p['t_max_c']} C, "
          f"Tmean {p['t_mean_recorded_c']} C, humidity {p['humidity_pct']}%, "
          f"rain {p['rain_mm']} mm")
    print(f"  daylight     sunrise {p['sunrise_h']:.2f}h, sunset "
          f"{p['sunset_h']:.2f}h, length {p['day_length_h']:.2f}h")
    print(f"  Theta peak   {p['peak_clock']} at {th.max():.2f} C; "
          f"min {th.min():.2f} C; mean {th.mean():.3f} C")
    print("\n  " + "  ".join(f"{cfg.slot_to_clock(i)} {th[i]:5.1f}"
                             for i in range(0, 48, 3)))
    print("\n  checks")
    ok = True
    for c in TM.validate(p):
        print(f"    [{'PASS' if c['pass'] else 'FAIL'}] {c['check']}"
              + (f"  ({c['detail']})" if c["detail"] else ""))
        ok &= c["pass"]

    out = cfg.CACHE_DIR / f"theta_{p['date']}.json"
    out.write_text(json.dumps(p, indent=1), encoding="utf-8")
    print(f"\n  cached -> {out}")

    if a.plot:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(11, 4))
        x = np.arange(48)
        ax.plot(x, th, color="darkorange", linewidth=2, label=r"$\Theta(t)$ reconstructed")
        ax.axhline(p["t_max_c"], color="tomato", linestyle=":", label="recorded $T_{max}$")
        ax.axhline(p["t_min_c"], color="steelblue", linestyle=":", label="recorded $T_{min}$")
        ax.axhline(p["t_mean_recorded_c"], color="grey", linestyle="--", label="recorded $T_{mean}$")
        ax.axvspan(p["sunrise_h"] * 2, p["sunset_h"] * 2, color="gold", alpha=0.12, label="daylight")
        for k, spec in (("cooling_ac", 24.0), ("cooling_fan", 26.0)):
            ax.axhline(spec, color="purple", alpha=0.35, linewidth=0.8)
            ax.text(0.5, spec + 0.1, f"{k} $\\theta_{{ref}}$", fontsize=7, color="purple")
        ax.set_xticks(range(0, 48, 4))
        ax.set_xticklabels([cfg.slot_to_clock(s) for s in range(0, 48, 4)], fontsize=8)
        ax.set_xlim(0, 47); ax.set_ylabel("$\\Theta$ (C)")
        ax.set_title(f"Reconstructed diurnal temperature, {p['date']} "
                     f"(Parton-Logan, mean-corrected)", fontsize=11)
        ax.legend(fontsize=8)
        fig.tight_layout()
        f = cfg.OUT_DIR / f"theta_profile_{p['date']}.png"
        fig.savefig(f, dpi=130, bbox_inches="tight")
        print(f"  wrote {f}")

        # True vector PDF (same figure, no rasterisation).
        f_pdf = cfg.PDF_DIR / f"theta_profile_{p['date']}.pdf"
        fig.savefig(f_pdf, bbox_inches="tight")
        print(f"  wrote {f_pdf}")
    return 0 if ok else 3

if __name__ == "__main__":
    sys.exit(main())
