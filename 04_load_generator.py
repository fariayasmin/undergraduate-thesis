#!/usr/bin/env python3
"""
Stage B runner - bottom-up load generation, Eqs. (2)-(7).

    python 04_load_generator.py --build              # reference day
    python 04_load_generator.py --date 2026-06-25    # any day, solves xi(d)
    python 04_load_generator.py --plot
    python 04_load_generator.py --xi-series 30       # xi(d) over the last N days
"""
import argparse, json, sys
import numpy as np, pandas as pd
from gentwin import (config as cfg, population as pop, load_model as lm,
                     temperature as TM, forecast as F, peak_window as pw)

def _bpdb():
    for p in (cfg.BPDB_CSV, cfg.DATA_DIR / "BPDB_Dhaka_City_Substations_Page3.csv"):
        if p.exists():
            return pd.read_csv(p, parse_dates=["Date"])
    raise FileNotFoundError(f"BPDB CSV not found at {cfg.BPDB_CSV}")

def build_day(bpdb, name, date, cat, sch):
    cons = pop.consumers_from_cache(name)
    theta = np.array(TM.profile_for_date(bpdb, date)["theta_c"])
    agg = lm.aggregate(cons, sch, cat, theta)
    lc = cfg.SUBSTATIONS[name]["load_col"]
    row = bpdb.loc[bpdb["Date"] == pd.Timestamp(date), lc]
    # The BPDB CSV is in MW; section 1.2 works in kW. Convert on read.
    observed = (float(row.iloc[0]) * 1000.0
                if len(row) and pd.notna(row.iloc[0]) else None)
    xi = lm.solve_xi(cons, sch, cat, theta, observed) if observed else None
    return cons, theta, agg, observed, xi

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--date", default=None)
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--xi-series", type=int, default=0)
    ap.add_argument("--substations", nargs="*", default=list(cfg.SUBSTATIONS))
    a = ap.parse_args()
    if not (a.build or a.date or a.plot or a.xi_series):
        ap.print_help(); return 1

    bpdb = _bpdb()
    cat = pop.load_catalogue(); sch = pop.all_schedules(cat)
    store = {}

    for name in a.substations:
        ref = pop.read_cache(name)["reference_day"]
        date = a.date or ref
        cons, theta, agg, observed, xi = build_day(bpdb, name, date, cat, sch)
        store[name] = (cons, theta, agg, date, observed, xi)

        if a.build or a.date:
            print(f"\n{name}  {date}"
                  + ("  (reference day)" if str(date) == ref else ""))
            print(f"  Lambda_i peak   {agg['peak_kw']:.2f} kW at {agg['peak_clock']}"
                  + (f"   observed {observed:.2f} kW" if observed else ""))
            print(f"  energy          {agg['energy_kwh']:.1f} kWh   "
                  f"load factor {agg['load_factor']:.3f}")
            print(f"  by class (kW peak)  " + "  ".join(
                f"{c} {v.max():.2f}" for c, v in agg["by_class_kw"].items()))
            if xi and xi.get("feasible"):
                print(f"  xi({date}) = {xi['xi']:.4f}  -> peak "
                      f"{xi['achieved_peak_kw']:.2f} kW at {xi['peak_clock']} "
                      f"({xi['peak_error_pct']:+.3f}%)")
            lam = agg["lambda_kw"]
            print("  profile kW: " + " ".join(
                f"{cfg.slot_to_clock(i)}:{lam[i]:.0f}" for i in range(0, 48, 4)))
            out = cfg.OUT_DIR / f"lambda_{name}_{date}.csv"
            pd.DataFrame({
                "slot": range(48),
                "clock": [cfg.slot_to_clock(s) for s in range(48)],
                "theta_c": theta,
                "lambda_kw": lam,
                **{f"{c}_kw": v for c, v in agg["by_class_kw"].items()},
                **{f"cat_{c}_kw": v for c, v in agg["by_category_kw"].items()},
            }).to_csv(out, index=False)
            print(f"  wrote {out}")

    if a.xi_series:
        print(f"\n{'=' * 74}\nxi(d) OVER THE LAST {a.xi_series} OBSERVED DAYS\n{'=' * 74}")
        for name in a.substations:
            cons = pop.consumers_from_cache(name)
            lc = cfg.SUBSTATIONS[name]["load_col"]
            obs = bpdb[["Date", lc]].dropna().tail(a.xi_series)
            rows = []
            for _, r in obs.iterrows():
                try:
                    th = np.array(TM.profile_for_date(bpdb, r["Date"])["theta_c"])
                except ValueError:
                    continue
                s = lm.solve_xi(cons, sch, cat, th, float(r[lc]) * 1000.0)
                if s.get("feasible"):
                    rows.append({"date": str(r["Date"].date()),
                                 "observed_kw": float(r[lc]) * 1000.0, "xi": s["xi"],
                                 "peak_clock": s["peak_clock"]})
            df = pd.DataFrame(rows)
            df.to_csv(cfg.OUT_DIR / f"xi_series_{name}.csv", index=False)
            print(f"{name}: xi mean {df['xi'].mean():.3f}, sd {df['xi'].std():.3f}, "
                  f"range {df['xi'].min():.3f}-{df['xi'].max():.3f}  (n={len(df)})")
            print(f"  |xi-1| > 0.25 on {(abs(df['xi']-1)>0.25).sum()} of {len(df)} days"
                  f"  -- large values mean the catalogue does not describe that day")

    if a.plot:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        def draw_by_category(ax, name, date, agg, observed, t_pk):
            x = np.arange(48)
            base = np.zeros(48)
            for c in ["Residential", "Commercial", "Educational", "Hospital",
                      "Government", "Industrial"]:
                if c not in agg["by_category_kw"]: continue
                v = agg["by_category_kw"][c]
                ax.fill_between(x, base, base + v, alpha=0.75, label=c)
                base = base + v
            ax.plot(x, agg["lambda_kw"], color="black", linewidth=1.6,
                    label=r"$\Lambda_i(t)$")
            if observed:
                ax.axhline(observed, color="red", linestyle="--", linewidth=1.3,
                           label=f"BPDB peak {observed:.1f} kW")
            ax.axvspan(min(t_pk) - .5, max(t_pk) + .5, color="orange", alpha=0.18,
                       label=f"$T^{{pk}}_i$ {pw.format_window(t_pk)}")
            ax.axvline(agg["peak_slot"], color="black", linestyle=":", linewidth=1)
            ax.set_xticks(range(0, 48, 4))
            ax.set_xticklabels([cfg.slot_to_clock(s) for s in range(0, 48, 4)], fontsize=8)
            ax.set_xlim(0, 47); ax.set_ylabel("kW")
            ax.set_title(f"{name} {date} - by category; peak {agg['peak_clock']}",
                         fontsize=11)
            ax.legend(fontsize=7.5, loc="upper left", ncol=2)

        def draw_by_class(ax, name, agg, theta, t_pk):
            x = np.arange(48)
            base = np.zeros(48)
            colors = {"critical": "#b03060", "shiftable": "#3878a8", "curtailable": "#4fa64f"}
            for c, v in agg["by_class_kw"].items():
                ax.fill_between(x, base, base + v, alpha=0.8, label=c, color=colors[c])
                base = base + v
            ax2 = ax.twinx()
            ax2.plot(x, theta, color="darkorange", linewidth=1.6, linestyle="--",
                     label=r"$\Theta(t)$")
            ax2.set_ylabel(r"$\Theta$ (C)", color="darkorange")
            ax.axvspan(min(t_pk) - .5, max(t_pk) + .5, color="orange", alpha=0.15)
            ax.set_xticks(range(0, 48, 4))
            ax.set_xticklabels([cfg.slot_to_clock(s) for s in range(0, 48, 4)], fontsize=8)
            ax.set_xlim(0, 47); ax.set_ylabel("kW")
            ax.set_title(f"{name} - by class C, with $\\Theta(t)$", fontsize=11)
            ax.legend(fontsize=8, loc="upper left")

        n = len(store)
        fig, axes = plt.subplots(2, n, figsize=(8 * n, 9), squeeze=False)
        for j, (name, (cons, theta, agg, date, observed, xi)) in enumerate(store.items()):
            t_pk = sorted(pw.get_peak_slots(name))
            draw_by_category(axes[0, j], name, date, agg, observed, t_pk)
            draw_by_class(axes[1, j], name, agg, theta, t_pk)
        fig.suptitle("Stage B: bottom-up synthetic load, Eqs. (2)-(6)",
                     fontsize=13, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        p = cfg.OUT_DIR / "load_generation.png"
        fig.savefig(p, dpi=130, bbox_inches="tight")
        print(f"\nwrote {p}")

        # True vector PDFs, one per panel, drawn fresh at full resolution.
        for name, (cons, theta, agg, date, observed, xi) in store.items():
            t_pk = sorted(pw.get_peak_slots(name))
            safe = name.replace(" ", "_")

            fig_i, ax_i = plt.subplots(figsize=(8, 4.5))
            draw_by_category(ax_i, name, date, agg, observed, t_pk)
            fig_i.tight_layout()
            p1 = cfg.PDF_DIR / f"load_generation_{safe}_by_category.pdf"
            fig_i.savefig(p1, bbox_inches="tight")
            plt.close(fig_i)
            print(f"wrote {p1}")

            fig_i, ax_i = plt.subplots(figsize=(8, 4.5))
            draw_by_class(ax_i, name, agg, theta, t_pk)
            fig_i.tight_layout()
            p2 = cfg.PDF_DIR / f"load_generation_{safe}_by_class.pdf"
            fig_i.savefig(p2, bbox_inches="tight")
            plt.close(fig_i)
            print(f"wrote {p2}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
