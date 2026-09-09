#!/usr/bin/env python3
"""
Stage A/B runner - synthetic consumer population.

    python 03_population_synthesis.py --build
    python 03_population_synthesis.py --report

Population is built from the OSM service-area mix (never at random), calibrated
to the BPDB reference-day peak, and stamped with the reference day so that
Assumption 1 cannot be violated silently.
"""
import argparse, sys
import numpy as np, pandas as pd
from gentwin import (config as cfg, population as pop, load_model as lm,
                     temperature as TM, forecast as F, service_area as sa)

def _bpdb():
    for p in (cfg.BPDB_CSV, cfg.DATA_DIR / "BPDB_Dhaka_City_Substations_Page3.csv"):
        if p.exists():
            return pd.read_csv(p, parse_dates=["Date"])
    raise FileNotFoundError(f"BPDB CSV not found at {cfg.BPDB_CSV}")

def build(bpdb, subs):
    cat = pop.load_catalogue()
    sch = pop.all_schedules(cat)
    for name in subs:
        krec = F.read_cache(name)
        if not krec:
            print(f"{name}: run 05c_forecast_to_kappa.py --build first "
                  f"(the reference day comes from there).")
            continue
        ref_date = krec["reference_day"]["date"]
        target = float(krec["reference_day"]["peak_kw"])
        F.assert_population_alignment(name, ref_date)   # Assumption 1

        theta = np.array(TM.profile_for_date(bpdb, ref_date)["theta_c"])
        mix = pop.demand_mix_for(name)
        osm = sa.read_cache(name)

        cons = pop.synthesise(name, cat, mix)
        diag = lm.set_weights(cons, sch, cat, theta, mix, target)
        agg = lm.aggregate(cons, sch, cat, theta)

        meta = {
            "reference_day": ref_date,
            "target_peak_kw": target,
            "osm_provenance": osm["provenance"],
            "osm_mix": mix,
            "calibration": {k: v for k, v in diag.items()},
            "seed": cfg.POPULATION_SEED,
            "catalogue": str(pop.CATALOGUE_PATH.name),
            "achieved_peak_kw": agg["peak_kw"],
            "achieved_peak_clock": agg["peak_clock"],
            "load_factor": agg["load_factor"],
            "energy_kwh": agg["energy_kwh"],
        }
        path = pop.write_cache(name, cons, meta)

        print(f"\n{name}")
        print(f"  reference day        {ref_date}   target peak {target:.2f} kW")
        print(f"  OSM provenance       {osm['provenance']}"
              + ("   [FALLBACK - not surveyed]"
                 if osm["provenance"] != "overpass" else ""))
        print(f"  representatives      {len(cons)}")
        print(f"  implied connections  {diag['total_connections_implied']:,.0f}")
        print(f"  calibration K        {diag['global_constant_K']:.6g}")
        print(f"  peak achieved        {agg['peak_kw']:.2f} kW at "
              f"{agg['peak_clock']}  ({diag['peak_error_pct']:+.4f}%)")
        print(f"  OSM mix reproduced   max share error "
              f"{diag['max_share_error']:.2e}")
        print(f"  {'category':13} {'share':>7} {'reps':>6} {'connections':>13} "
              f"{'peak kW':>9}")
        for c in cfg.CONSUMER_CATEGORIES:
            n = sum(1 for h in cons if h.category == c)
            if not n:
                continue
            w = diag["connections_by_category"].get(c, 0.0)
            pk = float(agg["by_category_kw"][c].max())
            print(f"  {c:13} {mix.get(c,0):7.3f} {n:6d} {w:13,.0f} {pk:9.2f}")
        print(f"  tiers                " + ", ".join(
            f"{t} {sum(1 for h in cons if h.archetype==t)}"
            for t in sorted({h.archetype for h in cons})))
        print(f"  cached -> {path}")

def report(subs):
    for name in subs:
        rec = pop.read_cache(name)
        print(f"\n{name}: {rec['n_consumers']} representatives, reference day "
              f"{rec['reference_day']}, OSM {rec['osm_provenance']}")
        cons = pop.consumers_from_cache(name)
        w = np.array([c.w for c in cons])
        print(f"  w_h: min {w.min():,.1f}  median {np.median(w):,.1f}  "
              f"max {w.max():,.1f}  total {w.sum():,.0f}")
        napp = np.array([len(c.appliances) for c in cons])
        print(f"  appliances per consumer: min {napp.min()} median "
              f"{int(np.median(napp))} max {napp.max()}")
        cl = np.array([c.connected_load_kw for c in cons])
        print(f"  connected load per consumer (kW): min {cl.min():,.2f} "
              f"median {np.median(cl):,.2f} max {cl.max():,.2f}")

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--substations", nargs="*", default=list(cfg.SUBSTATIONS))
    a = ap.parse_args()
    if not (a.build or a.report):
        ap.print_help(); return 1
    b = _bpdb()
    if a.build:
        print("=" * 74); print("POPULATION SYNTHESIS"); print("=" * 74)
        build(b, a.substations)
    if a.report:
        print("\n" + "=" * 74); print("POPULATION REPORT"); print("=" * 74)
        report(a.substations)
    return 0

if __name__ == "__main__":
    sys.exit(main())
