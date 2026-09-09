#!/usr/bin/env python3
"""
Stage B gate - validate the synthetic aggregate against BPDB.

    python 05_validate_against_bpdb.py --validate
    python 05_validate_against_bpdb.py --diagnose-mix    # only if T2 fails
    python 05_validate_against_bpdb.py --ablation        # category-mix ablation

T1 (peak LEVEL) is satisfied by construction. T2 (peak TIMING) is not fitted
anywhere and is the real test of the bottom-up model. The LP should not be run
on a population that fails T2 without the failure being understood.
"""
import argparse, json, sys
import numpy as np, pandas as pd
from gentwin import (config as cfg, population as pop, load_model as lm,
                     temperature as TM, validation as V, peak_window as pw,
                     peak_time_audit as pta)

def _bpdb():
    for p in (cfg.BPDB_CSV, cfg.DATA_DIR / "BPDB_Dhaka_City_Substations_Page3.csv"):
        if p.exists():
            return pd.read_csv(p, parse_dates=["Date"])
    raise FileNotFoundError(f"BPDB CSV not found at {cfg.BPDB_CSV}")

def _setup(bpdb, name):
    cat = pop.load_catalogue(); sch = pop.all_schedules(cat)
    rec = pop.read_cache(name)
    cons = pop.consumers_from_cache(name)
    theta = np.array(TM.profile_for_date(bpdb, rec["reference_day"])["theta_c"])
    agg = lm.aggregate(cons, sch, cat, theta)
    return cat, sch, rec, cons, theta, agg

def cmd_validate(bpdb, subs):
    gate_ok = True
    summary = {}
    for name in subs:
        cat, sch, rec, cons, theta, agg = _setup(bpdb, name)
        target = float(rec["target_peak_kw"])
        tc = cfg.SUBSTATIONS[name]["peak_time_col"]
        row = bpdb.loc[bpdb["Date"] == pd.Timestamp(rec["reference_day"]), tc]
        obs_slot = None
        if len(row) and isinstance(row.iloc[0], str) and ":" in str(row.iloc[0]):
            try:
                s = cfg.clock_to_slot(str(row.iloc[0]))
                obs_slot = None if s in cfg.PEAK_TIME_ARTEFACT_SLOTS else s
            except ValueError:
                pass

        av = pta.audit(bpdb, name)
        stab = V.timing_stability(name, cat, sch, theta,
                                  pop.demand_mix_for(name), target)
        checks = V.acceptance(name, agg, target, obs_slot,
                              audit_verdict=av, stability=stab)
        df = V.diversity_factor(cons, agg)

        print(f"\n{name}   reference day {rec['reference_day']}   "
              f"OSM {rec['osm_provenance']}")
        print(f"  Lambda peak {agg['peak_kw']:.2f} kW at {agg['peak_clock']}   "
              f"target {target:.2f} kW   T^pk_i "
              f"{pw.format_window(sorted(pw.get_peak_slots(name)))}")
        print(f"  load factor {agg['load_factor']:.3f}   diversity factor "
              f"{df:.2f}   energy {agg['energy_kwh']:.1f} kWh")
        if obs_slot is None:
            print(f"  note: the reference day's recorded peak TIME is unusable "
                  f"(00:00 artefact); T2b skipped, T2 uses the distributional "
                  f"window instead.")
        print(f"  Peak_Time audit: {av['verdict']} ({av['n_flags']} flags, "
              f"{av['half_hour_share']:.1%} of readings on :30)"
              + ("   -> T2 is CONDITIONAL, not a gate"
                 if not av["t2_target_usable"] else ""))
        print(f"  T2 seed stability: peak inside T^pk_i on "
              f"{stab['n_inside']}/{stab['n_seeds']} draws "
              f"({stab['share_inside']:.0%})  -> {stab['verdict']}")
        print(f"     draw distribution: "
              + ", ".join(f"{k} x{v}" for k, v in stab["distribution"].items()))
        for c in checks:
            tag = ("PASS" if c["pass"] else
                   ("COND" if c.get("conditional") else
                    "FAIL" if c["gate"] else "warn"))
            print(f"    [{tag}] {c['check']}")
            if c["detail"]:
                print(f"           {c['detail']}")
            if c["gate"] and not c["pass"]:
                gate_ok = False
        summary[name] = {
            "peak_kw": agg["peak_kw"], "peak_clock": agg["peak_clock"],
            "target_kw": target, "load_factor": agg["load_factor"],
            "diversity_factor": df,
            "gates_passed": all(c["pass"] for c in checks if c["gate"]),
            "timing_stability": stab,
            "checks": checks,
        }
    (cfg.OUT_DIR / "stageB_validation.json").write_text(
        json.dumps(summary, indent=1, default=str), encoding="utf-8")
    print(f"\nwrote {cfg.OUT_DIR / 'stageB_validation.json'}")
    print(f"\nSTAGE B GATE: {'PASS' if gate_ok else 'FAIL'}")
    print(f"  Per-substation verdicts - do NOT report these as a single "
          f"'both passed':")
    for name, v in summary.items():
        t2 = next((c for c in v["checks"]
                   if c["check"].startswith("T2 peak timing")), None)
        st = v["timing_stability"]
        verdict = ("ROBUST PASS" if st["share_inside"] >= 0.90 else
                   "ROBUST FAIL" if st["share_inside"] <= 0.10 else
                   "CONDITIONAL / UNDETERMINED")
        print(f"     {name:<12} T1 level PASS (by construction)   "
              f"T2 timing {verdict} "
              f"({st['n_inside']}/{st['n_seeds']} draws)"
              + ("   [Peak_Time target rated LIKELY_CONVENTION]"
                 if t2 and t2.get("conditional") else ""))
    if not gate_ok:
        print("  Run --diagnose-mix before proceeding to the LP.")
    return gate_ok

def cmd_diagnose(bpdb, subs):
    print("DIAGNOSTIC ONLY. The mix used by the pipeline always comes from the")
    print("OSM cache and is never set from this function.\n")
    for name in subs:
        cat, sch, rec, cons, theta, agg = _setup(bpdb, name)
        mix = pop.demand_mix_for(name)
        d = V.critical_residential_share(name, cat, sch, theta, mix,
                                         float(rec["target_peak_kw"]))
        print(f"{name}: residential share currently {d['current_share']:.3f}")
        if d["max_share_passing"] is None:
            print("  no residential share in 0.20-0.65 reproduces the observed "
                  "peak timing; the problem is not the mix.")
        else:
            print(f"  T2 passes for residential share <= {d['max_share_passing']:.2f}"
                  f"   (margin {d['margin']:+.3f} from the current value)")
        prev = None
        for r in d["rows"]:
            if prev is not None and r["inside_t_pk"] != prev:
                print(f"  flip at residential share ~{r['residential_share']:.2f}: "
                      f"peak moves to {r['peak_clock']}")
            prev = r["inside_t_pk"]
        pd.DataFrame(d["rows"]).to_csv(
            cfg.OUT_DIR / f"mix_diagnostic_{name}.csv", index=False)
        print(f"  wrote {cfg.OUT_DIR / f'mix_diagnostic_{name}.csv'}\n")

def cmd_ablation(bpdb, subs):
    print("Category-mix ablation: does the composition do real work, or is the")
    print("peak timing coming from the residential schedules alone?\n")
    for name in subs:
        cat, sch, rec, cons, theta, agg = _setup(bpdb, name)
        mix = pop.demand_mix_for(name)
        target = float(rec["target_peak_kw"])
        rows = [{"variant": "full mix", "peak_clock": agg["peak_clock"],
                 "peak_kw": round(agg["peak_kw"], 2)}]
        m2 = {"Residential": 1.0}
        c2 = pop.synthesise(name, cat, m2)
        lm.set_weights(c2, sch, cat, theta, m2, target)
        a2 = lm.aggregate(c2, sch, cat, theta)
        rows.append({"variant": "residential only", "peak_clock": a2["peak_clock"],
                     "peak_kw": round(a2["peak_kw"], 2)})
        nonres = {k: v for k, v in mix.items() if k != "Residential"}
        tot = sum(nonres.values())
        m3 = {k: v / tot for k, v in nonres.items()}
        c3 = pop.synthesise(name, cat, m3)
        lm.set_weights(c3, sch, cat, theta, m3, target)
        a3 = lm.aggregate(c3, sch, cat, theta)
        rows.append({"variant": "non-residential only", "peak_clock": a3["peak_clock"],
                     "peak_kw": round(a3["peak_kw"], 2)})
        print(f"{name}  (T^pk_i {pw.format_window(sorted(pw.get_peak_slots(name)))})")
        print(pd.DataFrame(rows).to_string(index=False))
        print()

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--diagnose-mix", action="store_true")
    ap.add_argument("--ablation", action="store_true")
    ap.add_argument("--substations", nargs="*", default=list(cfg.SUBSTATIONS))
    a = ap.parse_args()
    if not (a.validate or a.diagnose_mix or a.ablation):
        ap.print_help(); return 1
    b = _bpdb()
    ok = True
    if a.validate:
        print("=" * 74); print("STAGE B VALIDATION AGAINST BPDB"); print("=" * 74)
        ok = cmd_validate(b, a.substations)
    if a.ablation:
        print("\n" + "=" * 74); print("CATEGORY-MIX ABLATION"); print("=" * 74)
        cmd_ablation(b, a.substations)
    if a.diagnose_mix:
        print("\n" + "=" * 74); print("MIX DIAGNOSTIC"); print("=" * 74)
        cmd_diagnose(b, a.substations)
    return 0 if ok else 3

if __name__ == "__main__":
    sys.exit(main())
