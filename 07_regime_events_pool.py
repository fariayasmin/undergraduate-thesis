#!/usr/bin/env python3
"""
Stage H runner - regime, events and priority pooling, Eqs. (44)-(53).

    python 07_regime_events_pool.py --run
    python 07_regime_events_pool.py --run --scarcity   # read the scarcity solve
    python 07_regime_events_pool.py --run --pool-rule quantile
    python 07_regime_events_pool.py --plot

Reads the LP outputs written by 06_lp_optimiser.py. The regime is computed
AFTER the solve because Eq. (48) reads T_ij and nu_i, which are LP outputs;
s_{i,d} comes from the forecast and is a separate quantity.
"""

import argparse
import json
import sys
from dataclasses import asdict

import numpy as np
import pandas as pd

from gentwin import config as cfg, forecast as F, population as pop, regime as R


def _load_lp(tag=""):
    net = pd.read_csv(cfg.OUT_DIR / f"lp_network{tag}.csv")
    dec = pd.read_csv(cfg.OUT_DIR / f"lp_decisions{tag}.csv")
    thr = pd.read_csv(cfg.OUT_DIR / f"lp_thresholds{tag}.csv")
    summ = pd.read_csv(cfg.OUT_DIR / f"lp_summary{tag}.csv")
    return net, dec, thr, summ


def run(subs, tag="", pool_rule=None):
    net_df, dec_df, thr_df, summ = _load_lp(tag)
    POP = {i: pop.consumers_from_cache(i) for i in subs}
    krec = {i: F.read_cache(i) for i in subs}

    all_events, all_pools, all_entities, prev_pool = [], [], [], None

    for date in sorted(net_df["date"].unique()):
        nd = net_df[net_df["date"] == date]
        td = thr_df[thr_df["date"] == date] if len(thr_df) else thr_df

        network, regimes, frows = {}, {}, {}
        for i in subs:
            g = nd[nd["substation"] == i].sort_values("slot")
            network[i] = {
                "G": g["G_kw"].values, "Bch": g["Bch_kw"].values,
                "Bdis": g["Bdis_kw"].values, "T_out": g["T_out_kw"].values,
                "T_in": g["T_in_kw"].values, "nu": g["nu_kwh"].values,
                "Z_critical": g["Z_critical_kw"].values,
                "Z_shiftable": g["Z_shiftable_kw"].values,
                "Z_curtailable": g["Z_curtailable_kw"].values,
                "SoC": g["SoC_kwh"].values, "pi": g["pi_tk_per_kwh"].values,
                "post_response_kw": g["post_response_kw"].values,
            }
            frow = next(r for r in krec[i]["rows"] if r["target_date"] == date)
            frows[i] = frow
            regimes[i] = R.operating_regime(                       # (48)
                p_tilde_kw=frow["p_tilde_kw"],
                p_max_kw=cfg.SUBSTATIONS[i]["p_max_kw"],
                transfer_out_kwh=float(network[i]["T_out"].sum() * cfg.DELTA_H),
                nu_kwh=float(network[i]["nu"].sum()),
                s_stress=frow["s_stress"])

        # per-consumer response energies from the threshold table
        decisions = {"__network__": {}}
        for i in subs:
            zt = float(sum(network[i][f"Z_{c}"].sum()
                           for c in ("critical", "shiftable", "curtailable"))
                       * cfg.DELTA_H)
            decisions["__network__"][i] = {
                "Z_total_kwh": zt,
                "f_cost_tk": float(summ.loc[summ.date == date,
                                            f"f_cost_tk_{i}"].iloc[0])
                if f"f_cost_tk_{i}" in summ.columns else 0.0,
            }
        if len(td):
            for cid, g in td.groupby("consumer"):
                sub = g["substation"].iloc[0]
                decisions[cid] = {
                    "substation": sub,
                    "e_sh_kwh": float(g.loc[g.action == "shift", "energy_kwh"].sum()),
                    "e_cu_kwh": float(g.loc[g.action == "curtail", "energy_kwh"].sum()),
                    "x_max": float(g.loc[g.action == "shift", "value"].max())
                    if (g.action == "shift").any() else 0.0,
                    "y_max": float(g.loc[g.action == "curtail", "value"].max())
                    if (g.action == "curtail").any() else 0.0,
                }

        ents = R.build_entities(date, subs, POP, decisions, regimes, frows)
        ents = R.score_entities(ents, regimes, frows)
        pool = R.assign_pools(ents, pool_rule)
        pool["date"] = date
        if prev_pool is not None:
            pool["alpha_P"] = R.repooling_trigger(prev_pool, pool)     # (52)
        prev_pool = pool

        topn = R.top_n(ents)                                            # (53)
        pool["top_n"] = [{"entity_id": e.entity_id, "kind": e.kind,
                          "substation": e.substation, "chi": e.chi,
                          "pool": e.pool} for e in topn]

        events = R.generate_events(date, subs, regimes, frows, decisions,
                                   network, td.to_dict("records") if len(td) else [])
        all_events.extend(events)
        all_pools.append(pool)
        for e in ents:
            d = asdict(e)
            d["phi1"], d["phi2"], d["phi3"], d["phi4"] = e.phi
            d.pop("phi")
            d["value"] = json.dumps(d["value"])
            all_entities.append(d)

        print(f"\n  {date}")
        for i in subs:
            r = regimes[i]
            trg = [k for k, v in r["triggers_r3"].items() if v]
            print(f"     {i:<11} r={r['regime']} ({r['regime_name']:<9}) "
                  f"s={r['s_stress']}  p~={r['p_tilde_kw']:,.0f} kW  "
                  f"Def={r['deficit_kw']:,.0f}  T_out={r['transfer_out_kwh']:,.0f} kWh  "
                  f"nu={r['nu_kwh']:.1f} kWh"
                  + (f"  r3 triggers: {', '.join(trg)}" if trg else ""))
            print(f"     {'':<11} beta(r) = {r['beta']}")
        print(f"     entities {pool['n_entities']:,}  chi_bar {pool['chi_bar']:.4f}  "
              f"spread {pool['chi_spread']:.4f}  rule={pool['rule']}")
        for p in ("PH", "PM", "PL"):
            q = pool["pools"][p]
            print(f"     {p}: n={q['n']:>5}  W(P)={q['W']:.4f}  "
                  f"chi {q['chi_min']:.4f}-{q['chi_max']:.4f}")
        if pool["degenerate"]:
            print(f"     ! POOL DEGENERATE - an empty pool. Section 9.6 predicts "
                  f"this for the mean-relative rule on tight score "
                  f"distributions; re-run with --pool-rule quantile.")
        if "alpha_P" in pool:
            print(f"     alpha_P = " + ", ".join(f"{k} {v:.5f}"
                                                 for k, v in pool["alpha_P"].items()))
        print(f"     events: " + (", ".join(f"{e['trigger']}@{e['substation']}"
                                            for e in events) or "none"))

    pd.DataFrame(all_events).to_csv(cfg.OUT_DIR / f"events{tag}.csv", index=False)
    pd.DataFrame(all_entities).to_csv(cfg.OUT_DIR / f"entities{tag}.csv", index=False)
    (cfg.OUT_DIR / f"priority_pools{tag}.json").write_text(
        json.dumps(all_pools, indent=1, default=str), encoding="utf-8")
    ranked = pd.DataFrame([
        {"date": p["date"], "rank": k + 1, **e}
        for p in all_pools for k, e in enumerate(p["top_n"])])
    ranked.to_csv(cfg.OUT_DIR / f"priority_pool_ranked{tag}.csv", index=False)
    print(f"\n  wrote events{tag}.csv ({len(all_events)}), "
          f"entities{tag}.csv ({len(all_entities)}), "
          f"priority_pools{tag}.json, priority_pool_ranked{tag}.csv "
          f"({len(ranked)} rows)")
    return all_events, all_pools, all_entities


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--scarcity", action="store_true")
    ap.add_argument("--pool-rule", choices=["mean_relative", "quantile"],
                    default=None)
    ap.add_argument("--substations", nargs="*", default=list(cfg.SUBSTATIONS))
    a = ap.parse_args()
    if not a.run:
        ap.print_help(); return 1
    tag = "_scarcity" if a.scarcity else ""
    print("=" * 78)
    print(f"STAGE H - REGIME, EVENTS, PRIORITY POOL{'  [scarcity scenario]' if tag else ''}")
    print("=" * 78)
    run(a.substations, tag, a.pool_rule)
    return 0


if __name__ == "__main__":
    sys.exit(main())
