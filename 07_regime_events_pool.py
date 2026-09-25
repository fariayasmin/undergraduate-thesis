#!/usr/bin/env python3
"""
Stage H runner - regime, events and priority pooling, Eqs. (44)-(53).

    python 07_regime_events_pool.py --run
    python 07_regime_events_pool.py --run --scarcity        # read the scarcity solve
    python 07_regime_events_pool.py --run --tag _period     # read a --month/--period solve
    python 07_regime_events_pool.py --run --pool-rule quantile
    python 07_regime_events_pool.py --plot

Reads the LP outputs written by 06_lp_optimiser.py. The regime is computed
AFTER the solve because Eq. (48) reads T_ij and nu_i, which are LP outputs;
s_{i,d} comes from the forecast and is a separate quantity.

`--tag` must match whatever 06_lp_optimiser.py was run with (e.g. `_period`
for `--month 30` / `--period ...`), so this stage covers the same days the LP
solved rather than silently falling back to the untagged 3-day default.
"""

import argparse
import copy
import json
import sys
from dataclasses import asdict

import numpy as np
import pandas as pd

from gentwin import config as cfg, forecast as F, population as pop, regime as R


def _bpdb() -> pd.DataFrame:
    for p in (cfg.BPDB_CSV, cfg.DATA_DIR / "BPDB_Dhaka_City_Substations_Page3.csv"):
        if p.exists():
            return pd.read_csv(p, parse_dates=["Date"])
    raise FileNotFoundError(f"BPDB CSV not found at {cfg.BPDB_CSV}")


def _load_lp(tag=""):
    net = pd.read_csv(cfg.OUT_DIR / f"lp_network{tag}.csv")
    dec = pd.read_csv(cfg.OUT_DIR / f"lp_decisions{tag}.csv")
    thr = pd.read_csv(cfg.OUT_DIR / f"lp_thresholds{tag}.csv")
    summ = pd.read_csv(cfg.OUT_DIR / f"lp_summary{tag}.csv")
    return net, dec, thr, summ


def run(subs, tag="", pool_rule=None):
    net_df, dec_df, thr_df, summ = _load_lp(tag)
    bpdb = _bpdb()
    POP = {i: pop.consumers_from_cache(i) for i in subs}

    all_events, all_pools, all_entities, prev_pool, prev_membership = [], [], [], None, None
    candidate_log = []   # raw (never-reused) candidate W(P) per day, for --alpha-sweep
    precision_log = []   # item 2: precision@N vs random baseline, per day
    trigger_log = []     # item 3: alpha_P vs frac-changed re-pool decision, per day

    # Round-3 follow-up: read the EFFECTIVE (already-scaled) P^max/G^max that
    # 06_lp_optimiser.py actually solved against, per substation, from
    # lp_summary{tag}.csv - cfg.SUBSTATIONS in THIS process was never scaled
    # (apply_scenario() only mutated the LP-solve process's own memory), so
    # reading cfg directly here silently used the wrong (unscaled) P^max for
    # every scenario arm and the capacity_deficit trigger never fired.
    p_max_eff = {i: float(summ[f"p_max_kw_eff_{i}"].iloc[0])
                if f"p_max_kw_eff_{i}" in summ.columns
                else cfg.SUBSTATIONS[i]["p_max_kw"] for i in subs}
    scale_gmax = {i: float(summ[f"scale_gmax_{i}"].iloc[0])
                 if f"scale_gmax_{i}" in summ.columns else 1.0 for i in subs}
    scale_pmax = {i: float(summ[f"scale_pmax_{i}"].iloc[0])
                 if f"scale_pmax_{i}" in summ.columns else 1.0 for i in subs}
    if any(scale_gmax[i] != 1.0 or scale_pmax[i] != 1.0 for i in subs):
        print("  SCENARIO (read from lp_summary" + tag + "): " +
              ", ".join(f"{i} G^max x{scale_gmax[i]} P^max x{scale_pmax[i]} "
                        f"(P^max_eff={p_max_eff[i]:,.0f} kW)" for i in subs))

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
            # F.build_period(..., date, date) resolves ONE day whether it is
            # OBSERVED (inside the BPDB record - sigma=0, no forecast error)
            # or FORECAST (inside the horizon cache), unlike a bare lookup in
            # the forecast-horizon-only cache, which has no row at all for an
            # observed day. A --month/--period run mixes both kinds, so the
            # forecast-horizon-only cache is not sufficient here any more.
            frow = F.build_period(bpdb, i, date, date)[0]
            frows[i] = frow
            regimes[i] = R.operating_regime(                       # (48)
                p_tilde_kw=frow["p_tilde_kw"],
                p_max_kw=p_max_eff[i],
                transfer_out_kwh=float(network[i]["T_out"].sum() * cfg.DELTA_H),
                transfer_in_kwh=float(network[i]["T_in"].sum() * cfg.DELTA_H),
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

        # Issue E hysteresis: only re-pool (recompute cut points from
        # today's chi distribution) when the previous day's re-pooling
        # trigger exceeds ALPHA_REPOOL; otherwise reuse yesterday's pool
        # MEMBERSHIP (by entity_id, stable across days) instead of
        # recomputing it from today's scores, so households are not
        # reshuffled between pools by score noise alone.
        candidate = R.assign_pools(ents, pool_rule)     # always scores; may be discarded
        candidate["date"] = date
        candidate_log.append({"date": date,
                              "W": {p: candidate["pools"][p]["W"] for p in ("PH", "PM", "PL")}})

        # Item 3: compute BOTH re-pooling triggers every day, regardless of
        # which one actually governs the decision (cfg.REPOOL_TRIGGER_KIND),
        # so the two can be compared day-by-day rather than only in theory -
        # alpha_w is a w_e-weighted pool-total move and is structurally
        # blind to same-kind churn that leaves W(P) unchanged; frac_changed
        # catches that churn but does not care whether the entities that
        # moved carry any real weight.
        frac_changed = R.membership_change_fraction(ents, prev_membership)
        if prev_pool is None:
            alpha = None
            trigger_alpha_w, trigger_frac = True, True
        else:
            alpha = R.repooling_trigger(prev_pool, candidate)          # (52)
            trigger_alpha_w = max(alpha.values()) > cfg.ALPHA_REPOOL
            trigger_frac = frac_changed > cfg.ALPHA_REPOOL_FRAC
        active_trigger = (trigger_alpha_w if cfg.REPOOL_TRIGGER_KIND == "alpha_w"
                          else trigger_frac)
        trigger_log.append({"date": date, "frac_changed": frac_changed,
                            "alpha_P_max": max(alpha.values()) if alpha else None,
                            "trigger_alpha_w": trigger_alpha_w,
                            "trigger_frac_changed": trigger_frac,
                            "active_rule": cfg.REPOOL_TRIGGER_KIND,
                            "rules_agree": trigger_alpha_w == trigger_frac})

        if prev_pool is None:
            pool, repooled = candidate, True
        elif active_trigger:
            pool, repooled = candidate, True
        else:
            pool = R.apply_membership(ents, prev_membership)
            pool["date"] = date
            repooled = False
        pool["repooled"] = repooled
        if alpha is not None:
            pool["alpha_P"] = alpha
        pool["frac_changed"] = frac_changed
        prev_pool = pool
        prev_membership = {e.entity_id: e.pool for e in ents}

        topn = R.top_n(ents)                                            # (53), stratified by kind
        pool["top_n"] = [{"entity_id": e.entity_id, "kind": e.kind,
                          "substation": e.substation, "chi": e.chi,
                          "pool": e.pool} for e in topn]

        # Item 2 (and Round-3 follow-up): is the ranking finding entities
        # with real response energy, or just filling N seats? Same-day
        # precision_at_n() is CIRCULAR - phi2 is built from the exact
        # "acting" label being checked against (e_sh_kwh+e_cu_kwh>0), and
        # within one kind/substation/day phi1,phi3,phi4 are identical for
        # every entity, so any entity that acted outranks every one that
        # didn't by construction. Three non-circular variants, computed on
        # a COPY of `ents` so none of them disturb the real e.pool used by
        # `all_entities.csv`/hysteresis below:
        #   - mean_relative pooling's own top-N vs the SAME random baseline
        #     (item 2's original ask: quantile vs mean_relative vs random).
        #   - phi2 dropped from chi entirely (chi_excluding_phi2): what the
        #     OTHER three features alone can rank, with no acting-label leak.
        #   - next-day precision (below, after the loop's first day): does
        #     TODAY's top-N, chosen before tomorrow's response is known,
        #     predict who acts TOMORROW? This is the only one of the four
        #     that cannot leak the label, because the label does not exist
        #     yet when the ranking is made.
        prec = R.precision_at_n(ents, topn)
        prec_rand = R.precision_at_n(ents, R.random_topn(ents))
        ents_mr_scored = copy.deepcopy(ents)
        R.assign_pools(ents_mr_scored, "mean_relative")
        prec_mr = R.precision_at_n(ents, R.top_n(ents_mr_scored))
        chi_nop2 = R.chi_excluding_phi2(ents, regimes)
        prec_nop2 = R.precision_at_n(ents, R.topn_by_chi(ents, chi_nop2))
        prec["date"] = date
        prec["precision_random_draw"] = prec_rand["precision"]
        prec["precision_mean_relative"] = prec_mr["precision"]
        prec["precision_no_phi2"] = prec_nop2["precision"]
        prec["acting_ids"] = json.dumps(sorted(
            e.entity_id for e in ents if (e.e_sh_kwh + e.e_cu_kwh) > 1e-9))
        prec["topn_ids"] = json.dumps(sorted(e.entity_id for e in topn))
        # Item A5: today's ACTORS (not the ranking), sampled down to the same
        # per-kind quota as top_n - the baseline for "does the priority
        # score add anything over just watching whoever is active today."
        persist = R.persistence_topn(ents)
        prec["persistence_ids"] = json.dumps(sorted(e.entity_id for e in persist))
        precision_log.append(prec)

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
              f"spread {pool['chi_spread']:.4f}  rule={pool['rule']}  "
              f"[{'REPOOLED' if pool['repooled'] else 'reused prior membership'}]")
        for p in ("PH", "PM", "PL"):
            q = pool["pools"][p]
            print(f"     {p}: n={q['n']:>5}  W(P)={q['W']:.4f}  "
                  f"chi {q['chi_min']:.4f}-{q['chi_max']:.4f}")
        if pool["degenerate"]:
            print(f"     ! POOL DEGENERATE - an empty pool. Section 9.6 predicts "
                  f"this for the mean_relative rule on tight score "
                  f"distributions (its medium band can be an empty interval "
                  f"by construction - see docs/thesis_changes.md, Issue E); "
                  f"quantile is now the default (--pool-rule mean_relative "
                  f"to reproduce the base-paper rule instead).")
        if "alpha_P" in pool:
            print(f"     alpha_P = " + ", ".join(f"{k} {v:.5f}"
                                                 for k, v in pool["alpha_P"].items())
                  + f"   frac_changed={pool['frac_changed']:.4f}  "
                    f"active_rule={cfg.REPOOL_TRIGGER_KIND}")
        kind_counts = {k: sum(1 for e in topn if e.kind == k) for k in R.ENTITY_KINDS}
        print(f"     top-N ({len(topn)}) kind composition: " +
              ", ".join(f"{k}={v}" for k, v in kind_counts.items()))
        print(f"     precision@N={prec['precision']:.3f}  "
              f"random_draw={prec['precision_random_draw']:.3f}  "
              f"population_base_rate={prec['baseline_random']:.3f}  "
              f"lift={prec['lift']:.2f}x" if prec['n_acting'] else
              f"     precision@N: no acting entities today")
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

    prec_df = pd.DataFrame(precision_log)
    # Round-3 follow-up: next-day precision - day d's top-N (fixed before day
    # d+1's response is known) scored against day d+1's acting set. This is
    # the one precision variant that cannot leak the acting label into the
    # ranking, because the label does not exist yet when day d's ranking is
    # made, unlike same-day precision_at_n().
    # Item A5: the SAME next-day comparison, but for the persistence
    # baseline (today's actual actors, sampled to the same per-kind quota)
    # instead of the ranking's top-N - is the priority score adding
    # anything over just tracking who is already active?
    nextday, nextday_persist = [None] * len(prec_df), [None] * len(prec_df)
    for k in range(len(prec_df) - 1):
        topn_ids = set(json.loads(prec_df.iloc[k]["topn_ids"]))
        persist_ids = set(json.loads(prec_df.iloc[k]["persistence_ids"]))
        acting_next = set(json.loads(prec_df.iloc[k + 1]["acting_ids"]))
        n_total_next = int(prec_df.iloc[k + 1]["n_total"])
        nextday[k] = R.precision_at_n_ids(topn_ids, acting_next, n_total_next)
        nextday_persist[k] = R.precision_at_n_ids(persist_ids, acting_next, n_total_next)
    prec_df["precision_nextday"] = [d["precision"] if d else None for d in nextday]
    prec_df["precision_nextday_random"] = [
        d["baseline_random"] if d else None for d in nextday]
    prec_df["precision_nextday_lift"] = [d["lift"] if d else None for d in nextday]
    prec_df["precision_nextday_persistence"] = [
        d["precision"] if d else None for d in nextday_persist]
    prec_df["precision_nextday_persistence_lift"] = [
        d["lift"] if d else None for d in nextday_persist]
    prec_df.drop(columns=["acting_ids", "topn_ids", "persistence_ids"]).to_csv(
        cfg.OUT_DIR / f"precision_at_n{tag}.csv", index=False)
    trig_df = pd.DataFrame(trigger_log)
    trig_df.to_csv(cfg.OUT_DIR / f"repool_trigger_compare{tag}.csv", index=False)

    print(f"\n  wrote events{tag}.csv ({len(all_events)}), "
          f"entities{tag}.csv ({len(all_entities)}), "
          f"priority_pools{tag}.json, priority_pool_ranked{tag}.csv "
          f"({len(ranked)} rows), precision_at_n{tag}.csv, "
          f"repool_trigger_compare{tag}.csv")

    valid = prec_df[prec_df["n_acting"] > 0]
    if len(valid):
        print(f"\n  Item 2 summary, SAME-DAY (circular - phi2 is built from "
              f"the label being checked; see Round-3 follow-up) over "
              f"{len(valid)} days: quantile={valid['precision'].mean():.3f}  "
              f"mean_relative={valid['precision_mean_relative'].mean():.3f}  "
              f"random={valid['precision_random_draw'].mean():.3f}  "
              f"population_base_rate={valid['baseline_random'].mean():.3f}")
        print(f"  Item 2 summary, NON-CIRCULAR: chi excluding phi2 "
              f"(same-day, no label leak in the SCORE, still same-day) = "
              f"{valid['precision_no_phi2'].mean():.3f}")
        nd_valid = prec_df[prec_df["precision_nextday"].notna()
                          & (prec_df["n_acting"] > 0)]
        if len(nd_valid):
            print(f"  Item 2/A5 summary, NEXT-DAY (no leak possible - the "
                  f"label does not exist when the ranking is made) over "
                  f"{len(nd_valid)} day-pairs:")
            print(f"    ranking (top_n)     precision="
                  f"{nd_valid['precision_nextday'].mean():.3f}  "
                  f"lift={nd_valid['precision_nextday_lift'].mean():.2f}x")
            print(f"    persistence (today's actors, same quota) precision="
                  f"{nd_valid['precision_nextday_persistence'].mean():.3f}  "
                  f"lift={nd_valid['precision_nextday_persistence_lift'].mean():.2f}x")
            print(f"    random draw         precision="
                  f"{nd_valid['precision_nextday_random'].mean():.3f}  "
                  f"(the common baseline for both lifts above)")
    n_days_trig = len(trig_df)
    n_disagree = int((~trig_df["rules_agree"]).sum()) if n_days_trig else 0
    print(f"  Item 3 summary (re-pool trigger comparison over {n_days_trig} days, "
          f"active_rule={cfg.REPOOL_TRIGGER_KIND}): alpha_w repools "
          f"{int(trig_df['trigger_alpha_w'].sum())} days, frac_changed "
          f"(>{cfg.ALPHA_REPOOL_FRAC}) repools "
          f"{int(trig_df['trigger_frac_changed'].sum())} days, "
          f"disagree on {n_disagree}/{n_days_trig} days")

    return all_events, all_pools, all_entities, candidate_log


def alpha_sweep(candidate_log: list, alphas=None) -> pd.DataFrame:
    """
    Issue E: replay the re-pool/reuse decision (Eq. 52's trigger against
    ALPHA_REPOOL) across a grid of thresholds, using only the sequence of
    candidate W(P) values already computed by run() - no re-scoring needed,
    since the hysteresis DECISION is the only thing that varies with alpha.
    """
    alphas = alphas or [0.001, 0.005, 0.01, 0.02, 0.03, 0.05]
    rows = []
    for a in alphas:
        n_repool, prev_w = 0, None
        for rec in candidate_log:
            if prev_w is None:
                n_repool += 1
            else:
                alpha_p = {p: abs(rec["W"][p] - prev_w[p]) for p in prev_w}
                if max(alpha_p.values()) > a:
                    n_repool += 1
            prev_w = rec["W"]
        n = len(candidate_log)
        rows.append({"alpha_repool": a, "n_days": n, "n_repooled": n_repool,
                     "n_reused": n - n_repool,
                     "pct_reused": 100.0 * (n - n_repool) / n if n else 0.0})
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--scarcity", action="store_true")
    ap.add_argument("--tag", default=None,
                    help="LP output tag to read, e.g. _period for a "
                         "06_lp_optimiser.py --month/--period run. Overrides "
                         "--scarcity if both are given.")
    ap.add_argument("--pool-rule", choices=["mean_relative", "quantile"],
                    default=None)
    ap.add_argument("--alpha-sweep", action="store_true",
                    help="Issue E: report the re-pool/reuse split across "
                         "ALPHA_REPOOL in [0.001, 0.05] and exit.")
    ap.add_argument("--substations", nargs="*", default=list(cfg.SUBSTATIONS))
    a = ap.parse_args()
    if not a.run:
        ap.print_help(); return 1
    tag = a.tag if a.tag is not None else ("_scarcity" if a.scarcity else "")
    print("=" * 78)
    print(f"STAGE H - REGIME, EVENTS, PRIORITY POOL{f'  [tag={tag}]' if tag else ''}")
    print("=" * 78)
    _, _, _, candidate_log = run(a.substations, tag, a.pool_rule)
    if a.alpha_sweep:
        print(f"\n{'=' * 78}\nALPHA_REPOOL SWEEP (Issue E)\n{'=' * 78}")
        sweep = alpha_sweep(candidate_log)
        print(sweep.to_string(index=False))
        p = cfg.OUT_DIR / f"alpha_repool_sweep{tag}.csv"
        sweep.to_csv(p, index=False)
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
