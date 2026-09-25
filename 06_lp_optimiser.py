#!/usr/bin/env python3
"""
Stage E/F runner - the linear programme, Eqs. (17)-(35).

    python 06_lp_optimiser.py --solve                 # dispatch horizon
    python 06_lp_optimiser.py --solve --separate      # Remark 1: S={i}, L=empty
    python 06_lp_optimiser.py --solve --days 3
    python 06_lp_optimiser.py --sweep-theta           # section 9.6 open item
    python 06_lp_optimiser.py --plot

Solves day by day on a receding basis, carrying S_i(end of day d) into S^0_i
for day d+1, as the formulation describes. Within a day the solve is a single
simultaneous programme, so Corollary 1 holds.

Theta(t) is EACH DAY'S OWN profile (observed where BPDB has it, day-of-year
climatology beyond it - see gentwin/temperature.py), never the reference
day's. This is the corrected behaviour for Issue D: an earlier version of this
docstring claimed Theta(t) was pinned to the reference day "because kappa_i,d
is a level rescaling of that same reference profile" and warned that using a
different day's temperature would double-count the level change kappa already
carries - which was true only as long as kappa_{i,d} = p~_{i,d} / p_{i,d0}
(Eq. 16 as originally written, denominator = the FIXED reference-day peak).
That is exactly the bug: with per-day Theta already baked into the bottom-up
load Lambda_d(t;0,0) via the thermal response, a hot forecast day gets its
level raised ONCE by Theta_d inside the load model and a SECOND time by
kappa_d against the (cooler) reference day - the same physical driver counted
twice. Fixed by redefining the denominator to that day's OWN generated
baseline peak: kappa_d = p~_{i,d} / max_t Lambda_d(t;0,0;Theta_d), computed
from the SAME per-day population/weather run() already builds, before the
solve. Both the old (reference-day) and new (same-day) kappa are recorded in
lp_summary{tag}.csv as kappa_refday_<sub> / kappa_<sub> for traceability - see
docs/thesis_changes.md, Issue D.
"""

import argparse
import json
import sys

import numpy as np
import pandas as pd

from gentwin import (config as cfg, forecast as F, load_model as lm, lp,
                     peak_window as pw, population as pop, tariff as T,
                     temperature as TM)


def _bpdb():
    for p in (cfg.BPDB_CSV, cfg.DATA_DIR / "BPDB_Dhaka_City_Substations_Page3.csv"):
        if p.exists():
            return pd.read_csv(p, parse_dates=["Date"])
    raise FileNotFoundError(f"BPDB CSV not found at {cfg.BPDB_CSV}")


def _setup(subs):
    cat = pop.load_catalogue()
    sch = pop.all_schedules(cat)
    POP = {i: pop.consumers_from_cache(i) for i in subs}
    rec = {i: F.read_cache(i) for i in subs}
    for i in subs:
        if not rec[i]:
            raise FileNotFoundError(
                f"No kappa table for {i}; run 05c_forecast_to_kappa.py --build")
    return cat, sch, POP, rec


def _min_retail(subs, POP):
    """Eq. (43): the lowest retail tariff among the households at i."""
    out = {}
    for i in subs:
        rates = []
        for h in POP[i]:
            code = cfg.CONSUMER_ARCHETYPES[h.archetype]["tariff_class"]
            rates.append(T.retail_rate(code, 200.0)
                         if T.RETAIL_TARIFF[code].slabs else T.RETAIL_TARIFF[code].flat)
        out[i] = float(min(rates)) if rates else 0.0
    return out


def apply_scenario(scale_gmax: float | None, scale_pmax: float | None):
    """
    SCENARIO KNOB, not a model change.

    In the base case G^max_i = 200 MW against a ~145 MW post-response peak, so
    the import cap never binds, pi_i(t) sits flat at the producer price, and
    Z, nu and the tie-lines all stay at zero. That is a correct result for
    these parameters, but it means the emergency regime r=3 of Eq. (48) can
    never fire and Stage H has nothing to classify.

    Scaling G^max (and optionally P^max) down creates genuine scarcity so the
    reliability terms of Eq. (23), inter-substation transfer, and regimes 2-3
    are exercised. Every artefact records the scale factors, and the base case
    is always reported alongside.
    """
    orig = {}
    for i, spec in cfg.SUBSTATIONS.items():
        orig[i] = (spec["g_max_kw"], spec["p_max_kw"])
        if scale_gmax:
            spec["g_max_kw"] *= scale_gmax
        if scale_pmax:
            spec["p_max_kw"] *= scale_pmax
    return orig


def run(subs, n_days, couple=True, theta_w=None, verbose=True,
        period=None, individual_rationality=None, gamma_scale=None,
        no_dr=False):
    """
    Solve a run of days.

    `period` = (start, end) drives a BILLING PERIOD: each day gets its own
    Theta(t) and its own kappa_{i,d}, and days are marked `observed` or
    `forecast` by forecast.build_period(). Without it, the first `n_days` of
    the dispatch horizon are solved against the reference day's Theta, which
    is the single-day behaviour.

    `gamma_scale` (Round-3 item C14, default None - no change): multiplies
    every household's `gamma_sh_h`/`gamma_cu_h` (discomfort weight) by this
    factor, applied to the CACHED population objects in THIS process only -
    does not touch `gentwin/config.py`'s synthesis defaults or the cache on
    disk, so a fresh `population.consumers_from_cache()` call elsewhere is
    unaffected. `no_dr`: see `gentwin/lp.py::build_day`'s docstring.
    """
    bpdb = _bpdb()
    cat, sch, POP, rec = _setup(subs)
    if gamma_scale is not None:
        for i in subs:
            for h in POP[i]:
                h.gamma_sh_h = float(h.gamma_sh_h) * gamma_scale
                h.gamma_cu_h = float(h.gamma_cu_h) * gamma_scale
    ref_date = rec[subs[0]]["reference_day"]["date"]
    soc = {i: float(cfg.SUBSTATIONS[i]["s0_kwh"]) for i in subs}
    min_ret = _min_retail(subs, POP)

    if period:
        plans = {i: {r["date"]: r for r in
                     F.build_period(bpdb, i, period[0], period[1])}
                 for i in subs}
        dates = sorted(plans[subs[0]])
    else:
        plans = None
        dates = [r["target_date"] for r in rec[subs[0]]["rows"]
                 if r["in_dispatch_horizon"]][:n_days]

    solutions = []
    theta_cache = {}

    for date in dates:
        # Per-day Theta(t). Observed where the dataset has weather, day-of-year
        # climatology beyond it; the profile records which.
        if date not in theta_cache:
            theta_cache[date] = TM.profile_for_date(bpdb, date)
        tp = theta_cache[date]
        theta = np.array(tp["theta_c"])

        # Baseline Lambda_d(t;0,0), computed BEFORE the solve, from THIS day's
        # own population/weather run - Issue D. This is the denominator the
        # corrected kappa_d uses, and it is reused below for sol.baseline_kw
        # so the aggregation is only ever done once per day.
        baseline_kw = {}
        for i in subs:
            lam0 = np.zeros(cfg.SLOTS_PER_DAY)
            for h in POP[i]:
                pr = lm.consumer_profile(h, sch, cat, theta)
                lam0 += float(h.w) * (pr["critical"] + pr["shiftable"]
                                      + pr["curtailable"])
            baseline_kw[i] = lam0

        kappa, kappa_refday, s_flag, mode = {}, {}, {}, {}
        for i in subs:
            row = (plans[i][date] if plans
                   else next(r for r in rec[i]["rows"] if r["target_date"] == date))
            # OLD (Eq. 16 as originally written): p~_d / p_(i,d0), the FIXED
            # reference-day peak. Kept only for traceability (kappa_refday_*
            # in lp_summary{tag}.csv) - see the module docstring on Issue D.
            kappa_refday[i] = row["kappa_scale"]
            # NEW (Issue D fix): p~_d / max_t Lambda_d(t;0,0;Theta_d), that
            # SAME day's own bottom-up baseline peak. Removes the double
            # count: Theta_d's thermal response has already raised
            # Lambda_d(t;0,0) if the day is hot, so kappa_d now measures only
            # the residual gap between the robust FORECAST and what the
            # bottom-up model, with today's weather, already predicts.
            p_tilde = row.get("p_tilde_kw")
            base_peak = float(baseline_kw[i].max())
            kappa[i] = (float(p_tilde) / base_peak) if p_tilde is not None \
                and base_peak > 0 else row["kappa_scale"]
            s_flag[i] = row["s_stress"]
            mode[i] = row.get("mode", "forecast")

        # Issue 10 (optional, off by default): only the LAST date of the run
        # gets a terminal-SoC target, and only if TERMINAL_SOC is enabled.
        terminal_target = ({i: float(cfg.SUBSTATIONS[i]["s0_kwh"]) for i in subs}
                          if cfg.TERMINAL_SOC and date == dates[-1] else None)
        sol = lp.solve_day(date, subs, POP, sch, cat, theta, kappa, s_flag,
                           soc, couple=couple, theta_w=theta_w,
                           individual_rationality=individual_rationality,
                           terminal_target=terminal_target, no_dr=no_dr)
        if not sol.success:
            raise RuntimeError(
                f"{date}: LP solve failed ({sol.status}). Refusing to "
                f"silently truncate the period - fix the infeasibility "
                f"(scarcity scale, storage bounds, reserve floor) and rerun.")

        # Stage G, Eqs. (40)-(43), from the completed solve.
        post = {}
        for i in subs:
            blocks = {}
            lam_post = np.zeros(cfg.SLOTS_PER_DAY)
            tpk = list(pw.get_peak_slots(i))
            for h in POP[i]:
                prof = lm.consumer_profile(h, sch, cat, theta)
                w = float(h.w)
                xh = np.zeros(cfg.SLOTS_PER_DAY)
                yh = np.zeros(cfg.SLOTS_PER_DAY)
                for t in tpk:
                    xh[t] = sol.x.get((h.consumer_id, t), 0.0)
                for t in range(cfg.SLOTS_PER_DAY):
                    yh[t] = sol.y.get((h.consumer_id, t), 0.0)
                lam_post += w * lm.post_response_load(prof, xh, yh, tpk)
            post[i] = lam_post
        sol.stage_g = lp.stage_g(sol, post, min_ret)
        sol.post_response_kw = {i: post[i].tolist() for i in subs}
        sol.day_mode = mode
        sol.theta_source = tp["weather_source"]
        sol.theta_c = tp["theta_c"]
        sol.baseline_kw = {i: baseline_kw[i].tolist() for i in subs}
        sol.kappa_used = dict(kappa)
        sol.kappa_refday = dict(kappa_refday)

        for i in subs:
            soc[i] = sol.soc_end_kwh[i]
        solutions.append(sol)

        if verbose:
            d = sol.diagnostics
            modes = "/".join(sorted(set(mode.values())))
            print(f"\n  {date}  [{modes}, wx={tp['weather_source']}]   "
                  f"F = {sol.objective_tk:,.0f} Tk   "
                  f"({d['n_variables']:,} vars, {d['n_constraints']:,} rows, "
                  f"{'coupled' if d['coupled'] else 'separate'})")
            print(f"     max |C1 slack| {d['max_c1_slack_kw']:.3g} kW   "
                  f"x active {d['n_x_active']}   y active {d['n_y_active']}   "
                  f"C7 binding {d['n_c7_binding']}")
            for i in subs:
                ct, net, pi = sol.cost_terms[i], sol.network[i], sol.duals[i]
                g = sol.stage_g[i]
                print(f"     {i:<11} F_i {sol.F_by_substation[i]:>14,.0f} | "
                      f"cost {ct['f_cost_tk']:>12,.0f} rel {ct['f_rel_tk']:>9,.0f} "
                      f"wel {ct['f_wel_tk']:>13,.0f}")
                print(f"     {'':<11} G_id {ct['G_id_kwh']:>10,.0f} kWh  "
                      f"B_id {ct['B_id_kwh']:>7,.0f}  T_out {ct['T_ijd_kwh']:>8,.0f}  "
                      f"nu {net['nu'].sum():>6,.1f}  "
                      f"Z {sum(net[f'Z_{c}'].sum() for c in lp.CLASSES):>6,.1f}")
                print(f"     {'':<11} pi_i(t) {pi.min():.3f}-{pi.max():.3f} Tk/kWh  "
                      f"kappa {kappa[i]:.4f}  s={s_flag[i]}  "
                      f"SoC {sol.soc_end_kwh[i]:,.0f} kWh  "
                      f"varpi {g['varpi_tk_per_kwh']:.4f} Tk/kWh")
    return solutions, POP


def export(sols, subs, POP, tag="", scenario=None, ir_on=False, gamma_scale=None):
    """
    Everything Stages H and K need.

    `scenario` (Round-3 follow-up): {i: (scale_gmax, scale_pmax, g_max_kw_eff,
    p_max_kw_eff)}, or None for the base case (scale=1.0 for every
    substation). `apply_scenario()` mutates `cfg.SUBSTATIONS` in-place inside
    THIS process only; 07/08/09 run as separate `python` processes and import
    a fresh, unscaled `cfg`, so a scarcity run's effective P^max/G^max was
    silently invisible to every stage downstream of the LP solve itself
    (Stage H's capacity_deficit trigger, in particular, was computed against
    the wrong, unscaled P^max). Recording the scale factors and effective
    caps here, once, per substation, lets every downstream stage read and
    apply them instead of re-deriving them from a cfg that was never scaled
    in their process.
    """
    rows_dec, rows_thr, rows_net = [], [], []
    for sol in sols:
        for (cid, t), val in sol.x.items():
            rows_dec.append({"date": sol.date, "consumer_id": cid, "slot": t,
                             "clock": cfg.slot_to_clock(t), "var": "x", "value": val})
        for (cid, t), val in sol.y.items():
            rows_dec.append({"date": sol.date, "consumer_id": cid, "slot": t,
                             "clock": cfg.slot_to_clock(t), "var": "y", "value": val})
        for r in sol.thresholds:
            rows_thr.append({"date": sol.date, **r,
                             "clock": cfg.slot_to_clock(r["slot"])})
        for i in subs:
            net = sol.network[i]
            for t in range(cfg.SLOTS_PER_DAY):
                rows_net.append({
                    "date": sol.date, "substation": i, "slot": t,
                    "clock": cfg.slot_to_clock(t),
                    "G_kw": net["G"][t], "Bch_kw": net["Bch"][t],
                    "Bdis_kw": net["Bdis"][t], "T_out_kw": net["T_out"][t],
                    "T_in_kw": net["T_in"][t], "nu_kwh": net["nu"][t],
                    "Z_critical_kw": net["Z_critical"][t],
                    "Z_shiftable_kw": net["Z_shiftable"][t],
                    "Z_curtailable_kw": net["Z_curtailable"][t],
                    "SoC_kwh": net["SoC"][t], "pi_tk_per_kwh": sol.duals[i][t],
                    # Round-3 item C12 (social efficiency): C4's own dual,
                    # raw (not kappa/th3-scaled like threshold_table's
                    # `system_tk_per_kwh`) - exactly 0 whenever P^max isn't
                    # scaled down (C4 slack), nonzero only once it binds.
                    "pi4_tk_per_kwh": sol.duals_c4.get(
                        i, np.zeros(cfg.SLOTS_PER_DAY))[t],
                    "c1_slack_kw": sol.c1_slack_kw[i][t],
                    "post_response_kw": sol.post_response_kw[i][t],
                    "baseline_kw": sol.baseline_kw[i][t],
                    "theta_c": sol.theta_c[t],
                    "day_mode": sol.day_mode[i],
                    "weather_source": sol.theta_source,
                })
    for name, rows in (("decisions", rows_dec), ("thresholds", rows_thr),
                       ("network", rows_net)):
        pd.DataFrame(rows).to_csv(cfg.OUT_DIR / f"lp_{name}{tag}.csv", index=False)

    scenario = scenario or {i: (1.0, 1.0, cfg.SUBSTATIONS[i]["g_max_kw"],
                                cfg.SUBSTATIONS[i]["p_max_kw"]) for i in subs}
    summary = [{
        "date": s.date, "F_tk": s.objective_tk,
        **{f"F_{i}": s.F_by_substation[i] for i in subs},
        **{f"{k}_{i}": v for i in subs for k, v in s.cost_terms[i].items()},
        **{f"varpi_{i}": s.stage_g[i]["varpi_tk_per_kwh"] for i in subs},
        **{f"soc_end_{i}": s.soc_end_kwh[i] for i in subs},
        # Issue D: both kappa definitions, so the fix is auditable rather than
        # silently replacing one number with another.
        **{f"kappa_{i}": s.kappa_used[i] for i in subs},
        **{f"kappa_refday_{i}": s.kappa_refday[i] for i in subs},
        # Round-3 follow-up: the scenario scale factors AND the effective
        # (already-scaled) caps, so a downstream process never has to guess
        # whether cfg.SUBSTATIONS in ITS process reflects a scenario that was
        # only ever applied inside the LP-solve process's own memory.
        **{f"scale_gmax_{i}": scenario[i][0] for i in subs},
        **{f"scale_pmax_{i}": scenario[i][1] for i in subs},
        **{f"g_max_kw_eff_{i}": scenario[i][2] for i in subs},
        **{f"p_max_kw_eff_{i}": scenario[i][3] for i in subs},
        # Round-3 follow-up (item A2): same class of bug as the P^max/G^max
        # scenario above, but for rho/IR/curtail_scope - 09_monthly_billing.py
        # runs as its own process and was recomputing incentive_received_tk
        # (and hence saving/net_benefit/gained_*) with the DEFAULT rho
        # regardless of --rho 0, because nothing persisted the override.
        # Confirmed: _period_rho0's own billing_statistics.json reported
        # rho_tk_per_kwh=3.0 (the default) and identical gained-household
        # sets to the base arm, even though the LP itself solved under
        # rho=0. Recorded here so 09 can read and apply the ACTUAL value
        # used for this solve instead of assuming its own process's cfg.
        "rho_used_tk_per_kwh": cfg.RHO_REBATE_TK_PER_KWH,
        "rho_current_policy": cfg.RHO_CURRENT_POLICY,
        "individual_rationality_on": bool(ir_on),
        "curtail_scope_used": cfg.CURTAIL_SCOPE,
        # Round-4 item 2: same propagation gap as rho/IR/curtail_scope above,
        # for --gamma-scale. run() mutates POP's h.gamma_sh_h/h.gamma_cu_h in
        # THIS process's memory only; 09_monthly_billing.py calls
        # pop.consumers_from_cache() fresh in its own process and gets the
        # UNSCALED gamma back, so discomfort_J2_tk/net_benefit_tk/gained_*
        # for every C14 arm were computed against the wrong discomfort
        # weight even though the LP's own x/y decisions (which read gamma
        # directly from the mutated POP inside build_day) were correct.
        "gamma_scale_used": gamma_scale if gamma_scale is not None else 1.0,
        "max_c1_slack_kw": s.diagnostics["max_c1_slack_kw"],
        "n_c7_binding": s.diagnostics["n_c7_binding"],
        "n_ir_binding": s.diagnostics.get("n_ir_binding", 0),
    } for s in sols]
    pd.DataFrame(summary).to_csv(cfg.OUT_DIR / f"lp_summary{tag}.csv", index=False)
    print(f"\n  wrote lp_decisions{tag}.csv, lp_thresholds{tag}.csv, "
          f"lp_network{tag}.csv, lp_summary{tag}.csv")


def cmd_sweep(subs, n_days):
    print("theta sensitivity - formulation section 9.6 open item.\n")
    grid = [(1, 1, 1), (5, 1, 1), (1, 5, 1), (1, 1, 5),
            (1, 1, 0.2), (10, 1, 1)]
    rows = []
    for tw in grid:
        sols, POP = run(subs, n_days, theta_w=tw, verbose=False)
        if not sols:
            continue
        s = sols[0]
        dr = pd.DataFrame(s.thresholds)
        rows.append({
            "theta1": tw[0], "theta2": tw[1], "theta3": tw[2],
            "F_tk": s.objective_tk,
            "n_curtail": int((dr.action == "curtail").sum()) if len(dr) else 0,
            "n_shift": int((dr.action == "shift").sum()) if len(dr) else 0,
            "curtailed_kwh": float(dr.loc[dr.action == "curtail", "energy_kwh"].sum())
            if len(dr) else 0.0,
            "mean_pi": float(np.mean([s.duals[i].mean() for i in subs])),
            "transfer_kwh": float(sum(s.cost_terms[i]["T_ijd_kwh"] for i in subs)),
        })
    df = pd.DataFrame(rows)
    print(df.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
    df.to_csv(cfg.OUT_DIR / "lp_theta_sweep.csv", index=False)
    print("\nRaising theta_3 weights household welfare more heavily, which by "
          "Eq. (37) suppresses the system term kappa*pi/theta_3 and moves the "
          "solution toward the purely private comparison.")


def _draw_dispatch_panel(ax, i, s, net, tpk, x):
    ax.plot(x, net["G"], label="$G_i(t)$ import", color="steelblue", lw=1.6)
    ax.plot(x, net["Bdis"], label="$B^{dis}_i(t)$", color="seagreen", lw=1.3)
    ax.plot(x, -net["Bch"], label="$-B^{ch}_i(t)$", color="darkred", lw=1.3)
    ax.plot(x, net["T_out"], label="$T_{ij}(t)$ export", color="purple", lw=1.2)
    ax.plot(x, s.post_response_kw[i], "--", color="black", lw=1.4,
            label=r"post-response $L_h(t;x,y)$")
    ax.axvspan(min(tpk) - .5, max(tpk) + .5, color="orange", alpha=0.15,
               label="$T^{pk}_i$")
    ax.set_ylabel("kW"); ax.legend(fontsize=7.5); ax.set_xlim(0, 47)
    ax.set_xticks(range(0, 48, 4))
    ax.set_xticklabels([cfg.slot_to_clock(t) for t in range(0, 48, 4)], fontsize=8)
    ax.set_xlabel("time of day")
    ax.set_title(f"{i} - dispatch, {s.date}", fontsize=11)


def _draw_shadow_price_panel(ax, i, net, pi, x):
    ax.plot(x, pi, color="tomato", lw=1.8, label=r"$\pi_i(t)$ dual of C1")
    ax.set_ylabel("Tk/kWh"); ax.legend(fontsize=8); ax.set_xlim(0, 47)
    ax.set_xticks(range(0, 48, 4))
    ax.set_xticklabels([cfg.slot_to_clock(t) for t in range(0, 48, 4)], fontsize=8)
    ax.set_xlabel("time of day")
    ax2 = ax.twinx()
    ax2.plot(x, net["SoC"], color="steelblue", lw=1.2, ls=":", label="SoC")
    ax2.set_ylabel("kWh", color="steelblue")
    ax.set_title(f"{i} - shadow price and storage", fontsize=11)


def _draw_demand_response_panel(ax, i, s, tpk, x):
    thr = pd.DataFrame([t for t in s.thresholds if t["substation"] == i])
    if len(thr):
        cur = thr[thr.action == "curtail"]
        agg = cur.groupby("slot")["energy_kwh"].sum().reindex(x, fill_value=0)
        ax.bar(x, agg.values, color="seagreen", alpha=0.8, label="curtailed kWh")
        sh = thr[thr.action == "shift"]
        if len(sh):
            aggs = sh.groupby("slot")["energy_kwh"].sum().reindex(x, fill_value=0)
            ax.bar(x, aggs.values, color="steelblue", alpha=0.8,
                   bottom=agg.values, label="shifted kWh")
    ax.axvspan(min(tpk) - .5, max(tpk) + .5, color="orange", alpha=0.15)
    ax.set_ylabel("kWh"); ax.legend(fontsize=8); ax.set_xlim(0, 47)
    ax.set_xticks(range(0, 48, 4))
    ax.set_xticklabels([cfg.slot_to_clock(t) for t in range(0, 48, 4)], fontsize=8)
    ax.set_title(f"{i} - demand response by slot", fontsize=11)


def cmd_plot(sols, subs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = sols[0]
    fig, axes = plt.subplots(3, len(subs), figsize=(8 * len(subs), 11), squeeze=False)
    x = np.arange(cfg.SLOTS_PER_DAY)
    for j, i in enumerate(subs):
        net, pi = s.network[i], s.duals[i]
        tpk = sorted(pw.get_peak_slots(i))
        _draw_dispatch_panel(axes[0, j], i, s, net, tpk, x)
        _draw_shadow_price_panel(axes[1, j], i, net, pi, x)
        _draw_demand_response_panel(axes[2, j], i, s, tpk, x)
    fig.suptitle("Stage E/F: LP dispatch, shadow prices and demand response",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p = cfg.OUT_DIR / "lp_dispatch.png"
    fig.savefig(p, dpi=130, bbox_inches="tight")
    print(f"  wrote {p}")

    # True vector PDFs, one per panel, drawn fresh from the same solution.
    for i in subs:
        net, pi = s.network[i], s.duals[i]
        tpk = sorted(pw.get_peak_slots(i))
        safe = i.replace(" ", "_")

        fig_i, ax_i = plt.subplots(figsize=(8, 4.2))
        _draw_dispatch_panel(ax_i, i, s, net, tpk, x)
        fig_i.tight_layout()
        p1 = cfg.PDF_DIR / f"lp_dispatch_{safe}_dispatch.pdf"
        fig_i.savefig(p1, bbox_inches="tight")
        plt.close(fig_i)
        print(f"  wrote {p1}")

        fig_i, ax_i = plt.subplots(figsize=(8, 4.2))
        _draw_shadow_price_panel(ax_i, i, net, pi, x)
        fig_i.tight_layout()
        p2 = cfg.PDF_DIR / f"lp_dispatch_{safe}_shadow_price.pdf"
        fig_i.savefig(p2, bbox_inches="tight")
        plt.close(fig_i)
        print(f"  wrote {p2}")

        fig_i, ax_i = plt.subplots(figsize=(8, 4.2))
        _draw_demand_response_panel(ax_i, i, s, tpk, x)
        fig_i.tight_layout()
        p3 = cfg.PDF_DIR / f"lp_dispatch_{safe}_demand_response.pdf"
        fig_i.savefig(p3, bbox_inches="tight")
        plt.close(fig_i)
        print(f"  wrote {p3}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--solve", action="store_true")
    ap.add_argument("--sweep-theta", action="store_true")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--separate", action="store_true",
                    help="Remark 1: solve each substation alone (S={i}, L=empty)")
    ap.add_argument("--days", type=int, default=cfg.DISPATCH_DAYS)
    ap.add_argument("--period", nargs=2, metavar=("START", "END"),
                    help="billing period, e.g. --period 2026-06-14 2026-07-13")
    ap.add_argument("--month", type=int, default=None, metavar="N",
                    help="shorthand: the latest N-day period ending at the "
                         "forecast horizon (use 30 for a billing month)")
    ap.add_argument("--scale-gmax", type=float, default=None,
                    help="scenario: scale G^max_i to create scarcity "
                         "(e.g. 0.62). Base case leaves it unbound.")
    ap.add_argument("--scale-pmax", type=float, default=None,
                    help="scenario: scale P^max_i, which drives Def and r=3")
    ap.add_argument("--ir", action="store_true",
                    help="Issue I: add the individual-rationality constraint "
                         "(lambda_h*(J1(x,y)-J1(0,0)) + (1-lambda_h)*J2(x,y) "
                         "<= 0 per household per day). Off by default - the "
                         "base formulation does not require it.")
    ap.add_argument("--curtail-scope", choices=["all", "dr_window", "stress_days"],
                    default=None,
                    help="override CURTAIL_SCOPE: restrict y_{h,t} to the "
                         "DR-activation window ('dr_window') or to stressed "
                         "days only ('stress_days'), instead of every slot "
                         "of every day ('all', the default).")
    ap.add_argument("--rho", type=float, default=None,
                    help="Issue J: override RHO_REBATE_TK_PER_KWH. Pass 0 "
                         "for the CURRENT-POLICY arm (no stress rebate - rho "
                         "does not exist in Bangladesh today). Default: the "
                         "config value, i.e. the PROPOSED-mechanism rho.")
    ap.add_argument("--tag-suffix", default=None,
                    help="Override the auto-built --tag entirely with this "
                         "exact string (e.g. '_period_drw_ir_rho6') - needed "
                         "once more than one of --curtail-scope/--rho/--ir is "
                         "combined, since the auto-built tag only names one "
                         "axis at a time. Purely a filename/bookkeeping "
                         "choice - changes no computation.")
    ap.add_argument("--gamma-scale", type=float, default=None,
                    help="Round-3 item C14: multiply every household's "
                         "gamma_sh_h/gamma_cu_h (discomfort weight) by this "
                         "factor for THIS run only - a sensitivity check, "
                         "not a change to the config defaults or synthesis.")
    ap.add_argument("--no-dr", action="store_true",
                    help="Round-3 item C13: force x=y=0 for every household "
                         "(zero the shift/curtail bounds) - the load-shedding "
                         "counterfactual, so whatever the substation cannot "
                         "serve from G/battery/tie-line shows up as Z instead.")
    ap.add_argument("--substations", nargs="*", default=list(cfg.SUBSTATIONS))
    a = ap.parse_args()
    if not (a.solve or a.sweep_theta or a.plot):
        ap.print_help(); return 1

    subs = a.substations
    if a.sweep_theta:
        print("=" * 78); print("THETA SWEEP"); print("=" * 78)
        cmd_sweep(subs, 1)
        return 0

    print("=" * 78)
    print(f"LP SOLVE - {'separate (Remark 1)' if a.separate else 'coupled'}, "
          f"{a.days} dispatch day(s), Delta={cfg.DELTA_H} h, "
          f"{cfg.SLOTS_PER_DAY} slots/day")
    print("=" * 78)
    scenario = None
    if a.scale_gmax or a.scale_pmax:
        apply_scenario(a.scale_gmax, a.scale_pmax)
        scenario = {i: (a.scale_gmax or 1.0, a.scale_pmax or 1.0,
                        cfg.SUBSTATIONS[i]["g_max_kw"],
                        cfg.SUBSTATIONS[i]["p_max_kw"]) for i in subs}
        print(f"SCENARIO: G^max x{a.scale_gmax or 1.0}, P^max x{a.scale_pmax or 1.0} "
              f"- scarcity is imposed, not observed. Base case reported separately.")
    if a.rho is not None:
        cfg.RHO_REBATE_TK_PER_KWH = a.rho
        cfg.RHO_CURRENT_POLICY = (a.rho == 0.0)
        print(f"RHO OVERRIDE: rho = {a.rho} Tk/kWh"
              + ("  [CURRENT-POLICY ARM - no stress rebate exists in "
                 "Bangladesh today; this is what the grid actually offers]"
                 if cfg.RHO_CURRENT_POLICY else
                 "  [still the PROPOSED-mechanism rebate, not existing "
                 "BERC policy]"))
    if a.ir:
        print("INDIVIDUAL RATIONALITY (Issue I): ON - "
              "lambda_h*(J1(x,y)-J1(0,0)) + (1-lambda_h)*J2(x,y) <= 0 "
              "enforced per household per day.")
    if a.curtail_scope is not None:
        cfg.CURTAIL_SCOPE = a.curtail_scope
        print(f"CURTAIL_SCOPE OVERRIDE: y_" + "{h,t}" + f" restricted to "
              f"'{a.curtail_scope}'" +
              (" (curtailment is now PEAK-TIME scoped, not all-day)"
               if a.curtail_scope == "dr_window" else
               " (curtailment is now STRESS-DAY scoped, not all-day)"
               if a.curtail_scope == "stress_days" else ""))
    period = tuple(a.period) if a.period else None
    if a.month:
        period = F.period_bounds(_bpdb(), subs, a.month)
        print(f"Billing period: {period[0]} -> {period[1]} ({a.month} days)")
    if a.gamma_scale is not None:
        print(f"GAMMA-SCALE OVERRIDE (item C14): gamma_sh_h/gamma_cu_h x "
              f"{a.gamma_scale} for this run only.")
    if a.no_dr:
        print("NO-DR COUNTERFACTUAL (item C13): x=y=0 forced for every "
              "household - whatever cannot be served from G/battery/tie-line "
              "shows up as Z (unserved load) instead.")
    sols, POP = run(subs, a.days, couple=not a.separate, period=period,
                    individual_rationality=(True if a.ir else None),
                    gamma_scale=a.gamma_scale, no_dr=bool(a.no_dr))
    if not sols:
        return 3
    # Issue C: every variant writes distinct outputs so 07/08/09 can consume
    # the matching one via --tag.
    if a.tag_suffix is not None:
        tag = a.tag_suffix
    else:
        tag = "_separate" if a.separate else ""
        if a.month or a.period:
            tag += "_period"
        if a.rho == 0.0:
            tag += "_rho0"
        if a.ir:
            tag += "_ir"
        if a.curtail_scope and a.curtail_scope != "all":
            tag += f"_{a.curtail_scope}"
        if a.scale_gmax or a.scale_pmax:
            tag += "_scarcity"
        if a.no_dr:
            tag += "_nodr"
        if a.gamma_scale is not None:
            tag += f"_gamma{a.gamma_scale}"
    export(sols, subs, POP, tag=tag, scenario=scenario, ir_on=bool(a.ir),
          gamma_scale=a.gamma_scale)
    if a.plot:
        cmd_plot(sols, subs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
