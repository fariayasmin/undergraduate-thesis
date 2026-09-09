"""
GenTwin-SG :: monthly billing and savings
=========================================

Answers the thesis question directly: **how much can a consumer save on a
monthly electricity bill under this scheme, and at what comfort cost?**

Why this needs a month, not a day
---------------------------------
Three things in the model are monthly by construction and cannot be recovered
from a single day:

1. **The tariff is a cumulative monthly slab.** LT-A charges 6.18 Tk/kWh on
   the first 75 units, 8.50 on 76-200, 9.10 on 201-300, and so on. Which slab
   a household ends in depends on its WHOLE month. A 30 kWh day tells you
   nothing about whether the household lands at 200 or 450 units, and the
   marginal rate differs by 78% across that range. So savings computed
   day-by-day and multiplied by 30 are simply wrong.

2. **J^h_1 and J^h_2 are defined over T_m** - the slots of billing month m,
   Eqs. (20)-(21). Not over a day.

3. **The surcharge varpi_{i,m} of Eq. (42) is monthly**, computed from a
   completed solve and applied in the following month (Corollary 1).

Two bills, both reported
------------------------
There is a real distinction here and collapsing it would misreport the result:

  **J^h_1, the formulation's bill.** Linear: sum_t L_h(t;x,y) pi_h(t) Delta,
  less the curtailment rebate. `pi^ret_h` is the MARGINAL slab rate, which is
  correct for the LP - a household deciding whether to run its AC faces the
  rate on the next kWh, not its average. This is what drives Eq. (37).

  **The gazette bill.** What the consumer actually pays: cumulative slabs on
  the month's total kWh, plus the demand charge, plus 5% VAT, less the 0.5%
  prepaid rebate. Computed by `tariff.monthly_bill()`.

The two differ, and the difference is itself informative: it is the gap
between the marginal price that drives behaviour and the average price that
appears on the bill. Both are reported per consumer.

Baseline vs optimised
---------------------
The baseline is `L_h(t;0,0)` - the same population, same weather, same days,
with x = y = 0. It is NOT a separate simulation: it is the same profiles with
the decisions zeroed, so every difference is attributable to the LP and
nothing else.

Savings are reported in Tk, in %, and against the discomfort J^h_2 that buying
them cost. A saving quoted without its comfort cost is not a result.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config as cfg
from . import load_model as lm
from . import tariff as T


# =============================================================================
# Per-consumer monthly accumulation
# =============================================================================

def accumulate(consumers, daily_profiles: dict, decisions: dict,
               peak_slots: dict, s_stress_lookup: dict | None = None) -> pd.DataFrame:
    """
    Roll a multi-day solve up to one row per consumer.

    `daily_profiles[date][consumer_id]` -> the UNWEIGHTED class profiles in kW
    (one real consumer, not the represented population).
    `decisions[date][(cid, slot)]` -> {"x": .., "y": ..}
    """
    D = cfg.DELTA_H
    s_stress_lookup = s_stress_lookup or {}
    rows = []
    for h in consumers:
        cid = h.consumer_id
        base_kwh = dr_kwh = 0.0
        e_sh = e_cu = 0.0
        j2 = 0.0
        n_shift_slots = n_curtail_slots = 0
        n_dr_days = 0
        incentive = 0.0
        peak_base = peak_dr = 0.0
        rho = getattr(h, "rho_h", cfg.RHO_REBATE_TK_PER_KWH)

        for date, profs in daily_profiles.items():
            prof = profs[cid]
            tpk = peak_slots[h.substation]
            xh = np.zeros(cfg.SLOTS_PER_DAY)
            yh = np.zeros(cfg.SLOTS_PER_DAY)
            dec = decisions.get(date, {})
            for t in tpk:
                v = dec.get((cid, t, "x"))
                if v:
                    xh[t] = v
                    n_shift_slots += 1
            for t in range(cfg.SLOTS_PER_DAY):
                v = dec.get((cid, t, "y"))
                if v:
                    yh[t] = v
                    n_curtail_slots += 1

            base = prof["critical"] + prof["shiftable"] + prof["curtailable"]
            post = lm.post_response_load(prof, xh, yh, tpk)
            base_kwh += float(base.sum() * D)
            dr_kwh += float(post.sum() * D)
            peak_base = max(peak_base, float(base.max()))
            peak_dr = max(peak_dr, float(post.max()))

            esh_d = float((xh * prof["shiftable"]).sum() * D)      # Eq. (19)
            ecu_d = float((yh * prof["curtailable"]).sum() * D)
            e_sh += esh_d
            e_cu += ecu_d
            j2 += h.gamma_sh_h * esh_d + h.gamma_cu_h * ecu_d      # Eq. (21)
            if esh_d > 1e-9 or ecu_d > 1e-9:
                n_dr_days += 1
            # Eq. (20) rebate term: rho * E^cu * s_{i,d}, paid only on a
            # stressed day. This is the INCENTIVE the consumer receives.
            incentive += rho * ecu_d * float(
                s_stress_lookup.get(date, {}).get(h.substation, 0))

        rows.append({
            "consumer_id": cid, "substation": h.substation,
            "category": h.category, "archetype": h.archetype,
            "w_h": float(h.w), "lambda_h": float(h.lambda_h),
            "gamma_sh_h": float(h.gamma_sh_h), "gamma_cu_h": float(h.gamma_cu_h),
            "y_max_h": float(h.y_max_h),
            "tariff_class": cfg.CONSUMER_ARCHETYPES[h.archetype]["tariff_class"],
            "baseline_kwh": base_kwh, "optimised_kwh": dr_kwh,
            "energy_shifted_kwh": e_sh, "energy_curtailed_kwh": e_cu,
            "n_shift_slots": n_shift_slots, "n_curtail_slots": n_curtail_slots,
            "discomfort_J2_tk": j2,
            "incentive_received_tk": incentive,
            "n_dr_events": n_dr_days,
            "income_band": getattr(h, "income_band", ""),
            "monthly_income_bdt": getattr(h, "monthly_income_bdt", 0.0),
            "household_size": getattr(h, "household_size", 0),
            "peak_baseline_kw": peak_base, "peak_optimised_kw": peak_dr,
            "connected_load_kw": float(h.connected_load_kw),
        })
    return pd.DataFrame(rows)


# =============================================================================
# Bills
# =============================================================================

def j1_linear(prof_by_date: dict, decisions: dict, consumer, peak_slots,
              s_stress: dict, monthly_kwh: float, surcharge: float) -> dict:
    """
    Eq. (20): J^h_1 = sum_t L_h(t;x,y) pi_h(t) Delta - rho sum_t E^cu_{h,t} s.

    Uses the MARGINAL slab rate for pi^ret_h, which is what a household faces
    on the next kWh and therefore what Eq. (37) compares against.
    """
    D = cfg.DELTA_H
    code = cfg.CONSUMER_ARCHETYPES[consumer.archetype]["tariff_class"]
    p = np.asarray(T.mu_profile(code, substation=consumer.substation))
    base_rate = T.retail_rate(code, monthly_kwh)
    price = (base_rate + surcharge) * p
    rho = getattr(consumer, "rho_h", cfg.RHO_REBATE_TK_PER_KWH)

    bill_base = bill_dr = rebate = 0.0
    for date, profs in prof_by_date.items():
        prof = profs[consumer.consumer_id]
        tpk = peak_slots[consumer.substation]
        xh = np.zeros(cfg.SLOTS_PER_DAY)
        yh = np.zeros(cfg.SLOTS_PER_DAY)
        dec = decisions.get(date, {})
        for t in tpk:
            xh[t] = dec.get((consumer.consumer_id, t, "x"), 0.0)
        for t in range(cfg.SLOTS_PER_DAY):
            yh[t] = dec.get((consumer.consumer_id, t, "y"), 0.0)
        base = prof["critical"] + prof["shiftable"] + prof["curtailable"]
        post = lm.post_response_load(prof, xh, yh, tpk)
        bill_base += float((base * price).sum() * D)
        bill_dr += float((post * price).sum() * D)
        rebate += rho * float((yh * prof["curtailable"]).sum() * D) \
            * float(s_stress.get(date, {}).get(consumer.substation, 0))
    return {"J1_baseline_tk": bill_base, "J1_optimised_tk": bill_dr - rebate,
            "rebate_tk": rebate, "marginal_rate_tk_per_kwh": base_rate}


def gazette_bills(df: pd.DataFrame, days: int, surcharge: dict) -> pd.DataFrame:
    """
    The bill the consumer actually receives, under the BERC June-2026 order:
    cumulative slabs on the month's kWh, demand charge, 5% VAT.

    `days` scales a partial period to a nominal 30-day month so the slab
    boundaries are applied to a comparable quantity; the scale factor is
    reported so nothing is hidden.
    """
    scale = 30.0 / days if days else 1.0
    out = []
    for _, r in df.iterrows():
        code = r["tariff_class"]
        kw = max(r["connected_load_kw"], 0.5)
        b_base = T.monthly_bill(code, r["baseline_kwh"] * scale, sanctioned_kw=kw)
        b_dr = T.monthly_bill(code, r["optimised_kwh"] * scale, sanctioned_kw=kw)
        sur = surcharge.get(r["substation"], 0.0)
        sur_tk = sur * r["optimised_kwh"] * scale
        out.append({
            "period_days": days, "scale_to_30d": scale,
            "baseline_kwh_30d": r["baseline_kwh"] * scale,
            "optimised_kwh_30d": r["optimised_kwh"] * scale,
            "bill_baseline_tk": b_base["total_bdt"],
            "bill_optimised_tk": b_dr["total_bdt"] + sur_tk,
            "surcharge_tk": sur_tk,
            "energy_charge_baseline_tk": b_base["energy_charge_bdt"],
            "energy_charge_optimised_tk": b_dr["energy_charge_bdt"],
            "eff_rate_baseline": b_base["effective_rate_bdt_per_kwh"],
            "eff_rate_optimised": b_dr["effective_rate_bdt_per_kwh"],
            "slab_baseline": b_base["breakdown"][-1]["band"] if b_base["breakdown"] else "",
            "slab_optimised": b_dr["breakdown"][-1]["band"] if b_dr["breakdown"] else "",
        })
    return pd.DataFrame(out, index=df.index)


def qoe(df: pd.DataFrame) -> pd.Series:
    """C7's QoE, Eq. (35), realised over the billing period."""
    denom = df["baseline_kwh"].replace(0, np.nan)
    return 1.0 - ((cfg.QOE_WEIGHT_SHIFT * df["energy_shifted_kwh"]
                   + cfg.QOE_WEIGHT_CURTAIL * df["energy_curtailed_kwh"]) / denom)


def build(consumers, daily_profiles, decisions, peak_slots, s_stress,
          surcharge: dict, days: int) -> pd.DataFrame:
    """Per-consumer monthly billing table, both bill definitions."""
    df = accumulate(consumers, daily_profiles, decisions, peak_slots, s_stress)
    by_id = {h.consumer_id: h for h in consumers}

    j1 = [j1_linear(daily_profiles, decisions, by_id[r.consumer_id], peak_slots,
                    s_stress, r.baseline_kwh * (30.0 / days if days else 1.0),
                    surcharge.get(r.substation, 0.0))
          for r in df.itertuples()]
    df = pd.concat([df, pd.DataFrame(j1, index=df.index)], axis=1)
    df = pd.concat([df, gazette_bills(df, days, surcharge)], axis=1)

    df["saving_tk"] = df["bill_baseline_tk"] - df["bill_optimised_tk"]
    df["saving_pct"] = 100.0 * df["saving_tk"] / df["bill_baseline_tk"].replace(0, np.nan)
    df["J1_saving_tk"] = df["J1_baseline_tk"] - df["J1_optimised_tk"]
    df["energy_saving_kwh"] = df["baseline_kwh"] - df["optimised_kwh"]
    df["energy_saving_pct"] = 100.0 * df["energy_saving_kwh"] / df["baseline_kwh"].replace(0, np.nan)
    df["peak_reduction_kw"] = df["peak_baseline_kw"] - df["peak_optimised_kw"]
    df["peak_reduction_pct"] = 100.0 * df["peak_reduction_kw"] / df["peak_baseline_kw"].replace(0, np.nan)
    df["qoe"] = qoe(df)
    df["qoe_ok"] = df["qoe"] >= cfg.QOE_MIN - 1e-9
    # Net benefit: money saved less the Tk-equivalent discomfort it cost.
    # Net benefit = bill saving + incentive received - discomfort cost.
    # The incentive is Eq. (20)'s rho * E^cu * s term, already netted out of
    # J^h_1; it is shown separately here because the brief asks for it and
    # because it is the policy lever that decides whether participation is
    # individually rational.
    df["net_benefit_tk"] = (df["saving_tk"] + df["incentive_received_tk"]
                            - df["discomfort_J2_tk"])
    df["gained"] = df["net_benefit_tk"] > 0
    df["saving_per_discomfort"] = df["saving_tk"] / df["discomfort_J2_tk"].replace(0, np.nan)
    df["participated"] = (df["energy_shifted_kwh"] + df["energy_curtailed_kwh"]) > 1e-9
    return df


def comparison(df: pd.DataFrame) -> pd.DataFrame:
    """
    Scenario A (no demand response) against Scenario B (demand response), per
    household. Scenario A is not a separate simulation: it is the same
    profiles with x = y = 0, so every difference is attributable to the LP.
    """
    return pd.DataFrame({
        "household_id": df["consumer_id"], "substation": df["substation"],
        "category": df["category"], "archetype": df["archetype"],
        "income_band": df["income_band"], "w_h": df["w_h"],
        "A_energy_kwh": df["baseline_kwh_30d"],
        "B_energy_kwh": df["optimised_kwh_30d"],
        "energy_reduction_kwh": df["baseline_kwh_30d"] - df["optimised_kwh_30d"],
        "energy_reduction_pct": df["energy_saving_pct"],
        "A_bill_tk": df["bill_baseline_tk"], "B_bill_tk": df["bill_optimised_tk"],
        "bill_reduction_tk": df["saving_tk"], "bill_reduction_pct": df["saving_pct"],
        "incentive_tk": df["incentive_received_tk"],
        "discomfort_tk": df["discomfort_J2_tk"],
        "net_benefit_tk": df["net_benefit_tk"], "gained": df["gained"],
        "A_qoe": 1.0, "B_qoe": df["qoe"], "qoe_difference": 1.0 - df["qoe"],
        "A_peak_kw": df["peak_baseline_kw"], "B_peak_kw": df["peak_optimised_kw"],
        "peak_reduction_pct": df["peak_reduction_pct"],
        "n_dr_events": df["n_dr_events"],
        "A_slab": df["slab_baseline"], "B_slab": df["slab_optimised"],
    })


def distribution(df: pd.DataFrame) -> dict:
    """
    Requirement 5: the spread, not just the mean. Reported population-weighted
    so it describes connections, not representatives.
    """
    out = {}
    for scope, d in [("fleet", df)] + [(i, g) for i, g in df.groupby("substation")]:
        part = d[d["participated"]]
        gained = d[d["gained"]]
        lost = d[~d["gained"] & d["participated"]]
        v = d["saving_tk"]
        out[scope] = {
            "n_representatives": int(len(d)),
            "n_connections": float(d["w_h"].sum()),
            "n_participating": int(len(part)),
            "connections_participating": float(part["w_h"].sum()),
            "connections_gained_money": float(gained["w_h"].sum()),
            "connections_lost_money": float(lost["w_h"].sum()),
            "pct_connections_gained": 100.0 * float(gained["w_h"].sum())
            / max(float(d["w_h"].sum()), 1e-9),
            "saving_max_tk": float(v.max()), "saving_min_tk": float(v.min()),
            "saving_mean_tk": float(v.mean()), "saving_median_tk": float(v.median()),
            "saving_p25_tk": float(v.quantile(0.25)),
            "saving_p75_tk": float(v.quantile(0.75)),
            "saving_pct_mean": float(d["saving_pct"].mean()),
            "saving_pct_median": float(d["saving_pct"].median()),
            "net_benefit_median_tk": float(d["net_benefit_tk"].median()),
            "net_benefit_p25_tk": float(d["net_benefit_tk"].quantile(0.25)),
            "net_benefit_p75_tk": float(d["net_benefit_tk"].quantile(0.75)),
            "qoe_mean": float(d["qoe"].mean()), "qoe_min": float(d["qoe"].min()),
            "kwh_mean": float(d["baseline_kwh_30d"].mean()),
            "kwh_median": float(d["baseline_kwh_30d"].median()),
        }
    return out


def validate(df: pd.DataFrame, days: int) -> list:
    """
    Requirement 9. Warn, never fail silently.
    """
    out = []

    def _c(name, ok, detail=""):
        out.append({"check": name, "pass": bool(ok), "detail": str(detail)})

    res = df[df["category"] == "Residential"]
    lo, hi = PLAUSIBLE_KWH_PER_MONTH["Residential"]
    mean_kwh = float((res["baseline_kwh_30d"] * res["w_h"]).sum()
                     / max(res["w_h"].sum(), 1e-9))
    _c("residential mean consumption in the published Dhaka range",
       lo <= mean_kwh <= hi,
       f"{mean_kwh:,.0f} kWh/month against {lo}-{hi}")
    frac_in = float(((res["baseline_kwh_30d"] >= 30)
                     & (res["baseline_kwh_30d"] <= 900)).mean())
    _c("90%+ of residential households consume 30-900 kWh/month",
       frac_in >= 0.90, f"{frac_in:.1%} within range")

    _c("no negative energy", bool((df["baseline_kwh_30d"] >= 0).all()
                                 and (df["optimised_kwh_30d"] >= 0).all()))
    _c("optimised energy never exceeds baseline",
       bool((df["optimised_kwh_30d"] <= df["baseline_kwh_30d"] + 1e-6).all()),
       f"{int((df['optimised_kwh_30d'] > df['baseline_kwh_30d'] + 1e-6).sum())} violations")
    _c("bill_before >= bill_after wherever consumption fell",
       bool((df.loc[df["energy_saving_kwh"] > 1e-6, "saving_tk"] >= -1e-6).all()),
       f"{int((df.loc[df['energy_saving_kwh'] > 1e-6, 'saving_tk'] < -1e-6).sum())} violations")
    _c("no saving claimed where nothing was curtailed or shifted",
       bool((df.loc[~df["participated"], "saving_tk"].abs() < 1e-6).all()))
    _c("incentive is zero unless energy was curtailed",
       bool((df.loc[df["energy_curtailed_kwh"] <= 1e-9,
                    "incentive_received_tk"].abs() < 1e-6).all()))
    _c("incentive never exceeds rho * curtailed energy",
       bool((df["incentive_received_tk"]
             <= cfg.RHO_REBATE_TK_PER_KWH * df["energy_curtailed_kwh"] + 1e-6).all()))
    _c("QoE satisfies C7 for every household",
       bool((df["qoe"] >= cfg.QOE_MIN - 1e-6).all()),
       f"{int((df['qoe'] < cfg.QOE_MIN - 1e-6).sum())} violations of "
       f"QoE^min = {cfg.QOE_MIN}")
    # Slab monotonicity is tested on the ENERGY charge, not the all-in
    # effective rate. The effective rate includes the fixed demand charge, so
    # a household that consumes less legitimately shows a HIGHER Tk/kWh - the
    # fixed part is spread over fewer units. Testing the all-in rate would
    # flag correct behaviour as a fault.
    moved = df["energy_saving_kwh"] > 1e-6
    _c("slab assignment monotone: lower consumption never raises the energy charge",
       bool((df.loc[moved, "energy_charge_optimised_tk"]
             <= df.loc[moved, "energy_charge_baseline_tk"] + 1e-6).all()),
       f"{int((df.loc[moved, 'energy_charge_optimised_tk'] > df.loc[moved, 'energy_charge_baseline_tk'] + 1e-6).sum())} violations "
       f"(the all-in effective rate can rise, correctly, because the demand "
       f"charge is fixed)")
    _c("net benefit reconciles: saving + incentive - discomfort",
       bool((abs(df["net_benefit_tk"] - (df["saving_tk"]
                                         + df["incentive_received_tk"]
                                         - df["discomfort_J2_tk"])) < 1e-6).all()))
    _c("billing period is a full month",
       days >= 28, f"{days} days")
    return out


# =============================================================================
# Plausibility checks on the billing result
# =============================================================================

#: Published ranges for Dhaka. A synthetic population that lands outside these
#: puts consumers in the WRONG TARIFF SLAB, which distorts every bill figure -
#: LT-A runs from 6.18 Tk/kWh at 75 units to 17.35 above 600.
PLAUSIBLE_KWH_PER_MONTH = {
    "Residential": (100, 450),
    "Commercial": (300, 8000),
    "Industrial": (1000, 50000),
    "Hospital": (5000, 200000),
    "Educational": (500, 30000),
    "Government": (500, 20000),
}


def plausibility(df: pd.DataFrame, days: int) -> list:
    """
    Does the synthetic population consume like real Dhaka connections?

    This is the check that decides whether the BILL figures mean anything. The
    aggregate energy and peak results are pinned to the BPDB peak and are sound
    regardless; the bills are not, because the tariff slab a consumer lands in
    depends on its monthly kWh, and the slab rate varies by 180% across the
    range. A population of too-few, too-large consumers reports the right total
    energy at the wrong price.
    """
    out = []
    for (i, cat_), g in df.groupby(["substation", "category"]):
        w = g["w_h"]
        kwh = float((g["baseline_kwh_30d"] * w).sum() / w.sum())
        lo, hi = PLAUSIBLE_KWH_PER_MONTH.get(cat_, (0, 1e9))
        out.append({
            "substation": i, "category": cat_,
            "kwh_per_connection_month": kwh,
            "plausible_low": lo, "plausible_high": hi,
            "within_range": bool(lo <= kwh <= hi),
            "factor_off": (kwh / hi if kwh > hi else
                           (kwh / lo if kwh < lo else 1.0)),
            "connections": float(w.sum()),
            "mean_bill_tk": float((g["bill_baseline_tk"] * w).sum() / w.sum()),
        })
    return out


# =============================================================================
# Substation and fleet summaries
# =============================================================================

def summarise(df: pd.DataFrame, lp_summary: pd.DataFrame, subs, days: int) -> pd.DataFrame:
    """
    Substation-level rollup, population-weighted by w_h.

    Utility margin is reported as retail revenue less the modelled operator
    cost f^cost_i. It is INDICATIVE only: no transmission, distribution,
    metering or overhead cost is modelled, so it is not a P&L.
    """
    rows = []
    for i in subs:
        d = df[df["substation"] == i]
        w = d["w_h"]
        base_kwh = float((d["baseline_kwh"] * w).sum())
        opt_kwh = float((d["optimised_kwh"] * w).sum())
        bill_base = float((d["bill_baseline_tk"] * w).sum())
        bill_opt = float((d["bill_optimised_tk"] * w).sum())
        f_cost = float(lp_summary.get(f"f_cost_tk_{i}", pd.Series([0.0])).sum())
        part = d[d["participated"]]
        rows.append({
            "substation": i, "period_days": days,
            "n_representatives": len(d),
            "n_connections": float(w.sum()),
            "n_participating_connections": float(part["w_h"].sum()),
            "participation_rate_pct": 100.0 * float(part["w_h"].sum()) / float(w.sum()),
            "baseline_energy_kwh": base_kwh, "optimised_energy_kwh": opt_kwh,
            "energy_saved_kwh": base_kwh - opt_kwh,
            "energy_saved_pct": 100.0 * (base_kwh - opt_kwh) / base_kwh,
            "energy_shifted_kwh": float((d["energy_shifted_kwh"] * w).sum()),
            "energy_curtailed_kwh": float((d["energy_curtailed_kwh"] * w).sum()),
            "bill_baseline_tk": bill_base, "bill_optimised_tk": bill_opt,
            "consumer_saving_tk": bill_base - bill_opt,
            "consumer_saving_pct": 100.0 * (bill_base - bill_opt) / bill_base,
            "mean_saving_per_connection_tk": (bill_base - bill_opt) / float(w.sum()),
            "median_household_saving_tk": float(d["saving_tk"].median()),
            "mean_household_bill_tk": float((d["bill_baseline_tk"] * w).sum() / w.sum()),
            "mean_household_kwh": float((d["baseline_kwh_30d"] * w).sum() / w.sum()),
            "total_incentive_paid_tk": float((d["incentive_received_tk"] * w).sum()),
            "n_dr_participants": int(part.shape[0]),
            "connections_gained_money": float(d.loc[d["gained"], "w_h"].sum()),
            "discomfort_tk": float((d["discomfort_J2_tk"] * w).sum()),
            "net_benefit_tk": float((d["net_benefit_tk"] * w).sum()),
            "peak_reduction_pct_mean": float(d["peak_reduction_pct"].mean()),
            "qoe_mean": float(d["qoe"].mean()),
            "qoe_min": float(d["qoe"].min()),
            "qoe_violations": int((~d["qoe_ok"]).sum()),
            "operator_cost_tk": f_cost,
            "retail_revenue_tk": bill_opt,
            "utility_margin_tk_indicative": bill_opt - f_cost,
        })
    return pd.DataFrame(rows)


def by_category(df: pd.DataFrame) -> pd.DataFrame:
    """Who actually saves, and who pays for it in comfort."""
    g = df.groupby(["substation", "category"])
    out = g.apply(lambda d: pd.Series({
        "n_reps": len(d),
        "connections": float(d["w_h"].sum()),
        "participation_pct": 100.0 * float(d.loc[d["participated"], "w_h"].sum())
        / max(float(d["w_h"].sum()), 1e-9),
        "bill_baseline_tk": float((d["bill_baseline_tk"] * d["w_h"]).sum()),
        "saving_tk": float((d["saving_tk"] * d["w_h"]).sum()),
        "saving_pct": 100.0 * float((d["saving_tk"] * d["w_h"]).sum())
        / max(float((d["bill_baseline_tk"] * d["w_h"]).sum()), 1e-9),
        "mean_saving_per_connection_tk": float((d["saving_tk"] * d["w_h"]).sum())
        / max(float(d["w_h"].sum()), 1e-9),
        "discomfort_tk": float((d["discomfort_J2_tk"] * d["w_h"]).sum()),
        "qoe_mean": float(d["qoe"].mean()),
        "curtailed_kwh": float((d["energy_curtailed_kwh"] * d["w_h"]).sum()),
    }), include_groups=False)
    return out.reset_index()
