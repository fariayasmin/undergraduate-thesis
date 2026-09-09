"""
GenTwin-SG :: Stage K - knowledge graph
=======================================

    K = (V, E, lab, rho),   E subset of V x V          Eq. (57)

Builds the graph from the completed Stage E/F and Stage H outputs.

Design principle, from the brief
--------------------------------
The graph must preserve the CAUSAL CHAIN, not just the final outcome:

    Forecast -> Optimisation -> Decision variables -> Regime -> Events
             -> Priority Pool -> Affected entities

and it must be answerable WITHOUT re-running the LP. Two consequences drive
the whole schema:

1. **The explanation is stored, not recomputed.** Every DrAction node carries
   the Proposition 2 decomposition - `private_tk_per_kwh`, `system_tk_per_kwh`,
   `qoe_tk_per_kwh`, `discomfort_tk_per_kwh`, `margin`, `lambda_h`,
   `lambda_star`, `pi_i_t`, `kappa` - so "why did this consumer curtail at
   18:30" is three numbers on one node. Without this the later LLM layer would
   have to re-derive the threshold, which is exactly what the formulation's
   explainability claim is meant to avoid.

2. **Forecast and Regime are separate nodes.** s_{i,d} is computed from the
   forecast alone and enters the LP objective (Eq. 20), so it lives on
   Forecast. r_{i,d} reads T_ij and nu_i, which are LP outputs (Eq. 48), so it
   lives on a Regime node created after the solve. Collapsing them would
   destroy the ordering the formulation insists on.

Slot-level granularity
----------------------
TimeSlot is a first-class node, 48 per day, so Cypher can answer "which
consumers curtailed at 18:30" and "what was the state of charge at 17:00"
directly. Network decisions are stored DENSELY (including zeros) because there
are only 8 x 48 x n_days x n_substations of them and absence has to be
queryable. Household DR actions are stored SPARSELY - a node only where
x* > 0 or y* > 0 - because dense storage would be ~800 x 48 x 2 per day.

To keep absence queryable for consumers too, every consumer gets a per-day
ParticipationSummary carrying totals and a `never_curtailed` flag. So both
"who acted at 18:30" and "which hospital loads were never curtailed" are
single-hop queries.

Backward compatibility
----------------------
The earlier GraphRAG query pack used `:User`, `:DrEvent` and `gcs_score`. Those
labels are kept as ALIASES (`:Household:User`, `:Event:DrEvent`) and `chi` is
mirrored to `gcs_score`, so the existing eleven queries still run against the
new graph while the formulation-faithful names are the primary ones.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as cfg
from . import peak_window as pw
from . import tariff as T

SCHEMA_VERSION = 2

NODE_LABELS = [
    "Substation", "TieLine", "Meter", "Household", "Appliance", "ApplianceClass",
    "Archetype", "TariffClass", "TouWindow", "TimeSlot", "Day",
    "LoadProfile", "Forecast", "OptimizationRun", "Decision", "DrAction",
    "Regime", "Event", "PriorityPool", "PoolEntity", "ParticipationSummary",
    "Surcharge", "MonthlyBill", "BillingPeriod",
]

REL_TYPES = [
    "HAS_HOUSEHOLD", "LOCATED_AT", "HAS_METER", "FEEDS_FROM", "REGISTERS",
    "BELONGS_TO", "IN_ARCHETYPE", "SUBJECT_TO", "HAS_TOU_WINDOW",
    "PREDICTS", "AT_DAY", "AT_TIMESLOT", "SOLVED_FOR", "PRODUCED",
    "APPLIES_TO", "PERFORMED", "REDUCES", "COMPUTED_FROM", "UNDER_REGIME",
    "TRIGGERED_BY", "AFFECTS", "CAUSES", "CONTAINS", "SCORED_AS",
    "SUMMARISES", "CONNECTS", "HAS_PROFILE", "PASSED_THROUGH",
    "BILLED", "COVERS", "BILLED_UNDER", "HAS_EVENT", "HAS_APPLIANCE",
    "PARTICIPATED_IN",
]


class GraphBuilder:
    def __init__(self):
        self.nodes: dict[str, list] = {}
        self.rels: list = []

    def node(self, label: str, key: str, **props):
        props = {k: v for k, v in props.items() if v is not None}
        props["id"] = key
        self.nodes.setdefault(label, []).append(props)
        return key

    def rel(self, src_label, src, rtype, dst_label, dst, **props):
        self.rels.append({"src_label": src_label, "src": src, "type": rtype,
                          "dst_label": dst_label, "dst": dst,
                          **{k: v for k, v in props.items() if v is not None}})

    def counts(self):
        return ({k: len(v) for k, v in self.nodes.items()},
                pd.Series([r["type"] for r in self.rels]).value_counts().to_dict()
                if self.rels else {})


# =============================================================================
# Static layer: network, consumers, appliances, tariff, time
# =============================================================================

def build_static(g: GraphBuilder, subs, populations, dates,
                 forecast_dates=None):
    for i in subs:
        spec = cfg.SUBSTATIONS[i]
        tpk = sorted(pw.get_peak_slots(i))
        g.node("Substation", i, name=i, code=spec["code"],
               latitude=spec["latitude"], longitude=spec["longitude"],
               coords_provisional=spec["coords_provisional"],
               feeder_voltage_kv=spec["feeder_voltage_kv"],
               upstream_kv=spec["upstream_kv"],
               p_max_kw=spec["p_max_kw"], g_max_kw=spec["g_max_kw"],
               e_max_kwh=spec["e_max_kwh"], b_max_kw=spec["b_max_kw"],
               reserve_kwh=spec["reserve_kwh"], s0_kwh=spec["s0_kwh"],
               c_producer_tk_per_kwh=spec["c_producer_tk_per_kwh"],
               t_pk_slots=tpk, t_pk_label=spec["t_pk_label"],
               eta_ch=cfg.ETA_CH, eta_dis=cfg.ETA_DIS)
        # Adaptive local ToU window, with both halves' provenance
        g.node("TouWindow", f"TOU-{i}", substation=i, slots=tpk,
               window_label=spec["t_pk_label"], mu_peak=cfg.MU_PEAK,
               mu_off=cfg.MU_OFF,
               mu_source="BERC order 03 June 2026 (LT-class ratio)",
               window_source="GenTwin-SG adaptive local derivation",
               is_replacement_tariff=False)
        g.rel("Substation", i, "HAS_TOU_WINDOW", "TouWindow", f"TOU-{i}")

    for l in cfg.TIE_LINES:
        if l["from"] in subs and l["to"] in subs:
            key = f"TL-{l['from']}-{l['to']}"
            g.node("TieLine", key, **{k: l[k] for k in
                                      ("from", "to", "t_max_kw",
                                       "loss_fraction", "price_tk_per_kwh")})
            g.rel("Substation", l["from"], "CONNECTS", "TieLine", key)
            g.rel("TieLine", key, "CONNECTS", "Substation", l["to"])

    for c in ("critical", "shiftable", "curtailable"):
        g.node("ApplianceClass", c, name=c,
               voll_tk_per_kwh=cfg.VOLL_TK_PER_KWH[c],
               explanation_template={
                   "critical": "Cannot be interrupted without harm or spoilage.",
                   "shiftable": "The task must happen, but its timing is free.",
                   "curtailable": "Can be reduced in intensity at a comfort cost.",
               }[c])

    for a, spec in cfg.CONSUMER_ARCHETYPES.items():
        g.node("Archetype", a, name=a, category=spec["category"],
               tariff_class=spec["tariff_class"],
               description=spec["description"])
        g.rel("Archetype", a, "SUBJECT_TO", "TariffClass", spec["tariff_class"])

    for code in {s["tariff_class"] for s in cfg.CONSUMER_ARCHETYPES.values()}:
        tc = T.RETAIL_TARIFF[code]
        g.node("TariffClass", code, code=code, name_en=tc.name_en,
               voltage=tc.voltage, flat_tk_per_kwh=tc.flat,
               off_peak=tc.off_peak, peak=tc.peak,
               demand_charge=tc.demand_charge, has_tou=tc.has_tou,
               source=T.GAZETTE_REF)

    # Day nodes cover the whole FORECAST horizon, not just the dispatch days,
    # because look-ahead Forecast nodes attach to them (the formulation's
    # "classification and look-ahead rather than dispatch"). TimeSlot nodes are
    # created only for dispatch days, where there are decisions to attach.
    for d in sorted(set(dates) | set(forecast_dates or [])):
        g.node("Day", d, date=d, is_dispatch_day=bool(d in set(dates)))
    for d in dates:
        for s in range(cfg.SLOTS_PER_DAY):
            key = f"{d}-S{s:02d}"
            g.node("TimeSlot", key, date=d, slot_index=s,
                   clock=cfg.slot_to_clock(s), delta_h=cfg.DELTA_H)
            g.rel("TimeSlot", key, "AT_DAY", "Day", d)

    for i in subs:
        for h in populations[i]:
            hid = h.consumer_id
            g.node("Household", hid, household_id=hid, substation=i,
                   category=h.category, archetype=h.archetype,
                   w_h=float(h.w),
                   represents_population=bool(h.w > 1.5),
                   lambda_h=float(h.lambda_h),
                   gamma_sh_h=float(h.gamma_sh_h),
                   gamma_cu_h=float(h.gamma_cu_h),
                   y_max_h=float(h.y_max_h),
                   connected_load_kw=float(h.connected_load_kw),
                   n_appliances=len(h.appliances),
                   tariff_class=cfg.CONSUMER_ARCHETYPES[h.archetype]["tariff_class"])
            g.rel("Substation", i, "HAS_HOUSEHOLD", "Household", hid)
            g.rel("Household", hid, "LOCATED_AT", "Substation", i)
            g.rel("Household", hid, "IN_ARCHETYPE", "Archetype", h.archetype)

            mid = f"{hid}-M"
            g.node("Meter", mid, meter_id=mid, household_id=hid, substation=i,
                   n_appliances=len(h.appliances))
            g.rel("Household", hid, "HAS_METER", "Meter", mid)
            g.rel("Meter", mid, "FEEDS_FROM", "Substation", i)

            for k, a in enumerate(h.appliances):
                aid = f"{hid}-A{k:02d}"
                g.node("Appliance", aid, appliance_id=aid, household_id=hid,
                       name=a.name, cls=a.cls, quantity=a.n,
                       p_rat_kw=a.p_rat_kw, duty_cycle=a.duty_cycle,
                       schedule=a.schedule, thermal_response=a.thermal)
                g.rel("Meter", mid, "REGISTERS", "Appliance", aid)
                # Direct edge as well as the meter path: the brief's traversal
                # is (Household)-[:HAS_APPLIANCE]->(Appliance), and forcing
                # every appliance query through the meter adds a hop for no
                # information.
                g.rel("Household", hid, "HAS_APPLIANCE", "Appliance", aid)
                g.rel("Appliance", aid, "BELONGS_TO", "ApplianceClass", a.cls)


# =============================================================================
# Dynamic layer: forecast, run, decisions, regime, events, pools
# =============================================================================

def build_forecast(g, subs, krec):
    """Forecast nodes carry Eqs. (13)-(16). NO regime - that is Eq. (48)."""
    for i in subs:
        for r in krec[i]["rows"]:
            key = f"FC-{cfg.SUBSTATIONS[i]['code']}-{r['target_date']}"
            g.node("Forecast", key, substation=i, date=r["target_date"],
                   horizon_k=r["horizon_k"],
                   in_dispatch_horizon=r["in_dispatch_horizon"],
                   p_hat_kw=r["p_hat_kw"], sigma_kw=r["sigma_kw"],
                   z_beta=r["z_beta"], p_tilde_kw=r["p_tilde_kw"],
                   uncertainty_margin_kw=r["uncertainty_margin_kw"],
                   p_str_kw=r["p_str_kw"], s_stress=r["s_stress"],
                   deficit_kw=r["deficit_kw"], p_max_kw=r["p_max_kw"],
                   kappa=r["kappa_scale"], p_ref_kw=r["p_ref_kw"],
                   reference_day=krec[i]["reference_day"]["date"],
                   weather_source=r["weather_source"],
                   holiday_type=r["holiday_type"],
                   sigma_method=krec[i]["sigma_method"],
                   model="BiGRU (frozen)")
            g.rel("Forecast", key, "PREDICTS", "Substation", i)
            g.rel("Forecast", key, "AT_DAY", "Day", r["target_date"])


def build_dynamic(g, subs, net_df, thr_df, summ, events, entities, pools):
    """OptimizationRun -> Decision -> Regime -> Event -> PriorityPool."""
    for date in sorted(net_df["date"].unique()):
        nd = net_df[net_df["date"] == date]
        srow = summ[summ["date"] == date].iloc[0] if len(summ) else None
        run_key = f"RUN-{date}"
        g.node("OptimizationRun", run_key, date=date,
               objective_F_tk=float(srow["F_tk"]) if srow is not None else None,
               theta1=cfg.THETA_COST, theta2=cfg.THETA_RELIABILITY,
               theta3=cfg.THETA_WELFARE, delta_h=cfg.DELTA_H,
               slots_per_day=cfg.SLOTS_PER_DAY, solver=cfg.LP_SOLVER,
               qoe_min=cfg.QOE_MIN,
               max_c1_slack_kw=float(srow["max_c1_slack_kw"]) if srow is not None else None,
               n_c7_binding=int(srow["n_c7_binding"]) if srow is not None else None,
               formulation="GenTwinSG revised, Eqs. (17)-(35)")
        g.rel("OptimizationRun", run_key, "AT_DAY", "Day", date)

        for i in subs:
            g.rel("OptimizationRun", run_key, "SOLVED_FOR", "Substation", i)
            gi = nd[nd["substation"] == i].sort_values("slot")
            if srow is not None and f"f_cost_tk_{i}" in summ.columns:
                g.node("Surcharge", f"SUR-{i}-{date}", substation=i, date=date,
                       varpi_tk_per_kwh=float(srow.get(f"varpi_{i}", 0.0)),
                       f_cost_tk=float(srow.get(f"f_cost_tk_{i}", 0.0)),
                       f_rel_tk=float(srow.get(f"f_rel_tk_{i}", 0.0)),
                       f_wel_tk=float(srow.get(f"f_wel_tk_{i}", 0.0)),
                       G_id_kwh=float(srow.get(f"G_id_kwh_{i}", 0.0)),
                       B_id_kwh=float(srow.get(f"B_id_kwh_{i}", 0.0)),
                       T_ijd_kwh=float(srow.get(f"T_ijd_kwh_{i}", 0.0)),
                       F_i_tk=float(srow.get(f"F_{i}", 0.0)))
                g.rel("Substation", i, "PASSED_THROUGH", "Surcharge",
                      f"SUR-{i}-{date}")

            # Dense network decisions, Eq. (17) - zeros included on purpose
            for _, row in gi.iterrows():
                s = int(row["slot"])
                slot_key = f"{date}-S{s:02d}"
                for var, col, unit in (
                        ("G", "G_kw", "kW"), ("B_ch", "Bch_kw", "kW"),
                        ("B_dis", "Bdis_kw", "kW"), ("T_out", "T_out_kw", "kW"),
                        ("T_in", "T_in_kw", "kW"), ("nu", "nu_kwh", "kWh"),
                        ("Z_critical", "Z_critical_kw", "kW"),
                        ("Z_shiftable", "Z_shiftable_kw", "kW"),
                        ("Z_curtailable", "Z_curtailable_kw", "kW")):
                    dk = f"DEC-{cfg.SUBSTATIONS[i]['code']}-{date}-{s:02d}-{var}"
                    g.node("Decision", dk, var_type=var, value=float(row[col]),
                           unit=unit, substation=i, date=date, slot_index=s,
                           clock=cfg.slot_to_clock(s),
                           equation="(17)")
                    g.rel("OptimizationRun", run_key, "PRODUCED", "Decision", dk)
                    g.rel("Decision", dk, "AT_TIMESLOT", "TimeSlot", slot_key)
                    g.rel("Decision", dk, "APPLIES_TO", "Substation", i)
                # slot state: SoC and the shadow price
                lk = f"LP-{cfg.SUBSTATIONS[i]['code']}-{date}-{s:02d}"
                g.node("LoadProfile", lk, substation=i, date=date, slot_index=s,
                       clock=cfg.slot_to_clock(s),
                       post_response_kw=float(row["post_response_kw"]),
                       soc_kwh=float(row["SoC_kwh"]),
                       pi_tk_per_kwh=float(row["pi_tk_per_kwh"]),
                       c1_slack_kw=float(row["c1_slack_kw"]))
                g.rel("Substation", i, "HAS_PROFILE", "LoadProfile", lk)
                g.rel("LoadProfile", lk, "AT_TIMESLOT", "TimeSlot", slot_key)

    # Sparse household DR actions, with the Proposition 2 decomposition
    for _, r in thr_df.iterrows():
        date, s = r["date"], int(r["slot"])
        aid = (f"DR-{r['consumer']}-{date}-{s:02d}-"
               f"{'X' if r['action'] == 'shift' else 'Y'}")
        g.node("DrAction", aid, consumer_id=r["consumer"],
               substation=r["substation"], date=date, slot_index=s,
               clock=cfg.slot_to_clock(s), action=r["action"],
               value=float(r["value"]), energy_kwh=float(r["energy_kwh"]),
               # --- the stored explanation, Eqs. (36)/(37) ---
               private_tk_per_kwh=float(r["private_tk_per_kwh"]),
               system_tk_per_kwh=float(r["system_tk_per_kwh"]),
               qoe_tk_per_kwh=float(r.get("qoe_tk_per_kwh", 0.0)),
               discomfort_tk_per_kwh=float(r["discomfort_tk_per_kwh"]),
               margin=float(r["margin"]),
               c7_binding=bool(r.get("c7_binding", False)),
               pi_i_t=float(r["pi_i_t"]), kappa=float(r["kappa"]),
               lambda_h=float(r["lambda_h"]) if pd.notna(r.get("lambda_h")) else None,
               lambda_star=float(r["lambda_star"]) if pd.notna(r.get("lambda_star")) else None,
               y_max_h=float(r["y_max_h"]) if pd.notna(r.get("y_max_h")) else None,
               at_cap=bool(r["at_cap"]) if pd.notna(r.get("at_cap")) else None,
               threshold_form=("private + system - qoe > discomfort"
                               if r.get("c7_binding") else
                               "private + system > discomfort"),
               equation="(36)" if r["action"] == "shift" else "(37)")
        g.rel("Household", r["consumer"], "PERFORMED", "DrAction", aid)
        g.rel("DrAction", aid, "AT_TIMESLOT", "TimeSlot", f"{date}-S{s:02d}")
        g.rel("DrAction", aid, "REDUCES", "LoadProfile",
              f"LP-{cfg.SUBSTATIONS[r['substation']]['code']}-{date}-{s:02d}")

    # Regime nodes - computed AFTER the solve, Eq. (48)
    ev = pd.DataFrame(events)
    for (date, i), grp in ev.groupby(["date", "substation"]):
        row = grp.iloc[0]
        rk = f"REG-{cfg.SUBSTATIONS[i]['code']}-{date}"
        trig = row["regime_triggers"]
        if isinstance(trig, str):
            trig = json.loads(trig.replace("'", '"').replace("True", "true")
                              .replace("False", "false"))
        g.node("Regime", rk, substation=i, date=date, r=int(row["regime"]),
               regime_name=row["regime_name"],
               s_stress=int(row["stress_indicator"]),
               trigger_capacity_deficit=bool(trig.get("capacity_deficit")),
               trigger_transfer=bool(trig.get("inter_substation_transfer")),
               trigger_reserve_shortfall=bool(trig.get("reserve_shortfall")),
               beta1=row["beta_weights"][0] if isinstance(row["beta_weights"], list) else None,
               equation="(48)",
               computed_after_solve=True)
        g.rel("Regime", rk, "COMPUTED_FROM", "OptimizationRun", f"RUN-{date}")
        g.rel("Regime", rk, "COMPUTED_FROM", "Forecast",
              f"FC-{cfg.SUBSTATIONS[i]['code']}-{date}")
        g.rel("Regime", rk, "APPLIES_TO", "Substation", i)

    # Events
    for e in events:
        ek = e["event_id"]
        g.node("Event", ek, **{k: v for k, v in e.items()
                               if k not in ("regime_triggers", "beta_weights")})
        i, date = e["substation"], e["date"]
        # The brief's diagram is (Substation)-[:HAS_EVENT]->(Event) and
        # (Event)-[:AFFECTS]->(Household); AFFECTS is reserved for the latter.
        g.rel("Substation", i, "HAS_EVENT", "Event", ek)
        g.rel("Event", ek, "TRIGGERED_BY", "Forecast",
              f"FC-{cfg.SUBSTATIONS[i]['code']}-{date}")
        g.rel("Event", ek, "UNDER_REGIME", "Regime",
              f"REG-{cfg.SUBSTATIONS[i]['code']}-{date}")
        g.rel("Event", ek, "COMPUTED_FROM", "OptimizationRun", f"RUN-{date}")

        # (Event)-[:AFFECTS]->(Household) and (Event)-[:CAUSES]->(DrAction),
        # both from the brief. Without them "which customers did this event
        # affect" has to be reconstructed from dates, which is exactly the
        # kind of implicit join a knowledge graph exists to remove.
        acted = thr_df[(thr_df["date"] == date)
                       & (thr_df["substation"] == i)] if len(thr_df) else thr_df
        if len(acted):
            for hid in acted["consumer"].unique():
                g.rel("Event", ek, "AFFECTS", "Household", hid)
                g.rel("Household", hid, "PARTICIPATED_IN", "Event", ek)
            if e["trigger"] == "demand_response":
                for _, ar in acted.iterrows():
                    aid = (f"DR-{ar['consumer']}-{date}-{int(ar['slot']):02d}-"
                           f"{'X' if ar['action'] == 'shift' else 'Y'}")
                    g.rel("Event", ek, "CAUSES", "DrAction", aid)

    # Priority pools and scored entities
    for p in pools:
        date = p["date"]
        for name in ("PH", "PM", "PL"):
            q = p["pools"][name]
            pk = f"POOL-{name}-{date}"
            g.node("PriorityPool", pk, pool=name, date=date, n=q["n"],
                   W_P=q["W"], chi_mean=q["chi_mean"], chi_min=q["chi_min"],
                   chi_max=q["chi_max"], chi_bar=p["chi_bar"],
                   cut_low=p["cut_low"], cut_high=p["cut_high"],
                   rule=p["rule"], degenerate=p["degenerate"],
                   alpha_P=p.get("alpha_P", {}).get(name),
                   equation="(50)-(52)")
        for rank, e in enumerate(p["top_n"], start=1):
            g.node("PoolEntity", f"PE-{e['entity_id']}-{date}",
                   entity_id=e["entity_id"], kind=e["kind"],
                   substation=e["substation"], date=date, chi=e["chi"],
                   gcs_score=e["chi"],          # alias for the legacy pack
                   pool=e["pool"], rank=rank, equation="(49),(53)")
            g.rel("PriorityPool", f"POOL-{e['pool']}-{date}", "CONTAINS",
                  "PoolEntity", f"PE-{e['entity_id']}-{date}", rank=rank)


def build_participation(g, subs, populations, thr_df, dates):
    """
    Per (consumer, day) totals, so ABSENCE is queryable in one hop.
    "Which hospital loads were never curtailed" needs this; the sparse
    DrAction nodes alone cannot answer it.
    """
    acted = (thr_df.groupby(["date", "consumer", "action"])
             .agg(energy=("energy_kwh", "sum"), n=("value", "size"),
                  max_value=("value", "max")).reset_index()
             if len(thr_df) else pd.DataFrame())
    for date in dates:
        sub = acted[acted["date"] == date] if len(acted) else acted
        for i in subs:
            for h in populations[i]:
                hid = h.consumer_id
                rows = sub[sub["consumer"] == hid] if len(sub) else sub
                sh = rows[rows.action == "shift"] if len(rows) else rows
                cu = rows[rows.action == "curtail"] if len(rows) else rows
                key = f"PS-{hid}-{date}"
                g.node("ParticipationSummary", key, consumer_id=hid, date=date,
                       substation=i, archetype=h.archetype,
                       category=h.category, w_h=float(h.w),
                       n_shift_slots=int(sh["n"].sum()) if len(sh) else 0,
                       n_curtail_slots=int(cu["n"].sum()) if len(cu) else 0,
                       energy_shifted_kwh=float(sh["energy"].sum()) if len(sh) else 0.0,
                       energy_curtailed_kwh=float(cu["energy"].sum()) if len(cu) else 0.0,
                       max_y=float(cu["max_value"].max()) if len(cu) else 0.0,
                       never_curtailed=bool(len(cu) == 0),
                       never_shifted=bool(len(sh) == 0),
                       participated=bool(len(rows) > 0))
                g.rel("ParticipationSummary", key, "SUMMARISES", "Household", hid)
                g.rel("ParticipationSummary", key, "AT_DAY", "Day", date)


def build_billing(g: GraphBuilder, bills, period: dict, subs) -> None:
    """
    Requirement 8: monthly billing in the graph.

    One MonthlyBill per (household, period), carrying the whole Scenario A vs
    Scenario B comparison - energy, bill, incentive, discomfort, net benefit,
    QoE and DR participation. That makes "why is my bill what it is, and did
    demand response help me" answerable in one hop from the household, with no
    recomputation, which is the same discipline the DrAction nodes follow.
    """
    pk = f"BP-{period['start']}-{period['end']}"
    g.node("BillingPeriod", pk, start=period["start"], end=period["end"],
           days=period["days"], month_label=period.get("month_label", ""),
           observed_days=period.get("observed_days"),
           forecast_days=period.get("forecast_days"),
           tariff_source=T.GAZETTE_REF,
           rho_tk_per_kwh=cfg.RHO_REBATE_TK_PER_KWH,
           qoe_min=cfg.QOE_MIN)
    for d in period.get("dates", []):
        g.rel("BillingPeriod", pk, "COVERS", "Day", d)

    for r in bills:
        hid = r["household_id"] if "household_id" in r else r["consumer_id"]
        bk = f"MB-{hid}-{period['start']}"
        g.node("MonthlyBill", bk,
               household_id=hid, substation=r["substation"],
               period_start=period["start"], period_end=period["end"],
               days=period["days"],
               # Scenario A: no demand response
               energy_before_kwh=float(r["baseline_kwh_30d"]),
               bill_before_tk=float(r["bill_baseline_tk"]),
               # Scenario B: demand response enabled
               energy_after_kwh=float(r["optimised_kwh_30d"]),
               bill_after_tk=float(r["bill_optimised_tk"]),
               # outcome
               bill_saving_tk=float(r["saving_tk"]),
               bill_saving_pct=float(r["saving_pct"]) if pd.notna(r["saving_pct"]) else None,
               energy_saving_kwh=float(r["energy_saving_kwh"]),
               incentive_received_tk=float(r["incentive_received_tk"]),
               discomfort_cost_tk=float(r["discomfort_J2_tk"]),
               net_benefit_tk=float(r["net_benefit_tk"]),
               gained=bool(r["gained"]),
               qoe=float(r["qoe"]),
               n_dr_events=int(r["n_dr_events"]),
               n_shift_slots=int(r["n_shift_slots"]),
               n_curtail_slots=int(r["n_curtail_slots"]),
               energy_shifted_kwh=float(r["energy_shifted_kwh"]),
               energy_curtailed_kwh=float(r["energy_curtailed_kwh"]),
               peak_before_kw=float(r["peak_baseline_kw"]),
               peak_after_kw=float(r["peak_optimised_kw"]),
               peak_reduction_pct=float(r["peak_reduction_pct"])
               if pd.notna(r["peak_reduction_pct"]) else None,
               participated=bool(r["participated"]),
               tariff_class=r["tariff_class"],
               slab_before=r["slab_baseline"], slab_after=r["slab_optimised"],
               eff_rate_before=float(r["eff_rate_baseline"]),
               eff_rate_after=float(r["eff_rate_optimised"]),
               surcharge_tk=float(r["surcharge_tk"]),
               # the formulation's own bill, alongside the gazette one
               J1_before_tk=float(r["J1_baseline_tk"]),
               J1_after_tk=float(r["J1_optimised_tk"]),
               marginal_rate_tk_per_kwh=float(r["marginal_rate_tk_per_kwh"]),
               income_band=r.get("income_band", ""),
               monthly_income_bdt=float(r.get("monthly_income_bdt", 0.0)),
               household_size=int(r.get("household_size", 0)),
               w_h=float(r["w_h"]),
               equation="(20),(21) + BERC slab bill")
        g.rel("Household", hid, "BILLED", "MonthlyBill", bk)
        g.rel("MonthlyBill", bk, "BILLED_UNDER", "BillingPeriod", pk)
        g.rel("MonthlyBill", bk, "SUBJECT_TO", "TariffClass", r["tariff_class"])


# =============================================================================
# Export
# =============================================================================

def export_csv(g: GraphBuilder, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    for label, rows in g.nodes.items():
        df = pd.DataFrame(rows)
        for c in df.columns:
            if df[c].apply(lambda v: isinstance(v, (list, dict))).any():
                df[c] = df[c].apply(lambda v: json.dumps(v)
                                    if isinstance(v, (list, dict)) else v)
        p = out_dir / f"nodes_{label}.csv"
        df.to_csv(p, index=False)
        written[f"nodes_{label}"] = len(df)
    rdf = pd.DataFrame(g.rels)
    rdf.to_csv(out_dir / "relationships.csv", index=False)
    written["relationships"] = len(rdf)
    return written


CYPHER_HEADER = """// GenTwin-SG knowledge graph loader
// Generated by 08_knowledge_graph.py. K = (V, E, lab, rho), Eq. (57).
//
// Place the nodes_*.csv and relationships.csv files in Neo4j's import folder,
// then run this file in Neo4j Browser or cypher-shell.
//
// Backward compatibility: :Household also carries :User and :Event also
// carries :DrEvent, and PoolEntity.chi is mirrored to gcs_score, so the
// earlier GraphRAG query pack still runs.
"""


def export_cypher(g: GraphBuilder, path: Path, csv_dir_name="import") -> Path:
    lines = [CYPHER_HEADER, ""]
    lines.append("// ---- constraints and indexes ----")
    for label in g.nodes:
        lines.append(f"CREATE CONSTRAINT {label.lower()}_id IF NOT EXISTS "
                     f"FOR (n:{label}) REQUIRE n.id IS UNIQUE;")
    for label, prop in (("TimeSlot", "slot_index"), ("TimeSlot", "date"),
                        ("DrAction", "date"), ("DrAction", "clock"),
                        ("Decision", "var_type"), ("Decision", "clock"),
                        ("Event", "regime"), ("Household", "category"),
                        ("ParticipationSummary", "never_curtailed")):
        lines.append(f"CREATE INDEX {label.lower()}_{prop} IF NOT EXISTS "
                     f"FOR (n:{label}) ON (n.{prop});")
    lines.append("")

    for label, rows in g.nodes.items():
        if not rows:
            continue
        props = sorted({k for r in rows for k in r} - {"id"})
        alias = {"Household": ":User", "Event": ":DrEvent"}.get(label, "")
        setters = ", ".join(f"n.{p} = row.{p}" for p in props)
        lines.append(f"// ---- {label} ({len(rows)}) ----")
        lines.append(
            f"LOAD CSV WITH HEADERS FROM 'file:///nodes_{label}.csv' AS row\n"
            f"CALL {{ WITH row\n"
            f"  MERGE (n:{label}{alias} {{id: row.id}})\n"
            f"  SET {setters}\n"
            f"}} IN TRANSACTIONS OF 5000 ROWS;")
        lines.append("")

    lines.append("// ---- relationships ----")
    for rtype in sorted({r["type"] for r in g.rels}):
        pairs = sorted({(r["src_label"], r["dst_label"]) for r in g.rels
                        if r["type"] == rtype})
        for sl, dl in pairs:
            lines.append(
                f"LOAD CSV WITH HEADERS FROM 'file:///relationships.csv' AS row\n"
                f"WITH row WHERE row.type = '{rtype}' AND "
                f"row.src_label = '{sl}' AND row.dst_label = '{dl}'\n"
                f"CALL {{ WITH row\n"
                f"  MATCH (a:{sl} {{id: row.src}}), (b:{dl} {{id: row.dst}})\n"
                f"  MERGE (a)-[:{rtype}]->(b)\n"
                f"}} IN TRANSACTIONS OF 10000 ROWS;")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
