"""
GenTwin-SG :: Stage H - regime, events and priority pooling
===========================================================

    (44)  presence indicator      perp_e
    (45)  entity records          (k_e, v_e) for [:ap], [:mt], [:sb]
    (46)  phi_1 structural weight, phi_2 response energy
    (47)  phi_3 stress, phi_4 deficit
    (48)  operating regime        r_{i,d} in {1,2,3}
    (49)  criticality score       chi_e = sum_j beta_j(r) phi_j(e)
    (50)  pools                   P_H, P_M, P_L
    (51)  refresh weights         w_e, W(P)
    (52)  re-pooling trigger      alpha_P
    (53)  top-N selection         P_N

ORDERING, which the formulation is explicit about
-------------------------------------------------
The regime is "defined before the score that uses it", and Eq. (48) reads
T_ij and nu_i - which are LP OUTPUTS. So the order is strictly:

    forecast  ->  s_{i,d}, Def_{i,d}   (Stage C/D, before the solve)
    LP        ->  T_ij, nu_i, Z^c_i    (Stage E/F)
    regime    ->  r_{i,d}              (here, AFTER the solve)
    chi, pools ->                      (here, using beta_j(r))

s_{i,d} is NOT the regime. It is computed from the forecast alone and enters
the LP objective through the stress rebate in Eq. (20), so it must exist
before the solve. r_{i,d} cannot. Keeping them apart is why the knowledge
graph carries them on separate nodes.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np

from . import config as cfg

ENTITY_KINDS = ("ap", "mt", "sb")


# =============================================================================
# Eq. (48) - operating regime
# =============================================================================

def operating_regime(p_tilde_kw: float, p_max_kw: float, transfer_out_kwh: float,
                     nu_kwh: float, s_stress: int) -> dict:
    """
    r_{i,d} = max{1, 2*1[s=1], 3*1[p~ > P^max OR sum_j T_ij > 0 OR nu > 0]}

    Every trigger is recorded, not just the winner, so an Event can say WHY it
    reached its regime rather than only that it did.
    """
    t3 = {
        "capacity_deficit": bool(p_tilde_kw > p_max_kw),
        "inter_substation_transfer": bool(transfer_out_kwh > 1e-6),
        "reserve_shortfall": bool(nu_kwh > 1e-6),
    }
    r = max(1, 2 * int(s_stress == 1), 3 * int(any(t3.values())))
    return {
        "regime": int(r), "regime_name": cfg.REGIME_NAMES[int(r)],
        "s_stress": int(s_stress),
        "triggers_r3": t3, "any_r3": bool(any(t3.values())),
        "p_tilde_kw": float(p_tilde_kw), "p_max_kw": float(p_max_kw),
        "deficit_kw": float(max(0.0, p_tilde_kw - p_max_kw)),
        "transfer_out_kwh": float(transfer_out_kwh), "nu_kwh": float(nu_kwh),
        "beta": cfg.PRIORITY_WEIGHTS[int(r)],
    }


# =============================================================================
# Eq. (45) - entity records
# =============================================================================

@dataclass
class Entity:
    entity_id: str
    kind: str                        # ap | mt | sb
    substation: str
    date: str
    value: dict                      # v_e
    present: int = 1                 # perp_e, Eq. (44)
    degree: int = 0                  # deg(e)
    e_sh_kwh: float = 0.0
    e_cu_kwh: float = 0.0
    phi: tuple = (0.0, 0.0, 0.0, 0.0)
    chi: float = 0.0
    pool: str = ""
    w: float = 0.0


def build_entities(date, substations, populations, decisions, regimes,
                   forecast_rows) -> list:
    """
    One entity per appliance [:ap], per meter [:mt] (one meter per consumer),
    and per substation [:sb]. Degrees are the knowledge-graph incidence counts.

    Household response energy is allocated to appliances by class: shifted
    energy to the household's shiftable appliances in proportion to their
    rated contribution, curtailed energy to its curtailable ones. Eq. (45)
    stores x* and y* on the appliance record, so the allocation has to be
    consistent with them.
    """
    ents: list[Entity] = []
    for i in substations:
        reg = regimes[i]
        frow = forecast_rows[i]
        cons = populations[i]
        n_meters = len(cons)

        for h in cons:
            dec = decisions.get(h.consumer_id, {})
            e_sh = float(dec.get("e_sh_kwh", 0.0))
            e_cu = float(dec.get("e_cu_kwh", 0.0))
            x_star = float(dec.get("x_max", 0.0))
            y_star = float(dec.get("y_max", 0.0))

            sh_apps = [a for a in h.appliances if a.cls == "shiftable"]
            cu_apps = [a for a in h.appliances if a.cls == "curtailable"]
            sh_tot = sum(a.n * a.p_rat_kw for a in sh_apps) or 1.0
            cu_tot = sum(a.n * a.p_rat_kw for a in cu_apps) or 1.0

            for k, a in enumerate(h.appliances):
                share_sh = (a.n * a.p_rat_kw / sh_tot) if a.cls == "shiftable" else 0.0
                share_cu = (a.n * a.p_rat_kw / cu_tot) if a.cls == "curtailable" else 0.0
                ents.append(Entity(
                    entity_id=f"{h.consumer_id}-A{k:02d}", kind="ap",
                    substation=i, date=date,
                    value={"aID": f"{h.consumer_id}-A{k:02d}",
                           "hID": h.consumer_id, "class": a.cls,
                           "appliance": a.name, "n": a.n,
                           "p_rat_kw": a.p_rat_kw,
                           "x_star": x_star if a.cls == "shiftable" else 0.0,
                           "y_star": y_star if a.cls == "curtailable" else 0.0},
                    e_sh_kwh=e_sh * share_sh, e_cu_kwh=e_cu * share_cu,
                    degree=3))       # meter, class, household

            ents.append(Entity(
                entity_id=f"{h.consumer_id}-M", kind="mt", substation=i,
                date=date,
                value={"mID": f"{h.consumer_id}-M", "iID": i,
                       "N_ap": len(h.appliances), "hID": h.consumer_id,
                       "archetype": h.archetype, "w_h": float(h.w)},
                e_sh_kwh=e_sh, e_cu_kwh=e_cu,
                degree=1 + len(h.appliances) + 1))

        net = decisions.get("__network__", {}).get(i, {})
        ents.append(Entity(
            entity_id=f"SB-{i}", kind="sb", substation=i, date=date,
            value={"iID": i, "N_mt": n_meters,
                   "s_id": int(reg["s_stress"]),
                   "nu_kwh": float(reg["nu_kwh"]),
                   "Z_kwh": float(net.get("Z_total_kwh", 0.0)),
                   "regime": reg["regime"]},
            e_sh_kwh=sum(d.get("e_sh_kwh", 0.0) for k, d in decisions.items()
                         if k != "__network__" and d.get("substation") == i),
            e_cu_kwh=sum(d.get("e_cu_kwh", 0.0) for k, d in decisions.items()
                         if k != "__network__" and d.get("substation") == i),
            degree=n_meters + 1))
    return ents


# =============================================================================
# Eqs. (46)-(47) - criticality features
# =============================================================================

def score_entities(ents: list, regimes: dict, forecast_rows: dict) -> list:
    """phi_1..phi_4 then chi_e = sum_j beta_j(r_{i,d}) phi_j(e), Eq. (49)."""
    counts = {k: sum(1 for e in ents if e.kind == k) for k in ENTITY_KINDS}
    max_count = max(counts.values()) or 1
    max_resp = max((e.e_sh_kwh + e.e_cu_kwh for e in ents), default=0.0) or 1.0

    for e in ents:
        reg = regimes[e.substation]
        frow = forecast_rows[e.substation]
        p_max = float(cfg.SUBSTATIONS[e.substation]["p_max_kw"])

        phi1 = e.present * counts[e.kind] / max_count                    # (46)
        phi2 = (e.e_sh_kwh + e.e_cu_kwh) / max_resp                      # (46)
        phi3 = float(reg["s_stress"])                                    # (47)
        phi4 = float(frow["deficit_kw"]) / p_max if p_max > 0 else 0.0   # (47)
        phi4 = min(phi4, 1.0)

        beta = cfg.PRIORITY_WEIGHTS[reg["regime"]]
        e.phi = (phi1, phi2, phi3, phi4)
        e.chi = float(sum(b * p for b, p in zip(beta, e.phi)))           # (49)
    return ents


# =============================================================================
# Eqs. (50)-(53) - pools
# =============================================================================

def assign_pools(ents: list, rule: str | None = None) -> dict:
    """
    Eq. (50) as written: P_H above 3/4 chi_bar, P_M between 1/2 and 3/4,
    P_L at or below 1/2 chi_bar.

    Section 9.6 records that this mean-relative rule "degenerates on tight
    score distributions" and names quantile cut points as the intended fix.
    Both are implemented; the rule as written is the default and both are
    reported, so the degeneracy is visible rather than silently repaired.
    """
    rule = rule or cfg.POOL_CUT_RULE
    chi = np.array([e.chi for e in ents])
    chi_bar = float(chi.mean()) if len(chi) else 0.0

    if rule == "quantile":
        lo, hi = np.quantile(chi, cfg.POOL_QUANTILES)
    else:
        lo, hi = 0.5 * chi_bar, 0.75 * chi_bar

    for e in ents:
        e.pool = "PH" if e.chi > hi else ("PM" if e.chi > lo else "PL")

    n_ent = len(ents) or 1
    for e in ents:                                                       # (51)
        e.w = (1.0 / n_ent) if e.kind == "ap" else (e.degree / n_ent)

    pools = {}
    for p in ("PH", "PM", "PL"):
        members = [e for e in ents if e.pool == p]
        pools[p] = {
            "n": len(members),
            "W": float(sum(1 + e.degree for e in members) / n_ent),       # (51)
            "chi_mean": float(np.mean([e.chi for e in members])) if members else 0.0,
            "chi_min": float(min((e.chi for e in members), default=0.0)),
            "chi_max": float(max((e.chi for e in members), default=0.0)),
        }

    degenerate = (pools["PH"]["n"] == 0 or pools["PL"]["n"] == 0
                  or pools["PM"]["n"] == 0)
    return {
        "rule": rule, "chi_bar": chi_bar, "cut_low": float(lo),
        "cut_high": float(hi), "pools": pools, "n_entities": n_ent,
        "degenerate": bool(degenerate),
        "chi_spread": float(chi.max() - chi.min()) if len(chi) else 0.0,
    }


def repooling_trigger(w_prev: dict, w_now: dict) -> dict:
    """Eq. (52): alpha_P = |W(P)_t - W(P)_{t+1}| per pool."""
    return {p: abs(w_now["pools"][p]["W"] - w_prev["pools"][p]["W"])
            for p in ("PH", "PM", "PL")}


def top_n(ents: list, n: int | None = None) -> list:
    """
    Eq. (53): P_N = argmax over Q subset of P of sum chi_e, |Q| <= N.

    With a cardinality constraint and no interaction between entities, the
    maximiser is exactly the N highest-scoring entities - a sort, not a search.
    """
    n = n or cfg.POOL_N
    return sorted(ents, key=lambda e: -e.chi)[:n]


# =============================================================================
# Events
# =============================================================================

TRIGGERS = ("regime_stress", "demand_response", "reserve_shortfall",
            "unserved_load")


def generate_events(date, substations, regimes, forecast_rows, decisions,
                    network, thresholds) -> list:
    """
    One event per (substation, day, trigger) that actually fired. Each carries
    everything the brief asks for plus the causal links Stage K needs.
    """
    events = []
    for i in substations:
        reg = regimes[i]
        frow = forecast_rows[i]
        net = network[i]
        acting = {cid: d for cid, d in decisions.items()
                  if cid != "__network__" and d.get("substation") == i
                  and (d.get("e_sh_kwh", 0) > 0 or d.get("e_cu_kwh", 0) > 0)}
        thr_i = [t for t in thresholds if t["substation"] == i]

        fired = {
            "regime_stress": reg["regime"] >= 2,
            "demand_response": len(acting) > 0,
            "reserve_shortfall": float(net["nu"].sum()) > 1e-6,
            "unserved_load": float(sum(net[f"Z_{c}"].sum()
                                       for c in ("critical", "shiftable",
                                                 "curtailable"))) > 1e-6,
        }
        for trig, ok in fired.items():
            if not ok:
                continue
            e_sh = sum(d["e_sh_kwh"] for d in acting.values())
            e_cu = sum(d["e_cu_kwh"] for d in acting.values())
            events.append({
                "event_id": f"EV-{cfg.SUBSTATIONS[i]['code']}-{date}-{trig}",
                "timestamp": f"{date}T00:00:00",
                "date": date, "substation": i, "trigger": trig,
                "regime": reg["regime"], "regime_name": reg["regime_name"],
                "stress_indicator": reg["s_stress"],
                "forecast_peak_kw": frow["p_hat_kw"],
                "robust_peak_kw": frow["p_tilde_kw"],
                "p_str_kw": frow["p_str_kw"],
                "p_max_kw": frow["p_max_kw"],
                "deficit_kw": frow["deficit_kw"],
                "kappa": frow["kappa_scale"],
                "actual_peak_kw": float(np.max(net.get("post_response_kw",
                                                       [np.nan]))),
                "n_affected_consumers": len(acting),
                "energy_shifted_kwh": float(e_sh),
                "energy_curtailed_kwh": float(e_cu),
                "energy_reduction_kwh": float(e_cu),
                "reserve_shortfall_kwh": float(net["nu"].sum()),
                "unserved_kwh": float(sum(net[f"Z_{c}"].sum()
                                          for c in ("critical", "shiftable",
                                                    "curtailable"))
                                      * cfg.DELTA_H),
                "transfer_out_kwh": reg["transfer_out_kwh"],
                "cost_impact_tk": float(decisions.get("__network__", {})
                                        .get(i, {}).get("f_cost_tk", 0.0)),
                "pi_min_tk_per_kwh": float(net["pi"].min()),
                "pi_max_tk_per_kwh": float(net["pi"].max()),
                "n_threshold_records": len(thr_i),
                "regime_triggers": reg["triggers_r3"],
                "beta_weights": list(reg["beta"]),
            })
    return events
