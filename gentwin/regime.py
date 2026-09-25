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
                     nu_kwh: float, s_stress: int, transfer_in_kwh: float = 0.0,
                     trigger: str | None = None) -> dict:
    """
    r_{i,d} = max{1, 2*1[s=1], 3*1[p~ > P^max OR transfer-trigger OR nu > 0]}

    Every trigger is recorded, not just the winner, so an Event can say WHY it
    reached its regime rather than only that it did.

    `cfg.REGIME_TRANSFER_TRIGGER` (reviewer follow-up) decides which side of
    an inter-substation transfer counts as "emergency":

      "export"  (the previous, undocumented default) - the SENDING
                substation is flagged. Rationale: it is the one whose own
                reserve/capacity margin was drawn down to help a neighbour,
                which is a real cost to it regardless of why the transfer
                happened.
      "import"  (the new default) - the RECEIVING substation is flagged.
                Rationale: it is the one that could not meet its own C1
                supply-adequacy row without help - the more standard reading
                of "which substation is in trouble" in a distribution
                system, and the one an operator would usually act on first.
      "both"    - either side counts.

    Neither reading changes anything in the base-arm results reported so
    far: both study substations show zero transfers in every arm run to
    date. The choice matters only once a scarcity/coupling scenario
    actually produces a nonzero T_ij.
    """
    trigger = trigger or cfg.REGIME_TRANSFER_TRIGGER
    if trigger == "export":
        transfer_flag = transfer_out_kwh > 1e-6
    elif trigger == "both":
        transfer_flag = (transfer_out_kwh > 1e-6) or (transfer_in_kwh > 1e-6)
    else:   # "import", the default
        transfer_flag = transfer_in_kwh > 1e-6

    t3 = {
        "capacity_deficit": bool(p_tilde_kw > p_max_kw),
        "inter_substation_transfer": bool(transfer_flag),
        "reserve_shortfall": bool(nu_kwh > 1e-6),
    }
    r = max(1, 2 * int(s_stress == 1), 3 * int(any(t3.values())))
    return {
        "regime": int(r), "regime_name": cfg.REGIME_NAMES[int(r)],
        "s_stress": int(s_stress),
        "triggers_r3": t3, "any_r3": bool(any(t3.values())),
        "p_tilde_kw": float(p_tilde_kw), "p_max_kw": float(p_max_kw),
        "deficit_kw": float(max(0.0, p_tilde_kw - p_max_kw)),
        "transfer_out_kwh": float(transfer_out_kwh),
        "transfer_in_kwh": float(transfer_in_kwh),
        "transfer_trigger_rule": trigger,
        "nu_kwh": float(nu_kwh),
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
    # Issue E follow-up (tie-break): the appliance's own flexibility
    # potential (rated power * duty cycle, i.e. its typical draw), 0 for
    # non-appliance kinds. Used ONLY to break ties in chi - never enters chi
    # itself - so an appliance that could move more energy outranks an
    # identically-scored one that could move less, instead of the tie being
    # broken by an arbitrary string (entity_id).
    flex_potential: float = 0.0


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
                    degree=3,        # meter, class, household
                    flex_potential=float(a.n * a.p_rat_kw * a.duty_cycle)))

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
    """
    phi_1..phi_4 then chi_e = sum_j beta_j(r_{i,d}) phi_j(e), Eq. (49).

    Issue E fix, both KIND-LEVEL normalisations replaced by the correct
    scope:

      phi1  was e.present * counts[e.kind] / max_count(by kind) - a value
            IDENTICAL for every entity of a given kind (all "ap" get exactly
            1.0, since appliances are always the largest kind by count).
            That is why P_H contained all 5,416 appliances every day: phi1
            never discriminated between them. Corrected to the base paper's
            actual Eq. (3): phi1(e) = present_e * deg(e) / max_e' deg(e') -
            ENTITY-level, using the entity's own graph degree, normalised
            against the single highest-degree entity in the whole graph
            (typically a substation or a well-connected meter, not an
            appliance, which has a fixed degree of 3).

      phi2  was normalised by the GLOBAL max response energy across ALL
            kinds - but a substation ("sb") entity's response energy is the
            SUM over every consumer at it, dwarfing any single appliance's
            or meter's, so every "ap"/"mt" phi2 collapsed toward 0.
            Corrected to normalise WITHIN each kind: an appliance is scored
            against the most-responsive appliance, a meter against the
            most-responsive meter, a substation against... the other
            substation (n=2, but the scope is still correct in principle).
    """
    max_deg = max((e.degree for e in ents), default=0) or 1
    max_resp_by_kind = {
        k: max((e.e_sh_kwh + e.e_cu_kwh for e in ents if e.kind == k),
              default=0.0) or 1.0
        for k in ENTITY_KINDS
    }

    for e in ents:
        reg = regimes[e.substation]
        frow = forecast_rows[e.substation]
        p_max = float(cfg.SUBSTATIONS[e.substation]["p_max_kw"])

        phi1 = e.present * e.degree / max_deg                            # (46), corrected
        phi2 = (e.e_sh_kwh + e.e_cu_kwh) / max_resp_by_kind[e.kind]       # (46), corrected
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
    Eq. (50): P_H above 3/4 chi_bar, P_M between 1/2 and 3/4, P_L at or below
    1/2 chi_bar (the "mean_relative" rule, as the base paper writes it).

    Section 9.6 records that this rule "degenerates on tight score
    distributions" - and Issue E confirms it: at cfg.PRIORITY_WEIGHTS'
    scale, the medium pool (3/4 Avg <= p < Avg/2) the base paper's own Eq.
    (4) defines is provably an EMPTY interval whenever the score
    distribution is unimodal and continuous (3/4 chi_bar > 1/2 chi_bar for
    any chi_bar > 0, so nothing can be BOTH > 3/4 chi_bar cut... the
    intervals as literally written do not partition the line without a
    degenerate middle band under common distributions - see
    docs/thesis_changes.md, Issue E). QUANTILE is therefore now the
    DEFAULT (cfg.POOL_CUT_RULE); mean_relative is kept, selectable with
    `--pool-rule mean_relative`, as the base-paper replication ablation.
    """
    rule = rule or cfg.POOL_CUT_RULE
    chi = np.array([e.chi for e in ents])
    chi_bar = float(chi.mean()) if len(chi) else 0.0

    # w_e (Eq. 51) is computed FIRST, not after pool assignment as before -
    # it is now also used as part of the tie-break key below.
    n_ent = len(ents) or 1
    for e in ents:
        e.w = (1.0 / n_ent) if e.kind == "ap" else (e.degree / n_ent)

    tie_info = {"tie_break_used": False, "tied_at_lo": 0, "tied_at_hi": 0}

    if rule == "quantile":
        lo, hi = np.quantile(chi, cfg.POOL_QUANTILES)
        if hi > lo:
            for e in ents:
                e.pool = "PH" if e.chi > hi else ("PM" if e.chi > lo else "PL")
        else:
            # Tie at the cut point: a large enough cluster of entities shares
            # the EXACT SAME chi (common here - most non-acting appliances at
            # one substation get an identical phi1/phi3/phi4, since degree,
            # s_stress and deficit are all substation-level constants) that
            # the 50th and 75th percentile VALUES coincide, leaving PM empty
            # even under quantile.
            #
            # Follow-up fix: ties are broken by a MEANINGFUL key, not the
            # arbitrary entity_id used previously - flex_potential (an
            # appliance's rated power * duty cycle, i.e. how much load it
            # could actually move; 0 for meters/substations) first, then w_e.
            # A bigger, more flexible appliance tied on chi with a smaller
            # one is ranked above it, not decided by string comparison. This
            # still produces an exact 50/25/25 split by sorted POSITION
            # (cannot leave a pool empty for n_ent >= 4), but the split
            # WITHIN a tied cluster is now a real judgement, not noise -
            # reported honestly via tie_info below rather than hidden.
            order = sorted(range(len(ents)),
                           key=lambda k: (chi[k], ents[k].flex_potential, ents[k].w))
            n = len(ents)
            lo_cut, hi_cut = int(round(cfg.POOL_QUANTILES[0] * n)), \
                int(round(cfg.POOL_QUANTILES[1] * n))
            for rank, k in enumerate(order):
                ents[k].pool = ("PL" if rank < lo_cut else
                                "PM" if rank < hi_cut else "PH")
            # How many entities actually shared the value sitting at each cut
            # (i.e. how much of the split at that boundary was tie-broken
            # rather than a genuine chi difference) - reported, not hidden.
            chi_sorted = chi[[order[k] for k in range(n)]]
            lo_val = chi_sorted[max(lo_cut - 1, 0)]
            hi_val = chi_sorted[max(hi_cut - 1, 0)]
            tie_info = {
                "tie_break_used": True,
                "tied_at_lo": int(np.sum(chi == lo_val)),
                "tied_at_hi": int(np.sum(chi == hi_val)),
            }
    else:
        lo, hi = 0.5 * chi_bar, 0.75 * chi_bar
        for e in ents:
            e.pool = "PH" if e.chi > hi else ("PM" if e.chi > lo else "PL")

    pools = {}
    for p in ("PH", "PM", "PL"):
        members = [e for e in ents if e.pool == p]
        pools[p] = {
            "n": len(members),
            # Issue E fix: W(P) = sum_{e in P} w_e (Eq. 51), literally - the
            # previous formula, sum(1+degree)/n_ent, did not use w_e at all.
            "W": float(sum(e.w for e in members)),
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
        **tie_info,
    }


def apply_membership(ents: list, membership: dict) -> dict:
    """
    Issue E hysteresis: apply YESTERDAY's pool labels to TODAY's entities
    (entity_id is stable across days - see build_entities), instead of
    recomputing cut points from today's chi distribution. Recomputes w_e,
    W(P) and the summary dict exactly as assign_pools does, just from a
    carried-over `pool` label rather than a fresh chi_bar/quantile cut.
    Falls back to putting an entity seen for the first time in "PL" (the
    conservative default - it cannot yet be known to be high-priority).
    """
    for e in ents:
        e.pool = membership.get(e.entity_id, "PL")
    n_ent = len(ents) or 1
    for e in ents:
        e.w = (1.0 / n_ent) if e.kind == "ap" else (e.degree / n_ent)
    chi = np.array([e.chi for e in ents])
    pools = {}
    for p in ("PH", "PM", "PL"):
        members = [e for e in ents if e.pool == p]
        pools[p] = {
            "n": len(members),
            "W": float(sum(e.w for e in members)),
            "chi_mean": float(np.mean([e.chi for e in members])) if members else 0.0,
            "chi_min": float(min((e.chi for e in members), default=0.0)),
            "chi_max": float(max((e.chi for e in members), default=0.0)),
        }
    return {
        "rule": "reused", "chi_bar": float(chi.mean()) if len(chi) else 0.0,
        "cut_low": None, "cut_high": None, "pools": pools, "n_entities": n_ent,
        "degenerate": bool(pools["PH"]["n"] == 0 or pools["PM"]["n"] == 0
                          or pools["PL"]["n"] == 0),
        "chi_spread": float(chi.max() - chi.min()) if len(chi) else 0.0,
    }


def repooling_trigger(w_prev: dict, w_now: dict) -> dict:
    """Eq. (52): alpha_P = |W(P)_t - W(P)_{t+1}| per pool."""
    return {p: abs(w_now["pools"][p]["W"] - w_prev["pools"][p]["W"])
            for p in ("PH", "PM", "PL")}


def membership_change_fraction(ents: list, prev_membership: dict | None) -> float:
    """
    Alternative re-pooling trigger (reviewer follow-up): the share of
    entities whose pool label actually changed from yesterday to today's
    freshly-scored candidate, by entity_id. Unlike alpha_P (which only
    moves when entities of DIFFERENT w_e swap pools, since W(P) is a
    w_e-weighted sum), this is sensitive to any reshuffling at all,
    including same-kind churn driven by today's actions - the case alpha_P
    is structurally blind to. Returns 1.0 (treat as "always repool") when
    there is no previous membership to compare against.
    """
    if not prev_membership:
        return 1.0
    n = len(ents) or 1
    changed = sum(1 for e in ents if prev_membership.get(e.entity_id) != e.pool)
    return changed / n


POOL_RANK = {"PH": 0, "PM": 1, "PL": 2}


def top_n(ents: list, by_kind: dict | None = None) -> list:
    """
    Eq. (53): P_N = argmax over Q subset of P of sum chi_e, |Q| <= N.

    Issue E fix: a single flat top-N (sorted only by chi) was entirely
    appliances every day, for the same reason the pools were: appliances
    outnumber meters and substations ~7:1, so an unstratified sort never
    surfaces anything else. Stratified by kind (cfg.POOL_N_BY_KIND: all
    substations, top 30 meters, top 20 appliances by default), ranked
    within each kind by (pool order PH > PM > PL, chi desc, w_e desc) -
    still exactly a sort, not a search, since there is no interaction
    between entities of different kinds either.
    """
    by_kind = by_kind or cfg.POOL_N_BY_KIND

    def _key(e):
        return (POOL_RANK.get(e.pool, 3), -e.chi, -e.w)

    out = []
    for kind in ENTITY_KINDS:
        kind_ents = sorted((e for e in ents if e.kind == kind), key=_key)
        limit = by_kind.get(kind)
        out.extend(kind_ents if limit is None else kind_ents[:limit])
    return out


def random_topn(ents: list, by_kind: dict | None = None, seed: int = 42) -> list:
    """
    Reviewer follow-up: a same-size RANDOM baseline for top_n(), stratified
    by the SAME POOL_N_BY_KIND sizes, so precision_at_n() has a meaningful
    "no information" comparator rather than only comparing pooling rules
    against each other.
    """
    import numpy as np
    by_kind = by_kind or cfg.POOL_N_BY_KIND
    rng = np.random.default_rng(seed)
    out = []
    for kind in ENTITY_KINDS:
        kind_ents = [e for e in ents if e.kind == kind]
        limit = by_kind.get(kind)
        if limit is None or limit >= len(kind_ents):
            out.extend(kind_ents)
        else:
            idx = rng.choice(len(kind_ents), size=limit, replace=False)
            out.extend(kind_ents[i] for i in idx)
    return out


def precision_at_n(ents: list, topn: list) -> dict:
    """
    Reviewer follow-up: is the pooling actually finding anything, or are the
    acceptance checks (non-empty pools, mixed kinds in top-N) satisfied by
    construction alone? "Acting" = an entity with real response energy that
    day (e_sh_kwh + e_cu_kwh > 0) - the thing a demand-response priority
    list should actually be prioritising. precision = share of top_n that
    is acting; baseline_random = the prevalence of acting entities in the
    whole population (what a same-size RANDOM sample would be EXPECTED to
    hit, in the long run) - the number precision must beat for the ranking
    to be doing real work, not just filling seats.
    """
    n_total = len(ents) or 1
    n_acting = sum(1 for e in ents if (e.e_sh_kwh + e.e_cu_kwh) > 1e-9)
    n_topn = len(topn) or 1
    hits = sum(1 for e in topn if (e.e_sh_kwh + e.e_cu_kwh) > 1e-9)
    return {
        "n_total": n_total, "n_acting": n_acting, "n_topn": len(topn),
        "hits": hits, "precision": hits / n_topn,
        "baseline_random": n_acting / n_total,
        "lift": (hits / n_topn) / (n_acting / n_total) if n_acting else float("nan"),
    }


def precision_at_n_ids(topn_ids: set, acting_ids: set, n_total: int) -> dict:
    """
    Round-3 follow-up: the label-leakage-safe form of precision_at_n() - takes
    plain entity_id sets instead of live Entity objects, so it can score
    YESTERDAY's top-N (a fixed set of ids, chosen before today's response was
    known) against TODAY's acting set. Same-day precision_at_n() is
    circular because phi2 (the score's biggest lever within a kind) is built
    FROM the same-day acting label it is then checked against; this is not,
    because the ranking that produced `topn_ids` could not have seen
    `acting_ids` yet.
    """
    n_topn = len(topn_ids) or 1
    hits = len(topn_ids & acting_ids)
    n_acting = len(acting_ids)
    return {
        "n_total": n_total, "n_acting": n_acting, "n_topn": len(topn_ids),
        "hits": hits, "precision": hits / n_topn,
        "baseline_random": n_acting / n_total if n_total else 0.0,
        "lift": (hits / n_topn) / (n_acting / n_total)
        if n_acting and n_total else float("nan"),
    }


def chi_excluding_phi2(ents: list, regimes: dict) -> dict:
    """
    Round-3 follow-up: chi_e with phi2's term dropped from Eq. (49) -
    `beta1*phi1 + beta3*phi3 + beta4*phi4`, entity_id -> value. phi2 is built
    directly from the SAME "acting" label (e_sh_kwh+e_cu_kwh) that
    precision_at_n() checks against, so it is the term most responsible for
    same-day precision being circular; scoring without it shows how much
    signal the OTHER three features (graph centrality, stress, deficit) carry
    on their own. Diagnostic only - does not touch e.chi or e.pool.
    """
    out = {}
    for e in ents:
        beta = cfg.PRIORITY_WEIGHTS[regimes[e.substation]["regime"]]
        out[e.entity_id] = float(sum(b * p for j, (b, p) in enumerate(zip(beta, e.phi))
                                     if j != 1))
    return out


def topn_by_chi(ents: list, chi_map: dict, by_kind: dict | None = None) -> list:
    """
    Round-3 follow-up: top-N stratified by kind (same sizes as top_n()), but
    ranked purely by an externally supplied chi_map instead of e.chi/e.pool -
    for scoring a diagnostic chi variant (chi_excluding_phi2, or a different
    pool rule's chi) without disturbing the entities' real pool assignment.
    """
    by_kind = by_kind or cfg.POOL_N_BY_KIND
    out = []
    for kind in ENTITY_KINDS:
        kind_ents = sorted((e for e in ents if e.kind == kind),
                           key=lambda e: -chi_map.get(e.entity_id, 0.0))
        limit = by_kind.get(kind)
        out.extend(kind_ents if limit is None else kind_ents[:limit])
    return out


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
