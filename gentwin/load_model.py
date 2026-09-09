"""
GenTwin-SG :: Stage B - load model, Eqs. (2)-(8)
================================================

Builds the bottom-up load from the synthetic population and calibrates it
against the BPDB reference peak.

    (2)  P_{h,a}(t) = n_{h,a} P^rat_a epsilon_a u_{h,a}(t) f_a(Theta(t))
    (3)  L^c_h(t) = sum_{a in A^c_h} P_{h,a}(t),   L_h(t) = sum_c L^c_h(t)
    (4)  L_h(t;x,y) = L^cr + (1-x)L^sh + (1-y)L^cu + R_h(t;x)
    (5)  R_h(t;x)  = 1[t in T^off] / |T^off| * sum_{tau in T^pk} x_{h,tau} L^sh_h(tau)
    (6)  Lambda_i(t) = sum_{h in H_i} L_h(t;0,0),   p_{i,d} = max_t Lambda_i(t)
    (7)  Lambda_i(t) = Lambda^f_i(t) + xi(d) Lambda^v_i(t)
    (8)  the calibration band on the mean daily peak

Units: kW throughout, matching P^rat_a and L^c_h(t) in the formulation's
section 1.2. No conversion layer.

The fixed / variable split of Eq. (7)
-------------------------------------
Lambda^f is the CRITICAL class and Lambda^v is shiftable + curtailable. The
reasoning is physical rather than convenient: critical load is what the
consumer cannot vary - refrigeration, life safety, servers - and is therefore
the part that does not move with occupancy, weather or behaviour. Everything
xi(d) is asked to absorb is behavioural or thermal, which is what varies day to
day. It also means xi never rescales load the optimiser is forbidden to touch.

Equation (5) and the 48-slot grid
---------------------------------
Recovery spreads displaced peak energy evenly across the off-peak window of the
SAME day, now in half-hour slots and over the substation's own T^pk_i from the
adaptive local ToU stage. C6 (Eq. 34) forces x_{h,t} = 0 outside T^pk_i, so
recovery only ever returns energy that was actually displaced.
"""

from __future__ import annotations

import numpy as np

from . import config as cfg
from . import population as pop

CLASSES = ("critical", "shiftable", "curtailable")


# =============================================================================
# Eq. (2) and (3)
# =============================================================================

def consumer_profile(consumer, schedules: dict, cat: dict,
                     theta: np.ndarray) -> dict:
    """
    L^c_h(t) for each class and L_h(t), in kW, for ONE consumer.

    The per-consumer jitter (a slot roll and an amplitude factor, both fixed at
    synthesis time and stored on the consumer) is what gives the population
    diversity. Without it every consumer of a category would switch on in the
    same slot and the aggregate would be a spike rather than a load curve.
    """
    out = {c: np.zeros(cfg.SLOTS_PER_DAY) for c in CLASSES}
    for a in consumer.appliances:
        # u_{h,a}(t): the household's OWN binary on/off pattern, sampled at
        # synthesis. Falls back to the shared schedule for populations built
        # before on_slots existed, so old caches still load.
        if getattr(a, "on_slots", None):
            u = np.zeros(cfg.SLOTS_PER_DAY)
            u[np.asarray(a.on_slots, dtype=int)] = 1.0
        else:
            u = schedules[a.schedule]
            if consumer.jitter_offset:
                u = np.roll(u, consumer.jitter_offset)
        f = thermal_cache(cat, a.thermal, theta)
        # Eq. (2), in kW. `simultaneity` is the fraction of the n owned units
        # drawing at once - a household with three ACs rarely runs all three.
        p = (a.n * getattr(a, "simultaneity", 1.0) * a.p_rat_kw * a.duty_cycle
             * u * f * consumer.jitter_amplitude)
        out[a.cls] += p
    out["total"] = sum(out[c] for c in CLASSES)
    return out


_THERMAL_CACHE: dict = {}


def thermal_cache(cat: dict, kind: str, theta: np.ndarray) -> np.ndarray:
    key = (kind, theta.tobytes())
    if key not in _THERMAL_CACHE:
        _THERMAL_CACHE[key] = pop.thermal_factor(cat, kind, theta)
    return _THERMAL_CACHE[key]


# =============================================================================
# Eq. (6) - aggregation
# =============================================================================

def aggregate(consumers, schedules: dict, cat: dict, theta: np.ndarray,
              weighted: bool = True) -> dict:
    """
    Lambda_i(t) = sum_h w_h L_h(t;0,0), split by class and by category, in kW.
    """
    tot = {c: np.zeros(cfg.SLOTS_PER_DAY) for c in CLASSES}
    by_cat: dict = {}
    by_archetype: dict = {}
    per_consumer = {}

    for h in consumers:
        prof = consumer_profile(h, schedules, cat, theta)
        w = float(h.w) if weighted else 1.0
        per_consumer[h.consumer_id] = {c: prof[c] * w for c in CLASSES}
        for c in CLASSES:
            tot[c] += prof[c] * w
        by_cat.setdefault(h.category, np.zeros(cfg.SLOTS_PER_DAY))
        by_cat[h.category] += prof["total"] * w
        by_archetype.setdefault(h.archetype, np.zeros(cfg.SLOTS_PER_DAY))
        by_archetype[h.archetype] += prof["total"] * w

    lam = sum(tot[c] for c in CLASSES)
    return {
        "lambda_kw": lam,
        "by_class_kw": tot,
        "by_category_kw": by_cat,
        "by_archetype_kw": by_archetype,
        "per_consumer_kw": per_consumer,
        "peak_kw": float(lam.max()),
        "peak_slot": int(np.argmax(lam)),
        "peak_clock": cfg.slot_to_clock(int(np.argmax(lam))),
        "energy_kwh": float(lam.sum() * cfg.DELTA_H),
        "load_factor": float(lam.mean() / lam.max()) if lam.max() > 0 else 0.0,
    }


# =============================================================================
# Calibration - Eq. (7) and (8)
# =============================================================================

def set_weights(consumers, schedules: dict, cat: dict, theta: np.ndarray,
                mix: dict, target_peak_kw: float) -> dict:
    """
    Two-stage weighting, as described in population.py.

      1. v_h = s_c / E_c reproduces the OSM demand mix exactly (by energy).
      2. K = p_target / max_t Lambda_rel(t) fixes the absolute level, so that
         xi = 1 on the reference day.

    Mutates `consumers` in place (sets .w) and returns a diagnostics dict.
    """
    raw = {}
    energy_by_cat: dict = {}
    for h in consumers:
        prof = consumer_profile(h, schedules, cat, theta)
        raw[h.consumer_id] = prof
        e = float(prof["total"].sum() * cfg.DELTA_H)
        energy_by_cat[h.category] = energy_by_cat.get(h.category, 0.0) + e

    # Stage 1: composition.
    for h in consumers:
        s_c = float(mix.get(h.category, 0.0))
        e_c = energy_by_cat.get(h.category, 0.0)
        h.w = (s_c / e_c) if e_c > 0 else 0.0

    rel = np.zeros(cfg.SLOTS_PER_DAY)
    for h in consumers:
        rel += raw[h.consumer_id]["total"] * h.w
    rel_peak = float(rel.max())
    if rel_peak <= 0:
        raise ValueError("relative aggregate has zero peak; cannot calibrate")

    # Stage 2: level.
    K = float(target_peak_kw) / rel_peak
    for h in consumers:
        h.w *= K

    agg = aggregate(consumers, schedules, cat, theta)
    achieved_shares = {c: float(v.sum() * cfg.DELTA_H / agg["energy_kwh"])
                       for c, v in agg["by_category_kw"].items()}
    return {
        "global_constant_K": K,
        "target_peak_kw": float(target_peak_kw),
        "achieved_peak_kw": agg["peak_kw"],
        "peak_error_pct": 100.0 * (agg["peak_kw"] - target_peak_kw) / target_peak_kw,
        "peak_slot": agg["peak_slot"], "peak_clock": agg["peak_clock"],
        "total_connections_implied": float(sum(h.w for h in consumers)),
        "connections_by_category": {
            c: float(sum(h.w for h in consumers if h.category == c))
            for c in mix},
        "osm_share_target": dict(mix),
        "osm_share_achieved": achieved_shares,
        "max_share_error": max(
            (abs(achieved_shares.get(c, 0.0) - s) for c, s in mix.items()),
            default=0.0),
        "mean_w": float(np.mean([h.w for h in consumers])),
    }


def solve_xi(consumers, schedules: dict, cat: dict, theta: np.ndarray,
             target_peak_kw: float) -> dict:
    """
    Eq. (7): solve xi(d) such that max_t [Lambda^f(t) + xi Lambda^v(t)] equals
    the observed peak of day d.

    max over t of an affine function of xi is convex and non-decreasing in xi
    (Lambda^v >= 0), so a bisection is exact and needs no starting guess. This
    is the ONE degree of freedom the calibration has per day; xi far from 1.0
    is a signal that the catalogue does not describe that day, and is reported
    rather than absorbed.
    """
    agg = aggregate(consumers, schedules, cat, theta)
    fixed = agg["by_class_kw"]["critical"]
    var = agg["by_class_kw"]["shiftable"] + agg["by_class_kw"]["curtailable"]

    def peak(xi: float) -> float:
        return float((fixed + xi * var).max())

    lo, hi = 0.0, 1.0
    if peak(lo) > target_peak_kw:
        return {"xi": 0.0, "achieved_peak_kw": peak(0.0),
                "feasible": False,
                "note": ("critical load alone already exceeds the target peak; "
                         "xi cannot reach it. The catalogue's critical block is "
                         "too heavy for this substation.")}
    while peak(hi) < target_peak_kw and hi < 1e4:
        hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if peak(mid) < target_peak_kw:
            lo = mid
        else:
            hi = mid
    xi = 0.5 * (lo + hi)
    lam = fixed + xi * var
    return {
        "xi": float(xi), "achieved_peak_kw": float(lam.max()),
        "target_peak_kw": float(target_peak_kw),
        "peak_error_pct": 100.0 * (lam.max() - target_peak_kw) / target_peak_kw,
        "peak_slot": int(np.argmax(lam)),
        "peak_clock": cfg.slot_to_clock(int(np.argmax(lam))),
        "feasible": True,
        "lambda_kw": lam,
    }


# =============================================================================
# Eq. (4) and (5) - post-response load, for the LP builder
# =============================================================================

def recovery(x_h: np.ndarray, l_sh: np.ndarray, peak_slots) -> np.ndarray:
    """
    Eq. (5): R_h(t;x). Energy displaced from peak slots is returned evenly
    across the off-peak window of the same day.
    """
    peak_slots = set(peak_slots)
    off = [t for t in range(cfg.SLOTS_PER_DAY) if t not in peak_slots]
    displaced = float(sum(x_h[t] * l_sh[t] for t in peak_slots))
    r = np.zeros(cfg.SLOTS_PER_DAY)
    if off and displaced > 0:
        r[off] = displaced / len(off)
    return r


def post_response_load(prof: dict, x_h: np.ndarray, y_h: np.ndarray,
                       peak_slots) -> np.ndarray:
    """Eq. (4): L_h(t; x, y)."""
    return (prof["critical"]
            + (1.0 - x_h) * prof["shiftable"]
            + (1.0 - y_h) * prof["curtailable"]
            + recovery(x_h, prof["shiftable"], peak_slots))
