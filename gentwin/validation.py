"""
GenTwin-SG :: Stage B validation - the synthetic aggregate against BPDB
======================================================================

The gate before the LP. Eq. (6) requires

    Lambda_i(t) = sum_{h in H_i} L_h(t;0,0),      p_{i,d} = max_t Lambda_i(t)

and the calibration band of Eq. (8) requires the peak to reproduce the observed
one. Two acceptance tests matter, and they are not equally easy:

  T1  PEAK LEVEL     max_t Lambda_i(t) = p_{i,d} within AGG_PEAK_TOLERANCE.
                     This is satisfied BY CONSTRUCTION - the global constant K
                     in set_weights() is solved for exactly this. It confirms
                     the calibration ran; it is not evidence that the model is
                     right.

  T2  PEAK TIMING    argmax_t Lambda_i(t) falls inside the substation's
                     adaptive T^pk_i window. This is NOT fitted anywhere. The
                     schedules, ownership rates and thermal responses were all
                     chosen before the timing was tested, and the OSM mix is
                     the only thing standing between them and the answer.
                     T2 is the real test of whether the bottom-up model works.

Everything else here - load factor, class shares, diversity factor, category
decomposition - is a plausibility report rather than a pass/fail gate.

Why T2 is worth taking seriously
--------------------------------
A synthetic population can always be scaled to hit a peak LEVEL. Reproducing
the peak TIMING requires the category mix and their schedules to be jointly
right, and it is the one thing the BPDB dataset independently observes about
the shape of the day. If T2 passes, the composition is doing real work; if it
fails, something in the mix or the schedules is wrong and the diagnostic below
says which.

`critical_residential_share()` reports the residential share at which the peak
flips between the midday and evening candidate. It is a DIAGNOSTIC, not a
calibration knob: the mix used by the pipeline always comes from the OSM cache
and is never set from this function.
"""

from __future__ import annotations

import numpy as np

from . import config as cfg
from . import load_model as lm
from . import population as pop
from . import peak_window as pw


def acceptance(substation: str, agg: dict, target_peak_kw: float,
               observed_peak_slot: int | None = None,
               audit_verdict: dict | None = None,
               stability: dict | None = None) -> list:
    checks = []

    def _c(name, ok, detail="", gate=True):
        checks.append({"check": name, "pass": bool(ok), "detail": str(detail),
                       "gate": gate})

    lam = agg["lambda_kw"]
    err = abs(agg["peak_kw"] - target_peak_kw) / target_peak_kw

    # --- T1 -----------------------------------------------------------------
    _c(f"T1 peak level within {cfg.AGG_PEAK_TOLERANCE:.0%} of p_(i,d)",
       err <= cfg.AGG_PEAK_TOLERANCE,
       f"{agg["peak_kw"]:,.0f} kW vs target {target_peak_kw:,.0f} kW "
       f"({100 * (agg['peak_kw'] - target_peak_kw) / target_peak_kw:+.3f}%). "
       f"Satisfied by construction via the calibration constant K.")

    # --- T2 -----------------------------------------------------------------
    # T2 is only a valid GATE where the BPDB peak-time field is a measurement.
    # `audit_verdict` is read from gentwin/peak_time_audit.py, so this
    # downgrade is mechanical and applies to whatever the audit says - it is
    # not a judgement applied by hand to a substation we wanted to excuse.
    t_pk = set(pw.get_peak_slots(substation))
    inside = agg["peak_slot"] in t_pk
    near = min(abs(agg["peak_slot"] - s) for s in t_pk)
    usable = audit_verdict is None or audit_verdict.get("t2_target_usable", True)

    # The VERDICT is the seed-stable one, never a single draw. On a knife-edge
    # composition one draw can land either side of the boundary, so reporting
    # the default seed would be reporting noise in whichever direction it fell.
    if stability is not None:
        share = stability["share_inside"]
        inside = share >= 0.90
        detail = (f"inside T^pk_i on {stability['n_inside']}/"
                  f"{stability['n_seeds']} population draws ({share:.0%}); "
                  f"T^pk_i = {pw.format_window(sorted(t_pk))}; "
                  f"default-seed argmax {agg['peak_clock']}")
        if 0.10 < share < 0.90:
            detail += (". UNDETERMINED: the argmax is seed-dependent at this "
                       "composition, so neither pass nor fail can be claimed")
            usable = False
    else:
        detail = (f"argmax at {agg['peak_clock']}; T^pk_i = "
                  f"{pw.format_window(sorted(t_pk))}"
                  + ("" if inside else f"; {near} slot(s) outside"))
    if not usable:
        detail += (f". CONDITIONAL: the peak-time audit rates this "
                   f"substation's `Peak_Time` field {audit_verdict['verdict']} "
                   f"({audit_verdict['n_flags']} flags), so the target is not "
                   f"trustworthy enough to fail the model against. See "
                   f"docs/peak_time_reliability.md.")
    _c("T2 peak timing falls inside the adaptive T^pk_i window",
       inside, detail, gate=usable)
    if not usable:
        checks[-1]["conditional"] = True

    if observed_peak_slot is not None and usable:
        d = min(abs(agg["peak_slot"] - observed_peak_slot),
                cfg.SLOTS_PER_DAY - abs(agg["peak_slot"] - observed_peak_slot))
        _c("T2b argmax within one slot of the reference day's recorded peak time",
           d <= cfg.AGG_PEAK_SLOT_TOLERANCE,
           f"synthetic {agg['peak_clock']} vs recorded "
           f"{cfg.slot_to_clock(observed_peak_slot)} ({d} slot(s) apart)")

    # --- plausibility (reported, not gated) ---------------------------------
    lf = agg["load_factor"]
    _c("load factor in a plausible urban range (0.55-0.85)",
       0.55 <= lf <= 0.85, f"{lf:.3f}", gate=False)

    shares = {c: float(v.sum() * cfg.DELTA_H / agg["energy_kwh"])
              for c, v in agg["by_class_kw"].items()}
    _c("critical class share is 10-30% of energy",
       0.10 <= shares.get("critical", 0) <= 0.30,
       ", ".join(f"{k} {v:.3f}" for k, v in shares.items()), gate=False)

    flex = shares.get("shiftable", 0) + shares.get("curtailable", 0)
    _c("flexible energy share is large enough for DR to be meaningful (>50%)",
       flex > 0.50, f"{flex:.3f} of daily energy is shiftable or curtailable",
       gate=False)

    _c("no slot is negative or non-finite",
       bool(np.all(np.isfinite(lam)) and np.all(lam >= 0)))

    _c("night minimum is below 60% of the peak",
       float(lam.min()) < 0.60 * float(lam.max()),
       f"min {lam.min():,.0f} kW at {cfg.slot_to_clock(int(np.argmin(lam)))}, "
       f"{100 * lam.min() / lam.max():.1f}% of peak", gate=False)

    return checks


def timing_stability(substation: str, cat: dict, schedules: dict,
                     theta: np.ndarray, mix: dict, target_peak_kw: float,
                     n_seeds: int = 40) -> dict:
    """
    Is T2 a result, or a coin flip?

    The synthetic population is a random draw. Where the midday and evening
    peak candidates are close - at Dhanmondi they differ by under 1% - a
    different draw can move the argmax across the T^pk_i boundary. Reporting
    one seed would then be reporting noise, in whichever direction that seed
    happened to fall.

    This re-synthesises the population across `n_seeds` and reports the share
    of draws whose peak lands inside T^pk_i. Only a share near 0 or near 1 is
    a finding; anything in between means T2 is undetermined at this
    composition and must be reported as such.
    """
    from collections import Counter
    t_pk = set(pw.get_peak_slots(substation))
    clocks, inside = [], 0
    for seed in range(n_seeds):
        cons = pop.synthesise(substation, cat, mix, seed=seed)
        lm.set_weights(cons, schedules, cat, theta, mix, target_peak_kw)
        a = lm.aggregate(cons, schedules, cat, theta)
        clocks.append(a["peak_clock"])
        inside += int(a["peak_slot"] in t_pk)
    share = inside / n_seeds
    return {
        "n_seeds": n_seeds, "n_inside": inside, "share_inside": share,
        "distribution": dict(Counter(clocks).most_common()),
        "verdict": ("ROBUST PASS" if share >= 0.90 else
                    "ROBUST FAIL" if share <= 0.10 else
                    "UNDETERMINED - T2 is seed-dependent at this composition"),
    }


def diversity_factor(consumers, agg: dict) -> float:
    """
    DIVERSITY FACTOR, per IEEE 141:

        DF = sum of individual peaks / coincident system peak      (DF >= 1)

    Its reciprocal is the COINCIDENCE FACTOR, CF = 1/DF (CF <= 1), and the two
    are routinely confused. Urban residential feeders run CF ~ 0.2-0.4, i.e.
    DF ~ 2.5-5.0. Quote whichever you like but say which.

    ONE CAVEAT that bounds the interpretation: this is the diversity ACROSS THE
    REPRESENTATIVE AGENTS, not across the real population. Each agent stands
    for w_h identical households which by construction peak together, so the
    measured DF is a LOWER BOUND on the diversity a real population of
    sum(w_h) households would show. Raising N_REPRESENTATIVE_CONSUMERS tightens
    the bound; it does not change the aggregate, which is pinned to the BPDB
    peak either way.
    """
    non_coincident = sum(float(v["critical"].max() + v["shiftable"].max()
                               + v["curtailable"].max())
                         for v in agg["per_consumer_kw"].values())
    return non_coincident / agg["peak_kw"] if agg["peak_kw"] > 0 else float("nan")


def critical_residential_share(substation: str, cat: dict, schedules: dict,
                               theta: np.ndarray, mix: dict,
                               target_peak_kw: float,
                               grid=np.arange(0.20, 0.66, 0.01)) -> dict:
    """
    DIAGNOSTIC ONLY. At what residential share does the synthetic peak move
    between the midday and evening candidates?

    Never used to set the mix. Its purpose is to say how far the OSM estimate
    would have to be wrong for the T2 verdict to change - which is the honest
    way to report a near-miss.
    """
    t_pk = set(pw.get_peak_slots(substation))
    others = sum(v for k, v in mix.items() if k != "Residential")
    rows = []
    for rs in grid:
        m = {k: (float(rs) if k == "Residential" else v * (1 - rs) / others)
             for k, v in mix.items()}
        cons = pop.synthesise(substation, cat, m)
        lm.set_weights(cons, schedules, cat, theta, m, target_peak_kw)
        a = lm.aggregate(cons, schedules, cat, theta)
        rows.append({"residential_share": round(float(rs), 3),
                     "peak_slot": a["peak_slot"], "peak_clock": a["peak_clock"],
                     "inside_t_pk": a["peak_slot"] in t_pk})
    passing = [r["residential_share"] for r in rows if r["inside_t_pk"]]
    return {
        "rows": rows,
        "shares_passing_T2": passing,
        "max_share_passing": max(passing) if passing else None,
        "current_share": float(mix.get("Residential", 0.0)),
        "margin": (max(passing) - float(mix.get("Residential", 0.0)))
        if passing else None,
    }
