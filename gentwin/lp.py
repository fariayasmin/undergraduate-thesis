"""
GenTwin-SG :: Stages E/F - the linear programme, Eqs. (17)-(35)
==============================================================

One decision vector q, both layers, solved simultaneously. No demand-response
quantity or price is passed between two problems.

    (17)  q = ({x},{y},{G},{B^ch},{B^dis},{T},{Z},{nu})
    (18)  G_{i,d}, B_{i,d}, T_{ij,d}   daily totals, SUBSTITUTED not variables
    (19)  E^sh, E^cu
    (20)  J^h_1   net bill
    (21)  J^h_2   discomfort
    (22)  f^cost_i   DAILY procurement
    (23)  f^rel_i    Delta on Z only, not on nu
    (24)  f^wel_i
    (25)  F_i = theta1 f^cost + theta2 f^rel + theta3 f^wel
    (26)  F = sum_i F_i
    (27)  min F over F = {C1..C7}
    (28)  S_i(t)  substituted, not a constraint
    (29)  C1 supply adequacy
    (30)  C2 storage bounds
    (31)  C3 reserve floor
    (32)  C4 tie-line and export
    (33)  C5 resource bounds        -> variable bounds
    (34)  C6 household bounds       -> variable bounds
    (35)  C7 service quality

Units: kW, kWh, Tk, Tk/kWh. Delta = 0.5 h.

-----------------------------------------------------------------------------
Design decisions, and why
-----------------------------------------------------------------------------

**Daily totals add no variables.** Eq. (18) defines G_{i,d}, B_{i,d} and
T_{ij,d} as sums of the per-slot rates times Delta. They are substituted into
(22) exactly as the state of charge (28) is substituted into C2/C3. So the
objective coefficient on G_i(t) is simply c_{i,d} * Delta. Physics stays per
slot; only the bookkeeping is per day.

**Eq. (23) treats nu and Z differently, on purpose.** Delta attaches only to
Z^c_i(t), which is a power in kW. nu_i(t) is already an energy in kWh, because
C3 compares it against the reserve floor R_i in kWh. Multiplying both by Delta
- as an earlier version of this code did - understates the reserve penalty by
a factor of 1/Delta = 2.

**x exists only on peak slots.** C6 forces x_{h,t} = 0 off-peak via the
indicator, so off-peak x variables are never created rather than created and
pinned. At 6 peak slots against 48 this removes ~87% of the x block.

**The dual of C1 needs a Delta.** pi_i(t) is defined in section 1.7 as Tk/kWh,
but C1 is written in kW, so the solver's raw marginal on that row carries
Tk/kW. Relaxing the row by 1 kW over a slot of Delta hours is Delta kWh, so

    pi_i(t) = -marginal / Delta          [Tk/kWh]

The sign flips because the row is stored as a <= after moving the supply terms
to the left. At Delta = 1 h - the section 8 instance - the division is
invisible, which is why the formulation writes kappa*pi/theta_3 without it.
At Delta = 0.5 h it is a factor of two, and every Proposition 2 threshold
reported by this module uses the corrected value.

**Storage chains across days.** C2/C3 accumulate from S^0_i over tau <= t. A
multi-day run solves day by day on a receding basis (as the formulation
describes) and carries S_i(end of day d) into S^0_i for day d+1. Within a day
the solve is simultaneous, so Corollary 1 is untouched.

**Per-substation or coupled.** Remark 1 states that S = {i}, L = empty gives a
complete single-substation model with all propositions holding verbatim.
`solve_day(..., couple=False)` does exactly that, which is useful when one
substation's inputs are under question.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np
from scipy import sparse
from scipy.optimize import linprog

from . import config as cfg
from . import load_model as lm
from . import population as pop
from . import tariff as T
from . import peak_window as pw

CLASSES = ("critical", "shiftable", "curtailable")


# =============================================================================
# Variable index bookkeeping
# =============================================================================

class VarIndex:
    """Maps every decision variable of Eq. (17) to a column of the LP."""

    def __init__(self):
        self.n = 0
        self.slices: dict = {}
        self.meta: list = []

    def add(self, key, count: int, **meta):
        s = slice(self.n, self.n + count)
        self.slices[key] = s
        self.meta.extend([{"key": key, "j": j, **meta} for j in range(count)])
        self.n += count
        return s

    def __getitem__(self, key):
        return self.slices[key]


@dataclass
class DaySolution:
    date: str
    substations: list
    status: str
    success: bool
    objective_tk: float
    F_by_substation: dict
    cost_terms: dict
    duals: dict                      # pi_i(t), Tk/kWh (C1)
    duals_c4: dict                   # pi4_i(t), Tk/kWh (C4 export/capacity)
    x: dict                          # (consumer_id, slot) -> x*
    y: dict                          # (consumer_id, slot) -> y*
    network: dict                    # per substation, per slot arrays
    soc_end_kwh: dict
    c1_slack_kw: dict
    c7_binding: list
    ir_binding: list = field(default_factory=list)
    thresholds: list = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)


# =============================================================================
# Building blocks
# =============================================================================

def _consumer_blocks(consumers, schedules, cat, theta, peak_slots):
    """
    Per-consumer weighted class profiles in kW, plus the peak/off-peak slot
    sets. Everything the objective and C1/C4/C7 need about demand.
    """
    nT = cfg.SLOTS_PER_DAY
    off = [t for t in range(nT) if t not in set(peak_slots)]
    blocks = {}
    for h in consumers:
        prof = lm.consumer_profile(h, schedules, cat, theta)
        w = float(h.w)
        blocks[h.consumer_id] = {
            "consumer": h,
            "cr": prof["critical"] * w,
            "sh": prof["shiftable"] * w,
            "cu": prof["curtailable"] * w,
            "base": (prof["critical"] + prof["shiftable"]
                     + prof["curtailable"]) * w,
        }
    return blocks, list(peak_slots), off


def _price_vector(consumer, substation, surcharge_tk_per_kwh: float,
                  monthly_kwh: float | None) -> np.ndarray:
    """
    pi_h(t) = (pi^ret_h + varpi_{i,m}) mu_i(t), Eq. (20). Tk/kWh.

    Issue A: this is a BILLING price, so mu(t) must use the OFFICIAL BERC
    ToU window (17:00-23:00), never the adaptive DR-activation window. The
    two are different things with different owners: BERC fixes when a ToU
    class is actually billed peak/off-peak; the adaptive window (derived per
    substation from congestion history) only says when x_{h,t} may be
    non-zero for demand-response ACTIVATION, in lp.build_day. Passing
    `adaptive=False` here is what selects the official window - has_tou=False
    classes (LT-A, LT-D1) are unaffected either way (mu=1.0 always, see
    tariff.mu_profile).
    """
    code = cfg.CONSUMER_ARCHETYPES[consumer.archetype]["tariff_class"]
    base = T.retail_rate(code, monthly_kwh)
    mu = T.mu_profile(code, substation=substation, adaptive=False)
    return (base + surcharge_tk_per_kwh) * np.asarray(mu)


# =============================================================================
# The programme
# =============================================================================

def build_day(date, substations, populations, schedules, cat, theta,
              kappa: dict, s_stress: dict, soc0: dict,
              surcharge: dict | None = None, couple: bool = True,
              theta_w: tuple | None = None,
              individual_rationality: bool | None = None,
              terminal_target: dict | None = None,
              no_dr: bool = False):
    """
    Assemble the LP for one day. Returns (c, A_ub, b_ub, bounds, idx, ctx).

    `kappa[i]` is kappa_{i,d} from Eq. (16); `s_stress[i]` is s_{i,d} from
    Eq. (15); `soc0[i]` is S^0_i in kWh for this day.

    `individual_rationality` (Issue I), when True, adds one row per household
    per day: lambda_h*(J1(x,y)-J1(0,0)) + (1-lambda_h)*J2(x,y) <= 0. x=y=0 is
    always feasible.

    `terminal_target` (Issue 10, optional): {substation: kWh}. Adds
    S_i(end of day) >= target on the LAST slot only. The caller passes this
    only for whichever date it considers the end of the run.

    `no_dr` (Round-3 item C13, default False - no change to any existing
    result): forces x=y=0 for every household by zeroing their upper
    bounds, a BOUNDS-only override that touches neither the objective nor
    any config default. Used for the load-shedding counterfactual - with
    no shift/curtail available at all, whatever the substation cannot
    serve from G/battery/tie-line must appear as Z (unserved load) instead.
    """
    th1, th2, th3 = theta_w or (cfg.THETA_COST, cfg.THETA_RELIABILITY,
                                cfg.THETA_WELFARE)
    ir_on = cfg.INDIVIDUAL_RATIONALITY if individual_rationality is None \
        else individual_rationality
    surcharge = surcharge or {i: 0.0 for i in substations}
    nT, D = cfg.SLOTS_PER_DAY, cfg.DELTA_H
    idx = VarIndex()
    ctx = {"peak": {}, "off": {}, "blocks": {}, "prices": {}, "jx": {}, "jy": {}}

    # ---- variables ---------------------------------------------------------
    for i in substations:
        tpk = list(pw.get_peak_slots(i))
        blocks, tpk, off = _consumer_blocks(populations[i], schedules, cat,
                                            theta, tpk)
        ctx["peak"][i], ctx["off"][i], ctx["blocks"][i] = tpk, off, blocks
        for cid, b in blocks.items():
            h = b["consumer"]
            monthly = float(b["base"].sum() * D * 30.0 / max(h.w, 1e-9))
            ctx["prices"][cid] = _price_vector(h, i, surcharge[i], monthly)
            idx.add(("x", cid), len(tpk), substation=i, consumer=cid, kind="x")
            idx.add(("y", cid), nT, substation=i, consumer=cid, kind="y")
    for i in substations:
        for k in ("G", "Bch", "Bdis", "nu"):
            idx.add((k, i), nT, substation=i, kind=k)
        for c in CLASSES:
            idx.add(("Z", i, c), nT, substation=i, kind="Z", cls=c)
    links = [(l["from"], l["to"], l) for l in cfg.TIE_LINES
             if couple and l["from"] in substations and l["to"] in substations]
    for a, b_, spec in links:
        idx.add(("T", a, b_), nT, substation=a, to=b_, kind="T")

    n = idx.n
    c = np.zeros(n)

    # ---- objective ---------------------------------------------------------
    terms = {i: {"cost": 0.0, "rel": 0.0, "wel": 0.0} for i in substations}
    const_wel = {i: 0.0 for i in substations}

    for i in substations:
        spec = cfg.SUBSTATIONS[i]
        # (22) f^cost_i : daily totals, substituted -> price * Delta per slot
        c[idx[("G", i)]] += th1 * spec["c_producer_tk_per_kwh"] * D
        c[idx[("Bch", i)]] += th1 * cfg.C_BATTERY_TK_PER_KWH * D
        c[idx[("Bdis", i)]] += th1 * cfg.C_BATTERY_TK_PER_KWH * D
        # (23) f^rel_i : Delta on Z only; nu is already kWh
        c[idx[("nu", i)]] += th2 * cfg.PI_RESERVE_TK_PER_KWH
        for cls in CLASSES:
            c[idx[("Z", i, cls)]] += th2 * cfg.VOLL_TK_PER_KWH[cls] * D

    for a, b_, spec in links:
        c[idx[("T", a, b_)]] += th1 * spec["price_tk_per_kwh"] * D

    for i in substations:
        tpk, off = ctx["peak"][i], ctx["off"][i]
        n_off = max(len(off), 1)
        for cid, blk in ctx["blocks"][i].items():
            h = blk["consumer"]
            p = ctx["prices"][cid]
            lam, g_sh, g_cu = h.lambda_h, h.gamma_sh_h, h.gamma_cu_h
            rho = getattr(h, "rho_h", cfg.RHO_REBATE_TK_PER_KWH)
            s = float(s_stress[i])

            # J^h_1 constant part: baseline bill (does not change argmin, but
            # is carried so F* is reportable and the welfare surplus of
            # section 3.2 can be formed).
            const_wel[i] += lam * float((blk["base"] * p).sum() * D)

            # x_{h,tau}: bill effect = -L^sh(tau)*pi(tau) + recovery spread
            jx = np.zeros(len(tpk))
            for a_, tau in enumerate(tpk):
                bill = -blk["sh"][tau] * p[tau] * D
                bill += blk["sh"][tau] / n_off * float(p[off].sum()) * D
                disc = g_sh * blk["sh"][tau] * D          # (21)
                jx[a_] = lam * bill + (1.0 - lam) * disc
            c[idx[("x", cid)]] += th3 * jx

            # y_{h,t}: bill effect + rebate, against discomfort
            jy = (lam * (-blk["cu"] * p * D - rho * blk["cu"] * D * s)
                  + (1.0 - lam) * g_cu * blk["cu"] * D)
            c[idx[("y", cid)]] += th3 * jy

            # Stored (th3-unscaled) for the IR constraint below and for
            # threshold_table's IR-dual term - both need the household's own
            # lambda_h*DeltaJ1 + (1-lambda_h)*J2 per unit of x/y, which is
            # exactly jx/jy as computed here.
            ctx["jx"][cid] = jx
            ctx["jy"][cid] = jy

    # ---- constraints -------------------------------------------------------
    rows, rhs, rmeta = [], [], []

    def _row():
        return np.zeros(n)

    for i in substations:
        spec = cfg.SUBSTATIONS[i]
        k_i = float(kappa[i])
        tpk, off = ctx["peak"][i], ctx["off"][i]
        n_off = max(len(off), 1)
        blocks = ctx["blocks"][i]
        lam0 = np.zeros(nT)
        for blk in blocks.values():
            lam0 += blk["base"]

        # demand coefficients: d(load at t)/d(var)
        def load_coeffs(t):
            out = {}
            for cid, blk in blocks.items():
                cx = np.zeros(len(tpk))
                for a_, tau in enumerate(tpk):
                    if tau == t:
                        cx[a_] -= blk["sh"][tau]
                    if t in off:
                        cx[a_] += blk["sh"][tau] / n_off
                if cx.any():
                    out[("x", cid)] = cx
                out[("y", cid)] = -blk["cu"][t]
            return out

        for t in range(nT):
            lc = load_coeffs(t)

            # --- C1 (29) --------------------------------------------------
            r = _row()
            r[idx[("G", i)].start + t] = -1.0
            r[idx[("Bdis", i)].start + t] = -cfg.ETA_DIS
            r[idx[("Bch", i)].start + t] = 1.0
            for cls in CLASSES:
                r[idx[("Z", i, cls)].start + t] = -1.0
            for a_, b_, spec_l in links:
                if b_ == i:
                    r[idx[("T", a_, b_)].start + t] = -(1.0 - spec_l["loss_fraction"])
                if a_ == i:
                    r[idx[("T", a_, b_)].start + t] = 1.0
            for key, val in lc.items():
                sl = idx[key]
                if key[0] == "x":
                    r[sl] += k_i * val
                else:
                    r[sl.start + t] += k_i * val
            rows.append(sparse.csr_matrix(r))
            rhs.append(-k_i * lam0[t])
            rmeta.append({"type": "C1", "substation": i, "slot": t})

            # --- C4 (32) export limit -------------------------------------
            # Follow-up (Round 3): unserved load Z_c(t) must relax this row
            # exactly as it relaxes C1 - shed load is not drawn through the
            # transformer either, so it cannot still count against P^max.
            # Its absence made a P^max scenario INFEASIBLE (HiGHS status 8)
            # whenever the unavoidable load exceeded the scaled cap, because
            # the LP had no way to shed load against C4 even though C1 could
            # already shed against G^max; a real operator sheds load instead.
            r = _row()
            for a_, b_, _ in links:
                if a_ == i:
                    r[idx[("T", a_, b_)].start + t] = 1.0
            r[idx[("Bdis", i)].start + t] = -cfg.ETA_DIS
            for cls in CLASSES:
                r[idx[("Z", i, cls)].start + t] = -1.0
            for key, val in lc.items():
                sl = idx[key]
                if key[0] == "x":
                    r[sl] += k_i * val
                else:
                    r[sl.start + t] += k_i * val
            rows.append(sparse.csr_matrix(r))
            rhs.append(spec["p_max_kw"] - k_i * lam0[t])
            rmeta.append({"type": "C4", "substation": i, "slot": t})

            # --- C2 (30) storage bounds, cumulative -----------------------
            cum = _row()
            cum[idx[("Bch", i)].start:idx[("Bch", i)].start + t + 1] = cfg.ETA_CH * D
            cum[idx[("Bdis", i)].start:idx[("Bdis", i)].start + t + 1] = -D
            rows.append(sparse.csr_matrix(cum))
            rhs.append(spec["e_max_kwh"] - soc0[i])
            rmeta.append({"type": "C2u", "substation": i, "slot": t})

            rows.append(sparse.csr_matrix(-cum))
            rhs.append(soc0[i])
            rmeta.append({"type": "C2l", "substation": i, "slot": t})

            # --- C3 (31) reserve floor ------------------------------------
            r3 = -cum.copy()
            r3[idx[("nu", i)].start + t] = -1.0
            rows.append(sparse.csr_matrix(r3))
            rhs.append(soc0[i] - spec["reserve_kwh"])
            rmeta.append({"type": "C3", "substation": i, "slot": t})

            # --- TERMINAL_SOC (Issue 10, optional), last slot only --------
            # S_i(end of day) >= terminal_target[i]. Only added on whichever
            # day the caller marks as the horizon's last (06_lp_optimiser.py
            # passes terminal_target only for that date); every other day
            # is unaffected regardless of TERMINAL_SOC.
            if t == nT - 1 and terminal_target and i in terminal_target:
                r_term = -cum.copy()
                rows.append(sparse.csr_matrix(r_term))
                rhs.append(soc0[i] - float(terminal_target[i]))
                rmeta.append({"type": "TerminalSoC", "substation": i, "slot": t})

        # --- C7 (35) service quality, one row per consumer ----------------
        for cid, blk in blocks.items():
            r = _row()
            r[idx[("x", cid)]] = cfg.QOE_WEIGHT_SHIFT * blk["sh"][tpk] * D
            r[idx[("y", cid)]] = cfg.QOE_WEIGHT_CURTAIL * blk["cu"] * D
            e_base = float(blk["base"].sum() * D)
            rows.append(sparse.csr_matrix(r))
            rhs.append((1.0 - cfg.QOE_MIN) * e_base)
            rmeta.append({"type": "C7", "substation": i, "consumer": cid})

        # --- IR (Issue I), optional: individual rationality, one row per
        # consumer. lambda_h*(J1(x,y)-J1(0,0)) + (1-lambda_h)*J2(x,y) <= 0.
        # The row coefficients are literally ctx["jx"]/ctx["jy"] (the same
        # th3-unscaled per-unit welfare contribution already used to build
        # the objective for this household) - x=y=0 gives 0 <= 0, always
        # feasible.
        if ir_on:
            for cid, blk in blocks.items():
                r = _row()
                r[idx[("x", cid)]] = ctx["jx"][cid]
                r[idx[("y", cid)]] = ctx["jy"][cid]
                rows.append(sparse.csr_matrix(r))
                rhs.append(0.0)
                rmeta.append({"type": "IR", "substation": i, "consumer": cid})

    A_ub = sparse.vstack(rows, format="csr")
    b_ub = np.asarray(rhs, dtype=float)

    # ---- bounds: C5 (33), C6 (34) -----------------------------------------
    lo = np.zeros(n)
    hi = np.full(n, np.inf)
    scope = cfg.CURTAIL_SCOPE
    for i in substations:
        spec = cfg.SUBSTATIONS[i]
        hi[idx[("G", i)]] = spec["g_max_kw"]
        hi[idx[("Bch", i)]] = spec["b_max_kw"]
        hi[idx[("Bdis", i)]] = spec["b_max_kw"]
        tpk_i = ctx["peak"][i]
        stressed_today = float(s_stress[i]) >= 1
        for cid, blk in ctx["blocks"][i].items():
            hi[idx[("x", cid)]] = 0.0 if no_dr else 1.0
            cap = getattr(blk["consumer"], "y_max_h", cfg.Y_MAX_DEFAULT)
            y_sl = idx[("y", cid)]
            if no_dr:
                hi[y_sl] = 0.0
                continue
            if scope == "dr_window":
                # Follow-up (Issue: curtailment scope): restrict y to the
                # substation's own DR-activation window, so curtailment is
                # genuinely peak-time-scoped rather than all-day. x already
                # only exists on this window (unchanged); y previously had
                # no such restriction at all.
                y_hi = np.zeros(nT)
                for t in tpk_i:
                    y_hi[t] = cap
                hi[y_sl] = y_hi
            elif scope == "stress_days":
                # y allowed at any slot, but ONLY on a day this substation's
                # own s_stress flag is 1 - curtailment becomes event-scoped.
                hi[y_sl] = cap if stressed_today else 0.0
            else:   # "all" - unchanged, default
                hi[y_sl] = cap
    for a_, b_, spec_l in links:
        hi[idx[("T", a_, b_)]] = spec_l["t_max_kw"]

    ctx.update({"rmeta": rmeta, "links": links, "const_wel": const_wel,
                "theta": (th1, th2, th3), "kappa": kappa,
                "s_stress": s_stress, "soc0": soc0, "ir_on": ir_on})
    return c, A_ub, b_ub, list(zip(lo, hi)), idx, ctx


def solve_day(date, substations, populations, schedules, cat, theta,
              kappa, s_stress, soc0, surcharge=None, couple=True,
              theta_w=None, individual_rationality=None,
              terminal_target=None, no_dr=False) -> DaySolution:
    """Assemble, solve, and unpack one day."""
    c, A_ub, b_ub, bounds, idx, ctx = build_day(
        date, substations, populations, schedules, cat, theta, kappa,
        s_stress, soc0, surcharge, couple, theta_w, individual_rationality,
        terminal_target, no_dr)

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds,
                  method="highs", options=cfg.LP_OPTIONS)

    sol = DaySolution(
        date=str(date), substations=list(substations), status=res.message,
        success=bool(res.success), objective_tk=np.nan,
        F_by_substation={}, cost_terms={}, duals={}, duals_c4={}, x={}, y={},
        network={}, soc_end_kwh={}, c1_slack_kw={}, c7_binding=[], ir_binding=[])
    if not res.success:
        return sol

    v = res.x
    nT, D = cfg.SLOTS_PER_DAY, cfg.DELTA_H
    marg = np.asarray(res.ineqlin.marginals, dtype=float)

    # --- duals: pi_i(t) in Tk/kWh, sign- and Delta-corrected ---------------
    for i in substations:
        pi = np.zeros(nT)
        pi4 = np.zeros(nT)
        slack = np.zeros(nT)
        for r, m in enumerate(ctx["rmeta"]):
            if m["type"] == "C1" and m["substation"] == i:
                pi[m["slot"]] = -marg[r] / D
                slack[m["slot"]] = float(b_ub[r] - A_ub[r].dot(v)[0])
            # Follow-up (Issue G/6): C4's row has the IDENTICAL load-term
            # structure as C1 (same k_i * load_coeffs(t)), so its dual is a
            # second, genuine shadow price on the SAME x/y decisions -
            # export/capacity headroom, not supply adequacy. Extracted with
            # the same sign/Delta correction as C1's pi. It is exactly zero
            # whenever C4 is slack (the base-case default: G^max is never
            # binding), so this changes nothing in results where it doesn't
            # bind, and becomes nonzero precisely in a scarcity scenario -
            # where the old threshold_table silently omitted it.
            if m["type"] == "C4" and m["substation"] == i:
                pi4[m["slot"]] = -marg[r] / D
        sol.duals[i] = pi
        sol.duals_c4[i] = pi4
        sol.c1_slack_kw[i] = slack

    # --- decisions ----------------------------------------------------------
    for i in substations:
        tpk = ctx["peak"][i]
        for cid in ctx["blocks"][i]:
            xv = v[idx[("x", cid)]]
            yv = v[idx[("y", cid)]]
            for a_, t in enumerate(tpk):
                if xv[a_] > 1e-9:
                    sol.x[(cid, t)] = float(xv[a_])
            for t in range(nT):
                if yv[t] > 1e-9:
                    sol.y[(cid, t)] = float(yv[t])

        net = {k: v[idx[(k, i)]].copy() for k in ("G", "Bch", "Bdis", "nu")}
        for cls in CLASSES:
            net[f"Z_{cls}"] = v[idx[("Z", i, cls)]].copy()
        net["T_out"] = np.zeros(nT)
        net["T_in"] = np.zeros(nT)
        for a_, b_, spec_l in ctx["links"]:
            if a_ == i:
                net["T_out"] += v[idx[("T", a_, b_)]]
            if b_ == i:
                net["T_in"] += (1 - spec_l["loss_fraction"]) * v[idx[("T", a_, b_)]]
        soc = ctx["soc0"][i] + np.cumsum(
            cfg.ETA_CH * net["Bch"] - net["Bdis"]) * D
        net["SoC"] = soc
        sol.network[i] = net
        sol.soc_end_kwh[i] = float(soc[-1])

        # (18) daily totals, substituted
        spec = cfg.SUBSTATIONS[i]
        g_d = float(net["G"].sum() * D)
        b_d = float((net["Bch"] + net["Bdis"]).sum() * D)
        t_d = float(net["T_out"].sum() * D)
        # Minor fix: the tie-line cost must use EACH link's own price, not
        # links[0]'s price applied to every substation's total export. With
        # only two same-price links this was numerically invisible; it is
        # structurally wrong the moment link prices differ by direction.
        t_cost = sum(spec_l["price_tk_per_kwh"]
                    * float(v[idx[("T", a_, b_)]].sum() * D)
                    for a_, b_, spec_l in ctx["links"] if a_ == i)
        f_cost = (spec["c_producer_tk_per_kwh"] * g_d
                  + cfg.C_BATTERY_TK_PER_KWH * b_d
                  + t_cost)
        f_rel = float(cfg.PI_RESERVE_TK_PER_KWH * net["nu"].sum()
                      + sum(cfg.VOLL_TK_PER_KWH[cl] * net[f"Z_{cl}"].sum() * D
                            for cl in CLASSES))
        sol.cost_terms[i] = {
            "G_id_kwh": g_d, "B_id_kwh": b_d, "T_ijd_kwh": t_d,
            "f_cost_tk": f_cost, "f_rel_tk": f_rel,
        }

    # --- welfare and F_i ----------------------------------------------------
    th1, th2, th3 = ctx["theta"]
    for i in substations:
        wel = ctx["const_wel"][i]
        for cid in ctx["blocks"][i]:
            wel += float(c[idx[("x", cid)]].dot(v[idx[("x", cid)]]) / th3)
            wel += float(c[idx[("y", cid)]].dot(v[idx[("y", cid)]]) / th3)
        ct = sol.cost_terms[i]
        ct["f_wel_tk"] = wel
        sol.F_by_substation[i] = (th1 * ct["f_cost_tk"] + th2 * ct["f_rel_tk"]
                                  + th3 * wel)

    # F* reported INCLUDING the constant part of J^h_1. The LP objective omits
    # it (a constant cannot change the argmin) but section 3.2 forms the
    # household surplus against J^h_1(0,0), so the constant has to be carried
    # for F to be comparable across scenarios and to equal sum_i F_i.
    sol.objective_tk = float(sum(sol.F_by_substation.values()))
    sol.diagnostics_lp_fun = float(res.fun)

    # --- C7 binding rows ----------------------------------------------------
    for r, m in enumerate(ctx["rmeta"]):
        if m["type"] == "C7":
            s_ = float(b_ub[r] - A_ub[r].dot(v)[0])
            if s_ < 1e-6:
                sol.c7_binding.append({"consumer": m["consumer"],
                                       "substation": m["substation"],
                                       "dual": float(-marg[r])})

    ctx["c7_duals"] = {b["consumer"]: b["dual"] for b in sol.c7_binding}

    # --- IR binding rows (Issue I) ------------------------------------------
    if ctx.get("ir_on"):
        for r, m in enumerate(ctx["rmeta"]):
            if m["type"] == "IR":
                s_ = float(b_ub[r] - A_ub[r].dot(v)[0])
                if s_ < 1e-6:
                    sol.ir_binding.append({"consumer": m["consumer"],
                                           "substation": m["substation"],
                                           "dual": float(-marg[r])})
    ctx["ir_duals"] = {b["consumer"]: b["dual"] for b in sol.ir_binding}

    sol.thresholds = threshold_table(sol, ctx, idx)
    sol.diagnostics = {
        "n_variables": int(idx.n), "n_constraints": int(A_ub.shape[0]),
        "n_x_active": len(sol.x), "n_y_active": len(sol.y),
        "max_c1_slack_kw": max(float(np.abs(s).max()) for s in sol.c1_slack_kw.values()),
        "n_c7_binding": len(sol.c7_binding),
        "n_ir_binding": len(sol.ir_binding),
        "coupled": bool(ctx["links"]),
    }
    return sol


# =============================================================================
# Proposition 2 decomposition - stored, not verified
# =============================================================================

def threshold_table(sol: DaySolution, ctx, idx) -> list:
    """
    The three (four, with QoE) terms of Eq. (36)/(37) for every acted-upon
    (h, t).

    This is NOT proposition verification - it stores the decomposition so the
    knowledge graph can explain an action as

        private benefit + system value  >  discomfort ( + QoE scarcity)

    without re-running the solver. All terms in Tk/kWh.

    Issue G fix. Both actions now use the ACTUAL slot price `p[t]` from
    `ctx["prices"][cid]` (which is officially-windowed - Issue A), never the
    family ratio mu_pk/mu_off:

      shift    priv = lam * ( p[tau] - mean_{t in off} p[t] )
               exactly the per-unit bill effect build_day's `jx` coefficient
               already computes (Eq. 20's -p[tau] + recovery-spread term),
               so this is the LP's own number, not an approximation of it.
               `off` here is the ADAPTIVE off-peak complement (Eq. 5/14's
               recovery window - a load-model fact, unrelated to Issue A),
               but `p` itself is priced under the OFFICIAL BERC window, so
               "mean off-peak price" is the officially-correct one.

      curtail  priv = lam * (p[t] + rho*s)   (unchanged - already correct:
               curtailing genuinely avoids buying that kWh at whatever price
               applies at t, plus the stress rebate on a stressed day)

               lambda_star (Eq. 38) corrected to
                   (gamma_cu_h + qoe - syst) / (p[t] + rho*s + gamma_cu_h)
               using the SLOT price p[t] and rho*s (matching priv exactly,
               and including the QoE term in the numerator), not the global
               mu_pk and a bare, unstressed rho.
    """
    out = []
    th3 = ctx["theta"][2]
    D = cfg.DELTA_H
    c7d = ctx.get("c7_duals", {})
    ird = ctx.get("ir_duals", {})
    for i in sol.substations:
        pi = sol.duals[i]
        pi4 = sol.duals_c4.get(i, np.zeros(cfg.SLOTS_PER_DAY))
        tpk, off = ctx["peak"][i], ctx["off"][i]
        pi_off = float(pi[off].mean()) if off else 0.0
        pi4_off = float(pi4[off].mean()) if off else 0.0
        k_i = float(ctx["kappa"][i])
        s = float(ctx["s_stress"][i])
        for cid, blk in ctx["blocks"][i].items():
            h = blk["consumer"]
            p = ctx["prices"][cid]
            p_off_mean = float(p[off].mean()) if off else 0.0
            lam = h.lambda_h
            rho = getattr(h, "rho_h", cfg.RHO_REBATE_TK_PER_KWH)
            # Issue I: IR's dual (0 unless the IR constraint is active AND
            # binding for this household) is a scarcity price on the SAME
            # jx/jy row already used to build the objective - see build_day.
            ir_dual = ird.get(cid, 0.0)
            jx_arr = ctx["jx"].get(cid)
            jy_arr = ctx["jy"].get(cid)

            for a_, t in enumerate(tpk):
                xv = sol.x.get((cid, t))
                if xv is None:
                    continue
                priv = lam * (float(p[t]) - p_off_mean)
                # Follow-up (Issue 6): C4's dual added in - it is exactly
                # zero whenever C4 is slack (G^max unbound, the base-arm
                # default), so this changes nothing there; it becomes
                # nonzero, and belongs here, once export/capacity binds.
                syst = k_i * (pi[t] - pi_off) / th3 + k_i * (pi4[t] - pi4_off) / th3
                disc = (1 - lam) * h.gamma_sh_h
                # C7 is NOT a box constraint, so when it binds its dual enters
                # the reduced cost and the threshold gains a fourth term.
                # See docs/qoe_definition.md section 4.
                qoe = cfg.QOE_WEIGHT_SHIFT * c7d.get(cid, 0.0) / th3
                # Round-3 follow-up: the IR row's own coefficient on x_{h,tau}
                # is jx_arr[a_] itself (not a fixed weight like C7's
                # QOE_WEIGHT_SHIFT), so normalising it to the same per-kWh
                # units as priv/disc/qoe requires dividing by this SAME
                # slot's own sh[tau]*D energy factor - exactly how C7's own
                # QOE_WEIGHT_SHIFT*sh[tau]*D row coefficient collapses to the
                # constant QOE_WEIGHT_SHIFT once divided through by sh[tau]*D.
                # Missing that division left ir_term ~sh[tau]*D times too
                # large, which is why interior margins for the --ir arm were
                # off by hundreds to thousands (check 6 FAILED for --ir until
                # this fix; caught by running acceptance check 6 on that arm
                # specifically per this round's own instruction).
                sh_t = float(blk["sh"][t]) * D
                ir_term = (ir_dual * jx_arr[a_] / sh_t / th3
                          if jx_arr is not None and sh_t > 1e-12 else 0.0)
                out.append({
                    "qoe_tk_per_kwh": qoe, "c7_binding": cid in c7d,
                    "ir_tk_per_kwh": ir_term, "ir_binding": cid in ird,
                    "consumer": cid, "substation": i, "slot": t,
                    "action": "shift", "value": xv,
                    "private_tk_per_kwh": priv, "system_tk_per_kwh": syst,
                    "discomfort_tk_per_kwh": disc,
                    "margin": priv + syst - qoe - ir_term - disc,
                    "pi_i_t": float(pi[t]), "pi_off": pi_off, "kappa": k_i,
                    "price_tau_tk_per_kwh": float(p[t]),
                    "price_off_mean_tk_per_kwh": p_off_mean,
                    "energy_kwh": xv * float(blk["sh"][t]) * D,
                })
            for t in range(cfg.SLOTS_PER_DAY):
                yv = sol.y.get((cid, t))
                if yv is None:
                    continue
                p_t = float(p[t])
                priv = lam * (p_t + rho * s)
                syst = k_i * pi[t] / th3 + k_i * pi4[t] / th3   # + C4 dual, follow-up
                disc = (1 - lam) * h.gamma_cu_h
                qoe = cfg.QOE_WEIGHT_CURTAIL * c7d.get(cid, 0.0) / th3
                # Round-3 follow-up: see the shift-side comment above - same
                # fix, normalising by this slot's own cu[t]*D.
                cu_t = float(blk["cu"][t]) * D
                ir_term = (ir_dual * jy_arr[t] / cu_t / th3
                          if jy_arr is not None and cu_t > 1e-12 else 0.0)
                denom = p_t + rho * s + h.gamma_cu_h
                out.append({
                    "qoe_tk_per_kwh": qoe, "c7_binding": cid in c7d,
                    "ir_tk_per_kwh": ir_term, "ir_binding": cid in ird,
                    "consumer": cid, "substation": i, "slot": t,
                    "action": "curtail", "value": yv,
                    "private_tk_per_kwh": priv, "system_tk_per_kwh": syst,
                    "discomfort_tk_per_kwh": disc,
                    "margin": priv + syst - qoe - ir_term - disc,
                    "pi_i_t": float(pi[t]), "kappa": k_i,
                    "lambda_h": lam, "y_max_h": float(h.y_max_h),
                    "at_cap": bool(yv >= float(h.y_max_h) - 1e-6),
                    "price_tau_tk_per_kwh": p_t,
                    "lambda_star": ((h.gamma_cu_h + qoe - syst) / denom
                                    if denom > 0 else np.nan),   # Eq. (38)
                    "energy_kwh": yv * float(blk["cu"][t]) * D,
                })
    return out


# =============================================================================
# Stage G - Eqs. (40)-(43)
# =============================================================================

def stage_g(sol: DaySolution, post_response_kw: dict,
            min_retail: dict) -> dict:
    """
    Cost allocation. Eq. (40) net import cost, (41) energy served, (42)
    surcharge, (43) cap at the lowest retail tariff among the households at i.
    """
    out = {}
    D = cfg.DELTA_H
    for i in sol.substations:
        net = sol.network[i]
        spec = cfg.SUBSTATIONS[i]
        c_id = spec["c_producer_tk_per_kwh"]
        pi_ji = next((l["price_tk_per_kwh"] for l in cfg.TIE_LINES
                      if l["to"] == i), 0.0)
        # (40): only IMPORTS (j -> i) are charged
        imported_kwh = float(net["T_in"].sum() * D)
        delta_c = (pi_ji - c_id) * imported_kwh
        # (41): post-response delivered energy, less unserved. Explicitly NOT
        # scaled by kappa - "that factor is a robustness margin applied to a
        # constraint, not metered energy".
        served = float(post_response_kw[i].sum() * D
                       - sum(net[f"Z_{c}"].sum() for c in CLASSES) * D)
        varpi = (cfg.ZETA_PASSTHROUGH * delta_c / served) if served > 0 else 0.0
        cap = min_retail[i]                       # (43)
        out[i] = {"delta_C_tk": delta_c, "imported_kwh": imported_kwh,
                  "E_srv_kwh": served, "varpi_uncapped": varpi,
                  "varpi_tk_per_kwh": float(min(varpi, cap)),
                  "cap_tk_per_kwh": cap, "cap_binding": varpi > cap}
    return out
