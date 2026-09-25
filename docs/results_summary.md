# GenTwin-SG — Results Summary (Supervisor Meeting)

All numbers below are read directly from tag files under `outputs/` and
`outputs/billing/`; the tag is given under each table so every number is
independently reproducible without re-solving anything. Full derivations,
caveats and the audit trail (bugs found and fixed) are in
`docs/thesis_changes.md`. Nothing here required a new LP solve as of this
document's date except where the tag itself names a new arm.

## 1. Main arm comparison (full 30-day period, 2026-06-14 → 2026-07-13)

Tags: `_period` (base), `_period_dr_window`, `_period_rho0`, `_period_ir`,
`_period_drw_ir`, `_period_scarcity` (0.55/0.62).

| arm | curtail kWh | shift kWh | n participating (of 800 reps) | %gained, ALL connections, unweighted | %gained, ALL connections, λ-weighted | substation peak reduction % (Dhanmondi / Kalyanpur) | check 6 |
|---|---|---|---|---|---|---|---|
| base | 11,389,227 | 54,328 | 460 | 1.2% | 14.8% | 1.47 / 5.76 | PASS |
| dr_window | 3,088,306 | 135,290 | 458 | 2.2% | 15.3% | 3.15 / 7.48 | PASS |
| rho0 (current policy, rho=0) | 11,078,285 | 60,048 | 432 | 0.2% | 12.1% | 0.85 / 5.68 | PASS |
| ir | 5,171,419 | 93,995 | 207 | 2.0% | 17.6% | 0.92 / 4.82 | PASS |
| drw_ir (candidate main arm) | 1,324,212 | 135,290 | 189 | 3.0% | 17.7% | 3.15 / 7.44 | PASS |
| scarcity (compulsory, 0.55/0.62) | 12,746,020 | 1,155,314 | 750 | 1.2% | 14.8% | 15.27 / 23.57 | PASS |

**Reading:** kWh columns are already population-scale (~348,661
connections), not per-representative. `%gained` is share of ALL
connections, not just participants. Curtailment dominates shifting
everywhere except `dr_window`/`drw_ir` (residential has no time-of-use
exposure, so shifting has no private financial reason to occur outside a
scoped window). None of these arms show more than 3% of ALL connections
clearing the unweighted saving>discomfort bar.

## 2. Social efficiency — is curtailment worth it to the SYSTEM, or only to the household?

Test: `pi_i(t) + pi4_i(t) - gamma_{cu,h} > 0` (curtail) /
`(pi(tau)-mean(pi_off)) - gamma_{sh,h} > 0` (shift), unweighted.

| arm | share of curtailed kWh passing | share of shifted kWh passing |
|---|---|---|
| base / dr_window / rho0 / ir / stress_days / drw_ir | **0.0%** | **0.0%** |
| scarcity (compulsory, 0.55/0.62) | **11.0%** | 0.0% |

Under normal conditions, 0% of DR is planner-optimal — every curtailment
is privately rational (via the household's own λ-weighted comparison)
but not system-value-creating. Only real scarcity, where the shadow
price can reach the VoLL level, makes any of it pass the strict test.

## 3. Scarcity decomposition (severe scarcity, 0.55/0.62) — Z, energy not delivered, welfare cost

Tags: `_period_scarcity_nodr`, `_period_scarcity_drwonly`,
`_period_scarcity_ironly`, `_period_drw_ir_scar_rho0`,
`_period_drw_ir_scar_rho15`, `_period_scarcity`.

| arm | Z, unserved (kWh) | curtailed (kWh) | **energy not delivered = Z+curtailed** | VoLL cost (Tk) | welfare cost = VoLL·Z + γ·curtail + γ·shift (Tk) |
|---|---|---|---|---|---|
| no DR | 8,640,419 | 0 | **8,640,419** | 518,425,161 | 518,425,161 |
| `dr_window` ONLY (no IR) | 4,186,355 | 4,024,307 | **8,210,661** | 251,181,300 | 344,175,800 |
| IR ONLY (all-day scope) | 4,147,018 | 4,995,786 | **9,142,805** | 248,821,100 | 352,727,800 |
| `drw_ir` (both), rho=0 | 7,387,586 | 1,135,882 | **8,523,468** | 443,255,200 | 466,561,400 |
| `drw_ir` (both), rho=15 | 6,533,245 | 1,818,740 | **8,351,985** | 391,994,700 | 430,709,300 |
| **compulsory (all-scope)** | 19,237 | 12,746,020 | **12,765,257** | 1,154,217 | **279,528,400** |

**The headline is welfare cost, not Z.** Compulsory DR cuts Z by 99.8%,
but energy not delivered is actually **47.7% HIGHER** than no-DR (DR
substitutes voluntary curtailment for involuntary shedding — it does not
shrink the underlying shortfall). Welfare cost still falls **46.1%**
because voluntary curtailment is priced at γ (~12–37 Tk/kWh) instead of
VoLL (60–150 Tk/kWh) — that repricing, not shortfall elimination, is the
real benefit.

**Window-scoping and IR each recover about half the benefit alone;
together they recover barely a quarter.** `dr_window` alone and IR alone
each cut Z by ~52% vs no-DR; combined (`drw_ir`), they cut it by only
14.5–24.4% — an interaction effect (their eligible action-sets intersect
narrowly), not two additive costs.

**Scale of the stress test.** Kalyanpur's baseline load exceeds effective
P^max for a mean of **10.6 hours/day, every day** of the 30-day period
(Dhanmondi: 1.8 h/day, also every day) — this is a sustained
supply-shortage stress test, not a peak event.

**Caveat on `Z` itself.** The formulation sheds `Z_critical`/
`Z_shiftable` = 0 in every arm, but this reflects an ASSUMED capability
(smart-meter-level, per-class selective shedding, no explicit cap in the
LP) rather than a modelled protection. Bangladesh's actual practice is
feeder-level rotational shedding that takes all classes down together —
so the `no-DR` row above is a best case for load-shedding, not a
realistic one.

## 4. rho sweep on `drw_ir` (base and severe-scarcity conditions)

Tags: `_period_drw_ir_rho{0,3,6,9,12,15}`,
`_period_drw_ir_scar_rho{0,3,6,9,12,15}`.

| condition | rho | curtail kWh | shift kWh | %gained, ALL conn. (unw. / λ) | %gained, PARTICIPANTS (unw. / λ) | rebate paid (Tk) | avoided cost (Tk) | budget balance (Tk) |
|---|---|---|---|---|---|---|---|---|
| base | 0 | 1,135,706 | 135,290 | 0.93 / 11.97 | 6.9 / 88.7 | 0 | -7.92M | -7.92M |
| base | 15 | 1,818,642 | 135,290 | 28.96 / 33.45 | **86.2** / 99.5 | 17.72M | -11.68M | -29.40M |
| scarcity | 0 | 1,135,882 | 432,927 | 0.69 / 11.51 | 5.1 / 85.3 | 0 | +53.34M | +53.34M |
| scarcity | 15 | 1,818,740 | 694,716 | 25.20 / 32.73 | **75.0** / 97.4 | 17.72M | +86.47M | +68.74M |

(Full 12-row table in `thesis_changes.md`.) **rho\*** (smallest rho with
≥80% of PARTICIPATING connections gaining, unweighted — the only
informative definition; the λ-weighted one is a tautology of the IR
constraint itself, not a finding): **base = 15**; **scarcity = NOT
REACHED** even at rho=15 (75.0%). Avoided cost is negative under base at
every rho (DR is a wealth transfer to households, not a cost saving,
absent real scarcity) and strongly positive under scarcity.

**Peak ceiling (base condition).** Dhanmondi's post-response peak sits
OUTSIDE its own DR-activation window on **30/30 days at every rho** —
peak reduction is capped by geometry, not by rho: max achievable
reduction = baseline peak − highest out-of-window baseline slot = 4,194
kW (Dhanmondi) / 10,350 kW (Kalyanpur), and observed reduction sits right
at that ceiling regardless of rho. Rebate cost per kW of peak reduction
rises from Tk 0 (rho=0) to Tk 41.6/kW/day (rho=15) as spend keeps rising
against a flat, ceiling-pinned reduction.

## 5. Stakeholder ledger

| arm | operator: lost retail revenue (Tk) | operator: rebate paid (Tk) | residential lower-middle saving (Tk) | residential middle saving (Tk) | residential upper-middle saving (Tk) |
|---|---|---|---|---|---|
| base (original) | 155,966,726 | 15,665,847 | **0** | 11,483,690 | 84,324,700 |
| drw_ir, rho=0 | 17,621,047 | 0 | **0** | 0 | 12,287,820 |
| drw_ir, rho=15 | 27,299,901 | 17,721,186 | **0** | 288,374 | 17,279,270 |
| scarcity | 169,981,105 | 18,660,810 | 430,501 | 19,187,200 | 84,148,420 |

**Lower-middle income households receive zero bill saving in every arm
except scarcity.** Upper-middle captures 72–92% of all residential
savings in every arm. `Hospital`/`Government` categories show zero or
negligible savings throughout (protected archetypes).

## Bottom line for discussion

1. This DR mechanism, as parameterised, is **privately rational but not
   socially efficient** outside real scarcity (0% planner-test pass rate).
2. It is a **narrow, upper-income wealth transfer**, not a broadly shared
   benefit, under normal conditions.
3. Under real scarcity it delivers a **genuine ~46% welfare-cost
   reduction** — but by repricing the shortfall, not eliminating it.
4. **Voluntary (individually-rational, window-scoped) DR captures only a
   fraction of that benefit** — not because either restriction is
   expensive alone, but because requiring both together is
   disproportionately restrictive.
