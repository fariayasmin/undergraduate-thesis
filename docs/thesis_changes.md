# Thesis / formulation changes required by this audit

> **Follow-up audit (this section).** A second reviewer examined the first
> pass's own log output and found real remaining problems: the tie-break
> fix for degenerate quantile cuts was masking rather than solving the
> issue, the acceptance checks for pooling were tautological (guaranteed to
> pass by construction), the LP's all-slot curtailment bound combined with
> its objective double-counts a household's retail saving and the
> operator's wholesale saving, C4's dual was missing from the threshold
> decomposition, and three of four scenario arms had never actually been
> run. Each item below is a genuine fix or a genuine, honestly-reported
> limitation - not a reassertion that the first pass was already correct.

---

## Follow-up F1. Priority-pool tie-break was arbitrary, not meaningful

**What was wrong.** When quantile cut VALUES tied (Issue E's second finding),
entities were split by sorting on `(chi, entity_id)` - a string comparison
with no domain meaning. "0 degenerate days" was real, but WHICH entities
landed in P_M versus P_L at a tied boundary was decided by string luck.

**Fix.** `gentwin/regime.py`: added `Entity.flex_potential` (an appliance's
`n * p_rat_kw * duty_cycle` - its actual rated capacity to move load; 0 for
meters/substations). The tie-break key is now `(chi, flex_potential, w_e)`:
a bigger, more flexible appliance tied on chi with a smaller one now ranks
above it, which is a real judgement about which entity matters more, not
noise. `assign_pools()` also now reports `tie_break_used`, `tied_at_lo`,
`tied_at_hi` in its return dict, so the SIZE of every tie-broken boundary is
visible in `priority_pools{tag}.json` rather than hidden inside "0
degenerate days."

---

## Follow-up F2. Pooling acceptance checks were tautological

**What was wrong.** `POOL_N_BY_KIND` forces top-N to contain 3 kinds by
construction; the rank-based tie-break forces non-empty pools for
`n_ent >= 4` by construction. Neither check tests whether the RANKING is
doing anything useful.

**Fix.** `gentwin/regime.py::precision_at_n(ents, topn)` (new): measures
what share of top-N is actually "acting" that day (`e_sh_kwh+e_cu_kwh>0`)
against `baseline_random` (the population-wide prevalence of acting
entities - what a same-size random sample would hit in expectation) and
reports the `lift` ratio. `gentwin/regime.py::random_topn()` (new): a
same-size random comparator, stratified by the same `POOL_N_BY_KIND` sizes.
Together these let quantile, mean_relative, and random be compared on the
SAME day's scored entities. **Superseded by Follow-up F14**: same-day
precision turned out to be circular (phi2 is built from the same "acting"
label being checked against), so this is now only one of four reported
variants - see F14 and the measured-results table below for the
non-circular ones and what all four actually measure.

---

## Follow-up F3. The LP curtails all day, every day - not peak-time DR

**What was wrong, and it is the most consequential item in this whole
follow-up.** `y_{h,t}`'s upper bound was `y_max_h` at every one of the 48
slots, every day, with no dependence on congestion, stress, or the DR
window at all. Combined with the objective (Eq. 24) counting BOTH the
household's retail saving (`lambda_h * J1` falling as y rises) AND the
operator's own wholesale cost saving (`theta_1 * f_cost` falling as
imported energy falls) on the SAME displaced kWh - even though the retail
payment is a transfer from household to utility, not two independent
savings - the LP has a standing incentive to curtail toward the QoE floor
at every slot, regardless of whether the substation is under any stress at
all. The result, visible in the base-arm log: pi_i(t) flat at the flat
producer price all day (no congestion anywhere), Z/nu/T all zero, and
~9,500 y-variables active per day against ~70 x-variables. This is mostly
all-day energy reduction wearing the label "demand response."

**Fix (partial - the scope restriction; the double-count itself is
reported, not eliminated, since removing it is a bigger change to the
objective's structure than this pass should make without a broader
sign-off).** `cfg.CURTAIL_SCOPE` (new, default `"all"` - UNCHANGED base-arm
behaviour): `"dr_window"` restricts `y_{h,t}`'s bound to the substation's
own DR-activation window (matching where `x_{h,t}` already lives);
`"stress_days"` restricts it to days with `s_stress=1`. `06_lp_optimiser.py
--curtail-scope dr_window|stress_days`. See the acceptance-check table
below for the measured before/after under `dr_window`.

**Recommended formulation text.** State explicitly, adjacent to Eq. (20) and
Eq. (22): "the household's retail payment (part of `f_wel` via `J1`) and
the operator's avoided wholesale cost (part of `f_cost`) are DIFFERENT
accounting perspectives on the SAME transfer, not two independent sources
of value from the same curtailed kWh; the reported system-level saving is
therefore an upper bound unless one of them is removed from the objective
or the two are reconciled through an explicit tariff/wholesale settlement
term." Recommend restricting `y_{h,t}` to `"dr_window"` or `"stress_days"`
as the reported result, with `"all"` kept only as an explicit upper-bound
scenario, clearly labelled as such.

---

## Follow-up F4. C4's dual was missing from the Proposition-2 threshold

**What was wrong.** `threshold_table()`'s `syst` term used only C1's dual
`pi_i(t)`. C4 (export/capacity) has the IDENTICAL load-term structure
(`k_i * load_coeffs(t)`), so when C4 binds, its dual is a second, real
shadow price on the same x/y decisions that the stored explanation
silently dropped - invisible in the base arm (C4 is slack there, `G^max`
unbound) but wrong in any scenario where it binds, including the scarcity
arm.

**Fix.** `gentwin/lp.py::solve_day()` now extracts `pi4_i(t)` (C4's dual)
with the same sign/Delta correction as `pi_i(t)`, stored as `sol.duals_c4`.
`threshold_table()`'s `syst` is now `k_i*(pi[t]-pi_off)/th3 +
k_i*(pi4[t]-pi4_off)/th3` (shift) and `k_i*pi[t]/th3 + k_i*pi4[t]/th3`
(curtail) - exactly zero, and therefore no change to any previously-reported
number, whenever C4 is slack.

---

## Follow-up F5. Re-pooling trigger is nearly blind to same-kind churn

**What was wrong.** `W(P) = sum_e w_e` with `w_e` fixed per entity
(`1/n_ent` for an appliance, `deg(e)/n_ent` otherwise) and pool sizes pinned
to fixed quantile fractions means `alpha_P = |W(P)_t - W(P)_{t+1}|` can only
move when entities of DIFFERENT `w_e` (i.e. different kinds) swap pools.
Same-kind reshuffling - a household that curtailed heavily today swapping
places with one that acted yesterday, both appliances - moves `alpha_P` by
nothing, so a day can be "reused" even though today's actual DR activity
looks nothing like yesterday's. The sweep result is consistent with this:
re-pooling saturates at a fixed count (14/30 days) for every
`alpha >= 0.02`, suggesting the trigger is close to bimodal rather than
continuously informative.

**Fix.** `gentwin/regime.py::membership_change_fraction(ents,
prev_membership)` (new): the plain fraction of entities whose pool label
would change, entity-by-entity - sensitive to same-kind churn `alpha_P`
structurally cannot see. `cfg.REPOOL_TRIGGER_KIND` (`"alpha_w"` default,
unchanged behaviour, or `"frac_changed"`) selects which one gates the real
run; both are always computed and both should be reported side by side (see
the sweep note below for what has actually been run).

---

## Follow-up F6. Regime transfer-trigger direction

**What was raised.** The choice to flag the EXPORTING substation as
"emergency" (r=3) on a transfer was defended on the grounds that it bears a
real cost to its own margin. The counter-argument: the substation that
IMPORTS is the one that could not meet its own C1 row alone, which is the
more standard reading of "which substation is in trouble," and the one an
operator would act on first.

**Fix.** `cfg.REGIME_TRANSFER_TRIGGER` (new, default flipped to `"import"`):
`gentwin/regime.py::operating_regime()` now takes `transfer_in_kwh`
alongside `transfer_out_kwh` and a `trigger` argument (`"import"` |
`"export"` | `"both"`), with the full rationale for each reading in the
function's own docstring, including the counter-argument for the option NOT
chosen. Has no effect on any base-arm result reported so far - both
substations show zero transfers in every arm run to date - so this is a
default-direction correction made visible for when it does matter, not a
number that has changed.

---

## Follow-up F7. Battery never recharges across a multi-day run (reported, optional fix)

**What was observed.** The battery sits at its reserve floor for the whole
base-arm run and is never recharged - the myopic effect of a receding solve
with no condition on where the LAST day leaves the state of charge, and a
real per-kWh throughput cost (`C_BATTERY_TK_PER_KWH`) that gives the
objective no reason to pay for charging unless the reserve floor forces it.

**Fix (optional, off by default).** `cfg.TERMINAL_SOC` (new, default
`False`): when `True`, `06_lp_optimiser.py::run()` adds ONE constraint, on
the LAST day of the run only - `S_i(end of run) >= S^0_i` (the substation's
configured starting SoC) - via a new `terminal_target` parameter threaded
through `lp.build_day()`/`solve_day()`. Not claimed to be more "correct"
than the myopic default; it is a real modelling choice with its own cost
(the battery must now be recharged even when there is no operational reason
to), exposed for comparison rather than imposed.

---

## Follow-up F8. rho/IR/curtail-scope now labelled on KG and billing-statistics nodes

`09_monthly_billing.py`'s `billing_statistics{tag}.json` and
`gentwin/kg.py::build_billing()`'s `BillingPeriod` node now both carry
`rho_is_current_policy`, `individual_rationality_on`, and `curtail_scope`
directly (previously only printed to the CLI, and only `rho_tk_per_kwh`
itself was persisted), so a reader of either artefact alone - without the
command line that produced it - can tell which arm it is.

---

## Follow-up F9. Held-out T2 on 3 dates is too few; kappa range check

**What was raised.** The held-out timing check (Follow-up F1's catalogue
fix, Issue F) had only been run on 3 dates, too few to distinguish
generalisation from luck; and Dhanmondi's kappa range (0.67-1.33 across the
base-arm month) is wide enough that the tuned `beta_per_c=0.08` might be
over-reacting on hot days and under-reacting on cool ones - the overfitting
risk inherent in the grid-search fix itself.

**Run.** `05_validate_against_bpdb.py --held-out` with 24 dates (the 5th and
20th of every month across 2025, chosen for seasonal spread rather than
proximity to the tuned reference day 2026-06-29): **Dhanmondi 23/24** inside
`T^pk_i` (12:00-14:30) - the one miss, 2025-01-20, peaks at 18:30 instead,
disclosed rather than dropped; **Kalyanpur 24/24** inside `T^pk_i`
(18:00-21:00). Both substations generalise well beyond the single tuned
day. Full rows: `outputs/held_out_t2_{substation}.csv`.

Dhanmondi's kappa range itself (computed from the same, already-run 30-day
base arm - kappa is a property of the LP's own daily forecast-vs-baseline
comparison, not something a held-out run needs to add) is real and not
resolved by the wider held-out sample: it says the fix is well-CALIBRATED
IN TIMING (the peak always lands in the window) while still being
IMPRECISE IN LEVEL day-to-day (kappa swings further than 1.0 on hot/cool
days). This is a genuine, disclosed limitation of a single fixed
`beta_per_c`, not a contradiction of the held-out timing result - recommend
stating both in Chapter 4: the timing fix generalises, the day-to-day level
match does not perfectly, and a per-day or per-season `beta_per_c` is future
work, not something this pass changed.

---

## Follow-up F10. C4 had no load-shed term - a real formulation gap, not just a scenario being too harsh

**What was wrong.** C4 (Eq. 32, capacity/export row, H4 above) had no
`Z_c(t)` term, unlike C1 (Eq. 29, supply adequacy), which does. A round of
this audit first treated `--scale-gmax 0.55 --scale-pmax 0.62`'s
infeasibility (HiGHS status 8 on 2026-07-12) as the scenario simply being
too harsh, and picked a milder scale instead. That was the wrong diagnosis:
the LP had no way to shed load against C4 even after already shedding
against C1, so whenever the unavoidable load exceeded the scaled transformer
cap, there was genuinely no feasible point - a real operator sheds load
instead, which the model could not represent.

**Fix.** `gentwin/lp.py::build_day()`'s C4 row now includes
`-sum_c Z_c(t)` exactly like C1's. Shed load is not drawn through the
transformer either, so removing it from demand must relax C4 the same way
it relaxes C1. `0.55/0.62` (the ORIGINAL scenario scale, restored in
`run_pipeline.sh`/`run_pipeline.ps1` and matching `README.md`/`RUNBOOK.md`'s
existing examples) now solves and produces real `Z > 0` at the VoLL price
on the two most-stressed days (2026-07-12: Dhanmondi `Z=14,118` kWh;
2026-07-13: `Z=2,650` kWh) with `capacity_deficit` firing every day for
both substations once Follow-up F11 (below) is also applied - the emergency
regime this scenario was meant to exercise, previously never reached.

**Recommended formulation text.** Add `-sum_c Z_c(t)` to Eq. (32)/(3.31)'s
LHS, with a one-line note: unserved load reduces flow through the
transformer/feeder exactly as it reduces the supply-adequacy shortfall in
Eq. (29), so C4 must include it for the same reason C1 does.

---

## Follow-up F11. Scenario scale factors never reached 07/08/09 (separate processes)

**What was wrong.** `06_lp_optimiser.py::apply_scenario()` mutates
`cfg.SUBSTATIONS[i]["g_max_kw"/"p_max_kw"]` IN-PLACE, but only inside the
`python 06_lp_optimiser.py` process's own memory. `07_regime_events_pool.py`,
`08_knowledge_graph.py` and `09_monthly_billing.py` each run as a SEPARATE
`python` process and import a fresh, unscaled `cfg`, so a scarcity run's
effective P^max was invisible to every stage downstream of the LP solve
itself. Concretely: the scarcity arm's own log showed Dhanmondi at
`p~=166,478 kW` with `Def=0`, even though `P^max` was scaled to 130,000 kW
(0.65x) - `capacity_deficit` never fired, and Stage H's regime/pool/event
classification for the ENTIRE scarcity arm was silently computed against
the base case's P^max instead.

**Fix.** `06_lp_optimiser.py::export()` now writes
`scale_gmax_<sub>`/`scale_pmax_<sub>`/`g_max_kw_eff_<sub>`/
`p_max_kw_eff_<sub>` into `lp_summary{tag}.csv` (1.0/1.0/base values for the
base case, so old base-arm summaries stay readable). `07_regime_events_pool.py::run()`
now reads `p_max_kw_eff_<sub>` from the summary and passes it to
`R.operating_regime()` instead of reading `cfg.SUBSTATIONS` directly, and
prints the scenario it read whenever it differs from 1.0/1.0. Verified: the
scarcity arm now shows `Def=32,151` kW (not 0) and `capacity_deficit` firing
on every day for both substations, matching the LP's own P^max exactly.

---

## Follow-up F12. Billing Tk quantities mixed a scaled and an unscaled period

**What was wrong, and it invalidated most of the previous round's headline
billing numbers.** `gentwin/billing.py::gazette_bills()` scales
`baseline_kwh`/`optimised_kwh`/the resulting bill to a 30-day-equivalent
month via `scale = 30.0/days`, and reports `scale_to_30d` so nothing is
hidden - but `j1_linear()`'s `J1_baseline_tk`/`J1_optimised_tk`/`rebate_tk`
and `accumulate()`'s `discomfort_J2_tk`/`incentive_received_tk` all summed
over however many days the run actually covered, WITHOUT that scale. For a
full 30-day run (`scale=1`) this is invisible; for the 7-day comparison arms
this round introduced for speed, it silently mixed a 30-day-equivalent bill
saving with 7-day-actual discomfort/incentive inside
`net_benefit_tk`/`gained_unweighted`/`gained_lambda`, which:
- inflated the reported "% gained" from ~1-2% (the honest 30-day figure) to
  54.7% on the 7-day base arm - a ~40x distortion, entirely an artefact of
  the mismatched scale, not a real result;
- inverted the expected J1-vs-gazette relationship (J1 uses the marginal
  slab rate, which is >= the average rate the gazette bill uses, so J1
  saving should be >= gazette saving; the unscaled comparison showed the
  OPPOSITE, gazette saving ~3.8x LARGER than J1 saving).

**Fix.** `gentwin/billing.py::build()` now scales `J1_baseline_tk`,
`J1_optimised_tk`, `rebate_tk`, `discomfort_J2_tk` and
`incentive_received_tk` by the SAME `30.0/days` factor as the gazette bill,
immediately after both are computed, so every Tk column in the table is the
same 30-day-equivalent quantity regardless of the period actually solved.
Verified on the 7-day base arm: mean J1 saving moved from 580 Tk (wrong,
unscaled) to 2,488 Tk (now >= the 2,196 Tk gazette saving, as expected);
`%gained_unweighted` moved from 54.7% (inflated) to 1.24% (consistent with
the original 30-day run's own figure). `09_monthly_billing.py::load_run()`'s
short-run warning threshold was also tightened from `< 7` to `< 30` days,
since 7 days is not itself a safe minimum, only the bare floor the old
message implied.

**Recommended formulation text.** No equation changes - Eq. (20)-(24) are
period-length-agnostic; this was purely an implementation bug in how a
partial period gets normalised to a nominal month, now fixed at the source
rather than by only ever running 30-day periods.

---

## Follow-up F13. Household-level peak reduction is not what DR is judged on

**What was wrong.** `gentwin/billing.py::summarise()`'s
`peak_reduction_pct_mean` averages each HOUSEHOLD's own peak reduction. A
household's individual peak can fall while the SUBSTATION peak - what the
transformer/feeder and Eq. (48)'s regime actually see - is unmoved, or even
rises if shifted load recovers into a new coincident peak. No
substation-level number existed anywhere except inside a plotting function
(`09_monthly_billing.py::_draw_daily_peak_panel`), never turned into a
reported statistic.

**Fix.** `gentwin/billing.py::substation_peak_reduction(net, subs)` (new):
per substation-day, `max_t` of `lp_network{tag}.csv`'s `baseline_kw` vs
`post_response_kw` - the same series the existing dispatch/peak plots
already draw. Also flags `peak_shifted_not_shaved`: a day where the peak
value did not fall but its CLOCK SLOT did move, so a shift that only
relocates the peak rather than shaving it is visible rather than averaged
into a positive-looking mean. `09_monthly_billing.py --report` now prints
both numbers side by side, labelled, and exports
`outputs/billing/substation_peak_reduction{tag}.csv`.

---

## Follow-up F14. precision@N was circular - phi2 is built from the label it was checked against

**What was wrong.** The previous round's precision@N = 1.000 on every day
of every arm was presented as evidence the ranking works, on the grounds
that a same-size random draw was NOT 1.0. That defence misses the actual
mechanism: "acting" is defined as `e_sh_kwh+e_cu_kwh > 0`, which is exactly
what phi2 is built from (Eq. 46: `phi2(e) = resp(e) / max_resp(kind)`).
Within one kind, on one substation-day, phi1/phi3/phi4 are IDENTICAL for
every entity (phi1 depends only on a fixed appliance degree; phi3/phi4 are
substation-level constants that day), so ANY entity that acted has
`phi2 > 0` and outranks every entity that didn't, by construction. With
only 20 appliance / 30 meter slots against hundreds of acting entities,
precision@N=1.0 is close to guaranteed regardless of whether the ranking
carries any real signal - the random baseline not being 1.0 says only that
a random draw doesn't automatically inherit the same construction, not that
the ranking's OWN precision is meaningful.

**Fix - three non-circular measurements, all in `gentwin/regime.py` and
wired into `07_regime_events_pool.py::run()`'s per-day loop, none of which
disturb the real (hysteresis-governed) pool assignment:**
1. `chi_excluding_phi2(ents, regimes)` + `topn_by_chi()`: re-rank with
   phi2's term dropped from chi entirely, so whatever precision remains
   comes only from graph centrality (phi1), stress (phi3) and deficit
   (phi4) - still same-day, but the SCORE no longer contains the label.
2. `assign_pools(copy, "mean_relative")` + `top_n()`: the base paper's own
   pooling rule, scored against the SAME day, alongside quantile and
   random (item 2's original three-way ask: quantile vs mean_relative vs
   random).
3. `precision_at_n_ids(topn_ids, acting_ids, n_total)` (new, id-set form):
   day d's top-N (fixed BEFORE day d+1's response exists) scored against
   day d+1's acting set - the only variant that cannot leak the label,
   because the label does not exist when the ranking is made. This is the
   test that actually answers "does the pool predict who will act."

`outputs/precision_at_n{tag}.csv` now carries all of `precision`
(same-day, circular), `precision_mean_relative`, `precision_no_phi2`,
`precision_nextday`, and their respective random baselines/lifts, per day.
See the measured-results table below for what these numbers actually are -
not assumed here.

---

## Follow-up: measured results (Round 3 - all five arms, full 30-day period)

**Superseded the previous round's table entirely.** That round used a
7-day comparison window for speed; Follow-up F12 found this silently mixed
a 30-day-scaled bill saving with 7-day-actual discomfort/incentive, and the
explicit instruction for this round was to drop the 7-day comparison and
rerun everything on the full 30-day period (2026-06-14 -> 2026-07-13). Every
number below is from that full-period rerun, with every fix in this
document applied: the tie-break key (F1), C4's dual (F4), the
scenario-scale propagation to Stage H (F11), the C4 load-shed term (F10),
the billing scale fix (F12), the substation-level peak metric (F13), the
non-circular precision variants (F14), and the IR-term normalisation fix
below (F15, found BY running acceptance check 6 on the `ir` arm
specifically, exactly as this round's instructions required).

## Follow-up F15. IR's threshold contribution was missing an energy-factor division

**What was wrong.** `threshold_table()`'s `ir_term` (Follow-up-1's fix for
Issue I) was `ir_dual * jx_arr[a_] / th3` (shift) / `ir_dual * jy_arr[t] /
th3` (curtail). The IR row's own coefficient on `x_{h,tau}` IS
`jx_arr[a_]` (a per-slot welfare contribution that already contains the
household's own `sh[tau]*Delta` energy factor - see `build_day()`'s
`jx[a_] = lam*bill + (1-lam)*disc`, where `bill`/`disc` are both
proportional to `sh[tau]*Delta`). `priv`/`disc`/`qoe` in the same formula
are all normalised to per-kWh (price-like) units - `qoe`'s analogous
C7-dual term works out to the constant `QOE_WEIGHT_SHIFT` exactly BECAUSE
C7's row coefficient (`QOE_WEIGHT_SHIFT*sh[tau]*Delta`) is divided by that
same `sh[tau]*Delta` before being added to `qoe`. `ir_term` never did that
division, so it stayed in raw `jx`-units - roughly `sh[tau]*Delta` (often
0.01-1 kWh) times too large. Caught exactly as this round's instructions
intended: acceptance check 6 (`|margin| <= 1e-6` for interior actions) run
on the `ir` arm specifically FAILED, with 1,284 of 4,754 interior actions
violating it by up to -3,603 Tk/kWh.

**Fix.** `gentwin/lp.py::threshold_table()`: both `ir_term` computations
now divide by the same slot's `blk["sh"][t]*D` (shift) / `blk["cu"][t]*D`
(curtail) before dividing by `th3`, mirroring exactly how C7's row
coefficient collapses to a constant. Re-verified: check 6 now PASSES on
the `ir` arm with 0 violations (was 1,284). The LP's own decisions (x, y,
objective) are UNCHANGED by this fix - `ir_term` only feeds the diagnostic
threshold/margin report, never the constraint or objective itself, so
`shift_kwh`/`curtail_kwh`/`n_participating`/every billing number for the
`ir` arm is identical before and after.

---

**LP-level, full 30-day period:**

| arm | shift kWh | curtail kWh | n_events | degenerate pool-days | check 6 |
|---|---|---|---|---|---|
| base | 54,328 | 11,389,227 | 86 | 0/30 | PASS |
| dr_window | 135,290 | 3,088,306 | 86 | 0/30 | PASS |
| rho0 | 60,048 | 11,078,285 | 86 | 0/30 | PASS |
| ir | 93,995 | 5,171,419 | 86 | 0/30 | PASS (after F15) |
| scarcity (0.55/0.62) | 1,155,314 | 12,746,020 | 124 | 0/30 | PASS |

`dr_window` cuts curtailed energy by 73% versus `base` over the full month
(11.39M -> 3.09M kWh) while shifting more than doubles - the same
double-counting signature as the earlier 7-day check, now confirmed at the
full period. `scarcity` (0.55/0.62, restored to the ORIGINAL scale after
Follow-up F10's C4 fix) produces real unserved load: Dhanmondi hits
`Z=14,118 kWh` at the VoLL price (60 Tk/kWh) on its worst day, with several
other days in the 1,800-5,600 kWh range and `capacity_deficit` firing on
every one of the 30 days for both substations (verified via Follow-up
F11's fix - previously this trigger never fired at all, regardless of
scale, because it read the wrong, unscaled P^max).

**Billing-level, fleet-wide (all 800 representatives, full 30-day
period, POST billing-scale fix F12):**

| arm | n participating | median saving % (fleet-wide) | mean bill saving (Tk) | % gained, unweighted | % gained, lambda-weighted | substation peak reduction % |
|---|---|---|---|---|---|---|
| base | 460 | 5.80 | 2,167.7 | 1.2 | 14.8 | 3.61 |
| dr_window | 458 | 0.74 | 622.7 | 2.2 | 15.3 | 5.31 |
| rho0 | 432 | 4.00 | 2,145.5 | 1.2 | 14.8 | 3.27 |
| ir | 207 | 0.00 | 1,032.7 | 2.0 | 17.6 | 2.87 |
| scarcity | 750 | 8.69 | 2,380.3 | 1.2 | 14.8 | 19.42 |

These numbers are now roughly an ORDER OF MAGNITUDE more conservative than
the previous round's mistakenly-scaled ones (e.g. base's `%gained_unweighted`
1.2%, not 54.7%) and match the ORIGINAL 30-day run's own figures, which is
the internal-consistency check Follow-up F12's fix was supposed to restore.
`substation_peak_red_%` (Follow-up F13, `max_t` baseline vs post-response
per substation-day) is now reported instead of the misleading
household-averaged number; `scarcity`'s 19.42% is by far the largest -
consistent with its being the only arm where load-shedding and the
emergency regime are actually reached. `n_peak_moved_not_shaved` is 0 in
every arm: no day showed a peak that only relocated without falling, for
this run.

**Pooling, full 30-day period (Follow-up F14 - four precision variants,
none tautological by construction the way `precision@N=1.000` alone was):**

| arm | same-day (circular) | mean_relative (circular) | chi excl. phi2 (non-circular, same-day) | next-day (non-circular) | next-day random baseline | next-day lift |
|---|---|---|---|---|---|---|
| base | 1.000 | 1.000 | 0.706 | 1.000 | 0.265 | 3.80x |
| dr_window | 1.000 | 1.000 | 0.706 | 0.995 | 0.268 | 3.74x |
| rho0 | 1.000 | 1.000 | 0.695 | 1.000 | 0.255 | 3.94x |
| ir | 1.000 | 1.000 | 0.349 | 0.968 | 0.123 | 7.91x |
| scarcity | 1.000 | 1.000 | 0.806 | 1.000 | 0.452 | 2.31x |

Same-day precision is 1.000 on both quantile and mean_relative for every
arm - confirming the circularity diagnosis (F14): whichever pooling RULE
is used, phi2 alone (built from the acting label) separates acting from
non-acting entities perfectly within a kind/day, so same-day precision
cannot distinguish a good ranking from a tautological one. Dropping phi2
from chi (still same-day, but the score can no longer see the label) drops
precision to 0.35-0.81 - real signal from graph centrality/stress/deficit
alone, well above what a same-size random draw would get, but nowhere near
perfect. Next-day precision - the one variant that CANNOT leak the label,
since day d's top-N is fixed before day d+1's response exists - stays
remarkably high (0.968-1.000) with a substantially lower random baseline
(0.123-0.452), giving a real lift of 2.3x-7.9x: entities that were
top-ranked today are very likely to be the SAME entities acting tomorrow,
which is a genuine, non-circular finding about persistence of who responds,
not an artefact of the scoring formula.

Item 3 (`alpha_w` vs `frac_changed` re-pooling trigger, computed
side-by-side every day, `alpha_w` remains the active default rule): over
the full 30-day base arm the two triggers disagree on 2/30 days (`alpha_w`
repools 16 days; `frac_changed`, threshold 0.05, repools 14) - unchanged
from the 7-day-window check, since both triggers are computed from the same
underlying day-to-day score movement regardless of period length. Real and
small, consistent with Follow-up F5's claim that `alpha_w` is blind to
same-kind churn rather than useless.

Full per-day numbers: `outputs/precision_at_n{tag}.csv`,
`outputs/repool_trigger_compare{tag}.csv`,
`outputs/alpha_repool_sweep{tag}.csv`,
`outputs/billing/substation_peak_reduction{tag}.csv` for
`tag in {_period, _period_dr_window, _period_rho0, _period_ir,
_period_scarcity}`.

---

The thesis LaTeX source is not available to this audit, so every correction
below is written as: the passage/equation as it stands, the replacement, and
which files implement it. This supersedes `docs/chapter3_vs_implementation.md`
for anything the two disagree on (this file is newer and reflects the full
audit; the older file's residential-ToU section (#1) is superseded by Issue A
below, which generalises it).

Every item is classified exactly as the audit brief requires:

- **Category 1** - the implementation is a valid extension. Keep the code;
  this file tells you what to write in the thesis instead.
- **Category 2** - the written equation is inconsistent or wrong. The fix is
  entirely in this document; no code changed.
- **Category 3** - the code conflicted with the intended real-world
  (Bangladesh) assumption. The code was changed; this file records why.

---

## Issue A (Category 3, code changed) - billing window vs. DR activation window

**What was wrong.** `tariff.mu_profile()` took its peak/off-peak window from
`peak_window.get_peak_slots(substation)` - the ADAPTIVE, substation-derived
congestion window - for every use, including billing. So a Dhanmondi LT-C1/
LT-E consumer was being billed the peak rate at 12:00-14:30 and the off-peak
rate at 17:00-23:00, which is the reverse of what BERC's order actually
specifies (peak 17:00-23:00, national, every substation, footnote 6).

**Fix.** `gentwin/tariff.py::mu_profile()`/`effective_price()` are unchanged
in their capability (they can still return either window), but every BILLING
call site now passes `adaptive=False` explicitly, which forces
`OFFICIAL_TOU_PEAK` (17:00-23:00):

- `gentwin/lp.py::_price_vector()` - the price the LP's objective and
  Proposition-2 threshold use.
- `gentwin/billing.py::j1_linear()` - the "J1" bill reported in
  `09_monthly_billing.py`.

The DR-activation window is untouched: `lp.build_day()` still creates
`x_{h,t}` only on `peak_window.get_peak_slots(i)`'s slots, and the recovery
mechanism (Eq. 5/14) still spreads displaced energy over that same window's
complement. **This is the intended, load-bearing consequence, not a side
effect to work around**: because a household's shift decision now moves
energy out of a slot billed at the OFFICIAL off-peak rate (whenever the
adaptive window falls outside 17:00-23:00, as at Dhanmondi) and into a
recovery window that includes the OFFICIAL peak rate, shifting at a
misaligned substation now correctly RAISES the bill instead of lowering it.

**Replacement equation, Eq. (19)/(3.21):**

```
pi_h(t) = (pi^ret_h + varpi_{i,m}) * mu_c(t)

    mu_c(t) = 1                                         if class c has no gazette ToU row (LT-A, LT-D1)
    mu_c(t) = mu^pk  if t in [17:00,23:00), else mu^off   if class c has a gazette ToU row (has_tou = True)
```

`T^pk_i` (the substation-derived window in Eqs. 5/32/34) is a SEPARATE
object, unchanged, and enters ONLY the DR-activation bound
`1[t in T^pk_i]` on `x_{h,t}` - never `mu_c(t)`.

**Knowledge graph.** `TouWindow` now carries two explicit fields,
`official_billing_window_{slots,label}` (always 17:00-23:00) and
`dr_activation_window_{slots,label}` (the derived per-substation window),
plus `applies_to` stating which tariff classes the ratios apply to. The old
single `slots`/`window_label` pair (which silently meant the adaptive window)
is removed - any query reading `tw.slots` will now fail loudly (KeyError)
rather than silently reading the wrong window.

**Label bug, fixed alongside.** `Substation.t_pk_label` and the old
`TouWindow.window_label` were read from `cfg.SUBSTATIONS[i]["t_pk_label"]`,
a STATIC fallback literal ("12:00-15:00" for Dhanmondi) used only when no
derived cache exists. The actually-used slots (`tpk`, from
`peak_window.get_peak_slots`) already preferred the DERIVED window
(12:00-14:30, 5 slots), so the label and the slots could disagree - and did.
Both `gentwin/kg.py` and `00_tariff_bpdb.py::cmd_mu()` now read the label
from the same cache the slots come from (`peak_window.read_cache(i)["label"]`).

---

## Issue G (Category 3, code changed) - threshold_table bugs

**What was wrong**, in `gentwin/lp.py::threshold_table()`:

1. The shift threshold's private term used the family ratio
   `base_rate * (mu_pk - mu_off)` instead of the actual slot price. After
   Issue A this is doubly wrong: `mu_pk`/`mu_off` no longer even describe
   what a given adaptive-window slot is billed at.
2. The curtail threshold's `lambda_star` (Eq. 38) used `base_rate * mu_pk`
   (a class-level constant) and a bare `rho` (not `rho * s`), and omitted the
   QoE term entirely - inconsistent with the `priv`/`margin` fields on the
   very same row.

**Fix**, using the ACTUAL per-slot price `p[t]` from `ctx["prices"][cid]`
(officially-windowed per Issue A) throughout:

```
shift    priv = lambda_h * ( p[tau] - mean_{t in T^off_i} p[t] )
curtail  priv = lambda_h * ( p[t] + rho * s )                       (unchanged - already correct)
         lambda*_h = ( gamma^cu_h + qoe - syst ) / ( p[t] + rho*s + gamma^cu_h )     Eq. (38), corrected
```

Both are now exactly the LP's own per-unit objective coefficient (`jx`/`jy`
in `build_day`), not an approximation of it - the same number the solver
actually used, stored rather than re-derived.

**Recommended formulation text.** Eq. (38) should read as above (slot price,
`rho*s`, and the QoE numerator term), and a note added that `p(t)` there is
the SAME officially-windowed price of Eq. (19)/(3.21) as amended by Issue A.

---

## Issue D (Category 3, code changed) - kappa double-counted in a monthly run

**What was wrong.** `06_lp_optimiser.py::run()` gives every dispatch day its
OWN Theta(t) (observed or climatology - this was already true before this
audit, contrary to the module's own prior docstring, which claimed Theta was
pinned to the reference day). But `kappa_d` was still
`p~_{i,d} / p_{i,d0}` - the robust forecast divided by the FIXED
reference-day peak. A hot day's bottom-up load `Lambda_d(t;0,0)` is already
elevated by that day's own Theta_d through the thermal response (Eq. 2);
dividing the forecast by the (cooler) reference day's peak on top of that
counts the same weather signal twice.

**Fix.** `06_lp_optimiser.py::run()` now computes
`Lambda_d(t;0,0)` - that day's own bottom-up baseline, from that day's own
Theta_d - BEFORE the solve, and redefines:

```
kappa_d = p~_{i,d} / max_t Lambda_d(t;0,0;Theta_d)          (corrected)
kappa_refday_d = p~_{i,d} / p_{i,d0}                        (Eq. 16 as originally written, kept for traceability)
```

Both are written to `lp_summary{tag}.csv` as `kappa_<sub>` and
`kappa_refday_<sub>`.

**Recommended formulation text, Eq. (16)/(3.8):**

```
kappa_{i,d} = p~_{i,d} / p_{i,d}(Theta_d)
```

where `p_{i,d}(Theta_d) = max_t Lambda_{i,d}(t;0,0;Theta_d)` is that SAME
day's bottom-up baseline peak, not `p_{i,d0}` (the fixed reference-day peak
used elsewhere purely for the population-calibration constant K, Eq. 7-8,
which is a separate, one-time, non-double-counted use of the reference day).

---

## Issue I (Category 1, code added - not required by the base formulation)

**What it is.** An OPTIONAL, off-by-default linear constraint per household
per day:

```
lambda_h * ( J1_h(0,0) - J1_h(x,y) ) >= (1 - lambda_h) * J2_h(x,y)
```

equivalently `lambda_h*(J1(x,y)-J1(0,0)) + (1-lambda_h)*J2(x,y) <= 0` - the
household's own lambda-weighted welfare must not be made worse off by its
own actions. `x=y=0` always satisfies it, so turning it on never makes the
programme infeasible. Implemented in `gentwin/lp.py::build_day()`
(`individual_rationality` parameter, `cfg.INDIVIDUAL_RATIONALITY`), exposed
via `06_lp_optimiser.py --ir`. When binding, its dual enters the
Proposition-2 threshold as a fourth (fifth, with QoE) term, stored as
`ir_tk_per_kwh`/`ir_binding` alongside the existing `qoe_tk_per_kwh`/
`c7_binding` fields - same treatment as C7's dual, for the same reason (a
non-box constraint's dual belongs in the reduced cost whenever it binds).

**Recommended formulation text.** A new constraint, C8 (IR), stated as
optional and not required for Propositions 1-3 (its own dual only appears in
the threshold when the constraint is both enabled AND binding - Propositions
1-3 hold verbatim with IR off, which is the default).

**Why report it separately from what billing.py already showed.**
`billing.py` already reported `net_benefit_tk = saving + incentive - J2 > 0`
as a plain Tk comparison - call this `gained_unweighted`. The LP itself never
used that criterion; it used the lambda-weighted one. `gained_lambda`
(`lambda_h*(saving+incentive) > (1-lambda_h)*J2`) is the criterion the
optimiser actually applied when it chose to act at all, and the two can and
do disagree for any household whose `lambda_h != 0.5`. Both are now reported
(`billing.py::build()`, `distribution()`); `gained` is kept as an alias of
`gained_unweighted` for backward compatibility with existing readers.

---

## Issue J (Category 3, code changed) - rho is a proposed incentive, not policy

**What was wrong.** `rho` (the stress-day curtailment rebate, Eq. 20) was
always active at its config value with no way to see what happens without
it, and nothing in the reported outputs stated that Bangladesh has no such
rebate today.

**Fix.** `06_lp_optimiser.py --rho 0` (or `cfg.RHO_CURRENT_POLICY = True`,
set automatically when `--rho 0` is passed) runs the CURRENT-POLICY arm -
no stress rebate, which is what the grid actually offers. The default
(`--rho` omitted) is unchanged and is explicitly labelled the PROPOSED
mechanism, both in the `06_lp_optimiser.py` CLI output and in
`gentwin/config.py`'s comment on `RHO_REBATE_TK_PER_KWH`/
`RHO_CURRENT_POLICY`.

**Recommended formulation text.** A footnote on Eq. (20)'s `rho` term:
"rho is a PROPOSED demand-response incentive evaluated by this study; it is
not a rate BERC currently publishes. The current-policy comparison sets
rho = 0."

**KG/billing labelling - done, see Follow-up F8.** `BillingPeriod` and
`billing_statistics{tag}.json` now carry `rho_is_current_policy` (alongside
the pre-existing `rho_tk_per_kwh`), plus `individual_rationality_on` and
`curtail_scope`.

---

## Issue E (Category 2, formulation equations corrected; code also fixed)

Two DIFFERENT things were wrong here: the base paper's own Eq. (50) is
internally inconsistent (Category 2 - the equation itself needs correcting),
and this implementation's Eq. (46) `phi1`/`phi2` were normalised at the
wrong scope (also a correctness bug, fixed in code since "the code
implements the formulation faithfully" per the audit brief - the
faithfulness itself was the bug).

### E1. phi1, phi2 (Category 2/3 - both formulation and its literal-faithful
implementation were degenerate; code corrected)

**Eq. (46) as written:** `phi1(e) = present_e * |kind(e)| / max_k |k|` (a
KIND-level count ratio) and `phi2(e) = resp(e) / max_{e' in ALL} resp(e')`
(GLOBALLY normalised). Both make every entity of a kind identical or
vanishingly small respectively - see `gentwin/regime.py::score_entities`'s
docstring for the exact mechanism. This is why P_H contained all 5,416
appliances every day and every appliance/meter's phi2 collapsed toward 0
next to a substation-level aggregate.

**Corrected (matches the GenTwin base paper's own Eq. 3, which this
formulation's Eq. 46 mis-transcribed):**

```
phi1(e) = present_e * deg(e) / max_{e' in ALL} deg(e')        entity-level graph degree, not a kind count
phi2(e) = resp(e) / max_{e' in kind(e)} resp(e')                normalised WITHIN kind, not globally
```

### E2. W(P) (Category 2 - the formula did not implement its own definition)

**Eq. (51)** defines `w_e` (`1/n_ent` for an appliance, `deg(e)/n_ent`
otherwise) and `W(P) = sum_{e in P} w_e`. The code computed
`W(P) = sum_{e in P}(1 + deg(e)) / n_ent` - a different quantity that does
not reduce to `sum w_e` for any pool. Corrected to `W(P) = sum_{e in P} w_e`
literally, in `gentwin/regime.py::assign_pools()`/`apply_membership()`.

### E3. The medium pool, Eq. (50) (Category 2 - the base paper's equation)

**As written:** `P_H: chi_e > 3/4 chi_bar`, `P_M: 3/4 chi_bar <= chi_e >
1/2 chi_bar`(sic - the base paper's own inequality direction on the medium
band, `Avg <= p < Avg/2`, is internally contradictory for any
`chi_bar > 0`, since `3/4 chi_bar > 1/2 chi_bar` always: nothing can be
simultaneously `>= 3/4 chi_bar` and `< 1/2 chi_bar`, or the interval reduces
to `(1/2 chi_bar, 3/4 chi_bar]` (the sensible reading, which is what this
code always implemented) - and Section 9.6 of the local formulation already
records that *this* sensible reading still "degenerates on tight score
distributions." Either reading of Eq. (50) can produce an empty P_M for a
concentrated chi distribution.

**Recommended formulation text.** State the QUANTILE cut rule
(`cfg.POOL_QUANTILES = (0.50, 0.75)`, i.e. P_L below the median, P_M between
median and 75th percentile, P_H above) as the primary rule. Keep the
mean-relative rule as Section 9.6 already frames it: a named ablation for
base-paper replication, selectable with `--pool-rule mean_relative`, with
its degeneracy risk stated rather than hidden. `cfg.POOL_CUT_RULE` default
flipped from `"mean_relative"` to `"quantile"` to match.

**A second, independent tie found empirically and also fixed.** Comparing
`chi_e` against the quantile VALUES (`np.quantile`) is not itself immune to
degeneracy: a large enough cluster of entities sharing an IDENTICAL chi -
common here, since most non-acting appliances at one substation get the
same phi1 (fixed degree=3), phi3 (s_stress) and phi4 (deficit/P_max), all
substation-level constants that day - can make the 50th and 75th percentile
VALUES coincide, leaving P_M empty even under "quantile" (observed on 2 of
30 days in the base-arm run). Fixed in `gentwin/regime.py::assign_pools()`:
when the two cut VALUES tie, fall back to RANK-based assignment, cut at
fixed sorted POSITIONS 50%/75%, which reproduces an exact 50/25/25 split by
construction regardless of how many values are tied, for any `n_ent >= 4`.
**Superseded by Follow-up F1**: the rank tie-break key was originally
`(chi, entity_id)` (a string comparison with no domain meaning, sorting by
Python's object id in effect); it is now `(chi, flex_potential, w_e)` -
`flex_potential` is an appliance's `n * p_rat_kw * duty_cycle` (0 for
meters/substations), so a bigger, more flexible appliance tied on chi now
ranks above a smaller one, and the tie SIZE at every boundary is reported
(`tie_break_used`/`tied_at_lo`/`tied_at_hi` in `priority_pools{tag}.json`)
rather than only "0 degenerate days." Recommended formulation text: state
Eq. (50)'s quantile variant as a rank statistic with a flexibility-based
tie-break, not a value-threshold comparison or an arbitrary id ordering.

### E4. Re-pooling hysteresis, Eq. (52) (Category 1 - genuine extension)

Not previously acted on: `alpha_P` (Eq. 52) was computed and reported every
day but never used to decide anything. `cfg.ALPHA_REPOOL = 0.01` (new) now
gates whether a day RE-POOLS (recomputes cut points from today's chi
distribution) or REUSES yesterday's pool MEMBERSHIP
(`gentwin/regime.py::apply_membership()`, keyed by the stable `entity_id`).
`07_regime_events_pool.py --alpha-sweep` reports the repool/reuse split
across `ALPHA_REPOOL in [0.001, 0.05]`. Recommended formulation text: state
Eq. (52)'s trigger as an actual control rule - "re-pool when
`max_P alpha_P > ALPHA_REPOOL`; otherwise carry the previous assignment
forward" - rather than a quantity computed for its own sake.

### E5. Top-N stratification, Eq. (53) (Category 1 - genuine extension)

`P_N = argmax_{Q subset P, |Q|<=N} sum chi_e` reduces to a flat sort only
because there is no interaction between entities - true regardless of
whether the sort is stratified. A FLAT sort of ~6,200 entities dominated by
~5,400 appliances is, again, always appliances. `cfg.POOL_N_BY_KIND =
{"sb": None (all), "mt": 30, "ap": 20}` stratifies the same argmax by kind
(`gentwin/regime.py::top_n()`), ranked within kind by
`(pool order PH>PM>PL, chi desc, w_e desc)`. This changes nothing about
what Eq. (53) IS (still a sort, not a search); it only fixes which subset of
entities the reader ever sees represented.

### E6. Regime transfer trigger direction (Category 1/2 - superseded, see F6)

**Originally** (this section, first pass): `t3["inter_substation_transfer"]`
used only `transfer_out_kwh` (EXPORTS), flagging the SENDING substation as
"emergency" (r=3), defended on the grounds that it is the one whose own
reserve/capacity margin was drawn down to help a neighbour. Kept as-is at
the time, "documented rather than changed."

**Superseded by Follow-up F6**: a second reviewer pass argued the RECEIVING
substation is the more standard reading of "which substation is in
trouble," since it is the one that could not meet its own C1 supply-adequacy
row without help, and the one an operator would act on first.
`cfg.REGIME_TRANSFER_TRIGGER` (new, default flipped to `"import"`) now
selects which side counts, with `"export"` (this section's original
reading) and `"both"` also available;
`gentwin/regime.py::operating_regime()`'s docstring carries the full
rationale for each. Base/rho0/ir show zero inter-substation transfer, so
the choice is invisible there; the scarcity arm DOES produce a nonzero
`T_out` at Kalyanpur (it exports to help Dhanmondi), so this is the first
arm where "import" vs "export" actually changes which substation's regime
shows the transfer trigger - see the measured-results table.

---

## Issue B (Category 3, code changed) - KG Forecast nodes missing for observed days

**What was wrong.** `kg.build_forecast()` built one `Forecast` node per row
of `forecast.read_cache(i)["rows"]`, which covers only the 14-day
forecast-horizon cache. A `_period` run (30 days: ~16 observed history days
plus the 14-day horizon) has Events/Regimes on ALL 30 days pointing
`TRIGGERED_BY`/`COMPUTED_FROM` at a Forecast node keyed by date - which,
for the ~16 observed dates, was never created. 40 of 86 Events were
dangling.

**Fix.** `08_knowledge_graph.py::main()` now builds per-substation rows via
`forecast.build_period(bpdb, i, dates[0], dates[-1])` - the SAME resolver
`06_lp_optimiser.py` and `07_regime_events_pool.py` already use for exactly
this reason - whenever the tag indicates a period run, and passes them to
the generalised `kg.build_forecast(g, subs, krec, rows_by_sub=...)`, which
now reads every field with `.get()` and stamps `mode = "observed"|
"forecast"` (observed rows lack `horizon_k`, `in_dispatch_horizon` and
`holiday_type`, and key their date under `"date"` rather than
`"target_date"`).

**Self-check.** `GraphBuilder.dangling_relationships()` (new) walks every
relationship and confirms both endpoints were actually created as nodes;
`08_knowledge_graph.py` prints PASS/FAIL and raises if the count is nonzero,
so this class of bug cannot silently reappear.

---

## Issue F (Category 3, code/data changed) - Dhanmondi peak-timing calibration

Not a formulation issue - the load MODEL (population/appliance schedules)
did not reproduce the substation's own observed peak timing. See
`02_consumer_catalogue.json`'s `res_ac_affluent` and `cooling_ac` comments
for the full before/after numbers and the grid search that found the fix;
summarised in the acceptance-check results below. No equation changed;
`Theta(t)`, Eq. (2)'s thermal-response form, and Eq. (7)/(8)'s calibration
constant K are all unchanged in form - only two catalogue PARAMETERS
(`res_ac_affluent`'s midday schedule level, `cooling_ac`'s `beta_per_c`)
were corrected, each with a written justification in the catalogue file
itself.

---

## Issue H - Category 1 items (valid extensions; formulation text only)

None of these required a code change; each is the code already doing more
than the written formulation states, per the audit brief's instruction to
keep the extension and update the documentation.

| # | Extension already in code | Where | Recommended formulation text |
|---|---|---|---|
| H1 | theta1/theta2/theta3 weights, `F = sum_i (theta1 f^cost_i + theta2 f^rel_i + theta3 f^wel_i)` | `gentwin/config.py` (`THETA_*`), `lp.py::build_day` objective assembly | Add theta1/theta2/theta3 to Table 3.1's symbol list; restate Eq. (24)/(3.27) with the weights and the sum over i, not just a bare sum of the three terms |
| H2 | Household-specific lambda_h, gamma^sh_h, gamma^cu_h, y^max_h | `gentwin/config.py::sample_preferences`, `population.py::synthesise` | Add the `_h` subscript to every occurrence in Eqs. (20)-(24)/(3.20)-(3.24)'s text; already justified at length in `docs/formulation_v4_delta.md` Sec. 2-3 and `docs/y_max_h_derivation.md` (Remark 4: "household heterogeneity is individual") |
| H3 | Appliance power model: duty cycle, simultaneity, per-household on/off sampling `u_{h,a}(t)`, per-household amplitude jitter | `gentwin/load_model.py::consumer_profile` | Complete Eq. (2)/(3.10), currently under-specified (epsilon is not even defined in the thesis text as given): `P_{h,a}(t) = n_{h,a} * sim_a * P^rat_a * eps_a * u_{h,a}(t) * f_a(Theta(t)) * j_h`, defining `sim_a` (simultaneity fraction), `u_{h,a}(t)` (the household's own sampled on/off pattern) and `j_h` (per-household amplitude jitter) alongside the now-plain per-appliance constant `eps_a` (duty cycle) |
| H4 | C4 substation export/capacity row: `T_i(t) - eta_dis*B^dis_i(t) + kappa_{i,d}(t) sum_h L_h(t;x,y) <= P^max_i` **(superseded, see Follow-up F10 below - the row itself needed a code fix, not just documentation)** | `lp.py::build_day`, the "C4 export limit" rows | Add this inequality to C4/Eq. (32)/(3.31), WITH the `-sum_c Z_c(t)` load-shed term from Follow-up F10, alongside the tie-line box bound `0 <= T_ij(t) <= T^max_ij`, which is a variable BOUND in code, not this row |
| H5 | C7 QoE constraint, entirely missing from the thesis chapter as reviewed, cited there only as "Eq. 35" | `lp.py::build_day`'s C7 rows, `docs/qoe_definition.md` (full derivation already written) | Insert `docs/qoe_definition.md`'s definition verbatim as the thesis's own Eq. (35)/(3.32): `QoE_{h,m}(q) = 1 - (w^sh*sum E^sh + w^cu*sum E^cu)/E^base_{h,m} >= QoE^min`. Note: daily enforcement (one row per household per DISPATCH day) is a sufficient, slightly conservative substitute for monthly enforcement (one row per household per BILLING month) - satisfying the constraint on every day of the month implies satisfying the average-of-the-month version, but not vice versa, so this is a conservative approximation in the safe direction, not an error |
| H6 | sigma_k from backtest RMSE with a monotone envelope | `gentwin/forecast.py::load_sigma` | Add one sentence under Eq. (13)/(3.4): RMSE_k = sqrt(Var_k + bias_k^2) >= sigma_k, so using it is a documented, deliberately CONSERVATIVE approximation (widens the robust peak, never shrinks it), used because the frozen forecasting pipeline exports only the backtest summary; the exact variance is computed automatically whenever per-record residuals are supplied (`sigma_from_residuals()`) |
| H7 | pi^ret_h for LT-A = the MARGINAL slab rate, not the average | `gentwin/tariff.py::marginal_rate_lt_a`, `billing.py::j1_linear` | State explicitly under Eq. (19)/(3.21) that `pi^ret_h` for a cumulative-slab class is evaluated at the household's marginal slab (the rate on its NEXT kWh), which is what a rational household actually faces when deciding whether to use one more unit - the average rate would understate the incentive at a slab boundary |
| H8 | Individual rationality (Issue I above) | `lp.py`, `06_lp_optimiser.py --ir` | Add as optional constraint C8, off by default, per Issue I |

---

## Issue K - Chapter 4 passages that now contradict the corrected results

Both passages quoted below were read from the thesis text supplied for this
audit (section 4.2.6, "Load Classes and Consumer-Level Outcomes").

### K1. "the local time-of-use window provides an economic signal for shifting"

**As written:** *"This occurs because the local time-of-use window provides
an economic signal for shifting or reducing flexible demand, while the
forecast-based stress condition provides an additional incentive when the
system is under greater pressure."*

**Why it now contradicts the model.** After Issue A, the local
(DR-activation) window carries NO price signal at all - billing is always
by the official 17:00-23:00 window. Shifting inside a misaligned local
window (Dhanmondi) now REDUCES private benefit rather than providing one;
only the system-value term (kappa*pi_i(t)/theta_3) and, for curtailment
only, the genuine kWh-reduction saving, remain as drivers.

**Proposed replacement.** *"This occurs because curtailing genuinely
reduces the energy a household buys, which saves money under any tariff,
and because the forecast-based stress rebate rho (a proposed incentive, not
current BERC policy - Sec. [Issue J]) adds a further inducement on stressed
days. Shifting behaves differently by substation: at Kalyanpur, whose own
congestion window overlaps the official 17:00-23:00 peak, shifting out of
it still lowers the bill under the current tariff. At Dhanmondi, whose
midday congestion window does not overlap the official window at all,
shifting out of it under the CURRENT tariff raises the bill - demand
response there is driven entirely by the system-value term, i.e. by the
value of relieving physical congestion that the household's own bill does
not price. This asymmetry is itself a finding: it is the quantitative case
for a substation-aware ToU tariff, which does not exist today."*

### K2. "most residential observations remain below the saving-discomfort parity line"

**As written:** *"Most residential observations remain below the
saving-discomfort parity line, indicating that their bill savings are
greater than the monetary discomfort assigned to the selected response."*

**Status after the fixes: largely still true, but for a different reason
than implied.** Measured before/after the residential-ToU and Issue-A/G/D
fixes (see the acceptance-check table below): residential bill savings
barely moved, because they were already curtailment-dominated (shifting
under a flat tariff, or a misaligned ToU tariff, was never a material
private benefit once correctly priced). The passage's CONCLUSION survives;
its implicit MECHANISM does not.

**Proposed replacement.** *"Most residential observations remain below the
saving-discomfort parity line. This saving is driven almost entirely by
curtailment - using less energy, which saves money under any tariff -
rather than by shifting, since residential consumers face no time-of-use
pricing under current BERC policy (Sec. [Issue A]) and therefore have no
private financial reason to shift load between slots of otherwise-identical
price."*

---

## Equation numbering, unified

Four different numbering schemes are in play across the codebase, the
internal formulation document, this thesis, and Chapter 4's citations. This
table resolves them; use it to find "the same equation" across any two.

| Meaning | Code comment (`lp.py`, `regime.py`) | Internal formulation (`GenTwinSG_Formulation`) | This thesis (Ch. 3) | Chapter 4 citation |
|---|---|---|---|---|
| Household load, Eq. (2) form | Eq. (2) | Eq. (2) | Eq. (3.10) (under-specified - see H3) | - |
| Load classes / total, Eq. (3) | Eq. (3) | Eq. (3) | Eq. (3.11)-(3.12) | - |
| Post-response load, x/y | Eq. (4) | Eq. (4)/(29) | Eq. (3.13) | - |
| Recovery spread | Eq. (5)/(14) | Eq. (5) | Eq. (3.14) | - |
| Aggregation, Lambda_i(t) | Eq. (6) | Eq. (6) | Eq. (3.15)-(3.16) | - |
| Fixed/variable split, xi(d) | Eq. (7)-(8) | Eq. (7)-(8) | not present (Ch.4 4.1.3 describes it in prose) | - |
| Robust peak p~ | Eq. (14) | Eq. (14) | Eq. (3.5) | Fig. 4.11 |
| Stress indicator s, deficit Def | Eq. (15) | Eq. (15) | Eq. (3.6)-(3.7) | - |
| kappa (Issue D) | Eq. (16) | Eq. (16) | Eq. (3.8) - CORRECTED, see Issue D | Fig. 4.12 ("kappa") |
| mu_i(t) / mu_c(t) (Issue A) | Eq. (19)/(34) | Eq. (19)/(34) | Eq. (3.21) - CORRECTED, see Issue A | Sec. 4.1.5 |
| J1 (household bill) | Eq. (20) | Eq. (20) | Eq. (3.22) | Sec. 4.2.6 |
| J2 (discomfort) | Eq. (21) | Eq. (21) | Eq. (3.23) | - |
| f_wel | - | Eq. (24) | Eq. (3.24) | - |
| Objective F, theta weights (Issue H1) | `th1,th2,th3` in `build_day` | Eq. (24)-(26) | Eq. (3.27) - INCOMPLETE, see H1 | - |
| C1 supply adequacy | "C1", Eq. (29) | Eq. (29) | Eq. (3.31), C1 | - |
| C2 storage bounds | "C2u"/"C2l", Eq. (30) | Eq. (30) | Eq. (3.31), C2 | - |
| C3 reserve floor | "C3", Eq. (31) | Eq. (31) | Eq. (3.31), C3 | - |
| C4 tie-line/export (Issue H4) | "C4", Eq. (32) | Eq. (32) | Eq. (3.31), C4 - INCOMPLETE, see H4 | - |
| C5 resource bounds | variable bounds | Eq. (33) | Eq. (3.31), C5 | - |
| C6 household bounds | variable bounds | Eq. (34) | Eq. (3.31), C6 | - |
| C7 QoE (Issue H5) | "C7", `docs/qoe_definition.md` | Eq. (35) | **MISSING** - see H5 | **"Eq. 35"**, Fig. 4.16 caption |
| Proposition 2 threshold, shift | Eq. (36) | Eq. (36) | not present | - |
| Proposition 2 threshold, curtail (Issue G) | Eq. (37)/(38) - CORRECTED | Eq. (37)/(38) | not present | - |
| Stage G surcharge | Eq. (40)-(43) | Eq. (40)-(43) | not present | Sec. 4.2.6 (`varpi`) |
| Regime r_{i,d} | Eq. (48) | Eq. (46) | not present | - |
| Entity records / phi (Issue E1) | Eq. (44)-(47) - phi1/phi2 CORRECTED | Eq. (44)-(47) | not present (Sec. 3.4, prose only) | - |
| chi_e criticality score | Eq. (49) | Eq. (47) | not present | - |
| Pools P_H/P_M/P_L (Issue E3) | Eq. (50) - now quantile-default | Eq. (50) (base paper Eq. 4) | not present | - |
| W(P), w_e (Issue E2) | Eq. (51) - CORRECTED | Eq. (51) | not present | - |
| Re-pooling alpha_P (Issue E4) | Eq. (52) - now load-bearing | Eq. (52) | not present | - |
| Top-N (Issue E5) | Eq. (53) - now stratified | Eq. (53) | not present | - |

Chapter 4's "Eq. 35" (Fig. 4.16 caption, "C7, Eq. 35") is the ONLY place the
thesis text as given cites a numbered equation that Chapter 3 (as given)
never introduces. Resolving H5 (inserting `docs/qoe_definition.md`'s
definition as the thesis's own numbered equation, whatever number it ends
up at once the LaTeX is available) makes that citation resolve to something
that actually exists in the document, rather than to the internal
formulation's numbering.
