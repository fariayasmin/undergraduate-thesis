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
(18:00-21:00). Full rows: `outputs/held_out_t2_{substation}.csv`.

**Caveat, added per Follow-up F17 - read Kalyanpur's 24/24 as weak, not
strong, validation.** This "inside T^pk_i" test is lenient: Kalyanpur's
synthetic peak slot is a FIXED 18:30 every single day (an evening
residential/AC block that always dominates, never varies with weather),
which sits inside its own tuned window by construction - 24/24 here is
close to guaranteed, not independent evidence of generalisation. Follow-up
F17's STRICTER test (does the synthetic peak SLOT match the ACTUAL BPDB
peak TIME, not just "is it inside a window drawn around the synthetic
answer") found Kalyanpur's hit rate exactly EQUALS a naive always-guess-
the-mode baseline (35.7%=35.7%, zero lift) - the honest conclusion for
Kalyanpur is that the model's TIMING generalises to "always inside its own
window" (trivial) but NOT to "actually matches when Kalyanpur really
peaks" (real, and it does not). Dhanmondi does not have this problem - its
synthetic peak varies day to day and beats the naive baseline by a real
margin (87.5% vs 83.3%, Follow-up F17).

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

## Follow-up F16. rho never reached 09_monthly_billing.py either (item A2)

**What was wrong.** The same propagation gap as Follow-up F11
(`06_lp_optimiser.py::apply_scenario()` mutating `cfg` in its own process
only), but for `rho`, and this one changed a REPORTED NUMBER, not just a
label. `09_monthly_billing.py` computes `incentive_received_tk` (and, via
`j1_linear()`, `rebate_tk`) from `cfg.RHO_REBATE_TK_PER_KWH` fresh in its
own process - so the `rho0` (current-policy, no-rebate) arm's billing was
silently computed as if the default rho=3 Tk/kWh rebate still applied.
Confirmed directly: `billing_statistics_period_rho0.json` reported
`"rho_tk_per_kwh": 3.0`, and the exact same 10 household IDs, with
bit-for-bit identical `saving_tk`/`incentive_received_tk`/`optimised_kwh`,
registered as "gained" in `base`, `rho0` AND `scarcity` - not a coincidence
of the boundary being far from zero (verified separately for `scarcity`,
where the underlying numbers genuinely differ but the boolean happens to
agree), but for `rho0` a real computation done with the wrong rho. The
same gap affected `individual_rationality_on`/`curtail_scope` LABELS
(cosmetic only - no computed number depends on them) in both
`09_monthly_billing.py`'s JSON and `gentwin/kg.py::build_billing()`'s
`BillingPeriod` node.

**Fix.** `06_lp_optimiser.py::export()` now writes `rho_used_tk_per_kwh`,
`rho_current_policy`, `individual_rationality_on`, `curtail_scope_used`
into `lp_summary{tag}.csv`. `09_monthly_billing.py::main()` reads them and
OVERRIDES its own `cfg.RHO_REBATE_TK_PER_KWH`/`cfg.RHO_CURRENT_POLICY`
before computing anything, and uses the read values (not `cfg` directly)
for its JSON labels. `08_knowledge_graph.py` passes the same four fields
(read from the now-correct `billing_statistics{tag}.json`) into
`kg.build_billing()`, which now prefers them over its own process's `cfg`.

**Verified after re-solving `_period_rho0`:** `rho_tk_per_kwh` now reads
`0.0`, `rho_is_current_policy` reads `true`, total `incentive_received_tk`
is exactly `0.0` (was 146,255 Tk, matching the base arm's non-zero rebate
total almost exactly). `pct_connections_gained` moved from the buggy 1.24%
(identical to base) to the correct **0.20%**; `pct_connections_gained_lambda`
from 14.84% to **12.14%**. This is a real, previously-unreported difference
between the current-policy and proposed-mechanism arms that the bug had
been hiding.

---

## Follow-up F17. Kalyanpur's synthetic peak TIMING does not beat a naive baseline

**What was checked (item B6).** Distinct from the held-out T2 check
(Follow-up F9, which only asks "does the peak land inside the DR window"),
this compares the synthetic model's predicted peak SLOT against the
BPDB-recorded ACTUAL peak time for the same 24 held-out dates, and against
an "always guess the modal slot" baseline.

**Dhanmondi**: hit rate (within +/-1 slot) = 87.5% (21/24) vs the
always-modal-slot baseline's 83.3% (20/24) - a real, if modest, lift.
Peak-level ratio (synthetic/actual) averages 1.09 (well-calibrated).

**Kalyanpur**: hit rate = 35.7% (5/14 valid, non-`0:00`, records) - EXACTLY
equal to the always-modal-slot baseline's 35.7%. The synthetic model
provides ZERO timing lift over a trivial constant guess, because its own
predicted peak slot is itself CONSTANT (18:30, every single day, driven by
a fixed evening residential/AC block that always dominates regardless of
weather) while the actual BPDB peak time ranges from 12:00 to 23:00.
Peak-level ratio averages 1.53 - the synthetic model also overstates
Kalyanpur's peak magnitude by roughly half.

**A likely structural cause, found from the raw BPDB data itself**: over
all of 2025 (256 valid, non-`0:00` days out of 358), Kalyanpur's ACTUAL
peak lands in 12:00-14:00 on 39.5% of days - MORE often than in the
18:00-21:00 window (27.0%), which is both the substation's adaptive
DR-activation window (`T^pk_Kalyanpur`) AND the synthetic model's only
possible peak slot. 28.5% of 2025 records are `0:00` (a data-quality gap,
not necessarily a genuine midnight peak).

**Recommendation, not a fix made this round**: the audit brief's RULES for
this round exclude changing the load model itself; this is reported as a
genuine, disclosed limitation rather than corrected. Chapter 4 (or a
future pass) should state that the synthetic model's Kalyanpur peak timing
is well below a trivial baseline, and that BPDB's own actual data suggests
Kalyanpur may have a materially noon-peaking sub-population the current
synthetic mix under-represents.

---

## Follow-up F18. Scarcity fairness: exporters curtail their own load in the same slots (item B8)

**What was checked.** For the scarcity arm (0.55/0.62), per substation-day
where that substation is EXPORTING (`T_out_kw>0` in some slot) AND is
itself in `capacity_deficit` that day (using the correctly-propagated
`p_max_kw_eff`, Follow-up F11), how much of that substation's OWN
curtailment falls in the SAME slots it is exporting in.

**Result**: 18 of 30 exporting substation-days meet both conditions. On
the most extreme, 2026-06-28, Dhanmondi exports 74,792 kWh while
curtailing 144,772 kWh of its own households' load in those same slots -
almost double the exported amount. On 2026-07-12, Kalyanpur exports
81,138 kWh while curtailing 191,571 kWh of its own load in those slots.

**This is not presented as a bug** - the coupled LP is doing exactly what
Eq. (23)/(29)'s system-wide welfare objective asks: relieve the WORSE-OFF
substation even at a cost to the better (but still stressed) one, when
that raises total welfare. It is, however, a genuine DISTRIBUTIONAL/
fairness finding worth stating plainly in Chapter 4: under this scarcity
scenario, the model's coupling asks an already-stressed substation to
export power while curtailing its own residents in the same half-hours -
a burden-sharing pattern that has policy content (is it fair, and would a
BPDB operator actually do this) independent of whether the dispatch
itself is efficient.

---

## Follow-up F19. Social efficiency: curtailment is privately, not socially, rational under normal conditions (item C12)

**What was checked.** Post-hoc from `lp_thresholds{tag}.csv`, no LP or
objective change: what share of curtailed kWh satisfies
`pi_i(t) + pi4_i(t) - gamma_{cu,h} > 0` (the RAW, unweighted system-price
test - not the lambda-weighted comparison the LP itself uses), and what
share of shifted kWh satisfies `(pi(tau) - mean(pi_off)) - gamma_{sh,h} > 0`.

**Result: 0.0% in every arm without a scarcity scenario** (base,
dr_window, rho0, ir, stress_days, drw_ir) - for both curtailment and
shifting. The mechanism is structural, not a bug: `pi_i(t)` sits flat at
the producer price (8.39 Tk/kWh; C1 never binds without a scarcity
scenario) while `gamma_{cu,h}` (recovered from
`discomfort_tk_per_kwh/(1-lambda_h)`) ranges 11.6-36.5 Tk/kWh - the raw
price signal never clears the raw discomfort weight. Every curtailment the
LP chooses is worthwhile only under the household's OWN lambda-weighted
comparison, never under the unweighted, "value-to-the-grid-alone" one.

**Under scarcity (0.55/0.62, with pi4 now exported per Follow-up F20
below)**: 10.96% of curtailed kWh becomes socially efficient once C4's
dual is included (610 of 304,185 curtail-actions have `pi4>0`) - the
system price can now exceed gamma on the days C4 actually binds. Shifted
kWh remains 0.0% (the shift test does not involve `pi4` at all, and the
shift-window price spread stays small relative to `gamma_{sh,h}`).

**Recommendation for Chapter 4**: state this explicitly rather than
implying (as the current K2 passage's mechanism did) that curtailment
reflects system value. It reflects PRIVATE value, mediated by lambda_h;
system value only enters once real scarcity pushes the shadow price above
the discomfort weight, and even then covers roughly a tenth of the
curtailed energy. See the rewritten K2 above and the "Incentive
compatibility and benefit sharing" section below for the full
decomposition.

---

## Follow-up F20. C4's dual (pi4) had no CSV export

**What was missing.** `pi4_i(t)` (C4's dual, already computed and used
internally since Follow-up F4) was never written to any output file -
`lp_network{tag}.csv` carried `pi_tk_per_kwh` (C1's dual) but nothing for
C4, so Follow-up F19's social-efficiency check needed it and had none to
read for the scarcity arm (it is exactly 0, and thus omittable without
changing the RESULT, for every arm that never scales P^max).

**Fix.** `06_lp_optimiser.py::export()`'s network rows now include
`pi4_tk_per_kwh: sol.duals_c4.get(i, zeros)[t]`. Verified this is a pure
reporting addition: re-solving the scarcity arm to pick it up reproduced
`shift_kwh`/`curtail_kwh` and every check-6 margin bit-for-bit identical to
the pre-addition run.

---

**LP-level, full 30-day period:**

| arm | shift kWh | curtail kWh | n_events | degenerate pool-days | check 6 |
|---|---|---|---|---|---|
| base | 54,328 | 11,389,227 | 86 | 0/30 | PASS |
| dr_window | 135,290 | 3,088,306 | 86 | 0/30 | PASS |
| rho0 | 60,048 | 11,078,285 | 86 | 0/30 | PASS |
| ir | 93,995 | 5,171,419 | 86 | 0/30 | PASS (after F15) |
| scarcity (0.55/0.62) | 1,155,314 | 12,746,020 | 124 | 0/30 | PASS |

(shift/curtail kWh here are already population-scale, like every other
kWh figure in this document - see the Round-4/item-1 correction under
C10/C11 for the sanity check confirming this and the double-weighting
bug it caught in a DIFFERENT, downstream calculation.)

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
| rho0 | 432 | 4.00 | 2,145.5 | **0.20** | **12.14** | 3.27 |
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

| arm | same-day (circular) | mean_relative (circular) | chi excl. phi2 (non-circular, same-day) | next-day, RANKING (non-circular) | next-day, PERSISTENCE baseline | next-day random baseline | next-day lift (ranking) |
|---|---|---|---|---|---|---|---|
| base | 1.000 | 1.000 | 0.706 | 1.000 | 0.985 | 0.265 | 3.80x |
| dr_window | 1.000 | 1.000 | 0.706 | 0.995 | 0.976 | 0.268 | 3.74x |
| rho0 | 1.000 | 1.000 | 0.695 | 1.000 | 0.994 | 0.255 | 3.94x |
| ir | 1.000 | 1.000 | 0.349 | 0.968 | 0.985 | 0.123 | 7.91x |
| scarcity | 1.000 | 1.000 | 0.806 | 1.000 | 0.949 | 0.452 | 2.31x |

The "next-day, PERSISTENCE baseline" column (item A5) is the one that
matters for reading this table correctly - see the Round-4 correction
below.

Same-day precision is 1.000 on both quantile and mean_relative for every
arm - confirming the circularity diagnosis (F14): whichever pooling RULE
is used, phi2 alone (built from the acting label) separates acting from
non-acting entities perfectly within a kind/day, so same-day precision
cannot distinguish a good ranking from a tautological one. Dropping phi2
from chi (still same-day, but the score can no longer see the label) drops
precision to 0.35-0.81 - real signal from graph centrality/stress/deficit
alone, well above what a same-size random draw would get, but nowhere near
perfect.

**Round-4 correction (item 6).** Next-day precision (the one variant that
cannot leak the label) is high (0.968-1.000), but this round's own
persistence baseline (item A5 - today's actual actors, sampled to the same
per-kind quota, no ranking involved) is ALSO high and tracks it closely:
base 0.985 vs 1.000, dr_window 0.976 vs 0.995, rho0 0.994 vs 1.000, ir
0.985 vs 0.968 (persistence slightly EXCEEDS the ranking here), scarcity
0.949 vs 1.000. The gap between next-day precision and the persistence
baseline is at most ~0.05 in every arm, and negative in one. The correct
reading is therefore NOT "the ranking finds a genuine, non-circular
signal about persistence" (the previous round's claim) - it is that
MOST of next-day precision's lift over random comes from ordinary
day-to-day persistence of WHO is active in this population, which a
trivial "watch today's actors" rule captures almost as well as chi's
ranking does. The ranking's OWN marginal contribution over that trivial
baseline is small (and inconsistent in sign) in every arm tested. This
does not mean the pooling mechanism is worthless - `chi_excluding_phi2`
still beats random by a real margin same-day - but the specific claim that
next-day precision demonstrates the RANKING's predictive value, rather
than the population's own persistence, does not hold up against this
round's own comparison.

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

**The figure this passage describes** (`09_monthly_billing.py::_draw_savings_vs_discomfort_panel`,
panel of `monthly_billing{tag}.png`): a scatter of one point per household
representative, coloured by category, with **X-axis = discomfort J^h_2
(Tk)** and **Y-axis = bill saving (Tk)**, plus a dashed 45-degree line
`y=x` labelled "saving = discomfort" - the parity line. A point ABOVE the
line has `saving > discomfort`; a point BELOW it has `saving < discomfort`.

**Status: CONTRADICTED, not confirmed.** The passage's own two clauses
are inconsistent with each other on THIS figure's axis convention: being
"below" the `y=x` parity line means `saving < discomfort` (Y is below what
the line predicts for that X), not `saving > discomfort` as the passage's
second clause claims. Reading the passage's INTENDED claim charitably as
"residential participants mostly come out ahead" (saving exceeds
discomfort), the measured result is the opposite: `pct_connections_gained`
(unweighted, one Tk equals one Tk, share of **ALL 800 representatives /
~348,661 connections - not just participants**) is **1.2%**, and even
`pct_connections_gained_lambda` (the LP's OWN lambda-weighted criterion,
the most favourable reading available) is only **14.8%**. Under either
definition, the overwhelming majority of ALL residential connections -
not merely a minority of participants - sit BELOW the parity line
(`saving < discomfort`, or fail the lambda-weighted test). The passage's
claim that most observations show savings exceeding discomfort is
therefore directly contradicted by the measured figure, on its own axis
convention.

**A second, independent reason for the same conclusion (Round-3 item
C12/Follow-up F19).** Beyond residential's lack of time-of-use exposure
(below), literally NONE of the curtailed or shifted energy in ANY arm
without a scarcity scenario (base, dr_window, rho0, ir, stress_days,
drw_ir - checked directly against `lp_thresholds{tag}.csv`) satisfies the
SOCIAL-efficiency test `pi_i(t) - gamma_{cu,h} > 0` (curtail) or
`(pi(tau) - mean(pi_off)) - gamma_{sh,h} > 0` (shift): the raw system price
`pi_i(t)` sits flat at 8.39 Tk/kWh (no congestion, G^max unbound) while raw
`gamma_cu_h` ranges 11.6-36.5 Tk/kWh - the price signal alone never clears
the discomfort cost. Every household that curtails is doing so because its
OWN lambda-weighted comparison (`lambda_h * private benefit` against
`(1-lambda_h) * discomfort`) favours it, not because the unweighted,
planner's-eye view (or the figure's own unweighted axes) would call the
action worthwhile. This is not a bug - it is the honest reading of what
"demand response" is doing in this model under normal (non-scarcity)
conditions: a private transaction the LP mediates, not a system-value-
creating one, and not one that leaves most residential connections ahead
on a plain Tk-for-Tk basis either. Only the scarcity arm, where `pi_i(t)`
and C4's own dual can reach the VoLL price (60 Tk/kWh), gives the price
signal room to exceed gamma at all.

**Proposed replacement.** *"Most residential observations remain below
the saving-discomfort parity line (Fig. [X]: discomfort J^h_2 on the
x-axis, bill saving on the y-axis, dashed y=x parity line) - CONTRARY to
a reading where savings exceed discomfort for most participants. Measured
over the full population (not just participants): only 1.2% of ALL
residential connections show a bill saving exceeding their discomfort
under the plain, unweighted Tk criterion; even under the LP's own
lambda-weighted criterion (the most favourable available reading), only
14.8% do. This saving is driven almost entirely by curtailment - using
less energy, which saves money under any tariff - rather than by
shifting, since residential consumers face no time-of-use pricing under
current BERC policy (Sec. [Issue A]) and therefore have no private
financial reason to shift load between slots of otherwise-identical
price. Beyond the tariff structure, the curtailment itself is privately
but not socially rational under normal conditions: the system price never
rises far enough above the discomfort weight gamma_{cu,h} to justify
curtailment on value-to-the-grid grounds alone (Sec. [Incentive
compatibility and benefit sharing]) - only under the scarcity scenario,
where the price can reach the value of lost load, does that comparison
turn favourable."*

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

---

# Incentive compatibility and benefit sharing (Round-3 items C10-C16)

No LP/objective/config-default change underlies anything in this section.
Every number is read from a tag file this round actually produced; every
new arm is listed in "Exact commands" in the closing report. `dr_window +
IR` (tag `_period_drw_ir`) is evaluated here as the CANDIDATE main
reported arm, per this round's own framing - not as a replacement of the
base formulation, which every table below reports alongside it.

## IR's interpretation

The individual-rationality constraint
(`lambda_h*(J1(x,y)-J1(0,0)) + (1-lambda_h)*J2(x,y) <= 0`) guarantees, BY
CONSTRUCTION and verified numerically (Follow-up F15's fix, item A3), that
almost every participating household's own LP-marginal-price bill
(J1) improves enough to outweigh its lambda-weighted discomfort: 99.0% of
`ir` arm participants satisfy the lambda-weighted criterion under J1,
falling to 75.4% under the GAZETTE bill a household actually receives
(slab/VAT structure, not the LP's own criterion, explains the entire gap -
Follow-up "IR gained_lambda" note, item A3). IR is therefore best read as
a PARTICIPATION filter on the LP's own optimism, not a guarantee that the
household's real bill improves - Chapter 4 should state both numbers, not
just the LP's internal one.

## rho as a peak-time (stress-day) rebate

`rho` (Eq. 20's stress-rebate) is paid only when `s_{i,d}=1` (a stressed
day) AND the household curtails - `incentive = rho * E^cu_{h,d} * s_{i,d}`.
It is NOT a general demand-response payment; on a non-stressed day it pays
nothing regardless of how much a household curtails. This is why the rho
sweep below (item C11) shows curtailed kWh rising only modestly with rho
(most curtailment happens on non-stressed days, where rho never applies)
while `%gained` rises much more sharply - the households near the margin
are disproportionately the ones already curtailing on the FEW stressed
days rho actually pays on.

## The decomposition pi - gamma = (pi - p) + (p - gamma)

Three different comparisons matter for THREE different parties, and this
round's items keep confusing them if not stated separately:

- `pi_i(t) + pi4_i(t) - gamma_{cu,h}` (or the shift analogue): the PLANNER's
  test - is the action worth it to the SYSTEM alone, ignoring who pays whom.
  Item C12's result: 0.0% of curtailed/shifted kWh passes this without a
  scarcity scenario; 10.96% of curtailed kWh passes it under scarcity
  (Follow-up F19).
- `pi_i(t) + pi4_i(t) - p_h(t)` (the "avoided cost" side of item C11's
  ledger): the OPERATOR's test - does curtailment save more in system/
  wholesale value than it costs in foregone retail revenue. Structurally
  NEGATIVE under normal conditions (`pi` sits far below the household's
  retail rate `p_h(t)`) and strongly POSITIVE under scarcity (`pi` can
  reach the VoLL price, far above `p_h(t)`).
- `p_h(t) - gamma_{cu,h}` (embedded in `priv - disc`, what the LP's own
  lambda-weighted objective actually decides on, for lambda_h close to 1):
  the HOUSEHOLD's test - does its own bill saving exceed its own
  discomfort. This is the one the LP optimises; the other two are
  post-hoc, non-binding checks on the SAME decision.

The three tests can and do disagree - see item C12's finding that
curtailment is essentially always privately rational (the household's own
test) while essentially never planner-optimal (the system test) outside
scarcity.

## C10/C11: the rho sweep on dr_window+IR, base and severe-scarcity conditions

**Arm**: `_period_drw_ir` (`--curtail-scope dr_window --ir`), confirmed
with **0 dangling KG relationships** (acceptance check, see below). The
rho sweep runs `rho in {0,3,6,9,12,15}` on top of it, both without
scarcity (tag `_period_drw_ir_rho{v}`) and with the severe scarcity
scenario (0.55/0.62, tag `_period_drw_ir_scar_rho{v}`) - 12 runs total, all
check-6 PASS.

**Round-4 correction (item 1).** The avoided-cost/budget-balance columns
below were originally computed by multiplying `lp_thresholds{tag}.csv`'s
`energy_kwh` by `w_h` a SECOND time. `energy_kwh` is already population-
scale: `gentwin/lp.py::_consumer_blocks()` multiplies each household's raw
load profile by `h.w` BEFORE it ever reaches the LP (`blocks[...]["cu"] =
prof["curtailable"] * w`), so `yv * blk["cu"][t] * D` (how `energy_kwh` is
built in `threshold_table()`) is already a population total. Sanity check
confirming this (as this round's instructions specified): total curtailed
kWh across the whole month is 0.89% of total substation import energy
(`sum(G_kw)*Delta` = 127.0M kWh vs curtail_kwh = 1.14M kWh for
`drw_ir_rho0`) - a plausible SYSTEM-scale ratio, and matches
`billing.py`'s independently-computed population total energy (128.5M kWh)
to within 1%. Multiplying by `w_h` again squared the weight, inflating
avoided cost into the billions/tens-of-billions where it should read in
the millions/tens-of-millions. `rebate_paid_tk` (sourced from
`billing.py`'s OWN `incentive_received_tk`, which is genuinely
per-representative and DOES need one `w_h` multiplication - a DIFFERENT
code path with a different, correct convention) was never affected.
Corrected table below; the "*" footnote on kWh columns is also corrected -
they were mislabelled "per-representative" when they are already
population-scale.

| condition | rho | participation % | curtail kWh* | shift kWh* | peak red. Dhanmondi % | peak red. Kalyanpur % | %gained (unweighted) | %gained (lambda) | % of PARTICIPANTS gained (unweighted) | % of PARTICIPANTS gained (lambda) | rebate paid (Tk) | avoided cost (Tk) | budget balance (Tk) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | 0 | 13.49 | 1,135,706 | 135,290 | 3.15 | 7.23 | 0.93 | 11.97 | 6.9 | 88.7 | 0 | -7.92e6 | -7.92e6 |
| base | 3 | 18.51 | 1,324,212 | 135,290 | 3.15 | 7.44 | 3.00 | 17.70 | 16.2 | 95.6 | 2.06e6 | -9.05e6 | -1.11e7 |
| base | 6 | 23.22 | 1,484,208 | 135,290 | 3.15 | 7.39 | 6.60 | 22.18 | 28.4 | 95.5 | 5.08e6 | -9.89e6 | -1.50e7 |
| base | 9 | 27.06 | 1,619,749 | 135,290 | 3.15 | 7.42 | 13.51 | 26.52 | 49.9 | 98.0 | 8.84e6 | -1.06e7 | -1.94e7 |
| base | 12 | 30.49 | 1,708,545 | 135,290 | 3.15 | 7.45 | 22.65 | 30.24 | 74.3 | 99.2 | 1.29e7 | -1.11e7 | -2.39e7 |
| **base** | **15** | 33.60 | 1,818,642 | 135,290 | 3.15 | 7.39 | 28.96 | 33.45 | **86.2** | 99.5 | 1.77e7 | -1.17e7 | -2.94e7 |
| scarcity | 0 | 13.49 | 1,135,882 | 432,927 | 3.06 | 7.22 | 0.69 | 11.51 | 5.1 | 85.3 | 0 | +5.33e7 | +5.33e7 |
| scarcity | 3 | 18.51 | 1,324,397 | 525,635 | 3.03 | 7.14 | 1.86 | 17.22 | 10.0 | 93.0 | 2.06e6 | +6.34e7 | +6.13e7 |
| scarcity | 6 | 23.22 | 1,484,392 | 576,420 | 3.01 | 7.10 | 3.97 | 21.69 | 17.1 | 93.4 | 5.08e6 | +7.07e7 | +6.56e7 |
| scarcity | 9 | 27.06 | 1,619,849 | 625,771 | 2.99 | 7.06 | 9.14 | 24.88 | 33.8 | 91.9 | 8.84e6 | +7.71e7 | +6.83e7 |
| scarcity | 12 | 30.49 | 1,708,643 | 658,792 | 2.97 | 7.04 | 19.14 | 29.27 | 62.8 | 96.0 | 1.29e7 | +8.13e7 | +6.84e7 |
| scarcity | 15 | 33.60 | 1,818,740 | 694,716 | 2.96 | 7.01 | 25.20 | 32.73 | **75.0** | 97.4 | 1.77e7 | +8.65e7 | +6.87e7 |

*shift/curtail kWh are already POPULATION-SCALE (see the Round-4
correction above, and B/A4's earlier population comparisons - NOT
per-representative); every Tk column is likewise population-scale
(~348,661 connections).

**rho\* (smallest rho with >=80% of PARTICIPATING connections gaining).**
Only the UNWEIGHTED definition is informative here - report it as the
finding: **base = 15** (reaches 86.2%, first value at or above 80%);
**scarcity = NOT REACHED** (only 75.0% even at rho=15). The
LAMBDA-weighted definition is **not a finding**: it reads `rho*=0` for
BOTH conditions purely because the IR constraint (Follow-up-1's fix,
verified in item A3) is ITSELF written in lambda-weighted terms, so a
participant satisfying IR at all is close to satisfying the lambda-
weighted gain criterion by construction, independent of rho. Reporting
`rho*=0 (lambda)` as if it were a discovered result would be circular;
Chapter 4 should state plainly that the lambda-weighted rho\* is a
tautology of the IR constraint's own definition, not evidence that rho=0
is "enough."

**Reading the corrected table.** rho\*'s two answers (15 vs "not reached,"
under the one INFORMATIVE - unweighted - definition, across the two
conditions) still make the same point as before: real scarcity structurally
worsens fairness under the strict criterion, no rebate tested fixes it. The
avoided-cost/budget-balance columns, now correctly scaled (millions and
tens-of-millions of Tk, not billions), tell the same QUALITATIVE story as
before the fix, just at the right magnitude: under normal conditions every
rho tested leaves the operator's avoided-cost ledger negative (-7.9M to
-11.7M Tk/month) - curtailment costs more in foregone retail revenue than
it saves in wholesale terms, so the DR mechanism absent real scarcity is a
modest wealth transfer INTO participating households, not a cost-saving
one for the operator. Under scarcity, avoided cost is sharply positive
(+53.3M to +86.5M Tk/month) and budget balance stays strongly positive
(avoided cost exceeds rebate paid at every rho tested) - the operator could
afford considerably more than a 15 Tk/kWh rebate and still come out ahead
DURING SCARCITY specifically. (Per-kWh sanity check: avoided cost per
curtailed+shifted kWh is a believable -6.0 to -6.2 Tk/kWh under base and
+34.0 to +34.4 Tk/kWh under scarcity - consistent with `pi_i(t)`~8.4 Tk/kWh
vs typical retail rates ~14-17 Tk/kWh under base, and `pi_i(t)+pi4_i(t)`
reaching well above retail under scarcity.)

## Round-5 item 2: the severe scarcity scenario is a sustained shortage, not a peak event

For the severe scarcity scenario (0.55/0.62), the number of half-hour
slots per day where BASELINE load (before any DR) already exceeds the
scaled effective P^max:

| substation | mean slots/day over P^max | mean hours/day | min-max slots/day | days with any overage |
|---|---|---|---|---|
| Dhanmondi | 3.57 | 1.78 h | 2-7 | 30/30 |
| Kalyanpur | 21.20 | 10.60 h | 18-25 | 30/30 |

**Kalyanpur's baseline load exceeds its effective P^max for close to
HALF the day, EVERY day of the 30-day period** (mean 21.2 of 48 slots,
i.e. 10.6 hours) - this is a sustained supply-adequacy shortage, not a
brief peak excursion. Dhanmondi is milder (mean 1.78 hours/day) but still
shows overage on all 30 days, never a zero-overage day. Chapter 4 should
describe this scenario as a SUSTAINED supply-shortage stress test (most
of the day, every day, for Kalyanpur) rather than as a "peak event" -
the latter framing would understate how binding the scarcity scale
actually is and would misdescribe what the reported Z/VoLL/regime results
are actually stress-testing.

## Item 3: scarcity under IR - unserved load never goes away, even with the rebate

The severe scarcity scenario (0.55/0.62) UNDER `drw_ir` (dr_window-scoped
curtailment + individual rationality) leaves FAR more unserved load than
the unrestricted "compulsory" scarcity arm (item C13's original two rows):
total `Z` starts at 7.39M kWh (rho=0) and falls only to 6.53M kWh (rho=15,
a 12% reduction) - two to three orders of magnitude above the compulsory
arm's 19,237 kWh, because restricting curtailment to the DR window and to
individually-rational actions removes almost all of the flexibility that
arm had available to absorb the shock.

| rho | Dhanmondi Z_cu (kWh) | Kalyanpur Z_cu (kWh) | Total Z (kWh) | VoLL cost (Tk) |
|---|---|---|---|---|
| 0 | 3,286,799 | 4,100,787 | 7,387,586 | 443,255,200 |
| 3 | 3,145,116 | 3,986,060 | 7,131,176 | 427,870,600 |
| 6 | 3,047,242 | 3,890,014 | 6,937,256 | 416,235,400 |
| 9 | 2,962,427 | 3,808,242 | 6,770,669 | 406,240,200 |
| 12 | 2,905,138 | 3,755,033 | 6,660,171 | 399,610,200 |
| 15 | 2,857,933 | 3,675,312 | 6,533,245 | 391,994,700 |

`Z_critical`/`Z_shiftable` are exactly 0 at every rho, same as the
compulsory arm. See "Round-6 item 3" below (after C13) for why this is a
modelling ASSUMPTION about the mechanism available to the optimiser, not
a discovered protection behaviour worth reporting as a finding on its
own.

## Round-5 item 1: decomposing the voluntary-vs-compulsory gap - window restriction alone, IR alone, and both together

The previous round attributed the entire gap between "voluntary"
(`drw_ir`, both restrictions) and "compulsory" (all-scope, neither
restriction) DR to "restricting curtailment to individually-rational,
DR-window-scoped actions" as a single, combined cause. Two new
disentangling runs (severe scarcity 0.55/0.62 with EACH restriction
applied alone) show that attribution was too coarse, and the real
mechanism is an INTERACTION, not two additive costs:

| arm | Dhanmondi Z (kWh) | Kalyanpur Z (kWh) | Total Z (kWh) | VoLL cost (Tk) | vs no-DR |
|---|---|---|---|---|---|
| no DR (`--no-dr`) | 3,693,116 | 4,947,303 | 8,640,419 | 518,425,161 | - |
| `dr_window` ONLY (no IR) | 1,648,756 | 2,537,598 | 4,186,355 | 251,181,300 | -51.6% |
| IR ONLY (all-day scope, no window) | 1,869,502 | 2,277,516 | 4,147,018 | 248,821,100 | -52.0% |
| `drw_ir` (BOTH), rho=0 | 3,286,799 | 4,100,787 | 7,387,586 | 443,255,200 | -14.5% |
| `drw_ir` (BOTH), rho=15 | 2,857,933 | 3,675,312 | 6,533,245 | 391,994,700 | -24.4% |
| compulsory DR (all-scope, no IR) | 12,001 | 7,236 | 19,237 | 1,154,217 | -99.8% |

**Each restriction ALONE cuts unserved load by roughly half** (-51.6%
for `dr_window` alone, -52.0% for IR alone, essentially the same
magnitude from two different mechanisms). **Applied TOGETHER, they do
NOT compound additively to a bigger cut - they compound to a much
SMALLER one** (-14.5% at rho=0, only reaching -24.4% at rho=15): the
combined arm leaves MORE unserved load than either restriction does on
its own. The reason is set intersection, not set addition: `dr_window`
restricts curtailment to WHEN it can happen (only within the DR-activation
window); IR restricts it to WHICH households curtail (only where
individually rational, evaluated over the WHOLE day's welfare). A
household that is individually-rational to curtail at some slot OUTSIDE
the window, or a window-slot curtailment that is not individually
rational once its day-level welfare is checked, is excluded once BOTH
apply, even though EITHER restriction alone would have allowed it. The
intersection of "individually rational" and "inside the window" is
considerably smaller than either set alone - which is exactly what the
table shows.

**Revised conclusion (replaces the previous round's single-cause
attribution).** The gap between voluntary and compulsory DR is NOT mainly
"the cost of IR" or "the cost of window-scoping" individually - either
one alone would still recover roughly half of compulsory DR's benefit.
It is the cost of REQUIRING BOTH simultaneously, which is disproportionately
larger than either requirement's own cost, because their eligible sets
intersect narrowly rather than overlapping broadly. A policy design that
wants to keep IR (household consent) without paying this full compounded
price should consider relaxing the window restriction (or vice versa) -
this round's own data shows either relaxation alone recovers about half
of compulsory DR's system-level benefit, while paying for both restrictions
recovers less than a quarter of it even at the highest rebate tested.

## Item 4: peak ceiling - dr_window's own peak is often outside its own window

For the `drw_ir` BASE rho sweep, per-day post-response peak SLOT vs the
substation's own DR-activation window (`T^pk_i`):

| substation | rho | days with post-response peak OUTSIDE T^pk_i | mean daily peak reduction (kW) |
|---|---|---|---|
| Dhanmondi | any (0-15) | **30/30** | ~4,194 (flat across rho) |
| Kalyanpur | 0 | 13/30 | ~9,794 |
| Kalyanpur | 3-15 | 22/30 | ~10,000-10,090 |

**Dhanmondi's post-response peak lies outside its own DR window on EVERY
single day, at every rho tested** - the `dr_window`-scoped curtailment
mechanism never actually touches Dhanmondi's true peak slot at all.
Kalyanpur's peak stays outside its window on 13-22 of 30 days depending on
rho - worse as rho rises, though its peak reduction (~9,800-10,100 kW) is
much larger in absolute terms.

**Round-5 item 3 correction - the mechanism is a hard ceiling, not
"off-window curtailment leaking Follow-up F3's double-count."** The
previous round attributed the flat peak reduction to rho-driven
curtailment being spent off-window (the F3 double-counting signature).
That explanation does not fit the facts: peak reduction stays flat
regardless of rho because there is a HARD, geometric CEILING that no
amount of in-window curtailment can cross, independent of rho entirely.

The correct mechanism: `dr_window`-scoped curtailment can only ever lower
load INSIDE the window. Once in-window load is pushed low enough, the
window's own peak drops BELOW the highest load already occurring OUTSIDE
the window that day - at which point the substation's DAILY peak simply
BECOMES that out-of-window slot, and no further in-window curtailment can
reduce it at all (curtailing inside the window no longer touches whatever
the new peak is). The maximum a window-only mechanism can EVER achieve is
therefore capped at `baseline peak - highest out-of-window baseline
slot`, computed directly from `lp_network_period_drw_ir_rho0.csv`'s
`baseline_kw` (pre-DR, so independent of rho):

| substation | mean baseline peak (kW) | mean highest OUT-of-window baseline slot (kW) | ceiling = max achievable reduction (kW) | ceiling as % of baseline peak | observed mean peak reduction (kW) |
|---|---|---|---|---|---|
| Dhanmondi | 132,089 | 127,895 | **4,194** | 3.2% | ~4,194 |
| Kalyanpur | 135,642 | 125,292 | **10,350** | 7.6% | ~9,800-10,090 |

The observed peak reduction (from the rho sweep) matches this ceiling
almost exactly for Dhanmondi (4,194 vs a 4,194 ceiling - running EXACTLY
against it) and sits just under it for Kalyanpur (9,800-10,090 vs a 10,350
ceiling). This is why peak reduction stays flat as rho rises: rho can
still buy MORE curtailment (confirmed - curtailed kWh keeps rising with
rho), but that extra curtailment cannot buy any MORE peak reduction once
the mechanism is already pinned against its ceiling. Extra rho-driven
curtailment beyond the ceiling is not "wasted off-window" in the sense of
being misdirected - it is doing exactly what a window-scoped mechanism
can do (levelling load inside the window further), it simply cannot
translate into further DAILY peak reduction once the peak has already
relocated outside the window.

**Rebate cost per kW of peak reduction** (total rebate paid / mean daily
peak reduction, both substations, as a Tk-per-kW-per-day proxy): rises
from 0 (rho=0, no rebate) to Tk 4.8 (rho=3), 11.9 (rho=6), 20.7 (rho=9),
30.0 (rho=12), 41.6 (rho=15) - a steep, worsening marginal cost, because
peak reduction is pinned at its ceiling (~14,000-14,300 kW total, both
substations, regardless of rho) while the rebate paid keeps climbing
linearly with rho, entirely consistent with the ceiling mechanism above:
once pinned at the ceiling, ANY additional rebate spend buys zero
additional peak reduction, so cost-per-kW is mechanically guaranteed to
rise without bound as rho increases further.

## C12: social efficiency (see Follow-up F19 for the full write-up)

0.0% of curtailed/shifted kWh is planner-test-efficient in any
non-scarcity arm; 10.96% of curtailed kWh (0.0% of shifted) becomes
efficient under scarcity, once C4's dual is included.

## C13: load-shedding counterfactual, scarcity (0.55/0.62), system level only

No household-level Z-allocation rule is stated anywhere in this codebase's
formulation, so per this round's own instruction this is reported at
SYSTEM level only. Peaks: Dhanmondi 136,380 kW (from a 138,435 kW baseline)
and Kalyanpur 109,044 kW (from 141,326 kW) under the compulsory/all-scope
DR arm; UNCHANGED at baseline (138,435 / 141,326 kW) with no DR at all -
import/battery/tie-line alone cannot relieve either substation's peak.
See Item 3 above for the full four-row table (no DR / voluntary DR at
rho=0 and rho=15 / compulsory DR) and its reading.

**Round-6 item 1 correction - "99.8% reduction in unserved load" is true
of Z alone and is the WRONG headline number; energy not delivered and
welfare cost are what should be reported together.** `Z` (involuntary,
unserved load) does fall by 99.8% under compulsory DR relative to no DR
(19,237 vs 8,640,419 kWh). But `Z` is only ONE of two ways energy fails to
reach a household in this model - the other is VOLUNTARY curtailment
(`y`), which compulsory DR uses heavily instead. Adding them
(`energy not delivered = Z + curtailed kWh`) reverses the headline:

| arm | Z (kWh) | curtailed (kWh) | energy not delivered = Z+curtailed (kWh) | welfare cost (VoLL·Z + gamma·curtail + gamma·shift) (Tk) |
|---|---|---|---|---|
| no DR | 8,640,419 | 0 | 8,640,419 | 518,425,161 |
| dr_window ONLY | 4,186,355 | 4,024,307 | 8,210,661 | 344,175,800 |
| IR ONLY | 4,147,018 | 4,995,786 | 9,142,805 | 352,727,800 |
| drw_ir, rho=0 | 7,387,586 | 1,135,882 | 8,523,468 | 466,561,400 |
| drw_ir, rho=15 | 6,533,245 | 1,818,740 | 8,351,985 | 430,709,300 |
| compulsory (all-scope) | 19,237 | 12,746,020 | **12,765,257** | 279,528,400 |

**Compulsory DR does NOT reduce the total energy the population goes
without - it is HIGHER than every other arm, including no-DR (12.77M vs
8.64M kWh, +47.7%).** What compulsory DR actually does is CONVERT
involuntary shedding (priced at VoLL, 60-150 Tk/kWh) into voluntary
curtailment (priced at the much lower `gamma_{cu,h}`, ~12-37 Tk/kWh) -
which is why welfare cost still falls by 46.1% (518.4M -> 279.5M Tk) even
though the underlying energy shortfall does not shrink at all. The
correct statement for Chapter 4 is: "compulsory DR does not eliminate the
underlying supply shortfall (energy not delivered is, if anything,
larger); it changes WHO decides which energy is foregone and at what
implied price, cutting the welfare cost of the shortfall by roughly
half." Reporting only the 99.8%-Z-reduction number, without this pairing,
overstates what the mechanism achieves.

Voluntary (`drw_ir`) DR is worse than compulsory on BOTH measures at
every rho tested - higher energy not delivered (8.35-8.52M kWh, close to
the no-DR figure) AND higher welfare cost (430.7-466.6M Tk, well above
compulsory's 279.5M) - because it is far more constrained in WHICH
curtailment it can substitute for shedding (Round-5 item 1's
window/IR-intersection finding).

## Round-6 item 3: Z is class-selective by ASSUMPTION, not by a formulation bound - and that assumption is optimistic for Bangladesh

**What the formulation actually does.** `Z_{i,cls}(t)` (unserved load) is
split into THREE separate decision variables per substation-slot - one
per class (`critical`, `shiftable`, `curtailable`) - each with its OWN
VoLL price (150 / 100 / 60 Tk/kWh) and NO explicit upper bound in
`gentwin/lp.py::build_day()`'s `bounds` array (`hi[idx[("Z",i,cls)]]` is
never set, so it defaults to `+inf`). The reason `Z_critical`/
`Z_shiftable` are observed at exactly 0 in every arm (Round-4/5/6, no
exception found) is NOT a formulation constraint enforcing protection -
it is that the OPTIMISER, facing three interchangeable ways to shed a kWh
at three different prices, always prefers the cheapest (curtailable,
60 Tk/kWh) first, and only spills into the pricier classes once the
entire curtailable class is already exhausted at a given slot (verified
never to happen in any arm run so far - item 2's check found 0 violations
of `Z_curtailable_kw <= kappa * post-response curtailable load`, i.e. the
optimiser never even runs OUT of curtailable-class headroom to shed
from first).

**Why this matters for how "no DR" (and the whole Z accounting) should be
read.** The formulation implicitly assumes SMART-METER-LEVEL, per-class
SELECTIVE shedding - the operator can choose to interrupt exactly the
curtailable-class kWh at a household and leave its critical/shiftable
load fully served, at arbitrary granularity, with no capability
constraint modelled. Bangladesh's actual load-shedding practice is
FEEDER-LEVEL ROTATIONAL shedding: an entire feeder is de-energised for a
rotation slot, taking every class on it - critical, shiftable and
curtailable alike - down together, with no selective sparing. The
`no-DR` counterfactual (item C13/Round-5) is therefore a BEST CASE for
what load-shedding would achieve in Bangladesh today, not a realistic
model of it: a real rotational outage would shed critical and shiftable
load too, at a cost this model cannot show because those two classes are
never actually shed in any run. Chapter 4 should state the `no-DR`
counterfactual's Z/VoLL/welfare-cost numbers as a LOWER BOUND on the true
cost of load-shedding without DR, not as the cost itself, and should not
describe `Z_critical`/`Z_shiftable`'s observed value of 0 as a "finding"
that critical load is protected - it is an artefact of an idealised
shedding mechanism the formulation assumes but does not model the
capability constraints for.

## C14: gamma sensitivity on drw_ir

`--gamma-scale 0.7`/`1.3` (discomfort weight scaled down/up), each at
`rho=0` and `rho=15` (the base condition's unweighted rho\*, item C11).
All 4 runs check-6 PASS.

**Round-4 correction (item 2).** `--gamma-scale` mutates the cached
population's `h.gamma_sh_h`/`h.gamma_cu_h` inside `06_lp_optimiser.py::run()`'s
own process only - the SAME propagation gap as Follow-ups F11/F16, this
time for gamma. `09_monthly_billing.py` calls
`pop.consumers_from_cache()` fresh in its own process and got the
UNSCALED gamma back, so `discomfort_J2_tk`/`net_benefit_tk`/`gained_*`
for all four C14 arms were computed against the WRONG discomfort weight -
visible in the original table as `%gained_unweighted` being identical
(0.93%) across all three gamma scales at rho=0, which should not happen
if gamma is actually changing. The LP's own x/y decisions (curtail_kwh,
shift_kwh, participation) were NOT affected - `build_day()` reads gamma
directly from the mutated population, so those numbers were already
correct. Fixed by writing `gamma_scale_used` to `lp_summary{tag}.csv`
(mirroring F11/F16) and adding a `--gamma-scale` override to
`09_monthly_billing.py` itself, applied to its own `POP` before
`billing.build()` runs. Per this round's instruction, this did NOT require
a new LP solve - `09_monthly_billing.py --report --tag <tag> --gamma-scale
<gs>` was re-run against the EXISTING `lp_decisions`/`lp_thresholds` for
all four tags.

| gamma_scale | rho | participation % | curtail kWh | shift kWh | %gained (unweighted) | %gained (lambda) |
|---|---|---|---|---|---|---|
| 0.7 | 0 | 22.05 | 1,957,607 | 144,118 | **12.07** | 20.68 |
| 1.0 (baseline) | 0 | 13.49 | 1,135,706 | 135,290 | 0.93 | 11.97 |
| 1.3 | 0 | 6.87 | 512,898 | 133,036 | **0.00** | 5.38 |
| 0.7 | 15 | 48.37 | 2,506,614 | 144,118 | **48.09** | 48.29 |
| 1.0 (baseline) | 15 | 33.60 | 1,818,642 | 135,290 | 28.96 | 33.45 |
| 1.3 | 15 | 27.38 | 1,242,719 | 133,036 | **14.54** | 26.82 |

Corrected `%gained_unweighted` now moves monotonically and substantially
with gamma at BOTH rho values (0.00% to 12.07% at rho=0; 14.54% to 48.09%
at rho=15) - the flat 0.93% in the original table was the bug's signature,
not a real finding. Participation/curtail/shift kWh are unchanged from
the original table (the LP's decisions were always correct). This confirms
the model's `%gained` sensitivity to `gamma_{sh,h}`/`gamma_{cu,h}` is real
and substantial, which matters for how much weight Chapter 4 should put on
results that depend on the specific calibrated gamma values (Sec. [Issue
F]/[calibration]) - a +-30% gamma uncertainty band moves `%gained_unweighted`
by roughly a factor of 4-13x at these two rho values, not the ~1x the
uncorrected table implied.

## C15: stakeholder ledger (from available data only)

| arm | operator: lost retail revenue (Tk) | operator: rebate paid (Tk) | residential lower-middle bill saving (Tk) | residential middle bill saving (Tk) | residential upper-middle bill saving (Tk) |
|---|---|---|---|---|---|
| base (original, non-IR) | 155,966,726 | 15,665,847 | 0 | 11,483,690 | 84,324,700 |
| drw_ir, rho=0 | 17,621,047 | 0 | 0 | 0 | 12,287,820 |
| drw_ir, rho=15 (rho\*) | 27,299,901 | 17,721,186 | 0 | 288,374 | 17,279,270 |
| scarcity | 169,981,105 | 18,660,810 | 430,501 | 19,187,200 | 84,148,420 |

**The lower-middle income band receives ZERO bill saving in every arm
except scarcity** (where it gets a small, 430,501 Tk, share) - it simply
never participates in demand response under any of these mechanisms.
Savings are overwhelmingly captured by the upper-middle band in every arm
(72-92% of all residential savings). `Hospital` and `Government`
categories show zero or negligible savings in every arm (protected
classes, high `y_max_h`/`gamma` making curtailment prohibitively costly or
their archetype excludes them) - see `outputs/billing/household_monthly_bills{tag}.csv`
grouped by `category`/`income_band` for the full breakdown underlying this
table.

**Recommended interpretation for Chapter 4**: state plainly that this DR
mechanism, as currently parameterised, is (a) privately but not
socially rational outside real scarcity (item C12/F19: 0.0% of
curtailed/shifted kWh clears the planner's test without a scarcity
scenario), (b) a net transfer from the operator's retail revenue to a
NARROW, upper-income slice of participants rather than a broadly shared
benefit (this section: lower-middle income gets 0 Tk in every arm but
scarcity; the operator's own avoided-cost ledger runs -7.9M to -11.7M
Tk/month under normal conditions, item C10/C11, corrected), and (c) a
genuinely large, quantified WELFARE-COST reduction once real scarcity is
reached (item C13, Round-6 correction: avoided cost swings to +53.3M to
+86.5M Tk/month, and total welfare cost of the shortfall falls 46.1%
(518.4M -> 279.5M Tk) relative to no DR - NOT because the underlying
energy shortfall shrinks, which it does not - energy not delivered
(Z + curtailed kWh) is actually 47.7% LARGER under compulsory DR than
under no DR - but because DR converts costly involuntary shedding into
cheaper voluntary curtailment) - three different, non-contradictory
claims that should not be collapsed into a single "demand response saves
everyone money," or even "demand response reduces the shortfall,"
narrative.
A fourth, sharper point, revised by Round-5 item 1's decomposition: even
the "voluntary" (individually-rational, DR-window-scoped) version of the
mechanism captures only a small fraction of that emergency value -
unserved load under `drw_ir` during severe scarcity (6.5-7.4M kWh
depending on rho) sits far closer to the no-DR counterfactual (8.6M kWh)
than to the compulsory, unrestricted arm (19,237 kWh). This is NOT because
either restriction (window-scoping OR individual rationality) is
individually expensive - each ALONE recovers roughly half of compulsory
DR's benefit (dr_window alone: -51.6% unserved load vs no-DR; IR alone:
-52.0%). It is because REQUIRING BOTH AT ONCE intersects two narrow,
largely non-overlapping eligibility sets, recovering barely a quarter of
compulsory DR's benefit even at the highest rebate tested (rho=15:
-24.4%). The policy-relevant framing is therefore not "voluntary DR is
inherently weak" but "stacking window-scoping AND individual-rationality
requirements is far more costly than either alone" - a designer wanting
to preserve household consent (IR) without paying the full compounded
price should consider relaxing the window restriction, or vice versa.
