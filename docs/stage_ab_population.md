# Stage A/B — synthetic population and bottom-up load

Implements Eqs. (1)–(8). Forecast outputs are read-only throughout.

---

## 1. Result summary

| | Dhanmondi | Kalyanpur |
|---|---|---|
| Reference day | 2026-06-29 | 2026-06-29 |
| Target `p_{i,d}` | 136,500 kW | 138,000 kW |
| Achieved peak | 136,500 kW (−0.000%) | 138,000 kW (−0.000%) |
| **T1 peak level** | **PASS** | **PASS** |
| Synthetic argmax | 18:00 | 19:00 |
| Adaptive `T^pk_i` | 12:00–14:30 | 18:00–21:00 |
| **T2 peak timing** | **UNDETERMINED** (9/40 draws) | **ROBUST PASS** (40/40) |
| Load factor | 0.674 | 0.631 |
| Diversity factor | 1.39 | 1.37 |
| Implied connections | 41,187 | 70,122 |
| Flexible energy share | 0.852 | 0.853 |

**Stage B gate: PASS**, with Dhanmondi T2 recorded as UNDETERMINED rather than failed. Section 6 explains why, and why the
failure is more informative than a pass would have been.

## 2. The two acceptance tests are not equally meaningful

**T1 — peak level.** Satisfied *by construction*: the global calibration
constant `K` in `set_weights()` is solved for exactly this. T1 confirms the
calibration executed. It is not evidence the model is right, and the thesis
should not present it as such.

**T2 — peak timing.** Not fitted anywhere, and now reported as a
SEED-STABLE verdict rather than a single draw (see §9). Schedules, ownership rates, appliance
ratings and thermal coefficients were all fixed before timing was tested, and
the peak hour is the one thing the BPDB dataset independently observes about the
*shape* of the day. T2 is the real test.

The category-mix ablation shows T2 is doing real work rather than being
determined by the residential schedules alone:

| variant | Dhanmondi argmax | Kalyanpur argmax |
|---|---|---|
| full mix | 18:00 | 19:00 |
| residential only | 19:00 | 19:00 |
| non-residential only | **12:00** | **12:00** |

The midday and evening peaks come from genuinely different parts of the
population. The mix is what decides which wins.

## 3. How the population is built

```
OSM demand mix  ──► category allocation (400 reps, floor of 8/category)
                        │
                        ├─► tier assignment (residential_archetype_mix)
                        ├─► appliance sampling (ownership prob, n_range)
                        └─► per-consumer jitter (±2 slots, ±12% amplitude)
                                │
Θ(t) ──────────────────────────►│  Eq. (2)  P_{h,a}(t)
                                │  Eq. (3)  L^c_h(t), L_h(t)
                                ▼
        stage 1: v_h = s_c / E_c      → OSM mix reproduced exactly (1.1e-16)
        stage 2: K = p_target / max Λ → peak level exact, ξ = 1 on ref day
                                ▼
                          Eq. (6)  Λ_i(t)
                          Eq. (7)  Λ^f + ξ(d)Λ^v
```

**Weights.** `w_h` is the number of real connections a representative stands
for, stored on every consumer and carried into the knowledge graph. The
reduction is exact, not approximate: `L_h` scales by `w_h` while `x` and `y`
remain fractions, so the LP stays linear and the Proposition 2 reduced cost
divides through by `θ₃·w_h·L^cu_h(t)·Δ > 0`, still strictly positive. The
threshold is unchanged.

Within-category multiplicity is **uniform** — all representatives of a category
share one `w`. Diversity comes from appliance sampling and schedule jitter, not
unequal weights. Making `w` heterogeneous too would let one representative
dominate its category with nothing to justify it.

**ξ(d).** Eq. (7) splits `Λ = Λ^f + ξ(d)Λ^v` with `Λ^f` = critical class and
`Λ^v` = shiftable + curtailable. The split is physical, not convenient: critical
load is what the consumer cannot vary, so everything ξ absorbs is behavioural or
thermal — which is what actually varies day to day. It also means ξ never
rescales load the optimiser is forbidden to touch. `max_t(Λ^f + ξΛ^v)` is convex
and non-decreasing in ξ, so bisection is exact.

Over the last 30 observed days: ξ mean 0.808 (sd 0.201) at Dhanmondi, 0.885
(sd 0.076) at Kalyanpur. `|ξ−1| > 0.25` on 8/30 Dhanmondi days against 2/30 at
Kalyanpur — Dhanmondi's day-to-day peak is far more volatile (it ranges 92–137
MW in the last ten days alone), and ξ is absorbing that. A ξ far from 1 is
reported, never hidden: it means the catalogue does not describe that day.

## 4. Θ(t)

`Temp_Max_C`, `Temp_Min_C` and `Temp_Mean_C` are daily. Θ(t) at 48 slots is
**reconstructed**, not observed, using Parton–Logan (1981): a sine arc from Tmin
at sunrise to Tmax at a 2 h lag after solar noon, then exponential nocturnal
decay, then an affine correction so the slot-mean reproduces the recorded
`Temp_Mean_C` exactly.

Reference day: Tmin 26.2 / Tmax 33.3 / Tmean 29.8 °C, day length 13.46 h,
Θ peaking 33.18 °C at 13:30. All five reconstruction checks pass.

This matters more than it looks. The AC thermal response `f_a(Θ)` is what
creates the afternoon load rise, so the *shape* of Θ(t) partly determines the
peak hour — the very thing T2 tests. Every artefact built on it carries
`provenance: RECONSTRUCTED`.

## 5. Assumptions introduced (the honest list)

**Catalogue and schedules** — `02_consumer_catalogue.json`

1. **Schedules are the single largest assumption in Stage A/B.** No metered
   end-use survey exists for Dhaka. Each category's `u_{h,a}(t)` encodes a
   documented occupancy pattern. They are in a readable JSON with comments,
   not buried in code, so they can be challenged directly.
2. **`res_ac_affluent` is consequential and contestable.** It assumes affluent
   Dhanmondi households run AC through the hot afternoon (someone is home —
   domestic staff, home working, non-working household members). A purely
   evening-dominant residential AC profile makes a midday substation peak
   arithmetically impossible. This is stated in the catalogue itself.
3. **Thermal coefficients.** AC 6%/°C above 24 °C capped at 1.75; fans 3%/°C
   above 26 °C capped at 1.30; refrigeration 2%/°C above 22 °C capped at 1.30.
   Literature-order values, not measurements.
4. **Non-residential ratings are floor-area-derived**, from ASHRAE/IEEE 3001
   connected-load densities, not nameplate surveys of these specific buildings.

**Population**

5. **`residential_archetype_mix` is a scenario.** OSM carries no income data. The
   Res-A/B/C archetype split (Dhanmondi 0.45/0.40/0.15; Kalyanpur 0.20/0.50/0.30) is
   assigned from the documented character of each neighbourhood. It sets how
   much daytime AC load exists and therefore feeds directly into T2.
6. **400 representatives per substation**, floor of 8 per category so hospitals
   (3–9% of demand) are resolved by more than one or two consumers.
7. **Uniform `w` within a category** (§3).
8. **Jitter parameters** (±2 slots, ±12% amplitude) are chosen to give a
   plausible diversity factor, not measured. Achieved: 1.39 and 1.37.

**Calibration**

9. **ξ is applied to the variable classes only** (§3).
10. **The OSM mix is currently the FALLBACK**, `provenance: fallback_literature`,
    not a real Overpass fetch. This is the direct cause of the T2 failure — see
    §6.

**Known plausibility weakness**

11. **Too many, too small hospitals.** The implied count is 171 at Dhanmondi and
    55 at Kalyanpur. Dhanmondi has a dense private-hospital cluster, but 171 is
    high even counting clinics and diagnostic centres, and each is
    correspondingly under-sized (~35 kW connected). The category *total* is
    right by construction; its division into units is not. This affects
    per-consumer explanations in the knowledge graph, not the aggregate or the
    LP. Fixing it needs the real OSM feature counts.

## 6. Why Dhanmondi fails T2, and what it tells us

At 13:00 vs 19:00 on the reference day:

| category | 13:00 MW | 19:00 MW | Δ |
|---|---|---|---|
| Residential | 54.44 | 80.58 | **−26.14** |
| Commercial | 42.74 | 38.30 | +4.44 |
| Educational | 17.16 | 5.44 | +11.72 |
| Government | 8.58 | 2.57 | +6.01 |
| Hospital | 10.08 | 8.50 | +1.58 |
| Industrial | 1.61 | 0.19 | +1.43 |
| **Total** | **134.62** | **135.57** | **−0.96** |

The evening wins by **0.96 MW — 0.7%**. The model is on a knife edge, and the
residential share decides it:

| residential share | argmax | T2 |
|---|---|---|
| 0.42 – 0.50 | 12:00–12:30 | **pass** |
| **0.51** | 18:00 | fail ← flip point |
| 0.52 (current fallback) | 18:00 | fail |

**The flip is at 0.51. The fallback estimate is 0.52.** The T2 failure lies
entirely within the uncertainty of a number that OSM exists to measure and that
is currently a literature guess.

So the honest conclusion is not "the model is wrong" and not "the model is
right". It is:

> Dhanmondi's observed 13:00 peak is reproducible if and only if residential
> load is at most ~50% of connected demand in the service area. The bottom-up
> model is sharp enough to place a real constraint on the composition — but it
> cannot settle the question without the measurement, and **the Overpass fetch
> is therefore not optional for Dhanmondi**. It is the observation that decides
> whether the model reproduces the peak timing.

Dhanmondi has commercialised heavily (Satmasjid and Mirpur Road retail, a dense
private-hospital cluster, one of Dhaka's highest concentrations of schools and
coaching centres), so a real OSM floor-area inventory could plausibly return
below 50%. **We have not adjusted the fallback to make the test pass.** Doing so
after seeing the result would convert an independent test into a fitted
parameter and destroy the only real validation Stage B has.

`--diagnose-mix` reports the flip point. It never sets the mix; the pipeline
always reads the OSM cache.

## 7. Running

```bash
python 03b_temperature_profile.py --date 2026-06-29 --plot
python 03_population_synthesis.py --build --report
python 04_load_generator.py --build --plot --xi-series 30
python 05_validate_against_bpdb.py --validate --ablation
python 05_validate_against_bpdb.py --diagnose-mix     # on a T2 failure
```

To resolve the Dhanmondi failure:

```bash
python 01_service_area_osm.py --fetch     # online, once
python 03_population_synthesis.py --build
python 05_validate_against_bpdb.py --validate
```

## 8. Carry-forward to the LP

- `data/cache/population_*.json` — consumers, tiers, appliances, `w_h`
- `outputs/lambda_*_2026-06-29.csv` — Λ_i(t), by class and category, with Θ(t)
- `L^cr`, `L^sh`, `L^cu` per consumer per slot feed Eq. (4) directly
- `post_response_load()` and `recovery()` in `load_model.py` implement Eqs. (4)
  and (5) against the substation's own `T^pk_i`

**The LP should not be run on Dhanmondi until T2 is resolved or the failure is
explicitly accepted and recorded in the thesis.** Kalyanpur passes all gates and
is ready.


## 9. T2 is a distribution, not a single number

Added after the formulation revision, when a changed RNG stream moved
Dhanmondi's default-seed argmax from 18:00 to 12:30 — i.e. from fail to pass —
with no modelling change at all. On a composition where the two peak candidates
differ by under 1%, one draw carries no information.

`validation.timing_stability()` re-synthesises the population across 40 seeds
and reports the share of draws landing inside `T^pk_i`:

| | inside T^pk_i | draw distribution | verdict |
|---|---|---|---|
| Dhanmondi | **9/40 (22%)** | 18:00 ×28, 12:30 ×5, 18:30 ×3, 12:00 ×2, 13:00 ×2 | **UNDETERMINED** |
| Kalyanpur | **40/40 (100%)** | 19:00 ×17, 18:00 ×16, 18:30 ×7 | **ROBUST PASS** |

The T2 check now takes its verdict from this share, never from the default
seed. Thresholds: ≥90% ROBUST PASS, ≤10% ROBUST FAIL, anything between is
UNDETERMINED and is reported as neither.

Kalyanpur's result is strengthened by this: it is not one lucky draw but every
draw, and the peak lands at 18:00–19:00 in all forty.

Dhanmondi's is honestly weakened. The earlier "fails by 0.96 MW" framing was
already a knife edge; this quantifies it. The composition simply does not
determine the peak hour, and combined with the `Peak_Time` reliability audit
(`docs/peak_time_reliability.md`, verdict LIKELY_CONVENTION) there are now
**two independent reasons** not to treat Dhanmondi's timing as settled:
the target may not be real, and the model does not commit to an answer.


## 10. Verification of the reviewer's checks

Each was tested against the code and outputs rather than argued.

| # | Check | Result |
|---|---|---|
| 1 | Did the LP explode to ~376,000 variables? | **No.** 400 agents per substation, **43,568 variables**, 1,280 rows, ~1.7 s/day. `sum(w_h)` = 133,813 + 241,879 = 375,692 real connections. Option A, as intended. |
| 2 | Are bills per agent or per real customer? | **Per agent.** 800 rows, each a representative standing for w_h real connections. All population totals are w_h-weighted. Wording corrected throughout. |
| 3 | Stage B wording | **Fixed.** The gate now prints a per-substation verdict and warns against reporting a single "both passed". |
| 4 | Coincidence factor still low? | **My terminology was wrong.** 2.37/2.79 is the DIVERSITY factor (IEEE 141: Σ individual peaks / coincident peak). Its reciprocal, the COINCIDENCE factor, is **0.42 / 0.36** — Kalyanpur inside and Dhanmondi just outside the 0.2–0.4 urban residential band. Much closer to realistic than "still below 3–8" implied. |
| 5 | Which net-benefit equation? | **`saving + incentive − discomfort`**, verified numerically. No double count: the gazette bill contains no ρ term, so the incentive is added once. `J^h_1` *does* net the rebate per Eq. (20) and is reported separately, never summed with it. |
| 6 | LP price vs billing price | **Correct and now stated explicitly** in the report and in `billing_statistics.json` under `price_basis`. |
| 7 | Billing window naming | **Fixed** to "representative 30-day evaluation window", with `is_bpdb_billing_cycle: false` in the JSON. |
| 8 | KG relationships | **Genuine gap, fixed.** `Event→AFFECTS→Household` (2,511), `Event→CAUSES→DrAction` (28,457), `Household→PARTICIPATED_IN→Event` (2,511) and `Household→HAS_APPLIANCE→Appliance` (5,416) were missing; `Substation→Event` is now `HAS_EVENT`, freeing `AFFECTS` for its role in the brief. Zero dangling endpoints. |

### The diversity caveat that remains

The measured diversity factor is across the **400 agents**, not across the
375,692 real connections. Each agent stands for w_h identical households which
by construction peak together, so **2.37 is a lower bound** on what a real
population would show. Raising `N_REPRESENTATIVE_CONSUMERS` tightens the bound;
it does not move the aggregate, which is pinned to the BPDB peak either way.
This is now stated in `validation.diversity_factor()`.
