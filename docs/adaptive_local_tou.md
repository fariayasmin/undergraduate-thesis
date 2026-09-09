# Adaptive local time-of-use pricing

**BERC sets the ratios. The digital twin sets the window.**

This note documents the GenTwin-SG adaptive local ToU mechanism: what it is,
why it exists, what it changes in the formulation, and what it does not claim.

---

## 0. The one-sentence statement

> Time-of-use price *ratios* are taken from the BERC retail tariff order of
> 03 June 2026 (μ^pk = 1.20, μ^off = 0.90). The *activation window* over which
> the peak ratio applies is derived by the digital twin, per substation, from
> that substation's own congestion history.

This is **not a replacement for the official tariff** and no output of this
model may present it as one. It is a congestion-aware activation schedule laid
over the regulator's own multipliers. Every artefact records both halves and
their separate provenance:

```json
"mu_peak": 1.20,
"mu_source": "BERC order 03 June 2026 (LT-class peak/flat ratio)",
"window_source": "GenTwin-SG adaptive local derivation"
```

---

## 1. Why the fixed national window was insufficient

The BERC order (footnotes 4–6) defines one national window for every ToU
consumer in Bangladesh: peak 17:00–23:00, off-peak 23:00–17:00. It is a single
national instrument, and for the country as a whole it is defensible — the
national system peak is an evening peak.

It does not follow that it is right at every substation. The BPDB data says it
is not.

| | Dhanmondi | Kalyanpur |
|---|---|---|
| Adaptive window T^pk_i | **12:00–14:30** | **18:00–21:00** |
| National ToU peak | 17:00–23:00 | 17:00–23:00 |
| Overlap | **0 of 5 slots** | 6 of 6 slots |
| Congestion mass captured | 0.709 | 0.416 |
| … by the national window | **0.006** | 0.653 |
| Precision (mass per priced slot) | **0.1417** | 0.0693 |
| … national | **0.0005** | 0.0545 |
| **Precision gain** | **285×** | 1.3× |

Read the Dhanmondi row carefully. On its 121 most congested days, **0.6%** of
the peak-timing density falls inside the national ToU window. A peak tariff
applied 17:00–23:00 at Dhanmondi charges consumers more at hours when the
substation is not congested, and charges them nothing extra at 13:00 when it
is. The price signal and the physical constraint are pointing at different
hours of the day. That is not a small calibration error; it is a signal aimed
at the wrong target.

Kalyanpur is the opposite case and is equally instructive: the national window
*does* contain its peak. The adaptive window captures less total mass (0.416 vs
0.653) precisely because it is half as wide. **Mass alone flatters a wide
window** — pricing all 48 slots captures 100% of the density and signals
nothing at all. Per priced slot, the adaptive window is still 1.3× more
targeted. So the mechanism does not merely relocate the window; where the
national window is already right it tightens it, and where it is wrong it moves
it.

A single national instrument cannot do both. A digital twin that already holds
each substation's load history can.

---

## 2. Why local peak-aware pricing suits adaptive smart-grid management

Three reasons, in increasing order of importance to this thesis.

**It matches the object the model actually constrains.** C1 (Eq. 27) and C4
(Eq. 30) are written per substation `i`. The stress indicator `s_{i,d}`
(Eq. 15), the deficit `Def_{i,d}`, the scaling factor `κ_{i,d}` and the regime
`r_{i,d}` are all indexed by `i`. The shadow price `π_i(t)` — the quantity on
which the entire explainability layer rests — is a *per-substation, per-slot*
dual. The formulation is already substation-aware everywhere except the tariff.
Making μ substation-aware removes the last inconsistency rather than
introducing a new concept.

**It closes the loop between the price signal and the shadow price.**
Proposition 2 makes each household action a comparison of a private benefit
against a discomfort, corrected by `κ_{i,d}(t)·π_i(t)/θ₃`. When the tariff peak
and the congestion peak coincide, the private and system terms *reinforce* each
other and point at the same slots. When they are six hours apart, as at
Dhanmondi, the private term pushes consumers to respond in the evening while
the system term pushes them to respond at midday. The scheme still works — the
system term internalises the externality regardless — but it works *against*
the tariff instead of with it, and every unit of response has to be bought
entirely by the shadow price.

**It is what "adaptive" means operationally.** The window is derived from data
the twin already ingests and can be re-derived as load shapes change. The
mechanism is not a one-off calibration; it is a standing procedure. Kalyanpur
demonstrates why that matters — see the limitation in §6 about its recent load
shape.

---

## 3. How this improves demand-response effectiveness

Through Eq. (34). The shifting threshold is

$$x^\star_{h,t} = 1 \iff \underbrace{\lambda_g \pi^{ret}_h(\mu^{pk} - \mu^{off})}_{\text{private}} + \underbrace{\frac{\kappa_{i,d}(t)}{\theta_3}\left(\pi_i(t) - \pi^{off}_{i,d}\right)}_{\text{system}} > \underbrace{(1-\lambda_g)\gamma^{sh}_g}_{\text{discomfort}}$$

The private term is non-zero **only for slots inside T^pk_i**, because μ(t) is
what makes `μ^pk − μ^off = 0.30` appear. Outside the window the private term is
identically zero and a household will shift only if the system term alone
clears its discomfort.

At Dhanmondi under the national window, every congested slot is an off-peak
slot. The private term is zero exactly when the grid needs the response, so the
whole burden falls on `κ_{i,d}(t)π_i(t)/θ₃`. Under the adaptive window the
private term is `λ_g · π^ret_h · 0.30` — for a Res-A household at the fourth
slab (9.62 BDT/kWh) that is `0.60 × 9.62 × 0.30 = 1.73` BDT/kWh of private
benefit added to the left-hand side, at precisely the congested slots.

The same logic runs through Eq. (36): a larger left-hand side lowers the
critical preference λ*_g, so **more tiers cross the participation threshold at
the same shadow price**. Formulation §8 shows this exact effect — a shadow
price of 6.00 BDT/kWh pulls λ*_A down to 0.3424 and draws tier B into
curtailment, which is what lets local generation and storage substitute for
inter-substation transfer. Aligning the window is a second, independent lever
on the same threshold.

There is also a straightforward efficiency argument. Under the national window
at Dhanmondi, peak pricing is applied to 12 slots of which ~0 are congested:
consumers pay a peak rate for six hours with no reliability benefit purchased.
Under the adaptive window, 5 slots are priced and 70.9% of congestion density
sits inside them. Fewer consumer-hours are disrupted for more delivered
flexibility.

---

## 4. What changes in the formulation

The amendments are **index changes on exogenous data only**. No proof is
touched.

| Eq. | Before | After |
|---|---|---|
| (5) | recovery spread over `T^off_{d(t)}` | `T^off_{i,d(t)}` |
| (19) | `π_h(t) = (π^ret_h + ς_{i,m})μ(t)` | `μ_i(t)`, `i = i(h)` |
| (32) | `0 ≤ x_{h,t} ≤ 1[t ∈ T^pk_{d(t)}]` | `1[t ∈ T^pk_{i,d(t)}]` |
| (34) | `λ_g π^ret_h (μ^pk − μ^off)` | unchanged in form; the *set* of `t` at which it is non-zero is now `T^pk_i` |

Note that Eq. (34) already carries `π^off_{i,d} = |T^off_d|^{-1} Σ_{τ∈T^off_d} π_i(τ)`
— the dual is indexed by `i` while the averaging window is not. The amendment
makes the window `T^off_{i,d}` and removes that inconsistency.

**Why the propositions survive.** `T^pk_i` is an exogenous set of slot indices
and `μ_i(t)` an exogenous coefficient vector; both are fixed before the solve,
exactly as `κ_{i,d}` is under Assumption 2. Indexing exogenous data by `i`
changes no coefficient's *structure*:

- **Prop. 1** perturbs `Z^c_i`, `G_i`, `T_{ji}` only. Untouched.
- **Prop. 2** requires `F` affine and C6 a box. `μ_i(t)` enters `f_wel`
  linearly through `J^h_1`; `1[t ∈ T^pk_{i,d}]` is a per-variable constant
  upper bound, so C6 is still a box. Both threshold forms hold verbatim.
- **Prop. 3** prices supply resources only; `μ` appears in neither (37) nor its
  exchange argument.
- **Corollary 1** — `T^pk_i` is derived from history before the solve, so no
  fixed point is created. See §6 for the one way this could be violated.

## 5. What must *not* change: μ does not enter `f_cost`

The instruction "the objective cost term should become substation-aware,
`C_i(t) = P_i(t) × μ_i(t)`" is implemented as the **consumer-facing price**,
`π_h(t) = (π^ret_h + ς_{i,m})·μ_i(t)` in Eq. (19), which enters `f_wel`.

It is deliberately **not** applied to `f_cost` (Eq. 22), and this is worth
being explicit about because it would be an easy and damaging change to make.
`c_i(t)` in Eq. (22) is the operator's marginal **supply cost** in BDT/MWh —
here 8,390 BDT/MWh, the BPDB bulk tariff. `μ_i(t)` is a **retail** multiplier.
Multiplying the two would:

1. **Price the same congestion twice**, once through the tariff paid by
   consumers and once again through the operator's dispatch cost, when the
   physical scarcity being priced is a single event.
2. **Corrupt `π_i(t)`.** The dual of C1 is the marginal value of energy at
   `(i,t)`. It is the quantity the whole explainability layer reports, and
   Eq. (36) divides by `θ₃` to put it on the same scale as household
   discomfort. If `c_i(t)` were inflated by 1.20 inside the peak window,
   `π_i(t)` would inherit a tariff artefact, and every "system value" term
   shown to a user would be partly a pricing decision rather than a physical
   scarcity signal.
3. **Weaken Prop. 3.** Condition (37) requires a strict ordering at every slot.
   Scaling `c_i` and `π_ij` by the same `μ_i(t)` preserves the first
   inequality but compresses `θ₁π_ij` against the fixed `θ₂π^ν`, so the merit
   order could break inside the peak window at parameter values where it
   currently holds.

If the intent is instead a genuinely time-varying **supply** cost — a
congestion-priced `c_i(t)` — that is a separate and legitimate modelling
choice, supported by the LP (`c_i(t)` is already slot-indexed in Eq. 22) and
independent of the retail multiplier. Say the word and it becomes a config
option; it should not be spelled `μ`.

---

## 6. Assumptions and limitations

**A1 — No intra-day load profile exists.** The BPDB dataset gives one daily
peak and its clock time; there is no half-hourly series. "Average historical
load profile" is therefore unavailable, and the estimator works on the
*distribution of peak timing* across ~2,300 days instead. This is a real
constraint on what can be claimed: T^pk_i is an estimate of *when the peak
falls*, not of *the shape of the load curve*.

**A2 — Peak time is recorded on a 30-minute grid.** All values fall on `:00` or
`:30`, which fixes the resolution at 48 slots.

**A3 — Congestion days as the population.** T^pk_i is derived from days above
the p95 peak (121 days per substation), not all days, because the window exists
to price congestion. This is a choice; §2 of the runner output reports the
window under three alternative populations so it can be argued with.

**A4 — μ^pk = 1.20 for residential is a scenario.** LT-A carries no ToU row in
the order. The LT-family ratio is adopted — the regulator's own number for
every other low-tension class — but residential ToU does not exist in
Bangladesh and the model must not imply it does.

**L1 — Kalyanpur's window is statistically unstable.** Bootstrap width IQR is
5 slots against Dhanmondi's 0. The runner flags it `UNSTABLE - indicative
only`. Two causes, both reported:

  - a secondary density lobe at **22:00–24:00** (mass 0.202) dropped by the
    contiguity rule;
  - **13.1%** of Kalyanpur rows record a peak at exactly 00:00 (versus 1.0% at
    Dhanmondi), which is a not-recorded default rather than a midnight peak.
    Adjacent to the 23:00 cluster, this suggests an end-of-day recording
    artefact and means peak timing at Kalyanpur is less well observed than its
    row count implies.

**L2 — Kalyanpur's recent load shape may have moved.** On 2024+ data alone the
estimator returns **11:30–14:30**, a midday window, against 17:30–20:30 on the
full record. Either Kalyanpur's peak has genuinely shifted toward midday, or
recent peak-time recording has degraded (consistent with L1, since most of the
00:00 artefacts are recent). **This should be checked against BPDB operational
records before the Kalyanpur window is relied on.** The model currently uses
the full-record evening window.

**L3 — The window is static within a run.** T^pk_i is derived once from history
and held fixed, matching Assumption 2's treatment of κ. A window recomputed
*from the post-response load* would make C1 depend on the solution and destroy
the linearity, Prop. 1 and the closed-form thresholds — exactly the failure
Assumption 2 warns about. Re-derivation between billing periods is safe and is
the intended operational cadence; re-derivation inside a solve is not.

**L4 — Seasonality is not modelled.** One window per substation for the whole
record. Dhaka's load shape differs between monsoon and dry season, and a
seasonal T^pk_{i,season} is a natural extension the estimator already supports
(pass a filtered frame). Not done here, because the operational horizon is 14
days.

**L6 — Dhanmondi's window rests on a field that may be a reporting
convention.** THIS IS THE MOST SERIOUS LIMITATION IN THIS DOCUMENT and it was
found after the mechanism was built. The peak-time reliability audit
(`docs/peak_time_reliability.md`, `python 00b_peak_window.py --audit`) rates
Dhanmondi's `Peak_Time` field **LIKELY_CONVENTION**, on 6 of 6 flags — the
worst in the fleet. Only **2.7%** of its readings fall on a `:30` boundary
against 17.7% at Kalyanpur and 20.7% at Mirpur; its modal share is *highest*
in winter (75%) when load is 36% lower and there is no cooling demand; and the
share rose +25.3% from 2020 to 2026 while load fell. Ten substations share the
pattern and **all ten have a 13:00 mode**, which reads as one reporting
practice rather than ten coincident midday peaks.

Consequences, stated plainly:

- The **285× precision gain at Dhanmondi must not be quoted unqualified.** Both
  the adaptive window and the "congestion mass" it is scored against come from
  the same possibly-clerical field. If 13:00 is a logging hour, the comparison
  is between two windows neither of which is anchored to observed congestion.
- **Kalyanpur is unaffected** (audit: RELIABLE, 1 flag, and its modal slot moves
  seasonally — 19:00 in monsoon, 13:00 in winter — exactly as a physical
  distribution should). Its 1.3× precision gain stands.
- **The mechanism itself is not in question.** Deriving the activation window
  from local congestion history rather than a national rule is the
  contribution, and it holds whatever this field turns out to be. What is in
  question is one substation's *input data*, not the method.
- Resolving it needs a half-hourly SCADA series or a direct question to BPDB
  about how `Peak_Time` is recorded. See `docs/peak_time_reliability.md` §7.

**L5 — Estimator settings move the window.** Across 27 combinations of stress
quantile, density threshold and bandwidth, Dhanmondi returns 7 distinct windows
(modal 12:00–14:30, 22%) and Kalyanpur 11 (modal 18:00–21:00, 19%). The modal
window is the one used in both cases, and all Dhanmondi variants remain within
11:00–15:30 — a midday window under every setting, never an evening one. The
qualitative conclusion is robust even where the exact boundaries are not.

---

## 7. Reproducing

```bash
python 00b_peak_window.py --derive        # derive T^pk_i, cache, report
python 00b_peak_window.py --sensitivity   # 27 estimator settings
python 00b_peak_window.py --compare       # adaptive vs national, precision
python 00b_peak_window.py --plot          # outputs/adaptive_tou_windows.png
```

To run the national-window ablation, set `ADAPTIVE_LOCAL_TOU = False` in
`gentwin/config.py`, or pass `adaptive=False` to `tariff.mu_profile`. Both arms
should be reported in the results chapter: the comparison *is* the result.
