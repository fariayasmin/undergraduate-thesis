# Chapter 3 (GenTwinX-SG thesis) vs. this implementation

Every mismatch found between the thesis's Chapter 3 methodology and the
running code, classified per the review brief:

- **Category 1** - the code is a valid extension/generalisation. Keep the
  code; update the written formulation.
- **Category 2** - the written equation is inconsistent or wrong. Correct the
  formulation; the code was already right.
- **Category 3** - the code contradicted the intended real-world (Bangladesh)
  assumption. Fix the code.

Only one item is Category 3. Nothing here required redesigning the LP, the
population synthesis, the Digital Twin architecture, the Knowledge Graph, or
priority pooling - all of those stand as built.

---

## 1. Residential (and LT-D1) time-of-use pricing - CATEGORY 3, fixed

**What Chapter 3 implies.** Eq. (3.21), `π(t) = (π^ret + δ)μ(t)`, is written as
if μ(t) is simply "the corresponding time-dependent (peak time, off-peak time)
pricing factor" for any consumer, with no carve-out.

**What is actually true in Bangladesh.** LT-A (residential) carries no
time-of-use row in the BERC order at all - footnote 3 gives it cumulative
slabs only. The model's own tariff table (`gentwin/tariff.py`) already knows
this: `tou_profile="none"` on LT-A, and also on LT-B, LT-C2, **LT-D1**, LT-D2,
LT-T and MT-6. LT-D1 matters here because two of this model's consumer
archetypes - **Hospital** and **Educational** - are mapped to LT-D1
(`config.CONSUMER_ARCHETYPES`), so this was never only a residential issue.

**What the code did before this fix.** `tariff.mu_profile()` ignored
`tc.has_tou` entirely and unconditionally applied the LT-family ratio
(mu^pk=1.20, mu^off=0.90) to *any* `tariff_code` passed in, including LT-A and
LT-D1. `MU_RESIDENTIAL_IS_SCENARIO = True` existed as a comment/export flag
but was never actually read by any pricing function - it asserted the
scenario was scenario, then applied it as if it were policy anyway.

**Fix (Category 3, code changed).** `gentwin/tariff.py::mu_profile()` and
`mu_peak_off()` now return `mu(t) = 1.0` for any class where `tc.has_tou` is
`False`, unless `MU_RESIDENTIAL_IS_SCENARIO` is explicitly flipped to `True` -
now defaulted `False` and repositioned as an opt-in ablation switch ("what if
a flat-billed class had ToU"), never the reported default. Because every
downstream price (`lp.py::_price_vector`, `lp.py::threshold_table`,
`billing.py::j1_linear`) reads price through `mu_profile()`/`mu_peak_off()`,
the fix is centralised - no other file needed a functional change. See
`docs/chapter3_vs_implementation.md` header commit / `gentwin/tariff.py` for
the exact diff, and `docs/adaptive_local_tou.md` Assumption A4, which had
already stated the intended behaviour that the code now enforces.

**Recommended formulation change.** Eq. (3.21) should read

```
π_h(t) = (π^ret_h + δ) μ_c(t),     μ_c(t) = 1                         if tariff class c has no gazette ToU row
                                    μ_c(t) = μ^pk / μ^off (per T^pk_i)  if it does
```

with a footnote naming which archetypes fall on each side (LT-A residential
and LT-D1 Hospital/Educational: flat; LT-C1 Industrial, LT-E Commercial and
Government: real ToU).

**What is preserved.** Residential/LT-D1 demand response is NOT removed. A
flat-rate household's `y_{h,t}` (curtailment) still carries a genuine private
bill benefit (using less energy still costs less, under any tariff), plus the
stress rebate `ρ·s_{i,d}`. Its `x_{h,t}` (shifting) now correctly earns zero
private benefit (moving energy between two identically-priced slots saves
nothing) but can still be selected by the LP when the *system* term -
`κ_{i,d}(t)·π_i(t)/θ₃` in Eq. (36)/(37): congestion relief, reserve adequacy,
storage/tie-line substitution, QoE - makes it worthwhile net of discomfort.
This is exactly the channel the review asked to keep.

---

## 2. Battery state of charge, Eq. (3.26) - CATEGORY 2, formulation wrong

Thesis: `S(t+1) = S(t) + η^ch·B^ch(t)Δ − B^dis(t)Δ/η^dis`.

Code (`gentwin/lp.py`, C2/C3 rows and the reported SoC trace):

```python
cum[Bch...] = cfg.ETA_CH * D      # charge into storage: eta_ch applied
cum[Bdis...] = -D                 # discharge OUT of storage: NOT divided by eta_dis
...
soc = soc0 + np.cumsum(cfg.ETA_CH * net["Bch"] - net["Bdis"]) * D
```

`B^dis(t)` is the amount physically drawn *from* the battery. Efficiency is
applied exactly once, at the point of delivery to the grid, inside the C1
supply constraint: `r[Bdis] = -cfg.ETA_DIS` (matching Eq. 3.25, which also
writes `η^dis·B^dis(t)` as the delivered term). If the SoC recursion in
Eq. (3.26) *also* divides by `η^dis`, the same discharge is penalised twice -
once shrinking what reaches the grid, once inflating what leaves the battery
for the same discharge - which is not physically consistent, and is
inconsistent with Eq. (3.25) inside the same chapter.

**Classification: Category 2.** The code's convention (apply `η_dis` once,
at delivery) is the physically correct one and needs no change.

**Recommended formulation change.** Eq. (3.26) should read

```
S(t+1) = S(t) + η^ch B^ch(t)Δ − B^dis(t)Δ
```

with a note that `B^dis(t)` is the battery-side draw (not the grid-delivered
quantity), consistent with how it is used in Eq. (3.25).

---

## 3. Objective weights θ₁, θ₂, θ₃ - CATEGORY 1, extension not documented

Thesis Eq. (3.27): `min Σ_t [f^cost(t) + f^rel(t) + Σ_h f^wel_h(t)]` - no
weights, and Table 3.1 does not list θ as a symbol.

Code (`gentwin/config.py`: `THETA_COST`, `THETA_RELIABILITY`,
`THETA_WELFARE`; used throughout `lp.py::build_day`'s objective assembly, and
swept by `06_lp_optimiser.py --sweep-theta`) treats
`F_i = θ₁ f^cost_i + θ₂ f^rel_i + θ₃ f^wel_i` as a first-class, tunable
structure - not a simplification that happens to equal Eq. (3.27) when all
three equal 1 (the current default).

**Classification: Category 1.** Keep the weights; they are load-bearing for
the sensitivity study the codebase already runs. Chapter 3 and Table 3.1
should introduce θ₁, θ₂, θ₃ explicitly and restate Eq. (3.27) as
`F = Σ_i (θ₁ f^cost_i + θ₂ f^rel_i + θ₃ Σ_h f^wel_{i,h})`.

---

## 4. C4 - substation export/capacity row missing from Chapter 3 - CATEGORY 1

Thesis Eq. (3.31), C4, is only the tie-line box bound
`0 ≤ T_ij(t) ≤ T^max_ij`.

Code (`gentwin/lp.py::build_day`) enforces that bound *and* a second,
separate per-slot inequality (also labelled "C4" internally, matching the
underlying formulation's Eq. 30/32): net export plus the substation's
κ-scaled post-response load must not exceed its firm capacity `P^max_i`. This
row genuinely binds when export is large relative to headroom and is not
derivable from the box bound alone.

**Classification: Category 1.** Keep the constraint; add it to Chapter 3's
C4 as a second inequality:

```
T_i(t) − η^dis B^dis_i(t) + κ_{i,d}(t) Σ_h L_h(t;x,y) ≤ P^max_i      ∀i, t
0 ≤ T_ij(t) ≤ T^max_ij                                                ∀(i,j), t
```

---

## 5. QoE / C7 constraint - missing from Chapter 3 entirely - CATEGORY 1

Chapter 3's constraint list stops at C6 (Eq. 3.31). There is no C7 and no
Eq. (35) anywhere in the methodology chapter as given - yet Chapter 4 cites
"(C7, Eq. 35)" directly (Figure 4.16's caption) and reports `QoE_min = 0.85`
as a headline result throughout. `docs/qoe_definition.md` already contains
the missing definition in full:

```
QoE_{h,m}(q) = 1 - ( w^sh Σ_t E^sh_{h,t} + w^cu Σ_t E^cu_{h,t} ) / E^base_{h,m}     ,   QoE_{h,m}(q) ≥ QoE^min
```

with `w^sh=0.4`, `w^cu=1.0`, `QoE^min=0.85`, and it is implemented as a real
per-consumer, per-billing-period LP row in `gentwin/lp.py` (the C7 rows in
`build_day`), bounded by `cfg.QOE_MIN`/`QOE_WEIGHT_SHIFT`/`QOE_WEIGHT_CURTAIL`.

**Classification: Category 1.** Keep the constraint (it is real, it binds for
~78% of demand-response actions per the Chapter 4 case study, and Chapter 4's
own results depend on it). Chapter 3 should add C7 as Eq. (3.32) verbatim from
`docs/qoe_definition.md`, so Chapter 4's citation of "Eq. 35" resolves to an
equation that actually exists in this thesis rather than to the internal
formulation document's numbering.

---

## 6. Per-household appliance power, Eq. (3.10) - CATEGORY 1, under-specified

Thesis: `P_{h,a}(t) = n_{h,a} P^rat_a ε_{a,h}(t) f_a(Θ(t))` - four factors,
with `ε` written as a function of both household and time.

Code (`gentwin/load_model.py::consumer_profile`):

```python
p = a.n * simultaneity * a.p_rat_kw * a.duty_cycle * u * f * jitter_amplitude
```

seven factors. In the code, `duty_cycle` (ε) is a fixed per-appliance
catalogue constant, not a function of `(a,h,t)`. The household/time variation
the thesis compresses into `ε_{a,h}(t)` is really three distinct mechanisms:
a binary on/off schedule `u_{h,a}(t)` sampled per household at synthesis
time, a per-household amplitude jitter, and (for multi-unit appliances) a
`simultaneity` fraction (not every owned AC runs at once). This is what gives
a population of otherwise-identical households a realistic coincidence
factor instead of one spike.

**Classification: Category 1.** Keep the richer model. Expand Eq. (3.10) to

```
P_{h,a}(t) = n_{h,a} · sim_a · P^rat_a · ε_a · u_{h,a}(t) · f_a(Θ(t)) · j_h
```

and define `u_{h,a}(t)` (household's own sampled on/off pattern), `sim_a`
(simultaneity fraction) and `j_h` (per-household amplitude jitter) alongside
`ε_a` (now a plain per-appliance constant, not household/time indexed).

---

## 7. Forecast error σ_k, Eq. (3.4) - documented approximation, no change needed

Thesis: `σ_k² = Var(p_{d+k} − p̂_{d+k|d})`.

Code (`gentwin/forecast.py::load_sigma`) uses the rolling-origin backtest
**RMSE** as σ by default, not the literal variance -
`RMSE_k = sqrt(Var_k + bias_k²) ≥ σ_k` - because the frozen forecasting
pipeline exports only the backtest summary, not per-record residuals. The
module docstring already states this is a deliberately **conservative**
substitution (it can only widen the robust peak p̃, never shrink it) and
that the literal Eq. (3.4) computation (`sigma_from_residuals()`) is available
and used automatically whenever a per-record backtest is supplied.

**Classification: not a mismatch to fix** - already documented in code as an
intentional, conservative proxy. Recommended: add one sentence to Chapter 3
under Eq. (3.4) stating that σ_k is estimated from backtest RMSE by default
(a conservative upper bound on the true standard deviation) and that the
exact variance is used when per-record residuals are available.

---

## 8. Per-household λ, γ^sh, γ^cu, y^max - CATEGORY 1 (already partly documented)

Eqs. (3.20), (3.22)-(3.24) write `λ`, `γ^sh`, `γ^cu`, `y^max` as plain,
un-subscripted scalars. The code (`config.py::sample_preferences`,
`population.py::synthesise`) samples `λ_h`, `γ^sh_h`, `γ^cu_h`, `y_max_h`
individually per household from its archetype's distribution - already
justified at length in `docs/formulation_v4_delta.md` §2-3 and
`docs/y_max_h_derivation.md` under the internal formulation's Remark 4
("household heterogeneity is individual... grouping changes nothing
structural"). Chapter 3's own text is the odd one out here: it should carry
the `_h` subscript on every occurrence to match both the code and the
internal formulation it otherwise follows.

**Classification: Category 1**, already resolved in spirit by existing docs;
this file's job is only to point Chapter 3's math at them.

---

## Summary table

| # | Item | Category | Action |
|---|---|---|---|
| 1 | Residential/LT-D1 μ(t) applied unconditionally | **3** | **code fixed** (`gentwin/tariff.py`) |
| 2 | Battery SoC Eq. (3.26) double-applies η^dis | 2 | fix formulation |
| 3 | θ₁,θ₂,θ₃ absent from Eq. (3.27)/Table 3.1 | 1 | add to formulation |
| 4 | C4 export/capacity row absent from Eq. (3.31) | 1 | add to formulation |
| 5 | QoE/C7 (Eq. 35) absent from Chapter 3 | 1 | add to formulation, from `docs/qoe_definition.md` |
| 6 | Appliance power Eq. (3.10) under-specified | 1 | expand formulation |
| 7 | σ_k uses RMSE proxy by default | - | document as intentional |
| 8 | λ, γ^sh, γ^cu, y^max un-subscripted | 1 | add `_h` subscripts in Ch.3 text |

Only #1 was a code change. Everything else leaves the implementation as-is
and asks Chapter 3's text to catch up to it.
