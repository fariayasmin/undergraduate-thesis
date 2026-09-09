# Stage C/D — forecast uncertainty into the optimisation

Implements Eqs. (13)–(16). The forecasting model is frozen; this stage only
reads its outputs.

---

## 1. Notation: three symbols, kept apart

| Symbol | Equation | Role | Code |
|---|---|---|---|
| `κ^scale_{i,d+k\|d}` | (16) | robust **scaling factor** `p̃ / p_{i,d}`; multiplies the decision-dependent load inside C1 (27) and C4 (30) | `kappa_scale` |
| `κ^cap_g` | (32) | **curtailment cap**, upper bound on `y_{h,t}`, per tier | `kappa_cap` |
| `z_β` | (14) | **quantile multiplier** on σ; 1.645 at β = 0.95 | `Z_BETA` |

The formulation reuses κ for the first two (the same collision Remark 1 flags
for ν and ς). `κ^cap` never appears in `gentwin/forecast.py`; `κ^scale` never
appears in the C6 code. `z_β` is not a κ at all.

## 2. Eq. (14) must not be written with κ in place of z_β

The formulation writes

$$\tilde{p}_{i,d+k|d} = \hat{p}_{i,d+k|d} + z_\beta\,\sigma_{i,k} \tag{14}$$

The variant `p̃ = p̂ + κσ` is **circular**. Eq. (16) *defines*
`κ^scale = p̃ / p_{i,d}`, so `p̃` would appear on both sides of its own
definition. Solving the fixed point:

$$\tilde{p} = \hat{p} + \frac{\tilde{p}}{p_{i,d}}\sigma \quad\Longrightarrow\quad \tilde{p} = \frac{\hat{p}}{1 - \sigma/p_{i,d}}$$

That is a different quantity. It is undefined at `σ = p_{i,d}` and changes sign
for `σ > p_{i,d}`. At Dhanmondi (`p_{i,d}` = 136.5 MW, `σ_14` = 23.5 MW) it
would return 155.1 MW against Eq. (14)'s 166.5 MW at k = 14 — and the
divergence grows with σ, exactly where a robust bound most needs to be
well-behaved.

It would also strain **Assumption 2**, which requires κ to be "computed before
the solve and held fixed". A self-referential κ is a fixed point of the
forecast layer rather than an exogenous input.

The two symbols do different jobs: `z_β` sets **how conservative** the bound is
(a confidence choice); `κ^scale` converts that bound into a **multiplier on the
reference-day profile** (a units conversion). A validation check reconstructs
`p̃` from `p̂ + z_β σ` to machine precision on every row, so the circular form
cannot creep back in.

## 3. How uncertainty propagates into the MOLP

```
σ_{i,k}                 forecast error scale, by horizon        Eq. (13)
   ↓  × z_β
p̃_{i,d+k|d}             robust peak                             Eq. (14)
   ↓  ÷ p_{i,d}
κ^scale_{i,d+k|d}       scaling factor                          Eq. (16)
   ↓  multiplies Σ_h L_h(t;x,y)
C1 right-hand side      supply adequacy                         Eq. (27)
C4 export limit                                                 Eq. (30)
   ↓  dual of C1
π_i(t)                  marginal value of energy at (i,t)
   ↓  κ^scale·π_i(t)/θ₃
Proposition 2 threshold  the "system" term of (34)/(35)
   ↓
λ*_g                    critical tier preference                Eq. (36)
```

The key structural point: **κ rescales the decision-dependent load inside the
constraints**, so the forecast and the household response interact. It is not a
pre-computed planning profile bolted on beside the LP. Inflating κ raises the
right-hand side of C1, which raises `π_i(t)`, which raises the system term of
the Proposition 2 threshold, which lowers `λ*_g` — so *more tiers cross into
demand response*. Forecast uncertainty therefore reaches household behaviour
through a chain that is explainable at every link.

Also, per the formulation's "How the K-step horizon is used": only the
`DISPATCH_DAYS` steps inside the operational horizon enter C1/C4 through
κ^scale. The remaining steps supply `s_{i,d+k}` and `Def_{i,d+k}` for the
regime `r_{i,d}` (46) and the priority weights `β_j(r)` (47). This is what keeps
the dispatch programme a **single simultaneous solve** (Corollary 1).
`get_kappa_scale()` raises if asked for a look-ahead day's κ, so a look-ahead
step cannot silently enter the constraints.

## 4. What it costs, quantified

Sweeping `z_β` (`--sweep`) prices the conservatism:

**Dhanmondi** (p_{i,d} = 136.5 MW, τ = 151.0 MW)

| z_β | confidence | mean κ^scale | extra demand vs point forecast | stress days |
|---|---|---|---|---|
| 0.000 | 50% | 0.9169 | — | 0/14 |
| 1.282 | 90% | 1.0826 | +18.1% | 7/14 |
| **1.645** | **95%** | **1.1295** | **+23.2%** | **10/14** |
| 2.326 | 99% | 1.2175 | +32.8% | 12/14 |

**Kalyanpur** (p_{i,d} = 138.0 MW, τ = 129.0 MW)

| z_β | confidence | mean κ^scale | extra demand | stress days |
|---|---|---|---|---|
| 0.000 | 50% | 0.8266 | — | 0/14 |
| **1.645** | **95%** | **0.9449** | **+14.3%** | **8/14** |
| 2.326 | 99% | 0.9939 | +20.2% | 14/14 |

Read the first row of each: at `z_β = 0` — the bare point forecast — **not a
single day is flagged stressed** at either substation. The entire stress
regime, and therefore every demand-response event the framework will generate,
exists *because* the model reasons about forecast uncertainty rather than a
point prediction. That is the case for the robust formulation, stated in
numbers.

The cost is real too. At Dhanmondi the LP is asked to serve 23.2% more than the
point forecast implies. Buying reliability against a 95% bound is not free, and
`z_β` is the dial. It belongs in the θ sensitivity sweep.

## 5. σ_{i,k}: what we have

Eq. (13) asks for `σ² = Var(p − p̂)`. The frozen pipeline exports the
rolling-origin recursive backtest **summary** — RMSE by horizon over ~37 origins
each. Since

$$\mathrm{RMSE}_k = \sqrt{\mathrm{Var}_k + \mathrm{bias}_k^2} \;\ge\; \sigma_k$$

using RMSE as σ is **conservative**: it can only widen `p̃`, the safe direction.

**Non-monotonicity.** The raw series decreases with k at 4 of 13 steps for
Dhanmondi and 6 of 13 for Kalyanpur (e.g. Dhanmondi k=3 at 9.68 MW against k=1
at 12.18). With ~37 origins per horizon that is sampling noise, not a claim that
a 3-day forecast beats a 1-day one. `SIGMA_MONOTONE = "envelope"` applies a
running maximum; `"raw"` and `"isotonic"` (PAVA, no sklearn dependency) are
available for the ablation.

**Two independent estimates agree.** The backtest at k=1 gives 12.18 MW at
Dhanmondi and 7.11 MW at Kalyanpur; the one-day-ahead test panel in
`metrics_summary.csv` gives 11.97 and 6.78 MW. Agreement to +1.7% and +4.7%
from two separately-computed quantities is a real consistency check, and it is
run automatically.

**Residual shape justifies z_β = 1.645.** That value presumes normality. For a
Gaussian, MAE/RMSE = √(2/π) = 0.7979. Measured: **0.7987** (Dhanmondi) and
**0.7964** (Kalyanpur). The residuals are consistent with Gaussian at the
aggregate level, so the quantile multiplier is defensible rather than
conventional. A materially lower ratio would mean heavier tails and z_β would
under-cover.

**The one test we cannot run.** Empirical coverage — what fraction of realised
peaks fall at or below `p̃` — needs the **per-record** backtest (the frame
before `summarise_backtest()`), which was not exported. `coverage_test()` and
`sigma_from_residuals()` are written and will run the moment that file exists;
`--coverage PATH` is the entry point. Until then, σ = RMSE is a bound, not a
measurement, and the 95% coverage is asserted from the Gaussian shape check
rather than observed. **This is the weakest link in Stage C/D and should be
closed if the per-record backtest can be re-exported.**

## 6. τ_i must be estimated on a trailing window

Eq. (15) sets `s_{i,d+k} = 1[p̃ ≥ τ_i]`. Estimating τ on the full record
**fails**:

| | full-record p95 | trailing-365d p95 | trailing-365d max | stress days at z_β=1.645 |
|---|---|---|---|---|
| Dhanmondi | 187.0 MW | 151.0 MW | 163.0 MW | 0/14 → **10/14** |
| Kalyanpur | 123.0 MW | 129.0 MW | 141.0 MW | 14/14 → **8/14** |

Dhanmondi's full-record p95 (187.0 MW) sits **above its entire last-365-day
range**, a legacy of its much higher 2019–2023 level. A threshold there can
essentially never fire: `s ≡ 0`, no stress rebate in Eq. (19), no regime 2 or 3
in Eq. (46), no events, no priority pool. Kalyanpur shows the opposite failure —
the full-record threshold is too *low* and fires every single day, which is
equally uninformative.

A trailing window tracks the level each substation actually operates at now.
`TAU_TRAILING_DAYS = 365`, `TAU_QUANTILE = 0.95`, both in config. A validation
check refuses a τ above the trailing maximum, and another rejects a degenerate
stress indicator (all-zero or all-one over the horizon).

## 7. Assumption 1, enforced not assumed

The denominator of Eq. (16) must be the realised peak of the **same reference
day** whose profile builds C1's right-hand side.

- `resolve_reference_day()` pins `d` to the day immediately before the
  forecast's first target and **raises** if that day's peak is missing.
- `assert_population_alignment()` is called by Stage A/B and raises if the
  synthetic population was built for a different day.

Resolved here: **d = 2026-06-29**, `p_{i,d}` = 136.50 MW (Dhanmondi) and
138.00 MW (Kalyanpur); first forecast target 2026-06-30.

> **Carry-forward for Stage A/B**: the synthetic population must reproduce
> `max_t Λ_i(t)` = 136.50 MW at Dhanmondi and 138.00 MW at Kalyanpur, both for
> 2026-06-29. Anything else violates Assumption 1.

Note that Kalyanpur's reference-day peak *time* is unusable (it is one of the
00:00 artefact rows). The level is fine, so Eq. (16) is unaffected, but the
argmax-timing acceptance test in Stage B will have to fall back to the
distributional peak window rather than that day's recorded time.

## 8. Assumptions and limitations

**A1 — σ from RMSE is an upper bound.** Bias is not separable from the summary.
Conservative, but the bound's tightness is unknown.

**A2 — z_β = 1.645 presumes Gaussian residuals.** Supported by MAE/RMSE ≈
0.798 at the aggregate level, but that statistic is insensitive to tail
behaviour specifically. Skew and kurtosis need per-record residuals.

**A3 — σ_{i,k} is horizon-dependent but not state-dependent.** One σ per
horizon, applied on every day. In reality forecast error is larger on hot days
and around holidays. A conditional `σ_{i,k}(weather, holiday)` is a natural
extension and is why `weather_source` and `holiday_type` are carried through
into the κ table.

**A4 — The weather band is a scenario tool, not a confidence interval.** Both
arms are computed under `weather_mode="climatology"` at ±1σ while the central
case uses API weather, so the central legitimately falls **outside** the band —
on 3/14 Dhanmondi days and 6/14 Kalyanpur days, in every case a day with
`weather_source == "api"`. The band answers "how much would the forecast move
under ±1σ weather", not "where will the actual land". **Forecast uncertainty in
Eq. (14) comes from σ_{i,k} and never from this band.** A validation check
confirms the outside-band days are API days only.

**A5 — Empirical coverage untested.** See §5.

**A6 — Assumption 1 is exact only at the peak slot.** The formulation says so:
κ rescales a *level*, but Eq. (10) forecasts peaks, not profiles, so κ is exact
at the peak slot and approximate elsewhere. If the forecaster ever emits a full
profile, the slot-level `κ_{i,t} = Λ̃_i(t)/Λ_i(t)` substitutes with no
structural change.

## 9. Reproducing

```bash
python 05c_forecast_to_kappa.py --build     # Eqs. 13-16, 16 checks, cache
python 05c_forecast_to_kappa.py --table     # per-horizon table -> CSV
python 05c_forecast_to_kappa.py --sweep     # z_beta propagation
python 05c_forecast_to_kappa.py --plot      # outputs/forecast_to_kappa.png
python 05c_forecast_to_kappa.py --coverage per_record_backtest.csv
```

All 16 validation checks pass at both substations.
