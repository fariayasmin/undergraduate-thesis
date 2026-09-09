# GenTwin-SG — implementation

Forecast → Synthetic consumers → Aggregation validation → LP → Priority Pool → Knowledge Graph

Sole source of truth: **GenTwinSG_Formulation_v3.pdf**. Where the earlier
implementation (`06_molp_optimizer`, GCS scoring, `DrEvent`) conflicts, the
formulation wins. The forecasting pipeline (`load_prediction.py`,
`forecast_outputs/`) is **read-only** and is never retrained or modified.

Stages I (Twin Adapter) and J (LLM explanation) are deliberately **not** built.

---

## Start here

New to this? Read **`docs/HOW_IT_WORKS.md`** — a plain-language tour of what the
system does, with a worked household example and an honest account of which
numbers are measured and which are assumed. No equations.

## Status

| Stage | Module | State |
|---|---|---|
| — · Formulation revision | `docs/formulation_v4_delta.md` | **applied** |
| 0 · Tariff | `00_tariff_bpdb.py`, `gentwin/tariff.py` | **done, verified** |
| 0b · Adaptive local ToU | `00b_peak_window.py`, `gentwin/peak_window.py` | **done, verified** |
| A′ · Service area & OSM | `01_service_area_osm.py`, `gentwin/service_area.py` | **done** (cache seeded offline; run `--fetch` for real OSM) |
| — · Config | `gentwin/config.py` | **done** |
| C/D · Forecast → κ, Eq. 13–16 | `05c_forecast_to_kappa.py`, `gentwin/forecast.py` | **done, 16/16 checks pass** |
| A/B · Consumer catalogue | `02_consumer_catalogue.json` | **done** |
| A/B · Population synthesis | `03_population_synthesis.py`, `gentwin/population.py` | **done** |
| A/B · Temperature Θ(t) | `03b_temperature_profile.py`, `gentwin/temperature.py` | **done, 5/5 checks** |
| B · Load generation, Eq. 2–6 | `04_load_generator.py`, `gentwin/load_model.py` | **done** |
| B · Aggregation validation, Eq. 6–8 | `05_validate_against_bpdb.py`, `gentwin/validation.py` | **done — gate FAILS at Dhanmondi, see below** |
| E/F · LP, Eq. 17–33 | `06_lp_optimiser.py` | next |
| — · Proposition checks | `06a_verify_propositions.py` | next |
| G/H · Regime, events, pool, Eq. 38–51 | `07_regime_events_pool.py` | next |
| K · Knowledge graph, Eq. 55 | `08_knowledge_graph.py` | next |

---

## Run

```bash
pip install -r requirements.txt

python 00_tariff_bpdb.py --all            # tariff: verify, tables, mu(t), bills
python 01_service_area_osm.py --fetch     # OSM, ONLINE, run once then commit cache
python 01_service_area_osm.py --report    # summarise the cache
```

`--fetch` is the only step that touches the network. It writes
`data/cache/osm_<substation>.json`; everything downstream reads that cache, so
after one online run the whole framework is reproducible offline. If you cannot
run it, `--offline-seed` writes a documented fallback stamped
`provenance = "fallback_literature"` — the schema is identical, but no artefact
built from it can be mistaken for surveyed data.

`--print-query` emits the Overpass QL to paste into <https://overpass-turbo.eu>
if you would rather run it by hand.

---

## Key decisions, with their justification

**Tariff is real, not approximated.** `gentwin/tariff.py` transcribes the BERC
order of 03 June 2026 (memo 28.01.0000.012.04.003.26.2521) in full — all 9 LT,
8 MT, 4 HT and 2 EHT classes, the LT-A cumulative slabs, demand charges, 5%
VAT, the 0.5% prepaid rebate, the footnote-7 blend for ~80%-residential MT-5
consumers, and the footnote 4–6 ToU windows. `--verify` recomputes the
gazette's own peak/flat ratios from the transcribed numbers as a proof-reading
test; all 12 checks pass.

**μ(t) is derived, not invented.** LT-A has no ToU in the order. But LT-C1,
LT-E, LT-D3 and every MT/HT/EHT class do, and they land on two clean families:
`μ^pk = 1.20` for LT, `1.25` for MT/HT/EHT, `μ^off = 0.90` uniformly. The
residential scenario adopts the LT family — the ratio the regulator itself
applied to every other low-tension consumer. Flagged
`MU_RESIDENTIAL_IS_SCENARIO = True`.

**Adaptive local ToU (see `docs/adaptive_local_tou.md`).** BERC supplies the
peak/off-peak *ratios*; the twin derives the *activation window* per substation
from congestion history. T^pk_i comes from a load-weighted circular KDE over
peak timing on the p95 congestion days, cut at 0.50 × max density, reduced to
the longest contiguous run, and bootstrapped 500×.

| | Dhanmondi | Kalyanpur |
|---|---|---|
| T^pk_i | 12:00–14:30 | 18:00–21:00 |
| National ToU | 17:00–23:00 | 17:00–23:00 |
| Overlap | **0 of 5 slots** | 6 of 6 |
| Congestion mass captured | 0.709 vs 0.006 national | 0.416 vs 0.653 |
| Precision (mass/priced slot) | **285× national** | 1.3× national |
| Bootstrap stability | width IQR 0 — stable | width IQR 5 — indicative |

Not a replacement tariff: every artefact records `mu_source` (BERC) and
`window_source` (GenTwin-SG) separately. μ is applied to the consumer-facing
price in Eq. (19), **not** to `f_cost` — see §5 of the doc for why that would
corrupt π_i(t).

**Kalyanpur peak-time artefact.** 13.1% of Kalyanpur rows carry a peak time of
exactly 00:00 (180 in 2024+ alone) against 1.0% at Dhanmondi. A midnight peak
at a Dhaka distribution substation is not credible at that frequency; it is the
default written when the field was not recorded. Excluded from the T^pk
derivation via `PEAK_TIME_ARTEFACT_SLOTS` and reported.

**48 slots is native, not a refinement.** Every `_Peak_Time` value in the BPDB
CSV falls on `:00` or `:30` — `load_prediction.is_clocktime_artifact()` already
encodes this. Hourly slots would discard the only intra-day observation the
dataset contains.

**Stage A/B gate: Kalyanpur PASS, Dhanmondi FAIL** (`docs/stage_ab_population.md`).
Peak *level* is exact at both (136,500 / 138,000 kW) but that is satisfied by
construction. Peak *timing* is the real test: Kalyanpur's synthetic argmax lands
at 19:00 inside its 18:00–21:00 window; Dhanmondi's argmax is **seed-dependent** — inside the window on only 9 of 40
population draws — so T2 there is recorded as **UNDETERMINED**, neither pass
nor fail. Two independent reasons not to treat its timing as settled: the
`Peak_Time` target is rated LIKELY_CONVENTION by the reliability audit, and the
composition does not commit to an answer. The fallback mix has *not* been
adjusted.

**Units are kW / kWh / Tk**, per the revised formulation's symbol tables. The
frozen forecast CSVs and the BPDB dataset are in MW and are converted on read,
in one place each.

**Stage C/D is complete** (`docs/forecast_to_kappa.md`). Reference day
**2026-06-29**, p_{i,d} = 136,500 kW (Dhanmondi) / 138,000 kW (Kalyanpur).
κ^scale ranges 0.936–1.263 and 0.893–1.010. At z_β = 0 — the bare point
forecast — **zero** days are flagged stressed at either substation; at
z_β = 1.645, 10/14 and 8/14 are. The entire stress regime exists because the
model reasons about uncertainty rather than a point prediction, at a cost of
+23.2% / +14.3% demand on C1's right-hand side.

**σ_{i,k} from the recursive backtest.** The rolling-origin RMSE by horizon is
√(Var + bias²) ≥ σ, so using it in Eq. (14) is conservative in the direction
that matters for a robust peak. The raw series is non-monotonic at Dhanmondi
(k=3 RMSE 9.68 < k=1 12.18) — a sampling artefact of ~37 origins per horizon,
not a claim that a 3-day forecast beats a 1-day one — so
`SIGMA_MONOTONE = "envelope"` applies a running maximum. Both are reported.

---

## Documented additions to the formulation

Only two, both derived in full under `docs/`:

1. **`y^max → y^max_h`** (`docs/y_max_h_derivation.md`) — per-tier curtailment cap
   in C6, the single approved structural extension. Shown to preserve
   linearity and to leave the Proposition 2 threshold *identical*; only the
   value attained at the bound changes. Note the derivation opens with a
   notation warning: the formulation uses κ for both the curtailment cap
   (Eq. 32) and the robust scaling factor (Eq. 16), and Eq. (36) contains the
   *scaling factor*, so it is untouched by this change.

2. **QoE definition for C7** (`docs/qoe_definition.md`) — the formulation
   states C7 but never defines QoE. An affine definition is supplied.
   **This raises a real issue with Proposition 2**: C7 is not a box
   constraint, so if it binds, its dual enters the reduced cost and the
   threshold gains a fourth term. That is a property of the formulation, not
   of the definition. `06a_verify_propositions.py` will check C7 slackness at
   every optimum and report the three-term threshold only where it holds;
   where C7 binds, the dual is stored on the KG decision node as `qoe_term`.

---

## Hospitals

Not hard-coded as non-curtailable. They receive `λ = 0.02` and
`γ^cu = 5000 BDT/kWh`, so the Proposition 2 threshold reads roughly
`27 > 4900` — false at any plausible shadow price. Non-curtailment is an
**output** of the optimisation, explainable slot by slot, rather than a rule
written into the model.

---

## Layout

```
gentwin/
  config.py           every parameter, tagged EQ / SOURCE / WHY
  tariff.py           BERC June 2026 gazette, slabs, ToU, mu(t)
  service_area.py     Voronoi + Overpass + cache + fallback
00_tariff_bpdb.py     runner
01_service_area_osm.py runner
docs/
  y_max_h_derivation.md
  qoe_definition.md
data/
  cache/              osm_*.json  (commit these)
  forecast_outputs/   drop the forecast CSVs here
outputs/
```

Numbered runners keep your existing convention; importable logic lives in the
`gentwin/` package so modules can import each other (a module named `00_x.py`
cannot be imported normally).

---

## Still needed

1. **The PER-RECORD recursive backtest** (the frame before
   `summarise_backtest()`, with columns `horizon, actual_mw, pred_mw`). Only
   the summaries were exported. Without it, σ is bounded by RMSE rather than
   measured, and the empirical coverage of p̃ cannot be tested —
   the weakest link in Stage C/D. `--coverage PATH` is ready for it.
2. **`nodes_Substation.csv`** — coordinates are currently provisional and
   flagged as such. Three or more fleet sites also unlock the Voronoi service
   area instead of the 2 km disc.
3. **One online run of `01_service_area_osm.py --fetch`**, then commit the
   cache.

---

## Stages E/F, H and K — complete

```bash
python 06_lp_optimiser.py --solve --days 3 --plot
python 06_lp_optimiser.py --solve --scale-gmax 0.55 --scale-pmax 0.62   # scarcity
python 06_lp_optimiser.py --sweep-theta          # section 9.6 open item
python 07_regime_events_pool.py --run [--pool-rule quantile] [--scarcity]
python 08_knowledge_graph.py --build --queries
```

**LP** — 43,568 variables, 1,280 rows, ~1.7 s per day with HiGHS. Solved day by
day on a receding basis, chaining SoC; within a day it is one simultaneous
programme (Corollary 1). `--separate` gives Remark 1's `S={i}, L=∅`.

**Knowledge graph** — 62,782 nodes / 178,684 relationships over 3 days, zero
dangling endpoints. Every `DrAction` carries the stored Eq. (36)/(37)
decomposition, so an action is explainable without re-running the solver.
`Forecast` and `Regime` are separate nodes because `s_{i,d}` precedes the solve
and `r_{i,d}` follows it.
