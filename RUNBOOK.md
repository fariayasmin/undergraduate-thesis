# RUNBOOK — what to run, in what order

**New here?** `docs/HOW_IT_WORKS.md` is the non-technical overview — read that
first, then come back.

## TL;DR

```bash
pip install -r requirements.txt
bash run_pipeline.sh
```

That runs the whole thing, forecast inputs to knowledge graph, in about
3 minutes. Everything it needs is already in `data/`.

If you prefer to drive it yourself, **the single most important file to run is
`06_lp_optimiser.py`** — but only after `05c` and `03` have populated the
caches. Full order below.

---

## File structure

```
gentwin_sg/
│
├── run_pipeline.sh              <- runs every stage in order
├── RUNBOOK.md                   <- this file
├── README.md                    <- status, findings, key decisions
├── requirements.txt
│
├── gentwin/                     <- importable library (all logic lives here)
│   ├── __init__.py
│   ├── config.py                every parameter, tagged EQ / SOURCE / WHY
│   ├── tariff.py                BERC June-2026 gazette, slabs, ToU, mu_i(t)
│   ├── peak_window.py           adaptive local T^pk_i derivation
│   ├── peak_time_audit.py       is `Peak_Time` a measurement or a convention?
│   ├── service_area.py          Voronoi + Overpass + cache + fallback
│   ├── temperature.py           Theta(t), Parton-Logan reconstruction
│   ├── population.py            synthetic consumers, per-household preferences
│   ├── load_model.py            Eqs. (2)-(8)
│   ├── validation.py            Stage B gate, T1/T2, seed stability
│   ├── forecast.py              Eqs. (13)-(16): sigma, p~, kappa, P^str
│   ├── lp.py                    Eqs. (17)-(35), HiGHS, duals, Stage G
│   ├── regime.py                Eqs. (44)-(53): regime, events, pools
│   └── kg.py                    Eq. (57), graph builder and exporters
│
├── 00_tariff_bpdb.py            <- runners. Numbered = execution order.
├── 00b_peak_window.py
├── 01_service_area_osm.py
├── 02_consumer_catalogue.json      (data, not a script)
├── 03_population_synthesis.py
├── 03b_temperature_profile.py
├── 04_load_generator.py
├── 05_validate_against_bpdb.py
├── 05c_forecast_to_kappa.py
├── 06_lp_optimiser.py
├── 07_regime_events_pool.py
├── 08_knowledge_graph.py
│
├── data/
│   ├── BPDB_Dhaka_City_Substations_Page3.csv    input (given)
│   ├── metrics_summary.csv                      input (frozen forecaster)
│   ├── forecast_outputs/                        input (frozen, READ-ONLY)
│   │   ├── Dhanmondi_14day_forecast.csv
│   │   ├── Dhanmondi_recursive_backtest.csv
│   │   ├── Kalyanpur_14day_forecast.csv
│   │   └── Kalyanpur_recursive_backtest.csv
│   └── cache/                                   generated, commit these
│       ├── osm_*.json           service area + consumer mix
│       ├── peak_window_*.json   T^pk_i
│       ├── kappa_*.json         Eqs. (13)-(16) per horizon step
│       ├── population_*.json    400 consumers per substation
│       └── theta_*.json         Theta(t)
│
├── docs/                        <- the written record
│   ├── HOW_IT_WORKS.md              PLAIN-LANGUAGE OVERVIEW - start here
│   ├── formulation_v4_delta.md      what changed in the revision
│   ├── forecast_to_kappa.md         Stage C/D
│   ├── stage_ab_population.md       Stage A/B
│   ├── adaptive_local_tou.md        the ToU mechanism
│   ├── peak_time_reliability.md     the Peak_Time audit
│   ├── osm_preregistration.md       LOCKED settings; read before --fetch
│   ├── y_max_h_derivation.md        the one C6 extension
│   └── qoe_definition.md            C7, and its effect on Prop. 2
│
└── outputs/                     <- everything generated
    ├── *.png                    figures
    ├── *.csv / *.json           tables
    └── kg/                      THE DELIVERABLE FOR NEO4J
        ├── nodes_*.csv          22 node files
        ├── relationships.csv
        ├── load_kg.cypher       constraints, indexes, batched LOAD CSV
        ├── query_pack.cypher    11 worked queries
        └── schema.json          for the Twin Adapter to introspect
```

---

## Run order

Each stage writes a cache the next one reads, so order matters. Stages 1-2 need
running only once.

| # | Command | Needs | Produces | ~time |
|---|---|---|---|---|
| 1 | `python 00_tariff_bpdb.py --all` | — | tariff reference, 12 checks | 1 s |
| 2 | `python 01_service_area_osm.py --offline-seed` | — | `osm_*.json` | 1 s |
| 3 | `python 00b_peak_window.py --audit --derive --compare --plot` | BPDB csv | `peak_window_*.json` | 20 s |
| 4 | `python 05c_forecast_to_kappa.py --build --table --plot` | forecast_outputs | `kappa_*.json` | 5 s |
| 5 | `python 03b_temperature_profile.py --date 2026-06-29 --plot` | BPDB csv | `theta_*.json` | 1 s |
| 6 | `python 03_population_synthesis.py --build --report` | steps 2 & 4 | `population_*.json` | 10 s |
| 7 | `python 04_load_generator.py --build --plot --xi-series 30` | step 6 | Lambda_i(t) csv, figures | 40 s |
| 8 | `python 05_validate_against_bpdb.py --validate --ablation` | step 6 | **Stage B gate** | 60 s |
| 9 | **`python 06_lp_optimiser.py --solve --days 3 --plot`** | steps 4 & 6 | LP decisions, duals | 10 s |
| 10 | `python 07_regime_events_pool.py --run` | step 9 | events, pools | 15 s |
| 11 | `python 08_knowledge_graph.py --build --queries` | steps 9 & 10 | **`outputs/kg/`** | 25 s |
| 12 | `python 06_lp_optimiser.py --solve --month 30` | steps 4 & 6 | 30-day solve | 60 s |
| 13 | **`python 09_monthly_billing.py --report --plot`** | step 12 | **monthly bills, savings** | 40 s |
| 14 | `python 08_knowledge_graph.py --build --queries --billing ""` | steps 9–13 | KG **with** MonthlyBill | 25 s |

Steps 9–11 give the 3-day detail (the dispatch figure). Steps 12–13 give the
month: the billing period runs **2026-06-14 → 2026-07-13**, 16 observed days
plus the 14 forecast days, which is the operationally realistic case — a
utility bills a month running from the recent past into the near future.
`--month 30` picks that window automatically.

**Step 4 must run before step 6.** The reference day and `p_{i,d}` come from
the kappa table, and `assert_population_alignment()` refuses to build a
population for any other day (Assumption 1).

### Optional runs

```bash
# Scarcity scenario. The base case never binds G^max, so nothing above
# regime 2 fires without this. Exercises regime 3, transfers, unserved
# load and the Stage G surcharge.
python 06_lp_optimiser.py --solve --days 2 --scale-gmax 0.55 --scale-pmax 0.62
python 07_regime_events_pool.py --run --scarcity
python 08_knowledge_graph.py --build --scarcity

# Quantile pooling: fixes the Eq. (50) degeneracy the formulation predicts
python 07_regime_events_pool.py --run --pool-rule quantile

# theta sensitivity (section 9.6 open item)
python 06_lp_optimiser.py --sweep-theta

# Single-substation model, Remark 1: S = {i}, L = empty
python 06_lp_optimiser.py --solve --separate

# The real OSM fetch. READ docs/osm_preregistration.md FIRST.
# Needs internet; then re-run steps 6 and 8.
python 01_service_area_osm.py --fetch
```

---

## Loading into Neo4j

```bash
cp outputs/kg/nodes_*.csv outputs/kg/relationships.csv "$NEO4J_HOME/import/"
cypher-shell -u neo4j -p <password> -f outputs/kg/load_kg.cypher
```

Then open `outputs/kg/query_pack.cypher` in Neo4j Browser. It contains the six
traversals from your brief plus the slot-level questions ("who curtailed at
18:30", "battery state at 17:00", "which hospital loads were never curtailed").

Graph size for a 3-day run: **62,782 nodes, 178,684 relationships**, zero
dangling endpoints.

---

## Two things to know before reading the output

**The Stage B gate passes, but Dhanmondi's T2 is UNDETERMINED, not passed.**
Its peak lands inside `T^pk_i` on only 9 of 40 population draws, and the
`Peak_Time` field it is tested against rates LIKELY_CONVENTION in the audit
(2.7% of readings on `:30`, against 17.7% at Kalyanpur). Kalyanpur is a robust
pass at 40/40. See `docs/peak_time_reliability.md`.

**The base LP never binds.** G^max is 200 MW against ~145 MW of demand, so
pi_i(t) sits flat at the producer price, there are no transfers, no unserved
load, and regime 3 cannot fire. That is a correct result for these parameters,
not a bug. To see the emergency path, run the scarcity scenario, where pi rises
to 60.0 Tk/kWh — exactly `v_curtailable`, which is Proposition 3's merit order
made visible.

**Population realism was fixed.** `u_{h,a}(t)` is now sampled as the BINARY
on/off pattern Eq. (2) specifies: each household draws its own ON slots to a
realistic `daily_hours`, with a `simultaneity` factor for multi-unit
appliances. Residential consumption went from 1,110 to **221 kWh/month**,
connections from 37k to **376k fleet-wide**, and the diversity factor (IEEE 141: sum of individual peaks / coincident peak)
from 1.32 to **2.4–2.8**, i.e. a coincidence factor of 0.42/0.36 — inside the
0.2–0.4 band urban residential feeders show. All 12 billing validation checks pass.

**Net benefit is still negative**, and that is a finding rather than a fault.
The Tk-equivalent discomfort exceeds bill saving plus incentive, which follows
directly from Eq. (24): the LP minimises `λ_h·J1 + (1−λ_h)·J2`, so at λ = 0.6
an action is taken whenever `saving > 0.67 × discomfort`. The system term
`κ·π/θ₃` also buys grid value the consumer never sees on their bill. **At
ρ = 3 Tk/kWh the rebate does not make participation individually rational** —
sweeping ρ to find the level that does is a strong thesis result.

Not implemented, by instruction: `06a_verify_propositions.py`, Stage I (Twin
Adapter), Stage J (LLM explanation).
