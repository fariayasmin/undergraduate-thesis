#!/usr/bin/env bash
# GenTwin-SG :: full pipeline, forecast inputs -> knowledge graph.
#
# Usage:
#   bash run_pipeline.sh                 # base arm: current tariff, proposed
#                                         # DR, default rho, IR off
#   bash run_pipeline.sh --rho0          # + current-policy arm (rho = 0)
#   bash run_pipeline.sh --ir            # + individual-rationality arm
#   bash run_pipeline.sh --scarcity      # + scarcity scenario
#   bash run_pipeline.sh --dr-window     # + curtailment scoped to the DR window (Follow-up F3)
#   bash run_pipeline.sh --rho0 --ir --scarcity --dr-window   # all variants
#
# Issue C fix: the monthly flow (steps 9-13) now runs entirely against
# _period-tagged outputs - LP, regime/pooling, billing AND the knowledge
# graph. The earlier version restored an UNTAGGED 3-day LP/regime solve
# immediately before building the KG, so the KG never actually reflected
# the monthly billing period it was built alongside. That step is gone.
# The untagged 3-day solve is kept ONLY for the dispatch figure (step 8),
# and is never read by anything downstream of it.
set -euo pipefail
cd "$(dirname "$0")"

step () { printf '\n\033[1m===> %s\033[0m\n' "$1"; }

step "1/12  Tariff (BERC June 2026)"          ; python3 00_tariff_bpdb.py --all
step "2/12  Service area / consumer mix"      ; python3 01_service_area_osm.py --offline-seed
step "3/12  Peak-time audit + adaptive ToU"   ; python3 00b_peak_window.py --audit --derive --compare --plot
step "4/12  Forecast -> kappa (Eqs. 13-16)"   ; python3 05c_forecast_to_kappa.py --build --table --plot
step "5/12  Theta(t)"                         ; python3 03b_temperature_profile.py --date 2026-06-29 --plot
step "6/12  Population synthesis"             ; python3 03_population_synthesis.py --build --report
step "7/12  Load generation (Eqs. 2-8)"       ; python3 04_load_generator.py --build --plot --xi-series 30
step "8/12  Stage B gate vs BPDB"             ; python3 05_validate_against_bpdb.py --validate --ablation
step "9/12  LP, 3-day detail (dispatch figure ONLY - never fed to the KG)"
python3 06_lp_optimiser.py --solve --days 3 --plot
python3 07_regime_events_pool.py --run

step "10/12 Monthly LP - BASE arm (_period)"  ; python3 06_lp_optimiser.py --solve --month 30
step "11/12 Monthly regime/pool + billing (_period)"
python3 07_regime_events_pool.py --run --tag _period
python3 09_monthly_billing.py --report --plot
step "12/12 Knowledge graph, monthly (_period, with billing)"
python3 08_knowledge_graph.py --build --queries --tag _period --billing

if [[ "$*" == *"--rho0"* ]]; then
  step "extra  Current-policy arm: rho = 0 (_period_rho0)"
  python3 06_lp_optimiser.py --solve --month 30 --rho 0
  python3 07_regime_events_pool.py --run --tag _period_rho0
  python3 09_monthly_billing.py --report --tag _period_rho0
  python3 08_knowledge_graph.py --build --tag _period_rho0 --billing _period_rho0
fi

if [[ "$*" == *"--ir"* ]]; then
  step "extra  Individual-rationality arm (_period_ir)"
  python3 06_lp_optimiser.py --solve --month 30 --ir
  python3 07_regime_events_pool.py --run --tag _period_ir
  python3 09_monthly_billing.py --report --tag _period_ir
  python3 08_knowledge_graph.py --build --tag _period_ir --billing _period_ir
fi

if [[ "$*" == *"--scarcity"* ]]; then
  step "extra  Scarcity scenario (_period_scarcity, regime 3)"
  # 0.55/0.62 was INFEASIBLE before the Round-3 C4 fix (HiGHS status 8 on
  # 2026-07-12 - the LP correctly raised rather than truncating, but there
  # was genuinely no way to satisfy C4 without shedding load, which C4 could
  # not do). C4 now includes -sum_c Z_c(t), exactly like C1, so shed load
  # also relaxes the transformer/export cap; 0.55/0.62 now solves and
  # produces real Z>0 at the VoLL price plus capacity_deficit firing on
  # every day - the emergency regime this scenario was meant to exercise.
  # See docs/thesis_changes.md, Round-3 follow-up.
  python3 06_lp_optimiser.py --solve --month 30 --scale-gmax 0.55 --scale-pmax 0.62
  python3 07_regime_events_pool.py --run --tag _period_scarcity
  python3 09_monthly_billing.py --report --tag _period_scarcity
  python3 08_knowledge_graph.py --build --tag _period_scarcity --billing _period_scarcity
fi

if [[ "$*" == *"--dr-window"* ]]; then
  step "extra  Curtailment scoped to the DR-activation window (_period_dr_window)"
  python3 06_lp_optimiser.py --solve --month 30 --curtail-scope dr_window
  python3 07_regime_events_pool.py --run --tag _period_dr_window
  python3 09_monthly_billing.py --report --tag _period_dr_window
  python3 08_knowledge_graph.py --build --tag _period_dr_window --billing _period_dr_window
fi

printf '\n\033[1mDone.\033[0m Neo4j import files: outputs/kg/\n'
printf 'Load with:  cypher-shell -f outputs/kg/load_kg.cypher\n'
