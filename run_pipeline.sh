#!/usr/bin/env bash
# GenTwin-SG :: full pipeline, forecast inputs -> knowledge graph.
# Usage:  bash run_pipeline.sh [--scarcity]
set -euo pipefail
cd "$(dirname "$0")"

step () { printf '\n\033[1m===> %s\033[0m\n' "$1"; }

step "1/14  Tariff (BERC June 2026)"          ; python3 00_tariff_bpdb.py --all
step "2/14  Service area / consumer mix"      ; python3 01_service_area_osm.py --offline-seed
step "3/14  Peak-time audit + adaptive ToU"   ; python3 00b_peak_window.py --audit --derive --compare --plot
step "4/14  Forecast -> kappa (Eqs. 13-16)"   ; python3 05c_forecast_to_kappa.py --build --table --plot
step "5/14  Theta(t)"                         ; python3 03b_temperature_profile.py --date 2026-06-29 --plot
step "6/14  Population synthesis"             ; python3 03_population_synthesis.py --build --report
step "7/14  Load generation (Eqs. 2-8)"       ; python3 04_load_generator.py --build --plot --xi-series 30
step "8/14  Stage B gate vs BPDB"             ; python3 05_validate_against_bpdb.py --validate --ablation
step "9/14  LP, 3-day detail (the dispatch figure)" ; python3 06_lp_optimiser.py --solve --days 3 --plot
step "10/14 Regime, events, priority pool"    ; python3 07_regime_events_pool.py --run
step "11/14 LP over the 30-day billing period"; python3 06_lp_optimiser.py --solve --month 30
step "12/14 Monthly bills and savings"        ; python3 09_monthly_billing.py --report --plot
step "13/14 Restore 3-day detail for the KG"  ; python3 06_lp_optimiser.py --solve --days 3 --plot
                                                python3 07_regime_events_pool.py --run
step "14/14 Knowledge graph (with billing)"   ; python3 08_knowledge_graph.py --build --queries --billing ""

if [[ "${1:-}" == "--scarcity" ]]; then
  step "extra  Scarcity scenario (regime 3)"
  python3 06_lp_optimiser.py --solve --days 2 --scale-gmax 0.55 --scale-pmax 0.62
  python3 07_regime_events_pool.py --run --scarcity
  python3 08_knowledge_graph.py --build --scarcity
fi

printf '\n\033[1mDone.\033[0m Neo4j import files: outputs/kg/\n'
printf 'Load with:  cypher-shell -f outputs/kg/load_kg.cypher\n'
