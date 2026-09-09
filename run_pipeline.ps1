$ErrorActionPreference = "Stop"

function Step($msg) {
    Write-Host ""
    Write-Host "===> $msg" -ForegroundColor Cyan
}


Step "1/14 Tariff (BERC June 2026)"
python 00_tariff_bpdb.py --all

Step "2/14 Service area / consumer mix"
python 01_service_area_osm.py --offline-seed

Step "3/14 Peak-time audit + adaptive ToU"
python 00b_peak_window.py --audit --derive --compare --plot

Step "4/14 Forecast -> kappa"
python 05c_forecast_to_kappa.py --build --table --plot

Step "5/14 Theta(t)"
python 03b_temperature_profile.py --date 2026-06-29 --plot

Step "6/14 Population synthesis"
python 03_population_synthesis.py --build --report

Step "7/14 Load generation"
python 04_load_generator.py --build --plot --xi-series 30

Step "8/14 Stage B validation"
python 05_validate_against_bpdb.py --validate --ablation

Step "9/14 LP 3-day dispatch"
python 06_lp_optimiser.py --solve --days 3 --plot

Step "10/14 Regime events priority pool"
python 07_regime_events_pool.py --run

Step "11/14 Monthly LP"
python 06_lp_optimiser.py --solve --month 30

Step "12/14 Monthly billing"
python 09_monthly_billing.py --report --plot

Step "13/14 Restore 3-day KG data"
python 06_lp_optimiser.py --solve --days 3 --plot
python 07_regime_events_pool.py --run

Step "14/14 Knowledge Graph"

python 08_knowledge_graph.py --build --queries --billing ""

Write-Host ""
Write-Host "DONE. Neo4j files are in outputs/kg/" -ForegroundColor Green