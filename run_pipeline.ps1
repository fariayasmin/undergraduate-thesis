# GenTwin-SG :: full pipeline, forecast inputs -> knowledge graph.
#
# Usage:
#   .\run_pipeline.ps1                      # base arm
#   .\run_pipeline.ps1 -Rho0                # + current-policy arm (rho = 0)
#   .\run_pipeline.ps1 -Ir                  # + individual-rationality arm
#   .\run_pipeline.ps1 -Scarcity            # + scarcity scenario
#   .\run_pipeline.ps1 -DrWindow            # + curtailment scoped to the DR window (Follow-up F3)
#   .\run_pipeline.ps1 -Rho0 -Ir -Scarcity -DrWindow  # all variants
#
# Issue C fix: the monthly flow now runs entirely against _period-tagged
# outputs (LP, regime/pooling, billing AND the knowledge graph). The
# previous version restored an UNTAGGED 3-day LP/regime solve immediately
# before building the KG, so the KG never reflected the monthly billing
# period it was built alongside. That step is gone. The untagged 3-day
# solve is kept ONLY for the dispatch figure, and nothing downstream reads it.

param(
    [switch]$Rho0,
    [switch]$Ir,
    [switch]$Scarcity,
    [switch]$DrWindow
)

$ErrorActionPreference = "Stop"

function Step($msg) {
    Write-Host ""
    Write-Host "===> $msg" -ForegroundColor Cyan
}

Step "1/12 Tariff (BERC June 2026)"
python 00_tariff_bpdb.py --all

Step "2/12 Service area / consumer mix"
python 01_service_area_osm.py --offline-seed

Step "3/12 Peak-time audit + adaptive ToU"
python 00b_peak_window.py --audit --derive --compare --plot

Step "4/12 Forecast -> kappa"
python 05c_forecast_to_kappa.py --build --table --plot

Step "5/12 Theta(t)"
python 03b_temperature_profile.py --date 2026-06-29 --plot

Step "6/12 Population synthesis"
python 03_population_synthesis.py --build --report

Step "7/12 Load generation"
python 04_load_generator.py --build --plot --xi-series 30

Step "8/12 Stage B validation"
python 05_validate_against_bpdb.py --validate --ablation

Step "9/12 LP, 3-day detail (dispatch figure ONLY - never fed to the KG)"
python 06_lp_optimiser.py --solve --days 3 --plot
python 07_regime_events_pool.py --run

Step "10/12 Monthly LP - BASE arm (_period)"
python 06_lp_optimiser.py --solve --month 30

Step "11/12 Monthly regime/pool + billing (_period)"
python 07_regime_events_pool.py --run --tag _period
python 09_monthly_billing.py --report --plot

Step "12/12 Knowledge graph, monthly (_period, with billing)"
python 08_knowledge_graph.py --build --queries --tag _period --billing

if ($Rho0) {
    Step "extra Current-policy arm: rho = 0 (_period_rho0)"
    python 06_lp_optimiser.py --solve --month 30 --rho 0
    python 07_regime_events_pool.py --run --tag _period_rho0
    python 09_monthly_billing.py --report --tag _period_rho0
    python 08_knowledge_graph.py --build --tag _period_rho0 --billing _period_rho0
}

if ($Ir) {
    Step "extra Individual-rationality arm (_period_ir)"
    python 06_lp_optimiser.py --solve --month 30 --ir
    python 07_regime_events_pool.py --run --tag _period_ir
    python 09_monthly_billing.py --report --tag _period_ir
    python 08_knowledge_graph.py --build --tag _period_ir --billing _period_ir
}

if ($Scarcity) {
    Step "extra Scarcity scenario (_period_scarcity, regime 3)"
    # 0.55/0.62 was INFEASIBLE before the Round-3 C4 fix (HiGHS status 8 on
    # 2026-07-12 - the LP correctly raised rather than truncating, but there
    # was genuinely no way to satisfy C4 without shedding load, which C4
    # could not do). C4 now includes -sum_c Z_c(t), exactly like C1, so shed
    # load also relaxes the transformer/export cap; 0.55/0.62 now solves and
    # produces real Z>0 at the VoLL price plus capacity_deficit firing on
    # every day - the emergency regime this scenario was meant to exercise.
    # See docs/thesis_changes.md, Round-3 follow-up.
    python 06_lp_optimiser.py --solve --month 30 --scale-gmax 0.55 --scale-pmax 0.62
    python 07_regime_events_pool.py --run --tag _period_scarcity
    python 09_monthly_billing.py --report --tag _period_scarcity
    python 08_knowledge_graph.py --build --tag _period_scarcity --billing _period_scarcity
}

if ($DrWindow) {
    Step "extra Curtailment scoped to the DR-activation window (_period_dr_window)"
    python 06_lp_optimiser.py --solve --month 30 --curtail-scope dr_window
    python 07_regime_events_pool.py --run --tag _period_dr_window
    python 09_monthly_billing.py --report --tag _period_dr_window
    python 08_knowledge_graph.py --build --tag _period_dr_window --billing _period_dr_window
}

Write-Host ""
Write-Host "DONE. Neo4j files are in outputs/kg/" -ForegroundColor Green
