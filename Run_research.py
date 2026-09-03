"""
Run_research.py — PHASE 1: the model-selection experiment.

    python Run_research.py

Trains all 6 architectures on all 16 retained BPDB substations under the
MULTI-HORIZON objective (K = 15, Eq. 14) and selects the single architecture
with the highest mean VALIDATION R2 across all 15 leads. The winner is written
to forecast_outputs/global_model_selection.csv and deployed by
05b_substation_forecast.py.

Why multi-horizon selection
---------------------------
The earlier sweep trained with a scalar target, Dense(1), and selected on
one-day-ahead validation R2. The deployed forecaster emits 15 leads, and a
model that is best at lead 1 need not be best over a 15-day window. Selecting
on lead-1 skill while training and deploying a 15-day head is a
selection/evaluation mismatch. This phase removes it: the loss aggregates 15
leads and so does the selection criterion.

Protocol (leak-free, unchanged)
-------------------------------
    train      2019-2023
    validation 2024        early stopping AND model selection
    test       2025-2026   reported once, never used for any decision

Cost and safety
---------------
96 trainings, roughly 5-7 hours on CPU. The sweep is RESUMABLE: it
checkpoints after every model and skips completed cells when re-run, so an
interruption costs one model rather than the whole run.

Smoke-test first (about 3 minutes):

    MH_EPOCHS=3 MH_SUBSTATIONS="Kalyanpur Dhanmondi" python research_multihorizon.py

then delete forecast_outputs/_multihorizon_partial.csv before the real run.

Next step after this completes:  python Run_pipeline.py
"""

import runpy
import time

import multihorizon as MH

print("#" * 70)
print("# PHASE 1 — MULTI-HORIZON MODEL SELECTION EXPERIMENT")
print("# 6 architectures x 16 substations = 96 model trainings")
print(f"# Horizon K = {MH.HORIZON}; loss and selection both aggregate all leads")
print("# Split: train 2019-2023 | validation 2024 | test 2025-2026")
print("# Selection: highest mean VALIDATION R2 over ALL leads")
print("#            (test years reported only, never used to decide)")
print("# Resumable: safe to interrupt and re-run.")
print("#" * 70)

t0 = time.time()
runpy.run_path("research_multihorizon.py", run_name="__main__")

print("\n" + "#" * 70)
print(f"# PHASE 1 complete in {(time.time() - t0) / 60:.1f} min.")
print("# Winner written to forecast_outputs/global_model_selection.csv")
print("# Next step (deployment):  python Run_pipeline.py")
print("#" * 70)