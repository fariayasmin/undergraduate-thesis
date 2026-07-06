"""
run_pipeline.py — run the whole Kalyanpur + Dhanmondi low-level dataset
pipeline in order. Requires BPDB_Dhaka_City_Substations_Page3.csv next to
these scripts (or set the path in config.py).

    python run_pipeline.py

Two-level design:
  PHASE 1 (offline research, run once, NOT part of this pipeline):
      python load_prediction.py
    trains all 6 DL architectures on all 18 BPDB substations under a strict
    chronological split — train 2019-2023 | validation 2024 (early stopping
    + MODEL SELECTION) | test 2025-26 (final reporting only) — and writes
    forecast_outputs/global_model_selection.csv. The winner is determined
    experimentally by mean VALIDATION R2; the test set never influences the
    selection. NOTE: if Phase 1 was last run under the old protocol
    (train <= 2024, selection on test), it MUST be re-run before citing any
    metrics.
  PHASE 2 (this pipeline): 05b_substation_forecast.py reads the winner from
    that file and deploys ONLY the selected model for the two case-study
    substations (Kalyanpur, Dhanmondi); 06-08 consume its forecasts.
    05b stops with instructions if Phase 1 has not been run.
"""

import runpy
import time
from pathlib import Path

# ── pre-flight: Phase 1 must exist and be from the leak-free protocol ──
_sel_candidates = [Path("forecast_outputs/global_model_selection.csv"),
                   Path("global_model_selection.csv")]
_sel = next((p for p in _sel_candidates if p.exists()), None)
if _sel is None:
    raise SystemExit(
        "Run_pipeline: forecast_outputs/global_model_selection.csv not "
        "found.\nPhase 1 (model selection) has not been run. Run first:\n"
        "    python Run_research.py\nthen re-run this pipeline.")
with open(_sel, encoding="utf-8") as _f:
    if "mean_val_r2" not in _f.readline():
        raise SystemExit(
            f"Run_pipeline: {_sel} is from the OLD protocol (selection on "
            "test metrics).\nRe-run Phase 1 under the leak-free protocol "
            "first:\n    python Run_research.py\nthen re-run this "
            "pipeline.")
print(f"pre-flight OK: Phase-1 selection found at {_sel} "
      f"(leak-free protocol)")

STEPS = [
    "00_bd_calendar.py",
    "01_dhaka_weather.py",
    "03_feeder_composition.py",
    "04_load_generator.py",
    "05_validate_against_bpdb.py",
    "05b_substation_forecast.py",
    "06_molp_optimizer.py",
    "07_priority_pool.py",
    "08_knowledge_graph.py",
]

for step in STEPS:
    print(f"\n{'#'*70}\n# {step}\n{'#'*70}")
    t0 = time.time()
    runpy.run_path(step, run_name="__main__")
    print(f"[{step}] done in {time.time()-t0:.1f}s")

print("\nPipeline complete. All outputs in ./data/")