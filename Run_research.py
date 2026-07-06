"""
Run_research.py — PHASE 1: the model-selection experiment.

    python Run_research.py

Trains ALL SIX deep-learning architectures (BiLSTM, BiGRU, BiLSTM+BiGRU,
BiGRU+BiLSTM, CNN+BiLSTM+BiGRU, CNN+BiGRU+BiLSTM) on ALL 18 BPDB
substations under the strict chronological split
(train 2019-2023 | validation 2024 | test 2025-2026), ranks them by mean
VALIDATION R², and writes the winner to
forecast_outputs/global_model_selection.csv.

This is the offline research experiment of the thesis. It is long
(108 model trainings — typically a few hours on CPU) and only needs to be
re-run when the data or the training protocol changes.

Afterwards, run the operational pipeline:

    python Run_pipeline.py

which deploys ONLY the selected winner for the two case-study substations
(Kalyanpur, Dhanmondi) and drives MOLP → Priority Pool → Knowledge Graph.
"""

import runpy
import time

print("#" * 70)
print("# PHASE 1 — MODEL SELECTION EXPERIMENT")
print("# 6 architectures x 18 substations = 108 model trainings")
print("# Split: train 2019-2023 | validation 2024 | test 2025-2026")
print("# Selection: highest mean VALIDATION R2 (test reported only)")
print("# This takes a while — progress and ETA are printed per substation.")
print("#" * 70)

t0 = time.time()
runpy.run_path("load_prediction.py", run_name="__main__")

print("\n" + "#" * 70)
print(f"# PHASE 1 complete in {(time.time() - t0) / 60:.1f} min.")
print("# Winner written to forecast_outputs/global_model_selection.csv")
print("# Next step (deployment):  python Run_pipeline.py")
print("#" * 70)