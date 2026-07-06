"""
regenerate_selection.py — rebuild every post-training Phase-1 artefact from
the already-saved forecast_outputs/metrics_summary.csv, WITHOUT retraining.

Use after a crash in the plotting/reporting stage (e.g. the matplotlib
boxplot incompatibility): the 108 model trainings are not repeated — this
finishes the last ~2 seconds of the experiment.

    python regenerate_selection.py

Regenerates:
  forecast_outputs/global_model_selection.csv   (re-written, identical)
  forecast_outputs/global_model_selection.png   (selection table figure)
  forecast_outputs/model_r2_boxplot.png         (the plot that crashed)
  forecast_outputs/best_model_per_substation.csv
  forecast_outputs/global_best_model.txt        (justification text)
and prints the SINGLE MODEL SELECTED block.
"""

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")   # quiet TF import

import pandas as pd
import load_prediction as LP

OUT = LP.OUTPUT_DIR
summary = pd.read_csv(OUT / "metrics_summary.csv")
assert "val_r2" in summary.columns, (
    "metrics_summary.csv is from the old protocol (no val_r2 column) — "
    "the full experiment must be re-run: python Run_research.py")

print(f"Loaded {len(summary)} model×substation rows "
      f"({summary['substation'].nunique()} substations, "
      f"{summary['model'].nunique()} models) — no retraining needed.\n")

best_model, sel = LP.select_global_best_model(summary)
sel.to_csv(OUT / "global_model_selection.csv", index=False)
LP.plot_global_selection_table(sel, best_model, OUT)
LP.plot_model_r2_boxplot(summary, best_model, OUT)
LP.report_best_model_per_substation(summary, best_model, OUT)

best_row = sel[sel["model"] == best_model].iloc[0]
justification = (
    f"SINGLE MODEL SELECTED FOR THE PAPER: {best_model}\n"
    f"{'-'*60}\n"
    f"Selection criterion : highest MEAN VALIDATION R^2 (year "
    f"{LP.VAL_YEAR}) across all "
    f"{summary['substation'].nunique()} substations\n"
    f"Mean Val R^2        : {best_row['mean_val_r2']:.4f}\n"
    f"Worst-case Val R^2  : {best_row['min_val_r2']:.4f}\n"
    f"Mean rank (val)     : {best_row['mean_rank']:.2f}  (1 = best)\n"
    f"Wins (rank-1, val)  : {int(best_row['wins'])} / "
    f"{summary['substation'].nunique()} substations\n"
    f"Reported ONCE on the untouched test set (2025-26):\n"
    f"  Mean Test R^2     : {best_row['mean_test_r2']:.4f}\n"
    f"  Mean Test RMSE    : {best_row['mean_test_rmse']:.2f} MW   "
    f"Mean Test MAE: {best_row['mean_test_mae']:.2f} MW\n"
    f"{'-'*60}\n"
    f"This model is applied uniformly to every substation in the paper.\n"
    f"Per-substation metrics: best_model_per_substation.csv\n"
)
(OUT / "global_best_model.txt").write_text(justification, encoding="utf-8")
print(justification)
print(f"All Phase-1 artefacts regenerated in {OUT}.")
print("Next step (deployment):  python Run_pipeline.py")