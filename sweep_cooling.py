"""Commit 3 — bounded sensitivity analysis of non-residential cooling.

The pre-registered band is +/- 20% around the current strengths, swept in 10%
steps.  An extended diagnostic range is also evaluated, purely to characterise
the response curve and to report what WOULD be required to close the residual;
values outside the pre-registered band are not eligible for selection.

Residential cohorts do not depend on this parameter, so they are built once and
cached; only the non-residential unit profiles and the yearly count calibration
are recomputed per sweep point.
"""
import contextlib
import importlib.util
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Runnable from anywhere: resolve the project root explicitly rather than
# relying on the working directory. Assumes this file sits in the repo root or
# one level below it (e.g. analysis/).
_HERE = Path(__file__).resolve().parent
ROOT = _HERE if (_HERE / "config.py").exists() else _HERE.parent
sys.path.insert(0, str(ROOT))

import config as C

spec = importlib.util.spec_from_file_location(
    "lg", str(ROOT / "04_load_generator.py"))
lg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lg)

# cache the residential cohorts; skip the 120 detailed households entirely
_cache, _orig = {}, lg.build_tier_cohort_profile
def _cached(tier, substation):
    key = (substation, tier)
    if key not in _cache:
        _cache[key] = _orig(tier, substation)
    return _cache[key]
lg.build_tier_cohort_profile = _cached
lg.build_detailed_households = lambda substation: None

BP = pd.read_csv(C.BPDB_CSV,
                 parse_dates=["Date"]).set_index("Date").sort_index()

PRE_REGISTERED = [0.80, 0.90, 1.00, 1.10, 1.20]
DIAGNOSTIC = [1.40, 1.70, 2.00]


def score(substation):
    agg = pd.read_csv(C.DATA_DIR / f"04_{substation.lower()}_aggregate_30min.csv",
                      parse_dates=["timestamp"]).sort_values("timestamp")
    days = pd.DatetimeIndex(agg["timestamp"].values[::C.SLOTS_PER_DAY])
    syn = pd.Series(agg["total_mw"].values.reshape(-1, C.SLOTS_PER_DAY).max(1),
                    index=days)
    real = BP[f"{substation}_Load_MW"].where(
        BP[f"{substation}_Load_MW"] <= C.MAX_PLAUSIBLE_LOAD_MW).reindex(days)
    j = pd.DataFrame({"syn": syn, "bp": real}).dropna()
    mape = 100 * np.mean(np.abs(j.syn - j.bp) / j.bp)
    corr = j.syn.corr(j.bp)
    bias = (j.syn - j.bp).mean()
    m = j.index.year <= 2025
    w = m & np.isin(j.index.month, [12, 1, 2])
    s = m & np.isin(j.index.month, [4, 5, 6])
    return mape, corr, bias, j.syn[s].mean() / j.syn[w].mean()


def bpdb_swing(substation):
    real = BP[f"{substation}_Load_MW"].where(
        BP[f"{substation}_Load_MW"] <= C.MAX_PLAUSIBLE_LOAD_MW).dropna()
    real = real[(real.index >= C.START_DATE) & (real.index <= C.END_DATE)]
    m = real.index.year <= 2025
    w = m & np.isin(real.index.month, [12, 1, 2])
    s = m & np.isin(real.index.month, [4, 5, 6])
    return real[s].mean() / real[w].mean()


rows = []
for scale in PRE_REGISTERED + DIAGNOSTIC:
    C.NONRES_COOLING_SCALE = scale
    for sub in C.SUBSTATIONS:
        with contextlib.redirect_stdout(io.StringIO()):
            lg.build_substation(sub)
        mape, corr, bias, sw = score(sub)
        rows.append(dict(scale=scale, substation=sub, mape=mape, corr=corr,
                         bias_mw=bias, swing=sw,
                         band="pre-registered" if scale in PRE_REGISTERED
                              else "diagnostic-only"))
    print(f"  scale {scale:.2f} done", flush=True)

df = pd.DataFrame(rows)
df.to_csv(_HERE / "cooling_sensitivity.csv", index=False)

for sub, g in df.groupby("substation", sort=False):
    tgt = bpdb_swing(sub)
    print(f"\n{'='*78}\n{sub}   (BPDB swing {tgt:.3f})\n{'='*78}")
    print(f"{'scale':>7} {'strength':>9} | {'swing':>7} {'gap':>7} | "
          f"{'MAPE%':>7} {'corr':>7} {'bias MW':>8} |  band")
    for _, r in g.sort_values("scale").iterrows():
        print(f"{r.scale:>7.2f} {0.70*r.scale:>9.2f} | {r.swing:>7.3f} "
              f"{r.swing-tgt:>+7.3f} | {r.mape:>7.2f} {r["corr"]:>7.3f} "
              f"{r.bias_mw:>+8.2f} |  {r.band}")