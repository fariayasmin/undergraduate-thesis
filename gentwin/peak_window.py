"""
GenTwin-SG :: adaptive local ToU - deriving T^pk_i
==================================================

What this module decides
------------------------
The ACTIVATION WINDOW for time-of-use pricing at each substation, T^pk_i.
It does not decide the price ratios: mu^pk = 1.20 and mu^off = 0.90 come from
the BERC order of 03 June 2026 and are transcribed in `tariff.py`. The split
of responsibility is deliberate and must be stated wherever this is reported:

    BERC          sets the peak / off-peak price RATIOS (regulatory)
    GenTwin-SG    determines WHEN the peak ratio is active, per substation,
                  from that substation's own congestion history (adaptive)

This is not a replacement tariff. It is a congestion-aware activation schedule
laid over the regulator's own multipliers.

What data actually exists
-------------------------
An honest constraint, stated up front because it shapes every choice below:
**the BPDB dataset contains no intra-day load profile.** Each row carries one
daily peak (`*_Load_MW`) and the clock time at which it occurred
(`*_Peak_Time`). There is no half-hourly series to average.

So "average historical load profile" is not available to us. What IS available
is ~2,300 daily observations of WHEN the peak fell, each paired with HOW LARGE
it was. That supports a proper statistical estimator of the peak-timing
distribution, which is what this module builds. Individual daily peak times are
noisy; the smoothed, load-weighted distribution over thousands of days is not.

The estimator
-------------
Five steps, each with a reason:

1. FILTER. Keep rows with a parseable HH:MM peak time. Drop the artefact slots
   (config.PEAK_TIME_ARTEFACT_SLOTS): 13.1% of Kalyanpur rows report a peak at
   exactly 00:00 against 1.0% at Dhanmondi, which is a not-recorded default
   rather than a midnight peak at a Dhaka distribution substation.

2. SELECT STRESS DAYS. Keep days whose peak is at or above the
   `stress_quantile` (default 0.95) of that substation's peak distribution.
   Rationale: the window exists to price CONGESTION. The right question is not
   "when does this substation usually peak" but "when does it peak on the days
   that actually stress it". These are the days on which s_{i,d} = 1 in
   Eq. (15) and on which demand response has to work.

3. LOAD-WEIGHTED CIRCULAR KDE. Each retained day contributes a Gaussian kernel
   (bandwidth 2 slots = 1 hour) centred on its peak slot, weighted by its peak
   MW, wrapped circularly so 23:30 and 00:00 are neighbours. Smoothing is what
   converts noisy individual peaks into a stable density; the circular
   treatment is what stops an evening-peaking substation being cut in half at
   midnight.

4. DENSITY THRESHOLD, then CONTIGUITY. Retain slots whose density is at least
   `threshold_frac` (default 0.50) of the maximum, then take the single
   longest contiguous circular run among them. The threshold adapts the window
   width to how sharp the distribution is - a substation with a tight peak gets
   a tight window, which a fixed-mass rule cannot do. Contiguity is imposed
   because a ToU activation window with holes in it is not a usable tariff;
   any secondary lobe that is dropped is REPORTED, never silently discarded.

5. BOOTSTRAP. 500 resamples of the stress-day set, re-running steps 3-4, giving
   the IQR of the window's start, end and width. A window whose bootstrap IQR
   is wide is not a window you should price against, and the report says so.

Result on the two study substations
-----------------------------------
    Dhanmondi  12:00-14:30   bootstrap IQR of start and end both degenerate
                             (start=12:00, end=14:30 in >75% of resamples)
    Kalyanpur  18:00-21:00   start stable at 18:00; a secondary lobe at
                             22:00-24:00 is dropped by the contiguity rule and
                             reported (see the 23:00 note in the output)

Both are stable enough to price against. Dhanmondi's window has ZERO slot
overlap with the national 17:00-23:00 ToU window, which is the empirical
justification for this whole mechanism.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as cfg

DEFAULT_BANDWIDTH_SLOTS = 2.0      # 1 hour
DEFAULT_THRESHOLD_FRAC = 0.50
DEFAULT_STRESS_QUANTILE = 0.95
DEFAULT_BOOTSTRAP = 500
MIN_STRESS_DAYS = 40               # below this, widen the quantile


@dataclass
class PeakWindow:
    substation: str
    slots: list                    # T^pk_i, sorted slot indices
    label: str
    method: str
    n_days_total: int
    n_days_stress: int
    stress_quantile: float
    stress_threshold_mw: float
    bandwidth_slots: float
    threshold_frac: float
    density: list                  # smoothed density, 48 values
    mass_in_window: float
    secondary_lobes: list = field(default_factory=list)
    bootstrap: dict = field(default_factory=dict)
    artefact_slots_dropped: list = field(default_factory=list)
    artefact_share: float = 0.0
    national_overlap_slots: int = 0
    warnings: list = field(default_factory=list)

    @property
    def is_stable(self) -> bool:
        bs = self.bootstrap
        return bool(bs) and bs.get("width_iqr", 99) <= 2 and bs.get("start_iqr", 99) <= 2


# =============================================================================
# Core statistics
# =============================================================================

def _circular_distance(a: np.ndarray, b: int, n: int = cfg.SLOTS_PER_DAY):
    d = np.abs(a - b)
    return np.minimum(d, n - d)


def circular_kde(peak_slots: np.ndarray, weights: np.ndarray,
                 bandwidth: float = DEFAULT_BANDWIDTH_SLOTS) -> np.ndarray:
    """Load-weighted Gaussian KDE on the circle of 48 slots."""
    idx = np.arange(cfg.SLOTS_PER_DAY)
    dens = np.zeros(cfg.SLOTS_PER_DAY)
    for s, w in zip(peak_slots, weights):
        dens += w * np.exp(-0.5 * (_circular_distance(idx, int(s)) / bandwidth) ** 2)
    total = dens.sum()
    return dens / total if total > 0 else dens


def _circular_runs(slots: list) -> list:
    """Split a slot set into maximal contiguous circular runs."""
    if not slots:
        return []
    s = sorted(slots)
    if len(s) == cfg.SLOTS_PER_DAY:
        return [s]
    runs, cur = [], [s[0]]
    for a, b in zip(s, s[1:]):
        if b == a + 1:
            cur.append(b)
        else:
            runs.append(cur)
            cur = [b]
    runs.append(cur)
    # join a run ending at 47 with one starting at 0
    if len(runs) > 1 and runs[0][0] == 0 and runs[-1][-1] == cfg.SLOTS_PER_DAY - 1:
        runs[0] = runs[-1] + runs[0]
        runs.pop()
    return runs


def window_from_density(dens: np.ndarray,
                        threshold_frac: float = DEFAULT_THRESHOLD_FRAC
                        ) -> tuple[list, list]:
    """
    Slots above `threshold_frac` x max density, reduced to the single run
    carrying the most probability mass. Returns (window, dropped_lobes).
    """
    above = np.flatnonzero(dens >= threshold_frac * dens.max()).tolist()
    runs = _circular_runs(above)
    if not runs:
        return [], []
    masses = [float(dens[r].sum()) for r in runs]
    best = int(np.argmax(masses))
    dropped = [{"slots": r, "mass": m}
               for k, (r, m) in enumerate(zip(runs, masses)) if k != best]
    return runs[best], dropped


def format_window(slots: list) -> str:
    """Slot list -> 'HH:MM-HH:MM' (or a comma list for a split set)."""
    if not slots:
        return "-"
    out = []
    for run in _circular_runs(slots):
        a, b = run[0], (run[-1] + 1) % cfg.SLOTS_PER_DAY
        end = "24:00" if b == 0 else cfg.slot_to_clock(b)
        out.append(f"{cfg.slot_to_clock(a)}-{end}")
    return ", ".join(out)


# =============================================================================
# Data preparation
# =============================================================================

def peak_observations(df: pd.DataFrame, substation: str) -> tuple[pd.DataFrame, dict]:
    """
    Daily (date, peak_slot, peak_mw) for one substation, artefact slots removed.
    Returns the frame and a diagnostics dict.
    """
    spec = cfg.SUBSTATIONS[substation]
    lc, tc = spec["load_col"], spec["peak_time_col"]
    x = df[["Date", lc, tc]].dropna().copy()
    x[tc] = x[tc].astype(str)
    n_raw = len(x)

    ok = x[tc].str.match(r"^\d{1,2}:\d{2}$")
    x = x[ok].copy()
    n_parsed = len(x)

    hm = x[tc].str.split(":", expand=True).astype(int)
    bad_minute = ~hm[1].isin([0, 30])
    x = x[~bad_minute].copy()
    hm = hm[~bad_minute]
    x["peak_slot"] = (hm[0] * 2 + (hm[1] >= 30).astype(int)).values
    x["peak_mw"] = x[lc].values

    artefact = x["peak_slot"].isin(cfg.PEAK_TIME_ARTEFACT_SLOTS)
    n_artefact = int(artefact.sum())
    x = x[~artefact].reset_index(drop=True)

    diag = {
        "n_raw_rows": n_raw,
        "n_unparseable_time": n_raw - n_parsed,
        "n_off_grid_minute": int(bad_minute.sum()),
        "n_artefact_dropped": n_artefact,
        "artefact_share": n_artefact / n_parsed if n_parsed else 0.0,
        "artefact_slots": list(cfg.PEAK_TIME_ARTEFACT_SLOTS),
        "n_usable": len(x),
    }
    return x[["Date", "peak_slot", "peak_mw"]], diag


# =============================================================================
# Derivation
# =============================================================================

def derive_peak_window(df: pd.DataFrame, substation: str,
                       stress_quantile: float = DEFAULT_STRESS_QUANTILE,
                       bandwidth: float = DEFAULT_BANDWIDTH_SLOTS,
                       threshold_frac: float = DEFAULT_THRESHOLD_FRAC,
                       n_bootstrap: int = DEFAULT_BOOTSTRAP,
                       seed: int = 42) -> PeakWindow:
    obs, diag = peak_observations(df, substation)
    if obs.empty:
        raise ValueError(f"{substation}: no usable peak-time observations")

    q = stress_quantile
    thr = float(obs["peak_mw"].quantile(q))
    stress = obs[obs["peak_mw"] >= thr]
    warnings: list[str] = []
    while len(stress) < MIN_STRESS_DAYS and q > 0.50:
        q = round(q - 0.05, 2)
        thr = float(obs["peak_mw"].quantile(q))
        stress = obs[obs["peak_mw"] >= thr]
        warnings.append(
            f"stress quantile relaxed to {q:.2f} to reach {MIN_STRESS_DAYS} days")

    dens = circular_kde(stress["peak_slot"].values, stress["peak_mw"].values, bandwidth)
    window, dropped = window_from_density(dens, threshold_frac)
    mass = float(dens[window].sum()) if window else 0.0

    # Bootstrap the window over resamples of the stress-day set.
    rng = np.random.default_rng(seed)
    starts, ends, widths = [], [], []
    for _ in range(n_bootstrap):
        take = rng.integers(0, len(stress), len(stress))
        bs = stress.iloc[take]
        w, _ = window_from_density(
            circular_kde(bs["peak_slot"].values, bs["peak_mw"].values, bandwidth),
            threshold_frac)
        if w:
            starts.append(w[0])
            ends.append(w[-1])
            widths.append(len(w))

    def _q(a, p):
        return float(np.percentile(a, p)) if a else float("nan")

    bootstrap = {
        "n": len(widths),
        "start_median": _q(starts, 50), "start_q1": _q(starts, 25),
        "start_q3": _q(starts, 75), "start_iqr": _q(starts, 75) - _q(starts, 25),
        "end_median": _q(ends, 50), "end_q1": _q(ends, 25),
        "end_q3": _q(ends, 75), "end_iqr": _q(ends, 75) - _q(ends, 25),
        "width_median": _q(widths, 50), "width_q1": _q(widths, 25),
        "width_q3": _q(widths, 75), "width_iqr": _q(widths, 75) - _q(widths, 25),
    }

    if diag["artefact_share"] > 0.05:
        warnings.append(
            f"{diag['artefact_share']:.1%} of parsed rows carried an artefact "
            f"peak time and were dropped; peak timing at this substation is "
            f"less well observed than the row count suggests")
    if dropped:
        for lobe in dropped:
            warnings.append(
                f"secondary density lobe at {format_window(lobe['slots'])} "
                f"(mass {lobe['mass']:.3f}) dropped by the contiguity rule")
    if bootstrap["width_iqr"] > 2:
        warnings.append(
            f"window width is unstable under resampling "
            f"(IQR {bootstrap['width_iqr']:.0f} slots); treat T^pk as indicative")

    from . import tariff as T
    overlap = len(set(window) & T.OFFICIAL_TOU_PEAK)

    return PeakWindow(
        substation=substation, slots=window,
        label=f"{format_window(window)} (adaptive local, stress-day derived)",
        method=(f"load-weighted circular KDE (bw={bandwidth} slots) on days "
                f"above the p{int(q * 100)} peak, density threshold "
                f"{threshold_frac:.2f} x max, longest contiguous run"),
        n_days_total=len(obs), n_days_stress=len(stress),
        stress_quantile=q, stress_threshold_mw=thr,
        bandwidth_slots=bandwidth, threshold_frac=threshold_frac,
        density=[float(v) for v in dens], mass_in_window=mass,
        secondary_lobes=dropped, bootstrap=bootstrap,
        artefact_slots_dropped=list(cfg.PEAK_TIME_ARTEFACT_SLOTS),
        artefact_share=diag["artefact_share"],
        national_overlap_slots=overlap, warnings=warnings,
    )


def sensitivity(df: pd.DataFrame, substation: str) -> pd.DataFrame:
    """
    The window under alternative defensible estimator settings. If the choices
    in `derive_peak_window` were driving the answer rather than the data, this
    table is where it would show.
    """
    rows = []
    for q in (0.90, 0.95, 0.99):
        for frac in (0.35, 0.50, 0.65):
            for bw in (1.5, 2.0, 3.0):
                try:
                    pw = derive_peak_window(df, substation, stress_quantile=q,
                                            threshold_frac=frac, bandwidth=bw,
                                            n_bootstrap=0)
                    rows.append({"stress_quantile": q, "threshold_frac": frac,
                                 "bandwidth": bw, "n_stress_days": pw.n_days_stress,
                                 "window": format_window(pw.slots),
                                 "n_slots": len(pw.slots),
                                 "mass": round(pw.mass_in_window, 3)})
                except ValueError:
                    continue
    return pd.DataFrame(rows)


def unweighted_comparison(df: pd.DataFrame, substation: str) -> dict:
    """
    The same window under three other populations, for the documentation:
    all days unweighted, all days load-weighted, and recent days only. Reported
    so that the choice of the stress-day population is visible and arguable
    rather than buried.
    """
    obs, _ = peak_observations(df, substation)
    out = {}
    variants = {
        "all_days_unweighted": (obs, np.ones(len(obs))),
        "all_days_load_weighted": (obs, obs["peak_mw"].values),
        "recent_2024plus": (obs[obs["Date"].dt.year >= 2024], None),
    }
    for name, (sel, w) in variants.items():
        if len(sel) < 10:
            continue
        weights = w if w is not None else sel["peak_mw"].values
        win, _ = window_from_density(
            circular_kde(sel["peak_slot"].values, weights))
        out[name] = {"n": len(sel), "window": format_window(win),
                     "slots": win}
    return out


# =============================================================================
# Cache
# =============================================================================

def cache_path(substation: str) -> Path:
    return cfg.CACHE_DIR / f"peak_window_{substation.lower().replace(' ', '_')}.json"


def write_cache(pw: PeakWindow, extra: dict | None = None) -> Path:
    rec = asdict(pw)
    rec["schema_version"] = 1
    rec["mu_peak"] = cfg.MU_PEAK
    rec["mu_off"] = cfg.MU_OFF
    rec["mu_source"] = "BERC order 03 June 2026 (LT-class peak/flat ratio)"
    rec["window_source"] = "GenTwin-SG adaptive local derivation (this module)"
    if extra:
        rec.update(extra)
    p = cache_path(pw.substation)
    p.write_text(json.dumps(rec, indent=1, default=str), encoding="utf-8")
    return p


def read_cache(substation: str) -> dict | None:
    p = cache_path(substation)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def get_peak_slots(substation: str) -> tuple:
    """
    T^pk_i for downstream stages. Prefers the derived cache; falls back to the
    literal in config with a warning, so the pipeline still runs before
    00b_peak_window.py has been executed but never does so silently.
    """
    rec = read_cache(substation)
    if rec and rec.get("slots"):
        return tuple(rec["slots"])
    import warnings as _w
    _w.warn(
        f"No derived peak window for {substation}; falling back to the config "
        f"literal. Run: python 00b_peak_window.py --derive", stacklevel=2)
    return tuple(cfg.SUBSTATIONS[substation]["t_pk_slots"])
