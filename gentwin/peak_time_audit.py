"""
GenTwin-SG :: audit of the BPDB `*_Peak_Time` field
===================================================

Why this exists
---------------
Stage B's T2 test asks whether the synthetic aggregate peaks at the hour BPDB
says the substation peaks. That test is only meaningful if `*_Peak_Time` is a
MEASUREMENT. This module asks whether it is.

The question arose because Dhanmondi records a 13:00 peak on 64% of days, which
is a suspiciously concentrated distribution for a physical quantity. It matters
in two places:

  1. T2 at Dhanmondi. If 13:00 is a logging convention, the model's evening
     peak may be right and the target wrong - and tuning the population to
     reproduce 13:00 would be fitting to an artefact.
  2. T^pk_i for the adaptive local ToU, which is derived from this same field.

Five independent signals
------------------------
Each is a property a genuine peak-timing distribution should have, and a
clerical convention should not.

  S1  HALF-HOUR SHARE. The strongest discriminator. A peak read off a genuine
      half-hourly series should land on :30 about half the time. A value near
      zero means the time is being recorded to the hour.

  S2  ENTROPY of the slot distribution. Physical peak timing varies with
      weather, day of week and season, so it spreads. A convention concentrates.

  S3  SEASONAL MODE SHIFT. Cooling-driven peaks move between monsoon and
      winter. A modal slot that is identical in every season, and MORE
      concentrated in the season with least load, is not thermal behaviour.

  S4  WEEKEND / HOLIDAY DROP. If the modal share falls sharply when offices are
      closed, the reading is likelier tied to staff presence than to load.

  S5  TREND. A rising modal share over years, especially while load falls, is
      the signature of consolidating reporting practice, not of physics.

Verdict
-------
Flags are counted, not weighted, because none of the five is decisive alone and
the point is convergent evidence:

    0-1 flags   RELIABLE          usable as a T2 target
    2-3 flags   QUESTIONABLE      usable with the caveat stated
    4-5 flags   LIKELY_CONVENTION not a safe T2 target

The audit reads only the peak-time field and the calendar. No model, residual
or fitted quantity enters it, so it cannot be tuned to a convenient answer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as cfg

# Thresholds. Chosen against the fleet distribution, which separates cleanly
# into two groups, and stated here rather than inline so they are arguable.
HALF_HOUR_SHARE_MIN = 0.10     # S1: below this, the field is hour-only
ENTROPY_MIN_BITS = 2.50        # S2
MODAL_SHARE_MAX = 0.50         # S2 companion
WEEKEND_DROP_MAX = 0.10        # S4: absolute drop in modal share
TREND_MAX = 0.10               # S5: rise in modal share, first to last year


def _slots(bpdb: pd.DataFrame, substation: str) -> pd.DataFrame:
    spec = cfg.SUBSTATIONS.get(substation)
    lc = spec["load_col"] if spec else f"{substation}_Load_MW"
    tc = spec["peak_time_col"] if spec else f"{substation}_Peak_Time"
    cols = [c for c in ["Date", lc, tc, "Season", "Is_Weekend", "Holiday_type"]
            if c in bpdb.columns]
    x = bpdb[cols].dropna(subset=[lc, tc]).copy()
    x[tc] = x[tc].astype(str)
    x = x[x[tc].str.match(r"^\d{1,2}:\d{2}$")]
    if x.empty:
        return x
    hm = x[tc].str.split(":", expand=True).astype(int)
    x = x[hm[1].isin([0, 30])]
    hm = hm[hm[1].isin([0, 30])]
    x["slot"] = (hm[0] * 2 + (hm[1] >= 30).astype(int)).values
    x["on_half_hour"] = (hm[1] == 30).values
    x["load_mw"] = x[lc].values
    x["artefact"] = x["slot"].isin(cfg.PEAK_TIME_ARTEFACT_SLOTS)
    return x


def audit(bpdb: pd.DataFrame, substation: str) -> dict:
    x = _slots(bpdb, substation)
    if len(x) < 200:
        return {"substation": substation, "verdict": "INSUFFICIENT_DATA",
                "n": len(x), "flags": [], "n_flags": 0}

    artefact_share = float(x["artefact"].mean())
    clean = x[~x["artefact"]]
    if clean.empty:
        return {"substation": substation, "verdict": "INSUFFICIENT_DATA",
                "n": 0, "flags": [], "n_flags": 0}

    p = clean["slot"].value_counts(normalize=True)
    modal_slot = int(clean["slot"].mode().iloc[0])
    modal_share = float(p.max())
    entropy = float(-(p * np.log2(p)).sum())
    half_share = float(clean["on_half_hour"].mean())

    # S3 seasonal
    seasonal = {}
    for s, g in clean.groupby("Season") if "Season" in clean else []:
        if len(g) < 50:
            continue
        seasonal[str(s)] = {
            "n": int(len(g)), "mode_slot": int(g["slot"].mode().iloc[0]),
            "mode_clock": cfg.slot_to_clock(int(g["slot"].mode().iloc[0])),
            "modal_share_global_mode": float((g["slot"] == modal_slot).mean()),
            "median_load_mw": float(g["load_mw"].median()),
        }
    modes = {v["mode_slot"] for v in seasonal.values()}
    no_seasonal_shift = len(modes) <= 1 and len(seasonal) >= 3
    # The sharper version: is the modal share HIGHEST in the lightest-load season?
    inverted = False
    if len(seasonal) >= 3:
        light = min(seasonal.values(), key=lambda v: v["median_load_mw"])
        heavy = max(seasonal.values(), key=lambda v: v["median_load_mw"])
        inverted = (light["modal_share_global_mode"]
                    > heavy["modal_share_global_mode"] + 0.05)

    # S4 weekend / holiday
    wk = clean[clean["Is_Weekend"] == 1] if "Is_Weekend" in clean else clean.iloc[:0]
    hol = (clean[clean["Holiday_type"] != "No Holiday"]
           if "Holiday_type" in clean else clean.iloc[:0])
    wk_share = float((wk["slot"] == modal_slot).mean()) if len(wk) > 30 else np.nan
    hol_share = float((hol["slot"] == modal_slot).mean()) if len(hol) > 30 else np.nan
    weekend_drop = (modal_share - wk_share) if np.isfinite(wk_share) else 0.0

    # S5 trend
    by_year = clean.groupby(clean["Date"].dt.year).apply(
        lambda g: pd.Series({"n": len(g),
                             "modal_share": float((g["slot"] == modal_slot).mean()),
                             "median_load": float(g["load_mw"].median())}))
    by_year = by_year[by_year["n"] >= 100]
    trend = (float(by_year["modal_share"].iloc[-1] - by_year["modal_share"].iloc[0])
             if len(by_year) >= 3 else 0.0)

    flags = []
    if half_share < HALF_HOUR_SHARE_MIN:
        flags.append(f"S1 only {half_share:.1%} of readings fall on :30 "
                     f"(expect ~50% from a genuine half-hourly series)")
    if entropy < ENTROPY_MIN_BITS or modal_share > MODAL_SHARE_MAX:
        flags.append(f"S2 distribution over-concentrated: entropy {entropy:.2f} bits, "
                     f"modal share {modal_share:.1%} at "
                     f"{cfg.slot_to_clock(modal_slot)}")
    if no_seasonal_shift:
        flags.append(f"S3 modal slot identical in every season "
                     f"({cfg.slot_to_clock(modal_slot)}); cooling-driven peaks "
                     f"should move")
    if inverted:
        flags.append("S3b modal share is HIGHER in the lightest-load season than "
                     "the heaviest - the opposite of thermal behaviour")
    if weekend_drop > WEEKEND_DROP_MAX:
        flags.append(f"S4 modal share falls {weekend_drop:.1%} at weekends "
                     f"({modal_share:.1%} -> {wk_share:.1%})")
    if trend > TREND_MAX:
        flags.append(f"S5 modal share rose {trend:+.1%} from "
                     f"{int(by_year.index[0])} to {int(by_year.index[-1])}")

    n = len(flags)
    verdict = ("RELIABLE" if n <= 1 else
               "QUESTIONABLE" if n <= 3 else "LIKELY_CONVENTION")

    return {
        "substation": substation, "n": int(len(clean)),
        "artefact_share": artefact_share,
        "modal_slot": modal_slot, "modal_clock": cfg.slot_to_clock(modal_slot),
        "modal_share": modal_share, "entropy_bits": entropy,
        "distinct_slots": int(clean["slot"].nunique()),
        "half_hour_share": half_share,
        "weekend_modal_share": wk_share, "holiday_modal_share": hol_share,
        "weekend_drop": weekend_drop, "trend_first_to_last_year": trend,
        "seasonal": seasonal, "no_seasonal_shift": bool(no_seasonal_shift),
        "inverted_seasonality": bool(inverted),
        "flags": flags, "n_flags": n, "verdict": verdict,
        "t2_target_usable": verdict != "LIKELY_CONVENTION",
    }


def audit_fleet(bpdb: pd.DataFrame, substations: list | None = None) -> pd.DataFrame:
    """
    Run the audit across the whole fleet. Fleet context is what turns a single
    odd distribution into evidence: if every substation looked like Dhanmondi
    the field would simply be coarse, but they do not.
    """
    if substations is None:
        substations = sorted({c[:-8] for c in bpdb.columns if c.endswith("_Load_MW")})
    rows = []
    for s in substations:
        try:
            a = audit(bpdb, s)
        except Exception:                                          # noqa: BLE001
            continue
        if a.get("verdict") == "INSUFFICIENT_DATA":
            continue
        rows.append({k: a[k] for k in
                     ("substation", "n", "modal_clock", "modal_share",
                      "entropy_bits", "distinct_slots", "half_hour_share",
                      "artefact_share", "n_flags", "verdict")})
    return pd.DataFrame(rows).sort_values("half_hour_share").reset_index(drop=True)
