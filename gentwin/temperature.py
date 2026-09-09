"""
GenTwin-SG :: Stage A/B - diurnal temperature Theta(t)
=====================================================

Why this module exists
----------------------
Equation (2) needs a thermal response f_a(Theta(t)) at every one of the 48
slots. The BPDB dataset carries only DAILY weather: Temp_Max_C, Temp_Min_C,
Temp_Mean_C. There is no sub-daily temperature series anywhere in the inputs.

So Theta(t) is RECONSTRUCTED, not observed, and every figure that depends on it
inherits that status. It matters more than it might appear: the air-conditioning
response is what produces Dhanmondi's afternoon peak, so the shape of Theta(t)
substantially determines whether the synthetic aggregate peaks at the right
hour. Getting it wrong would not merely add noise, it would move the peak.

Method
------
The Parton-Logan (1981) two-piece form, which is the standard reconstruction in
agrometeorology and crop modelling and is used here for the same reason it is
used there - it needs only Tmin, Tmax and day length:

  DAY   (sunrise -> sunset)   a sine arc rising from Tmin at sunrise to Tmax at
                              a lag after solar noon. The lag is real: peak air
                              temperature follows peak insolation by ~2-3 h.

  NIGHT (sunset -> sunrise)   exponential decay from the sunset temperature
                              towards the next morning's Tmin.

Then an affine correction rescales the curve so its 48-slot mean reproduces the
recorded Temp_Mean_C exactly. Without it the reconstruction's mean floats a
degree or so away from the observation, and Temp_Mean_C is the one intra-day
aggregate the dataset does record - it should be honoured.

Dhaka-specific constants are in config-free module scope here because they are
astronomical, not tunable: latitude 23.81 N gives a day length of roughly
13.4 h at the June solstice and 10.6 h at the December solstice.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as cfg

DHAKA_LATITUDE_DEG = 23.8103
TEMP_LAG_HOURS = 2.0          # peak air temperature after solar noon
NIGHT_DECAY_COEF = 2.2        # Parton-Logan nocturnal decay constant
SUNRISE_OFFSET_H = 0.0        # Tmin assumed at sunrise


def day_length_hours(day_of_year: int, latitude_deg: float = DHAKA_LATITUDE_DEG) -> float:
    """
    Astronomical day length by the standard declination formula. Used to place
    sunrise and sunset, which anchor both pieces of the curve.
    """
    lat = np.radians(latitude_deg)
    decl = np.radians(23.45) * np.sin(2 * np.pi * (284 + day_of_year) / 365.0)
    cos_h = -np.tan(lat) * np.tan(decl)
    cos_h = float(np.clip(cos_h, -1.0, 1.0))
    return 2.0 * np.degrees(np.arccos(cos_h)) / 15.0


def sun_times(day_of_year: int) -> tuple[float, float]:
    """(sunrise, sunset) in decimal hours, solar noon fixed at 12:00 local."""
    dl = day_length_hours(day_of_year)
    return 12.0 - dl / 2.0, 12.0 + dl / 2.0


def diurnal_profile(t_min: float, t_max: float, day_of_year: int,
                    t_mean: float | None = None) -> np.ndarray:
    """
    Theta(t) over the 48 half-hour slots, in degrees C.

    `t_mean` is optional; when given, the curve is affinely rescaled so its
    slot-mean matches it exactly while preserving the range (t_max - t_min).
    """
    sunrise, sunset = sun_times(day_of_year)
    t_peak = min(sunset - 0.5, 12.0 + TEMP_LAG_HOURS)
    day_len = sunset - sunrise
    night_len = 24.0 - day_len

    # Slot centres, so a slot's temperature represents the whole half hour.
    hours = np.arange(cfg.SLOTS_PER_DAY) * 0.5 + 0.25
    theta = np.empty(cfg.SLOTS_PER_DAY, dtype=float)

    # Sunset temperature, needed as the starting point of the night decay.
    frac_ss = (sunset - sunrise) / (t_peak - sunrise) if t_peak > sunrise else 1.0
    t_sunset = t_min + (t_max - t_min) * np.sin(np.pi / 2 * min(frac_ss, 2.0))

    for i, h in enumerate(hours):
        if sunrise <= h <= sunset:
            # Daytime sine arc: Tmin at sunrise, Tmax at t_peak.
            frac = (h - sunrise) / max(t_peak - sunrise, 1e-6)
            theta[i] = t_min + (t_max - t_min) * np.sin(np.pi / 2 * min(frac, 2.0))
        else:
            # Night: hours elapsed since sunset, wrapping past midnight.
            elapsed = (h - sunset) if h > sunset else (h + 24.0 - sunset)
            decay = np.exp(-NIGHT_DECAY_COEF * elapsed / max(night_len, 1e-6))
            theta[i] = t_min + (t_sunset - t_min) * decay

    if t_mean is not None and np.isfinite(t_mean):
        theta = theta + (float(t_mean) - float(theta.mean()))

    return theta


def _smooth_circular(src: pd.Series, window: int = 15) -> pd.Series:
    """Day-of-year aggregate on a full 1..366 index, smoothed circularly so
    31 Dec and 1 Jan are neighbours."""
    full = pd.Series(index=np.arange(1, 367), dtype=float)
    for k, v in src.items():
        if 1 <= int(k) <= 366 and pd.notna(v):
            full.loc[int(k)] = float(v)
    if full.notna().sum() == 0:
        return full.fillna(0.0)
    full = full.interpolate(limit_direction="both")
    tiled = pd.concat([full, full, full], ignore_index=True)
    sm = tiled.rolling(window, center=True, min_periods=1).mean()
    return pd.Series(sm.iloc[366:732].values, index=np.arange(1, 367))


_CLIM_CACHE: dict = {}


def climatology(bpdb: pd.DataFrame) -> dict:
    """
    Smoothed day-of-year normals for Tmin, Tmax and Tmean, from the whole
    BPDB weather record. Used for dates the dataset does not cover.
    """
    key = id(bpdb)
    if key in _CLIM_CACHE:
        return _CLIM_CACHE[key]
    w = bpdb.dropna(subset=["Temp_Max_C", "Temp_Min_C"]).copy()
    doy = pd.to_datetime(w["Date"]).dt.dayofyear
    out = {}
    for col in ("Temp_Min_C", "Temp_Max_C", "Temp_Mean_C"):
        if col in w.columns:
            out[col] = _smooth_circular(w.groupby(doy)[col].mean())
    out["_n_years"] = int(pd.to_datetime(w["Date"]).dt.year.nunique())
    out["_n_days"] = len(w)
    _CLIM_CACHE[key] = out
    return out


def profile_for_date(bpdb: pd.DataFrame, date, allow_climatology: bool = True) -> dict:
    """
    Theta(t) for one calendar date, read from the BPDB daily weather columns.
    Returns the profile plus the inputs it was built from, so any downstream
    artefact can state its provenance.
    """
    d = pd.Timestamp(date)
    row = bpdb.loc[bpdb["Date"] == d]
    weather_source = "observed"
    extra = {}

    have = (not row.empty
            and pd.notna(row.iloc[0].get("Temp_Min_C"))
            and pd.notna(row.iloc[0].get("Temp_Max_C")))

    if have:
        r = row.iloc[0]
        t_min = float(r["Temp_Min_C"])
        t_max = float(r["Temp_Max_C"])
        t_mean = float(r["Temp_Mean_C"]) if pd.notna(r.get("Temp_Mean_C")) else None
    elif allow_climatology:
        # Beyond the dataset (the forecast horizon runs past 2026-06-29). The
        # frozen forecaster's own weather values are not exported, only a
        # `weather_source` label, so day-of-year normals are the honest
        # substitute. Every artefact carries weather_source="climatology".
        clim = climatology(bpdb)
        doy = int(d.dayofyear)
        t_min = float(clim["Temp_Min_C"].loc[doy])
        t_max = float(clim["Temp_Max_C"].loc[doy])
        t_mean = (float(clim["Temp_Mean_C"].loc[doy])
                  if "Temp_Mean_C" in clim else None)
        weather_source = "climatology"
        r = row.iloc[0] if not row.empty else pd.Series(dtype=object)
        extra = {"climatology_years": clim["_n_years"],
                 "climatology_days": clim["_n_days"]}
    else:
        raise ValueError(f"No weather for {d.date()} and climatology disabled")

    if not (np.isfinite(t_min) and np.isfinite(t_max)):
        raise ValueError(f"{d.date()}: Temp_Min_C / Temp_Max_C missing")
    if t_max < t_min:
        raise ValueError(f"{d.date()}: Temp_Max_C {t_max} < Temp_Min_C {t_min}")

    doy = int(d.dayofyear)
    theta = diurnal_profile(t_min, t_max, doy, t_mean)
    sunrise, sunset = sun_times(doy)

    return {
        "date": str(d.date()),
        "theta_c": [float(v) for v in theta],
        "t_min_c": t_min, "t_max_c": t_max,
        "t_mean_recorded_c": t_mean,
        "t_mean_reconstructed_c": float(theta.mean()),
        "humidity_pct": float(r["Humidity_Mean_Pct"])
        if pd.notna(r.get("Humidity_Mean_Pct")) else None,
        "rain_mm": float(r["Rain_Sum_mm"]) if pd.notna(r.get("Rain_Sum_mm")) else None,
        "day_of_week": str(r.get("Day_of_Week", "")),
        "is_weekend": int(r.get("Is_Weekend", 0)),
        "holiday_type": str(r.get("Holiday_type", "No Holiday")),
        "season": str(r.get("Season", "")),
        "sunrise_h": sunrise, "sunset_h": sunset,
        "day_length_h": sunset - sunrise,
        "peak_slot": int(np.argmax(theta)),
        "peak_clock": cfg.slot_to_clock(int(np.argmax(theta))),
        "method": ("Parton-Logan (1981) sine-day / exponential-night, "
                   "mean-corrected to Temp_Mean_C"),
        "weather_source": weather_source,
        "provenance": ("RECONSTRUCTED from observed daily Tmin/Tmax/Tmean"
                       if weather_source == "observed" else
                       "RECONSTRUCTED from day-of-year CLIMATOLOGY - the date "
                       "is beyond the BPDB record"),
        **extra,
    }


def validate(profile: dict) -> list:
    """Checks that the reconstruction respects the observations it came from."""
    out = []

    def _c(name, ok, detail=""):
        out.append({"check": name, "pass": bool(ok), "detail": str(detail)})

    th = np.array(profile["theta_c"])
    _c("min(Theta) is within 0.5 C of the recorded Temp_Min_C",
       abs(th.min() - profile["t_min_c"]) < 0.5,
       f"{th.min():.2f} vs {profile['t_min_c']:.2f}")
    _c("max(Theta) is within 0.5 C of the recorded Temp_Max_C",
       abs(th.max() - profile["t_max_c"]) < 0.5,
       f"{th.max():.2f} vs {profile['t_max_c']:.2f}")
    if profile["t_mean_recorded_c"] is not None:
        _c("slot-mean(Theta) reproduces the recorded Temp_Mean_C",
           abs(th.mean() - profile["t_mean_recorded_c"]) < 0.05,
           f"{th.mean():.3f} vs {profile['t_mean_recorded_c']:.2f}")
    if profile.get("weather_source") == "climatology":
        _c("climatological day flagged", True,
           "weather_source=climatology; Tmin/Tmax are day-of-year normals, "
           "not observations")
    _c("Theta peaks in the afternoon (13:00-16:00)",
       26 <= profile["peak_slot"] <= 32,
       f"peak at {profile['peak_clock']}")
    _c("Theta is smooth (no slot-to-slot jump > 2 C)",
       float(np.abs(np.diff(th)).max()) < 2.0,
       f"max step {np.abs(np.diff(th)).max():.3f} C")
    return out
