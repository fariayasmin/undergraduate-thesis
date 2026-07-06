"""
01_dhaka_weather.py — SIMPLIFIED Dhaka weather, 30-minute resolution, 2024-01-01 → 2026-06-28.

The daily anchors (Temp_Max_C, Temp_Min_C, Temp_Mean_C, Humidity_Mean_Pct)
are taken directly from the REAL weather columns of the BPDB substation CSV,
so the synthetic 30-min weather is consistent with the exact series the
substation forecaster uses. A physically sensible diurnal curve fills each
day (minimum ~05:30, maximum ~14:30).

Per supervisor feedback the attribute list is kept minimal — precipitation,
wind, cloud, radiation, ac/fan multipliers, rain events and comfort indices
are all REMOVED. Temperature-driven appliance behaviour (fans/AC) is handled
INSIDE the load generator from temperature_c alone.

Output: data/01_dhaka_weather_30min.csv  (43,728 rows × 4 cols)

Columns
-------
timestamp               30-min timestamp
temperature_c           air temperature (°C)
relative_humidity_pct   relative humidity (%)
season                  winter | summer | monsoon | autumn
"""

import numpy as np
import pandas as pd
import config as C

SEASON_BY_MONTH = {12: "winter", 1: "winter", 2: "winter",
                   3: "summer", 4: "summer", 5: "summer",
                   6: "monsoon", 7: "monsoon", 8: "monsoon", 9: "monsoon",
                   10: "autumn", 11: "autumn"}


def load_daily_anchors() -> pd.DataFrame:
    bp = pd.read_csv(C.BPDB_CSV, usecols=[
        "Date", "Temp_Max_C", "Temp_Min_C", "Temp_Mean_C", "Humidity_Mean_Pct"])
    bp["Date"] = pd.to_datetime(bp["Date"])
    bp = bp[(bp["Date"] >= C.START_DATE) & (bp["Date"] <= C.END_DATE)]
    bp = bp.set_index("Date").sort_index()

    # reindex to every calendar day and fill the few gaps by interpolation
    full = pd.date_range(C.START_DATE, C.END_DATE, freq="D")
    bp = bp.reindex(full).interpolate(limit_direction="both")
    bp.index.name = "date"
    return bp


def diurnal_temperature(tmin: float, tmax: float, hours: np.ndarray) -> np.ndarray:
    """Half-cosine rise 05:30→14:30, half-cosine fall 14:30→(next)05:30."""
    t = np.empty_like(hours, dtype=float)
    rise = (hours >= 5.5) & (hours <= 14.5)
    t[rise] = tmin + (tmax - tmin) * 0.5 * (
        1 - np.cos(np.pi * (hours[rise] - 5.5) / 9.0))
    fall = ~rise
    h = np.where(hours[fall] > 14.5, hours[fall], hours[fall] + 24.0)  # 14.5→29.5
    t[fall] = tmax - (tmax - tmin) * 0.5 * (1 - np.cos(np.pi * (h - 14.5) / 15.0))
    return t


def main():
    rng = np.random.default_rng(C.SEED)
    anchors = load_daily_anchors()

    idx = pd.date_range(C.START_DATE, f"{C.END_DATE} 23:30", freq=C.FREQ)
    hours_of_day = idx.hour.values + idx.minute.values / 60.0

    temps, hums = [], []
    for day, row in anchors.iterrows():
        h = hours_of_day[:C.SLOTS_PER_DAY]
        t = diurnal_temperature(row["Temp_Min_C"], row["Temp_Max_C"], h)
        # humidity moves inversely to temperature around the daily mean
        t_norm = (t - t.mean()) / max(t.max() - t.min(), 1e-6)
        rh = row["Humidity_Mean_Pct"] - 24.0 * t_norm
        temps.append(t)
        hums.append(np.clip(rh, 25, 100))

    temperature = np.concatenate(temps) + rng.normal(0, 0.25, len(idx))
    humidity    = np.concatenate(hums)  + rng.normal(0, 1.0,  len(idx))

    df = pd.DataFrame({
        "timestamp": idx,
        "temperature_c": np.round(temperature, 2),
        "relative_humidity_pct": np.round(np.clip(humidity, 25, 100), 1),
        "season": [SEASON_BY_MONTH[m] for m in idx.month],
    })

    out = C.DATA_DIR / "01_dhaka_weather_30min.csv"
    df.to_csv(out, index=False)
    print(f"Weather written: {out}  ({len(df)} rows)")
    print(df.groupby("season")["temperature_c"].agg(["mean", "max"]).round(1))


if __name__ == "__main__":
    main()