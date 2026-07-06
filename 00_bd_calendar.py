"""
00_bd_calendar.py — SIMPLIFIED Bangladesh calendar, 30-minute resolution, 2024-01-01 → 2026-06-28.

Design decision (per supervisor feedback): too many calendar attributes hurt
the forecaster. Only the drivers that measurably move Dhaka electricity
consumption are kept, and every minor government holiday is collapsed into a
single "Public Holiday" class — exactly mirroring the Holiday_type encoding
of the BPDB substation CSV {No Holiday / Public Holiday / Eid / Durga Puja}.

Output: data/00_bd_calendar_30min.csv   (43,728 rows × 11 cols)

Columns
-------
timestamp        30-min timestamp, e.g. 2024-03-11 18:30:00
date             calendar date
hour             0–23
minute           0 or 30
day_of_week      0=Mon … 6=Sun
day_name         Monday …
is_weekend       1 = Friday/Saturday (Bangladesh weekend)
is_ramadan       1 during Ramadan (2024, 2025 and 2026 spans)
day_type         Weekday | Weekend | Public Holiday | Eid | Durga Puja
is_holiday       1 for any of {Public Holiday, Eid, Durga Puja}
load_multiplier  simple daily demand factor used by the load generator
"""

import pandas as pd
import config as C


def build_calendar() -> pd.DataFrame:
    idx = pd.date_range(C.START_DATE, f"{C.END_DATE} 23:30", freq=C.FREQ)
    df = pd.DataFrame({"timestamp": idx})
    df["date"]        = df["timestamp"].dt.date.astype(str)
    df["hour"]        = df["timestamp"].dt.hour
    df["minute"]      = df["timestamp"].dt.minute
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["day_name"]    = df["timestamp"].dt.day_name()
    df["is_weekend"]  = df["day_of_week"].isin([4, 5]).astype(int)  # Fri, Sat

    ram = []
    for lo, hi in C.RAMADAN_RANGES:
        ram += list(pd.date_range(lo, hi).astype(str))
    df["is_ramadan"] = df["date"].isin(ram).astype(int)

    # day_type priority: Eid > Durga Puja > Public Holiday > Weekend > Weekday
    df["day_type"] = "Weekday"
    df.loc[df["is_weekend"] == 1,                 "day_type"] = "Weekend"
    df.loc[df["date"].isin(C.PUBLIC_HOLIDAYS),    "day_type"] = "Public Holiday"
    df.loc[df["date"].isin(C.DURGA_PUJA_DAYS),    "day_type"] = "Durga Puja"
    df.loc[df["date"].isin(C.EID_DAYS),           "day_type"] = "Eid"

    df["is_holiday"] = df["day_type"].isin(
        ["Public Holiday", "Eid", "Durga Puja"]).astype(int)

    # Simple whole-day demand factor (substation level, Dhaka evidence:
    # Eid days drop hardest — city partially empties; weekends mild drop).
    mult = {"Weekday": 1.00, "Weekend": 0.94, "Public Holiday": 0.92,
            "Durga Puja": 0.95, "Eid": 0.78}
    df["load_multiplier"] = df["day_type"].map(mult)

    return df


def main():
    df = build_calendar()
    out = C.DATA_DIR / "00_bd_calendar_30min.csv"
    df.to_csv(out, index=False)
    print(f"Calendar written: {out}  ({len(df)} rows)")
    print(df["day_type"].value_counts().to_string())


if __name__ == "__main__":
    main()