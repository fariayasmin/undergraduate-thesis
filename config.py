"""
config.py — shared configuration for the Kalyanpur + Dhanmondi
low-level (appliance / building) synthetic dataset pipeline.

Period: 2024-01-01 → 2026-06-28 (last date of the BPDB substation record),
30-minute resolution. Calibrated per YEAR against BPDB daily peaks with a
realistic customer composition — counts of households / buildings, never
profile scaling.
"""

from pathlib import Path
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).parent
DATA_DIR  = BASE_DIR / "data"
KG_DIR    = BASE_DIR / "kg"
DATA_DIR.mkdir(exist_ok=True)
KG_DIR.mkdir(exist_ok=True)

_BPDB_CANDIDATES = [
    BASE_DIR / "BPDB_Dhaka_City_Substations_Page3.csv",
    Path("/mnt/user-data/uploads/BPDB_Dhaka_City_Substations_Page3.csv"),
    Path("BPDB_Dhaka_City_Substations_Page3.csv"),
]
BPDB_CSV = next((p for p in _BPDB_CANDIDATES if p.exists()), _BPDB_CANDIDATES[0])

# ── Time base ────────────────────────────────────────────────────────────────
START_DATE = "2024-01-01"
END_DATE   = "2026-06-28"        # last date with BPDB substation data
FREQ = "30min"
SLOTS_PER_DAY = 48
SEED = 42
YEARS = [2024, 2025, 2026]

SUBSTATIONS = ["Kalyanpur", "Dhanmondi"]
MAX_PLAUSIBLE_LOAD_MW = 450      # same artefact filter as load_prediction.py


def bpdb_yearly_targets(substation: str) -> dict:
    """Mean daily peak (MW) per year from the real BPDB record."""
    bp = pd.read_csv(BPDB_CSV, usecols=["Date", f"{substation}_Load_MW"])
    bp["Date"] = pd.to_datetime(bp["Date"])
    bp = bp[(bp["Date"] >= START_DATE) & (bp["Date"] <= END_DATE)]
    s = bp[f"{substation}_Load_MW"].where(
        bp[f"{substation}_Load_MW"] <= MAX_PLAUSIBLE_LOAD_MW)
    return s.groupby(bp["Date"].dt.year).mean().round(2).to_dict()


# 2024 anchors used to SIZE the initial composition (per-year k refines it)
BPDB_MEAN_DAILY_PEAK_MW = {"Kalyanpur": 96.6, "Dhanmondi": 119.8}

# ── Customer composition (shares of mean daily peak) ─────────────────────────
COMPOSITION = {
    "Kalyanpur": {"residential": 0.66, "commercial": 0.17, "educational": 0.04,
                  "health": 0.02, "industrial": 0.06, "infrastructure": 0.05},
    "Dhanmondi": {"residential": 0.54, "commercial": 0.21, "educational": 0.07,
                  "health": 0.12, "industrial": 0.01, "infrastructure": 0.05},
}

TIER_MIX = {
    "Kalyanpur": {"lower_middle": 0.35, "middle": 0.50, "upper_middle": 0.15},
    "Dhanmondi": {"lower_middle": 0.12, "middle": 0.43, "upper_middle": 0.45},
}

TIER_COINCIDENT_PEAK_KW = {"lower_middle": 0.35, "middle": 0.95,
                           "upper_middle": 2.60}

TIER_INCOME_BDT = {"lower_middle": (15000, 35000), "middle": (35001, 65000),
                   "upper_middle": (65001, 120000)}

UNIT_PEAK_KW = {"commercial_shop": 3.0, "office_smb": 8.0, "school": 18.0,
                "clinic_small": 80.0, "workshop": 12.0,
                "streetlight_block": 12.0, "wasa_pump": 150.0, "mosque": 10.0}

DHANMONDI_HOSPITALS = {
    "Labaid Specialized Hospital": 2000, "Labaid Cardiac Hospital": 900,
    "Popular Medical College Hospital": 1600, "Ibn Sina Hospital Dhanmondi": 1300,
    "Anwer Khan Modern Hospital": 1200, "Bangladesh Medical College Hosp.": 1100,
    "Central Hospital Dhanmondi": 900, "Gonoshasthaya Nagar Hospital": 700,
    "City Hospital Ltd": 700, "Medinova Medical Services": 600,
    "Renaissance Hospital": 500, "Comfort Diagnostic Centre": 500,
}

SUBSTATION_META = {   # for the knowledge graph
    "Kalyanpur": {"utility": "DESCO", "latitude": 23.7776, "longitude": 90.3560,
                  "feeder_voltage_kv": 33.0},
    "Dhanmondi": {"utility": "DPDC",  "latitude": 23.7461, "longitude": 90.3742,
                  "feeder_voltage_kv": 33.0},
}

N_DETAILED_HH = 120
N_ARCHETYPES  = 36

# ── Bangladesh calendar 2024–2026 (simplified; lunar dates approximate,
#    edit here if the observed dates differed) ───────────────────────────────
RAMADAN_RANGES = [("2024-03-12", "2024-04-09"),
                  ("2025-03-02", "2025-03-30"),
                  ("2026-02-19", "2026-03-19")]

EID_DAYS = [
    # Eid-ul-Fitr spans
    "2024-04-10", "2024-04-11", "2024-04-12",
    "2025-03-30", "2025-03-31", "2025-04-01",
    "2026-03-20", "2026-03-21", "2026-03-22",
    # Eid-ul-Adha spans
    "2024-06-16", "2024-06-17", "2024-06-18",
    "2025-06-06", "2025-06-07", "2025-06-08",
    "2026-05-26", "2026-05-27", "2026-05-28",
]

DURGA_PUJA_DAYS = [
    "2024-10-09", "2024-10-10", "2024-10-11", "2024-10-12", "2024-10-13",
    "2025-09-28", "2025-09-29", "2025-09-30", "2025-10-01", "2025-10-02",
]

PUBLIC_HOLIDAYS = [
    # 2024
    "2024-02-21", "2024-02-26", "2024-03-17", "2024-03-26", "2024-04-07",
    "2024-04-14", "2024-05-01", "2024-05-22", "2024-07-17", "2024-08-15",
    "2024-08-26", "2024-09-16", "2024-12-16", "2024-12-25",
    # 2025 (Mar 17 & Aug 15 dropped from the official list after Aug 2024)
    "2025-02-15", "2025-02-21", "2025-03-26", "2025-03-28", "2025-04-14",
    "2025-05-01", "2025-05-11", "2025-07-06", "2025-08-16", "2025-09-05",
    "2025-12-16", "2025-12-25",
    # 2026 (through 28 Jun)
    "2026-02-04", "2026-02-21", "2026-03-16", "2026-03-26", "2026-04-14",
    "2026-05-01", "2026-06-25",
]

# ── MOLP / tariff parameters ─────────────────────────────────────────────────
TOU = {"off_peak": {"hours": (23, 6),  "multiplier": 0.80},
       "peak":     {"hours": (17, 23), "multiplier": 1.35},
       "standard": {"hours": None,     "multiplier": 1.00}}

TIER_RATE_BDT_KWH   = {"lower_middle": 6.5, "middle": 8.0, "upper_middle": 10.5}
SGO_SUPPLY_COST     = {"off_peak": 5.5, "standard": 7.5, "peak": 11.5}  # BDT/kWh
TIER_ALPHA          = {"lower_middle": 0.80, "middle": 0.60, "upper_middle": 0.35}
DISCOMFORT_SHIFT    = {"lower_middle": 2.0, "middle": 4.0, "upper_middle": 8.0}
DISCOMFORT_CURTAIL  = {"lower_middle": 4.0, "middle": 7.0, "upper_middle": 14.0}
MAX_CURTAIL_FRAC    = 0.5
DR_STRESS_QUANTILE  = 0.90      # top 10% synthetic peak days = grid stress
DR_REBATE_BDT_KWH   = 3.0
