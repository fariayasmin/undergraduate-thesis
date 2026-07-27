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
# Monte-Carlo cohort size. Convergence study (3 seeds x N in 36/72/150/300/
# 600): the mean seasonal swing and MAPE are stable from N=72, but the
# between-seed sd of the swing keeps falling to N~300 (Dhanmondi 0.023 ->
# 0.016) and of MAPE to 0.05. 300 is the variance/cost knee.
N_ARCHETYPES  = 300

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
# alpha_upper raised 0.35 -> 0.40 (commit 5b). Under the real BERC marginal
# slab an upper-middle household sits in SLAB_7 (14.61 BDT/kWh, peak 19.72),
# so Eq. 31 gives alpha* = d_c/(r*mu_pk + rho + d_c) = 14/(19.72+3+14) = 0.381.
# At alpha = 0.40 the tier curtails on STRESS days only (0.40*22.72 = 9.09 >
# 0.60*14 = 8.40) and never routinely (0.40*19.72 = 7.89 < 8.40), and still
# never shifts (0.40*8.04 = 3.21 < 4.80). The rebate is what buys the highest-
# consumption tier's participation, and only when the grid needs it.
TIER_ALPHA          = {"lower_middle": 0.80, "middle": 0.60, "upper_middle": 0.40}
DISCOMFORT_SHIFT    = {"lower_middle": 2.0, "middle": 4.0, "upper_middle": 8.0}
DISCOMFORT_CURTAIL  = {"lower_middle": 4.0, "middle": 7.0, "upper_middle": 14.0}
MAX_CURTAIL_FRAC    = 0.5
DR_STRESS_QUANTILE  = 0.90      # top 10% synthetic peak days = grid stress
DR_REBATE_BDT_KWH   = 3.0


# ── Non-residential cooling response ─────────────────────────────────────────
# unit_profile applies ac_factor(s) = 1 + s * sigmoid((T - 28.5) / 1.7) to the
# temperature-sensitive non-residential classes. `s` is the cooling strength of
# that class; NONRES_COOLING_SCALE is a single global multiplier used as the
# calibration parameter for the residual seasonal amplitude (see CHANGELOG).
# Household AC penetration is NOT used for this: the catalogue already implies
# a weighted residential penetration at or above what the BBS/BSVS urban series
# supports, so the residual is absorbed by commercial cooling, which household
# surveys do not constrain.
NONRES_COOLING_STRENGTH = {
    "commercial_shop": 0.70,
    "office_smb":      0.70,
    "school":          0.25,
    "hospital":        0.75,
    "clinic_small":    0.75,
    "mosque":          0.30,
    "workshop":        0.00,
}
NONRES_COOLING_SCALE = 1.00


# ── Retail tariff: BERC residential slabs ────────────────────────────────────
# Single source of truth. Previously this table lived only in
# 08_knowledge_graph.py while 06_molp_optimizer.py billed at a flat invented
# tier rate (6.5 / 8.0 / 10.5), so the knowledge graph exposed slab rates that
# no bill had ever been computed from -- a household's MonthlyBill could not be
# reconciled with its own SUBJECT_TO slab, breaking the Eq. 50 grounding
# constraint and the GF = 1 claim.  Both layers now read this table.
#   (slab_id, from_kwh, to_kwh, rate_bdt_per_kwh)
TARIFF_SLABS = [
    (1,   0,     50, 4.63), (2,   0,     75, 5.26), (3,  76,    200,  7.20),
    (4, 201,    300, 7.59), (5, 301,    400, 8.02), (6, 401,    600, 12.67),
    (7, 601, 100000, 14.61),
]


def slab_for(monthly_kwh: float) -> int:
    """Slab id containing a monthly consumption."""
    for sid, lo, hi, _ in TARIFF_SLABS:
        if lo <= monthly_kwh <= hi:
            return sid
    return TARIFF_SLABS[-1][0]


def marginal_slab_rate(monthly_kwh: float) -> float:
    """Marginal retail rate (BDT/kWh) at a given monthly consumption.

    The MOLP thresholds (Eqs. 30-31) are exact only because the objective is
    LINEAR in (x, y).  A full block tariff is piecewise linear, so the marginal
    rate is used instead of the flat tier rate: evaluated at the household's
    BASELINE monthly consumption it is a per-household CONSTANT, independent of
    the decision variables, and the bang-bang result is preserved unchanged
    with r_g replaced by r_slab(h).

    Baseline (not optimised) consumption is used deliberately: making the rate
    depend on the post-DR bill would make the threshold depend on its own
    solution.
    """
    for _, lo, hi, rate in TARIFF_SLABS:
        if lo <= monthly_kwh <= hi:
            return rate
    return TARIFF_SLABS[-1][3]