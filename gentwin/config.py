"""
GenTwin-SG :: central configuration
===================================

Single source of truth for every parameter in the pipeline.

Every entry carries three annotations, per the brief:

    EQ      which equation of GenTwinSG_Formulation_v3.pdf it serves
    SOURCE  one of
              MEASURED   - read from the BPDB dataset
              GAZETTE    - BERC retail tariff order, 03 June 2026
              LITERATURE - published value, reference given
              SCENARIO   - engineering assumption, not available from BPDB
    WHY     why the parameter is needed at all

Nothing downstream may hard-code a number that belongs here.

UNIT CONVENTION (fixed by the formulation, section 1)
-----------------------------------------------------
    power       kW
    energy      kWh
    price       Tk/kWh
    time        30-minute slots, 48 per day, Delta = 0.5 h
    appliances  catalogued in kW, matching P^rat_a of Eq. (2)

These are the units the formulation gives in its symbol tables, so no
conversion layer is needed anywhere: the BERC gazette is already in Tk/kWh,
and gamma, rho and pi^ret are all Tk/kWh natively. Tk and BDT are the same
currency; the formulation writes Tk, so does this code.
"""

from __future__ import annotations

from pathlib import Path

# =============================================================================
# 0. Paths
# =============================================================================

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
OUT_DIR = ROOT / "outputs"
# One true vector PDF per individual graph (not a raster PNG wrapped in a PDF
# container): every multi-panel figure also renders each of its panels here
# as its own standalone Figure, drawn straight from the same data.
PDF_DIR = ROOT / "outputs_pdf"

# Inputs produced by the (frozen) forecasting stage. Never written to.
BPDB_CSV = DATA_DIR / "BPDB_Dhaka_City_Substations_Page3.csv"
FORECAST_DIR = DATA_DIR / "forecast_outputs"

for _d in (DATA_DIR, CACHE_DIR, OUT_DIR, PDF_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 1. Time discretisation
# =============================================================================
# EQ     (2)-(6), (17)-(33) - every equation indexed by t
# SOURCE MEASURED - the BPDB `*_Peak_Time` column only ever takes values on
#        :00 or :30 boundaries (load_prediction.is_clocktime_artifact enforces
#        `m in (0, 30)`), so 30 minutes is the native resolution of the only
#        intra-day observation in the dataset. Anything coarser discards it.

SLOTS_PER_DAY = 48
DELTA_H = 0.5                      # Delta in equations (18), (22), (23), (26)


def slot_to_clock(slot: int) -> str:
    """Slot index 0..47 -> 'HH:MM' of the slot's start."""
    return f"{slot // 2:02d}:{(slot % 2) * 30:02d}"


def clock_to_slot(clock: str) -> int:
    """'HH:MM' -> slot index 0..47. Raises on a non half-hour boundary."""
    h, m = (int(x) for x in clock.split(":"))
    if m not in (0, 30):
        raise ValueError(f"{clock!r} is not on a 30-minute boundary")
    return h * 2 + (1 if m == 30 else 0)


# =============================================================================
# 2. Substations
# =============================================================================
# Coordinates
#   SOURCE SCENARIO (provisional). The brief promises nodes_Substation.csv;
#   service_area.load_substation_coords() prefers that file whenever it is
#   present in data/ and only falls back to these. Values below are the
#   approximate BPDB 132/33 kV site locations and are flagged provisional so
#   they cannot be mistaken for surveyed coordinates.
#
# T_pk (peak window)  -- FALLBACK LITERALS ONLY
#   EQ     (5), (32), (34) - T^pk_i,d and its complement T^off_i,d
#   SOURCE MEASURED, derived by gentwin/peak_window.py and cached to
#          data/cache/peak_window_*.json. The literals below are a fallback
#          used only if that derivation has not been run; peak_window
#          .get_peak_slots() warns loudly when it falls back. They are the
#          same values the estimator converges on.
#          Derived from the empirical `_Peak_Time` distribution:
#            Dhanmondi  12:00 (11.0%), 13:00 (63.8%), 14:00 (6.5%)
#                       -> slots 24..29 = 12:00-15:00 covers 81.3% of days
#            Kalyanpur  18:00 (16.7%), 18:30 (10.2%), 19:00 (11.9%),
#                       19:30 (5.2%) -> slots 36..41 = 18:00-21:00 covers 44%,
#                       the largest coherent cluster once the 00:00 artefact
#                       (see PEAK_TIME_ARTEFACT_SLOTS) is removed.
#
# P_max, tau, storage, tie-line
#   SOURCE SCENARIO - BPDB publishes none of these. See section 5.

SUBSTATIONS = {
    "Dhanmondi": {
        "code": "DHN",
        "load_col": "Dhanmondi_Load_MW",
        "peak_time_col": "Dhanmondi_Peak_Time",
        "latitude": 23.7465,            # SCENARIO (provisional, see above)
        "longitude": 90.3760,
        "coords_provisional": True,
        "feeder_voltage_kv": 33,
        "upstream_kv": 132,
        "t_pk_slots": tuple(range(24, 30)),   # 12:00 - 15:00
        "t_pk_label": "12:00-15:00 (observed midday peak)",
        # EQ (15), (30), (45), (46) - firm capacity P^max_i
        # SCENARIO: 1.08 x the historical maximum observed daily peak
        # (185.7 MW, 2025+). A substation is not normally operated at its
        # nameplate, so a modest headroom over the observed record is the
        # least arbitrary defensible choice.
        "p_max_kw": 200_000.0,
        # EQ (15) - stress threshold P^str_i, in kW (the revised formulation
        # renames tau_i to P^str_i and notes it "carries the unit of a peak
        # load, so that (15) compares two quantities in kW"; tau is reserved
        # for a slot index). Derived from data - see forecast.derive_p_str().
        "p_str_kw": None,               # None => derive from data
        # EQ (26), (28), (29), (31) - storage. SCENARIO: a hypothetical BESS.
        # There is no battery at either substation today; this is a design
        # study, and the thesis must say so.
        "e_max_kwh": 40_000.0,
        "b_max_kw": 10_000.0,
        "reserve_kwh": 8_000.0,
        "s0_kwh": 20_000.0,   # = 0.5 * e_max
        # EQ (18), (22), (39) - producer price c_{i,d} of the day-d energy
        # block, Tk/kWh. The revised formulation makes procurement DAILY: the
        # substation commits to one block per day at a price fixed at purchase,
        # then draws on it slot by slot. So this is one price per day, not a
        # per-slot c_i(t).
        # SOURCE: BPDB bulk (wholesale) tariff 8.39 Tk/kWh after the June 2026
        # revision (from 7.00).
        "c_producer_tk_per_kwh": 8.39,
        # EQ (31) - import cap G^max_i. SCENARIO: firm capacity.
        "g_max_kw": 200_000.0,
    },
    "Kalyanpur": {
        "code": "KLN",
        "load_col": "Kalyanpur_Load_MW",
        "peak_time_col": "Kalyanpur_Peak_Time",
        "latitude": 23.7960,            # SCENARIO (provisional)
        "longitude": 90.3610,
        "coords_provisional": True,
        "feeder_voltage_kv": 33,
        "upstream_kv": 132,
        "t_pk_slots": tuple(range(36, 42)),   # 18:00 - 21:00
        "t_pk_label": "18:00-21:00 (observed evening peak)",
        "p_max_kw": 160_000.0,              # 1.13 x observed max 141.0 MW
        "p_str_kw": None,
        "e_max_kwh": 30_000.0,
        "b_max_kw": 8_000.0,
        "reserve_kwh": 6_000.0,
        "s0_kwh": 15_000.0,
        "c_producer_tk_per_kwh": 8.39,
        "g_max_kw": 160_000.0,
    },
}

# EQ (26), (27), (28) - round-trip efficiencies eta^ch, eta^dis
# SOURCE LITERATURE: 0.95 one-way is the standard planning value for
# grid-scale Li-ion (IRENA, Electricity Storage and Renewables, 2017).
ETA_CH = 0.95
ETA_DIS = 0.95

# EQ (18), (22) - battery throughput cost c^B, Tk/kWh charged or discharged.
# SCENARIO: degradation-equivalent, deliberately small but strictly positive so
# that Proposition 1's "all prices strictly positive" premise holds and the LP
# does not cycle the battery for free.
C_BATTERY_TK_PER_KWH = 1.50

# EQ (23) - reserve shortfall price pi^nu, Tk/kWh.
# NOTE on units, from the revised Eq. (23): Delta attaches ONLY to Z^c_i(t),
# which is a power in kW. nu_i(t) is ALREADY an energy in kWh, because C3
# compares it against the reserve floor R_i in kWh. The previous
# implementation multiplied both by Delta, understating the reserve penalty by
# a factor of 1/Delta = 2. Corrected.
PI_RESERVE_TK_PER_KWH = 25.0

# EQ (23), (39) - value of lost load v_c, Tk/kWh, by class.
# SOURCE LITERATURE (order of magnitude): VoLL for South Asian distribution
# networks is commonly placed at 5-20x the retail rate. Strictly ordered
# v_cr > v_sh > v_cu, as section 1.3 requires, so the merit-order premise of
# Proposition 3 is satisfiable.
VOLL_TK_PER_KWH = {
    "critical": 150.0,
    "shiftable": 100.0,
    "curtailable": 60.0,
}

# =============================================================================
# 3. Tie-lines
# =============================================================================
# EQ (29), (32), (40) - L, T^max_ij, loss fraction l_ij, transfer price pi_ij
# SOURCE SCENARIO. Dhanmondi and Kalyanpur are ~6 km apart. Whether a 33 kV
# tie exists between exactly these two substations is not public; the study
# models one, which is the whole point of testing whether household
# flexibility can substitute for it (Formulation section 8, closing remark).

TIE_LINES = [
    {
        "from": "Dhanmondi",
        "to": "Kalyanpur",
        "t_max_kw": 20_000.0,
        "loss_fraction": 0.02,          # l_ij in (27)
        "price_tk_per_kwh": 9.50,       # pi_ij, > c_{i,d} so (39) can hold
    },
    {
        "from": "Kalyanpur",
        "to": "Dhanmondi",
        "t_max_kw": 20_000.0,
        "loss_fraction": 0.02,
        "price_tk_per_kwh": 9.50,
    },
]

# =============================================================================
# 4. Data-quality flags
# =============================================================================
# MEASURED, and a finding in its own right: 13.1% of Kalyanpur rows carry a
# peak time of exactly 00:00 (180 of them in 2024+ alone), against 1.0% at
# Dhanmondi. A midnight peak at a Dhaka distribution substation is not
# physically credible at that frequency; it is the default written when the
# field was not recorded. Excluded when deriving T^pk, and reported.
PEAK_TIME_ARTEFACT_SLOTS = (0,)

# =============================================================================
# 5. Adaptive local time-of-use
# =============================================================================
# EQ (19), (34) - mu_i(t), and EQ (5), (32) - T^pk_i
#
# DIVISION OF RESPONSIBILITY, and this must be stated wherever the mechanism
# is reported:
#
#   BERC       sets the peak / off-peak price RATIOS. mu^pk = 1.20 and
#              mu^off = 0.90 are read off the order of 03 June 2026 as the
#              LT-class peak/flat and off-peak/flat ratios (LT-C1 1.1995 /
#              0.8995, LT-E 1.2012 / 0.8997). SOURCE: GAZETTE.
#   GenTwin-SG determines WHEN the peak ratio is active, per substation, from
#              that substation's own congestion history. SOURCE: MEASURED,
#              derived by gentwin/peak_window.py.
#
# This is NOT a replacement for the official tariff. It is a congestion-aware
# activation schedule laid over the regulator's own multipliers. The national
# 17:00-23:00 window is retained in tariff.OFFICIAL_TOU_PEAK for the ablation.

MU_PEAK = 1.20                     # mu^pk, GAZETTE (LT family)
MU_OFF = 0.90                      # mu^off, GAZETTE (all ToU classes)
ADAPTIVE_LOCAL_TOU = True          # False => use the national window instead

# Estimator settings for T^pk_i. See gentwin/peak_window.py for the rationale
# behind each; sensitivity across all of them is reported by
# 00b_peak_window.py --sensitivity.
PEAK_WINDOW_STRESS_QUANTILE = 0.95   # congestion days only
PEAK_WINDOW_BANDWIDTH_SLOTS = 2.0    # KDE bandwidth, 1 hour
PEAK_WINDOW_THRESHOLD_FRAC = 0.50    # keep slots >= 0.50 x max density
PEAK_WINDOW_BOOTSTRAP = 500

# =============================================================================
# 6. Optimisation weights and robustness
# =============================================================================
# EQ (24) - theta_1 (operator cost), theta_2 (reliability), theta_3 (welfare)
# SOURCE SCENARIO. Formulation section 9.5 lists a sensitivity sweep of these
# as an open item; run 06b_sensitivity_theta.py.
THETA_COST = 1.0
THETA_RELIABILITY = 1.0
THETA_WELFARE = 1.0

# EQ (14) - z_beta in the robust peak p~ = p_hat + z_beta * sigma
# SOURCE LITERATURE: one-sided normal quantile at beta = 0.95.
Z_BETA = 1.645
BETA_CONFIDENCE = 0.95

# Formulation, "How the K-step horizon is used": only the steps covered by the
# operational horizon T enter the constraints through kappa^scale_{i,d(t)}. The
# remaining steps of (10) supply the stress indicator s_{i,d+k} and the deficit
# Def_{i,d+k} of (15), and hence the regime r_{i,d} of (46) and the priority
# weights beta_j(r) of (47). Keeping DISPATCH_DAYS < FORECAST_HORIZON is what
# makes the dispatch programme a single simultaneous solve (Corollary 1)
# rather than a rolling multi-day one.
FORECAST_HORIZON = 14              # K, as exported by the frozen pipeline
DISPATCH_DAYS = 3                  # days entering C1 / C4

# EQ (13) - how sigma_{i,k} is obtained.# The rolling-origin recursive backtest reports RMSE by horizon k, which is
# sqrt(Var + bias^2) >= sigma. Using it as sigma is therefore conservative in
# the direction that matters for a robust peak (it can only widen p~).
# The raw series is non-monotonic in k at Dhanmondi (k=3 RMSE 9.68 < k=1
# 12.18), which is a sampling artefact of ~37 origins per horizon, not a
# claim that a 3-day forecast beats a 1-day one. "envelope" applies a
# running maximum so uncertainty never shrinks with horizon.
SIGMA_SOURCE = "backtest_rmse"
SIGMA_MONOTONE = "envelope"        # "envelope" | "raw" | "isotonic"

# EQ (15) - stress threshold P^str_i.
# MUST be estimated on a TRAILING window, not the full record. Both study
# substations have shifted level: Dhanmondi's full-record p95 is 187,000 kW,
# which is above its entire last-365-day range (max 163,000 kW) and above its
# 2024+ maximum. A threshold estimated on 2019-2023 behaviour can essentially never
# fire, and s_{i,d} would be identically zero - no stress rebate in Eq. (19),
# no regime 2 or 3 in Eq. (46), no events. Trailing p95 gives 10/14 stress
# days at Dhanmondi and 8/14 at Kalyanpur over the current forecast horizon.
P_STR_TRAILING_DAYS = 365
P_STR_QUANTILE = 0.95

# ---------------------------------------------------------------------------
# NOTATION. The revised formulation RESOLVED the kappa collision that v3 had:
# the curtailment cap is now y^max (Eq. 34), so kappa unambiguously means the
# robust scaling factor everywhere.
#
#   kappa_{i,d+k|d}   robust SCALING FACTOR, Eq. (16) = p~ / p_{i,d}.
#                     Multiplies the decision-dependent load inside C1 (29)
#                     and C4 (32). Code: `kappa_scale`.
#   y^max (y^max_h)   CURTAILMENT CAP, Eq. (34), upper bound on y_{h,t}.
#                     Per-household by the approved extension.
#                     Code: `y_max` / consumer.y_max.
#   z_beta            QUANTILE MULTIPLIER on sigma, Eq. (14). Not a kappa.
#
# Eq. (14) is  p~ = p_hat + z_beta * sigma  -- NOT p_hat + kappa * sigma.
# Writing kappa there would be circular: kappa is DEFINED as p~/p_{i,d} in
# Eq. (16), so p~ would appear on both sides. See docs/forecast_to_kappa.md.
# ---------------------------------------------------------------------------

# =============================================================================
# 7. Population synthesis
# =============================================================================
# Representative consumers per substation, each carrying a multiplicity
# weight w_h = number of real connections it stands for. Exact, not an
# approximation: L_h scales by w_h while x and y remain fractions, so the
# programme stays linear and the Proposition 2 reduced cost merely divides
# through by theta_3 * w_h * L^cu_h(t) * Delta > 0.
N_REPRESENTATIVE_CONSUMERS = 400
POPULATION_SEED = 42

# EQ (7), (8) - aggregation acceptance tolerances
AGG_PEAK_TOLERANCE = 0.02          # |max_t Lambda_i(t) - p_id| / p_id
AGG_PEAK_SLOT_TOLERANCE = 1        # argmax may differ by <= 1 slot (30 min)

# =============================================================================
# 8. Consumer categories and tiers
# =============================================================================
# EQ (20)-(21), (24), (34), (36), (37) - household preferences.
#
# THE REVISED FORMULATION DROPS THE TIER SET G. Section 1.3 defines lambda_h,
# gamma^sh_h and gamma^cu_h PER HOUSEHOLD, and Remark 4 is explicit:
#
#   "Household heterogeneity is individual. Preferences are carried by
#    lambda_h, gamma^sh_h, gamma^cu_h per household, with no grouping into
#    classes. Grouping, if wanted later, is a restriction of these parameters
#    and changes nothing structural."
#
# So the archetypes below are a GENERATOR, not the unit of preference. Each
# synthetic consumer samples its own lambda_h, gamma^sh_h, gamma^cu_h and
# y^max_h from its archetype's distribution, and those values are stored on
# the consumer and carried into every threshold, explanation and knowledge-
# graph node. Proposition 2 then speaks about THIS household rather than about
# its class, which is what the explainability layer wants.
#
# CALIBRATION. Central values are anchored to the worked example in section 8,
# which now states them outright rather than leaving them to be inferred:
#     lambda_h1 = 0.60, lambda_h2 = 0.35, lambda_h3 = 0.20
#     gamma^cu_h = 20 Tk/kWh, pi^ret_h mu^pk = 12, rho = 3, y^max = 0.40
# (Our earlier reconstruction of these from the section 8.2 tables agreed with
# every one of them.) gamma^sh central values come from the same tables:
# 0.80 = 0.4*gamma^sh_h1 and 0.975 = 0.65*gamma^sh_h2.
#
# y^max_h: the ONE structural extension to C6, approved. Eq. (34) gives a
# single global y^max; making it per-household is the natural completion now
# that every other preference already is. See docs/y_max_h_derivation.md.
#
# HOSPITALS ARE NOT HARD-CODED AS NON-CURTAILABLE. They draw a near-zero
# lambda_h and a very large gamma^cu_h, so the Proposition 2 threshold (37)
# is unreachable at any plausible shadow price. Non-curtailment is a derived
# result of the optimisation, not an exclusion rule.

PREFERENCE_UNIT = "Tk/kWh"          # gamma and rho, matching section 1.3

#: Archetype -> (mean, sd) for each per-household preference. sd = 0 pins the
#: value. Sampling is truncated to the stated bounds at synthesis time.
CONSUMER_ARCHETYPES = {
    # --- Residential (LT-A) -------------------------------------------------
    "Res-A": {
        "category": "Residential", "tariff_class": "LT-A",
        "lambda_h": (0.60, 0.08), "gamma_sh_h": (2.0, 0.4),
        "gamma_cu_h": (20.0, 3.0), "y_max_h": (0.40, 0.05),
        "description": "High-income residential, AC-owning",
    },
    "Res-B": {
        "category": "Residential", "tariff_class": "LT-A",
        "lambda_h": (0.35, 0.07), "gamma_sh_h": (1.5, 0.3),
        "gamma_cu_h": (20.0, 3.0), "y_max_h": (0.40, 0.05),
        "description": "Middle-income residential",
    },
    "Res-C": {
        "category": "Residential", "tariff_class": "LT-A",
        "lambda_h": (0.20, 0.05), "gamma_sh_h": (3.0, 0.5),
        "gamma_cu_h": (30.0, 4.0), "y_max_h": (0.30, 0.05),
        "description": "Low-income residential, lifeline or first-slab",
    },
    # --- Non-residential ----------------------------------------------------
    "Commercial": {
        "category": "Commercial", "tariff_class": "LT-E",
        "lambda_h": (0.55, 0.10), "gamma_sh_h": (2.5, 0.5),
        "gamma_cu_h": (25.0, 4.0), "y_max_h": (0.30, 0.06),
        "description": "Shops, offices, restaurants",
    },
    "Industrial": {
        "category": "Industrial", "tariff_class": "LT-C1",
        "lambda_h": (0.65, 0.10), "gamma_sh_h": (2.0, 0.4),
        "gamma_cu_h": (22.0, 4.0), "y_max_h": (0.35, 0.06),
        "description": "Small industry, light manufacturing",
    },
    "Hospital": {
        "category": "Hospital", "tariff_class": "LT-D1",
        "lambda_h": (0.02, 0.01), "gamma_sh_h": (200.0, 20.0),
        "gamma_cu_h": (5000.0, 400.0), "y_max_h": (0.05, 0.01),
        "description": "Hospitals and clinics; threshold (37) unreachable "
                       "by parameter value, not by rule",
    },
    "Educational": {
        "category": "Educational", "tariff_class": "LT-D1",
        "lambda_h": (0.40, 0.08), "gamma_sh_h": (4.0, 0.8),
        "gamma_cu_h": (40.0, 6.0), "y_max_h": (0.25, 0.05),
        "description": "Schools, colleges, universities",
    },
    "Government": {
        "category": "Government", "tariff_class": "LT-E",
        "lambda_h": (0.30, 0.06), "gamma_sh_h": (5.0, 1.0),
        "gamma_cu_h": (45.0, 7.0), "y_max_h": (0.20, 0.04),
        "description": "Government and public-service offices",
    },
}

#: Truncation bounds applied to every sampled preference.
PREFERENCE_BOUNDS = {
    "lambda_h": (0.0, 1.0),         # section 1.3: lambda_h in [0,1]
    "gamma_sh_h": (0.1, 1e5),
    "gamma_cu_h": (0.1, 1e5),
    "y_max_h": (0.0, 1.0),          # section 1.4: y^max in [0,1]
}

#: Eq. (34) global fallback, used if a consumer carries no sampled cap.
Y_MAX_DEFAULT = 0.40

CONSUMER_CATEGORIES = [
    "Residential", "Commercial", "Industrial",
    "Hospital", "Educational", "Government",
]

# EQ (20) - stress rebate rho, Tk/kWh of curtailed energy on a stressed day.
# SCENARIO: no such rebate exists in Bangladesh. Set to 3.0, the value the
# revised section 8 states explicitly (private = lambda_h(pi^ret mu^pk + rho),
# 9.00 = 0.60 * (12 + 3)).
RHO_REBATE_TK_PER_KWH = 3.0

# EQ (35) - C7 service quality.
# The formulation states QoE_{h,m}(q) >= QoE^min but never defines QoE. The
# definition below is the one addition to the constraint set and is chosen to
# be AFFINE in q so the programme stays linear and Propositions 1-3 survive:
#
#     QoE_{h,m}(q) = 1 - ( w_sh * E^sh_{h,m} + w_cu * E^cu_{h,m} ) / E^base_{h,m}
#
# i.e. a cap on the discomfort-weighted share of a consumer's monthly energy
# that may be displaced. Documented in docs/qoe_definition.md.
QOE_MIN = 0.85
QOE_WEIGHT_SHIFT = 0.4             # w_sh: shifting is less intrusive
QOE_WEIGHT_CURTAIL = 1.0           # w_cu: curtailment is fully felt

# =============================================================================
# 9. Priority pooling
# =============================================================================
# EQ (49) - beta_j(r), reproduced exactly from the formulation's table.
PRIORITY_WEIGHTS = {
    1: (0.55, 0.25, 0.10, 0.10),   # nominal
    2: (0.30, 0.30, 0.20, 0.20),   # stressed
    3: (0.15, 0.25, 0.25, 0.35),   # emergency
}
REGIME_NAMES = {1: "nominal", 2: "stressed", 3: "emergency"}

# EQ (50) - pool cut points. "mean_relative" is the rule as written
# (3/4 and 1/2 of the mean score). Formulation section 9.5 records that it
# degenerates on tight score distributions and names quantile cut points as
# the intended fix; both are implemented and both are reported.
POOL_CUT_RULE = "mean_relative"    # "mean_relative" | "quantile"
POOL_QUANTILES = (0.50, 0.75)

# EQ (53) - top-N pool size
POOL_N = 50

# =============================================================================
# 10. Cost allocation (Stage G)
# =============================================================================
# EQ (42) - pass-through share zeta_i in [0, 1]
ZETA_PASSTHROUGH = 0.5
# EQ (43) - surcharge cap. The revised formulation caps varpi_{i,m} at the
# LOWEST retail tariff among the households at i outright, "so no household's
# effective price is more than doubled". v3's delta^max multiplier is gone.
SURCHARGE_CAP_RULE = "min_retail_tariff"

# =============================================================================
# 11. Solver
# =============================================================================
# HiGHS via scipy.optimize.linprog. Chosen over CBC because it returns dual
# variables for inequality rows (res.ineqlin.marginals), and pi_i(t) - the
# dual of C1 - is not optional here: Proposition 2, the critical preference
# lambda*_g of (36), and the entire knowledge-graph explanation layer are
# built on it.
LP_SOLVER = "highs"
LP_OPTIONS = {"presolve": True, "time_limit": 600}


def derive_p_str(daily_peaks, quantile: float = 0.95) -> float:
    """EQ (15): stress threshold P^str_i from the empirical peak distribution."""
    import numpy as np
    return float(np.quantile(np.asarray(daily_peaks, dtype=float), quantile))


def sample_preferences(archetype: str, rng) -> dict:
    """
    Draw one household's own lambda_h, gamma^sh_h, gamma^cu_h and y^max_h.

    Implements Remark 4 directly: preferences are individual. The archetype
    supplies the distribution; the household carries the draw.
    """
    spec = CONSUMER_ARCHETYPES[archetype]
    out = {}
    for key in ("lambda_h", "gamma_sh_h", "gamma_cu_h", "y_max_h"):
        mean, sd = spec[key]
        lo, hi = PREFERENCE_BOUNDS[key]
        out[key] = float(min(max(rng.normal(mean, sd), lo), hi)) if sd > 0 \
            else float(mean)
    return out
