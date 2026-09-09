"""
GenTwin-SG :: Stage 0, tariff
=============================

Source document
---------------
Bangladesh Energy Regulatory Commission (BERC),
"বিদ্যুতের খুচরা মূল্যহার পুনর্নির্ধারণ সংক্রান্ত বিজ্ঞপ্তি" (Notification on
the re-fixing of retail electricity tariffs), memo 28.01.0000.012.04.003.26.2521,
dated 03 June 2026 / 20 Jyaistha 1433 BS. Signed by the Secretary, BERC.
Effective from the June 2026 billing month (clause 5).

Every rate in RETAIL_TARIFF below is transcribed from that order. Nothing here
is estimated. Where the pipeline needs a quantity the order does not define -
principally a residential time-of-use multiplier - it is derived from the
order's own ratios and flagged SCENARIO.

What the order settles for us
-----------------------------
1. LT-A (residential) is a pure cumulative-slab tariff with NO time-of-use
   component. This is not an omission in our reading; the order lists flat /
   off-peak / peak rows for LT-C1, LT-D3, LT-E and every MT, HT and EHT class,
   and pointedly does not for LT-A. Bangladesh has no residential ToU.

2. The classes that DO have ToU give us the multiplier empirically:

       LT-C1  peak 15.27 / flat 12.73 = 1.1995   off 11.45 / 12.73 = 0.8995
       LT-E   peak 18.45 / flat 15.36 = 1.2012   off 13.82 / 15.36 = 0.8997
       LT-D3  peak 14.20 / flat 11.36 = 1.2500   off 10.22 / 11.36 = 0.8997
       MT-1   peak 15.62 / flat 12.50 = 1.2496   off 11.25 / 12.50 = 0.9000
       MT-3   peak 16.06 / flat 12.85 = 1.2498   off 11.56 / 12.85 = 0.8996
       HT-2   peak 17.05 / flat 13.64 = 1.2500   off 12.28 / 13.64 = 0.9002

   Two clean families: LT classes at mu^pk = 1.20, MT/HT/EHT at 1.25, and a
   uniform mu^off = 0.90 across all of them. So mu(t) for the residential
   scenario is not invented - it is the LT-class ratio the regulator itself
   applied to every other low-tension consumer.

3. Peak and off-peak windows, from the order's footnotes 4-6:
       peak         17:00 - 23:00        (all ToU classes)
       off-peak     23:00 - 17:00        (all except LT-D3 / MT-7)
       LT-D3, MT-7  off-peak 23:00-05:00 and 09:00-17:00,
                    super off-peak 05:00-09:00

Peak window used by the model
-----------------------------
T^pk in equations (5), (32) and (34) is set from each substation's OBSERVED
peak period, not from the gazette's 17:00-23:00 window - per instruction.
The two disagree at Dhanmondi, which peaks at 13:00 on 63.8% of days. That
disagreement is a result, not a nuisance: it says the national ToU window is
misaligned with this substation's physical peak, and the model's mu(t) is
therefore a *proposed* substation-specific ToU tariff. The gazette window is
retained as OFFICIAL_TOU_WINDOWS so the ablation can be run either way.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import config as cfg

GAZETTE_REF = (
    "BERC memo 28.01.0000.012.04.003.26.2521, 03 June 2026; "
    "effective June 2026 billing month"
)

# =============================================================================
# 1. The gazette, transcribed
# =============================================================================
# energy rates in Tk/kWh, demand charges in Tk/kW/month.

#: LT-A residential cumulative slabs, (upper_bound_kwh, rate). None = no cap.
LT_A_SLABS = [
    (75, 6.18),      # প্রথম ধাপ    0-75
    (200, 8.50),     # দ্বিতীয় ধাপ  76-200
    (300, 9.10),     # তৃতীয় ধাপ    201-300
    (400, 9.62),     # চতুর্থ ধাপ    301-400
    (600, 15.01),    # পঞ্চম ধাপ    401-600
    (None, 17.35),   # ষষ্ঠ ধাপ     above 600
]

#: Lifeline rate. Footnote 3: available ONLY to consumers whose total monthly
#: consumption is <= 50 units. Exceed 50 and the ordinary slabs apply from the
#: first unit - this is not a slab, it is a separate qualifying tariff.
LT_A_LIFELINE_LIMIT_KWH = 50
LT_A_LIFELINE_RATE = 5.32

VAT_RATE = 0.05                    # clause on VAT, 5% of the energy charge
PREPAID_REBATE = 0.005             # clause 4: 0.5% rebate, net of VAT


@dataclass(frozen=True)
class TariffClass:
    code: str
    name_bn: str
    name_en: str
    voltage: str
    flat: float | None = None
    off_peak: float | None = None
    peak: float | None = None
    super_off_peak: float | None = None
    demand_charge: float | None = None
    slabs: list | None = None
    lifeline: bool = False
    tou_profile: str = "standard"   # "standard" | "ev" | "none"

    @property
    def has_tou(self) -> bool:
        return self.peak is not None and self.off_peak is not None

    def mu_peak(self) -> float | None:
        """Observed peak multiplier mu^pk = peak rate / flat rate."""
        if not self.has_tou or not self.flat:
            return None
        return self.peak / self.flat

    def mu_off(self) -> float | None:
        if not self.has_tou or not self.flat:
            return None
        return self.off_peak / self.flat


RETAIL_TARIFF: dict[str, TariffClass] = {
    # --- ক. নিম্নচাপ (LT): 230/400 V -----------------------------------------
    "LT-A": TariffClass(
        "LT-A", "আবাসিক", "Residential", "LT",
        demand_charge=42.00, slabs=LT_A_SLABS, lifeline=True,
        tou_profile="none",
    ),
    "LT-B": TariffClass(
        "LT-B", "সেচ/কৃষিকাজে ব্যবহৃত পাম্প", "Irrigation / agricultural pump",
        "LT", flat=6.04, demand_charge=42.00, tou_profile="none",
    ),
    "LT-C1": TariffClass(
        "LT-C1", "ক্ষুদ্র শিল্প", "Small industry", "LT",
        flat=12.73, off_peak=11.45, peak=15.27, demand_charge=48.00,
    ),
    "LT-C2": TariffClass(
        "LT-C2", "নির্মাণ", "Construction", "LT",
        flat=18.09, demand_charge=120.00, tou_profile="none",
    ),
    "LT-D1": TariffClass(
        "LT-D1", "শিক্ষা, ধর্মীয় ও দাতব্য প্রতিষ্ঠান এবং হাসপাতাল",
        "Education, religious and charitable institutions, and hospitals", "LT",
        flat=9.05, demand_charge=60.00, tou_profile="none",
    ),
    "LT-D2": TariffClass(
        "LT-D2", "রাস্তার বাতি ও পানির পাম্প", "Street lighting and water pumps",
        "LT", flat=11.46, demand_charge=90.00, tou_profile="none",
    ),
    "LT-D3": TariffClass(
        "LT-D3", "ইলেকট্রিক ভেহিকেল ও ব্যাটারি চার্জিং স্টেশন",
        "EV and battery charging station", "LT",
        flat=11.36, off_peak=10.22, super_off_peak=9.09, peak=14.20,
        demand_charge=90.00, tou_profile="ev",
    ),
    "LT-E": TariffClass(
        "LT-E", "বাণিজ্যিক ও অফিস", "Commercial and office", "LT",
        flat=15.36, off_peak=13.82, peak=18.45, demand_charge=90.00,
    ),
    "LT-T": TariffClass(
        "LT-T", "অস্থায়ী", "Temporary", "LT",
        flat=23.81, demand_charge=120.00, tou_profile="none",
    ),
    # --- খ. মধ্যমচাপ (MT): 11 kV ---------------------------------------------
    "MT-1": TariffClass("MT-1", "আবাসিক", "Residential", "MT",
                        flat=12.50, off_peak=11.25, peak=15.62, demand_charge=90.00),
    "MT-2": TariffClass("MT-2", "বাণিজ্যিক ও অফিস", "Commercial and office", "MT",
                        flat=13.93, off_peak=12.54, peak=17.41, demand_charge=90.00),
    "MT-3": TariffClass("MT-3", "শিল্প", "Industry", "MT",
                        flat=12.85, off_peak=11.56, peak=16.06, demand_charge=90.00),
    "MT-4": TariffClass("MT-4", "নির্মাণ", "Construction", "MT",
                        flat=17.16, off_peak=15.44, peak=21.45, demand_charge=120.00),
    "MT-5": TariffClass("MT-5", "সাধারণ", "General", "MT",
                        flat=12.58, off_peak=11.32, peak=15.92, demand_charge=90.00),
    "MT-6": TariffClass("MT-6", "অস্থায়ী", "Temporary", "MT",
                        flat=22.56, demand_charge=120.00, tou_profile="none"),
    "MT-7": TariffClass("MT-7", "ইলেকট্রিক ভেহিকেল ও ব্যাটারি চার্জিং স্টেশন",
                        "EV and battery charging station", "MT",
                        flat=11.31, off_peak=10.18, super_off_peak=9.05, peak=14.14,
                        demand_charge=90.00, tou_profile="ev"),
    "MT-8": TariffClass("MT-8", "সেচ/কৃষিকাজে ব্যবহৃত পাম্প",
                        "Irrigation / agricultural pump", "MT",
                        flat=7.38, off_peak=6.64, peak=9.23, demand_charge=90.00),
    # --- গ. উচ্চচাপ (HT): 33 kV ----------------------------------------------
    "HT-1": TariffClass("HT-1", "সাধারণ", "General", "HT",
                        flat=12.58, off_peak=11.28, peak=15.67, demand_charge=90.00),
    "HT-2": TariffClass("HT-2", "বাণিজ্যিক ও অফিস", "Commercial and office", "HT",
                        flat=13.64, off_peak=12.28, peak=17.05, demand_charge=90.00),
    "HT-3": TariffClass("HT-3", "শিল্প", "Industry", "HT",
                        flat=12.75, off_peak=11.47, peak=15.93, demand_charge=90.00),
    "HT-4": TariffClass("HT-4", "নির্মাণ", "Construction", "HT",
                        flat=15.96, off_peak=14.36, peak=19.95, demand_charge=90.00),
    # --- ঘ. অতি উচ্চচাপ (EHT): 132 / 230 kV ----------------------------------
    "EHT-1": TariffClass("EHT-1", "সাধারণ", "General", "EHT",
                         flat=12.66, off_peak=11.39, peak=15.82, demand_charge=90.00),
    "EHT-2": TariffClass("EHT-2", "সাধারণ", "General", "EHT",
                         flat=12.61, off_peak=11.35, peak=15.76, demand_charge=90.00),
}

# =============================================================================
# 2. ToU windows, from footnotes 4-6 of the order
# =============================================================================
# Expressed as sets of 30-minute slot indices, 0..47.

def _slots(start_h: float, end_h: float) -> set[int]:
    """Half-open [start, end) in hours -> slot indices, wrapping past midnight."""
    s, e = int(start_h * 2), int(end_h * 2)
    return set(range(s, e)) if s < e else set(range(s, 48)) | set(range(0, e))


#: Footnote 6: বিকাল ৫:০০ টা হতে রাত ১১:০০ টা - peak for all applicable classes.
OFFICIAL_TOU_PEAK = _slots(17, 23)
#: Footnote 4: রাত ১১:০০ টা হতে পরদিন বিকাল ৫:০০ টা - off-peak, standard classes.
OFFICIAL_TOU_OFFPEAK = _slots(23, 17)
#: Footnote 4/5, LT-D3 and MT-7 only.
EV_TOU_OFFPEAK = _slots(23, 5) | _slots(9, 17)
EV_TOU_SUPER_OFFPEAK = _slots(5, 9)

OFFICIAL_TOU_WINDOWS = {
    "standard": {"peak": OFFICIAL_TOU_PEAK, "off_peak": OFFICIAL_TOU_OFFPEAK},
    "ev": {"peak": OFFICIAL_TOU_PEAK, "off_peak": EV_TOU_OFFPEAK,
           "super_off_peak": EV_TOU_SUPER_OFFPEAK},
}

# =============================================================================
# 3. mu_i(t) - the ADAPTIVE LOCAL time-of-use multiplier, Eqs. (19) and (34)
# =============================================================================
# The mechanism has two halves with two different owners:
#
#   RATIOS   mu^pk = 1.20, mu^off = 0.90.  SOURCE: this gazette. They are the
#            LT-family peak/flat and off-peak/flat ratios the regulator itself
#            applied to LT-C1 (1.1995 / 0.8995) and LT-E (1.2012 / 0.8997).
#            LT-A carries no ToU row, so adopting the LT-family ratio for
#            residential is a documented SCENARIO - but it is the regulator's
#            own number, not an invented one.
#
#   WINDOW   T^pk_i.  SOURCE: derived per substation from its congestion
#            history by gentwin/peak_window.py. NOT from this gazette.
#
# So mu_i(t) is a congestion-aware activation schedule laid over the
# regulator's multipliers. It is NOT a replacement tariff, and no output of
# this model may present it as one. The national window survives in
# OFFICIAL_TOU_PEAK and drives the `--national` ablation.

MU_FAMILIES = {
    "LT": {"peak": 1.20, "off_peak": 0.90},
    "MT": {"peak": 1.25, "off_peak": 0.90},
    "HT": {"peak": 1.25, "off_peak": 0.90},
    "EHT": {"peak": 1.25, "off_peak": 0.90},
}
MU_RESIDENTIAL_IS_SCENARIO = True


def peak_slots_for(substation: str, adaptive: bool | None = None) -> tuple:
    """
    T^pk_i. Adaptive local window by default; the national gazette window when
    `adaptive` is False, which is the ablation arm.
    """
    adaptive = cfg.ADAPTIVE_LOCAL_TOU if adaptive is None else adaptive
    if not adaptive:
        return tuple(sorted(OFFICIAL_TOU_PEAK))
    from . import peak_window as pw
    return pw.get_peak_slots(substation)


def mu_profile(tariff_code: str, substation: str | None = None,
               peak_slots=None, adaptive: bool | None = None) -> list[float]:
    """
    EQ (19), (34): mu_i(t) over the 48 slots of a day.

        mu_i(t) = mu^pk   if t in T^pk_i
                  mu^off  otherwise

    Give either `substation` (window looked up, the normal path) or an
    explicit `peak_slots` (used by the verification harness and the ablation).
    """
    tc = RETAIL_TARIFF[tariff_code]
    fam = MU_FAMILIES[tc.voltage]
    if peak_slots is None:
        if substation is None:
            raise ValueError("give either substation or peak_slots")
        peak_slots = peak_slots_for(substation, adaptive)
    peak_slots = set(peak_slots)
    return [fam["peak"] if s in peak_slots else fam["off_peak"]
            for s in range(cfg.SLOTS_PER_DAY)]


def mu_peak_off(tariff_code: str) -> tuple[float, float]:
    """(mu^pk, mu^off) for the class - the two scalars Eq. (34) differences."""
    fam = MU_FAMILIES[RETAIL_TARIFF[tariff_code].voltage]
    return fam["peak"], fam["off_peak"]


# =============================================================================
# 4. Billing
# =============================================================================

def lt_a_energy_charge(kwh: float) -> tuple[float, list[dict]]:
    """
    Cumulative slab billing for LT-A, with the lifeline qualifying rule.

    Returns (charge_bdt, breakdown). The lifeline rate is NOT a first slab:
    footnote 3 makes it a separate tariff available only to consumers whose
    whole month falls at or below 50 units. A 51-unit consumer pays 6.18 on
    all 51 units, not 5.32 on the first 50.
    """
    if kwh <= 0:
        return 0.0, []
    if kwh <= LT_A_LIFELINE_LIMIT_KWH:
        charge = kwh * LT_A_LIFELINE_RATE
        return charge, [{"band": "lifeline 0-50", "kwh": kwh,
                         "rate": LT_A_LIFELINE_RATE, "charge": charge}]

    charge, remaining, lower = 0.0, kwh, 0.0
    breakdown = []
    for upper, rate in LT_A_SLABS:
        cap = float("inf") if upper is None else upper
        band = min(remaining, cap - lower)
        if band <= 0:
            break
        amount = band * rate
        charge += amount
        breakdown.append({
            "band": f"{int(lower) + 1}-{'above' if upper is None else upper}",
            "kwh": band, "rate": rate, "charge": amount,
        })
        remaining -= band
        lower = cap
        if remaining <= 0:
            break
    return charge, breakdown


def marginal_rate_lt_a(kwh: float) -> float:
    """
    pi^ret_h in equations (19) and (34): the rate the consumer faces on the
    NEXT kWh. Under a cumulative slab tariff this is the marginal rate, not
    the average - a household at 350 units/month decides against 9.62, not
    against its 8.4 Tk/kWh average bill. Using the average here would
    understate the private benefit term of Proposition 2.
    """
    if kwh <= LT_A_LIFELINE_LIMIT_KWH:
        return LT_A_LIFELINE_RATE
    lower = 0.0
    for upper, rate in LT_A_SLABS:
        cap = float("inf") if upper is None else upper
        if kwh <= cap:
            return rate
        lower = cap
    return LT_A_SLABS[-1][1]


def retail_rate(tariff_code: str, monthly_kwh: float | None = None) -> float:
    """
    pi^ret_h for any class. Slab classes use the marginal rate at the
    consumer's monthly consumption; flat and ToU classes use the flat rate,
    with mu(t) supplying the time variation.
    """
    tc = RETAIL_TARIFF[tariff_code]
    if tc.slabs is not None:
        if monthly_kwh is None:
            raise ValueError(f"{tariff_code} is a slab tariff; monthly_kwh required")
        return marginal_rate_lt_a(monthly_kwh)
    if tc.flat is None:
        raise ValueError(f"{tariff_code} has no flat rate")
    return tc.flat


def effective_price(tariff_code: str, slot: int, substation: str | None = None,
                    peak_slots=None, monthly_kwh: float | None = None,
                    surcharge_bdt_per_kwh: float = 0.0,
                    adaptive: bool | None = None) -> float:
    """
    EQ (19), substation-aware form:

        pi_h(t) = ( pi^ret_h + varpi_{i,m} ) * mu_i(t)             [Tk/kWh]

    This is the "C_i(t) = P_i(t) x mu_i(t)" term: P_i(t) is the price faced by
    a consumer at substation i - the regulated retail rate plus the Stage G
    surcharge - and mu_i(t) is the adaptive local multiplier. The whole product
    is a CONSUMER-FACING price and enters f_wel through J^h_1, Eq. (19).

    It does NOT enter f_cost, Eq. (22). See docs/adaptive_local_tou.md section
    5: c_{i,d} there is the operator's marginal SUPPLY cost in Tk/kWh, not a
    retail price, and multiplying it by a retail ToU multiplier would price the
    same congestion twice - once through the tariff and once through dispatch -
    and would misstate the shadow price pi_i(t) that every Proposition 2
    explanation is built on.

    `surcharge_bdt_per_kwh` is varpi_{i,m} from Eq. (42), computed from the
    previous month's completed solve. Zero in month one.
    """
    base = retail_rate(tariff_code, monthly_kwh)
    mu = mu_profile(tariff_code, substation=substation,
                    peak_slots=peak_slots, adaptive=adaptive)[slot]
    return (base + surcharge_bdt_per_kwh) * mu


def monthly_bill(tariff_code: str, monthly_kwh: float,
                 sanctioned_kw: float = 0.0, prepaid: bool = False) -> dict:
    """
    Full bill: energy charge + demand charge + 5% VAT, less the 0.5% prepaid
    rebate (clause 4, applied to the net bill excluding VAT).
    """
    tc = RETAIL_TARIFF[tariff_code]
    if tc.slabs is not None:
        energy, breakdown = lt_a_energy_charge(monthly_kwh)
    else:
        energy = monthly_kwh * tc.flat
        breakdown = [{"band": "flat", "kwh": monthly_kwh,
                      "rate": tc.flat, "charge": energy}]
    demand = sanctioned_kw * (tc.demand_charge or 0.0)
    net = energy + demand
    rebate = net * PREPAID_REBATE if prepaid else 0.0
    net -= rebate
    vat = net * VAT_RATE
    return {
        "tariff_class": tariff_code, "monthly_kwh": monthly_kwh,
        "energy_charge_bdt": energy, "demand_charge_bdt": demand,
        "prepaid_rebate_bdt": rebate, "vat_bdt": vat,
        "total_bdt": net + vat, "breakdown": breakdown,
        "effective_rate_bdt_per_kwh": (net + vat) / monthly_kwh if monthly_kwh else 0.0,
        "source": GAZETTE_REF,
    }


def mt5_residential_blend(monthly_kwh: float) -> dict:
    """
    Footnote 7: MT-5 consumers whose use is predominantly (~80%) residential -
    university halls, cantonment dormitories - are billed 20% at the MT-5
    energy rate (12.58), 72% at the mean of LT-A's third and fourth steps
    ((9.10 + 9.62)/2 = 9.36), and 8% at LT-A's sixth step (17.35).

    Relevant here because the Educational category around Dhanmondi includes
    exactly this kind of consumer.
    """
    r_mt5 = RETAIL_TARIFF["MT-5"].flat
    r_mid = (LT_A_SLABS[2][1] + LT_A_SLABS[3][1]) / 2      # 9.36
    r_top = LT_A_SLABS[5][1]                                # 17.35
    parts = [(0.20, r_mt5), (0.72, r_mid), (0.08, r_top)]
    charge = sum(share * monthly_kwh * rate for share, rate in parts)
    return {
        "energy_charge_bdt": charge,
        "blended_rate_bdt_per_kwh": charge / monthly_kwh if monthly_kwh else 0.0,
        "components": [{"share": s, "rate": r} for s, r in parts],
        "note": "BERC 03 June 2026 order, footnote 7",
    }


def tariff_for_archetype(archetype: str) -> str:
    """Map a GenTwin consumer archetype to its gazette tariff class."""
    return cfg.CONSUMER_ARCHETYPES[archetype]["tariff_class"]


# =============================================================================
# 5. Self-check
# =============================================================================

def verify() -> dict:
    """
    Reproduce the gazette's own ToU ratios from the transcribed rates. If the
    transcription were wrong, these would not land on the two clean families
    the order uses, so this doubles as a proofreading test of the table.
    """
    report = {"observed_ratios": [], "families": MU_FAMILIES, "checks": []}
    for code, tc in RETAIL_TARIFF.items():
        if tc.has_tou:
            report["observed_ratios"].append({
                "class": code, "voltage": tc.voltage,
                "mu_peak": round(tc.mu_peak(), 4),
                "mu_off": round(tc.mu_off(), 4),
            })

    def _check(name, ok, detail=""):
        report["checks"].append({"check": name, "pass": bool(ok), "detail": detail})

    lt = [r for r in report["observed_ratios"] if r["voltage"] == "LT"
          and r["class"] != "LT-D3"]
    mt = [r for r in report["observed_ratios"] if r["voltage"] in ("MT", "HT", "EHT")
          and r["class"] != "MT-7"]
    _check("LT peak ratios ~ 1.20",
           all(abs(r["mu_peak"] - 1.20) < 0.01 for r in lt),
           str([r["mu_peak"] for r in lt]))
    _check("MT/HT/EHT peak ratios ~ 1.25",
           all(abs(r["mu_peak"] - 1.25) < 0.02 for r in mt),
           str([r["mu_peak"] for r in mt]))
    _check("off-peak ratio ~ 0.90 everywhere",
           all(abs(r["mu_off"] - 0.90) < 0.01 for r in report["observed_ratios"]),
           str(sorted({r["mu_off"] for r in report["observed_ratios"]})))
    _check("LT-A has no ToU", not RETAIL_TARIFF["LT-A"].has_tou)

    # Slab arithmetic, hand-checked.
    #   50 units  -> lifeline: 50 * 5.32 = 266.00
    #   51 units  -> ordinary: 51 * 6.18 = 315.18  (lifeline lost entirely)
    #   250 units -> 75*6.18 + 125*8.50 + 50*9.10 = 463.5 + 1062.5 + 455 = 1981.00
    for kwh, want in ((50, 266.00), (51, 315.18), (250, 1981.00)):
        got, _ = lt_a_energy_charge(kwh)
        _check(f"LT-A energy charge at {kwh} kWh", abs(got - want) < 0.01,
               f"got {got:.2f}, expected {want:.2f}")

    _check("marginal rate at 350 kWh is 4th step (9.62)",
           abs(marginal_rate_lt_a(350) - 9.62) < 1e-9)
    # 0.20*12.58 + 0.72*9.36 + 0.08*17.35 = 2.5160 + 6.7392 + 1.3880 = 10.6432
    _check("footnote-7 blend = 10.6432 Tk/kWh",
           abs(mt5_residential_blend(1000)["blended_rate_bdt_per_kwh"] - 10.6432) < 0.01,
           f"{mt5_residential_blend(1000)['blended_rate_bdt_per_kwh']:.4f}")

    # Official windows partition the day for standard classes.
    _check("official peak + off-peak tile the 48 slots",
           len(OFFICIAL_TOU_PEAK | OFFICIAL_TOU_OFFPEAK) == 48
           and not (OFFICIAL_TOU_PEAK & OFFICIAL_TOU_OFFPEAK),
           f"peak={len(OFFICIAL_TOU_PEAK)} slots, off={len(OFFICIAL_TOU_OFFPEAK)} slots")
    _check("EV windows tile the 48 slots",
           len(OFFICIAL_TOU_PEAK | EV_TOU_OFFPEAK | EV_TOU_SUPER_OFFPEAK) == 48)

    report["all_pass"] = all(c["pass"] for c in report["checks"])
    return report
