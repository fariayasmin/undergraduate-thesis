"""
04_load_generator.py — bottom-up 30-min load generation for the Kalyanpur
and Dhanmondi substations, 2024-01-01 → 2026-06-28.

Three layers
------------
1. RESIDENTIAL COHORTS  — for each socioeconomic tier, N_ARCHETYPES households
   are simulated at APPLIANCE level (Monte-Carlo); their mean profile is
   normalised to the tier's design coincident peak (ADMD) and multiplied by
   the tier's household COUNT from the feeder composition. This is standard
   cohort aggregation — composition-based, not profile scaling.
2. NON-RESIDENTIAL UNITS — shops, offices, schools, clinics, named Dhanmondi
   hospitals, workshops, street lights, WASA pumps, mosques; each has a
   physically-motivated daily shape, day-type behaviour (Eid closures etc.)
   and a temperature-driven AC component.
3. DETAILED SAMPLE — N_DETAILED_HH smart-metered study households per
   substation, simulated individually at appliance level (for NILM / MOLP /
   QoE work). Their sum is NOT the substation total; they are a metered
   sample of the residential cohorts.

Temperature response (fans / AC / fridge / geyser) is computed INSIDE this
generator from weather temperature_c only — no multiplier attributes leak
into the datasets (per supervisor feedback: keep attributes minimal).

Outputs (data/):
  04_<sub>_aggregate_30min.csv          category-wise kW + total MW
  04_<sub>_households_30min.csv         120 detailed households (kW, wide)
  04_<sub>_household_inventory.csv      appliance inventory (project format)
  04_<sub>_appliance_sample_7days.csv   appliance-level, 10 HH × 7 days (long)
"""

import hashlib
import json
import numpy as np
import pandas as pd
import config as C

rng_global = np.random.default_rng(C.SEED)


def stable_seed(*parts) -> int:
    """Deterministic seed from string parts.

    Python's built-in hash() is salted per interpreter process (PEP 456), so
    hash((substation, tier, SEED)) returns a different value on every run and
    the cohort draw is NOT reproducible. md5 is stable across processes,
    platforms and Python versions; it is used here purely as a deterministic
    string-to-int map, not for any security purpose.
    """
    digest = hashlib.md5("|".join(map(str, parts)).encode("utf-8")).hexdigest()
    return int(digest[:8], 16)

# ── Load shared inputs ────────────────────────────────────────────────────────
CAL = pd.read_csv(C.DATA_DIR / "00_bd_calendar_30min.csv",
                  parse_dates=["timestamp"])
WX  = pd.read_csv(C.DATA_DIR / "01_dhaka_weather_30min.csv",
                  parse_dates=["timestamp"])
with open(C.BASE_DIR / "02_appliance_catalogue.json") as f:
    CATALOGUE = {k: v for k, v in json.load(f).items() if not k.startswith("_")}

T          = len(CAL)                                   # 17,568
TS         = CAL["timestamp"].values
HOUR_F     = CAL["hour"].values + CAL["minute"].values / 60.0
DOW        = CAL["day_of_week"].values
IS_WEEKEND = CAL["is_weekend"].values.astype(bool)
IS_RAMADAN = CAL["is_ramadan"].values.astype(bool)
DAY_TYPE   = CAL["day_type"].values
IS_EID     = DAY_TYPE == "Eid"
IS_HOLIDAY = CAL["is_holiday"].values.astype(bool)
MONTH      = CAL["timestamp"].dt.month.values
TEMP       = WX["temperature_c"].values
DAY_IDX    = (CAL["timestamp"].dt.dayofyear - 1).values   # 0..365
YEAR       = CAL["timestamp"].dt.year.values

DUSK_BY_MONTH = {1:17.4, 2:17.7, 3:18.1, 4:18.4, 5:18.6, 6:18.8,
                 7:18.8, 8:18.6, 9:18.2, 10:17.7, 11:17.3, 12:17.2}
DUSK = np.array([DUSK_BY_MONTH[m] for m in MONTH])

def win(lo, hi):
    """Boolean mask for hour-of-day window [lo, hi); supports wrap-around."""
    if lo <= hi:
        return (HOUR_F >= lo) & (HOUR_F < hi)
    return (HOUR_F >= lo) | (HOUR_F < hi)

SIG = lambda x: 1.0 / (1.0 + np.exp(-x))

# occupancy uplift for daytime on weekends/holidays
DAY_OCC = np.where((IS_WEEKEND | IS_HOLIDAY) & win(9, 17), 1.25, 1.0)


# ══════════════════════════════════════════════════════════════════════════════
# 1. RESIDENTIAL — appliance-level probability model
# ══════════════════════════════════════════════════════════════════════════════

def usage_probability(usage: str) -> np.ndarray:
    """Per-slot ON-probability template for one appliance unit (before duty)."""
    p = np.zeros(T)

    if usage == "always_on":
        p[:] = 1.0

    elif usage == "comfort":                      # fans
        base = np.clip((TEMP - 23.0) / 8.5, 0.02, 0.98)
        occ  = np.where(win(22, 7), 0.9, np.where(win(18, 22), 0.85, 0.5))
        p = base * occ * DAY_OCC

    elif usage == "cooling":                      # split AC
        need = SIG((TEMP - 28.0) / 1.5)
        slot = np.where(win(21, 7), 1.0,
               np.where(win(13, 17), 0.55, 0.18))
        p = need * slot

    elif usage == "lighting":
        p = np.where(win(5, 6.5), 0.40, 0.0)
        p = np.where((HOUR_F >= DUSK) & (HOUR_F < 23.5), 0.80, p)
        p = np.where(win(23.5, 5), 0.12, p)
        # Ramadan: pre-dawn sehri lighting
        p = np.where(IS_RAMADAN & win(3.0, 4.75), 0.75, p)

    elif usage == "entertainment":                # TV
        p = np.where(win(18, 23.5), 0.65, 0.0)
        p = np.where((IS_WEEKEND | IS_HOLIDAY) & win(13, 17), 0.35, p)

    elif usage in ("cooking", "cooking_short"):
        p = np.where(win(6.5, 8.5), 0.35, 0.0)
        p = np.where(win(11.5, 13.5), 0.50, p)
        p = np.where(win(19.0, 21.0), 0.60, p)
        # Ramadan: lunch suppressed, sehri + iftar preparation added
        p = np.where(IS_RAMADAN & win(11.5, 13.5), 0.05, p)
        p = np.where(IS_RAMADAN & win(3.0, 4.5), 0.55, p)
        p = np.where(IS_RAMADAN & win(16.5, 18.75), 0.70, p)
        if usage == "cooking_short":
            p = p * 0.7

    elif usage == "chore_morning":                # iron, washing machine
        wd = np.where(win(8, 11), 0.12, 0.0)
        we = np.where(win(9, 13), 0.20, 0.0)
        p  = np.where(IS_WEEKEND | IS_HOLIDAY, we, wd)

    elif usage == "pump":
        p = np.where(win(6.5, 8.5), 0.60, 0.0)
        p = np.where(win(17.5, 19.0), 0.40, p)

    elif usage == "evening_charge":               # IPS battery charging
        p = np.where(win(18, 23), 0.50, np.where(win(23, 2), 0.20, 0.0))

    elif usage == "winter_morning":               # geyser
        cold = (TEMP < 21.0) & np.isin(MONTH, [11, 12, 1, 2])
        p = np.where(cold & win(6, 9), 0.50, 0.0)
        p = np.where(cold & win(19, 21), 0.15, p)

    elif usage == "evening_work":                 # computer / laptop
        p = np.where(win(9, 17), 0.30, 0.0)
        p = np.where(win(19, 23.5), 0.50, p)

    return np.clip(p, 0.0, 1.0)


SUBSLOT_FRAC = {"cooking_short": 0.18, "cooking": 0.70, "chore_morning": 0.80,
                "winter_morning": 0.75, "pump": 0.85}

# probability templates are appliance-type level → precompute once
P_TEMPLATE = {name: usage_probability(spec["usage"])
              for name, spec in CATALOGUE.items()}


def simulate_household(tier: str, rng: np.random.Generator,
                       collect_appliances: bool = False):
    """
    Simulate one household for the full year at 30-min resolution.
    Returns (total_w[T], inventory_rows, appliance_traces{aid: w[T]} | None)
    """
    behaviour = rng.normal(1.0, 0.12)             # household-level intensity
    total = np.zeros(T, dtype=np.float32)
    cat_loads = {"Critical": np.zeros(T, np.float32),
                 "Schedulable": np.zeros(T, np.float32),
                 "Curtailable": np.zeros(T, np.float32)}
    inv_rows, traces = [], ({} if collect_appliances else None)

    # Eid exodus: 35 % of households leave Dhaka during Eid spans
    away = rng.random() < 0.35
    eid_factor = np.where(IS_EID, 0.25 if away else 1.05, 1.0).astype(np.float32)

    for name, spec in CATALOGUE.items():
        if rng.random() > spec["ownership_prob"][tier]:
            continue
        qm, qs = spec["quantity_dist"][tier]
        qty = max(1, int(round(rng.normal(qm, qs))))
        actual_w = max(5.0, rng.normal(spec["power_w"], spec["power_std_w"]))
        duty     = spec["duty_cycle"]
        frac     = SUBSLOT_FRAC.get(spec["usage"], 1.0)

        p = np.clip(P_TEMPLATE[name] * behaviour, 0, 1)

        if spec["usage"] == "always_on":
            d = duty
            if spec.get("temp_driven"):           # fridge/freezer compressor
                d = duty * (1 + 0.02 * (TEMP - 25.0))
            load = qty * actual_w * np.clip(d, 0.1, 0.95) \
                   * (1 + rng.normal(0, 0.03, T))
        elif spec["usage"] == "lighting":
            on_units = qty * p * (1 + rng.normal(0, 0.10, T))
            load = np.clip(on_units, 0, qty) * actual_w
        else:
            on_units = rng.binomial(qty, p)       # vectorised over T
            load = on_units * actual_w * duty * frac

        load = (load * eid_factor).astype(np.float32)
        # Critical appliances keep running even when family is away
        if away and spec["category"] != "Critical":
            pass                                   # eid_factor already applied
        elif away and spec["category"] == "Critical":
            load = np.where(IS_EID, load / 0.25 * 1.0, load).astype(np.float32)

        total += load
        cat_loads[spec["category"]] += load
        if collect_appliances:
            traces[name] = traces.get(name, np.zeros(T, np.float32)) + load
        inv_rows.append({
            "appliance_name": name, "unit_index": 1, "category": spec["category"],
            "rated_power_w": spec["power_w"],
            "actual_power_w": round(actual_w, 1), "duty_cycle": duty,
            "quantity": qty,
        })
    return total, inv_rows, traces, cat_loads


def build_tier_cohort_profile(tier: str, substation: str) -> np.ndarray:
    """Mean per-household profile (kW) normalised to the tier's design ADMD."""
    rng = np.random.default_rng(stable_seed(C.SEED, substation, tier))
    acc = np.zeros(T, dtype=np.float64)
    for _ in range(C.N_ARCHETYPES):
        w, _, _, _ = simulate_household(tier, rng)
        acc += w
    prof_kw = acc / C.N_ARCHETYPES / 1000.0
    prof_kw = pd.Series(prof_kw).rolling(3, center=True, min_periods=1).mean().to_numpy().copy()

    daily_max = prof_kw.reshape(-1, C.SLOTS_PER_DAY).max(axis=1)
    prof_kw *= C.TIER_COINCIDENT_PEAK_KW[tier] / daily_max.mean()
    return prof_kw


# ══════════════════════════════════════════════════════════════════════════════
# 2. NON-RESIDENTIAL unit shapes (kW per unit, normalised to design peak)
# ══════════════════════════════════════════════════════════════════════════════

def ac_factor(strength: float) -> np.ndarray:
    return 1.0 + strength * SIG((TEMP - 28.5) / 1.7)


def cooling_strength(subcat: str) -> float:
    """Cooling response of a non-residential class, times the global
    calibration multiplier C.NONRES_COOLING_SCALE. Named hospitals share the
    generic 'hospital' entry."""
    key = "hospital" if str(subcat).startswith("hospital") else str(subcat)
    return C.NONRES_COOLING_STRENGTH.get(key, 0.0) * C.NONRES_COOLING_SCALE

def _norm(shape: np.ndarray, peak_kw: float, rng) -> np.ndarray:
    shape = shape * (1 + rng.normal(0, 0.04, T))
    daily_max = shape.reshape(-1, C.SLOTS_PER_DAY).max(axis=1)
    return shape * (peak_kw / daily_max.mean())

def unit_profile(subcat: str, peak_kw: float, rng) -> np.ndarray:
    if subcat == "commercial_shop":
        a = np.where(win(10, 17), 0.55, 0.12)
        a = np.where(win(17, 21.5), 1.0, a)
        a *= np.where(IS_EID, 0.20, np.where(DAY_TYPE == "Public Holiday", 0.70,
             np.where(DAY_TYPE == "Durga Puja", 0.85, 1.0)))
        a = a * ac_factor(cooling_strength(subcat))
        # Ramadan: shops stay open late for iftar/Eid shopping
        a = np.where(IS_RAMADAN & win(19, 23), a * 1.25, a)

    elif subcat == "office_smb":
        openh = win(9, 17) & ~IS_WEEKEND & ~IS_HOLIDAY
        a = np.where(openh, 1.0, 0.08) * ac_factor(cooling_strength(subcat))

    elif subcat == "school":
        openh = win(8, 13.5) & ~IS_WEEKEND & ~IS_HOLIDAY
        a = np.where(openh, 1.0, 0.05) * ac_factor(cooling_strength(subcat))

    elif subcat.startswith("hospital") or subcat == "clinic_small":
        a = np.where(win(8, 14), 1.0, np.where(win(14, 21), 0.78, 0.55))
        a = a * ac_factor(cooling_strength(subcat))                    # hospitals never close

    elif subcat == "workshop":
        openh = win(9, 18) & (DOW != 4) & ~IS_EID  # closed Friday + Eid
        a = np.where(openh, 1.0, 0.07)

    elif subcat == "streetlight_block":
        a = np.where((HOUR_F >= DUSK) | (HOUR_F < 5.5), 1.0, 0.0)

    elif subcat == "wasa_pump":
        a = np.where(win(4, 9), 1.0, np.where(win(17, 21), 0.8,
            np.where(win(23, 2), 0.5, 0.2)))

    elif subcat == "mosque":
        a = np.full(T, 0.08)
        for lo, hi, v in [(4.9, 5.9, 0.5), (13.0, 14.0, 0.5),
                          (16.5, 17.5, 0.4)]:
            a = np.where(win(lo, hi), v, a)
        a = np.where((HOUR_F >= DUSK) & (HOUR_F < DUSK + 1.0), 0.6, a)
        a = np.where((DOW == 4) & win(12.5, 14.0), 1.0, a)          # Jummah
        a = np.where(IS_RAMADAN & (HOUR_F >= DUSK + 1.0)
                     & (HOUR_F < DUSK + 3.0), 1.0, a)               # Tarawih
        a = a * ac_factor(cooling_strength(subcat))
    else:
        raise ValueError(subcat)

    return _norm(a.astype(np.float64), peak_kw, rng)


# ══════════════════════════════════════════════════════════════════════════════
# 3. Per-substation build
# ══════════════════════════════════════════════════════════════════════════════

def build_substation(substation: str):
    print(f"\n{'='*60}\n{substation}\n{'='*60}")
    comp = pd.read_csv(C.DATA_DIR / f"03_{substation.lower()}_composition.csv")
    target = C.BPDB_MEAN_DAILY_PEAK_MW[substation]

    # ── Pass 1: build per-unit profiles for every composition row ────────────
    components = []            # (column, subcategory, units, per_unit_profile_kw)
    for tier in C.TIER_MIX[substation]:
        n = int(comp.loc[(comp["category"] == "residential") &
                         (comp["subcategory"] == tier), "units"].iloc[0])
        prof = build_tier_cohort_profile(tier, substation)
        components.append((f"residential_{tier}_kw", "residential", tier, n, prof))

    rng = np.random.default_rng(C.SEED + 7)
    for _, r in comp[comp["category"] != "residential"].iterrows():
        prof = unit_profile(r["subcategory"], r["per_unit_peak_kw"], rng)
        components.append((f"{r['category']}_kw", r["category"],
                           r["subcategory"], int(r["units"]), prof))

    # ── Pass 2: calibrate UNIT COUNTS per YEAR (not output scaling).
    #    (a) category peaks do not coincide in time → first pass undershoots;
    #    (b) the customer base evolves year to year (new connections, feeder
    #        transfers — e.g. Dhanmondi 2026 load moved to adjacent feeders).
    #    Both are represented by year-wise customer counts.
    #
    #    NAMED INSTITUTIONS ARE NOT SCALABLE. The Dhanmondi hospitals are real,
    #    individually identified sites; there cannot be 1.3 of Labaid. Under
    #    the previous formulation they entered the multiplier as
    #    max(1, round(1*k)) == 1, so they silently absorbed none of the
    #    calibration and the substation systematically undershot its target by
    #    their share (~12 MW of 119.8 MW at Dhanmondi). They are now held at a
    #    fixed count and k is solved over the SCALABLE population only, so the
    #    calibration still adjusts counts and never rescales a load profile.
    is_fixed = lambda subcat: str(subcat).startswith("hospital::")
    fixed_w = np.zeros(T)
    scalable_w = np.zeros(T)
    for _, _, subcat, n, prof in components:
        (fixed_w if is_fixed(subcat) else scalable_w).__iadd__(n * prof)

    day_year = YEAR.reshape(-1, C.SLOTS_PER_DAY)[:, 0]

    def mean_daily_peak(k: float, ymask: np.ndarray) -> float:
        dm = (fixed_w + k * scalable_w).reshape(
            -1, C.SLOTS_PER_DAY).max(axis=1) / 1000.0
        return dm[ymask].mean()

    def solve_k(target_mw: float, ymask: np.ndarray) -> float:
        """Bisect for the count multiplier on the scalable population.
        mean_daily_peak is monotone increasing in k, so this is well posed."""
        lo, hi = 1e-3, 10.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if mean_daily_peak(mid, ymask) < target_mw:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    targets = C.bpdb_yearly_targets(substation)

    # Anchor each year's multiplier at the centroid of that year's coverage and
    # interpolate linearly between anchors (flat outside the first/last), so the
    # customer base evolves continuously.  A step function at 1 January implied
    # e.g. ~4,000 Dhanmondi customers disconnecting overnight (86.1 -> 68.6 MW
    # between 2025-12-31 and 2026-01-01), a discontinuity a forecaster would
    # learn as a structural break.  The genuine year-on-year level change is
    # preserved: Dhanmondi's Jan-Jun mean really does fall ~19% from 2025 to
    # 2026 in the BPDB record.
    #
    # Smoothing changes each year's mean, so the anchors are refined by fixed
    # point until every year still reproduces its BPDB target.
    day_index = np.arange(len(day_year), dtype=float)
    anchor_x = np.array([day_index[day_year == y].mean() for y in C.YEARS])
    anchor_k = np.array([1.0 for _ in C.YEARS])
    tgt_vec = np.array([targets.get(y, target) for y in C.YEARS])

    # start from the piecewise-constant solution, then relax
    for i, y in enumerate(C.YEARS):
        anchor_k[i] = solve_k(tgt_vec[i], day_year == y)

    for _ in range(40):
        k_day = np.interp(day_index, anchor_x, anchor_k)
        dm = (fixed_w + np.repeat(k_day, C.SLOTS_PER_DAY) * scalable_w).reshape(
            -1, C.SLOTS_PER_DAY).max(axis=1) / 1000.0
        achieved = np.array([dm[day_year == y].mean() for y in C.YEARS])
        if np.max(np.abs(achieved - tgt_vec)) < 1e-3:
            break
        anchor_k *= (tgt_vec / achieved) ** 0.7      # damped fixed point

    k_day = np.interp(day_index, anchor_x, anchor_k)
    k_year = {y: float(anchor_k[i]) for i, y in enumerate(C.YEARS)}
    for i, y in enumerate(C.YEARS):
        print(f"  {y}: achieved {mean_daily_peak(1.0, day_year == y):.1f} MW → "
              f"target {tgt_vec[i]:.1f} MW  (anchor multiplier "
              f"k={anchor_k[i]:.3f} on scalable customers; named institutions "
              f"fixed; interpolated between anchors)")
    k_slot = np.repeat(k_day, C.SLOTS_PER_DAY)         # per-slot multiplier

    agg = pd.DataFrame({"timestamp": TS})
    cal_rows = []
    for col, cat, subcat, n, prof in components:
        fixed = is_fixed(subcat)
        k_eff = np.ones(T) if fixed else k_slot
        n_cal_slot = np.maximum(1, np.round(n * k_eff)).astype(np.int32)
        contrib = np.round(prof * n_cal_slot, 1)
        agg[col] = (agg[col] + contrib) if col in agg.columns else contrib
        pk = prof.reshape(-1, C.SLOTS_PER_DAY).max(axis=1).mean()
        row = {"substation": substation, "category": cat,
               "subcategory": subcat, "units_initial": n,
               "scalable": not fixed,
               "per_unit_mean_daily_peak_kw": round(pk, 3)}
        for y in C.YEARS:
            ky = 1.0 if fixed else k_year[y]
            row[f"units_{y}"] = max(1, int(round(n * ky)))
            row[f"k_{y}"] = round(ky, 4)
        cal_rows.append(row)
        label = f"{cat}/{subcat}" if cat == "residential" else subcat
        counts = " → ".join(str(row[f"units_{y}"]) for y in C.YEARS)
        print(f"  {label:<38} units {counts}")

    pd.DataFrame(cal_rows).to_csv(
        C.DATA_DIR / f"03b_{substation.lower()}_composition_calibrated.csv",
        index=False)

    kw_cols = [c for c in agg.columns if c.endswith("_kw")]
    agg["total_mw"] = np.round(agg[kw_cols].sum(axis=1) / 1000.0, 3)
    out = C.DATA_DIR / f"04_{substation.lower()}_aggregate_30min.csv"
    agg.to_csv(out, index=False)
    dp = agg["total_mw"].values.reshape(-1, C.SLOTS_PER_DAY).max(axis=1)
    print(f"  TOTAL mean daily peak: {dp.mean():.1f} MW "
          f"(BPDB target {target} MW)")
    print(f"  written: {out}")

    build_detailed_households(substation)


def build_detailed_households(substation: str):
    """120 smart-metered study households + inventory + appliance sample."""
    mix   = C.TIER_MIX[substation]
    tiers = (["lower_middle"] * int(round(C.N_DETAILED_HH * mix["lower_middle"]))
           + ["middle"]       * int(round(C.N_DETAILED_HH * mix["middle"])))
    tiers += ["upper_middle"] * (C.N_DETAILED_HH - len(tiers))
    prefix = "K" if substation == "Kalyanpur" else "D"

    rng = np.random.default_rng(C.SEED + (1 if substation == "Kalyanpur" else 2))
    hh_loads, inv_all, sample_traces, cat_store = {}, [], {}, {}

    for i, tier in enumerate(tiers, start=1):
        hh_id = f"{prefix}HH{i:04d}"
        collect = i <= 10                       # appliance-level sample HHs
        w, inv_rows, traces, cat_loads = simulate_household(tier, rng, collect)
        hh_loads[hh_id] = np.round(w / 1000.0, 4)          # kW
        cat_store[hh_id] = cat_loads
        lo, hi = C.TIER_INCOME_BDT[tier]
        income = int(rng.integers(lo, hi))
        size   = int(np.clip(rng.normal(4.2, 1.3), 1, 9))
        for j, r in enumerate(inv_rows, start=1):
            inv_all.append({
                "household_id": hh_id, "substation": substation, "tier": tier,
                "household_size": size, "monthly_income_bdt": income,
                **{k: r[k] for k in ("appliance_name", "unit_index", "category",
                                     "rated_power_w", "actual_power_w",
                                     "duty_cycle", "quantity")},
                "meter_id": f"{prefix}SM{i:04d}",
                "appliance_id": f"{prefix}SM{i:04d}_{r['appliance_name']}_1",
            })
        if collect:
            sample_traces[hh_id] = traces

    wide = pd.DataFrame({"timestamp": TS, **hh_loads})
    out1 = C.DATA_DIR / f"04_{substation.lower()}_households_30min.csv"
    wide.to_csv(out1, index=False)

    np.savez_compressed(
        C.DATA_DIR / f"04_{substation.lower()}_hh_category_loads.npz",
        household_ids=np.array(list(hh_loads.keys())),
        tiers=np.array(tiers),
        critical_w=np.stack([cat_store[h]["Critical"] for h in hh_loads], 1),
        schedulable_w=np.stack([cat_store[h]["Schedulable"] for h in hh_loads], 1),
        curtailable_w=np.stack([cat_store[h]["Curtailable"] for h in hh_loads], 1),
    )

    inv = pd.DataFrame(inv_all)
    out2 = C.DATA_DIR / f"04_{substation.lower()}_household_inventory.csv"
    inv.to_csv(out2, index=False)

    # 7-day appliance-level long sample (11–17 Mar: includes Ramadan start)
    mask = (CAL["timestamp"] >= "2024-03-11") & (CAL["timestamp"] < "2024-03-18")
    rows = []
    for hh_id, traces in sample_traces.items():
        for name, w in traces.items():
            spec = CATALOGUE[name]
            rows.append(pd.DataFrame({
                "timestamp": TS[mask.values], "household_id": hh_id,
                "appliance_name": name, "category": spec["category"],
                "power_w": np.round(w[mask.values], 1),
            }))
    sample = pd.concat(rows, ignore_index=True)
    out3 = C.DATA_DIR / f"04_{substation.lower()}_appliance_sample_7days.csv"
    sample.to_csv(out3, index=False)

    print(f"  detailed households: {out1.name}, {out2.name}, {out3.name}")


def main():
    for sub in C.SUBSTATIONS:
        build_substation(sub)


if __name__ == "__main__":
    main()