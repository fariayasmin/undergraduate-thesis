"""
GenTwin-SG :: Stage C/D - forecast to kappa
===========================================

Implements Eqs. (13)-(16) of GenTwinSG_Formulation_v3: forecast error scale,
the robust peak, the stress indicator and deficit, and the scaling factor that
carries the forecast into the optimisation.

The forecasting model is FROZEN. This module only reads what
load_prediction.py produced. Nothing here retrains, reweights or re-selects.

-----------------------------------------------------------------------------
NOTATION - three symbols that are easy to confuse, kept apart on purpose
-----------------------------------------------------------------------------

    kappa^scale_{i,d+k|d}   ROBUST SCALING FACTOR, Eq. (16)

                                kappa^scale = p~_{i,d+k|d} / p_{i,d}

                            Rescales the decision-dependent load inside C1
                            (27) and C4 (30). Dimensionless, near 1.0.
                            Code: `kappa_scale`.

    y^max (y^max_h)         CURTAILMENT CAP, Eq. (34). Upper bound on
                            y_{h,t}, per household by the approved C6
                            extension. Never appears in this module.

    z_beta                  QUANTILE MULTIPLIER on sigma in Eq. (14).
                            1.645 at beta = 0.95. NOT a kappa.

Equation (14), as the formulation writes it:

    p~_{i,d+k|d}  =  p_hat_{i,d+k|d}  +  z_beta * sigma_{i,k}            (14)

Why it must not be written with kappa in place of z_beta
--------------------------------------------------------
The form  p~ = p_hat + kappa * sigma  is CIRCULAR. Eq. (16) *defines*
kappa^scale as p~ / p_{i,d}, so p~ would appear on both sides of its own
definition. Solving the fixed point gives

    p~ = p_hat / (1 - sigma / p_{i,d})

which is a different quantity, is undefined at sigma = p_{i,d}, and changes
sign for sigma > p_{i,d}. At Dhanmondi (p_{i,d} = 136.5 MW, sigma_14 = 23.5 MW)
it would inflate the day+14 bound by ~21% over Eq. (14) for no stated reason;
at a substation with a low reference peak it would diverge.

It would also break Assumption 2. The formulation requires kappa to be
"computed before the solve and held fixed"; a self-referential kappa is a fixed
point of the forecast layer, and the exogeneity argument that keeps C1 linear
no longer reads cleanly.

The two symbols have different jobs. z_beta sets HOW CONSERVATIVE the bound is
(a confidence choice). kappa^scale converts that bound into a MULTIPLIER on the
reference-day load profile (a units conversion). This module implements (14)
and (16) exactly as written.

-----------------------------------------------------------------------------
sigma_{i,k}: what we have, and what it actually is
-----------------------------------------------------------------------------
Eq. (13) asks for sigma^2 = Var(p_{i,d+k} - p_hat_{i,d+k|d}). What the frozen
pipeline exports is the rolling-origin recursive backtest SUMMARY: RMSE by
horizon over ~37 origins per horizon. Since

    RMSE_k = sqrt( Var_k + bias_k^2 )  >=  sigma_k

using RMSE as sigma is CONSERVATIVE - it can only widen p~, the safe direction
for a robust bound. Separating variance from bias needs the per-record
backtest; if that is supplied, `sigma_from_residuals()` computes (13) as
written and `coverage_test()` becomes available.

The raw RMSE series is non-monotonic in k (Dhanmondi k=3 is 9.68 against k=1
at 12.18). With ~37 origins per horizon that is sampling noise, not a claim
that a 3-day-ahead forecast beats a 1-day-ahead one. SIGMA_MONOTONE controls
the treatment; "envelope" (running maximum) is the default, because forecast
uncertainty should not be modelled as shrinking with horizon.

-----------------------------------------------------------------------------
Assumption 1, enforced rather than assumed
-----------------------------------------------------------------------------
The denominator p_{i,d} of Eq. (16) must be the realised peak of the SAME
reference day whose profile L^c_h(t) builds the right-hand side of C1.
`resolve_reference_day()` pins d to the day immediately before the forecast's
first target and refuses to proceed if that day's peak is missing.
`assert_population_alignment()` is called by Stage A/B so the synthetic
population cannot be built for a different day without an error.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as cfg

GAUSSIAN_MAE_RMSE = float(np.sqrt(2.0 / np.pi))     # 0.79788, shape reference
MW_TO_KW = 1000.0   # the frozen forecast CSVs are in MW; section 1.2 is in kW


# =============================================================================
# 1. Inputs
# =============================================================================

def _fc_path(substation: str) -> Path:
    stem = substation.replace(" ", "_")
    for n in (14, 7, 30):
        p = cfg.FORECAST_DIR / f"{stem}_{n}day_forecast.csv"
        if p.exists():
            return p
    return cfg.FORECAST_DIR / f"{stem}_14day_forecast.csv"


def _bt_path(substation: str) -> Path:
    return cfg.FORECAST_DIR / f"{substation.replace(' ', '_')}_recursive_backtest.csv"


def load_point_forecast(substation: str) -> pd.DataFrame:
    """p_hat and the weather-scenario band, as load_prediction.py exports them."""
    p = _fc_path(substation)
    if not p.exists():
        raise FileNotFoundError(
            f"Forecast not found: {p}\nCopy the frozen pipeline's "
            f"*_14day_forecast.csv into {cfg.FORECAST_DIR}")
    df = pd.read_csv(p, parse_dates=["date"])
    need = {"date", "day_ahead", "predicted_mw",
            "predicted_mw_cool_wet", "predicted_mw_hot_dry"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"{p.name} missing columns: {sorted(missing)}")
    df = df.sort_values("day_ahead").reset_index(drop=True)
    # The FROZEN forecasting pipeline exports MW; the formulation works in kW.
    # Convert on read, once, here - never in the caller.
    for col in ("predicted_mw", "predicted_mw_cool_wet", "predicted_mw_hot_dry"):
        df[col.replace("_mw", "_kw")] = df[col] * MW_TO_KW
    return df


def _pava(y: np.ndarray) -> np.ndarray:
    """Pool-adjacent-violators: least-squares isotonic fit, no sklearn needed."""
    y = np.asarray(y, dtype=float).copy()
    w = np.ones_like(y)
    i = 0
    while i < len(y) - 1:
        if y[i] <= y[i + 1] + 1e-12:
            i += 1
            continue
        tot_w = w[i] + w[i + 1]
        avg = (y[i] * w[i] + y[i + 1] * w[i + 1]) / tot_w
        y[i] = avg
        w[i] = tot_w
        y = np.delete(y, i + 1)
        w = np.delete(w, i + 1)
        if i > 0:
            i -= 1
    return y


def load_sigma(substation: str, monotone: str | None = None) -> pd.DataFrame:
    """EQ (13): sigma_{i,k} by horizon, from the rolling-origin backtest."""
    monotone = monotone or cfg.SIGMA_MONOTONE
    p = _bt_path(substation)
    if not p.exists():
        raise FileNotFoundError(f"Backtest not found: {p}")
    bt = pd.read_csv(p).sort_values("horizon").reset_index(drop=True)

    out = pd.DataFrame({
        "horizon": bt["horizon"].astype(int),
        "n_origins": bt["n"].astype(int),
        "rmse": bt["rmse"].astype(float),
        "mae": bt["mae"].astype(float),
        "r2": bt["r2"].astype(float),
    })
    out["sigma_raw"] = out["rmse"] * MW_TO_KW   # backtest RMSE is MW

    if monotone == "raw":
        out["sigma"] = out["sigma_raw"]
    elif monotone == "envelope":
        out["sigma"] = np.maximum.accumulate(out["sigma_raw"].values)
    elif monotone == "isotonic":
        fit = _pava(out["sigma_raw"].values)
        # PAVA collapses blocks; re-expand to the original length.
        expanded, j = [], 0
        for _ in range(len(out)):
            expanded.append(fit[min(j, len(fit) - 1)])
            j += 1 if j < len(fit) - 1 else 0
        out["sigma"] = np.maximum.accumulate(np.array(expanded))
    else:
        raise ValueError(f"unknown SIGMA_MONOTONE: {monotone}")

    out["mae_rmse_ratio"] = out["mae"] / out["rmse"]
    out.attrs["method"] = f"backtest_rmse/{monotone}"
    return out


def sigma_from_residuals(residuals: pd.DataFrame) -> pd.DataFrame:
    """
    Preferred path when the PER-RECORD backtest is available (columns:
    horizon, actual_mw, pred_mw). Computes Eq. (13) as written - the variance
    of the error, with the bias reported separately instead of folded into it.
    """
    rows = []
    for h, g in residuals.groupby("horizon"):
        e = (g["actual_mw"] - g["pred_mw"]).values
        rows.append({
            "horizon": int(h), "n_origins": len(e),
            "bias": float(e.mean()),
            "sigma_raw": float(e.std(ddof=1)),
            "rmse": float(np.sqrt((e ** 2).mean())),
            "mae": float(np.abs(e).mean()),
            "skew": float(pd.Series(e).skew()),
            "excess_kurtosis": float(pd.Series(e).kurtosis()),
        })
    out = pd.DataFrame(rows).sort_values("horizon").reset_index(drop=True)
    out["sigma"] = np.maximum.accumulate(out["sigma_raw"].values)
    out["mae_rmse_ratio"] = out["mae"] / out["rmse"]
    out.attrs["method"] = "residual_std/envelope"
    return out


# =============================================================================
# 2. Assumption 1 - the reference day
# =============================================================================

@dataclass
class ReferenceDay:
    substation: str
    date: str
    peak_kw: float                 # p_{i,d}, denominator of Eq. (16)
    peak_slot: int | None
    peak_clock: str | None
    note: str


def resolve_reference_day(substation: str, bpdb: pd.DataFrame,
                          first_forecast_date: pd.Timestamp) -> ReferenceDay:
    """
    ASSUMPTION 1, enforced. Raises rather than let numerator and denominator of
    Eq. (16) refer to different baselines.
    """
    spec = cfg.SUBSTATIONS[substation]
    lc, tc = spec["load_col"], spec["peak_time_col"]
    obs = bpdb[["Date", lc, tc]].dropna(subset=[lc]).sort_values("Date")
    if obs.empty:
        raise ValueError(f"{substation}: no observed load in the dataset")

    d = pd.Timestamp(first_forecast_date) - pd.Timedelta(days=1)
    row = obs[obs["Date"] == d]
    if row.empty:
        raise ValueError(
            f"ASSUMPTION 1 VIOLATED for {substation}. The forecast's first "
            f"target is {pd.Timestamp(first_forecast_date).date()}, so the "
            f"reference day must be {d.date()}, but no peak is observed on "
            f"that date (last observation {obs['Date'].max().date()}). "
            f"Numerator and denominator of Eq. (16) would refer to different "
            f"baselines. Refusing to proceed.")

    p_id = float(row[lc].iloc[0]) * MW_TO_KW   # BPDB CSV is MW
    if not np.isfinite(p_id) or p_id <= 0:
        raise ValueError(f"{substation}: reference-day peak is {p_id}")

    slot, clock = None, None
    raw = str(row[tc].iloc[0]) if tc in row.columns else ""
    if ":" in raw:
        try:
            s = cfg.clock_to_slot(raw)
            if s not in cfg.PEAK_TIME_ARTEFACT_SLOTS:
                slot, clock = s, cfg.slot_to_clock(s)
        except ValueError:
            pass

    return ReferenceDay(
        substation=substation, date=str(d.date()), peak_kw=p_id,
        peak_slot=slot, peak_clock=clock,
        note=(f"p_(i,d) = {p_id:,.0f} kW realised on {d.date()}"
              + (f", peaking at {clock}" if clock else ", peak time unusable")
              + ". Stage A/B must calibrate the synthetic population to this "
                "same day (Assumption 1)."))


def assert_population_alignment(substation: str, population_date: str) -> None:
    """Called by Stage A/B. Fails loudly on a baseline mismatch."""
    rec = read_cache(substation)
    if not rec:
        raise FileNotFoundError(
            f"Run 05c_forecast_to_kappa.py --build before synthesising the "
            f"{substation} population; the reference day comes from there.")
    ref = rec["reference_day"]["date"]
    if str(population_date) != ref:
        raise ValueError(
            f"ASSUMPTION 1 VIOLATED for {substation}: the population was built "
            f"for {population_date} but Eq. (16) divides by the peak of {ref}. "
            f"Rebuild the population for {ref}, or re-run Stage C/D against a "
            f"forecast whose origin is {population_date}.")


# =============================================================================
# 3. Stress threshold tau_i, Eq. (15)
# =============================================================================

def derive_p_str(substation: str, bpdb: pd.DataFrame,
               trailing_days: int | None = None,
               quantile: float | None = None) -> dict:
    """
    P^str_i on a TRAILING window.

    Estimating P^str on the full record fails at Dhanmondi, whose 2019-2023 level
    was far above today's: the full-record p95 is 187.0 MW, above its entire
    last-365-day range (max 163.0). s_{i,d} would be identically zero and no
    stress rebate (19), regime (46) or event would ever fire. A trailing window
    tracks the level the substation actually operates at now.
    """
    trailing_days = cfg.P_STR_TRAILING_DAYS if trailing_days is None else trailing_days
    quantile = cfg.P_STR_QUANTILE if quantile is None else quantile
    lc = cfg.SUBSTATIONS[substation]["load_col"]
    obs = bpdb[["Date", lc]].dropna().sort_values("Date")
    cutoff = obs["Date"].max() - pd.Timedelta(days=trailing_days)
    recent = obs[obs["Date"] >= cutoff]
    if len(recent) < 60:
        recent = obs.tail(180)
    return {
        "p_str_kw": float(recent[lc].quantile(quantile)) * MW_TO_KW,
        "quantile": quantile, "trailing_days": trailing_days,
        "n_days_used": int(len(recent)),
        "window_start": str(recent["Date"].min().date()),
        "window_end": str(recent["Date"].max().date()),
        "recent_max_kw": float(recent[lc].max()) * MW_TO_KW,
        "recent_mean_kw": float(recent[lc].mean()) * MW_TO_KW,
        "full_record_p_str_kw": float(obs[lc].quantile(quantile)) * MW_TO_KW,
    }


# =============================================================================
# 4. Eqs. (14) - (16)
# =============================================================================

@dataclass
class ForecastHorizon:
    substation: str
    reference_day: dict
    p_str: dict
    p_max_kw: float
    z_beta: float
    beta: float
    sigma_method: str
    rows: list
    diagnostics: dict = field(default_factory=dict)
    checks: list = field(default_factory=list)

    @property
    def all_pass(self) -> bool:
        return bool(self.checks) and all(c["pass"] for c in self.checks)


def build_horizon(substation: str, bpdb: pd.DataFrame,
                  z_beta: float | None = None,
                  dispatch_days: int | None = None) -> ForecastHorizon:
    """
    The full Stage C/D chain:

        (13)  sigma_{i,k}                   from the recursive backtest
        (14)  p~ = p_hat + z_beta * sigma    robust peak
        (15)  s = 1[p~ >= P^str], Def = (p~ - P^max)_+
        (16)  kappa^scale = p~ / p_{i,d}     multiplier into C1 and C4
    """
    z_beta = cfg.Z_BETA if z_beta is None else z_beta
    dispatch_days = cfg.DISPATCH_DAYS if dispatch_days is None else dispatch_days
    spec = cfg.SUBSTATIONS[substation]

    fc = load_point_forecast(substation)
    sg = load_sigma(substation)
    ref = resolve_reference_day(substation, bpdb, fc["date"].iloc[0])
    p_str = derive_p_str(substation, bpdb)
    p_max = float(spec["p_max_kw"])
    smap = dict(zip(sg["horizon"], sg["sigma"]))
    rawmap = dict(zip(sg["horizon"], sg["sigma_raw"]))

    rows = []
    for _, f in fc.iterrows():
        k = int(f["day_ahead"])
        sigma_k = float(smap.get(k, sg["sigma"].iloc[-1]))
        p_hat = float(f["predicted_kw"])

        p_tilde = p_hat + z_beta * sigma_k                        # (14)
        s_flag = int(p_tilde >= p_str["p_str_kw"])                    # (15)
        deficit = max(0.0, p_tilde - p_max)                       # (15)
        kappa_scale = p_tilde / ref.peak_kw                       # (16)

        band = {}
        for label, col in (("cool_wet", "predicted_kw_cool_wet"),
                           ("hot_dry", "predicted_kw_hot_dry")):
            ph = float(f[col])
            pt = ph + z_beta * sigma_k
            band[label] = {"p_hat_kw": ph, "p_tilde_kw": pt,
                           "kappa_scale": pt / ref.peak_kw,
                           "s_stress": int(pt >= p_str["p_str_kw"]),
                           "deficit_kw": max(0.0, pt - p_max)}

        rows.append({
            "horizon_k": k,
            "target_date": str(pd.Timestamp(f["date"]).date()),
            "in_dispatch_horizon": bool(k <= dispatch_days),
            "p_hat_kw": p_hat,
            "sigma_kw": sigma_k,
            "sigma_raw_kw": float(rawmap.get(k, sigma_k)),
            "z_beta": z_beta,
            "uncertainty_margin_kw": z_beta * sigma_k,
            "uncertainty_margin_pct": 100.0 * z_beta * sigma_k / p_hat,
            "p_tilde_kw": p_tilde,
            "p_str_kw": p_str["p_str_kw"],
            "s_stress": s_flag,
            "p_max_kw": p_max,
            "deficit_kw": deficit,
            "p_ref_kw": ref.peak_kw,
            "kappa_scale": kappa_scale,
            "kappa_scale_cool_wet": band["cool_wet"]["kappa_scale"],
            "kappa_scale_hot_dry": band["hot_dry"]["kappa_scale"],
            "s_cool_wet": band["cool_wet"]["s_stress"],
            "s_hot_dry": band["hot_dry"]["s_stress"],
            "weather_source": str(f.get("weather_source", "n/a")),
            "holiday_type": str(f.get("holiday_type", "n/a")),
        })

    diagnostics = {
        "horizon_K": len(rows),
        "dispatch_days": dispatch_days,
        "lookahead_days": len(rows) - dispatch_days,
        "sigma_monotone": cfg.SIGMA_MONOTONE,
        "sigma_raw_decreases": int((np.diff(sg["sigma_raw"].values) < 0).sum()),
        "sigma_min_kw": float(sg["sigma"].min()),
        "sigma_max_kw": float(sg["sigma"].max()),
        "sigma_growth_ratio": float(sg["sigma"].iloc[-1] / sg["sigma"].iloc[0]),
        "mae_rmse_ratio_mean": float(sg["mae_rmse_ratio"].mean()),
        "mae_rmse_ratio_gaussian": GAUSSIAN_MAE_RMSE,
        "n_stress_days": int(sum(r["s_stress"] for r in rows)),
        "n_deficit_days": int(sum(r["deficit_kw"] > 0 for r in rows)),
        "n_stress_days_hot_dry": int(sum(r["s_hot_dry"] for r in rows)),
        "n_stress_days_cool_wet": int(sum(r["s_cool_wet"] for r in rows)),
        "kappa_scale_min": min(r["kappa_scale"] for r in rows),
        "kappa_scale_max": max(r["kappa_scale"] for r in rows),
        "kappa_scale_mean": float(np.mean([r["kappa_scale"] for r in rows])),
    }

    fh = ForecastHorizon(
        substation=substation, reference_day=asdict(ref), p_str=p_str,
        p_max_kw=p_max, z_beta=z_beta, beta=cfg.BETA_CONFIDENCE,
        sigma_method=sg.attrs.get("method", "backtest_rmse"),
        rows=rows, diagnostics=diagnostics)
    fh.checks = validate(fh, sg)
    return fh


# =============================================================================
# 5. Validation
# =============================================================================

def validate(fh: ForecastHorizon, sg: pd.DataFrame) -> list:
    checks: list = []

    def _c(name, ok, detail=""):
        checks.append({"check": name, "pass": bool(ok), "detail": str(detail)})

    rows = fh.rows
    d = fh.diagnostics

    # --- Eq. (14) form ------------------------------------------------------
    _c("(14) p~ >= p_hat at every horizon",
       all(r["p_tilde_kw"] >= r["p_hat_kw"] for r in rows),
       f"margin {min(r['uncertainty_margin_kw'] for r in rows):.2f}-"
       f"{max(r['uncertainty_margin_kw'] for r in rows):.2f} MW "
       f"({min(r['uncertainty_margin_pct'] for r in rows):.1f}-"
       f"{max(r['uncertainty_margin_pct'] for r in rows):.1f}%)")

    _c("(14) uses z_beta, not kappa (no circular definition)",
       all(abs(r["p_tilde_kw"] - (r["p_hat_kw"] + r["z_beta"] * r["sigma_kw"]))
           < 1e-9 for r in rows),
       "p~ reconstructed from p_hat + z_beta*sigma to machine precision")

    _c("z_beta matches the stated confidence level",
       abs(fh.z_beta - 1.645) < 1e-9 and abs(fh.beta - 0.95) < 1e-9,
       f"z_beta={fh.z_beta}, beta={fh.beta}")

    # --- Eq. (13) treatment -------------------------------------------------
    _c("sigma non-decreasing in k after treatment",
       bool(np.all(np.diff(sg["sigma"].values) >= -1e-9)),
       f"{d['sigma_raw_decreases']} raw decreases repaired by "
       f"'{d['sigma_monotone']}'; sigma {d['sigma_min_kw']:.2f} -> "
       f"{d['sigma_max_kw']:.2f} MW ({d['sigma_growth_ratio']:.2f}x)")

    ratio = d["mae_rmse_ratio_mean"]
    _c("residual shape consistent with Gaussian (MAE/RMSE ~ 0.798)",
       abs(ratio - GAUSSIAN_MAE_RMSE) < 0.05,
       f"MAE/RMSE = {ratio:.4f} vs {GAUSSIAN_MAE_RMSE:.4f}. Below this means "
       f"heavier tails, in which case z_beta=1.645 would under-cover")

    # --- Eq. (16) -----------------------------------------------------------
    ks = [r["kappa_scale"] for r in rows]
    _c("(16) kappa^scale finite and positive",
       all(np.isfinite(k) and k > 0 for k in ks),
       f"range {min(ks):.4f} - {max(ks):.4f}")
    _c("(16) kappa^scale within a plausible band (0.5, 2.0)",
       all(0.5 < k < 2.0 for k in ks),
       f"range {min(ks):.4f} - {max(ks):.4f}, mean {d['kappa_scale_mean']:.4f}")
    _c("(16) kappa^scale reconstructs as p~ / p_(i,d)",
       all(abs(r["kappa_scale"] - r["p_tilde_kw"] / r["p_ref_kw"]) < 1e-9
           for r in rows))

    # --- Assumption 1 -------------------------------------------------------
    ref_date = pd.Timestamp(fh.reference_day["date"])
    first_target = pd.Timestamp(rows[0]["target_date"])
    _c("Assumption 1: reference day is the day before the first forecast target",
       (first_target - ref_date).days == 1,
       fh.reference_day["note"])

    # --- Eq. (15) -----------------------------------------------------------
    _c("(15) Def > 0 exactly when p~ > P^max",
       all((r["deficit_kw"] > 0) == (r["p_tilde_kw"] > r["p_max_kw"])
           for r in rows),
       f"{d['n_deficit_days']} deficit day(s), P^max = {fh.p_max_kw:.1f} MW")
    _c("(15) s = 1 exactly when p~ >= P^str_i",
       all(r["s_stress"] == int(r["p_tilde_kw"] >= r["p_str_kw"]) for r in rows))

    t = fh.p_str
    _c("P^str_i is attainable at current load levels",
       t["p_str_kw"] <= t["recent_max_kw"],
       f"P^str={t['p_str_kw']:,.0f} kW vs trailing-{t['trailing_days']}d max "
       f"{t['recent_max_kw']:,.0f} kW (full-record value would be "
       f"{t['full_record_p_str_kw']:,.0f} kW)")
    _c("stress indicator is not degenerate over the horizon",
       0 < d["n_stress_days"] < len(rows),
       f"{d['n_stress_days']}/{len(rows)} stress days; "
       f"cool-wet {d['n_stress_days_cool_wet']}, hot-dry {d['n_stress_days_hot_dry']}")

    # --- Cross-check against the independent one-day-ahead test panel -------
    try:
        ms = pd.read_csv(cfg.DATA_DIR / "metrics_summary.csv")
        r = ms[(ms["substation"] == fh.substation) & (ms["model"] == "BiGRU")]
        if len(r):
            # metrics_summary.csv is exported by the frozen pipeline in MW.
            panel = float(r["test_rmse"].iloc[0]) * MW_TO_KW
            bt1 = float(sg.loc[sg["horizon"] == 1, "sigma_raw"].iloc[0])
            _c("sigma_(i,1) agrees with the one-day-ahead test panel RMSE",
               abs(bt1 - panel) / panel < 0.15,
               f"backtest k=1 {bt1:,.0f} kW vs test panel {panel:,.0f} kW "
               f"({100 * (bt1 - panel) / panel:+.1f}%) - two independent "
               f"estimates of the same quantity")
    except Exception:                                             # noqa: BLE001
        pass

    # --- Weather band -------------------------------------------------------
    # The band is a CLIMATOLOGY SENSITIVITY band, not a confidence interval.
    # Both arms are produced by the frozen pipeline under
    # weather_mode="climatology" at sigma = -1 and +1, while the central case
    # uses API weather. On days where the API forecast departs from
    # climatology, the central legitimately falls OUTSIDE the band. Asserting
    # containment would be asserting something upstream never promised - so we
    # check what the band does guarantee and report the rest as a diagnostic.
    _c("weather band has positive width at every horizon",
       all(abs(r["kappa_scale_hot_dry"] - r["kappa_scale_cool_wet"]) > 0
           for r in rows),
       f"kappa^scale band width "
       f"{min(abs(r['kappa_scale_hot_dry'] - r['kappa_scale_cool_wet']) for r in rows):.4f}"
       f" - "
       f"{max(abs(r['kappa_scale_hot_dry'] - r['kappa_scale_cool_wet']) for r in rows):.4f}")

    outside = [r for r in rows
               if not (min(r["kappa_scale_cool_wet"], r["kappa_scale_hot_dry"]) - 1e-9
                       <= r["kappa_scale"] <=
                       max(r["kappa_scale_cool_wet"], r["kappa_scale_hot_dry"]) + 1e-9)]
    api_only = all(r["weather_source"] == "api" for r in outside)
    _c("central outside the band only on API-weather days",
       api_only,
       f"{len(outside)}/{len(rows)} day(s) outside, all with "
       f"weather_source='api'. Expected: the band is climatology +/-1 sigma, "
       f"the central is API weather. The band is a SCENARIO tool - forecast "
       f"UNCERTAINTY in Eq. (14) comes from sigma_(i,k), never from this band")
    fh.diagnostics["n_central_outside_band"] = len(outside)
    fh.diagnostics["central_outside_band_days"] = [r["horizon_k"] for r in outside]

    # --- Horizon usage, per the formulation's "How the K-step horizon is used"
    n_disp = sum(r["in_dispatch_horizon"] for r in rows)
    _c("dispatch horizon is a strict prefix of the forecast horizon",
       0 < n_disp <= len(rows)
       and all(r["in_dispatch_horizon"] for r in rows[:n_disp])
       and not any(r["in_dispatch_horizon"] for r in rows[n_disp:]),
       f"{n_disp} day(s) enter C1/C4 via kappa^scale; "
       f"{len(rows) - n_disp} day(s) supply s and Def for the regime only")

    return checks


def coverage_test(fh: ForecastHorizon, residuals: pd.DataFrame) -> dict:
    """
    The validation that matters most, available ONLY with the per-record
    backtest: what fraction of realised peaks fall at or below p~?

    Should be about beta = 0.95. Materially below means the bound under-covers
    and z_beta or the sigma treatment needs revisiting; far above means it is
    needlessly conservative and the LP is handed inflated demand every day.
    """
    smap = {r["horizon_k"]: r["sigma_kw"] for r in fh.rows}
    out = []
    for h, g in residuals.groupby("horizon"):
        s = smap.get(int(h))
        if s is None:
            continue
        pt = g["pred_mw"] + fh.z_beta * s
        out.append({"horizon": int(h), "n": int(len(g)),
                    "coverage": float((g["actual_mw"] <= pt).mean())})
    if not out:
        return {"available": False}
    mean_cov = float(np.mean([o["coverage"] for o in out]))
    return {
        "available": True, "by_horizon": out, "overall_coverage": mean_cov,
        "target": fh.beta,
        "verdict": ("under-covers" if mean_cov < fh.beta - 0.05
                    else "over-covers" if mean_cov > fh.beta + 0.05
                    else "adequate"),
    }


def propagation_sweep(substation: str, bpdb: pd.DataFrame,
                      z_values=(0.0, 1.0, 1.282, 1.645, 2.0, 2.326)) -> pd.DataFrame:
    """
    How forecast uncertainty propagates into the MOLP. Sweeping z_beta shows
    the whole chain: sigma -> p~ -> kappa^scale -> C1 right-hand side -> the
    shadow price -> the Proposition 2 threshold. Documented in
    docs/forecast_to_kappa.md section 4.
    """
    rows = []
    for z in z_values:
        fh = build_horizon(substation, bpdb, z_beta=z)
        ks = [r["kappa_scale"] for r in fh.rows]
        rows.append({
            "z_beta": z,
            "confidence": {0.0: 0.50, 1.0: 0.841, 1.282: 0.90,
                           1.645: 0.95, 2.0: 0.977, 2.326: 0.99}.get(z, np.nan),
            "kappa_mean": float(np.mean(ks)),
            "kappa_max": float(np.max(ks)),
            "extra_demand_pct_vs_z0": np.nan,
            "n_stress_days": fh.diagnostics["n_stress_days"],
            "n_deficit_days": fh.diagnostics["n_deficit_days"],
        })
    base = rows[0]["kappa_mean"]
    for r in rows:
        r["extra_demand_pct_vs_z0"] = 100.0 * (r["kappa_mean"] - base) / base
    return pd.DataFrame(rows)


# =============================================================================
# 6. Cache
# =============================================================================

def cache_path(substation: str) -> Path:
    return cfg.CACHE_DIR / f"kappa_{substation.lower().replace(' ', '_')}.json"


def write_cache(fh: ForecastHorizon) -> Path:
    rec = asdict(fh)
    rec["schema_version"] = 1
    rec["all_pass"] = fh.all_pass
    rec["notation"] = {
        "kappa_scale": "Eq. (16) robust scaling factor p~/p_(i,d); multiplies "
                       "the decision-dependent load in C1 (27) and C4 (30)",
        "kappa_cap": "Eq. (34) per-tier curtailment cap on y_(h,t); a "
                     "DIFFERENT quantity that never appears in this file",
        "z_beta": "Eq. (14) quantile multiplier on sigma; NOT a kappa",
    }
    p = cache_path(fh.substation)
    p.write_text(json.dumps(rec, indent=1, default=str), encoding="utf-8")
    return p


def read_cache(substation: str) -> dict | None:
    p = cache_path(substation)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def get_kappa_scale(substation: str, target_date) -> float:
    """kappa^scale_(i,d(t)) for the LP builder. Fails loudly if not built."""
    rec = read_cache(substation)
    if not rec:
        raise FileNotFoundError(
            f"No kappa table for {substation}. "
            f"Run: python 05c_forecast_to_kappa.py --build")
    key = str(pd.Timestamp(target_date).date())
    for r in rec["rows"]:
        if r["target_date"] == key:
            if not r["in_dispatch_horizon"]:
                raise ValueError(
                    f"{key} (k={r['horizon_k']}) is a LOOK-AHEAD day for "
                    f"{substation}, not a dispatch day. Its kappa^scale must "
                    f"not enter C1/C4; it supplies s and Def for the regime "
                    f"only. See the formulation, 'How the K-step horizon is "
                    f"used'.")
            return float(r["kappa_scale"])
    raise KeyError(f"{key} is outside the forecast horizon for {substation}")


def get_stress_indicator(substation: str, target_date) -> int:
    """s_(i,d) for Eq. (19)'s rebate term. Valid across the whole horizon K."""
    rec = read_cache(substation)
    if not rec:
        raise FileNotFoundError(f"No kappa table for {substation}.")
    key = str(pd.Timestamp(target_date).date())
    for r in rec["rows"]:
        if r["target_date"] == key:
            return int(r["s_stress"])
    raise KeyError(f"{key} is outside the forecast horizon for {substation}")


# =============================================================================
# 7. Billing-period plan - mixing observed and forecast days
# =============================================================================

def build_period(bpdb, substation: str, start, end,
                 z_beta: float | None = None) -> list:
    """
    A day-by-day plan for a BILLING PERIOD, which will usually straddle the
    forecast origin.

    Two kinds of day, and the distinction is real:

      OBSERVED  the day is in the BPDB record. Its peak p_{i,d'} is a
                measurement, so there is no forecast error to hedge:
                z_beta * sigma = 0 and kappa_{i,d'} = p_{i,d'} / p_{i,d}.
                This is still exactly Eq. (16) - kappa is a level rescaling of
                the reference-day profile (Assumption 1) - with the numerator
                known rather than predicted.

      FORECAST  the day is beyond the record. p~ = p_hat + z_beta*sigma from
                Eqs. (13)-(14) and kappa from the cached table.

    Every row records `mode`, so no figure can present a backcast day as a
    forecast one. A period made only of OBSERVED days is a perfect-foresight
    backcast and should be labelled as such.
    """
    z_beta = cfg.Z_BETA if z_beta is None else z_beta
    rec = read_cache(substation)
    if not rec:
        raise FileNotFoundError(
            f"No kappa table for {substation}; run 05c_forecast_to_kappa.py --build")
    p_ref = float(rec["reference_day"]["peak_kw"])
    p_str = float(rec["p_str"]["p_str_kw"])
    p_max = float(rec["p_max_kw"])
    fmap = {r["target_date"]: r for r in rec["rows"]}

    lc = cfg.SUBSTATIONS[substation]["load_col"]
    obs = bpdb[["Date", lc]].dropna().set_index("Date")[lc]

    out = []
    for d in pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="D"):
        key = str(d.date())
        if d in obs.index:
            p_real = float(obs.loc[d]) * MW_TO_KW
            out.append({
                "date": key, "mode": "observed",
                "p_hat_kw": p_real, "sigma_kw": 0.0, "z_beta": 0.0,
                "p_tilde_kw": p_real, "uncertainty_margin_kw": 0.0,
                "kappa_scale": p_real / p_ref, "p_ref_kw": p_ref,
                "p_str_kw": p_str, "s_stress": int(p_real >= p_str),
                "p_max_kw": p_max, "deficit_kw": max(0.0, p_real - p_max),
                "weather_source": "observed", "horizon_k": None,
            })
        elif key in fmap:
            r = dict(fmap[key])
            r["mode"] = "forecast"
            r["date"] = key
            out.append(r)
        else:
            raise ValueError(
                f"{substation}: {key} is neither in the BPDB record nor in the "
                f"{len(fmap)}-day forecast horizon. The billing period must lie "
                f"within observed history plus the forecast horizon "
                f"(latest available: {max(fmap)}).")
    return out


def period_bounds(bpdb, substations, days: int = 30) -> tuple:
    """
    The latest `days`-long window that ends at the forecast horizon. This
    straddles the origin, which is the operationally realistic case: a utility
    bills a month that runs from the recent past into the near future.
    """
    ends = []
    for i in substations:
        rec = read_cache(i)
        ends.append(max(r["target_date"] for r in rec["rows"]))
    end = pd.Timestamp(min(ends))
    return str((end - pd.Timedelta(days=days - 1)).date()), str(end.date())
