"""
06_molp_optimizer.py — Multi-Objective Linear Programming (MOLP) demand
response for the detailed study households, 2024-01-01 → 2026-06-28.

Objectives
----------
  f1  minimise customer electricity COST        (TOU tariff, BDT)
  f2  maximise customer QoE                     (minimise discomfort, BDT-eq)
  f3  maximise SGO PROFIT                       (revenue − supply cost, BDT)

Decision variables, per household h, 30-min slot t in the evening peak:
  x[h,t] ∈ [0,1]  fraction of SCHEDULABLE load shifted to off-peak (23–06)
  y[h,t] ∈ [0,MAX_CURTAIL_FRAC]  fraction of CURTAILABLE load curtailed
CRITICAL load (fridge, lighting, medical...) is never touched — hard MOLP
constraint from nodes_Category (must_run).

Solution method
---------------
Weighted-sum scalarisation  max  α·(cost saving) − (1−α)·(discomfort).
Because the objectives are LINEAR in x, y and the constraints are per-variable
boxes, the scalarised optimum is bang-bang (each variable at a bound), so the
threshold rule below is the EXACT optimiser, not a heuristic. Sweeping α
traces the Pareto frontier; the main run uses tier-specific α (equity: lower
tiers weight bill savings, upper tiers weight comfort).

SGO-level forecast integration (UPDATED 2026-07-05)
--------------------------------------------------
Per the thesis architecture (LSTM Forecasting -> MOLP), the day-ahead
substation demand forecast produced by 05b_substation_forecast.py
(global best model BiGRU+BiLSTM, selected in global_model_selection.csv)
now drives MOLP at the SGO level:

  * GRID-STRESS DAYS are declared DAY-AHEAD from the FORECASTED substation
    daily peak (top DR_STRESS_QUANTILE of forecasted peaks), exactly how an
    SGO would trigger a DR event — no hindsight on realised loads.
    Fallback: if the forecast CSV is missing, the original realised
    synthetic-aggregate rule is used (with a warning).
  * The forecast is logged per event as substation_pred_peak_mw, giving the
    Priority Pool and the LLM/KG layer the grid context of each decision.

This module contains NO deep-learning code and NO TensorFlow dependency:
forecasting is a separate pipeline step (Single Responsibility).

The event log keeps the GenTwin schema (optimization_events_log.csv):
predicted_load_w is the SIMULATED COUNTERFACTUAL BASELINE evening-peak load
of the household (what it would have drawn without DR), optimised_load_w the
post-MOLP load, and power_deviation_ratio = (baseline-optimised)/baseline is
the MOLP throttle magnitude used by the GCS. The two proxy ensemble-weight
columns (ensemble_w_bilstm / ensemble_w_bigru) are REMOVED: the single
stacked BiGRU+BiLSTM network has no ensemble weights (07/08 never read
them); substation_pred_peak_mw takes their place.

Outputs (data/):
  06_<sub>_molp_events_log.csv        household × day event log
                                      (+ substation_pred_peak_mw column)
  06_<sub>_molp_household_summary.csv per-household costs, savings, QoE
  06_<sub>_stress_calendar.csv        daily forecast vs stress threshold
                                      (grounds "why was there NO DR" too)
  06_<sub>_molp_monthly_bills.csv     per-household MONTHLY bills:
                                      baseline vs optimised (net of DR
                                      rebate), saving, energy, discomfort
                                      — feeds the KG's MonthlyBill nodes
  06_molp_pareto.csv                  α sweep → Pareto frontier points
  06_molp_pareto.png                  frontier plot
  06_<sub>_peakday_profile.png        substation before/after on worst day
  06_sgo_summary.csv                  per substation × year SGO profit + peak cut
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config as C

CAL = pd.read_csv(C.DATA_DIR / "00_bd_calendar_30min.csv",
                  parse_dates=["timestamp"])
T = len(CAL)
HOUR = CAL["hour"].values + CAL["minute"].values / 60.0
DATE = CAL["timestamp"].dt.date.values
YEARV = CAL["timestamp"].dt.year.values
N_DAYS = T // C.SLOTS_PER_DAY

PEAK = (HOUR >= 17) & (HOUR < 23)
OFFP = (HOUR >= 23) | (HOUR < 6)
TOU_PERIOD = np.where(PEAK, "peak", np.where(OFFP, "off_peak", "standard"))
TOU_MULT = np.where(PEAK, C.TOU["peak"]["multiplier"],
            np.where(OFFP, C.TOU["off_peak"]["multiplier"], 1.0))
SUPPLY = np.where(PEAK, C.SGO_SUPPLY_COST["peak"],
           np.where(OFFP, C.SGO_SUPPLY_COST["off_peak"],
                    C.SGO_SUPPLY_COST["standard"]))
H = 0.5  # hours per slot


def stress_days(substation: str):
    """SGO-level DR trigger. Primary: grid-stress days are declared
    DAY-AHEAD from the BiGRU+BiLSTM substation forecast written by
    05b_substation_forecast.py (top DR_STRESS_QUANTILE of forecasted daily
    peaks). Fallback (forecast CSV missing): original hindsight rule on the
    realised synthetic aggregate.

    Returns (is_stress_day (N_DAYS,) bool, sub_pred_mw (N_DAYS,) float —
    NaN in fallback mode)."""
    days = pd.to_datetime(pd.Series(DATE[::C.SLOTS_PER_DAY]))
    fpath = C.DATA_DIR / f"05b_{substation.lower()}_daily_forecast.csv"
    if fpath.exists():
        fc = pd.read_csv(fpath, parse_dates=["date"]).set_index("date")
        pred = (fc["pred_mw"].reindex(days)
                  .interpolate(limit_direction="both").values)
        actual = (fc["actual_mw"].reindex(days)
                    .interpolate(limit_direction="both").values)
        thr = np.quantile(pred, C.DR_STRESS_QUANTILE)
        # export the SGO's daily DR decision calendar: lets the KG/LLM
        # explain NON-events ("no DR yesterday: forecast 88 MW was below
        # the 105 MW stress threshold") as well as events
        pd.DataFrame({"date": days.dt.strftime("%Y-%m-%d"),
                      "pred_peak_mw": np.round(pred, 2),
                      "actual_peak_mw": np.round(actual, 2),
                      "forecast_error_mw": np.round(pred - actual, 2),
                      "stress_threshold_mw": round(float(thr), 2),
                      "is_stress_day": (pred >= thr).astype(int)}
                     ).to_csv(C.DATA_DIR /
                              f"06_{substation.lower()}_stress_calendar.csv",
                              index=False)
        return pred >= thr, pred
    print(f"  WARNING: {fpath.name} not found — run "
          f"05b_substation_forecast.py first. Falling back to hindsight "
          f"stress days from the realised synthetic aggregate.")
    agg = pd.read_csv(C.DATA_DIR / f"04_{substation.lower()}_aggregate_30min.csv",
                      usecols=["total_mw"])
    dp = agg["total_mw"].values.reshape(-1, C.SLOTS_PER_DAY).max(axis=1)
    thr = np.quantile(dp, C.DR_STRESS_QUANTILE)
    return dp >= thr, np.full(N_DAYS, np.nan)


def optimise_substation(substation: str, alphas_override=None):
    z = np.load(C.DATA_DIR / f"04_{substation.lower()}_hh_category_loads.npz",
                allow_pickle=True)
    hh_ids = z["household_ids"]; tiers = z["tiers"]
    sched_kw = z["schedulable_w"] / 1000.0        # (T, n_hh)
    curt_kw  = z["curtailable_w"] / 1000.0
    crit_kw  = z["critical_w"] / 1000.0
    n_hh = len(hh_ids)

    # Marginal BERC slab rate at each household's BASELINE monthly consumption
    # (commit 5b). Per-household constant -> Eqs. 30-31 stay exact.
    _months = pd.PeriodIndex(CAL["timestamp"], freq="M")
    _n_months = _months.nunique()
    _base_kwh = (sched_kw + curt_kw + crit_kw).sum(axis=0) * H / _n_months
    rate  = np.array([C.marginal_slab_rate(k) for k in _base_kwh])   # (n_hh,)
    alpha = (np.full(n_hh, alphas_override) if alphas_override is not None
             else np.array([C.TIER_ALPHA[t] for t in tiers]))
    d_s   = np.array([C.DISCOMFORT_SHIFT[t] for t in tiers])
    d_c   = np.array([C.DISCOMFORT_CURTAIL[t] for t in tiers])

    is_stress_day, sub_pred_mw = stress_days(substation)
    stress_slot = np.repeat(is_stress_day, C.SLOTS_PER_DAY)          # (T,)

    # ── Exact bang-bang MOLP decisions (per household, evening-peak slots) ──
    # Shift:   per-kWh gain = rate·(1.35−0.80); discomfort d_s
    shift_gain = rate * (C.TOU["peak"]["multiplier"] - C.TOU["off_peak"]["multiplier"])
    do_shift = (alpha * shift_gain - (1 - alpha) * d_s) > 0          # (n_hh,)
    # Curtail: per-kWh gain = rate·1.35 (+ DR rebate on stress days)
    curt_gain_norm   = rate * C.TOU["peak"]["multiplier"]
    curt_gain_stress = curt_gain_norm + C.DR_REBATE_BDT_KWH
    do_curt_norm   = (alpha * curt_gain_norm   - (1 - alpha) * d_c) > 0
    do_curt_stress = (alpha * curt_gain_stress - (1 - alpha) * d_c) > 0

    x = PEAK[:, None] & do_shift[None, :]                            # (T,n_hh)
    y = PEAK[:, None] & np.where(stress_slot[:, None],
                                 do_curt_stress[None, :],
                                 do_curt_norm[None, :])

    shifted_kw   = sched_kw * x                                      # removed at peak
    curtailed_kw = curt_kw * y * C.MAX_CURTAIL_FRAC

    # shifted energy lands uniformly in the same day's off-peak slots
    day_idx = np.arange(T) // C.SLOTS_PER_DAY
    shifted_kwh_day = np.zeros((N_DAYS, n_hh))
    np.add.at(shifted_kwh_day, day_idx, shifted_kw * H)
    n_off = np.bincount(day_idx, weights=OFFP.astype(float))         # slots/day
    landed_kw = (shifted_kwh_day[day_idx] / (n_off[day_idx, None] * H)) \
                * OFFP[:, None]

    base_kw = crit_kw + sched_kw + curt_kw
    opt_kw  = base_kw - shifted_kw - curtailed_kw + landed_kw

    # ── Objective accounting ────────────────────────────────────────────────
    price = (TOU_MULT[:, None] * rate[None, :])                      # BDT/kWh
    base_cost = (base_kw * price * H)
    opt_cost  = (opt_kw  * price * H)
    rebate    = (curtailed_kw * H * C.DR_REBATE_BDT_KWH
                 * stress_slot[:, None])
    opt_cost -= rebate
    discomfort = (shifted_kw * H * d_s[None, :]
                  + curtailed_kw * H * d_c[None, :])
    sgo_base = (base_kw * (price - SUPPLY[:, None]) * H)
    sgo_opt  = (opt_kw  * (price - SUPPLY[:, None]) * H) - rebate

    # ── Household summary ───────────────────────────────────────────────────
    hh_sum = pd.DataFrame({
        "household_id": hh_ids, "substation": substation, "tier": tiers,
        "alpha": alpha,
        "baseline_cost_bdt":  base_cost.sum(0).round(0),
        "optimised_cost_bdt": opt_cost.sum(0).round(0),
        "saving_pct": (100 * (base_cost.sum(0) - opt_cost.sum(0))
                       / base_cost.sum(0)).round(2),
        "energy_shifted_kwh":   (shifted_kw.sum(0) * H).round(1),
        "energy_curtailed_kwh": (curtailed_kw.sum(0) * H).round(1),
        "discomfort_bdt_eq": discomfort.sum(0).round(0),
        "qoe_score": (100 * (1 - discomfort.sum(0)
                     / np.maximum(base_cost.sum(0), 1))).round(2),
    })

    # ── Daily events log — SAME SCHEMA as the earlier GenTwin
    #    optimization_events_log.csv, at household × day granularity ─────────
    act_shift = shifted_kw.reshape(N_DAYS, C.SLOTS_PER_DAY, n_hh).sum(1) * H
    act_curt  = curtailed_kw.reshape(N_DAYS, C.SLOTS_PER_DAY, n_hh).sum(1) * H
    pk_slots  = PEAK.reshape(N_DAYS, C.SLOTS_PER_DAY)[0].sum()       # 12
    base_pk_w = base_kw.reshape(N_DAYS, C.SLOTS_PER_DAY, n_hh)[
        :, PEAK.reshape(N_DAYS, C.SLOTS_PER_DAY)[0], :].mean(1) * 1000
    opt_pk_w  = opt_kw.reshape(N_DAYS, C.SLOTS_PER_DAY, n_hh)[
        :, PEAK.reshape(N_DAYS, C.SLOTS_PER_DAY)[0], :].mean(1) * 1000

    # Household-level "predicted" load = SIMULATED COUNTERFACTUAL BASELINE
    # (the load the household would have drawn without DR). Forecasting is
    # SGO-level only (see stress_days); households are downstream simulation.
    pred_w = pd.DataFrame(base_pk_w)

    days = pd.to_datetime(pd.Series(DATE[::C.SLOTS_PER_DAY]))
    price_mult_norm = np.where(is_stress_day, 1.0, 0.5)   # price-shock signal
    rows = []
    for j, hh in enumerate(hh_ids):
        mask = ((act_shift[:, j] > 0.01) | (act_curt[:, j] > 0.01)) \
               & pred_w[j].notna().values
        if not mask.any():
            continue
        curt_m = act_curt[mask, j] > 0.01
        action = np.where(curt_m,
                          np.where(is_stress_day[mask],
                                   "DR_CURTAIL+SHIFT", "CURTAIL+SHIFT"),
                          "SHIFT")
        # appliance_alpha: priority weight of the highest-priority affected
        # category (Curtailable 0.7 > Schedulable 0.4); Critical never appears
        app_alpha = np.where(curt_m, 0.7, 0.4)
        pw = pred_w[j].values[mask]
        ow = opt_pk_w[mask, j]
        rows.append(pd.DataFrame({
            "timestamp": days[mask].values,
            "household_id": hh, "substation": substation, "tier": tiers[j],
            "tou_period": "peak",
            "tou_multiplier": C.TOU["peak"]["multiplier"],
            "tou_rate_bdt_kwh": round(
                rate[j] * C.TOU["peak"]["multiplier"], 3),
            "predicted_load_w": pw.round(1),
            "optimised_load_w": ow.round(1),
            "power_deviation_w": (pw - ow).round(1),
            "power_deviation_ratio": np.clip((pw - ow) /
                                             np.maximum(pw, 1), 0, 1).round(4),
            "price_multiplier_norm": price_mult_norm[mask],
            "appliance_alpha": app_alpha,
            "molp_alpha": alpha[j],
            "action_taken": action,
            "is_stress_day": is_stress_day[mask].astype(int),
            "energy_shifted_kwh": act_shift[mask, j].round(3),
            "energy_curtailed_kwh": act_curt[mask, j].round(3),
            "substation_pred_peak_mw": np.round(sub_pred_mw[mask], 2),
        }))
    log = (pd.concat(rows, ignore_index=True) if rows
           else pd.DataFrame())
    return hh_sum, log, {
        "hh_ids": hh_ids, "tiers": tiers,
        "base_cost": base_cost, "opt_cost": opt_cost,
        "discomfort": discomfort, "sgo_base": sgo_base, "sgo_opt": sgo_opt,
        "base_kw": base_kw, "opt_kw": opt_kw,
        "shifted_kw": shifted_kw, "curtailed_kw": curtailed_kw,
        "is_stress_day": is_stress_day,
    }


def substation_scale(substation: str, res: dict) -> pd.DataFrame:
    """Extrapolate the household sample to the residential cohorts and report
    SGO-side profit + peak reduction per year."""
    calc = pd.read_csv(
        C.DATA_DIR / f"03b_{substation.lower()}_composition_calibrated.csv")
    tiers = res["tiers"]
    out = []
    for y in C.YEARS:
        ymask = YEARV == y
        if not ymask.any():
            continue
        peak_cut_mw = 0.0
        sgo_delta = 0.0
        for tier in C.TIER_MIX[substation]:
            sel = tiers == tier
            n_cohort = int(calc.loc[(calc["category"] == "residential") &
                                    (calc["subcategory"] == tier),
                                    f"units_{y}"].iloc[0])
            per_hh_cut = (res["shifted_kw"][ymask][:, sel]
                          + res["curtailed_kw"][ymask][:, sel])
            # mean reduction during evening peak slots
            pk = PEAK[ymask]
            cut_kw = per_hh_cut[pk].mean() if pk.any() else 0.0
            peak_cut_mw += cut_kw * n_cohort / 1000.0
            per_hh_sgo = (res["sgo_opt"][ymask][:, sel].sum(0).mean()
                          - res["sgo_base"][ymask][:, sel].sum(0).mean())
            sgo_delta += per_hh_sgo * n_cohort
        out.append({"substation": substation, "year": y,
                    "evening_peak_reduction_mw": round(peak_cut_mw, 2),
                    "sgo_profit_delta_bdt": round(sgo_delta, 0),
                    "n_stress_days": int(res["is_stress_day"]
                        [np.unique(np.arange(T)[ymask] // C.SLOTS_PER_DAY)].sum())})
    return pd.DataFrame(out)


def monthly_bills(substation: str, res: dict) -> pd.DataFrame:
    """Per-household MONTHLY billing table. baseline_bill_bdt = TOU cost of
    the unoptimised load; optimised_bill_bdt = post-MOLP cost NET of the DR
    rebate (opt_cost already includes it). Grounds KG/LLM answers to
    'how much did household X save last month, and at what comfort cost?'"""
    per = pd.PeriodIndex(CAL["timestamp"], freq="M")
    codes, uniq = pd.factorize(per)
    n_m, n_hh = len(uniq), len(res["hh_ids"])

    def agg(arr):
        out = np.zeros((n_m, n_hh))
        np.add.at(out, codes, arr)
        return out

    base_b = agg(res["base_cost"]); opt_b = agg(res["opt_cost"])
    disc   = agg(res["discomfort"])
    energy = agg(res["base_kw"] * H)
    shift  = agg(res["shifted_kw"] * H)
    curt   = agg(res["curtailed_kw"] * H)

    months = np.repeat(uniq.astype(str), n_hh)
    hh     = np.tile(res["hh_ids"], n_m)
    tiers  = np.tile(res["tiers"], n_m)
    saving = base_b - opt_b
    return pd.DataFrame({
        "household_id": hh, "substation": substation, "tier": tiers,
        "month": months,
        "baseline_bill_bdt":  base_b.ravel().round(1),
        "optimised_bill_bdt": opt_b.ravel().round(1),
        "saving_bdt":         saving.ravel().round(1),
        "saving_pct": (100 * saving / np.maximum(base_b, 1e-9)
                       ).ravel().round(2),
        "energy_kwh":           energy.ravel().round(1),
        "energy_shifted_kwh":   shift.ravel().round(2),
        "energy_curtailed_kwh": curt.ravel().round(2),
        "discomfort_bdt_eq":    disc.ravel().round(1),
    })


def pareto_sweep(substation: str) -> pd.DataFrame:
    pts = []
    for a in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        hh_sum, _, res = optimise_substation(substation, alphas_override=a)
        pts.append({
            "substation": substation, "alpha": a,
            "mean_saving_pct": hh_sum["saving_pct"].mean().round(3),
            "mean_discomfort_bdt": hh_sum["discomfort_bdt_eq"].mean().round(1),
            "sgo_profit_delta_bdt": (res["sgo_opt"].sum() -
                                     res["sgo_base"].sum()).round(0),
        })
    return pd.DataFrame(pts)


def plot_peak_day(substation: str, res: dict):
    tot_b = res["base_kw"].sum(1); tot_o = res["opt_kw"].sum(1)
    dp = tot_b.reshape(-1, C.SLOTS_PER_DAY).max(1)
    d = int(np.argmax(dp)); sl = slice(d * 48, (d + 1) * 48)
    hrs = HOUR[sl]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(hrs, tot_b[sl], label="Baseline (120 HH sample)", color="steelblue")
    ax.plot(hrs, tot_o[sl], label="MOLP optimised", color="tomato")
    ax.axvspan(17, 23, alpha=0.12, color="orange", label="Evening peak (TOU)")
    ax.set_xlabel("Hour of day"); ax.set_ylabel("kW")
    ax.set_title(f"{substation} — worst synthetic peak day "
                 f"({str(DATE[sl.start])}): MOLP before/after")
    ax.legend()
    fig.tight_layout()
    fig.savefig(C.DATA_DIR / f"06_{substation.lower()}_peakday_profile.png",
                dpi=130)
    plt.close(fig)


def main():
    all_sgo, all_pareto = [], []
    for sub in C.SUBSTATIONS:
        print(f"\n=== MOLP: {sub} ===")
        hh_sum, log, res = optimise_substation(sub)
        hh_sum.to_csv(C.DATA_DIR / f"06_{sub.lower()}_molp_household_summary.csv",
                      index=False)
        log.to_csv(C.DATA_DIR / f"06_{sub.lower()}_molp_events_log.csv",
                   index=False)
        monthly_bills(sub, res).to_csv(
            C.DATA_DIR / f"06_{sub.lower()}_molp_monthly_bills.csv",
            index=False)
        print(hh_sum.groupby("tier")[["saving_pct", "qoe_score",
              "energy_shifted_kwh", "energy_curtailed_kwh"]]
              .mean().round(2).to_string())
        sgo = substation_scale(sub, res)
        print(sgo.to_string(index=False))
        all_sgo.append(sgo)
        plot_peak_day(sub, res)
        all_pareto.append(pareto_sweep(sub))

    pd.concat(all_sgo).to_csv(C.DATA_DIR / "06_sgo_summary.csv", index=False)
    par = pd.concat(all_pareto)
    par.to_csv(C.DATA_DIR / "06_molp_pareto.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    for sub, g in par.groupby("substation"):
        ax.plot(g["mean_discomfort_bdt"], g["mean_saving_pct"], "o-", label=sub)
        for _, r in g.iterrows():
            ax.annotate(f"α={r['alpha']:.1f}",
                        (r["mean_discomfort_bdt"], r["mean_saving_pct"]),
                        fontsize=7, xytext=(4, 3), textcoords="offset points")
    ax.set_xlabel("Mean household discomfort (BDT-equivalent)")
    ax.set_ylabel("Mean bill saving (%)")
    ax.set_title("MOLP Pareto frontier — cost saving vs QoE (α sweep)")
    ax.legend(); fig.tight_layout()
    fig.savefig(C.DATA_DIR / "06_molp_pareto.png", dpi=130)
    plt.close(fig)
    print("\nMOLP outputs written to data/")


if __name__ == "__main__":
    main()