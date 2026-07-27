"""
08_knowledge_graph.py — build the Neo4j knowledge graph for LLM-based
explanations of MOLP demand-response decisions.

Schema (same names as the earlier project pipeline, so existing loaders work):

  (:User)-[:HAS_METER]->(:Meter)-[:FEEDS_FROM]->(:Substation)
  (:Meter)-[:LOCATED_IN]->(:Ward)-[:PART_OF]->(:Substation)
  (:User)-[:IN_TIER]->(:SocioeconomicTier)
  (:Meter)-[:REGISTERS]->(:Appliance)
  (:Appliance)-[:INSTALLED_AT]->(:Meter)
  (:Appliance)-[:BELONGS_TO {flexibility_index}]->(:Category)
  (:User)-[:SUBJECT_TO {monthly_kwh_est}]->(:TariffSlab)
  (:User)-[:EXPERIENCED {date, energy_shifted_kwh, ...}]->(:DrEvent)
  (:DrEvent)-[:DURING]->(:TouPeriod)
  (:User)-[:BILLED {month}]->(:MonthlyBill)     # per-household monthly bill
  (:ForecastDay)-[:FORECAST_FOR]->(:Substation) # every day's forecast +
                                                #   stress threshold ("why
                                                #   was there NO DR?")

Category nodes carry the natural-language explanation_template used by the
LLM to justify each action to the customer; DrEvent nodes carry a rendered
explanation string, so a RAG pipeline can answer "why was my AC rescheduled
on 2025-05-14?" by graph lookup + template filling.

Outputs:
  kg/nodes_*.csv, kg/rels_*.csv      Neo4j bulk-import format
  kg/load_kg.cypher                  LOAD CSV script
  kg/llm_context.md                  how to serialise the graph for the LLM

------------------------------------------------------------------------------
ENCODING FIX (2026-07-04)
------------------------------------------------------------------------------
Root cause of the crash on Windows:

    UnicodeEncodeError: 'charmap' codec can't encode character '\\u03b1'
    in position 1213: character maps to <undefined>

`pathlib.Path.write_text()` and the builtin `open()` default to
`encoding=None`, which resolves to `locale.getpreferredencoding(False)`.
On Linux/macOS that is virtually always UTF-8, so the bug is invisible
there. On Windows it resolves to the legacy ANSI code page — `cp1252`
for most EN-US/EN-GB installs — which is a single-byte encoding that
cannot represent the Greek letter alpha (U+03B1, used in the phrase
"tier (alpha weight)" inside `write_llm_context()`), or any other
character outside its 256-slot table (µ, °, β, —, ', ✓, etc.).

Fix applied below: every `write_text()` call now passes
`encoding="utf-8"` explicitly, and stdout/stderr are reconfigured to
UTF-8 on Windows so future console prints of non-ASCII text (BDT taka
symbols, Bengali place names, Greek symbols in formulas, etc.) won't
crash either. `pandas.DataFrame.to_csv()` was never affected — pandas
hardcodes UTF-8 by default regardless of OS — but `encoding="utf-8"`
is now passed explicitly there too, purely so every disk-write call in
this file states its encoding rather than relying on any default.
------------------------------------------------------------------------------
"""

import sys
import numpy as np
import pandas as pd
import config as C

# ── Windows console hardening ────────────────────────────────────────────
# Without this, print()/logging of non-ASCII text (Bengali names, BDT
# symbols, Greek letters in formulas) can raise the same UnicodeEncodeError
# at the *console* layer even after file writes are fixed, because Windows
# terminals also default to cp1252 for stdout/stderr.
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

rng = np.random.default_rng(C.SEED)

# BERC Feb-2024 residential slab tariff (BDT/kWh) — same as project file
TARIFF_SLABS = C.TARIFF_SLABS   # single source of truth: config.py
_LEGACY_SLABS = [
    (1, 0,   50,  4.63), (2, 0,   75,  5.26), (3, 76, 200, 7.20),
    (4, 201, 300, 7.59), (5, 301, 400, 8.02), (6, 401, 600, 12.67),
    (7, 601, 100000, 14.61),
]

# flexibility_index measures GRID-CONTROL flexibility (capacity to contribute
# to a DR action), not ease of rescheduling in time -- it is consumed to
# justify WHICH appliance was actioned first. A thermostatically controlled
# load can be throttled continuously and is the single largest residential
# load, so Curtailable now ranks above Schedulable. This also preserves the
# model document's worked ratio: AC 0.70*eps = 0.753 vs refrigerator
# 0.05*eps = 0.046, i.e. ~16x.
FLEX_INDEX = {"Critical": 0.05, "Schedulable": 0.45, "Curtailable": 0.70}

CATEGORY_ROWS = [
    {"nodeId": "CAT_Critical", "name": "Critical", "curtailment_priority": 3,
     "can_curtail": False, "can_schedule": False, "label": "Category",
     "molp_constraint": "must_run",
     "explanation_template": ("The {appliance} in your home is classified as "
        "Critical. It will NOT be affected during demand response events to "
        "protect food safety and essential services.")},
    {"nodeId": "CAT_Schedulable", "name": "Schedulable", "curtailment_priority": 2,
     "can_curtail": False, "can_schedule": True, "label": "Category",
     "molp_constraint": "flexible_start",
     "explanation_template": ("Your {appliance} can be shifted to run during "
        "off-peak hours (23:00-06:00) to reduce your electricity cost. This "
        "does not affect its function, only its start time.")},
    {"nodeId": "CAT_Curtailable", "name": "Curtailable", "curtailment_priority": 1,
     "can_curtail": True, "can_schedule": False, "label": "Category",
     "molp_constraint": "power_reduction_allowed",
     "explanation_template": ("During grid stress events, your {appliance} may "
        "be temporarily reduced. You are notified in advance and compensated "
        "with a bill credit of 3 BDT per curtailed kWh.")},
]

TOU_ROWS = [
    {"nodeId": "TOU_peak", "name": "peak", "hours": "17:00-23:00",
     "multiplier": 1.35, "label": "TouPeriod"},
    {"nodeId": "TOU_standard", "name": "standard", "hours": "06:00-17:00",
     "multiplier": 1.00, "label": "TouPeriod"},
    {"nodeId": "TOU_off_peak", "name": "off_peak", "hours": "23:00-06:00",
     "multiplier": 0.80, "label": "TouPeriod"},
]


HOT_DAY_C = 33.0          # daily max >= 33 C counts as a "hot day" (Dhaka)


def load_weather_context():
    """Monthly and daily weather stats from 01_dhaka_weather_30min.csv so
    the LLM can ground heat-related explanations ("June was hot -> AC ran
    more -> bill rose") in real numbers instead of inventing them.
    Returns (monthly: dict 'YYYY-MM' -> stats, daily_max: dict 'YYYY-MM-DD'
    -> max temp C); empty dicts (with a warning) if the CSV is missing."""
    wpath = C.DATA_DIR / "01_dhaka_weather_30min.csv"
    if not wpath.exists():
        print(f"  WARNING: {wpath.name} not found — weather context "
              f"omitted from MonthlyBill/DrEvent nodes.")
        return {}, {}
    w = pd.read_csv(wpath,
                    usecols=["timestamp", "temperature_c",
                             "relative_humidity_pct"],
                    parse_dates=["timestamp"])
    day = w["timestamp"].dt.strftime("%Y-%m-%d")
    ym = w["timestamp"].dt.strftime("%Y-%m")
    daily_max = w.groupby(day)["temperature_c"].max()
    month = daily_max.index.str[:7]
    monthly = pd.DataFrame({
        "mean_temp_c": w.groupby(ym)["temperature_c"].mean().round(1),
        "max_temp_c": daily_max.groupby(month).max().round(1),
        "hot_days": (daily_max >= HOT_DAY_C).groupby(month).sum()
                     .astype(int),
        "mean_humidity_pct": w.groupby(ym)["relative_humidity_pct"]
                              .mean().round(1),
    })
    # monthly rainfall from the REAL BPDB record (Rain_Sum_mm, daily)
    try:
        bp = pd.read_csv(C.BPDB_CSV, usecols=["Date", "Rain_Sum_mm"])
        bp["Date"] = pd.to_datetime(bp["Date"])
        bp = bp[(bp["Date"] >= C.START_DATE) & (bp["Date"] <= C.END_DATE)]
        monthly["rain_mm"] = (bp.groupby(bp["Date"].dt.strftime("%Y-%m"))
                                ["Rain_Sum_mm"].sum().round(1))
    except Exception as exc:
        print(f"  NOTE: rainfall unavailable ({exc}); rain_mm omitted.")
        monthly["rain_mm"] = None
    return monthly.to_dict("index"), daily_max.round(1).to_dict()


def slab_for(kwh: float) -> int:
    for sid, lo, hi, _ in TARIFF_SLABS:
        if lo <= kwh <= hi:
            return sid
    return 7


def main():
    users, meters, appliances, wards = [], [], [], []
    r_hasmeter, r_feeds, r_located, r_partof = [], [], [], []
    r_intier, r_registers, r_installed, r_belongs, r_subject = [], [], [], [], []
    dr_events, r_experienced, r_during = [], [], []
    bills, r_billed = [], []
    fdays, r_fcast = [], []
    wx_month, wx_day = load_weather_context()

    substations = []
    for sub in C.SUBSTATIONS:
        meta = C.SUBSTATION_META[sub]
        targets = C.bpdb_yearly_targets(sub)
        sub_id = f"SUB_{sub}"
        substations.append({
            "nodeId": sub_id, "name": f"{sub} 132/33kV Substation",
            "city": "Dhaka", "division": "Dhaka", "utility": meta["utility"],
            "mean_daily_peak_mw_2024": targets.get(2024),
            "mean_daily_peak_mw_2025": targets.get(2025),
            "mean_daily_peak_mw_2026": targets.get(2026),
            "feeder_voltage_kv": meta["feeder_voltage_kv"],
            "latitude": meta["latitude"], "longitude": meta["longitude"],
            "label": "Substation"})

        for w in range(1, 6):                    # 5 supply wards per substation
            wards.append({"nodeId": f"WARD_{sub[:3].upper()}_{w}",
                          "name": f"{sub} supply ward {w}",
                          "substation_id": sub_id, "label": "Ward"})

        inv = pd.read_csv(
            C.DATA_DIR / f"04_{sub.lower()}_household_inventory.csv")
        wide = pd.read_csv(
            C.DATA_DIR / f"04_{sub.lower()}_households_30min.csv", nrows=1)
        hh_cols = [c for c in wide.columns if c != "timestamp"]
        # monthly kWh estimate per household from the full series
        full = pd.read_csv(
            C.DATA_DIR / f"04_{sub.lower()}_households_30min.csv",
            usecols=hh_cols)
        n_months = 30                             # 2024-01 … 2026-06
        monthly_kwh = (full.sum() * 0.5 / n_months).round(1)

        # ── per-household MONTHLY bills (from 06) → MonthlyBill nodes ──
        bills_path = C.DATA_DIR / f"06_{sub.lower()}_molp_monthly_bills.csv"
        avg_bill = {}
        if bills_path.exists():
            bdf = pd.read_csv(bills_path)
            avg_bill = bdf.groupby("household_id")["optimised_bill_bdt"] \
                          .mean().round(0).to_dict()
            for _, b in bdf.iterrows():
                bill_id = f"BILL_{b['household_id']}_{b['month'].replace('-', '_')}"
                bills.append({
                    "nodeId": bill_id, "household_id": b["household_id"],
                    "month": b["month"],
                    "baseline_bill_bdt": b["baseline_bill_bdt"],
                    "optimised_bill_bdt": b["optimised_bill_bdt"],
                    "saving_bdt": b["saving_bdt"],
                    "saving_pct": b["saving_pct"],
                    "energy_kwh": b["energy_kwh"],
                    "energy_shifted_kwh": b["energy_shifted_kwh"],
                    "energy_curtailed_kwh": b["energy_curtailed_kwh"],
                    "discomfort_bdt_eq": b["discomfort_bdt_eq"],
                    "mean_temp_c": wx_month.get(b["month"], {}).get("mean_temp_c"),
                    "max_temp_c": wx_month.get(b["month"], {}).get("max_temp_c"),
                    "hot_days": wx_month.get(b["month"], {}).get("hot_days"),
                    "mean_humidity_pct": wx_month.get(b["month"], {})
                                                 .get("mean_humidity_pct"),
                    "rain_mm": wx_month.get(b["month"], {}).get("rain_mm"),
                    "label": "MonthlyBill"})
                r_billed.append((f"USER_{b['household_id']}", bill_id,
                                 "BILLED", b["month"]))
        else:
            print(f"  WARNING: {bills_path.name} not found — run "
                  f"06_molp_optimizer.py first; MonthlyBill nodes skipped "
                  f"and User.avg_monthly_bill_bdt left empty.")

        # ── ForecastDay nodes: EVERY day's forecast + stress threshold —
        # lets the LLM answer "why was there NO demand response on date X?"
        # (stress rule identical to 06_molp_optimizer.stress_days)
        fpath = C.DATA_DIR / f"05b_{sub.lower()}_daily_forecast.csv"
        if fpath.exists():
            fc = pd.read_csv(fpath, parse_dates=["date"]).set_index("date")
            days = pd.date_range(C.START_DATE, C.END_DATE, freq="D")
            pred = (fc["pred_mw"].reindex(days)
                      .interpolate(limit_direction="both"))
            actual = (fc["actual_mw"].reindex(days)
                        .interpolate(limit_direction="both"))
            thr = float(np.quantile(pred, C.DR_STRESS_QUANTILE))
            fc_ctx = {}      # date -> structured grid evidence for DrEvents
            for d, p in pred.items():
                ds = d.strftime("%Y-%m-%d")
                act = float(actual.loc[d])
                fc_ctx[ds] = {"actual": round(act, 2),
                              "error": round(float(p) - act, 2),
                              "threshold": round(thr, 2)}
                fid = f"FC_{sub[:3].upper()}_{ds.replace('-', '_')}"
                fdays.append({
                    "nodeId": fid, "date": ds, "substation": sub,
                    "pred_peak_mw": round(float(p), 2),
                    "actual_peak_mw": round(act, 2),
                    "forecast_error_mw": round(float(p) - act, 2),
                    "stress_threshold_mw": round(thr, 2),
                    "is_stress_day": bool(p >= thr),
                    "temp_max_c": wx_day.get(ds),
                    "label": "ForecastDay"})
                r_fcast.append((fid, sub_id, "FORECAST_FOR"))
        else:
            fc_ctx = {}
            print(f"  WARNING: {fpath.name} not found — ForecastDay nodes "
                  f"skipped for {sub} (run 05b first).")

        hh_meta = inv.drop_duplicates("household_id").set_index("household_id")
        for hh_id, m in hh_meta.iterrows():
            ward_id = f"WARD_{sub[:3].upper()}_{rng.integers(1, 6)}"
            user_id, meter_id = f"USER_{hh_id}", f"MTR_{m['meter_id']}"
            mk = float(monthly_kwh.get(hh_id, 150.0))
            users.append({"nodeId": user_id, "household_id": hh_id,
                          "tier": m["tier"], "household_size": m["household_size"],
                          "monthly_income_bdt": m["monthly_income_bdt"],
                          "avg_monthly_bill_bdt": avg_bill.get(hh_id),
                          "substation": sub, "label": "User"})
            meters.append({"nodeId": meter_id, "meter_id": m["meter_id"],
                           "household_id": hh_id, "meter_type": "smart_prepaid",
                           "substation_id": sub_id, "ward_id": ward_id,
                           "monthly_kwh_est": mk, "label": "Meter"})
            r_hasmeter.append((user_id, meter_id, "HAS_METER"))
            r_feeds.append((meter_id, sub_id, "FEEDS_FROM"))
            r_located.append((meter_id, ward_id, "LOCATED_IN"))
            r_intier.append((user_id, f"TIER_{m['tier']}", "IN_TIER",
                             m["monthly_income_bdt"]))
            r_subject.append((user_id, f"SLAB_{slab_for(mk)}", "SUBJECT_TO", mk))

        for _, a in inv.iterrows():
            app_id = f"APP_{a['appliance_id']}"
            meter_id = f"MTR_{a['meter_id']}"
            appliances.append({
                "nodeId": app_id, "appliance_id": a["appliance_id"],
                "appliance_name": a["appliance_name"],
                "category": a["category"], "rated_power_w": a["rated_power_w"],
                "actual_power_w": a["actual_power_w"],
                "duty_cycle": a["duty_cycle"], "quantity": a["quantity"],
                "household_id": a["household_id"], "label": "Appliance"})
            r_registers.append((meter_id, app_id, "REGISTERS"))
            r_installed.append((app_id, meter_id, "INSTALLED_AT"))
            flex = round(FLEX_INDEX[a["category"]]
                         * float(rng.normal(1.0, 0.08)), 3)
            r_belongs.append((app_id, f"CAT_{a['category']}", "BELONGS_TO", flex))

        # ── DR events: HIGH-IMPACT events selected by the Priority Pool ─────
        # (GCS > 0.50 — proposal flow: MOLP → Priority Pool → Twin Adapter/KG)
        pool = pd.read_csv(C.DATA_DIR / "07_priority_pool_high_impact.csv")
        log = pool[pool["substation"] == sub].rename(
            columns={"timestamp": "date"})
        sample = log.head(400)
        for i, (_, e) in enumerate(sample.iterrows()):
            ev_id = f"DR_{sub[:3].upper()}_{i:05d}"
            fc_mw = e.get("substation_pred_peak_mw")
            day_key = str(e["date"])[:10]
            day_tmax = wx_day.get(day_key)
            grid = fc_ctx.get(day_key, {})
            rebate_bdt = round(float(e["energy_curtailed_kwh"])
                               * C.DR_REBATE_BDT_KWH, 2)
            fc_txt = (f" The day-ahead BiGRU+BiLSTM forecast projected a "
                      f"substation peak of {fc_mw:.1f} MW, which triggered "
                      f"the demand-response window."
                      if pd.notna(fc_mw) else "")
            expl = (f"On {str(e['date'])[:10]}, a high-impact event "
                    f"(GCS={e['gcs_score']:.2f}) at {sub} substation "
                    f"passed the Priority Pool filter. Household {e['household_id']} "
                    f"({e['tier']} tier) shifted "
                    f"{e['energy_shifted_kwh']:.2f} kWh of schedulable load "
                    f"to off-peak hours and curtailed "
                    f"{e['energy_curtailed_kwh']:.2f} kWh of curtailable load "
                    f"during the 17:00-23:00 peak, earning a rebate of "
                    f"{e['energy_curtailed_kwh'] * C.DR_REBATE_BDT_KWH:.1f} BDT. "
                    f"Critical appliances were not affected." + fc_txt)
            dr_events.append({"nodeId": ev_id, "date": e["date"],
                              "substation": sub, "action": e["action_taken"],
                              "energy_shifted_kwh": e["energy_shifted_kwh"],
                              "energy_curtailed_kwh": e["energy_curtailed_kwh"],
                              "molp_alpha": e["molp_alpha"],
                              "gcs_score": e["gcs_score"],
                              "predicted_load_w": e["predicted_load_w"],
                              "optimised_load_w": e["optimised_load_w"],
                              "power_deviation_ratio": e["power_deviation_ratio"],
                              "substation_pred_peak_mw":
                                  e.get("substation_pred_peak_mw"),
                              # structured evidence — the LLM composes its
                              # explanation from these numbers; the stored
                              # explanation string is a debugging/template
                              # fallback, NOT the primary evidence
                              "actual_substation_peak_mw": grid.get("actual"),
                              "forecast_error_mw": grid.get("error"),
                              "stress_threshold_mw": grid.get("threshold"),
                              "rebate_bdt": rebate_bdt,
                              "temp_max_c": day_tmax,
                              # GCS decomposition (0.4*PDR + 0.3*PMN + 0.3*AA)
                              "price_multiplier_norm":
                                  e.get("price_multiplier_norm"),
                              "appliance_alpha": e.get("appliance_alpha"),
                              "explanation": expl, "label": "DrEvent"})
            r_experienced.append((f"USER_{e['household_id']}", ev_id,
                                  "EXPERIENCED", e["date"]))
            r_during.append((ev_id, "TOU_peak", "DURING"))

    # ── write node files ─────────────────────────────────────────────────────
    # encoding="utf-8" passed explicitly on every to_csv call below. pandas
    # already defaults to UTF-8 regardless of OS, so this is not required to
    # fix a bug — it is defensive/explicit so no writer in this file relies
    # on an implicit platform default.
    pd.DataFrame(substations).to_csv(C.KG_DIR / "nodes_Substation.csv", index=False, encoding="utf-8")
    pd.DataFrame(wards).to_csv(C.KG_DIR / "nodes_Ward.csv", index=False, encoding="utf-8")
    pd.DataFrame(users).to_csv(C.KG_DIR / "nodes_User.csv", index=False, encoding="utf-8")
    pd.DataFrame(meters).to_csv(C.KG_DIR / "nodes_Meter.csv", index=False, encoding="utf-8")
    pd.DataFrame(appliances).to_csv(C.KG_DIR / "nodes_Appliance.csv", index=False, encoding="utf-8")
    pd.DataFrame(CATEGORY_ROWS).to_csv(C.KG_DIR / "nodes_Category.csv", index=False, encoding="utf-8")
    pd.DataFrame(TOU_ROWS).to_csv(C.KG_DIR / "nodes_TouPeriod.csv", index=False, encoding="utf-8")
    pd.DataFrame(dr_events).to_csv(C.KG_DIR / "nodes_DrEvent.csv", index=False, encoding="utf-8")
    if bills:
        pd.DataFrame(bills).to_csv(C.KG_DIR / "nodes_MonthlyBill.csv", index=False, encoding="utf-8")
    if fdays:
        pd.DataFrame(fdays).to_csv(C.KG_DIR / "nodes_ForecastDay.csv", index=False, encoding="utf-8")
    pd.DataFrame([{"nodeId": f"TIER_{k}", "name": k.replace("_", " ").title(),
                   "income_min_bdt": v[0], "income_max_bdt": v[1],
                   "label": "SocioeconomicTier"}
                  for k, v in C.TIER_INCOME_BDT.items()]
                 ).to_csv(C.KG_DIR / "nodes_SocioeconomicTier.csv", index=False, encoding="utf-8")
    pd.DataFrame([{"nodeId": f"SLAB_{s}", "slab_id": s, "from_kwh": lo,
                   "to_kwh": hi, "rate_bdt_kwh": r, "label": "TariffSlab",
                   "tou_peak_rate": round(r * 1.35, 2),
                   "tou_offpeak_rate": round(r * 0.80, 2)}
                  for s, lo, hi, r in TARIFF_SLABS]
                 ).to_csv(C.KG_DIR / "nodes_TariffSlab.csv", index=False, encoding="utf-8")

    # ── write relationship files ─────────────────────────────────────────────
    def rel(rows, cols, name):
        pd.DataFrame(rows, columns=cols).to_csv(C.KG_DIR / name, index=False, encoding="utf-8")

    rel(r_hasmeter, [":START_ID", ":END_ID", ":TYPE"], "rels_HAS_METER.csv")
    rel(r_feeds, [":START_ID", ":END_ID", ":TYPE"], "rels_FEEDS_FROM.csv")
    rel(r_located, [":START_ID", ":END_ID", ":TYPE"], "rels_LOCATED_IN.csv")
    rel([(w["nodeId"], w["substation_id"], "PART_OF") for w in wards],
        [":START_ID", ":END_ID", ":TYPE"], "rels_PART_OF.csv")
    rel(r_intier, [":START_ID", ":END_ID", ":TYPE", "income_bdt"],
        "rels_IN_TIER.csv")
    rel(r_registers, [":START_ID", ":END_ID", ":TYPE"], "rels_REGISTERS.csv")
    rel(r_installed, [":START_ID", ":END_ID", ":TYPE"], "rels_INSTALLED_AT.csv")
    rel(r_belongs, [":START_ID", ":END_ID", ":TYPE", "flexibility_index"],
        "rels_BELONGS_TO.csv")
    rel(r_subject, [":START_ID", ":END_ID", ":TYPE", "monthly_kwh_est"],
        "rels_SUBJECT_TO.csv")
    rel(r_experienced, [":START_ID", ":END_ID", ":TYPE", "date"],
        "rels_EXPERIENCED.csv")
    rel(r_during, [":START_ID", ":END_ID", ":TYPE"], "rels_DURING.csv")
    if r_billed:
        rel(r_billed, [":START_ID", ":END_ID", ":TYPE", "month"],
            "rels_BILLED.csv")
    if r_fcast:
        rel(r_fcast, [":START_ID", ":END_ID", ":TYPE"],
            "rels_FORECAST_FOR.csv")

    write_cypher()
    write_llm_context()
    print(f"KG written to {C.KG_DIR}: "
          f"{len(users)} users, {len(appliances)} appliances, "
          f"{len(dr_events)} DR events, {len(bills)} monthly bills, "
          f"{len(fdays)} forecast days")


def write_cypher():
    # FIX: encoding="utf-8" added — without it, Path.write_text() falls back
    # to locale.getpreferredencoding(False), which is cp1252 on Windows and
    # cannot represent non-Latin-1 characters that may appear in future edits
    # to this Cypher script (e.g. non-ASCII comments, Bengali labels).
    (C.KG_DIR / "load_kg.cypher").write_text("""\
// Load the Kalyanpur+Dhanmondi DR knowledge graph (put CSVs in /import)
CREATE CONSTRAINT IF NOT EXISTS FOR (n:Node) REQUIRE n.nodeId IS UNIQUE;
LOAD CSV WITH HEADERS FROM 'file:///nodes_Substation.csv' AS r
  CREATE (:Substation:Node {nodeId:r.nodeId, name:r.name, utility:r.utility,
    latitude:toFloat(r.latitude), longitude:toFloat(r.longitude)});
LOAD CSV WITH HEADERS FROM 'file:///nodes_Ward.csv' AS r
  CREATE (:Ward:Node {nodeId:r.nodeId, name:r.name});
LOAD CSV WITH HEADERS FROM 'file:///nodes_User.csv' AS r
  CREATE (:User:Node {nodeId:r.nodeId, household_id:r.household_id, tier:r.tier,
    monthly_income_bdt:toInteger(r.monthly_income_bdt), substation:r.substation});
LOAD CSV WITH HEADERS FROM 'file:///nodes_Meter.csv' AS r
  CREATE (:Meter:Node {nodeId:r.nodeId, meter_id:r.meter_id,
    monthly_kwh_est:toFloat(r.monthly_kwh_est)});
LOAD CSV WITH HEADERS FROM 'file:///nodes_Appliance.csv' AS r
  CREATE (:Appliance:Node {nodeId:r.nodeId, appliance_name:r.appliance_name,
    category:r.category, rated_power_w:toInteger(r.rated_power_w)});
LOAD CSV WITH HEADERS FROM 'file:///nodes_Category.csv' AS r
  CREATE (:Category:Node {nodeId:r.nodeId, name:r.name,
    explanation_template:r.explanation_template, molp_constraint:r.molp_constraint});
LOAD CSV WITH HEADERS FROM 'file:///nodes_SocioeconomicTier.csv' AS r
  CREATE (:SocioeconomicTier:Node {nodeId:r.nodeId, name:r.name});
LOAD CSV WITH HEADERS FROM 'file:///nodes_TariffSlab.csv' AS r
  CREATE (:TariffSlab:Node {nodeId:r.nodeId, slab_id:toInteger(r.slab_id),
    rate_bdt_kwh:toFloat(r.rate_bdt_kwh)});
LOAD CSV WITH HEADERS FROM 'file:///nodes_TouPeriod.csv' AS r
  CREATE (:TouPeriod:Node {nodeId:r.nodeId, name:r.name,
    multiplier:toFloat(r.multiplier)});
LOAD CSV WITH HEADERS FROM 'file:///nodes_DrEvent.csv' AS r
  CREATE (:DrEvent:Node {nodeId:r.nodeId, date:r.date, action:r.action,
    substation_pred_peak_mw:toFloat(r.substation_pred_peak_mw),
    temp_max_c:toFloat(r.temp_max_c),
    power_deviation_ratio:toFloat(r.power_deviation_ratio),
    price_multiplier_norm:toFloat(r.price_multiplier_norm),
    appliance_alpha:toFloat(r.appliance_alpha),
    gcs_score:toFloat(r.gcs_score),
    actual_substation_peak_mw:toFloat(r.actual_substation_peak_mw),
    forecast_error_mw:toFloat(r.forecast_error_mw),
    stress_threshold_mw:toFloat(r.stress_threshold_mw),
    rebate_bdt:toFloat(r.rebate_bdt),
    explanation:r.explanation});
LOAD CSV WITH HEADERS FROM 'file:///nodes_ForecastDay.csv' AS r
  CREATE (:ForecastDay:Node {nodeId:r.nodeId, date:r.date,
    substation:r.substation, pred_peak_mw:toFloat(r.pred_peak_mw),
    actual_peak_mw:toFloat(r.actual_peak_mw),
    forecast_error_mw:toFloat(r.forecast_error_mw),
    stress_threshold_mw:toFloat(r.stress_threshold_mw),
    is_stress_day:toBoolean(r.is_stress_day),
    temp_max_c:toFloat(r.temp_max_c)});
LOAD CSV WITH HEADERS FROM 'file:///nodes_MonthlyBill.csv' AS r
  CREATE (:MonthlyBill:Node {nodeId:r.nodeId, month:r.month,
    baseline_bill_bdt:toFloat(r.baseline_bill_bdt),
    optimised_bill_bdt:toFloat(r.optimised_bill_bdt),
    saving_bdt:toFloat(r.saving_bdt), saving_pct:toFloat(r.saving_pct),
    energy_kwh:toFloat(r.energy_kwh),
    mean_temp_c:toFloat(r.mean_temp_c), max_temp_c:toFloat(r.max_temp_c),
    hot_days:toInteger(r.hot_days),
    mean_humidity_pct:toFloat(r.mean_humidity_pct),
    rain_mm:toFloat(r.rain_mm),
    discomfort_bdt_eq:toFloat(r.discomfort_bdt_eq)});
// Relationships (repeat pattern for every rels_*.csv)
LOAD CSV WITH HEADERS FROM 'file:///rels_HAS_METER.csv' AS r
  MATCH (a:Node {nodeId:r.`:START_ID`}), (b:Node {nodeId:r.`:END_ID`})
  CREATE (a)-[:HAS_METER]->(b);
LOAD CSV WITH HEADERS FROM 'file:///rels_FEEDS_FROM.csv' AS r
  MATCH (a:Node {nodeId:r.`:START_ID`}), (b:Node {nodeId:r.`:END_ID`})
  CREATE (a)-[:FEEDS_FROM]->(b);
LOAD CSV WITH HEADERS FROM 'file:///rels_LOCATED_IN.csv' AS r
  MATCH (a:Node {nodeId:r.`:START_ID`}), (b:Node {nodeId:r.`:END_ID`})
  CREATE (a)-[:LOCATED_IN]->(b);
LOAD CSV WITH HEADERS FROM 'file:///rels_PART_OF.csv' AS r
  MATCH (a:Node {nodeId:r.`:START_ID`}), (b:Node {nodeId:r.`:END_ID`})
  CREATE (a)-[:PART_OF]->(b);
LOAD CSV WITH HEADERS FROM 'file:///rels_IN_TIER.csv' AS r
  MATCH (a:Node {nodeId:r.`:START_ID`}), (b:Node {nodeId:r.`:END_ID`})
  CREATE (a)-[:IN_TIER {income_bdt:toInteger(r.income_bdt)}]->(b);
LOAD CSV WITH HEADERS FROM 'file:///rels_REGISTERS.csv' AS r
  MATCH (a:Node {nodeId:r.`:START_ID`}), (b:Node {nodeId:r.`:END_ID`})
  CREATE (a)-[:REGISTERS]->(b);
LOAD CSV WITH HEADERS FROM 'file:///rels_INSTALLED_AT.csv' AS r
  MATCH (a:Node {nodeId:r.`:START_ID`}), (b:Node {nodeId:r.`:END_ID`})
  CREATE (a)-[:INSTALLED_AT]->(b);
LOAD CSV WITH HEADERS FROM 'file:///rels_BELONGS_TO.csv' AS r
  MATCH (a:Node {nodeId:r.`:START_ID`}), (b:Node {nodeId:r.`:END_ID`})
  CREATE (a)-[:BELONGS_TO {flexibility_index:toFloat(r.flexibility_index)}]->(b);
LOAD CSV WITH HEADERS FROM 'file:///rels_SUBJECT_TO.csv' AS r
  MATCH (a:Node {nodeId:r.`:START_ID`}), (b:Node {nodeId:r.`:END_ID`})
  CREATE (a)-[:SUBJECT_TO {monthly_kwh_est:toFloat(r.monthly_kwh_est)}]->(b);
LOAD CSV WITH HEADERS FROM 'file:///rels_EXPERIENCED.csv' AS r
  MATCH (a:Node {nodeId:r.`:START_ID`}), (b:Node {nodeId:r.`:END_ID`})
  CREATE (a)-[:EXPERIENCED {date:r.date}]->(b);
LOAD CSV WITH HEADERS FROM 'file:///rels_DURING.csv' AS r
  MATCH (a:Node {nodeId:r.`:START_ID`}), (b:Node {nodeId:r.`:END_ID`})
  CREATE (a)-[:DURING]->(b);
LOAD CSV WITH HEADERS FROM 'file:///rels_BILLED.csv' AS r
  MATCH (a:Node {nodeId:r.`:START_ID`}), (b:Node {nodeId:r.`:END_ID`})
  CREATE (a)-[:BILLED {month:r.month}]->(b);
LOAD CSV WITH HEADERS FROM 'file:///rels_FORECAST_FOR.csv' AS r
  MATCH (a:Node {nodeId:r.`:START_ID`}), (b:Node {nodeId:r.`:END_ID`})
  CREATE (a)-[:FORECAST_FOR]->(b);
""", encoding="utf-8")


def write_llm_context():
    # FIX: encoding="utf-8" added here too. This is the function that was
    # actually raising the UnicodeEncodeError, because the docstring below
    # contains the Greek letter alpha (U+03B1) in "tier (alpha weight)" and
    # cp1252 (Windows' locale default) has no byte mapping for it.
    (C.KG_DIR / "llm_context.md").write_text("""\
# Using this knowledge graph with an LLM (GraphRAG)

Purpose: let an LLM answer customer questions such as
"Why was my AC rescheduled on 2025-05-14?" or
"How much did household DHH0042 save last month, and at what comfort cost?"

## Retrieval pattern
1. Resolve the entity (household / appliance / date) from the question.
2. Cypher-retrieve the local subgraph, e.g.:
   MATCH (u:User {household_id:$hh})-[:EXPERIENCED]->(e:DrEvent)
   WHERE e.date = $date
   OPTIONAL MATCH (u)-[:HAS_METER]->(m)-[:REGISTERS]->(a:Appliance)
            -[:BELONGS_TO]->(c:Category)
   RETURN u, e, collect({appliance:a.appliance_name, category:c.name,
                         template:c.explanation_template})
3. Serialise the returned nodes/edges as short triples
   (User KHH0007) -IN_TIER-> (middle, 35k-65k BDT)
   (DrEvent DR_KAL_00012) date=2025-05-14 shifted=1.42kWh curtailed=0.36kWh
4. Feed triples + the Category.explanation_template into the LLM prompt and
   ask it to compose the answer. DrEvent.explanation already contains a
   rendered fallback sentence, so even a template-only system is grounded.

## Why the graph grounds the LLM
- Every DR action links to the exact MOLP inputs: tier (alpha weight),
  category constraint (must_run / flexible_start / power_reduction_allowed),
  TOU period and tariff slab -- the LLM never has to invent numbers.
- flexibility_index on BELONGS_TO lets the LLM rank which appliance was
  chosen first and say so ("your air conditioner is 16x more flexible
  than your refrigerator").
- Substation nodes carry the real BPDB yearly peak means, so grid-level
  answers ("why was 2025-05-14 a stress day?") tie back to observed data.
- MonthlyBill nodes ((:User)-[:BILLED {month}]->(:MonthlyBill)) hold each
  household's baseline vs optimised bill, saving and discomfort per month,
  so "how much did DHH0042 save last month, and at what comfort cost?" is a
  single-hop lookup:
    MATCH (u:User {household_id:$hh})-[:BILLED]->(b:MonthlyBill {month:$ym})
    RETURN b.baseline_bill_bdt, b.optimised_bill_bdt, b.saving_bdt,
           b.discomfort_bdt_eq
""", encoding="utf-8")


if __name__ == "__main__":
    main()