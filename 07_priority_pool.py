"""
07_priority_pool.py — Priority Pooling with Grid Criticality Score (GCS),
exactly as specified in the thesis proposal and the earlier GenTwin
Module 07:

    GCS = 0.4 × PDR + 0.3 × PMN + 0.3 × AA          (threshold 0.50)

      PDR  power_deviation_ratio   — magnitude of the MOLP throttle
      PMN  price_multiplier_norm   — pricing shock (1.0 on grid-stress days)
      AA   appliance_alpha         — priority of the highest affected
                                     category (Curtailable 0.7 > Schedulable
                                     0.4; Critical never appears by MOLP
                                     constraint)

Role in the architecture (proposal, slide 11):
    LSTM Forecasting → MOLP Optimization → PRIORITY POOL → Twin Adapter /
    LLM Reasoning → Neo4j KG → Explainability

MOLP emits ~10⁵ household-day events; feeding all of them to the LLM would
overflow its context. The Priority Pool keeps only high-GCS events (an EV
charger delayed 10 min is dropped; an AC throttled 50% during a price spike
is flagged High-Impact and passed to the Twin Adapter).

Outputs (data/):
  07_optimization_events_log.csv     full MOLP log enriched with gcs_score
                                     (same schema as the old GenTwin file)
  07_priority_pool_high_impact.csv   stress-day events with GCS > 0.50
                                     → LLM/KG layer (old Module-07 gating)
  07_llm_payload.jsonl               top-N events serialised as grounded
                                     context blocks for the LLM Reasoning
                                     Engine (Twin Adapter input)
  07_priority_pool_stats.csv         context-reduction statistics
  07_gcs_distribution.png            GCS histogram with the 0.50 threshold
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config as C

GCS_THRESHOLD = 0.50                     # same as earlier GenTwin module
W_PDR, W_PMN, W_AA = 0.4, 0.3, 0.3
TOP_N_LLM = 500                          # events serialised for the LLM


def compute_gcs(df: pd.DataFrame) -> pd.Series:
    return (W_PDR * df["power_deviation_ratio"]
            + W_PMN * df["price_multiplier_norm"]
            + W_AA * df["appliance_alpha"]).round(4)


def build_llm_payload(pool: pd.DataFrame) -> list:
    """Serialise the highest-GCS events as grounded context blocks that the
    Twin Adapter feeds to the LLM Reasoning Engine (see proposal prompt:
    decision log + customer-facing XAI explanation)."""
    inv = {}
    for sub in C.SUBSTATIONS:
        i = pd.read_csv(C.DATA_DIR / f"04_{sub.lower()}_household_inventory.csv")
        inv[sub] = i.groupby(["household_id", "category"])["appliance_name"] \
                    .apply(lambda s: sorted(set(s))[:6]).unstack(fill_value=[])

    payload = []
    for _, e in pool.head(TOP_N_LLM).iterrows():
        hh, sub = e["household_id"], e["substation"]
        apps = inv[sub].loc[hh] if hh in inv[sub].index else {}
        payload.append({
            "event_id": f"EVT_{sub[:3].upper()}_{e.name:06d}",
            "gcs_score": float(e["gcs_score"]),
            "grid_context": {
                "substation": sub, "date": str(e["timestamp"])[:10],
                "tou_period": e["tou_period"],
                "tou_multiplier": float(e["tou_multiplier"]),
                "grid_stress_event": bool(e["is_stress_day"]),
                "price_shock_norm": float(e["price_multiplier_norm"]),
                # day-ahead BiGRU+BiLSTM substation forecast (05b) that
                # triggered/graded this DR event at the SGO level
                "substation_pred_peak_mw": (
                    float(e["substation_pred_peak_mw"])
                    if "substation_pred_peak_mw" in e
                    and pd.notna(e["substation_pred_peak_mw"]) else None),
            },
            "molp_decision": {
                "predicted_load_w": float(e["predicted_load_w"]),
                "optimised_load_w": float(e["optimised_load_w"]),
                "throttle_ratio": float(e["power_deviation_ratio"]),
                "action_taken": e["action_taken"],
                "molp_alpha": float(e["molp_alpha"]),
                "energy_shifted_kwh": float(e["energy_shifted_kwh"]),
                "energy_curtailed_kwh": float(e["energy_curtailed_kwh"]),
                "rebate_bdt": round(float(e["energy_curtailed_kwh"])
                                    * C.DR_REBATE_BDT_KWH
                                    * int(e["is_stress_day"]), 2),
            },
            "household_context": {
                "household_id": hh, "tier": e["tier"],
                "tariff_rate_bdt_kwh": float(e["tou_rate_bdt_kwh"]),
                "schedulable_appliances": list(apps.get("Schedulable", [])),
                "curtailable_appliances": list(apps.get("Curtailable", [])),
                "critical_protected": list(apps.get("Critical", [])),
            },
            "llm_task": ("Explain to the customer what action was taken, "
                         "why (grid context), and how it benefits them "
                         "(cost/QoE). Ground every number in this payload; "
                         "do not invent values."),
        })
    return payload


def main():
    logs = []
    for sub in C.SUBSTATIONS:
        logs.append(pd.read_csv(
            C.DATA_DIR / f"06_{sub.lower()}_molp_events_log.csv",
            parse_dates=["timestamp"]))
    events = pd.concat(logs, ignore_index=True)

    events["gcs_score"] = compute_gcs(events)
    events.to_csv(C.DATA_DIR / "07_optimization_events_log.csv", index=False)

    # Old GenTwin Module-07 gating: "Grid event log (stress events only,
    # GCS > threshold)" — routine daily shifts/curtailments are dropped, only
    # grid-stress events with a high criticality score reach the LLM layer.
    pool = (events[(events["is_stress_day"] == 1)
                   & (events["gcs_score"] > GCS_THRESHOLD)]
            .sort_values("gcs_score", ascending=False).reset_index(drop=True))
    pool.to_csv(C.DATA_DIR / "07_priority_pool_high_impact.csv", index=False)

    payload = build_llm_payload(pool)
    with open(C.DATA_DIR / "07_llm_payload.jsonl", "w") as f:
        for p in payload:
            f.write(json.dumps(p) + "\n")

    # ── stats: how much LLM context did the pool save? ───────────────────────
    stats = pd.DataFrame([{
        "molp_events_total": len(events),
        "high_impact_events": len(pool),
        "context_reduction_pct": round(100 * (1 - len(pool) / len(events)), 2),
        "gcs_threshold": GCS_THRESHOLD,
        "llm_payload_events": len(payload),
        "mean_gcs_all": events["gcs_score"].mean().round(4),
        "mean_gcs_pool": pool["gcs_score"].mean().round(4),
        "pct_pool_on_stress_days": round(100 * pool["is_stress_day"].mean(), 1),
    }])
    stats.to_csv(C.DATA_DIR / "07_priority_pool_stats.csv", index=False)
    print(stats.T.to_string(header=False))
    print(pool.groupby(["substation", "action_taken"])["gcs_score"]
              .agg(["count", "mean"]).round(3).to_string())

    # ── plot ─────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.hist(events["gcs_score"], bins=60, color="steelblue", alpha=0.8)
    ax.axvline(GCS_THRESHOLD, color="tomato", ls="--", lw=1.5,
               label=f"GCS threshold = {GCS_THRESHOLD} "
                     f"(drops {100*(1-len(pool)/len(events)):.1f}% of events)")
    ax.set_xlabel("Grid Criticality Score  (0.4·PDR + 0.3·PMN + 0.3·AA)")
    ax.set_ylabel("MOLP events")
    ax.set_title("Priority Pool — GCS distribution and high-impact cut-off")
    ax.legend()
    fig.tight_layout()
    fig.savefig(C.DATA_DIR / "07_gcs_distribution.png", dpi=130)
    plt.close(fig)
    print(f"\nPriority Pool outputs written to {C.DATA_DIR}")


if __name__ == "__main__":
    main()