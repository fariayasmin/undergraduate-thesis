"""
03_feeder_composition.py — realistic customer composition for the Kalyanpur
and Dhanmondi substations.

Matching is BY CONSUMPTION COMPOSITION, not by scaling: the number of
households (per socioeconomic tier), shops, schools, clinics, workshops,
hospitals, street-light blocks, pumps and mosques is chosen so that the sum
of their design coincident peaks equals the substation's observed mean daily
peak in the BPDB data. Named Dhanmondi hospitals are real, map-verifiable
institutions on that feeder.

Outputs:
  data/03_kalyanpur_composition.csv
  data/03_dhanmondi_composition.csv
"""

import pandas as pd
import config as C


def build_composition(substation: str) -> pd.DataFrame:
    target_mw = C.BPDB_MEAN_DAILY_PEAK_MW[substation]
    shares    = C.COMPOSITION[substation]
    rows = []

    # ── Residential: households per tier ─────────────────────────────────────
    res_target_kw = shares["residential"] * target_mw * 1000
    mix = C.TIER_MIX[substation]
    avg_kw = sum(mix[t] * C.TIER_COINCIDENT_PEAK_KW[t] for t in mix)
    n_households = int(round(res_target_kw / avg_kw, -2))   # round to 100s
    for tier, frac in mix.items():
        n = int(round(n_households * frac))
        rows.append({
            "category": "residential", "subcategory": tier, "units": n,
            "per_unit_peak_kw": C.TIER_COINCIDENT_PEAK_KW[tier],
            "design_peak_kw": round(n * C.TIER_COINCIDENT_PEAK_KW[tier], 1),
        })

    # ── Commercial ────────────────────────────────────────────────────────────
    com_kw = shares["commercial"] * target_mw * 1000
    n_shops   = int(round(0.75 * com_kw / C.UNIT_PEAK_KW["commercial_shop"], -1))
    n_offices = int(round(0.25 * com_kw / C.UNIT_PEAK_KW["office_smb"], -1))
    rows.append({"category": "commercial", "subcategory": "commercial_shop",
                 "units": n_shops, "per_unit_peak_kw": C.UNIT_PEAK_KW["commercial_shop"],
                 "design_peak_kw": round(n_shops * C.UNIT_PEAK_KW["commercial_shop"], 1)})
    rows.append({"category": "commercial", "subcategory": "office_smb",
                 "units": n_offices, "per_unit_peak_kw": C.UNIT_PEAK_KW["office_smb"],
                 "design_peak_kw": round(n_offices * C.UNIT_PEAK_KW["office_smb"], 1)})

    # ── Educational ───────────────────────────────────────────────────────────
    edu_kw = shares["educational"] * target_mw * 1000
    n_schools = max(1, int(round(edu_kw / C.UNIT_PEAK_KW["school"])))
    rows.append({"category": "educational", "subcategory": "school",
                 "units": n_schools, "per_unit_peak_kw": C.UNIT_PEAK_KW["school"],
                 "design_peak_kw": round(n_schools * C.UNIT_PEAK_KW["school"], 1)})

    # ── Health ────────────────────────────────────────────────────────────────
    health_kw = shares["health"] * target_mw * 1000
    if substation == "Dhanmondi":
        named_kw = sum(C.DHANMONDI_HOSPITALS.values())
        for name, pk in C.DHANMONDI_HOSPITALS.items():
            rows.append({"category": "health", "subcategory": f"hospital::{name}",
                         "units": 1, "per_unit_peak_kw": pk,
                         "design_peak_kw": pk})
        rem = max(0.0, health_kw - named_kw)
        n_clinics = int(round(rem / C.UNIT_PEAK_KW["clinic_small"]))
    else:
        n_clinics = max(1, int(round(health_kw / C.UNIT_PEAK_KW["clinic_small"])))
    rows.append({"category": "health", "subcategory": "clinic_small",
                 "units": n_clinics, "per_unit_peak_kw": C.UNIT_PEAK_KW["clinic_small"],
                 "design_peak_kw": round(n_clinics * C.UNIT_PEAK_KW["clinic_small"], 1)})

    # ── Small industrial workshops ────────────────────────────────────────────
    ind_kw = shares["industrial"] * target_mw * 1000
    n_ws = max(0, int(round(ind_kw / C.UNIT_PEAK_KW["workshop"], -1)))
    rows.append({"category": "industrial", "subcategory": "workshop",
                 "units": n_ws, "per_unit_peak_kw": C.UNIT_PEAK_KW["workshop"],
                 "design_peak_kw": round(n_ws * C.UNIT_PEAK_KW["workshop"], 1)})

    # ── Infrastructure: street light blocks, WASA pumps, mosques ─────────────
    infra_kw = shares["infrastructure"] * target_mw * 1000
    n_pumps  = max(2, int(round(0.45 * infra_kw / C.UNIT_PEAK_KW["wasa_pump"])))
    n_sl     = max(5, int(round(0.35 * infra_kw / C.UNIT_PEAK_KW["streetlight_block"])))
    n_mosque = max(5, int(round(0.20 * infra_kw / C.UNIT_PEAK_KW["mosque"])))
    for sub_cat, n in [("wasa_pump", n_pumps), ("streetlight_block", n_sl),
                       ("mosque", n_mosque)]:
        rows.append({"category": "infrastructure", "subcategory": sub_cat,
                     "units": n, "per_unit_peak_kw": C.UNIT_PEAK_KW[sub_cat],
                     "design_peak_kw": round(n * C.UNIT_PEAK_KW[sub_cat], 1)})

    df = pd.DataFrame(rows)
    df.insert(0, "substation", substation)
    df["share_of_design_peak_pct"] = (
        100 * df["design_peak_kw"] / df["design_peak_kw"].sum()).round(2)
    return df


def main():
    for sub in C.SUBSTATIONS:
        df = build_composition(sub)
        out = C.DATA_DIR / f"03_{sub.lower()}_composition.csv"
        df.to_csv(out, index=False)
        tot = df["design_peak_kw"].sum() / 1000
        print(f"\n{sub}: design peak {tot:.1f} MW "
              f"(BPDB 2024 mean daily peak {C.BPDB_MEAN_DAILY_PEAK_MW[sub]} MW)")
        print(df.groupby("category")["design_peak_kw"].sum().round(0).to_string())
        print(f"written: {out}")


if __name__ == "__main__":
    main()