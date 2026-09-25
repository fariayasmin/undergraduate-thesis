#!/usr/bin/env python3
"""
Stage 0 runner - BERC/BPDB retail tariff.

    python 00_tariff_bpdb.py --verify     # proof-read the transcribed gazette
    python 00_tariff_bpdb.py --table      # full rate schedule
    python 00_tariff_bpdb.py --mu         # mu(t) under each substation's T^pk
    python 00_tariff_bpdb.py --bills      # worked example bills
    python 00_tariff_bpdb.py --export     # write outputs/tariff_reference.json

--verify recomputes the gazette's own peak/flat and off-peak/flat ratios from
the transcribed numbers. If a rate had been mistyped the ratios would not land
on the two families the order actually uses, so it doubles as a proof-reading
test of the transcription.
"""

import argparse
import json
import sys

from gentwin import config as cfg
from gentwin import peak_window as pw
from gentwin import tariff as T


def cmd_verify() -> bool:
    r = T.verify()
    print(f"Source: {T.GAZETTE_REF}\n")
    print(f"{'class':7} {'V':5} {'flat':>7} {'off':>7} {'peak':>7} "
          f"{'mu_off':>7} {'mu_pk':>7}")
    print("-" * 52)
    for x in r["observed_ratios"]:
        tc = T.RETAIL_TARIFF[x["class"]]
        print(f"{x['class']:7} {x['voltage']:5} {tc.flat:7.2f} {tc.off_peak:7.2f} "
              f"{tc.peak:7.2f} {x['mu_off']:7.4f} {x['mu_peak']:7.4f}")
    print(f"\nDerived families: {json.dumps(T.MU_FAMILIES)}")
    print("\nChecks")
    print("-" * 52)
    for c in r["checks"]:
        print(f"  [{'PASS' if c['pass'] else 'FAIL'}] {c['check']}"
              + (f"\n         {c['detail']}" if c["detail"] else ""))
    print(f"\nALL PASS: {r['all_pass']}")
    return r["all_pass"]


def cmd_table() -> None:
    print(f"BERC retail tariff, effective June 2026 billing month")
    print(f"{T.GAZETTE_REF}\n")
    print("LT-A residential slabs (cumulative, Tk/kWh)")
    print(f"  lifeline 0-{T.LT_A_LIFELINE_LIMIT_KWH} units   "
          f"{T.LT_A_LIFELINE_RATE:6.2f}   "
          f"(qualifying tariff: forfeited entirely above 50 units)")
    lower = 0
    for upper, rate in T.LT_A_SLABS:
        band = f"{lower + 1}-{upper}" if upper else f"above {lower}"
        print(f"  {band:<20}   {rate:6.2f}")
        lower = upper if upper else lower
    print(f"\n  demand charge {T.RETAIL_TARIFF['LT-A'].demand_charge:.2f} Tk/kW/month"
          f"   VAT {T.VAT_RATE:.0%}   prepaid rebate {T.PREPAID_REBATE:.1%}\n")

    print(f"{'class':7} {'name':46} {'flat':>7} {'off':>7} {'peak':>7} {'demand':>7}")
    print("-" * 84)
    for code, tc in T.RETAIL_TARIFF.items():
        f = f"{tc.flat:7.2f}" if tc.flat else ("  slabs" if tc.slabs else "      -")
        o = f"{tc.off_peak:7.2f}" if tc.off_peak else "      -"
        p = f"{tc.peak:7.2f}" if tc.peak else "      -"
        print(f"{code:7} {tc.name_en[:46]:46} {f} {o} {p} {tc.demand_charge:7.2f}")

    print("\nOfficial ToU windows (order footnotes 4-6)")
    def _fmt(s):
        ss = sorted(s)
        runs, start = [], ss[0]
        for a, b in zip(ss, ss[1:] + [None]):
            if b != (a + 1):
                runs.append(f"{cfg.slot_to_clock(start)}-"
                            f"{cfg.slot_to_clock((a + 1) % 48)}")
                start = b
        return ", ".join(runs)
    print(f"  peak             {_fmt(T.OFFICIAL_TOU_PEAK)}")
    print(f"  off-peak         {_fmt(T.OFFICIAL_TOU_OFFPEAK)}")
    print(f"  off-peak (EV)    {_fmt(T.EV_TOU_OFFPEAK)}")
    print(f"  super off-peak   {_fmt(T.EV_TOU_SUPER_OFFPEAK)}   (LT-D3, MT-7 only)")


def cmd_mu() -> None:
    print("mu(t) as used by the model - EQ (19), (34)  [Issue A]\n")
    print("BILLING and the DR ACTIVATION window are two separate things now:")
    print("  BILLING       mu(t) always uses the OFFICIAL gazette window,")
    print("                17:00-23:00, for any class with a real ToU row.")
    print("                Never varies by substation.")
    print("  DR ACTIVATION x_{h,t} exists only on the substation's own")
    print("                OBSERVED (adaptive) peak window - it decides WHEN")
    print("                demand response may fire, never what is billed.\n")
    print("LT-A (residential) and LT-D1 (this model's Hospital and Educational")
    print("archetypes) carry NO gazette ToU row at all: mu(t) = 1.0 at every")
    print("slot under EITHER window, by policy, not by approximation.")
    print("Shifting energy earns these consumers nothing; only curtailing")
    print("(using less) can still save them money.")
    print(f"(MU_RESIDENTIAL_IS_SCENARIO = {T.MU_RESIDENTIAL_IS_SCENARIO} - "
          f"set True only to run an explicit 'what if a flat-rate class had "
          f"ToU' ablation, never to report current policy.)\n")
    for code in ("LT-A", "LT-D1"):
        mu_pk, mu_off = T.mu_peak_off(code)
        print(f"  {code:6} mu^pk {mu_pk:.2f} / mu^off {mu_off:.2f}  "
              f"(flat - {T.RETAIL_TARIFF[code].name_en})")
    print()
    official = sorted(T.OFFICIAL_TOU_PEAK)
    official_label = (f"{cfg.slot_to_clock(official[0])}-"
                      f"{cfg.slot_to_clock(official[-1] + 1)}")
    mu_billed = T.mu_profile("LT-C1", peak_slots=T.OFFICIAL_TOU_PEAK)   # what LT-C1/LT-E etc. actually pay
    print(f"Billing window (all ToU classes, e.g. LT-C1): {official_label}")
    print(f"  mu^pk {max(mu_billed):.2f} / mu^off {min(mu_billed):.2f}   "
          f"({len(official)} peak slots, {48 - len(official)} off-peak) - "
          f"IDENTICAL at both substations\n")
    for name in cfg.SUBSTATIONS:
        observed = sorted(pw.get_peak_slots(name))   # dynamic - Issue A/label bug
        # What LT-C1 is actually BILLED, evaluated AT the DR-activation
        # slots - this is the number that matters for the "does shifting out
        # of the DR window help or hurt the bill" question.
        billed_at_dr_slots = [mu_billed[t] for t in observed]
        overlap = set(official) & set(observed)
        print(f"{name}  DR activation window (Industrial/LT-C1, illustrative)")
        print(f"  T^pk (observed, DR activation)  "
              f"{pw.format_window(observed)}")
        print(f"  T^pk (gazette, BILLING)         {official_label}")
        print(f"  overlap                         {len(overlap)} of "
              f"{len(observed)} slots"
              + ("   << MISALIGNED - the case for a substation-specific "
                 "ToU proposal, not what is billed today"
                 if not overlap else ""))
        print(f"  mu(t) ACTUALLY BILLED at the DR-activation slots: "
              f"{min(billed_at_dr_slots):.2f}-{max(billed_at_dr_slots):.2f} "
              + ("(all OFF-PEAK by the gazette clock, despite this being "
                 "the substation's own physical peak - shifting load OUT "
                 "of this window and into the recovery window, which "
                 "includes the real 17:00-23:00 billing peak, RAISES the "
                 "bill)" if set(billed_at_dr_slots) == {min(mu_billed)}
                 else "(billing and DR windows overlap here, so shifting "
                      "out still behaves the way the family ratio implies)"))
        print()


def cmd_bills() -> None:
    print("Worked bills, BERC June 2026\n")
    print(f"{'class':7} {'kWh':>6} {'energy':>10} {'demand':>8} {'VAT':>8} "
          f"{'total':>10} {'eff':>7} {'marginal':>9}")
    print("-" * 70)
    for code, kwh, kw in [("LT-A", 45, 2.0), ("LT-A", 180, 2.0),
                          ("LT-A", 350, 3.0), ("LT-A", 720, 5.0),
                          ("LT-E", 2500, 15.0), ("LT-D1", 40000, 200.0),
                          ("LT-C1", 12000, 60.0)]:
        b = T.monthly_bill(code, kwh, sanctioned_kw=kw)
        marg = (f"{T.marginal_rate_lt_a(kwh):9.2f}" if code == "LT-A"
                else f"{T.RETAIL_TARIFF[code].flat:9.2f}")
        print(f"{code:7} {kwh:6} {b['energy_charge_bdt']:10.2f} "
              f"{b['demand_charge_bdt']:8.2f} {b['vat_bdt']:8.2f} "
              f"{b['total_bdt']:10.2f} {b['effective_rate_bdt_per_kwh']:7.2f} {marg}")

    blend = T.mt5_residential_blend(50000)
    print(f"\nFootnote 7 (MT-5 with ~80% residential use - university halls, "
          f"cantonments):\n  blended rate "
          f"{blend['blended_rate_bdt_per_kwh']:.4f} Tk/kWh "
          f"= 0.20x12.58 + 0.72x9.36 + 0.08x17.35")

    print("\nArchetype -> tariff class mapping used by the model.")
    print("(The archetype is a GENERATOR only: the revised formulation carries")
    print(" lambda_h, gamma^sh_h, gamma^cu_h and y^max_h per HOUSEHOLD.)")
    print(f"  {'archetype':12} {'category':13} {'tariff':7} "
          f"{'lambda_h':>16} {'gamma^cu_h':>14} {'y^max_h':>12}")
    for arch, spec in cfg.CONSUMER_ARCHETYPES.items():
        tc = T.RETAIL_TARIFF[spec["tariff_class"]]
        lm_, ls = spec["lambda_h"]; gm, gs = spec["gamma_cu_h"]
        ym, ys = spec["y_max_h"]
        print(f"  {arch:12} {spec['category']:13} {spec['tariff_class']:7} "
              f"{lm_:8.2f} +/-{ls:<4.2f} {gm:8.1f} +/-{gs:<4.1f} "
              f"{ym:6.2f} +/-{ys:<4.2f}   ({tc.name_en})")


def cmd_export() -> None:
    payload = {
        "source": T.GAZETTE_REF,
        "verification": T.verify(),
        "lt_a_slabs": T.LT_A_SLABS,
        "lt_a_lifeline": {"limit_kwh": T.LT_A_LIFELINE_LIMIT_KWH,
                          "rate": T.LT_A_LIFELINE_RATE},
        "classes": {c: {k: v for k, v in vars(tc).items() if k != "slabs"}
                    for c, tc in T.RETAIL_TARIFF.items()},
        "mu_families": T.MU_FAMILIES,
        "mu_residential_is_scenario": T.MU_RESIDENTIAL_IS_SCENARIO,
        "official_tou": {k: sorted(v) for k, v in
                         {"peak": T.OFFICIAL_TOU_PEAK,
                          "off_peak": T.OFFICIAL_TOU_OFFPEAK,
                          "ev_off_peak": T.EV_TOU_OFFPEAK,
                          "ev_super_off_peak": T.EV_TOU_SUPER_OFFPEAK}.items()},
        "model_peak_windows": {
            n: {"slots": list(s["t_pk_slots"]), "label": s["t_pk_label"],
                "source": "observed BPDB peak time distribution"}
            for n, s in cfg.SUBSTATIONS.items()},
        "vat_rate": T.VAT_RATE,
        "prepaid_rebate": T.PREPAID_REBATE,
    }
    p = cfg.OUT_DIR / "tariff_reference.json"
    p.write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")
    print(f"wrote {p}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    for flag in ("verify", "table", "mu", "bills", "export"):
        p.add_argument(f"--{flag}", action="store_true")
    p.add_argument("--all", action="store_true")
    a = p.parse_args()
    if not any(vars(a).values()):
        p.print_help()
        return 1

    ok = True
    for name, fn in (("verify", cmd_verify), ("table", cmd_table),
                     ("mu", cmd_mu), ("bills", cmd_bills), ("export", cmd_export)):
        if getattr(a, name) or a.all:
            print(f"\n{'=' * 72}\n{name.upper()}\n{'=' * 72}")
            res = fn()
            if name == "verify":
                ok = bool(res)
    return 0 if ok else 3


if __name__ == "__main__":
    sys.exit(main())
