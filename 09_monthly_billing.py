#!/usr/bin/env python3
"""
Monthly billing and savings.

    python 09_monthly_billing.py --report
    python 09_monthly_billing.py --report --plot
    python 09_monthly_billing.py --report --tag _period_scarcity

Answers: how much can a consumer save on a monthly bill, and at what comfort
cost? Reads a multi-day LP solve (run 06_lp_optimiser.py --solve --month 30
first) and accumulates it over the billing period T_m.

Two bills are reported for every consumer and they are not the same thing:

  J^h_1        the formulation's bill, Eq. (20). Linear, at the MARGINAL slab
               rate - the price a household faces on its next kWh, which is
               what drives the Eq. (37) threshold.
  gazette      what the consumer actually pays: cumulative BERC slabs on the
               month's kWh, plus demand charge, plus 5% VAT.

The gap between them is the gap between the marginal price that drives
behaviour and the average price that appears on the bill.
"""

import argparse
import sys

import numpy as np
import pandas as pd

from gentwin import (billing, config as cfg, forecast as F, load_model as lm,
                     peak_window as pw, population as pop, temperature as TM)


def _bpdb():
    for p in (cfg.BPDB_CSV, cfg.DATA_DIR / "BPDB_Dhaka_City_Substations_Page3.csv"):
        if p.exists():
            return pd.read_csv(p, parse_dates=["Date"])
    raise FileNotFoundError(f"BPDB CSV not found at {cfg.BPDB_CSV}")


def load_run(tag: str, subs):
    net = pd.read_csv(cfg.OUT_DIR / f"lp_network{tag}.csv")
    dec = pd.read_csv(cfg.OUT_DIR / f"lp_decisions{tag}.csv")
    summ = pd.read_csv(cfg.OUT_DIR / f"lp_summary{tag}.csv")
    dates = sorted(net["date"].unique())
    if len(dates) < 7:
        print(f"  WARNING: only {len(dates)} day(s) in this run. A monthly bill "
              f"needs the whole period - the tariff slab is cumulative over the "
              f"month, so a short run cannot be scaled up honestly.\n"
              f"  Run: python 06_lp_optimiser.py --solve --month 30")
    return net, dec, summ, dates


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--tag", default="_period")
    ap.add_argument("--substations", nargs="*", default=list(cfg.SUBSTATIONS))
    a = ap.parse_args()
    if not (a.report or a.plot):
        ap.print_help(); return 1

    subs = a.substations
    bpdb = _bpdb()
    cat = pop.load_catalogue()
    sch = pop.all_schedules(cat)
    POP = {i: pop.consumers_from_cache(i) for i in subs}
    net, dec, summ, dates = load_run(a.tag, subs)
    days = len(dates)

    print("=" * 78)
    print(f"REPRESENTATIVE {days}-DAY EVALUATION WINDOW - "
          f"{dates[0]} to {dates[-1]}")
    print(f"  NOT a BPDB billing cycle: actual meter-reading dates are not "
          f"published, so this is a\n  {days}-day window chosen to end at the "
          f"forecast horizon. Slab boundaries are applied to\n  the window's "
          f"total, scaled to a nominal 30 days.")
    print("=" * 78)

    modes = (net.groupby("day_mode")["date"].nunique().to_dict()
             if "day_mode" in net.columns else {})
    if modes:
        print(f"  day composition: " + ", ".join(f"{k} {v}" for k, v in modes.items())
              + "   (observed days carry no forecast-error margin; forecast days "
                "use p~ = p_hat + z_beta*sigma)")

    # --- rebuild the unweighted per-consumer profiles, day by day -----------
    peak_slots = {i: list(pw.get_peak_slots(i)) for i in subs}
    daily, s_stress = {}, {}
    for d in dates:
        theta = np.array(TM.profile_for_date(bpdb, d)["theta_c"])
        daily[d] = {}
        for i in subs:
            for h in POP[i]:
                daily[d][h.consumer_id] = lm.consumer_profile(h, sch, cat, theta)
        row = net[net["date"] == d]
        s_stress[d] = {}
        for i in subs:
            plan = F.build_period(bpdb, i, d, d)[0]
            s_stress[d][i] = plan["s_stress"]

    decisions = {}
    for d, g in dec.groupby("date"):
        decisions[d] = {(r.consumer_id, int(r.slot), r.var): float(r.value)
                        for r in g.itertuples()}

    surcharge = {i: float(summ.get(f"varpi_{i}", pd.Series([0.0])).mean())
                 for i in subs}

    allc = [h for i in subs for h in POP[i]]
    df = billing.build(allc, daily, decisions, peak_slots, s_stress,
                       surcharge, days)
    summary = billing.summarise(df, summ, subs, days)
    bycat = billing.by_category(df)
    comp = billing.comparison(df)
    dist = billing.distribution(df)
    checks = billing.validate(df, days)
    plaus = billing.plausibility(df, days)

    # ---- requirement 7: outputs/billing/ -------------------------------
    bdir = cfg.OUT_DIR / "billing"
    bdir.mkdir(parents=True, exist_ok=True)
    sfx = a.tag if a.tag != "_period" else ""
    df.rename(columns={"consumer_id": "household_id"}).to_csv(
        bdir / f"household_monthly_bills{sfx}.csv", index=False)
    comp.to_csv(bdir / f"household_monthly_comparison{sfx}.csv", index=False)
    summary.to_csv(bdir / f"substation_monthly_summary{sfx}.csv", index=False)
    bycat.to_csv(bdir / f"monthly_by_category{sfx}.csv", index=False)
    pd.DataFrame([{"scope": k, **v} for k, v in dist.items()]).to_csv(
        bdir / f"monthly_bill_distribution{sfx}.csv", index=False)
    import json as _json
    (bdir / f"billing_statistics{sfx}.json").write_text(_json.dumps({
        "period": {"start": dates[0], "end": dates[-1], "days": days,
                   "day_composition": modes,
                   "label": f"representative {days}-day evaluation window",
                   "is_bpdb_billing_cycle": False,
                   "note": "Actual BPDB meter-reading dates are not published. "
                           "This window ends at the forecast horizon and mixes "
                           "observed and forecast days."},
        "price_basis": {
            "optimisation": "J^h_1, Eq. (20), at the MARGINAL slab rate - the "
                            "price a household faces on its next kWh, which is "
                            "what the Eq. (37) threshold compares against",
            "reported_bill": "BERC June-2026 cumulative slabs + demand charge "
                             "+ 5% VAT + surcharge varpi_{i,m}",
            "why_they_differ": "A household decides at the margin but is "
                               "billed on the average. Using the average in the "
                               "LP would understate the incentive to respond in "
                               "a high slab; using the marginal rate as the "
                               "bill would misstate what is actually paid.",
        },
        "net_benefit_definition": "bill_saving + incentive_received - discomfort_J2",
        "population_basis": {
            "representatives_per_substation": cfg.N_REPRESENTATIVE_CONSUMERS,
            "note": "Each row is a REPRESENTATIVE consumer standing for w_h "
                    "real connections, not one metered customer. Population "
                    "totals are weighted by w_h.",
        },
        "distribution": dist,
        "substation_summary": summary.to_dict("records"),
        "by_category": bycat.to_dict("records"),
        "validation": checks,
        "plausibility": plaus,
        "tariff_source": __import__("gentwin.tariff", fromlist=["x"]).GAZETTE_REF,
        "rho_tk_per_kwh": cfg.RHO_REBATE_TK_PER_KWH,
        "qoe_min": cfg.QOE_MIN,
    }, indent=1, default=str), encoding="utf-8")
    # legacy paths kept so nothing downstream breaks
    df.to_csv(cfg.OUT_DIR / f"monthly_bills{a.tag}.csv", index=False)
    summary.to_csv(cfg.OUT_DIR / f"monthly_summary{a.tag}.csv", index=False)
    bycat.to_csv(cfg.OUT_DIR / f"monthly_by_category{a.tag}.csv", index=False)

    # ---- report ------------------------------------------------------------
    for _, r in summary.iterrows():
        print(f"\n  {r['substation']}")
        print(f"     connections            {r['n_connections']:>14,.0f}   "
              f"({r['n_representatives']} representatives)")
        print(f"     participating          {r['n_participating_connections']:>14,.0f}"
              f"   ({r['participation_rate_pct']:.1f}%)")
        print(f"     energy  baseline       {r['baseline_energy_kwh']:>14,.0f} kWh")
        print(f"             optimised      {r['optimised_energy_kwh']:>14,.0f} kWh"
              f"   ({r['energy_saved_pct']:+.2f}%)")
        print(f"             shifted        {r['energy_shifted_kwh']:>14,.0f} kWh")
        print(f"             curtailed      {r['energy_curtailed_kwh']:>14,.0f} kWh")
        print(f"     bill    baseline       {r['bill_baseline_tk']:>14,.0f} Tk")
        print(f"             optimised      {r['bill_optimised_tk']:>14,.0f} Tk")
        print(f"     CONSUMER SAVING        {r['consumer_saving_tk']:>14,.0f} Tk"
              f"   ({r['consumer_saving_pct']:.2f}%)")
        print(f"     per connection         {r['mean_saving_per_connection_tk']:>14,.2f} Tk/month")
        print(f"     incentive paid         {r['total_incentive_paid_tk']:>14,.0f} Tk"
              f"   (rho = {cfg.RHO_REBATE_TK_PER_KWH} Tk/kWh on stressed days)")
        print(f"     discomfort cost J2     {r['discomfort_tk']:>14,.0f} Tk")
        print(f"     net benefit            {r['net_benefit_tk']:>14,.0f} Tk")
        print(f"     median hh saving       {r['median_household_saving_tk']:>14,.2f} Tk")
        print(f"     mean hh bill / kWh     {r['mean_household_bill_tk']:>14,.0f} Tk"
              f" / {r['mean_household_kwh']:,.0f} kWh")
        print(f"     mean peak reduction    {r['peak_reduction_pct_mean']:>14,.2f} %")
        print(f"     QoE mean / min         {r['qoe_mean']:>14.3f} / {r['qoe_min']:.3f}"
              f"   violations {int(r['qoe_violations'])}")
        print(f"     operator cost          {r['operator_cost_tk']:>14,.0f} Tk")
        print(f"     utility margin         {r['utility_margin_tk_indicative']:>14,.0f} Tk"
              f"   (INDICATIVE: no network or overhead cost modelled)")

    print(f"\n  Saving by consumer category (population-weighted)")
    print(f"  {'substation':<12} {'category':<13} {'connections':>12} "
          f"{'part.%':>7} {'saving Tk':>13} {'sav%':>7} {'Tk/conn':>9} {'QoE':>6}")
    for _, r in bycat.iterrows():
        print(f"  {r['substation']:<12} {r['category']:<13} {r['connections']:>12,.0f} "
              f"{r['participation_pct']:>7.1f} {r['saving_tk']:>13,.0f} "
              f"{r['saving_pct']:>7.2f} {r['mean_saving_per_connection_tk']:>9.2f} "
              f"{r['qoe_mean']:>6.3f}")

    # ---- requirement 9: validation ---------------------------------------
    print(f"\n  VALIDATION")
    for c in checks:
        print(f"    [{'PASS' if c['pass'] else 'WARN'}] {c['check']}"
              + (f"\n           {c['detail']}" if c["detail"] else ""))
    nfail = sum(1 for c in checks if not c["pass"])
    print(f"    {len(checks) - nfail}/{len(checks)} passed"
          + ("" if not nfail else f"   -- {nfail} WARNING(S), see above"))

    # ---- requirement 5: distribution -------------------------------------
    print(f"\n  DISTRIBUTION OF OUTCOMES (population-weighted by w_h)")
    print(f"  {'scope':<12} {'connections':>12} {'gained':>12} {'lost':>12} "
          f"{'%gained':>8} {'med Tk':>9} {'p25':>9} {'p75':>9} {'max':>10}")
    for k, v in dist.items():
        print(f"  {k:<12} {v['n_connections']:>12,.0f} "
              f"{v['connections_gained_money']:>12,.0f} "
              f"{v['connections_lost_money']:>12,.0f} "
              f"{v['pct_connections_gained']:>8.1f} "
              f"{v['saving_median_tk']:>9,.0f} {v['saving_p25_tk']:>9,.0f} "
              f"{v['saving_p75_tk']:>9,.0f} {v['saving_max_tk']:>10,.0f}")

    # ---- plausibility gate on the bill figures --------------------------
    pl = plaus
    bad = [r for r in pl if not r["within_range"]]
    print(f"\n  PLAUSIBILITY OF PER-CONNECTION CONSUMPTION")
    print(f"  {'substation':<12} {'category':<13} {'kWh/conn/mo':>12} "
          f"{'plausible':>16} {'factor':>8}  {'bill Tk/mo':>11}")
    for r in pl:
        flag = "" if r["within_range"] else "  <-- OUT OF RANGE"
        print(f"  {r['substation']:<12} {r['category']:<13} "
              f"{r['kwh_per_connection_month']:>12,.0f} "
              f"{str(r['plausible_low']) + '-' + str(r['plausible_high']):>16} "
              f"{r['factor_off']:>8.2f}  {r['mean_bill_tk']:>11,.0f}{flag}")
    if bad:
        print(f"\n  !! {len(bad)} of {len(pl)} category/substation cells consume "
              f"outside the published Dhaka range.")
        print(f"     The AGGREGATE energy and peak results are unaffected - they "
              f"are pinned to the\n     BPDB peak by construction. The BILL "
              f"figures ARE affected, because a consumer\n     consuming 3x too "
              f"much lands in a higher LT-A slab (17.35 Tk/kWh above 600 units\n"
              f"     against 9.10 at 201-300), so both the bill and the saving "
              f"are priced wrongly.")
        print(f"     Root cause: the synthetic population's DIVERSITY is too low. "
              f"Real feeders show\n     DIVERSITY factors of 2.5-5 "
              f"(coincidence factor 0.2-0.4) for residential.\n     "
              f"substation peak is reached with too few, too large consumers. "
              f"Fix is in Stage A/B\n     (per-slot stochastic appliance usage), "
              f"not here.")
    pd.DataFrame(pl).to_csv(cfg.OUT_DIR / f"billing_plausibility{a.tag}.csv",
                            index=False)

    # ---- net benefit: is participation individually rational? ------------
    neg = summary["net_benefit_tk"] < 0
    if neg.any():
        print(f"\n  NET BENEFIT IS NEGATIVE at "
              f"{', '.join(summary.loc[neg, 'substation'])}: the Tk-equivalent "
              f"discomfort J^h_2\n  exceeds the bill saving. This is NOT an "
              f"error - it follows from Eq. (24). The LP\n  minimises "
              f"lambda_h*J1 + (1-lambda_h)*J2, so with lambda_h = 0.6 an action "
              f"is taken when\n  0.6*saving > 0.4*discomfort, i.e. whenever "
              f"saving > 0.67*discomfort. On top of that\n  the system term "
              f"kappa*pi/theta_3 (~9 Tk/kWh here) buys GRID value the consumer "
              f"does not\n  see on their bill - the externality being "
              f"internalised, exactly as section 4 intends.\n  The consequence "
              f"is a real policy finding: at rho = 3 Tk/kWh the rebate does not "
              f"make\n  participation individually rational. Sweep rho to find "
              f"the level that does.")

    d = df[df["participated"]]
    print(f"\n  Per-REPRESENTATIVE distribution among the {len(d)} that "
          f"participated.\n  Each represents a cluster of w_h similar real "
          f"connections; it is not one metered customer.")
    for col, lab, unit in (("saving_tk", "saving", "Tk"),
                           ("saving_pct", "saving", "%"),
                           ("qoe", "QoE", ""),
                           ("net_benefit_tk", "net benefit", "Tk")):
        v = d[col].dropna()
        if len(v):
            print(f"     {lab + ' (' + unit + ')':<18} min {v.min():>10,.2f}  "
                  f"p25 {v.quantile(.25):>10,.2f}  median {v.median():>10,.2f}  "
                  f"p75 {v.quantile(.75):>10,.2f}  max {v.max():>10,.2f}")

    ex = d.nlargest(3, "saving_tk")[
        ["consumer_id", "category", "archetype", "baseline_kwh_30d",
         "optimised_kwh_30d", "bill_baseline_tk", "bill_optimised_tk",
         "saving_tk", "saving_pct", "qoe", "slab_baseline", "slab_optimised"]]
    print(f"\n  Largest individual savings")
    print(ex.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))

    print(f"\n  WHY THE OPTIMISER AND THE BILL USE DIFFERENT PRICES")
    print(f"     The LP maximises against J^h_1 at the MARGINAL slab rate, "
          f"because that is the\n     price a household faces on its next kWh "
          f"and is what Eq. (37) compares against.\n     The reported bill uses "
          f"the BERC cumulative slabs, because that is what is paid.\n     "
          f"Decide at the margin, bill on the average - both are correct, and "
          f"neither\n     substitutes for the other.")
    print(f"\n  J^h_1 (formulation, marginal rate) vs gazette bill (slabs+VAT):")
    print(f"     mean J1 saving   {df['J1_saving_tk'].mean():>10,.2f} Tk")
    print(f"     mean bill saving {df['saving_tk'].mean():>10,.2f} Tk")
    print(f"     The two differ because J^h_1 prices every kWh at the marginal "
          f"slab rate\n     while the gazette bill charges each slab in turn. "
          f"J^h_1 drives the LP;\n     the gazette bill is what the consumer pays.")

    if a.plot:
        _plot(df, summary, bycat, net, subs, dates, a.tag)
    print(f"\n  wrote outputs/billing/: household_monthly_bills{sfx}.csv "
          f"({len(df)} households), household_monthly_comparison{sfx}.csv, "
          f"substation_monthly_summary{sfx}.csv,\n"
          f"        monthly_bill_distribution{sfx}.csv, "
          f"billing_statistics{sfx}.json, monthly_by_category{sfx}.csv")
    return 0


def _draw_daily_peak_panel(ax, net, subs, dates):
    for i in subs:
        g = net[net["substation"] == i]
        pk = g.groupby("date")[["baseline_kw", "post_response_kw"]].max()
        ax.plot(pk.index, pk["baseline_kw"] / 1000, "o-", lw=1.5, ms=3,
                label=f"{i} baseline")
        ax.plot(pk.index, pk["post_response_kw"] / 1000, "s--", lw=1.5, ms=3,
                label=f"{i} optimised")
    if "day_mode" in net.columns:
        obs = sorted(net.loc[net.day_mode == "observed", "date"].unique())
        if obs:
            ax.axvspan(obs[0], obs[-1], color="steelblue", alpha=0.07)
            ax.text(obs[len(obs) // 2], ax.get_ylim()[0], " observed days",
                    fontsize=8, color="steelblue", va="bottom")
    ax.set_ylabel("daily peak (MW)")
    ax.set_title(f"Daily peak over the billing period, baseline vs optimised "
                 f"({dates[0]} to {dates[-1]})", fontsize=12)
    ax.tick_params(axis="x", rotation=60, labelsize=7)
    ax.legend(fontsize=8, ncol=2)


def _draw_savings_distribution_panel(ax, d, subs):
    for i in subs:
        v = d.loc[d.substation == i, "saving_pct"].dropna()
        if len(v):
            ax.hist(v, bins=30, alpha=0.6, label=f"{i} (n={len(v)})")
    ax.set_xlabel("monthly bill saving (%)")
    ax.set_ylabel("representatives")
    ax.set_title("Distribution of consumer bill savings", fontsize=11)
    ax.legend(fontsize=8)


def _draw_savings_vs_discomfort_panel(ax, d):
    for c, g in d.groupby("category"):
        ax.scatter(g["discomfort_J2_tk"], g["saving_tk"], s=14, alpha=0.6, label=c)
    lim = max(d["discomfort_J2_tk"].max(), d["saving_tk"].max())
    ax.plot([0, lim], [0, lim], "k--", lw=1, label="saving = discomfort")
    ax.set_xlabel("discomfort $J^h_2$ (Tk)")
    ax.set_ylabel("bill saving (Tk)")
    ax.set_title("Is the saving worth the comfort cost?", fontsize=11)
    ax.legend(fontsize=7.5)


def _draw_savings_by_category_panel(ax, bycat):
    piv = bycat.pivot(index="category", columns="substation",
                      values="mean_saving_per_connection_tk").fillna(0)
    piv.plot(kind="barh", ax=ax, width=0.75)
    ax.set_xlabel("mean saving per connection (Tk/month)")
    ax.set_ylabel("")
    ax.set_title("Who saves, by consumer category", fontsize=11)
    ax.legend(fontsize=8)


def _draw_service_quality_panel(ax, d):
    order = sorted(d["category"].unique())
    ax.boxplot([d.loc[d.category == c, "qoe"].dropna().values for c in order],
               tick_labels=order, patch_artist=True,
               boxprops=dict(facecolor="#cfe2f3"))
    ax.axhline(cfg.QOE_MIN, color="red", ls="--", lw=1.2,
               label=f"$QoE^{{min}}$ = {cfg.QOE_MIN}")
    ax.set_ylabel("realised QoE")
    ax.set_title("Service quality by category (C7, Eq. 35)", fontsize=11)
    ax.tick_params(axis="x", rotation=25, labelsize=8)
    ax.legend(fontsize=8)


def _plot(df, summary, bycat, net, subs, dates, tag):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = df[df["participated"]]

    fig = plt.figure(figsize=(17, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.38, wspace=0.22)
    _draw_daily_peak_panel(fig.add_subplot(gs[0, :]), net, subs, dates)
    _draw_savings_distribution_panel(fig.add_subplot(gs[1, 0]), d, subs)
    _draw_savings_vs_discomfort_panel(fig.add_subplot(gs[1, 1]), d)
    _draw_savings_by_category_panel(fig.add_subplot(gs[2, 0]), bycat)
    _draw_service_quality_panel(fig.add_subplot(gs[2, 1]), d)

    fig.suptitle("GenTwin-SG monthly billing: savings, comfort cost and peak "
                 "reduction", fontsize=14, fontweight="bold")
    p = cfg.OUT_DIR / f"monthly_billing{tag}.png"
    fig.savefig(p, dpi=125, bbox_inches="tight")
    print(f"  wrote {p}")

    # True vector PDFs, one per panel, drawn fresh from the same dataframes.
    panels = [
        ("daily_peak", (17, 5), lambda ax: _draw_daily_peak_panel(ax, net, subs, dates)),
        ("savings_distribution", (8, 5), lambda ax: _draw_savings_distribution_panel(ax, d, subs)),
        ("savings_vs_discomfort", (8, 5), lambda ax: _draw_savings_vs_discomfort_panel(ax, d)),
        ("savings_by_category", (8, 5), lambda ax: _draw_savings_by_category_panel(ax, bycat)),
        ("service_quality", (8, 5), lambda ax: _draw_service_quality_panel(ax, d)),
    ]
    for suffix, figsize, draw in panels:
        fig_i, ax_i = plt.subplots(figsize=figsize)
        draw(ax_i)
        fig_i.tight_layout()
        p_i = cfg.PDF_DIR / f"monthly_billing_period_{suffix}.pdf"
        fig_i.savefig(p_i, bbox_inches="tight")
        plt.close(fig_i)
        print(f"  wrote {p_i}")


if __name__ == "__main__":
    sys.exit(main())
