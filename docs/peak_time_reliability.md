# Peak-time reliability audit

Is BPDB's `*_Peak_Time` a measurement or a clerical convention? Stage B's T2
test and the adaptive ToU window `T^pk_i` both depend on the answer.

Reproduce with `python 00b_peak_window.py --audit`.

---

## 1. Why the question was asked

Dhanmondi records its daily peak at exactly 13:00 on 64% of days. That is a
very concentrated distribution for a quantity that should move with weather,
day of week and season. The concern was concrete: Stage B's T2 test compares
the synthetic argmax against this field, and Dhanmondi failed by 0.96 MW. If
the target is an artefact, tuning the population to reproduce it would be
fitting to a clerical habit.

## 2. Five signals

Each is something a genuine peak-timing distribution should show and a
convention should not. Thresholds are in `gentwin/peak_time_audit.py`.

| | Signal | Rationale |
|---|---|---|
| S1 | share of readings on `:30` | A peak read off a half-hourly series should land on `:30` about half the time. Near zero means the field is recorded to the hour. |
| S2 | entropy / modal share | Physical timing spreads; a convention concentrates. |
| S3 | seasonal mode shift | Cooling-driven peaks move between monsoon and winter. |
| S3b | inverted seasonality | Modal share *higher* in the lightest-load season is the opposite of thermal behaviour. |
| S4 | weekend drop | A share that collapses when offices close is tied to staff presence, not load. |
| S5 | multi-year trend | Rising concentration while load falls is consolidating practice, not physics. |

Verdict by flag count: 0–1 RELIABLE, 2–3 QUESTIONABLE, 4–5+ LIKELY_CONVENTION.
The audit reads only the peak-time field and the calendar — no model, residual
or fitted quantity — so it cannot be tuned toward a convenient answer.

## 3. Fleet result

24 substations audited. `half_hour_share` separates them almost perfectly, and
the split is not subtle:

| group | n | modal slot | `:30` share | entropy | verdict |
|---|---|---|---|---|---|
| Motijheel, Narinda, Gulshan, Postogola, Agargaon, Banani, Cantonment, Lalbag, Aftabnagar, **Dhanmondi** | 10 | **all 13:00** | 0.027 – 0.073 | 2.04 – 3.31 | LIKELY_CONVENTION |
| Uttara New, Uttara, Bashundhara, Purbachal, Kamrangirchar, Ullon, Keranigonj | 7 | mostly 13:00 | 0.087 – 0.231 | 3.15 – 3.67 | QUESTIONABLE |
| Madartek, Savar, **Kalyanpur**, Mirpur, Kodda, Hasnabad, Metrorail ×2 | 8 | 12:00–23:00, varied | 0.098 – 0.177 | 3.19 – 3.98 | RELIABLE |

**Every substation in the LIKELY_CONVENTION group has a 13:00 mode.** Ten
independent substations converging on the same clock hour, all with almost no
`:30` readings, is a shared reporting practice — not ten coincident midday
peaks.

## 4. Dhanmondi: 6 of 6 flags, the worst in the fleet

```
modal 13:00 @ 64.4%, entropy 2.04 bits, 2.7% on :30, 19/47 distinct slots

S1  only 2.7% of readings on :30 (expect ~50%)
S2  entropy 2.04 bits, modal share 64.4%
S3  modal slot 13:00 in every season
S3b modal share higher in the lightest-load season than the heaviest
S4  modal share falls 18.1% at weekends (64.4% -> 46.3%)
S5  modal share rose +25.3% from 2020 to 2026
```

The seasonal table is the most telling single piece of evidence:

| season | n | mode | modal share | median load |
|---|---|---|---|---|
| Monsoon | 682 | 13:00 | 57.6% | **140.0 MW** |
| Pre-monsoon | 617 | 13:00 | 59.0% | 129.0 MW |
| Post-monsoon | 387 | 13:00 | 68.7% | 125.0 MW |
| Winter | 599 | 13:00 | **75.0%** | **89.0 MW** |

Dhanmondi's midday peak is *most* consistent in winter, when load is 36% lower
and there is essentially no cooling demand. A thermally-driven midday peak
would behave in exactly the opposite way. Combined with the weekend drop and
the rising trend, the reading is that a value is being logged at 13:00 on
working days rather than a peak being observed.

**Verdict: `t2_target_usable = False` for Dhanmondi.**

## 5. Kalyanpur: RELIABLE

```
modal 18:00 @ 19.2%, entropy 3.50 bits, 17.7% on :30, 22/47 distinct slots
```

One flag only (S3b), and its seasonal behaviour is what a physical
distribution looks like:

| season | mode | modal share | median load |
|---|---|---|---|
| Monsoon | 19:00 | 3.7% | 102.0 MW |
| Pre-monsoon | 19:00 | 3.7% | 96.0 MW |
| Post-monsoon | 18:00 | 48.9% | 87.0 MW |
| Winter | **13:00** | 28.7% | 66.0 MW |

The mode *moves* — evening in the cooling seasons, midday in winter when
commercial load dominates and there is no evening AC. That is the seasonal
signature S3 looks for, and Dhanmondi does not have it.

## 6. Consequences

**Stage B T2.** Dhanmondi's T2 result is downgraded from FAIL to
**CONDITIONAL**. The model's 18:00 peak may well be correct; the target is not
trustworthy enough to call it a failure. `validation.py` reads the audit verdict
and reports T2 as conditional wherever `t2_target_usable` is False, so this is
mechanical rather than a judgement applied by hand. Kalyanpur's T2 remains a
genuine PASS.

**Adaptive local ToU.** `T^pk_Dhanmondi = 12:00–14:30` is derived from the same
field. The reported 285× precision gain over the national window therefore
carries a caveat — see `docs/adaptive_local_tou.md` §6, limitation L6. The
Kalyanpur window is unaffected. The *mechanism* stands regardless: deriving the
window from local congestion history rather than a national rule is the
contribution, and it survives whatever this field turns out to be.

**Do not tune to it.** The Dhanmondi composition has not been adjusted, and must
not be.

## 7. How to settle it

Cheapest first:

1. **Ask BPDB how `Peak_Time` is recorded** at Dhanmondi and at Kalyanpur —
   automatic scan of a half-hourly series, or a manual reading. One question,
   and it is decisive. The fleet split gives a specific thing to ask about.
2. **Request a half-hourly or 15-minute SCADA series** for a few weeks at
   Dhanmondi. That resolves T2 directly, and would also remove the largest
   limitation in Stage A/B — the absence of any intra-day load observation.
3. Failing both, treat the RELIABLE group as the validation set and report
   Dhanmondi's timing as unverifiable.

Option 2 would improve more than T2: it would let κ be applied at slot level
(Assumption 1 becomes exact rather than exact-only-at-the-peak), and it would
replace the schedule assumptions in `02_consumer_catalogue.json` with measured
shape.
