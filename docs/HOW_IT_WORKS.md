# How GenTwin-SG works — a plain-language overview

This is the non-technical tour. It explains what the system does, walks through
one household, and — importantly — is honest about which numbers are **measured**
and which we had to **invent**.

No equations. If you want those, see `docs/formulation_v4_delta.md`.

---

## 1. What problem is this solving?

Two electricity substations in Dhaka — **Dhanmondi** and **Kalyanpur** — each
serve a few hundred thousand connections. On hot days they get close to their
limits.

One answer is to build more capacity. Another is to ask some customers to use a
little less at exactly the right moment, and pay them for it. That second answer
only works if you can say, for every customer and every half hour:

> *Should this household turn its air conditioner down right now — and is that
> fair to them?*

GenTwin-SG answers that question, and then **stores the reasoning** so anyone can
ask "why?" afterwards without re-running anything.

---

## 2. The chain, end to end

```
   Weather + past load          "It'll be 33°C on Tuesday, and the substation
          ↓                      will probably peak around 145 MW."
     FORECAST
          ↓
  SYNTHETIC HOUSEHOLDS          "Here are 400 made-up but realistic customers
          ↓                      whose combined demand matches the real data."
    OPTIMISATION
          ↓                     "Given prices, comfort and the grid's limits,
      DECISIONS                  here's who should do what, half hour by half hour."
          ↓
   EVENTS + PRIORITY            "These moments mattered. Here's the ranking."
          ↓
  KNOWLEDGE GRAPH               "And here's why, permanently recorded."
```

Each step feeds the next. Nothing is guessed twice.

---

## 3. One household, start to finish

Meet **DHNC0065** — a made-up upper-middle-income household of six in Dhanmondi
with air conditioning. Over a 30-day window:

| | Without demand response | With demand response |
|---|---|---|
| Electricity used | 447 kWh | 380 kWh |
| Tariff band reached | 401–600 units | **301–400 units** |
| Monthly bill | 4,512 Tk | **3,569 Tk** |

**It saved 944 Tk — about 21%.** And notice *why* it saved so much: by using
less, it dropped into a **cheaper tariff band**. Bangladesh's residential tariff
charges more per unit the more you use, so falling one band is worth far more
than the energy alone.

That's the good news. Here's the honest part:

| | |
|---|---|
| Bill saving | +944 Tk |
| Incentive received | +66 Tk |
| Discomfort (being warmer) | −1,677 Tk |
| **Net** | **−667 Tk** |

By this household's own stated preferences, **it came out behind**. It gave up
more comfort than it got back in money. We'll come back to this in §7 — it's the
most important result in the study.

---

## 4. Things we had to invent (and why)

This is the section a careful reader should actually read.

### Temperature every half hour

The dataset gives one high and one low per day — "26°C to 33°C, Monday". It does
**not** say what the temperature was at 2 p.m.

But air conditioning is the whole story here, and AC use depends on the
temperature *right now*. So we **reconstruct** an hour-by-hour curve from the
daily high and low, using a standard method from agricultural science: warm up
in a smooth curve after sunrise, peak about two hours after noon, cool off
overnight. We then nudge the curve so its daily average matches the recorded
average exactly.

**This is a reconstruction, not a measurement.** It's flagged as such on every
output. It matters: get the shape wrong and you move the afternoon peak.

### 400 households standing in for 375,000

We don't simulate every customer. We build **400 representative households per
substation** and give each one a **weight** — a number saying "this one stands
for 624 real connections like it".

Think of it like an opinion poll. You don't ask 375,000 people; you ask a
carefully chosen sample and scale up.

This keeps the maths solvable (about 44,000 decisions instead of tens of
millions — under two seconds per day) and it's exact, not approximate: the
weights carry the scale.

> ⚠️ **Read the numbers correctly.** When a report says "800 households", that
> means **800 representatives** covering **375,692 real connections**. Every
> population total is weighted. A count of representatives is not a count of
> customers.

### Households that behave differently from each other

An early version had every household of the same type behaving identically —
same appliances, same hours. That's not how people live, and it produced a
household consuming 1,110 kWh a month, roughly four times a real Dhaka home.

The fix: each household now **draws its own hours**. Your neighbour runs their AC
5 hours a day, you run yours 6, and not at the same times. Also, a home with
three ACs rarely runs all three at once.

After the fix, households consume **221 kWh/month on average** — squarely in the
real range — and the population is genuinely varied.

### The priority weights

When something goes wrong, the system ranks what matters most. Four things feed
the ranking: how structurally important something is, how much energy it moved,
whether the day was stressed, and how far over capacity we are.

How much each counts depends on how bad things are:

| Situation | Structure | Energy moved | Stress | Capacity gap |
|---|---|---|---|---|
| **Normal** | 0.55 | 0.25 | 0.10 | 0.10 |
| **Stressed** | 0.30 | 0.30 | 0.20 | 0.20 |
| **Emergency** | 0.15 | 0.25 | 0.25 | 0.35 |

Read across: on a normal day the system mostly cares about what things *are*. In
an emergency it mostly cares about **how far over capacity we are** (0.10 → 0.35).
Attention shifts from bookkeeping to the fire.

These four weights per row are **fixed by the mathematical model** — we did not
tune them. They're the model's judgement about what should matter when, and each
row sums to 1.

---

## 5. Where every number comes from

| Kind | Examples | Source |
|---|---|---|
| **Measured** | daily peak load, peak time, daily high/low temperature | BPDB dataset |
| **Official** | every electricity rate, tariff bands, VAT, peak/off-peak ratios | BERC order, 3 June 2026 |
| **Model output** | forecasts and their uncertainty | the frozen forecasting model |
| **Reconstructed** | half-hourly temperature | standard method, flagged |
| **Assumed** | appliance wattages, how long people run things, income bands, comfort values, battery size, how much capacity headroom exists | stated openly, with reasons |

The assumed group is real and unavoidable — nobody publishes what's plugged in
inside Dhanmondi homes. Every one of them lives in a single file (`config.py` or
the appliance catalogue) with a comment saying *why it's needed*, *which part of
the model uses it*, and *whether it came from literature or is our judgement*.

Nothing important is buried in code.

**A note on prices.** Two different prices appear, on purpose. The optimiser uses
the price of your *next* unit — that's what you actually weigh when deciding
whether to run the AC. Your bill uses the official banded calculation, because
that's what you pay. Decide at the margin, bill on the average. Both are right;
neither substitutes for the other.

---

## 6. When does the system say "something happened"?

An **event** is recorded whenever one of four things is true on a given day:

1. **The day is stressed** — tomorrow's likely peak is above the threshold.
2. **Customers responded** — somebody shifted or reduced load.
3. **Reserves ran short** — the battery dipped below its floor.
4. **Some demand went unserved** — the worst case.

Every event carries its own explanation: the forecast that triggered it, how bad
things got, who was affected, how much energy moved, and what it cost.

A real example from the run:

> **30 June 2026, Kalyanpur.** Expected peak 139,439 kW against a stress
> threshold of 129,000 kW — over the line. Recorded as **stressed**. 215
> representative customers responded, reducing 188,599 kWh. No reserve shortfall,
> nothing unserved.

Three severity levels exist — **normal**, **stressed**, **emergency**. Emergency
means capacity was exceeded, or power had to be pulled from the neighbouring
substation, or reserves ran short. Under normal settings the system never reaches
emergency, because there's enough headroom — which is a correct result, not a
missing feature. There's a deliberate "what if capacity were tighter" scenario to
exercise that path.

---

## 7. What the results actually say

**Customers save money.** About 10% off the monthly bill — 365 Tk/month for a
typical Dhanmondi home, 185 Tk in Kalyanpur. Hospitals were never asked to
reduce anything.

**But most customers end up behind once comfort is counted.** Only about **1.6%**
of connections come out genuinely ahead.

This isn't a bug — it's the model being honest. Two reasons:

1. **The system asks for more than it pays for.** Some of the value of reducing
   load goes to *the grid* — avoided strain, avoided imports. The household bears
   the discomfort but doesn't see that value on its bill.
2. **The current incentive is too small.** The reward is 3 Tk per unit reduced, on
   stressed days only. That doesn't cover the discomfort.

So the honest headline is:

> Demand response works technically and cuts bills by around 10% — but at the
> current incentive level, most households would be rational to decline. The
> incentive is the lever, and it needs to be bigger.

That's a finding, not a failure. It's also a concrete, testable recommendation.

---

## 8. What we're not claiming

Being clear about limits is part of the work:

- **Dhanmondi's peak timing is unresolved.** The recorded "time of daily peak"
  looks more like a clerical habit than a measurement — 64% of days land on
  exactly 13:00, and it's *more* consistent in winter when there's no cooling
  load. Kalyanpur's data looks genuine. So Kalyanpur passes validation; Dhanmondi
  is marked **undetermined**, not passed.
- **The customer mix is estimated**, not surveyed. A real map query would improve
  it; the settings for that are locked in advance so the answer can't be nudged.
- **The 30-day window is representative**, not a real billing cycle — actual
  meter-reading dates aren't published.
- **The utility margin is indicative.** No network or overhead costs are modelled.

---

## 9. Why the knowledge graph exists

Everything ends up in a searchable graph — customers, appliances, forecasts,
decisions, events, bills — with the *reasoning* attached, not just the outcome.

That means someone can ask, in plain terms:

- *Why was my bill high this month?*
- *Why was my AC turned down at 6:30 p.m.?*
- *Which hospitals were ever asked to reduce load?* (Answer: none.)
- *What was the battery doing at 5 p.m.?*
- *Who gained and who lost, by income group?*

and get an answer grounded in stored numbers — no re-computation, no guesswork.

That's the point of calling it a **digital twin**: not just a model that produces
an answer, but one that can still explain itself afterwards.
