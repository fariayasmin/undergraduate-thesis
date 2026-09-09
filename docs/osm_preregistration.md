# Pre-registration: OSM service-area estimator

**Status: LOCKED. Written before any Overpass fetch has been run.**

---

## Why pre-register

We already know the answer we would like. Stage B established that Dhanmondi's
synthetic peak moves into the 12:00–14:30 window if and only if the residential
share of connected demand is **≤ 0.50**, and the current fallback estimate is
**0.52**. The flip point is 0.51.

The OSM estimator has at least six settings, and every one of them moves the
residential share:

| setting | current value | direction of effect on residential share |
|---|---|---|
| service radius | 2.0 km | larger radius reaches outward into residential fringe → **raises** |
| `building:levels` default, residential | 5 | higher → **raises** |
| `building:levels` default, commercial | 6 | higher → **lowers** |
| intensity, residential | 15 W/m² | higher → **raises** |
| intensity, commercial | 45 W/m² | higher → **lowers** |
| tag mapping | `building=yes` → Residential | reassigning `yes` → **lowers** substantially |

Any of these could be nudged post-hoc to land under 0.51, and the resulting
"PASS" would be worthless. Freezing them in writing first is what keeps the OSM
result an independent measurement rather than a second calibration knob.

The `docs/peak_time_reliability.md` audit has since found that Dhanmondi's
13:00 target is itself probably a reporting convention, so the T2 comparison
may not be decidable at all. That **strengthens** the case for pre-registration
rather than weakening it: with an unreliable target, an unconstrained estimator
could be steered toward either answer.

## Locked settings

These are the values in `gentwin/service_area.py` and `01_service_area_osm.py`
as committed. **Running `--fetch` with anything else invalidates the result.**

### Geometry

```
SERVICE_RADIUS_KM      = 2.0
MAX_SERVICE_RADIUS_KM  = 3.0
method                 = voronoi if >= 3 fleet coordinates, else radius disc
```

Voronoi is preferred and requires `nodes_Substation.csv` with three or more
sites. Whichever runs is recorded in the cache as `service_area.method`.

### Tag → category mapping

First matching rule wins, so `amenity=hospital` beats `building=yes`. Full
ordered list in `service_area.TAG_RULES`; the consequential entries:

```
Hospital     amenity {hospital, clinic, doctors, dentist, nursing_home}
             healthcare {hospital, clinic, centre, laboratory, doctor}
             building {hospital}
Educational  amenity/building {school, college, university, kindergarten, library}
Government   office {government, administrative}
             amenity {townhall, courthouse, police, fire_station, post_office,
                      public_building}
             building {government, civic, public}
Industrial   building {industrial, warehouse, factory, manufacture}
             landuse {industrial}; man_made {works}
Commercial   building {commercial, retail, office, supermarket, kiosk, hotel,
                       shop, mall}
             shop=* (any); office=* (any other)
             amenity {restaurant, cafe, fast_food, bank, marketplace, cinema,
                      pharmacy}
Residential  building {apartments, residential, house, detached, dormitory,
                       terrace, bungalow, yes}
             landuse {residential}
```

**`building=yes` → Residential is the single most consequential rule.** In Dhaka
`yes` is the most common building value and covers a large share of untagged
mixed-use structures. Assigning it to Residential is the *conservative* choice
for our hypothesis: it inflates the residential share and therefore makes T2
*harder* to pass. It is locked in that direction deliberately, so that a pass
cannot be an artefact of a favourable mapping.

A sensitivity arm reassigning `building=yes` proportionally to the tagged mix
is declared in §"Pre-declared sensitivity" and must be reported alongside the
primary result, not instead of it.

### Levels defaults (used where `building:levels` is untagged)

```
Residential 5    Commercial 6    Industrial 2
Hospital    8    Educational 4   Government 6
```

The **tagged fraction is reported per category** so the reader can see how much
of the inventory is measured and how much inferred.

### Energy intensity (W/m² of floor area)

```
Residential 15    Commercial 45    Industrial 40
Hospital    60    Educational 25   Government 40
```

Only the *ratios* between these matter, because the absolute level is pinned to
the BPDB peak by ξ(d) in Eq. (8). An error common to all categories cancels.

### Floor area

`footprint_area × levels`, footprint by shoelace on a local equirectangular
projection. `node`-only features are excluded — a node has no footprint and
cannot be floor-area weighted.

## Pre-declared outcomes

Written before the fetch. All four are reportable results; none is a failure.

- **O1** Residential share **≤ 0.50** → Dhanmondi T2 would pass on the recorded
  target. Given the reliability audit, this is reported as *consistent with*
  the recorded 13:00 peak, **not** as confirmation of it.
- **O2** Residential share **0.51–0.55** → straddles the flip point. Report as
  indeterminate. Do not adjust anything.
- **O3** Residential share **> 0.55** → the synthetic evening peak stands. Given
  the audit verdict, the most likely reading is that the model is right and the
  13:00 field is clerical.
- **O4** Fetch fails or coverage is too sparse → keep the fallback, keep its
  `fallback_literature` provenance stamp, and report the composition as
  unmeasured.

**In no case is the composition adjusted by hand, and in no case is a setting
above changed after seeing the result.**

## Pre-declared sensitivity

Run and report all of these, whatever the primary returns:

1. Radius 1.6 / 2.0 / 2.4 km (±20%)
2. `building=yes` → Residential (primary) vs redistributed pro-rata to the
   tagged mix (arm)
3. Residential intensity 12 / 15 / 18 W/m²
4. Residential default levels 4 / 5 / 6

The headline is the **range** across these, not the point estimate. If the
range straddles 0.51 — which on the fallback's own numbers it plausibly will —
then the honest conclusion is that OSM cannot settle T2 at Dhanmondi either,
and the question moves to the SCADA request in
`docs/peak_time_reliability.md` §7.

## Procedure

```bash
# 1. Confirm nothing has drifted since this note was written
git diff --exit-code gentwin/service_area.py 01_service_area_osm.py

# 2. Fetch (online, once)
python 01_service_area_osm.py --fetch

# 3. Commit the cache immediately, before looking at downstream results
git add data/cache/osm_*.json && git commit -m "OSM fetch, pre-registered settings"

# 4. Only now regenerate and validate
python 03_population_synthesis.py --build
python 05_validate_against_bpdb.py --validate
```

Step 3 matters: committing the raw cache before running Stage B makes any
later change to the estimator visible in the history.

## What this note does not cover

The residential **tier** mix (Res-A/B/C) in `02_consumer_catalogue.json` is not
derived from OSM — there is no income data in OSM — and remains a documented
scenario. It also moves the daytime AC load and therefore the peak timing. It
is frozen at its current values (Dhanmondi 0.45/0.40/0.15; Kalyanpur
0.20/0.50/0.30) and is likewise not to be adjusted after seeing a T2 result.
