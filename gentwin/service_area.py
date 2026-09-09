"""
GenTwin-SG :: Stage A', geographic service area and building inventory
======================================================================

Purpose
-------
Decide the CONSUMER MIX at each substation from geography rather than by
drawing categories at random. OSM fixes the mix (what proportion of connected
demand is residential, commercial, hospital, ...); BPDB fixes the LEVEL (the
observed daily peak p_{i,d}). Equation (8)'s scaling xi(d) reconciles the two.
Neither source is asked to do the other's job.

Why floor area and not building counts
--------------------------------------
A hospital is not one house. Counting `building=*` features would make
Dhanmondi look ~95% residential and would destroy the very thing the category
split exists to reproduce - the midday institutional/commercial peak. The
estimator is therefore

    connected_demand_W  =  footprint_area_m2 x levels x intensity_W_per_m2

with intensity per category. `building:levels` is sparsely tagged in Dhaka, so
missing values fall back to a per-category default and the TAGGED FRACTION is
reported alongside the result: the reader can see how much of the inventory is
measured and how much is inferred.

Service-area geometry
---------------------
OSM has no substation service territories; no utility in Bangladesh publishes
them. Two approximations, in order of preference:

  voronoi  Thiessen polygons over the whole substation fleet, clipped to
           MAX_SERVICE_RADIUS_KM. Requires >= 3 fleet coordinates, i.e. a
           nodes_Substation.csv covering more than the two study sites.
  radius   A disc of SERVICE_RADIUS_KM around the site. Used when the fleet
           geometry is unavailable.

Both are approximations and both are recorded in the cache under
`service_area.method`, so no downstream figure can silently present one as the
other. A +/-20% radius sensitivity is available via `radius_km` override.

Network
-------
Overpass is not reachable from every environment (it is not on the sandbox
allow-list used to develop this module). The design consequence is deliberate:
`fetch_overpass()` is the ONLY function that touches the network, it writes
`data/cache/osm_<substation>.json`, and every other function in the pipeline
reads that cache. Run it once, commit the cache, and the whole framework is
reproducible offline forever after.

    python 01_service_area_osm.py --fetch            # online, once
    python 01_service_area_osm.py --offline-seed     # documented fallback
    python 01_service_area_osm.py --report           # read cache, summarise

`--offline-seed` writes a cache with `provenance = "fallback_literature"`
instead of `"overpass"`. Downstream code does not branch on provenance - the
schema is identical - but every figure and table carries the flag, so a
fallback run can never be mistaken for a surveyed one.
"""

from __future__ import annotations

import json
import math
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, asdict
from pathlib import Path

from . import config as cfg

OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]

SERVICE_RADIUS_KM = 2.0        # SCENARIO: typical 33 kV urban feeder reach
MAX_SERVICE_RADIUS_KM = 3.0    # Voronoi clip
OVERPASS_TIMEOUT_S = 180

# =============================================================================
# 1. OSM tag -> GenTwin consumer category
# =============================================================================
# Ordered: the first rule that matches wins, so `amenity=hospital` beats
# `building=yes`. Rules are (osm_key, {values}, category).

TAG_RULES = [
    ("amenity", {"hospital", "clinic", "doctors", "dentist", "nursing_home"},
     "Hospital"),
    ("healthcare", {"hospital", "clinic", "centre", "laboratory", "doctor"},
     "Hospital"),
    ("building", {"hospital"}, "Hospital"),

    ("amenity", {"school", "college", "university", "kindergarten", "library"},
     "Educational"),
    ("building", {"school", "college", "university", "kindergarten"},
     "Educational"),

    ("office", {"government", "administrative"}, "Government"),
    ("amenity", {"townhall", "courthouse", "police", "fire_station", "post_office",
                 "public_building"}, "Government"),
    ("building", {"government", "civic", "public"}, "Government"),

    ("building", {"industrial", "warehouse", "factory", "manufacture"},
     "Industrial"),
    ("landuse", {"industrial"}, "Industrial"),
    ("man_made", {"works"}, "Industrial"),

    ("building", {"commercial", "retail", "office", "supermarket", "kiosk",
                  "hotel", "shop", "mall"}, "Commercial"),
    ("shop", None, "Commercial"),            # any shop=* value
    ("office", None, "Commercial"),          # any other office=*
    ("amenity", {"restaurant", "cafe", "fast_food", "bank", "marketplace",
                 "cinema", "pharmacy"}, "Commercial"),

    ("building", {"apartments", "residential", "house", "detached", "dormitory",
                  "terrace", "bungalow", "yes"}, "Residential"),
    ("landuse", {"residential"}, "Residential"),
]

# =============================================================================
# 2. Category physical parameters
# =============================================================================
# intensity_w_per_m2  peak connected demand density
#   SOURCE LITERATURE (order of magnitude). Values are consistent with the
#   ranges used for South Asian urban distribution planning and with ASHRAE
#   connected-load densities; they are used only to set RELATIVE shares
#   between categories, because the absolute level is pinned to the BPDB peak
#   by xi(d) in equation (8). An error common to all categories cancels; only
#   the ratios between them survive into the model.
# default_levels
#   SOURCE SCENARIO, used where `building:levels` is untagged.

CATEGORY_PHYSICS = {
    "Residential":  {"intensity_w_per_m2": 15.0, "default_levels": 5,
                     "mean_floor_area_per_connection_m2": 90.0},
    "Commercial":   {"intensity_w_per_m2": 45.0, "default_levels": 6,
                     "mean_floor_area_per_connection_m2": 120.0},
    "Industrial":   {"intensity_w_per_m2": 40.0, "default_levels": 2,
                     "mean_floor_area_per_connection_m2": 600.0},
    "Hospital":     {"intensity_w_per_m2": 60.0, "default_levels": 8,
                     "mean_floor_area_per_connection_m2": 2500.0},
    "Educational":  {"intensity_w_per_m2": 25.0, "default_levels": 4,
                     "mean_floor_area_per_connection_m2": 1500.0},
    "Government":   {"intensity_w_per_m2": 40.0, "default_levels": 6,
                     "mean_floor_area_per_connection_m2": 800.0},
}

# =============================================================================
# 3. Documented offline fallback
# =============================================================================
# SOURCE SCENARIO / LITERATURE. Shares of CONNECTED DEMAND (not building
# counts) by category, for use when Overpass cannot be reached. Grounded in
# the published character of the two service areas:
#
#   Dhanmondi  A planned residential area that has converted heavily to mixed
#              use. Dense private-hospital cluster (Labaid, Popular, Ibn Sina,
#              Gonoshasthaya Nagar), a large concentration of schools and
#              coaching centres, and the Satmasjid Road / Mirpur Road retail
#              corridor. Almost no industry. This composition is the reason the
#              substation peaks at 13:00 rather than in the evening.
#   Kalyanpur  Predominantly residential (Kalyanpur, Shyamoli, Darus Salam
#              fringe), with the Mirpur Road commercial strip and pockets of
#              light industry and workshops toward Gabtoli. Institutional load
#              is comparatively thin, which is consistent with its 18:00-19:00
#              evening peak.
#
# These are estimates. They exist so the pipeline can run end-to-end without
# network access; they are NOT a substitute for the Overpass fetch, and every
# artefact built from them is stamped provenance="fallback_literature".

FALLBACK_MIX = {
    "Dhanmondi": {
        "Residential": 0.52, "Commercial": 0.24, "Educational": 0.09,
        "Hospital": 0.09, "Government": 0.05, "Industrial": 0.01,
    },
    "Kalyanpur": {
        "Residential": 0.68, "Commercial": 0.17, "Educational": 0.05,
        "Hospital": 0.03, "Government": 0.03, "Industrial": 0.04,
    },
}


# =============================================================================
# 4. Geometry
# =============================================================================

@dataclass
class ServiceArea:
    substation: str
    method: str                  # "voronoi" | "radius"
    centre_lat: float
    centre_lon: float
    radius_km: float
    polygon: list                # [[lat, lon], ...] closed ring
    coords_provisional: bool
    note: str = ""


def load_substation_coords(csv_path: Path | None = None) -> dict:
    """
    Prefer nodes_Substation.csv when present; fall back to the provisional
    coordinates in config, with the provisional flag preserved so that any
    figure built on them can say so.
    """
    csv_path = csv_path or (cfg.DATA_DIR / "nodes_Substation.csv")
    coords = {}
    if csv_path.exists():
        import pandas as pd
        df = pd.read_csv(csv_path)
        name_col = next((c for c in ("name", "substation", "nodeId")
                         if c in df.columns), df.columns[0])
        for _, r in df.iterrows():
            raw = str(r[name_col])
            key = next((k for k in cfg.SUBSTATIONS if k.lower() in raw.lower()), raw)
            try:
                coords[key] = {"latitude": float(r["latitude"]),
                               "longitude": float(r["longitude"]),
                               "provisional": False, "raw_name": raw}
            except (KeyError, TypeError, ValueError):
                continue
    for name, s in cfg.SUBSTATIONS.items():
        coords.setdefault(name, {"latitude": s["latitude"],
                                 "longitude": s["longitude"],
                                 "provisional": True, "raw_name": name})
    return coords


def _km_per_deg(lat: float) -> tuple[float, float]:
    """Local metric scale. Adequate over a 3 km neighbourhood."""
    return 110.574, 111.320 * math.cos(math.radians(lat))


def _disc(lat: float, lon: float, radius_km: float, n: int = 72) -> list:
    kla, klo = _km_per_deg(lat)
    ring = [[lat + radius_km / kla * math.sin(2 * math.pi * i / n),
             lon + radius_km / klo * math.cos(2 * math.pi * i / n)]
            for i in range(n)]
    ring.append(ring[0])
    return ring


def build_service_area(substation: str, coords: dict,
                       radius_km: float = SERVICE_RADIUS_KM) -> ServiceArea:
    """
    Voronoi cell clipped to MAX_SERVICE_RADIUS_KM when the fleet geometry is
    available (>= 3 sites), otherwise a disc.
    """
    me = coords[substation]
    lat, lon = me["latitude"], me["longitude"]
    fleet = [(n, c) for n, c in coords.items()]

    if len(fleet) >= 3:
        try:
            import numpy as np
            from scipy.spatial import Voronoi
            from shapely.geometry import Point, Polygon

            kla, klo = _km_per_deg(lat)
            pts = np.array([[(c["longitude"] - lon) * klo,
                             (c["latitude"] - lat) * kla] for _, c in fleet])
            idx = [n for n, _ in fleet].index(substation)
            # Distant sentinels bound the diagram so the study cells are finite.
            far = 500.0
            aug = np.vstack([pts, [[far, far], [-far, far], [far, -far], [-far, -far]]])
            vor = Voronoi(aug)
            region = vor.regions[vor.point_region[idx]]
            if region and -1 not in region:
                cell = Polygon([vor.vertices[i] for i in region])
                clipped = cell.intersection(
                    Point(pts[idx]).buffer(MAX_SERVICE_RADIUS_KM, quad_segs=32))
                if not clipped.is_empty and clipped.area > 0:
                    ring = [[lat + y / kla, lon + x / klo]
                            for x, y in clipped.exterior.coords]
                    return ServiceArea(
                        substation, "voronoi", lat, lon, MAX_SERVICE_RADIUS_KM,
                        ring, me["provisional"],
                        f"Thiessen cell over {len(fleet)} fleet sites, "
                        f"clipped at {MAX_SERVICE_RADIUS_KM} km",
                    )
        except Exception as exc:                                  # noqa: BLE001
            print(f"  Voronoi unavailable for {substation} ({exc}); using radius.")

    return ServiceArea(
        substation, "radius", lat, lon, radius_km, _disc(lat, lon, radius_km),
        me["provisional"],
        f"{radius_km} km disc; fleet geometry unavailable "
        f"({len(fleet)} coordinate(s) known)",
    )


# =============================================================================
# 5. Overpass
# =============================================================================

def build_overpass_query(area: ServiceArea) -> str:
    """
    One query per service area. `way` and `relation` only - a `node` has no
    footprint and cannot be floor-area weighted. `out center` gives us the
    centroid; geometry comes back for ways so footprint area is computable.
    """
    poly = " ".join(f"{lat:.6f} {lon:.6f}" for lat, lon in area.polygon[:-1])
    sel = "\n".join(
        f'  way(poly:"{poly}")[{k}];\n  relation(poly:"{poly}")[{k}];'
        for k in ("building", "amenity", "shop", "office", "healthcare", "landuse")
    )
    return f"[out:json][timeout:{OVERPASS_TIMEOUT_S}];\n(\n{sel}\n);\nout geom;"


def fetch_overpass(area: ServiceArea, retries: int = 2) -> dict:
    """
    THE ONLY networked function in this codebase. Writes nothing; the caller
    caches. Raises on total failure so `--fetch` fails loudly rather than
    silently degrading to the fallback.
    """
    query = build_overpass_query(area)
    last = None
    for endpoint in OVERPASS_ENDPOINTS:
        for attempt in range(retries):
            try:
                print(f"  Overpass: {endpoint} (attempt {attempt + 1})...")
                req = urllib.request.Request(
                    endpoint,
                    data=urllib.parse.urlencode({"data": query}).encode(),
                    headers={"User-Agent": "GenTwin-SG/1.0 (thesis research)"},
                )
                with urllib.request.urlopen(req, timeout=OVERPASS_TIMEOUT_S + 30) as r:
                    payload = json.loads(r.read().decode("utf-8"))
                print(f"  -> {len(payload.get('elements', []))} elements")
                return payload
            except Exception as exc:                              # noqa: BLE001
                last = exc
                print(f"     failed: {exc}")
                time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"All Overpass endpoints failed. Last error: {last}")


# =============================================================================
# 6. Classification and floor-area weighting
# =============================================================================

def classify(tags: dict) -> str | None:
    for key, values, category in TAG_RULES:
        if key in tags and (values is None or tags[key] in values):
            return category
    return None


def _ring_area_m2(geometry: list, lat0: float) -> float:
    """Shoelace on a local equirectangular projection."""
    if not geometry or len(geometry) < 3:
        return 0.0
    kla, klo = _km_per_deg(lat0)
    pts = [((p["lon"]) * klo * 1000.0, (p["lat"]) * kla * 1000.0) for p in geometry]
    a = sum(pts[i][0] * pts[(i + 1) % len(pts)][1]
            - pts[(i + 1) % len(pts)][0] * pts[i][1] for i in range(len(pts)))
    return abs(a) / 2.0


def inventory_from_overpass(payload: dict, area: ServiceArea) -> dict:
    """
    Turn raw Overpass elements into a per-category demand inventory.

    Reports `levels_tagged_fraction` per category, because the floor-area
    estimate is only as good as that number and the thesis should quote it.
    """
    cats = {c: {"n_features": 0, "footprint_m2": 0.0, "floor_area_m2": 0.0,
                "levels_tagged": 0, "connected_demand_w": 0.0}
            for c in cfg.CONSUMER_CATEGORIES}
    unclassified = 0

    for el in payload.get("elements", []):
        tags = el.get("tags") or {}
        cat = classify(tags)
        if cat is None:
            unclassified += 1
            continue
        foot = _ring_area_m2(el.get("geometry", []), area.centre_lat)
        if foot <= 0:
            continue
        phys = CATEGORY_PHYSICS[cat]
        lv_raw = tags.get("building:levels") or tags.get("levels")
        try:
            levels = max(1.0, float(lv_raw))
            tagged = 1
        except (TypeError, ValueError):
            levels = float(phys["default_levels"])
            tagged = 0
        floor = foot * levels
        c = cats[cat]
        c["n_features"] += 1
        c["footprint_m2"] += foot
        c["floor_area_m2"] += floor
        c["levels_tagged"] += tagged
        c["connected_demand_w"] += floor * phys["intensity_w_per_m2"]

    total_w = sum(c["connected_demand_w"] for c in cats.values())
    for cat, c in cats.items():
        c["demand_share"] = (c["connected_demand_w"] / total_w) if total_w else 0.0
        c["levels_tagged_fraction"] = (
            c["levels_tagged"] / c["n_features"] if c["n_features"] else 0.0)
        c["est_connections"] = (
            round(c["floor_area_m2"]
                  / CATEGORY_PHYSICS[cat]["mean_floor_area_per_connection_m2"])
            if c["floor_area_m2"] else 0)
    return {"categories": cats, "unclassified_elements": unclassified,
            "total_connected_demand_w": total_w}


def inventory_from_fallback(substation: str) -> dict:
    """Same schema, from FALLBACK_MIX. Shares are given directly."""
    mix = FALLBACK_MIX[substation]
    cats = {}
    for cat in cfg.CONSUMER_CATEGORIES:
        share = mix.get(cat, 0.0)
        cats[cat] = {
            "n_features": None, "footprint_m2": None, "floor_area_m2": None,
            "levels_tagged": None, "levels_tagged_fraction": None,
            "connected_demand_w": None, "demand_share": share,
            "est_connections": None,
        }
    return {"categories": cats, "unclassified_elements": None,
            "total_connected_demand_w": None}


# =============================================================================
# 7. Cache
# =============================================================================

def cache_path(substation: str) -> Path:
    return cfg.CACHE_DIR / f"osm_{substation.lower().replace(' ', '_')}.json"


def write_cache(substation: str, area: ServiceArea, inventory: dict,
                provenance: str, raw: dict | None = None) -> Path:
    record = {
        "schema_version": 1,
        "substation": substation,
        "provenance": provenance,          # "overpass" | "fallback_literature"
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "service_area": asdict(area),
        "category_physics": CATEGORY_PHYSICS,
        "inventory": inventory,
        "n_raw_elements": len(raw.get("elements", [])) if raw else None,
    }
    p = cache_path(substation)
    p.write_text(json.dumps(record, indent=1), encoding="utf-8")
    return p


def read_cache(substation: str) -> dict:
    p = cache_path(substation)
    if not p.exists():
        raise FileNotFoundError(
            f"No OSM cache for {substation} at {p}.\n"
            f"Run:  python 01_service_area_osm.py --fetch            (online)\n"
            f"  or: python 01_service_area_osm.py --offline-seed     (fallback)")
    return json.loads(p.read_text(encoding="utf-8"))


def demand_mix(substation: str) -> dict:
    """The one thing Stage 03 needs: category -> share of connected demand."""
    rec = read_cache(substation)
    return {c: v["demand_share"] for c, v in rec["inventory"]["categories"].items()}
