#!/usr/bin/env python3
"""
Stage A' runner - geographic service area and building inventory.

    python 01_service_area_osm.py --fetch          # online, run once, then commit
                                                   # data/cache/osm_*.json
    python 01_service_area_osm.py --offline-seed   # documented fallback mix
    python 01_service_area_osm.py --report         # summarise whatever is cached
    python 01_service_area_osm.py --print-query    # emit the Overpass QL to run
                                                   # by hand at overpass-turbo.eu

--fetch is the only mode that needs the network. Everything downstream reads
the cache, so the pipeline is reproducible offline once this has been run.
"""

import argparse
import json
import sys

from gentwin import config as cfg
from gentwin import service_area as sa


def _summarise(substation: str) -> None:
    rec = sa.read_cache(substation)
    area = rec["service_area"]
    prov = rec["provenance"]
    flag = "  [FALLBACK - not surveyed]" if prov != "overpass" else ""
    print(f"\n{substation}{flag}")
    print(f"  provenance     : {prov}")
    print(f"  generated      : {rec['generated_utc']}")
    print(f"  service area   : {area['method']}, r={area['radius_km']} km, "
          f"centre ({area['centre_lat']:.4f}, {area['centre_lon']:.4f})"
          f"{'  [PROVISIONAL COORDS]' if area['coords_provisional'] else ''}")
    print(f"  note           : {area['note']}")
    if rec["n_raw_elements"] is not None:
        print(f"  OSM elements   : {rec['n_raw_elements']} raw, "
              f"{rec['inventory']['unclassified_elements']} unclassified")
    print(f"  {'category':13} {'share':>7} {'features':>9} "
          f"{'floor m2':>12} {'levels tagged':>14} {'connections':>12}")
    for cat, v in rec["inventory"]["categories"].items():
        n = v["n_features"]
        fa = v["floor_area_m2"]
        lt = v["levels_tagged_fraction"]
        ec = v["est_connections"]
        print(f"  {cat:13} {v['demand_share']:7.3f} "
              f"{('-' if n is None else n):>9} "
              f"{('-' if fa is None else f'{fa:,.0f}'):>12} "
              f"{('-' if lt is None else f'{lt:.0%}'):>14} "
              f"{('-' if ec is None else f'{ec:,}'):>12}")
    tot = sum(v["demand_share"] for v in rec["inventory"]["categories"].values())
    print(f"  {'TOTAL':13} {tot:7.3f}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fetch", action="store_true", help="query Overpass (online)")
    p.add_argument("--offline-seed", action="store_true",
                   help="write the documented fallback mix instead")
    p.add_argument("--report", action="store_true", help="summarise the cache")
    p.add_argument("--print-query", action="store_true",
                   help="print Overpass QL for manual execution")
    p.add_argument("--radius-km", type=float, default=sa.SERVICE_RADIUS_KM,
                   help="service radius; use +/-20%% for the sensitivity check")
    p.add_argument("--substations", nargs="*", default=list(cfg.SUBSTATIONS))
    args = p.parse_args()

    if not any((args.fetch, args.offline_seed, args.report, args.print_query)):
        p.print_help()
        return 1

    coords = sa.load_substation_coords()
    if any(c["provisional"] for c in coords.values()):
        prov = [n for n, c in coords.items() if c["provisional"]]
        print(f"NOTE: provisional coordinates in use for {', '.join(prov)}.")
        print(f"      Drop nodes_Substation.csv into {cfg.DATA_DIR} to override.\n")

    for name in args.substations:
        area = sa.build_service_area(name, coords, radius_km=args.radius_km)

        if args.print_query:
            print(f"\n{'=' * 70}\n-- {name}: paste at https://overpass-turbo.eu\n"
                  f"{'=' * 70}")
            print(sa.build_overpass_query(area))
            continue

        if args.fetch:
            print(f"\nFetching {name} ({area.method}, r={area.radius_km} km)")
            try:
                raw = sa.fetch_overpass(area)
            except RuntimeError as exc:
                print(f"  ERROR: {exc}")
                print("  Not falling back silently. Either re-run with network "
                      "access, or run --offline-seed deliberately.")
                return 2
            inv = sa.inventory_from_overpass(raw, area)
            path = sa.write_cache(name, area, inv, "overpass", raw)
            print(f"  cached -> {path}")

        elif args.offline_seed:
            inv = sa.inventory_from_fallback(name)
            path = sa.write_cache(name, area, inv, "fallback_literature")
            print(f"{name}: fallback mix cached -> {path}")

    if args.report or args.fetch or args.offline_seed:
        print(f"\n{'=' * 70}\nSERVICE AREA / CONSUMER MIX\n{'=' * 70}")
        for name in args.substations:
            try:
                _summarise(name)
            except FileNotFoundError as exc:
                print(f"\n{name}: {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
