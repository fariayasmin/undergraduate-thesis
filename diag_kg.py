"""
diag_kg.py - READ-ONLY diagnosis of why outputs/kg differs from an earlier build.
Writes nothing, modifies nothing. Run from the repository root:

    python diag_kg.py                      # auto-detects tags
    python diag_kg.py --tag _period_drw_ir # also runs the population check for one tag
"""
import argparse, glob, json, os, time
from pathlib import Path
import pandas as pd

OUT = Path("outputs"); KG = OUT / "kg"; CACHE = Path("data") / "cache"
EXPECTED = {"DrAction": 272102, "Appliance": 5448, "PoolEntity": 1560, "PriorityPool": 90}
CURRENT = {"DrAction": 288348, "Appliance": 6012, "PoolEntity": 1700, "PriorityPool": 90}

def ts(p):
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(p))) if os.path.exists(p) else "MISSING"

def nrows(p):
    return sum(1 for _ in open(p, encoding="utf-8")) - 1

def hr(t): print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)

ap = argparse.ArgumentParser(); ap.add_argument("--tag", default=None); a = ap.parse_args()

# 1. schema.json vs CSVs on disk ------------------------------------------------
hr("1. schema.json vs CSV row counts in outputs/kg")
schema = json.loads((KG / "schema.json").read_text(encoding="utf-8"))
nc = schema.get("node_counts", {}); rc = schema.get("relationship_counts", {})
print(f"schema.json written: {ts(KG / 'schema.json')}")
mism = 0
for f in sorted(KG.glob("nodes_*.csv")):
    lab = f.stem.replace("nodes_", ""); n = nrows(f); s = nc.get(lab)
    flag = "" if s == n else "   <-- MISMATCH"; mism += bool(flag)
    print(f"  {lab:<22} csv={n:>9,}  schema={s if s is not None else '-':>9}  {ts(f)}{flag}")
rel = KG / "relationships.csv"
if rel.exists():
    r = pd.read_csv(rel, usecols=lambda c: c.lower() in ("type", ":type", "rel_type"))
    col = r.columns[0]; counts = r[col].value_counts()
    for k, v in counts.items():
        s = rc.get(k); flag = "" if s == v else "   <-- MISMATCH"; mism += bool(flag)
        print(f"  REL {k:<18} csv={v:>9,}  schema={s if s is not None else '-':>9}{flag}")
    print(f"  TOTAL relationships csv={len(r):,}")
print(f"  total nodes csv={sum(nrows(f) for f in KG.glob('nodes_*.csv')):,}")
print("  RESULT:", "schema.json MATCHES the CSVs" if not mism else f"{mism} mismatches")

# 2. provenance ----------------------------------------------------------------
hr("2. Provenance of outputs/kg")
bp = KG / "nodes_BillingPeriod.csv"
if bp.exists():
    print(pd.read_csv(bp).T.to_string())
day = KG / "nodes_Day.csv"
if day.exists():
    d = pd.read_csv(day); dc = [c for c in d.columns if "date" in c.lower()][0]
    print(f"Day nodes: {len(d)}  range {d[dc].min()} -> {d[dc].max()}")

# 5. DrAction candidates per tag -----------------------------------------------
hr("5. Active actions per tag (compare with DrAction 272,102 / 288,348)")
tags = sorted(p.replace(str(OUT / "lp_thresholds"), "").replace(".csv", "")
              for p in glob.glob(str(OUT / "lp_thresholds*.csv")))
for t in tags:
    p = OUT / f"lp_thresholds{t}.csv"
    try:
        th = pd.read_csv(p, usecols=lambda c: c in ("value", "action"))
        act = int((th["value"] > 1e-9).sum()) if "value" in th else -1
        tag_hit = " <-- matches EARLIER" if act == EXPECTED["DrAction"] else \
                  (" <-- matches CURRENT" if act == CURRENT["DrAction"] else "")
        print(f"  {t or '(untagged)':<34} rows={len(th):>9,} active={act:>9,}  {ts(p)}{tag_hit}")
    except Exception as e:
        print(f"  {t:<34} unreadable ({e})")

# 6. PoolEntity per day --------------------------------------------------------
hr("6. Ranked-pool rows per day (expect 52/day)")
for p in sorted(glob.glob(str(OUT / "priority_pool_ranked*.csv"))):
    df = pd.read_csv(p); dc = [c for c in df.columns if "date" in c.lower()]
    per = df.groupby(dc[0]).size() if dc else pd.Series(dtype=int)
    extra = [c for c in df.columns if c.lower() in ("method", "ranking", "comparator", "variant")]
    print(f"  {Path(p).name:<48} rows={len(df):>6,} days={len(per):>3} "
          f"min/max per day={per.min() if len(per) else '-'}/{per.max() if len(per) else '-'} "
          f"days>52={int((per > 52).sum()) if len(per) else '-'} comparator_col={extra or 'none'}")

# 3/4. Population consistency --------------------------------------------------
hr("3/4. Population cache vs LP outputs vs KG")
caches = sorted(CACHE.glob("population_*.json"))
pop_cons, pop_apps, pop_lam = set(), set(), {}
for c in caches:
    rec = json.loads(c.read_text(encoding="utf-8"))
    print(f"  {c.name:<34} written {ts(c)}  n_consumers={rec.get('n_consumers')}  "
          f"seed={rec.get('seed', rec.get('rng_seed', '?'))}")
    for h in rec["consumers"]:
        cid = h.get("consumer_id"); pop_cons.add(cid); pop_lam[cid] = h.get("lambda_h")
        for i, ap_ in enumerate(h.get("appliances", [])):
            pop_apps.add(ap_.get("appliance_id", f"{cid}-A{i:02d}"))
print(f"  population cache: {len(pop_cons)} consumers, {len(pop_apps)} appliances")
kga = KG / "nodes_Appliance.csv"
if kga.exists():
    kg_apps = set(pd.read_csv(kga)["appliance_id"])
    print(f"  KG Appliance nodes: {len(kg_apps)}; in cache: {len(kg_apps & pop_apps)}; "
          f"only in KG: {len(kg_apps - pop_apps)}; only in cache: {len(pop_apps - kg_apps)}")

if a.tag is not None:
    p = OUT / f"lp_thresholds{a.tag}.csv"
    th = pd.read_csv(p, usecols=lambda c: c in ("consumer", "lambda_h"))
    lp_cons = set(th["consumer"])
    print(f"\n  tag {a.tag}: LP written {ts(p)}; population caches written "
          f"{', '.join(ts(c) for c in caches)}")
    print(f"  consumers in LP not in cache: {len(lp_cons - pop_cons)}; "
          f"in cache not in LP: {len(pop_cons - lp_cons)}")
    if "lambda_h" in th:
        lam = th.drop_duplicates("consumer").set_index("consumer")["lambda_h"]
        diff = [c for c, v in lam.items()
                if c in pop_lam and pop_lam[c] is not None and abs(v - pop_lam[c]) > 1e-9]
        print(f"  lambda_h mismatches LP vs cache: {len(diff)} of {len(lam)}"
              + ("   <-- POPULATION REGENERATED AFTER THE LP SOLVE" if diff else "   (consistent)"))
print("\nDone. Nothing was written or modified.")
