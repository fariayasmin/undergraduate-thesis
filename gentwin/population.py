"""
GenTwin-SG :: Stage A/B - population synthesis
==============================================

Turns the OSM service-area mix into a population of representative consumers,
each carrying a multiplicity weight w_h.

Division of labour, restated because it governs every choice here:

    OSM      fixes the MIX      - what share of connected demand is
                                  residential, commercial, hospital, ...
    BPDB     fixes the LEVEL    - the observed daily peak p_{i,d}
    xi(d)    reconciles them    - Eq. (7)-(8)

Neither source is asked to do the other's job. The catalogue supplies the
shape; the mix supplies the composition; the peak supplies the scale.

Representative consumers and the weight w_h
-------------------------------------------
Dhanmondi's peak is ~136,500 kW. At 1-3 kW per household that is tens of thousands
of connections; enumerating them would make both the LP and the knowledge graph
unusable. Instead N_REPRESENTATIVE_CONSUMERS (400) consumers per substation are
synthesised, each with a weight w_h = the number of real connections it stands
for.

This is EXACT, not an approximation. L_h scales by w_h while x_{h,t} and
y_{h,t} remain fractions, so the programme stays linear and the Proposition 2
reduced cost merely divides through by theta_3 * w_h * L^cu_h(t) * Delta > 0,
which is still strictly positive. The threshold is unchanged.

w_h is stored on the knowledge-graph node, so any future explanation can state
whether it is describing one consumer or a weighted population.

How the weights are set
-----------------------
Two stages, in this order, because they answer two different questions:

  1. WITHIN-SUBSTATION COMPOSITION. For category c with OSM demand share s_c,
     every representative in c gets a relative weight v_h = s_c / E_c, where
     E_c is the summed daily energy of the unweighted representatives in c.
     By construction sum_{h in c} v_h * e_h = s_c exactly, so the synthetic
     population reproduces the OSM mix to machine precision.

  2. ABSOLUTE LEVEL. A single global constant K scales every v_h so that
     max_t Lambda_i(t) equals the BPDB reference peak p_{i,d} exactly, with
     xi = 1 on the reference day. Other days then use xi(d) of Eq. (7)-(8).

So the reference day calibrates the population once, and xi(d) carries the
day-to-day variation. w_h = K * v_h is then literally a count of real
connections, and `--report` prints the implied total so it can be sanity-
checked against what a Dhaka 132/33 kV substation plausibly serves.

Within-category multiplicity is uniform: all representatives of a category
share the same w. Diversity comes from appliance sampling and schedule jitter,
not from unequal weights. Making w heterogeneous as well would let a single
representative dominate its category with no evidence to support it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np

from . import config as cfg
from . import service_area as sa

CATALOGUE_PATH = cfg.ROOT / "02_consumer_catalogue.json"
MIN_REPS_PER_CATEGORY = 8       # so a small category is still represented
JITTER_SLOTS = 2                # +/- slots of per-consumer schedule offset
JITTER_AMPLITUDE_SD = 0.12      # per-consumer amplitude spread


# =============================================================================
# 1. Catalogue
# =============================================================================

def load_catalogue(path: Path | None = None) -> dict:
    p = path or CATALOGUE_PATH
    if not p.exists():
        raise FileNotFoundError(f"Consumer catalogue not found at {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def expand_schedule(spec: dict) -> np.ndarray:
    """
    Piecewise {from, to, level} blocks -> a 48-slot array. Later blocks win on
    overlap; uncovered slots take `default`. Blocks may wrap past midnight.
    """
    u = np.full(cfg.SLOTS_PER_DAY, float(spec.get("default", 0.0)))
    for b in spec.get("blocks", []):
        a, z, lv = float(b["from"]), float(b["to"]), float(b["level"])
        sa_, sz = int(round(a * 2)), int(round(z * 2))
        idx = (list(range(sa_, cfg.SLOTS_PER_DAY)) + list(range(0, sz))
               if sz <= sa_ else list(range(sa_, sz)))
        for i in idx:
            u[i % cfg.SLOTS_PER_DAY] = lv
    return u


def sample_on_slots(schedule: np.ndarray, daily_hours: float, rng,
                    spread: float = 0.25) -> list:
    """
    Draw one household's u_{h,a}(t), Eq. (2), as a BINARY on/off pattern.

    The schedule level is a PROBABILITY WEIGHT, not a duty fraction: it says
    how likely a slot is to be one of this appliance's ON slots. The number of
    ON slots comes from `daily_hours`, jittered per household.

    This is what makes households genuinely different from one another. Two
    Res-A households with identical appliances will run their air conditioning
    at different hours, which is why a population of them has a realistic
    coincidence factor instead of behaving like one scaled-up household.
    """
    n_slots = int(round(max(rng.normal(daily_hours, spread * daily_hours), 0.0)
                        / cfg.DELTA_H))
    n_slots = int(min(max(n_slots, 0), cfg.SLOTS_PER_DAY))
    if n_slots == 0:
        return []
    if n_slots >= cfg.SLOTS_PER_DAY:
        return list(range(cfg.SLOTS_PER_DAY))
    w = np.asarray(schedule, dtype=float).clip(min=0)
    if w.sum() <= 0:
        w = np.ones(cfg.SLOTS_PER_DAY)
    # Floor the weights at a small fraction of the peak. A schedule with hard
    # zeros (a chore block, say) would otherwise have fewer eligible slots than
    # the runtime needs; the floor also admits the occasional off-pattern hour,
    # which real households have.
    w = w + 1e-3 * w.max()
    p = w / w.sum()
    return sorted(rng.choice(cfg.SLOTS_PER_DAY, size=n_slots, replace=False,
                             p=p).tolist())


def schedule_implied_hours(schedule: np.ndarray) -> float:
    """Hours a schedule implies if read as a duty fraction - the fallback for
    non-residential appliances, which are already plausible."""
    return float(np.asarray(schedule).sum() * cfg.DELTA_H)


def all_schedules(cat: dict) -> dict:
    return {k: expand_schedule(v) for k, v in cat["schedules"].items()
            if not k.startswith("_")}


def thermal_factor(cat: dict, kind: str, theta: np.ndarray) -> np.ndarray:
    """
    f_a(Theta(t)) of Eq. (2), from the `thermal_responses` table.
    Linear above a reference temperature, capped.
    """
    spec = cat["thermal_responses"][kind]
    beta = float(spec["beta_per_c"])
    if beta == 0.0:
        return np.ones_like(theta)
    ref = float(spec["theta_ref_c"])
    cap = float(spec["cap"])
    return np.clip(1.0 + beta * np.maximum(theta - ref, 0.0), 1.0, cap)


# =============================================================================
# 2. Consumers
# =============================================================================

@dataclass
class Appliance:
    name: str
    cls: str                    # critical | shiftable | curtailable
    n: int                      # n_{h,a}
    p_rat_kw: float             # P^rat_a, kW
    duty_cycle: float           # epsilon_a
    schedule: str
    thermal: str
    simultaneity: float = 1.0   # fraction of the n units drawing at once
    on_slots: list = field(default_factory=list)   # u_{h,a}(t), Eq. (2)
    daily_hours: float = 0.0


@dataclass
class Consumer:
    consumer_id: str
    substation: str
    category: str
    archetype: str              # socio-economic band; see Remark 4
    lambda_h: float             # section 1.3, per-household
    gamma_sh_h: float
    gamma_cu_h: float
    y_max_h: float              # Eq. (34), per-household by the extension
    appliances: list
    w: float = 1.0              # multiplicity weight
    jitter_offset: int = 0
    jitter_amplitude: float = 1.0
    connected_load_kw: float = 0.0
    income_band: str = ""
    monthly_income_bdt: float = 0.0
    household_size: int = 0

    def to_record(self) -> dict:
        r = asdict(self)
        r["appliances"] = [asdict(a) if not isinstance(a, dict) else a
                           for a in self.appliances]
        return r


def _allocate_counts(mix: dict, n_total: int) -> dict:
    """
    Split N representatives across categories by demand share, with a floor so
    that a small but electrically important category (hospitals at 3-9%) is
    still resolved by more than one or two consumers.
    """
    present = {c: s for c, s in mix.items() if s > 0}
    counts = {c: max(MIN_REPS_PER_CATEGORY, int(round(s * n_total)))
              for c, s in present.items()}
    # Rescale back to n_total, never below the floor.
    while sum(counts.values()) > n_total:
        c = max((c for c in counts if counts[c] > MIN_REPS_PER_CATEGORY),
                key=lambda c: counts[c], default=None)
        if c is None:
            break
        counts[c] -= 1
    while sum(counts.values()) < n_total:
        c = max(present, key=lambda c: present[c] / counts[c])
        counts[c] += 1
    return counts


def synthesise(substation: str, cat: dict, mix: dict,
               n_total: int | None = None, seed: int | None = None,
               schedules: dict | None = None) -> list:
    """Build the unweighted representative population for one substation."""
    n_total = cfg.N_REPRESENTATIVE_CONSUMERS if n_total is None else n_total
    rng = np.random.default_rng(
        (cfg.POPULATION_SEED if seed is None else seed)
        + sum(ord(ch) for ch in substation))

    schedules = schedules if schedules is not None else all_schedules(cat)
    counts = _allocate_counts(mix, n_total)
    arch_mix = cat["residential_archetype_mix"].get(substation, {})
    consumers: list[Consumer] = []
    idx = 0

    for category, n_c in counts.items():
        spec = cat["categories"][category]
        archetypes = spec["archetypes"]
        for _ in range(n_c):
            if category == "Residential" and arch_mix:
                names = [k for k in arch_mix if not k.startswith("_")]
                probs = np.array([arch_mix[t] for t in names], dtype=float)
                arch = str(rng.choice(names, p=probs / probs.sum()))
            else:
                arch = archetypes[0]
            prefs = cfg.sample_preferences(arch, rng)
            socio = cat.get("socioeconomic", {}).get(arch, {})
            if socio:
                lo, hi = socio["income_bdt"]
                income = float(rng.uniform(lo, hi))
                s_lo, s_hi = socio["household_size"]
                hh_size = int(rng.integers(s_lo, s_hi + 1))
            else:
                income, hh_size = 0.0, 0

            apps, connected = [], 0.0
            for a in spec["appliances"]:
                if rng.random() > float(a["ownership"].get(arch, 0.0)):
                    continue
                lo, hi = a["n_range"].get(arch, [1, 1])
                n = int(rng.integers(lo, hi + 1))
                sched = a.get("schedule_override", {}).get(arch, a["schedule"])
                u_prob = schedules[sched]

                dh = a.get("daily_hours")
                if isinstance(dh, dict):
                    dh = dh.get(arch)
                if dh is None:
                    dh = schedule_implied_hours(u_prob)
                sim = float(a.get("simultaneity", 1.0))

                apps.append(Appliance(
                    name=a["name"], cls=a["class"], n=n,
                    p_rat_kw=float(a["p_rat_kw"]),
                    duty_cycle=float(a["duty_cycle"]),
                    schedule=sched, thermal=a["thermal"],
                    simultaneity=sim, daily_hours=float(dh),
                    on_slots=sample_on_slots(u_prob, float(dh), rng)))
                connected += n * sim * float(a["p_rat_kw"])

            if not apps:            # a consumer with no appliances is not one
                continue

            idx += 1
            consumers.append(Consumer(
                consumer_id=f"{cfg.SUBSTATIONS[substation]['code']}C{idx:04d}",
                substation=substation, category=category, archetype=arch,
                income_band=socio.get("band", ""),
                monthly_income_bdt=income,
                household_size=hh_size,
                lambda_h=prefs["lambda_h"], gamma_sh_h=prefs["gamma_sh_h"],
                gamma_cu_h=prefs["gamma_cu_h"], y_max_h=prefs["y_max_h"],
                appliances=apps,
                jitter_offset=int(rng.integers(-JITTER_SLOTS, JITTER_SLOTS + 1)),
                jitter_amplitude=float(np.clip(
                    rng.normal(1.0, JITTER_AMPLITUDE_SD), 0.6, 1.4)),
                connected_load_kw=connected))
    return consumers


def demand_mix_for(substation: str) -> dict:
    """OSM category shares, normalised. Fails loudly if the cache is absent."""
    mix = sa.demand_mix(substation)
    total = sum(mix.values())
    if total <= 0:
        raise ValueError(f"{substation}: OSM demand mix is empty")
    return {c: v / total for c, v in mix.items()}


# =============================================================================
# 3. Cache
# =============================================================================

def cache_path(substation: str) -> Path:
    return cfg.CACHE_DIR / f"population_{substation.lower().replace(' ', '_')}.json"


def write_cache(substation: str, consumers: list, meta: dict) -> Path:
    rec = {"schema_version": 1, "substation": substation,
           "n_consumers": len(consumers), **meta,
           "consumers": [c.to_record() for c in consumers]}
    p = cache_path(substation)
    p.write_text(json.dumps(rec, indent=1, default=str), encoding="utf-8")
    return p


def read_cache(substation: str) -> dict:
    p = cache_path(substation)
    if not p.exists():
        raise FileNotFoundError(
            f"No population for {substation}. Run: python 03_population_synthesis.py --build")
    return json.loads(p.read_text(encoding="utf-8"))


def consumers_from_cache(substation: str) -> list:
    rec = read_cache(substation)
    out = []
    for c in rec["consumers"]:
        apps = [Appliance(**a) for a in c["appliances"]]
        d = {k: v for k, v in c.items() if k != "appliances"}
        out.append(Consumer(appliances=apps, **d))
    return out
