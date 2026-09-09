# Formulation revision: what changed, and what it costs in code

Diff of `GenTwinSG_Formulation__2_.pdf` against the v3 used for Stages 0–B.
Nine substantive changes plus renumbering. Two of them resolve problems this
implementation had previously flagged.

---

## 1. Units: kW / kWh / Tk (was MW / MWh)

Every dimensional symbol is now given a unit in §1, and they are **kW, kWh and
Tk/kWh** throughout. The previous implementation worked in MW/MWh with prices
converted to BDT/MWh.

**Effect:** the internal unit convention flips to kW/kWh/Tk. This *simplifies*
the code — the BERC gazette is already in Tk/kWh and `γ`, `ρ`, `π^ret` are all
catalogued in Tk/kWh, so the `gamma_to_bdt_per_mwh()` conversion layer
disappears entirely. Appliance ratings move from W to kW at Eq. (2), matching
`P^rat_a` in kW.

Substation peaks become 136,500 kW rather than 136.50 MW. Coefficient spread in
the LP is comparable either way.

## 2. Preferences are per-household, not per-tier

v3 carried `λ_g`, `γ^sh_g`, `γ^cu_g` indexed by a tier set `G`. The revision
drops `G` entirely: §1.3 defines `λ_h`, `γ^sh_h`, `γ^cu_h` per household, and
Remark 4 is explicit —

> "Household heterogeneity is individual. Preferences are carried by `λ_h`,
> `γ^sh_h`, `γ^cu_h` per household, with no grouping into classes. Grouping, if
> wanted later, is a restriction of these parameters and changes nothing
> structural."

**Effect:** `CONSUMER_TIERS` becomes `CONSUMER_ARCHETYPES` — distributions from
which each consumer's own `λ_h`, `γ^sh_h`, `γ^cu_h` are *sampled* at synthesis
time and stored on the consumer. The archetypes remain as the generator (which
Remark 4 explicitly permits) but are no longer the unit of preference. Every
threshold, explanation and knowledge-graph node now carries the household's own
parameters.

This is strictly better for the explainability layer: Proposition 2 becomes a
statement about *this* household rather than about its class.

## 3. The curtailment cap is `y^max`, not `κ` — collision resolved

v3 used `κ` for both the curtailment cap (C6) and the robust scaling factor
(Eq. 16), which `docs/y_max_h_derivation.md` opened by flagging. The revision
renames the cap to **`y^max`** (§1.4, Eq. 34).

**Effect:** the notation warning is obsolete and the code renames
`kappa_cap` / `kappa_g` → `y_max` / `y_max_h`. `κ` now unambiguously means the
scaling factor of Eq. (16) everywhere.

The approved per-consumer extension survives as `y^max → y^max_h`. It is now a
*smaller* extension than before: with preferences already per-household, a
per-household cap is the natural completion rather than a new axis. The
derivation is unchanged in substance — see the rewritten
`docs/y_max_h_derivation.md`.

## 4. Stress threshold is `P^str_i`, not `τ_i`

§1.4 and Eq. (15). The document adds: *"The stress threshold is written `P^str_i`
and carries the unit of a peak load, so that (15) compares two quantities in kW.
The symbol τ is reserved throughout for a slot index."*

**Effect:** `tau_mw` → `p_str_kw` throughout `forecast.py` and `config.py`. Our
trailing-window estimator is unchanged; only the name and unit move. Note the
revision also frees `τ`, which we had not been using as a slot index — no
conflict introduced.

## 5. NEW — procurement is daily, dispatch is per slot

The largest structural change. Eq. (18) defines three **daily** totals as
substituted quantities, not new variables:

```
G_{i,d}  := Σ_{t∈T_d} G_i(t)Δ            kWh purchased that day
B_{i,d}  := Σ_{t∈T_d} (B^ch + B^dis)Δ     kWh throughput
T_{ij,d} := Σ_{t∈T_d} T_ij(t)Δ            kWh exported
```

and `f^cost_i` (Eq. 22) is rewritten over **days**:

```
f^cost_i = Σ_d [ c_{i,d}·G_{i,d} + c^B·B_{i,d} + Σ_j π_ij·T_{ij,d} ]
```

The producer price `c_{i,d}` is now **one flat price per day**, fixed when the
block is purchased — not a per-slot `c_i(t)`.

**Effect:** the LP objective is assembled from daily energy totals rather than
per-slot costs. Since each `G_{i,d}` is a linear functional of the per-slot
`G_i(t)`, this adds **no variable and no constraint** — exactly like the state
of charge. Config gains `c_producer_tk_per_kwh` as a per-day series.

Proposition 3's merit order (39) now reads `θ₁c_{i,d} < θ₁π_ij < θ₂π^ν <
θ₂ min_c v_c`, with `c_{i,d}` daily.

## 6. `f^rel` — Δ attaches only to `Z`, not to `ν`

Eq. (23):

```
f^rel_i = Σ_t ( π^ν·ν_i(t) + Σ_c v_c·Z^c_i(t)·Δ )
```

with the reason given: *"the factor Δ attaches only to `Z^c_i(t)`, which is a
power in kW; the shortfall `ν_i(t)` is already an energy in kWh, since it is
compared against the reserve floor `R_i`."*

**Effect:** a genuine correction. Our previous objective multiplied both terms
by Δ, which at Δ = 0.5 h **understated the reserve-shortfall penalty by a factor
of two**. Fixed. `ν` is now typed as kWh in the variable table and the code.

## 7. NEW — `E^srv_{i,m}` is defined explicitly (Eq. 41)

v3 used `E^srv_{i,m}` in the Stage G denominator without defining it. The
revision defines it and states three properties: it uses the **post-response**
load `L_h(t;x*,y*)`, it **subtracts unserved energy** `Z^{c*}_i(t)`, and it
**does not include κ** — *"that factor is a robustness margin applied to a
constraint, not metered energy."*

**Effect:** removes an ambiguity we would otherwise have had to resolve by
choice. The definition is directly verifiable against billing data, which the
document notes.

## 8. Surcharge: `ϖ_{i,m}`, and the cap changed

`ς_{i,m}` → `ϖ_{i,m}` (the symbol v3's Remark 1 offered as an alternative). The
cap (43) is now

```
ϖ_{i,m} ≤ min_{h∈H_i} π^ret_h
```

with the rationale *"no household's effective price is more than doubled"* —
where v3 had `ς ≤ δ^max · min_h π^ret_h`.

**Effect:** `DELTA_MAX_SURCHARGE` is deleted from config; the cap is now the
minimum retail tariff itself. Eq. (40) also now sums over importing links
`j:(j,i)∈L` and uses the daily `c_{i,d(t)}`.

## 9. Objective assembled per substation, then summed

`F_i` (25) then `F = Σ_i F_i` (26). Remark 1 spells out that `S = {i}`, `L = ∅`
gives a complete single-substation model with all propositions holding verbatim.

**Effect:** structurally equivalent to v3's single sum, but the code should
build `f^cost_i`, `f^rel_i`, `f^wel_i` per substation and report them per
substation. That is better for the knowledge graph anyway — `OptimizationRun`
can carry a per-substation cost breakdown.

It also means **Dhanmondi and Kalyanpur can legitimately be solved separately**
if the tie-line is dropped, which is a useful fallback given Dhanmondi's
outstanding data question.

## 10. Renumbering and symbol hygiene (no code effect beyond references)

| v3 | revised | |
|---|---|---|
| (17)–(25) objective block | (17)–(26) | `F_i` inserted |
| (26) state of charge | (28) | |
| (27)–(33) C1–C7 | (29)–(35) | |
| (34)(35) Prop 2 | (36)(37) | |
| (36) λ* | (38) | now `λ*_h` |
| (37) merit order | (39) | |
| (38)–(40) Stage G | (40)–(43) | (41) is new |
| (41)–(51) Stage H | (44)–(53) | |
| (46) regime | (48) | |
| (47) χ weights | (49) | |
| (52)–(58) Stage I–J | (54)–(60) | |

Symbol renames: `W`,`F` (window/features) → `n_w`,`n_f`; `r_n`,`r̂_n` →
`b_n`,`b̂_n`; `d_e` → `deg(e)`; `n` → `n_ent`; `W_p` → `W(P)`; `α_p` → `α_P`;
`z` → `z^adp`; `cl_Ω` → `cl_tol`; `ℓ` (KG label map) → `lab`.

## 11. Section 8 instance — now fully specified

The revision states what we previously had to infer: `y^max = 0.40`,
`λ_{h1} = 0.60`, `λ_{h2} = 0.35`, `λ_{h3} = 0.20`, `γ^cu_h = 20`,
`π^ret_h μ^pk = 12`, `ρ = 3`, `Δ = 1`, `P^max_1 = 13.0`. **Every value we
reverse-engineered from the §8.2 tables in the earlier reconstruction is
confirmed correct.** The instance also gains daily totals
(`G_{1,d+1} = 25.8532` kWh, `B_{1,d+1} = 2.5` kWh, `T_{12,d+1} = 0`) and a
Stage G check (`ΔC_{1,m} = 0`, `ϖ_{1,m} = 0`).

This makes `06a_verify_propositions.py` a much stronger test than planned: the
missing parameters are `c_{i,d+1}`, `c^B`, `π^ν`, `v_c`, `E^max`, `B^max`, `R`,
`T^max`, `ℓ` and the three households' `L^c_h(t)` — and `G_1`, `S_1` and the
duals in §8.1 constrain most of those.

---

## What still needs care: the Δ in Proposition 2

Unchanged from v3 and still worth stating. Eq. (37)'s system term is written
`κ_{i,d}(t)·π_i(t)/θ₃`. The proof divides the reduced cost by `θ₃L^cu_h(t)Δ`,
so the term is really `κπ/(θ₃Δ)`.

The document defines `π_i(t)` in Tk/**kWh** while C1 is written in **kW**, so
the LP's raw dual of C1 has units Tk/kW and must be divided by Δ to become the
Tk/kWh quantity §1.7 describes. At Δ = 1 h — the Section 8 instance — the two
coincide and the distinction is invisible. **At our Δ = 0.5 h it is a factor of
two**, so the implementation divides the raw dual by Δ before reporting `π_i(t)`
or using it in any threshold, and `06a_verify_propositions.py` checks the
reconstruction at both Δ = 1 and Δ = 0.5.

## Change log for the code

| file | change |
|---|---|
| `gentwin/config.py` | units → kW/kWh/Tk; `CONSUMER_TIERS` → `CONSUMER_ARCHETYPES` with sampled per-household preferences; `tau` → `p_str_kw`; `y_max` replaces `kappa_g`; `DELTA_MAX_SURCHARGE` removed; daily `c_{i,d}` |
| `gentwin/tariff.py` | Tk/kWh throughout, MWh conversion layer deleted |
| `gentwin/forecast.py` | `tau` → `P^str_i`; kW |
| `gentwin/population.py` | per-household λ, γ^sh, γ^cu, y^max sampled and stored; kW |
| `gentwin/load_model.py` | W→kW at Eq. (2); MW→kW throughout |
| `gentwin/validation.py` | kW |
| `02_consumer_catalogue.json` | `p_rat_kw`; archetype preference distributions |
| `docs/y_max_h_derivation.md` | → `docs/y_max_h_derivation.md`, rewritten |
| all docs | equation numbers, units |
