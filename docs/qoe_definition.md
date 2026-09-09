# C7: defining QoE, and what it does to Proposition 2

## 1. The gap

Constraint C7, Eq. (35), reads

$$QoE_{h,m}(q) \ge QoE^{min} \qquad \forall h, m$$

and the formulation never defines `QoE_{h,m}`. It appears in the notation
table only implicitly and in no other equation. An implementation cannot
proceed without a definition, so one is supplied here and flagged as an
addition rather than a transcription.

## 2. The definition adopted

Using the response energies already defined in Eq. (19),

$$E^{sh}_{h,t} = x_{h,t}L^{sh}_h(t)\Delta, \qquad E^{cu}_{h,t} = y_{h,t}L^{cu}_h(t)\Delta$$

define, over billing period `T_m`,

$$QoE_{h,m}(q) = 1 - \frac{w^{sh}\sum_{t \in T_m} E^{sh}_{h,t} + w^{cu}\sum_{t \in T_m} E^{cu}_{h,t}}{E^{base}_{h,m}}, \qquad E^{base}_{h,m} = \sum_{t \in T_m} L_h(t;0,0)\Delta$$

with `w^sh = 0.4`, `w^cu = 1.0`, `QoE^min = 0.85` (config).

In words: quality of experience is one minus the discomfort-weighted share of
a consumer's own baseline energy that the optimiser is allowed to displace.
`w^sh < w^cu` because shifting a wash to midnight is less keenly felt than
losing the air conditioner — the same ordering the formulation already asserts
through `gamma^sh_h < gamma^cu_h`.

`E^base_{h,m}` uses `L_h(t;0,0)`, the pre-response load of Eq. (6), which is a
**constant** with respect to `q`.

## 3. Why this definition and not another

Three requirements had to be met simultaneously:

1. **Affine in `q`.** `E^sh` and `E^cu` are linear in `x` and `y`;
   `E^base_{h,m}` is a constant. So C7 rearranges to

   $$w^{sh}\sum_{t} L^{sh}_h(t)\Delta \; x_{h,t} \;+\; w^{cu}\sum_{t} L^{cu}_h(t)\Delta \; y_{h,t} \;\le\; (1 - QoE^{min})\,E^{base}_{h,m}$$

   a single affine inequality per `(h, m)`. `F` remains a polyhedron and (25)
   remains a linear programme, which every structural result depends on.

2. **An inequality, not an equality.** Section 3.3 states that after substituting the
   state of charge, *"every element of `F` is an inequality in the decision
   variables"*, and Remark 2 makes the absence of equality constraints a claim
   of the formulation. A QoE definition requiring an equality would break that
   claim.

3. **Per-consumer and per-billing-period**, matching the `(h, m)` indexing of
   Eq. (35) exactly.

## 4. The consequence for Proposition 2 — read this part

Proposition 2's proof establishes the threshold by computing the reduced cost
of `y_{h,t}` on the premise that *"`F` is affine and C6 is a box"*. It
accounts for exactly two contributions: the objective term from `f_wel`, and
the relaxation of **C1** worth `kappa_{i,d}(t)L^{cu}_h(t)\pi_i(t)`.

C7 is **not a box**. It couples `x_{h,t}` and `y_{h,t}` across all `t` in the
billing period. So if C7 is *active* at the optimum, its dual `chi_{h,m} >= 0`
enters the reduced cost of `y_{h,t}` and the threshold acquires a fourth term:

$$\lambda_g\left(\pi^{ret}_h\mu(t) + \rho s\right) + \frac{\kappa_{i,d}(t)}{\theta_3}\pi_i(t) \; - \; \underbrace{\frac{w^{cu}\chi_{h,m}}{\theta_3}}_{\text{QoE scarcity}} \; > \; (1-\lambda_g)\gamma^{cu}_g$$

**This is a property of the formulation, not of the definition above.** Any
non-box C7 does the same thing; Proposition 2 as stated tacitly assumes C7 is
slack. The implementation handles it explicitly rather than quietly:

- `QoE^min` is set loose enough (0.85) that C7 is expected to be slack at the
  optimum, so the published three-term threshold is exact.
- `06a_verify_propositions.py` **checks C7 slack at every `(h, m)`** and
  reports the count of binding rows. The published threshold is only claimed
  where the row is slack.
- Where C7 *does* bind, the dual `chi_{h,m}` is captured and stored on the
  knowledge-graph decision node as `qoe_term`, and the explanation shows four
  terms instead of three. Nothing is hidden and no consumer's action is
  reported with a threshold that does not reconcile.

This is the same discipline the formulation itself applies to degeneracy in
§8.2 — the exceptional case is characterised and reported rather than
suppressed.

## 5. What to write in the thesis

> Constraint C7 is stated in the formulation but not defined. We adopt an
> affine definition of QoE as one minus the discomfort-weighted share of
> displaced baseline energy, which preserves linearity, keeps the feasible set
> free of equalities, and matches the `(h, m)` indexing of Eq. (35). Because
> C7 is not a box constraint, an active C7 would add its dual to the
> Proposition 2 threshold; we therefore verify C7 slackness at every optimum
> and report the three-term threshold only where it holds.
