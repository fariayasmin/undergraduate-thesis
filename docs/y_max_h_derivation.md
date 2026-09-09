# The `y^max_h` extension to C6

This is the **only** structural modification made to the constraint set of
GenTwinSG_Formulation_v3. It is derived here in full, as required.

## 0. A notation warning that has to come first

The formulation uses the letter kappa for **two unrelated quantities**:

| Symbol | Meaning | Where |
|---|---|---|
| `y^max` | curtailment cap, a scalar bound on `y_{h,t}` | notation table; C6, Eq. (32) |
| `kappa_{i,d+k\|d}` | robust scaling factor `p~ / p`, Eq. (16) | C1 (27), C4 (30), Props. 2-3, Eq. (38) |

This is the same class of collision the formulation already flags in Remark 1
for `nu` (reserve shortfall) and `varsigma` (surcharge). It matters here
because the extension below touches the **cap** and leaves the **scaling
factor** completely alone, and Eq. (38) contains the scaling factor, not the
cap. To keep the two apart, the implementation names them:

```python
tier["kappa_g"]      # the cap,            C6 / Eq. (32)
forecast["kappa"]    # the scaling factor, Eq. (16)
```

In the write-up the cap is suggested as `y^max_h`, by analogy with
Remark 1's `varpi_{i,m}`.

## 1. The change

Equation (36), as written:

$$0 \le x_{h,t} \le \mathbb{1}\!\left[t \in T^{pk}_{d(t)}\right], \qquad 0 \le y_{h,t} \le y^{max}$$

Equation (34'), as implemented:

$$0 \le x_{h,t} \le \mathbb{1}\!\left[t \in T^{pk}_{d(t)}\right], \qquad 0 \le y_{h,t} \le y^{max}_h$$

A single global cap becomes one cap per household. Nothing else changes:
the shifting bound, the indicator, and every other constraint are untouched.

## 2. Why it is needed, and why it is now a smaller step

The brief requires six consumer categories with *different flexibility limits*.
Under a single global `y^max` a hospital and a shop may curtail the same
maximum fraction of their curtailable load, which contradicts that directly.

The revision makes this extension **smaller than it was**. Section 1.3 already
carries `lambda_h`, `gamma^sh_h` and `gamma^cu_h` per household, and Remark 4
states that "household heterogeneity is individual". `y^max` is now the only
preference-like quantity the formulation does *not* index by `h`. Indexing it
is completing a pattern the document has already established, not opening a
new axis.

## 3. Linearity is preserved

`F` is a polyhedron because every element of the feasible set is an affine
inequality in `q` (§3.3). Replacing the constant `y^max` in the row
`y_{h,t} <= kappa` with the constant `y^max_h` changes the
right-hand side of that row and nothing else. The row remains

$$y_{h,t} - y^{max}_h \le 0,$$

affine in `q` with coefficient 1 on a single variable. `F` is therefore still
a polyhedron and (25) is still a linear programme. **A box with per-variable
bounds is still a box.**

## 4. Proposition 1 is unaffected

The proof of Proposition 1 perturbs `Z^c_i`, `G_i` and `T_{ji}` only. It
argues that if C1 has slack at some `(i,t)`, one of those supply terms can be
reduced, preserving feasibility and strictly lowering `F`. No step of the
argument reads, uses, or perturbs `y_{h,t}`, so no bound on `y` enters it.
Supply adequacy is still active at every optimum.

## 5. Proposition 2 holds, with the threshold *unchanged*

This is the substantive check, and it is worth being precise about why it
passes.

The proof of Proposition 2 runs: *"`F` is affine and C6 is a box, so the
reduced cost of `y_{h,t}` is the derivative of the Lagrangian. Per unit of
`y_{h,t}`, `f_wel` changes by ... while C1 is relaxed by
`kappa_{i,d}(t) L^{cu}_h(t)`, worth `kappa_{i,d}(t) L^{cu}_h(t) pi_i(t)`.
**The variable takes its upper bound exactly when the sum is negative;**
dividing by `theta_3 L^{cu}_h(t) Delta > 0` gives (37)."*

Two observations:

1. The proof requires only that C6 **is a box** — that `y_{h,t}` is bounded
   above by a constant independent of every other decision variable. It never
   uses the *value* of that constant. `y^max_h` is such a constant.

2. The reduced cost of `y_{h,t}` is a derivative of the Lagrangian with
   respect to `y_{h,t}`. The bound's value appears in the Lagrangian only
   multiplied by the bound's own multiplier, and contributes nothing to
   `partial L / partial y_{h,t}`.

Hence the **threshold in (37) is literally unchanged**:

$$y^\star_{h,t} = y^{max}_h \iff \underbrace{\lambda_h\left(\pi^{ret}_h \mu(t) + \rho s_{i,d(t)}\right)}_{\text{private}} + \underbrace{\frac{\kappa_{i,d}(t)}{\theta_3}\pi_i(t)}_{\text{system}} > \underbrace{(1-\lambda_h)\gamma^{cu}_h}_{\text{discomfort}}$$

Only the **value the variable takes at its bound** changes, from `y^max` to
`y^max_h`. The comparison of two scalars in the same unit — the
property the explainability layer depends on — survives intact.

The same argument covers `x_{h,t}` in (36), whose bound is not modified at all.

## 6. Equation (36) is unaffected

The critical tier preference

$$\lambda^\star_g = \frac{\gamma^{cu}_h - \kappa_{i,d}(t)\pi_i(t)/\theta_3}{\pi^{ret}_h \mu^{pk} + \rho + \gamma^{cu}_h}$$

is obtained by solving (37) at equality for `lambda_h`. Since (37) is
unchanged and the `y^max` appearing in (36) is the **scaling factor** of
Eq. (16), not the cap, (36) is unchanged. The section 8 arithmetic
`(20 - 1.3360 x 6.0000)/(12 + 3 + 20) = 0.3424` involves no cap at all.

## 7. Proposition 3 is unaffected

The merit-order condition (37) prices only supply resources. Demand response
is explicitly excluded from it — *"Demand response does not appear in (37),
because it is not a priced supply resource"* — and its position is set
endogenously through Proposition 2, which §5 shows is unchanged. The exchange
argument perturbs supply resources only.

## 8. Hospitals: a derived result, not an exclusion

`y^max_h ~ 0.05` for hospitals is a *cap*, not a prohibition. Hospitals are kept
out of curtailment by the **threshold**, not the bound: with
`lambda_h ~ 0.02` and `gamma^cu_h ~ 5000` BDT/kWh, (37) reads

```
LHS  =  0.02 x (9.05 x 1.20 + 3.00)  +  1.34 x 20.0        ~  27  BDT/kWh
RHS  =  (1 - 0.02) x 5000                                  = 4900  BDT/kWh
```

so `y*_{h,t} = 0` at any shadow price the network can plausibly produce. Non-
curtailment of hospital load is therefore an **output of the optimisation**,
explainable slot by slot as a comparison of two scalars, rather than a rule
written into the model. This is what the brief asked for, and it is a
strictly stronger claim than hard-coding the exclusion.

## 9. Summary

| Claim | Status under `kappa -> y^max_h` |
|---|---|
| (25) is a linear programme | holds; box bound with per-variable RHS |
| Prop. 1, supply adequacy active | holds; proof never touches `y` |
| Prop. 2, threshold form of (36) | holds, threshold identical |
| Prop. 2, threshold form of (37) | holds, threshold identical; only the attained value changes |
| Eq. (38), critical preference | holds, unchanged |
| Prop. 3, merit order | holds; DR is not a priced supply resource |
| Corollary 1, acyclicity | holds; no new inter-period coupling |


## 10. What changed when the formulation was revised

| | earlier draft | revised |
|---|---|---|
| symbol for the cap | `kappa` (collided with the scaling factor) | `y^max` |
| preference indexing | per tier `g in G` | per household `h` |
| this extension | `kappa -> kappa^cap_g` | `y^max -> y^max_h` |
| size of the step | new per-tier axis | completes an existing per-household pattern |

The substance of the derivation is unchanged: the bound is still a constant
independent of every other decision variable, so C6 is still a box, the
programme is still linear, and the Proposition 2 thresholds are still
identical - only the value attained at the bound differs.
