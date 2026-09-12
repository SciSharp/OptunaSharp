# AGE-MOEA (2019)

**Adaptive GEometry Estimation based Multi-Objective Evolutionary Algorithm** — Annibale Panichella,
*An Adaptive Evolutionary Algorithm based on Non-Euclidean Geometry for Many-objective Optimization*,
GECCO 2019. DOI: [10.1145/3321707.3321839](https://doi.org/10.1145/3321707.3321839).

AGE-MOEA is the direct predecessor of [AGE-MOEA-2](age-moea-2.md) and the middle link in the lineage
[NSGA-II](nsga-ii.md) → AGE-MOEA → AGE-MOEA-2. Its one big idea: keep NSGA-II's entire evolutionary
skeleton, but **replace the crowding distance with a survival score that adapts to the *shape* of the
Pareto front**.

## The idea

NSGA-II's crowding distance implicitly measures spread with rectilinear (Euclidean) geometry, which
suits a flat front but misjudges convex or concave ones. Reference-point methods such as NSGA-III place
their reference points on a flat hyperplane (Das–Dennis), so they too assume a flat front. AGE-MOEA
instead **estimates the front's curvature every generation** and measures both convergence and
diversity in the matching geometry.

**The `Lₚ` manifold.** AGE-MOEA models the first non-dominated front as the unit manifold of the
`Lₚ` norm:

```
(a_1^p + a_2^p + … + a_M^p)^(1/p) = 1
```

The exponent `p` *is* the curvature: `p = 1` is a flat hyperplane, `p = 2` a hypersphere (concave), and
`p < 1` a hyperbolic (convex) surface. Fitting the front means finding the `p` whose unit surface best
matches it.

**Per-generation recipe** (all inside the NSGA-II loop, on the first front `F₁`):

1. **Normalize** `F₁` NSGA-III-style — shift the ideal point `Z^min` to the origin, then divide by the
   intercepts of the `M`-dimensional hyperplane through the front's extreme points (falling back to
   per-axis min–max if that system is singular).
2. **Estimate `p`** cheaply (`O(MN)`) from a single representative point — the front's *central point*
   `C` (the one nearest the diagonal bisector) — via the closed form
   `p = log(M) / (log(M) − log(Σ Cᵢ))`.
3. **Score = diversity / proximity**, both in the `Lₚ` norm with that `p`:
   - *proximity* `= ‖f(S)‖ₚ` — distance to the ideal point (convergence);
   - *diversity* — a greedy maximin: pre-seed the extreme solutions with a score of `+∞`, then
     repeatedly add the point whose summed distance to its two nearest already-selected neighbors is
     largest, recording that sum as its score.
   - Non-first fronts get `score = 1 / proximity` (proximity only).

Environmental selection then fills front by front and splits the last front by descending survival
score — the NSGA-II structure, with this score in place of crowding distance.

## In OptunaSharp

AGE-MOEA is validated internally as the **base** of the AGE-MOEA-2 port: `AgeMoea2Survival` derives
from `AgeMoeaSurvival` (the 2019 survival, with `FindCornerSolutions`, normalization, and the greedy
`SurvivalScore`), and the byte-exact Layer-1 driver `AgeMoea2Optimizer` accepts either survival
strategy. Passing the 2019 base recovers **AGE-MOEA-1 for free** — the reference implementation runs
it by supplying the default geometry/distance kernels to the same code:

```csharp
// (internal Layer-1 optimizer, used by the parity harness)
var optimizer = new AgeMoea2Optimizer(problem, popSize, nGen, seed, survival: new AgeMoeaSurvival());
```

AGE-MOEA has no public Optuna-idiomatic sampler of its own; the public, ask/tell entry point is
[`AgeMoea2Sampler`](age-moea-2.md#in-optunasharp) (the improved 2022 variant). Like AGE-MOEA-2, AGE-MOEA
is **absent from Optuna core**, so it lives in OptunaSharp as a beyond-core algorithm validated against
its reference implementation (pymoo), not against Optuna.

## Relationship to the lineage

- **From [NSGA-II](nsga-ii.md):** inherits the whole (μ+λ) skeleton — `Rₜ = Pₜ ∪ Qₜ`, fast
  non-dominated sort, front-by-front filling, binary tournament + SBX + polynomial mutation. Only the
  crowding distance is replaced.
- **To [AGE-MOEA-2](age-moea-2.md):** the 2022 version keeps everything here and overrides exactly two
  kernels — the curvature estimate (a single-point heuristic → Newton–Raphson) and the pairwise
  distance (raw `Lₚ` → geodesic).

## See also

[AGE-MOEA-2 (2022)](age-moea-2.md) · [NSGA-II (2002)](nsga-ii.md) ·
[Samplers](../samplers.md) · [Multi-objective Optimization](../multi-objective.md) ·
the [API reference](../../api/index.md).
