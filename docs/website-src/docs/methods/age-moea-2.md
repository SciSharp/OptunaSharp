# AGE-MOEA-2 (2022)

**An Improved Pareto Front Modeling Algorithm for Large-scale Many-Objective Optimization** — Annibale
Panichella, GECCO 2022. DOI: [10.1145/3512290.3528732](https://doi.org/10.1145/3512290.3528732).

AGE-MOEA-2 is the newest member of the lineage [NSGA-II](nsga-ii.md) → [AGE-MOEA](age-moea.md) →
AGE-MOEA-2. It keeps [AGE-MOEA](age-moea.md)'s entire framework — which in turn keeps
[NSGA-II](nsga-ii.md)'s evolutionary skeleton — and **refines exactly two kernels** inside the survival
step: how the front's curvature is estimated, and how distances between solutions are measured.

## The idea

AGE-MOEA already models the first non-dominated front as the unit manifold of the `Lₚ` norm
`(a_1^p + … + a_M^p)^(1/p) = 1`, where `p` is the curvature (`p = 1` flat, `p = 2` spherical/concave,
`p < 1` hyperbolic/convex). AGE-MOEA-2 improves the two places that manifold is used.

### 1. Curvature `p` by Newton–Raphson (was: a single-point heuristic)

AGE-MOEA approximated `p` from one representative "central" point, which makes the estimate sensitive
to *which* point is chosen. AGE-MOEA-2 instead solves `Σᵢ aᵢ^p = 1` directly with the Newton–Raphson
root-finder on `f(p) = log(Σᵢ aᵢ^p)` (the ideal point sits at the origin after normalization):

```
p_{n+1} = p_n + log(Σᵢ aᵢ^{p_n}) / (Σᵢ aᵢ^{p_n} · log aᵢ)      seed p_0 = 1
```

It converges in about three iterations (error `< 1e-4`, at most 4–5 iterations) and is **independent of
the reference point**. In the paper's worked example — three points on a `p = 2` circle — Newton–Raphson
returns `p ≈ 2.0000` for *every* point, whereas the old heuristic returns `1.628` for an off-center one.
Cost is `O(M·G)` for `G` iterations.

### 2. Diversity by geodesic distance (was: the raw `Lₚ` distance)

On a curved front, the straight-line (Minkowski `Lₚ`) distance between two solutions understates how
far apart they really are along the surface. AGE-MOEA-2 measures a **geodesic** distance — the length of
a path along the manifold — approximated with two straight segments through the projected midpoint.
First every point is projected onto the unit manifold:

```
A⊥ = A / (Σᵢ aᵢ^p)^(1/p)
```

then, for the projected midpoint `C⊥` of two points `A` and `B`:

```
gd(A, B) = ‖A − C⊥‖₂ + ‖C⊥ − B‖₂
```

This geodesic distance replaces the `Lₚ` distance in AGE-MOEA's greedy diversity loop; everything else
about the survival score (proximity, the maximin selection, extremes pre-seeded with `+∞`, the
`1/proximity` score on later fronts) is inherited unchanged. Cost is `O(M·N²)`.

Empirically the paper reports AGE-MOEA-2 achieving higher hypervolume than AGE-MOEA, NSGA-III, MOEA/D,
VaEA, and LMEA across 17 WFG and SMOP problems at `M ∈ {3, 6, 9}` objectives with `D = 100` decision
variables.

## In OptunaSharp

AGE-MOEA-2 is **absent from Optuna core** (which stops at NSGA-II / NSGA-III / MO-TPE), so — like
LM-CMA and JANUS — it is a *beyond-core* sampler validated against its reference implementation (pymoo
0.6.1.3), not against Optuna. It is implemented in two layers:

**Layer 2 — `AgeMoea2Sampler` (the public, Optuna-idiomatic face).** An ask/tell sampler that
reconstructs the population from completed trials exactly like `NSGAIISampler`, but performs AGE-MOEA-2's
geometry-based environmental selection. It works across Optuna's full parameter space (float / int /
categorical, and conditional parameters):

```csharp
using OptunaSharp;
using OptunaSharp.Samplers;

var study = Optuna.CreateStudy(
    directions: new[] { StudyDirection.Minimize, StudyDirection.Minimize },
    sampler: new AgeMoea2Sampler(populationSize: 50, seed: 42));

study.Optimize((Func<Trial, IReadOnlyList<double>>)(t =>
{
    double x = t.SuggestFloat("x", 0, 1);
    double y = t.SuggestFloat("y", 0, 1);
    return new[] { x, 1 - x + y };
}), nTrials: 500);
```

Children use SBX (`eta = 15`) and polynomial mutation (`eta = 20`) over Optuna's transformed search
space; `populationSize`, `crossover`, `mutation`, `crossoverProb`, `swappingProb`, `seed`, and
`conditionalParams` are configurable. Because it uses Optuna's operators (a different RNG stream than
pymoo's), Layer 2 is validated **behaviorally**, not bit-for-bit. Multi-objective, unconstrained.

**Layer 1 — `AgeMoea2Optimizer` (the byte-perfect reference driver, internal).** A standalone
`minimize()`-shaped loop with pymoo-faithful operators that is **generation-by-generation bit-exact
against `pymoo.minimize`**. This is where the "byte-perfect" guarantee lives; it is exercised offline
against a committed corpus and, every CI run, by a live pythonnet parity gate that drives the real pymoo
in-process (`AgeMoea2ParityTests`). Supplying the AGE-MOEA base survival to this optimizer recovers
[AGE-MOEA-1](age-moea.md) for free.

## Relationship to the lineage

- **From [AGE-MOEA (2019)](age-moea.md):** inherits normalization, corner/extreme solutions, the greedy
  survival loop, environmental selection, and the operators. Overrides only `compute_geometry`
  (→ Newton–Raphson) and `pairwise_distances` (→ geodesic).
- **From [NSGA-II (2002)](nsga-ii.md):** inherits (transitively) the whole elitist (μ+λ) skeleton —
  combine parents and offspring, non-dominated sort, fill fronts, split the last one by the survival
  score, reproduce by tournament + SBX + polynomial mutation.

## See also

[AGE-MOEA (2019)](age-moea.md) · [NSGA-II (2002)](nsga-ii.md) ·
[Samplers](../samplers.md) · [Multi-objective Optimization](../multi-objective.md) ·
the [API reference](../../api/index.md).
