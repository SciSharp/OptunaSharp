# NSGA-III (2014)

**Non-dominated Sorting Genetic Algorithm III** — Kalyanmoy Deb and Himanshu Jain, *An Evolutionary
Many-Objective Optimization Algorithm Using Reference-Point-Based Non-dominated Sorting Approach,
Part I: Solving Problems With Box Constraints*, IEEE Transactions on Evolutionary Computation
**18**(4):577–601, 2014. DOI:
[10.1109/TEVC.2013.2281535](https://doi.org/10.1109/TEVC.2013.2281535).

NSGA-III is [NSGA-II](nsga-ii.md)'s **many-objective sibling**: it keeps the exact same elitist
evolutionary skeleton and replaces only the diversity mechanism — crowding distance gives way to
**reference-point niching**. It is a sibling of [AGE-MOEA](age-moea.md) (which instead replaces crowding
with a geometry-adaptive survival score), and AGE-MOEA in fact borrows NSGA-III's adaptive
normalization. OptunaSharp ships it as **`NSGAIIISampler`**, a first-class port of Optuna's own
sampler.

## The idea

As the number of objectives `M` grows past three, almost every solution in a random population becomes
non-dominated, so NSGA-II's crowding distance can no longer tell good spread from bad. NSGA-III keeps
NSGA-II's `Rₜ = Pₜ ∪ Qₜ` combine, the fast non-dominated sort, and the fill-fronts-then-split-the-last
structure, but decides *who survives from the splitting front* by spreading solutions across a set of
**structured reference points** rather than by local crowding.

**Reference points on a hyperplane.** A set of reference points is placed on the normalized
`(M−1)`-dimensional unit simplex with Das and Dennis's systematic approach: with `p` divisions per
objective, `H = C(M + p − 1, p)` points (e.g. `M = 3, p = 4` → 15 points). For high `M` two layers
(a boundary layer and an inside layer with small `p`) keep `H` manageable. NSGA-III sets the population
size `N ≈ H`, so it is effectively parameter-free beyond the usual GA parameters.

**Adaptive normalization** (recomputed every generation, so it copes with differently-scaled
objectives): find the ideal point `z^min` (per-objective minima over the selected set) and translate to
it; find each axis's *extreme* point by minimizing the achievement scalarizing function
`ASF(x, w) = maxᵢ f'ᵢ(x) / wᵢ`; solve the `M`-point hyperplane through those extremes for its axis
intercepts `aᵢ`; normalize `f^n_i = f'_i / (aᵢ − z_i^min)` so the front lies on `Σᵢ f^n_i = 1`.

**Association.** Draw a *reference line* from the origin through each reference point; associate every
solution with the reference point whose line it is closest to (smallest perpendicular distance in the
normalized objective space).

**Niche-preservation** (the selection that fills the last `K` slots). Track a *niche count* `ρⱼ` — how
many already-selected solutions are associated with reference point `j`. Repeat `K` times: pick a
reference point with the smallest niche count (ties broken at random); if the splitting front has
solutions associated with it, add one — the closest by perpendicular distance if the point was empty
(`ρⱼ = 0`), otherwise a random associate — and increment `ρⱼ`; if the front has none, drop that
reference point for this generation. This deterministically favors sparsely-populated reference
directions, preserving spread across the whole front.

## In OptunaSharp

`NSGAIIISampler` reuses all of NSGA-II's genetic machinery and swaps in reference-point niching for
elite selection. It suits many-objective studies:

```csharp
using OptunaSharp;
using OptunaSharp.Samplers;

var study = Optuna.CreateStudy(
    directions: new[]
    {
        StudyDirection.Minimize, StudyDirection.Minimize,
        StudyDirection.Minimize, StudyDirection.Minimize,
    },
    sampler: new NSGAIIISampler(
        populationSize: 92,
        dividingParameter: 3,   // p — density of the default Das–Dennis reference points
        seed: 42));
```

Supply your own `referencePoints` (a `double[,]` with `n_objectives` columns) to bias the search toward
preferred regions, or leave it `null` to generate uniformly-spread points from `dividingParameter`. The
crossover / mutation operators, `crossoverProb`, and `swappingProb` are shared with
[`NSGAIISampler`](nsga-ii.md). To optimize conditional / branch-defined parameters, use
**`NSGAIIICptSampler`** (NSGA-III-CPT) — a standalone copy of this sampler that takes a `ConditionalParams`
rule set; this sampler itself is a bit-exact Optuna port with no such extension (see the
[Samplers](../samplers.md) guide).

**Parity.** The default path stays in exact MT19937 lock-step with Optuna — generation attribution,
parent cache, the tournament / crossover / mutation draws, and the niching `rng.shuffle` — and the
objective-value niching is bit-exact: the hyperplane-fit `matrix_rank` / `solve` route through
NumSharp's OpenBLAS LAPACK backend (byte-identical to NumPy), confirmed byte-for-byte over a 400-config
differential plus the live parity seeds. The one documented divergence is near-tie reference-point
selection, which can pick a different equally-close reference point.

## Relationship to the lineage

- **From [NSGA-II (2002)](nsga-ii.md):** inherits the whole elitist (μ+λ) skeleton unchanged; only
  crowding distance → reference-point niching. NSGA-III uses a large SBX distribution index and, in the
  unconstrained form, picks parents at random rather than by tournament.
- **Beside [AGE-MOEA (2019)](age-moea.md) / [AGE-MOEA-2 (2022)](age-moea-2.md):** all three are
  NSGA-II with a many-objective diversity mechanism grafted on — NSGA-III uses structured reference
  points, the AGE family uses the front's estimated `Lₚ` geometry. AGE-MOEA reuses **NSGA-III's adaptive
  normalization** as the first step of its survival score.

## See also

[NSGA-II (2002)](nsga-ii.md) · [AGE-MOEA (2019)](age-moea.md) · [AGE-MOEA-2 (2022)](age-moea-2.md) ·
[Samplers](../samplers.md) · [Multi-objective Optimization](../multi-objective.md) ·
the [API reference](../../api/index.md).
