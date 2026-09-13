# NSGA-II (2002)

**Non-dominated Sorting Genetic Algorithm II** — Kalyanmoy Deb, Amrit Pratap, Sameer Agarwal, and
T. Meyarivan, *A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II*, IEEE Transactions on
Evolutionary Computation **6**(2):182–197, 2002. DOI:
[10.1109/4235.996017](https://doi.org/10.1109/4235.996017).

NSGA-II is the elitist multi-objective genetic algorithm that most later MOEAs build on — including
[AGE-MOEA](age-moea.md) and [AGE-MOEA-2](age-moea-2.md), both of which reuse its evolutionary skeleton
verbatim and swap out only the diversity mechanism. OptunaSharp ships NSGA-II as **`NSGAIISampler`**,
Optuna's default sampler for multi-objective studies, and the `OptunaSharp.Samplers.AgeMoea` family
reuses the same (μ+λ) loop and operators.

## The idea

NSGA-II fixed three problems with the original NSGA — an `O(MN³)` sorting cost, no elitism, and a
hand-tuned sharing parameter — and in doing so defined the template the field still uses (`M` = number
of objectives, `N` = population size).

**Fast non-dominated sort — `O(MN²)`.** For every solution `p`, compute a domination count `n_p` (how
many solutions dominate it) and the set `S_p` it dominates. The first front is `{p : n_p = 0}`. Then
"peel": for each `p` in the current front and each `q` in `S_p`, decrement `n_q`; when `n_q` reaches 0,
`q` joins the next front. Repeat until every solution is ranked.

**Crowding distance — a parameterless density estimate.** Within a front, for each objective sort the
solutions and give each the normalized gap between its two neighbors; the two boundary solutions get an
infinite distance so they are always kept. Sum the per-objective contributions. A larger crowding
distance means a more isolated (less crowded) solution.

**Crowded-comparison operator `≺ₙ`.** `i ≺ₙ j` when `i` has the lower (better) front rank, or — on a
rank tie — the larger crowding distance. This single order drives both parent selection and survival.

**Elitist main loop.** Combine parents and offspring into `Rₜ = Pₜ ∪ Qₜ` (size `2N`); non-dominated
sort `Rₜ`; fill the next population front by front; when the next whole front would overflow `N`, sort
that one splitting front by `≺ₙ` and take just enough of its least-crowded members. Offspring `Qₜ₊₁`
come from binary-tournament selection (on `≺ₙ`) with simulated binary crossover (SBX) and polynomial
mutation. Combining parents with offspring is what makes the algorithm elitist; overall cost is
`O(MN²)`, dominated by the sort.

## In OptunaSharp

`NSGAIISampler` reconstructs the population from completed trials and evolves it exactly as above,
across Optuna's full parameter space (float / int / categorical):

```csharp
using OptunaSharp;
using OptunaSharp.Samplers;
using OptunaSharp.Samplers.Nsgaii;

var study = Optuna.CreateStudy(
    directions: new[] { StudyDirection.Minimize, StudyDirection.Minimize },
    sampler: new NSGAIISampler(
        populationSize: 64,
        crossover: new SBXCrossover(),
        mutation: new PolynomialMutation(eta: 20.0),
        seed: 42));
```

Available crossover operators (in `OptunaSharp.Samplers.Nsgaii`): `UniformCrossover` (the default),
`BLXAlphaCrossover`, `SBXCrossover`, `VSBXCrossover`, `SPXCrossover`, `UNDXCrossover`; mutation:
`PolynomialMutation`.

**Parity.** The default path (`UniformCrossover` + no mutation object) is **bit-exact** against Optuna:
uniform crossover copies parent values verbatim, mutated-away parameters are resampled by the
independent `RandomSampler`, and every draw flows through a `LazyRandomState` on NumSharp's MT19937 —
never `System.Random`. The arithmetic crossovers (BLX-α/SBX/vSBX/SPX) stay bit-exact; `UNDXCrossover`
is tolerance-bounded (managed QR vs LAPACK) but keeps the RNG in lock-step. See the
[Samplers](../samplers.md) and [Multi-objective Optimization](../multi-objective.md) guides for the
full API and documented divergences. (Conditional / branch-defined parameters are handled by the
separate [`NSGAIIICptSampler`](nsga-iii.md), NSGA-III-CPT; this sampler is a bit-exact Optuna port.)

## What the successors keep and change

Both later algorithms in this section reuse the NSGA-II skeleton unchanged — `Rₜ = Pₜ ∪ Qₜ`,
non-dominated sort, front-by-front filling with a secondary score on the splitting front,
tournament + SBX + polynomial mutation — and replace only the **secondary score**:

- [AGE-MOEA (2019)](age-moea.md) swaps NSGA-II's crowding distance for a *survival score* that adapts
  to the estimated geometry (curvature) of the Pareto front.
- [AGE-MOEA-2 (2022)](age-moea-2.md) keeps AGE-MOEA's framework and refines two kernels inside it.

## See also

[AGE-MOEA (2019)](age-moea.md) · [AGE-MOEA-2 (2022)](age-moea-2.md) ·
[Samplers](../samplers.md) · [Multi-objective Optimization](../multi-objective.md) ·
the [API reference](../../api/index.md).
