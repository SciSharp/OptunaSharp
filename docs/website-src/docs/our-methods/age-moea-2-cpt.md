# AGE-MOEA-2-CPT — Categorical Parameter Trees

**AGE-MOEA-2 with Categorical Parameter Trees** — an **OptunaSharp original** with no upstream Optuna
analog. It joins [AGE-MOEA-2 (2022)](../methods/age-moea-2.md)'s geometry-based environmental selection to
the same conditional-parameter machinery [NSGA-III-CPT](nsga-iii-cpt.md) introduced, and ships as
**`AgeMoea2CptSampler`** (`OptunaSharp.Samplers`). It is the AGE-MOEA-2 sibling of
[NSGA-III-CPT](nsga-iii-cpt.md).

## The problem it solves

This is the same gap [NSGA-III-CPT](nsga-iii-cpt.md) closes, one host up the lineage. Optuna's
define-by-run API lets an objective suggest a parameter only under a branch:

```csharp
string optimizer = t.SuggestCategorical("optimizer", new[] { "adam", "sgd" });
if (optimizer == "adam") t.SuggestFloat("beta1", 0.5, 0.999);   // only exists under adam
else                     t.SuggestFloat("momentum", 0.0, 0.99); // only exists under sgd
```

but the GA samplers evolve the *intersection* search space, so a branch-defined knob leaves that
intersection after the first trial that lacks it and is thereafter resampled uniformly at random — no
crossover, no mutation, no inheritance. AGE-MOEA-2-CPT keeps such declared parameters in the evolved space
whenever they are *active*, exactly as [NSGA-III-CPT](nsga-iii-cpt.md) does.

## The idea

CPT (Categorical Parameter Trees) is **host-agnostic**: it is entirely a *child-generation / search-space*
concern — which parameters stay in the relative space, and how they are crossed, mutated and inherited
when only some parents carry them — and it knows nothing about *survival*. The host is only the
elite-selection strategy. So AGE-MOEA-2-CPT is literally [NSGA-III-CPT](nsga-iii-cpt.md) with the survival
rule and operator defaults swapped:

- **Survival** → AGE-MOEA-2's `AgeMoea2ElitePopulationSelectionStrategy` (adaptive front-geometry
  estimation) instead of NSGA-III's reference-point niching.
- **Conditional machinery** → the *same* reused seams (`NsgaiiiCptSearchSpace`,
  `NsgaiiiCptChildGenerationStrategy`, `ConditionalParams`), not a re-implementation — so the activation
  algebra, math-neutrality property, and opt-in dormant-memory mode are identical to
  [NSGA-III-CPT](nsga-iii-cpt.md).
- **Operators** → AGE-MOEA-2's own defaults (polynomial mutation `eta = 20`, SBX `eta = 15`), so
  `conditionalParams: null` is bit-identical to [`AgeMoea2Sampler`](../methods/age-moea-2.md).

## Why reach for this over NSGA-III-CPT

Both handle conditional parameters identically; they differ only in **survival**. NSGA-III niches against a
fixed reference-point simplex whose size grows as `C(M+2, 3)`, so past roughly five objectives most of the
population is mutually non-dominated and the front stops selecting; AGE-MOEA-2 estimates the front's
geometry adaptively and keeps selection pressure in that regime. So AGE-MOEA-2-CPT is the one to use when
the objective count is (or may grow) high; at 2–4 objectives the two are close and
[NSGA-III-CPT](nsga-iii-cpt.md) remains the conservative default. **This is a genuine tradeoff, not a
strict improvement** — validate the swap on the actual objective set before adopting it.

## In OptunaSharp

```csharp
using OptunaSharp;
using OptunaSharp.Samplers;
using OptunaSharp.Samplers.NsgaiiiCpt;

var rules = new ConditionalParams()
    .When("optimizer", "adam", "adamw").Activates("beta1", "beta2")
    .When("optimizer", "sgd").Activates("momentum")
    .When("n_layers", v => (long)v! >= 3).Activates("width_3");

var study = Optuna.CreateStudy(
    directions: new[] { StudyDirection.Minimize, StudyDirection.Minimize, StudyDirection.Minimize },
    sampler: new AgeMoea2CptSampler(populationSize: 32, seed: 0, conditionalParams: rules));
```

Constructed with `conditionalParams: null`, `AgeMoea2CptSampler` is behaviorally identical to
[`AgeMoea2Sampler`](../methods/age-moea-2.md) (the CPT search space and child generation fall through to
the shared fold, and the elite strategy, operators and RNG draw order match), so it is a strict superset.

**Unconstrained multi-objective only.** AGE-MOEA-2's survival does not model constraints, so — unlike
[`NSGAIIICptSampler`](nsga-iii-cpt.md) — this class exposes no `constraintsFunc`. If you need constrained
conditional many-objective search, use [NSGA-III-CPT](nsga-iii-cpt.md).

## Parity & isolation

The conditional feature is the sole responsibility of the `OptunaSharp.Samplers.NsgaiiiCpt` family; the
ported samplers ([`NSGAIISampler`](../methods/nsga-ii.md), [`NSGAIIISampler`](../methods/nsga-iii.md),
[`AgeMoea2Sampler`](../methods/age-moea-2.md), [`RveaSampler`](../methods/rvea.md)) stay bit-exact and do
**not** accept `conditionalParams`. AGE-MOEA-2-CPT lives *beside* [`AgeMoea2Sampler`](../methods/age-moea-2.md)
exactly as [`NSGAIIICptSampler`](nsga-iii-cpt.md) lives beside [`NSGAIIISampler`](../methods/nsga-iii.md), so
the port stays pristine.

## Relationship to the lineage

- **From [AGE-MOEA-2 (2022)](../methods/age-moea-2.md):** the entire many-objective genetic core —
  adaptive geometry estimation (Newton–Raphson curvature + geodesic distance) survival, normalization, and
  the operators — is inherited unchanged. Only the parameter-set shaping is conditional-aware.
- **From [NSGA-III-CPT](nsga-iii-cpt.md):** the identical CPT machinery (the same classes), including the
  math-neutrality integration and the opt-in dormant-memory mode.

## See also

[NSGA-III-CPT](nsga-iii-cpt.md) · [AGE-MOEA-2 (2022)](../methods/age-moea-2.md) ·
[JANUS-AGE-MOEA-2-CMIX-CPT](janus-age-moea-2-cmix-cpt.md) · [Samplers](../samplers.md) ·
[Multi-objective Optimization](../multi-objective.md) · the [API reference](../../api/index.md).
