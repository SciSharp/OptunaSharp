# NSGA-III-CPT — Categorical Parameter Trees

**NSGA-III with Categorical Parameter Trees** — an **OptunaSharp original** with no upstream Optuna
analog. It extends [NSGA-III (2014)](../methods/nsga-iii.md) to optimize *conditional* (define-by-run,
branch-defined) hyperparameters, and ships as **`NSGAIIICptSampler`** (`OptunaSharp.Samplers`).

## The problem it solves

Optuna's define-by-run API lets an objective suggest a parameter only under a branch:

```csharp
string optimizer = t.SuggestCategorical("optimizer", new[] { "adam", "sgd" });
if (optimizer == "adam") t.SuggestFloat("beta1", 0.5, 0.999);   // only exists under adam
else                     t.SuggestFloat("momentum", 0.0, 0.99); // only exists under sgd
```

But the GA samplers evolve the *intersection* search space — the parameters present in **every**
completed trial. `beta1` and `momentum` each leave that intersection after the first trial that lacks
them, and are thereafter resampled uniformly at random on every trial: no crossover, no mutation, no
inheritance — no evolutionary pressure at all, precisely on the knobs that make a branch choice
meaningful. (Optuna's own answer for the *TPE* sampler is `TPESampler(group: true)`; the GA family had
no analog.)

## The idea

NSGA-III-CPT takes a declarative rule set — **`ConditionalParams`** — that says which parameters are
conditional on which activator values. Declared parameters stay in the relative (evolved) search space
and are crossed over, mutated and inherited whenever they are *active*, and are simply absent — never
blended, never counted — whenever they are *inactive*. The rules form a **tree** of
categorically-gated parameters (hence *Categorical Parameter Trees*): an activator gates its dependents,
a dependent may itself gate others (chains), several rules on one activator **OR**, several activators on
one dependent **AND**, and cycles are rejected at declaration.

The genetic core is [NSGA-III](../methods/nsga-iii.md) unchanged — reference-point niching, adaptive
normalization, and the shared crossovers / mutations / ranking. Only three parameter-set-shaping seams
become conditional-aware (search-space construction, crossover, child generation), and a
**math-neutrality** property holds: a parameter that is inactive for a child perturbs neither the
produced individual nor the RNG stream — bit-identical to a counterfactual world in which that parameter
never existed.

**Dynamic mode (opt-in).** With `PreserveDormantValues`, an inactive gene's last value is parked in a
latent side channel and *resumed* when its branch switches back on, so a fluctuating genome does not
repeatedly relearn the same gene. It is out of every equation while dormant (never suggested, scored,
crossed or mutated) and consumes no sampler RNG.

## In OptunaSharp

```csharp
using OptunaSharp;
using OptunaSharp.Samplers;
using OptunaSharp.Samplers.NsgaiiiCpt;

var rules = new ConditionalParams()
    .When("optimizer", "adam", "adamw").Activates("beta1", "beta2")   // a value set (OR within an activator)
    .When("optimizer", "sgd").Activates("momentum")                    // a single value
    .When("n_layers", v => (long)v! >= 3).Activates("width_3");        // a numeric predicate

var study = Optuna.CreateStudy(
    directions: new[] { StudyDirection.Minimize, StudyDirection.Minimize },
    sampler: new NSGAIIICptSampler(populationSize: 32, seed: 0, conditionalParams: rules));
```

Constructed with `conditionalParams: null`, `NSGAIIICptSampler` is behaviorally identical to
[`NSGAIIISampler`](../methods/nsga-iii.md) — the extension is entirely opt-in, and it is a strict
superset of NSGA-III.

## Parity & isolation

NSGA-III-CPT is the **sole home** of the conditional-parameter feature. The ported samplers —
[`NSGAIISampler`](../methods/nsga-ii.md), [`NSGAIIISampler`](../methods/nsga-iii.md),
[`AgeMoea2Sampler`](../methods/age-moea-2.md), [`RveaSampler`](../methods/rvea.md) — are bit-exact Optuna
ports and do **not** accept `conditionalParams`. Everything CPT-specific — the sampler, its search
space, crossover, child-generation strategy, and the `ConditionalParams` rule type itself — lives in the
`OptunaSharp.Samplers.NsgaiiiCpt` namespace, so the ported classes stay pristine.

## Relationship to the lineage

- **From [NSGA-III (2014)](../methods/nsga-iii.md):** identical many-objective genetic core
  (reference-point niching, adaptive normalization, the shared operators). Only the parameter-set shaping
  is conditional-aware; selection, ranking and niching are untouched.
- **Prior art:** conditional / hierarchical, variable-length genomes are classical in genetic programming
  and neuroevolution. NSGA-III-CPT's contribution is a precise *math-neutrality* integration into an
  otherwise bit-exact sampler, plus the non-destructive *dormant-memory* variant — expressed declaratively
  rather than inferred.

## See also

The full design write-up (motivation, the activation algebra, the three integration seams, the
math-neutrality property, and the dynamic dormant-memory mode) lives in
[`docs/NSGA-III-DCPARA.md`](../../../NSGA-III-DCPARA.md). The user-facing guide, with the rule syntax and
the null-control guarantees, is in [Samplers § Conditional parameters](../samplers.md). Also:
[NSGA-III (2014)](../methods/nsga-iii.md) · [Search Spaces](../search-spaces.md) ·
[Multi-objective Optimization](../multi-objective.md) · the [API reference](../../api/index.md).
