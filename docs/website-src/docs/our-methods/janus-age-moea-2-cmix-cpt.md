# JANUS-AGE-MOEA-2-CMIX-CPT — Categorical Parameter Trees

**JANUS-AGE-MOEA-2-CMIX with Categorical Parameter Trees** — an **OptunaSharp original** with no upstream
Optuna analog. It carries the [NSGA-III-CPT](nsga-iii-cpt.md) conditional-parameter machinery onto a
JANUS-augmented AGE-MOEA-2 host, and ships as **`JanusAgeMoea2CmixCptSampler`**
(`OptunaSharp.Samplers`). It is simultaneously the conditional-parameter sibling of
[JANUS-AGE-MOEA-2-CMIX](../methods/janus-age-moea-2-cmix.md) and the JANUS-augmented sibling of
[AGE-MOEA-2-CPT](age-moea-2-cpt.md).

## What it combines

Three ideas stack, each cleanly separable:

1. **[AGE-MOEA-2 (2022)](../methods/age-moea-2.md)** survival — adaptive front-geometry environmental
   selection (the many-objective host).
2. **[Categorical Parameter Trees](age-moea-2-cpt.md)** — the OptunaSharp-original conditional-parameter
   machinery, so branch-defined hyperparameters are *evolved* rather than resampled at random. This is the
   part that makes it an "Our Method": AGE-MOEA-2 and the JANUS CMIX idea are published, the conditional
   integration is OptunaSharp's own.
3. **JANUS CMIX infill** ([Yu et al., arXiv:2608.22862](https://arxiv.org/abs/2608.22862)) — past a
   warm-up, a small fraction of children are replaced by an inverse-curvature Gaussian sample drawn from
   the recent window (using `component_losses` residuals, or a diagonal-quadratic scalar fallback).

Concretely, `JanusAgeMoea2CmixCptSampler` extends [`AgeMoea2CptSampler`](age-moea-2-cpt.md): the
AGE-MOEA-2-CPT host runs first (its conditional child + dormant carry, RNG stream **identical** to the
bare host), then the shared JANUS CMIX engine decides whether to substitute the child.

## The one thing to know: when CMIX fires

The CMIX engine needs an **all-numeric relative search space** — it builds `Float`/`Int` transforms to fit
its inverse-curvature metric and bails to null on any categorical. So:

- On a **categorical-dominant or conditional-categorical** space (e.g. gene-tree genomes), the gate
  **never fires** and this sampler is *exactly* [`AgeMoea2CptSampler`](age-moea-2-cpt.md). That is the
  intended, useful behavior — the value there is AGE-MOEA-2's survival plus CPT, not the infill.
- On a **continuous (all-numeric) conditional** multi-objective space, CMIX fires; the injected sample
  spans the relative space without regard to conditional activation, so the substituted child is then
  **pruned against the rules** so it cannot carry inactive-branch parameters.

In short: reserve the CMIX layer for continuous multi-objective problems. It is carried here as the named
JANUS variant so a single class covers both regimes.

## In OptunaSharp

```csharp
using OptunaSharp;
using OptunaSharp.Samplers;
using OptunaSharp.Samplers.NsgaiiiCpt;

// A NUMERIC activator keeps the whole relative space Float/Int, so the CMIX gate can fire on it. A
// categorical activator would keep the space non-numeric and CMIX would never fire (see below).
var rules = new ConditionalParams()
    .When("n_layers", v => (long)v! >= 3).Activates("width_3", "dropout_3");

var study = Optuna.CreateStudy(
    directions: new[] { StudyDirection.Minimize, StudyDirection.Minimize },
    sampler: new JanusAgeMoea2CmixCptSampler(
        populationSize: 40, seed: 0, cmixProb: 0.02, conditionalParams: rules));

study.Optimize((Func<Trial, IReadOnlyList<double>>)(t =>
{
    long n = t.SuggestInt("n_layers", 1, 5);
    double w = t.SuggestFloat("base_width", 0, 1);
    double extra = 0;
    if (n >= 3)   // width_3 / dropout_3 exist only on this branch, and CMIX can act on them
    {
        extra = t.SuggestFloat("width_3", 0, 1) + t.SuggestFloat("dropout_3", 0, 1);
    }
    // t.SetUserAttr("component_losses", new[] { ... });   // optional per-objective residuals for CMIX
    return new[] { w * w + extra, (w - 1) * (w - 1) + extra };
}), nTrials: 500);
```

The AGE-MOEA-2 operators (`mutation`, `crossover`, `crossoverProb`, `swappingProb`, `populationSize`) and
`conditionalParams` are forwarded to [`AgeMoea2CptSampler`](age-moea-2-cpt.md); the CMIX knobs
(`cmixProb`, `cmixWindow`, `cmixRidge`, `cmixLambda`, `cmixSigma`, `cmixStartGeneration` /
`cmixStartTrial`, `componentAttr`, `scalarFallback`) are the same ones documented on
[JANUS-MOO-CMIX](../methods/janus-moo-cmix.md), including its PCG64-RNG divergence note. With
`conditionalParams: null` the host reduces to a plain [`AgeMoea2Sampler`](../methods/age-moea-2.md) under
the CMIX layer.

## Relationship to the lineage

- **From [AGE-MOEA-2-CPT](age-moea-2-cpt.md):** inherits the whole AGE-MOEA-2 survival + CPT stack
  (subclass); adds only the CMIX substitution after the host's child is produced.
- **From [JANUS-AGE-MOEA-2-CMIX](../methods/janus-age-moea-2-cmix.md):** the identical shared `JanusMooCmix`
  engine, now over an AGE-MOEA-2-**CPT** host instead of the bit-exact AGE-MOEA-2 port — so it evolves
  conditional parameters that the non-CPT variant would resample at random.

## See also

[AGE-MOEA-2-CPT](age-moea-2-cpt.md) · [NSGA-III-CPT](nsga-iii-cpt.md) ·
[JANUS-AGE-MOEA-2-CMIX (2026)](../methods/janus-age-moea-2-cmix.md) ·
[JANUS-MOO-CMIX (2026)](../methods/janus-moo-cmix.md) · [Samplers](../samplers.md) ·
[Multi-objective Optimization](../multi-objective.md) · the [API reference](../../api/index.md).
