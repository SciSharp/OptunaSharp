# Our Methods

This section covers **OptunaSharp originals** — methods with no upstream Optuna analog that we designed
and ship as first-class samplers. It sits above the [Methods](../methods/index.md) reference, which
documents our *ports* of published algorithms; everything here is our own contribution.

The distinction is the point. Under [Methods](../methods/index.md), every entry is a faithful,
parity-checked port of a published algorithm — bit-for-bit lock-step with Optuna wherever Optuna has the
same method. The methods in *this* section are things Optuna does not have. They are strictly opt-in and
never change the behavior of the ported samplers.

- ✅ — full deep-dive page.

## Samplers

**Categorical Parameter Trees (CPT) — conditional-parameter evolution.** A host-agnostic machinery that
keeps *branch-defined* (define-by-run) hyperparameters in the evolved search space instead of resampling
them at random. The same CPT seams are wired onto three multi-objective hosts; the difference between them
is entirely the *survival* rule (and, for the last, an optional JANUS infill layer).

| | Method | Built on |
|---|---|---|
| ✅ | [NSGA-III-CPT — Categorical Parameter Trees](nsga-iii-cpt.md) | [NSGA-III (2014)](../methods/nsga-iii.md) |
| ✅ | [AGE-MOEA-2-CPT — Categorical Parameter Trees](age-moea-2-cpt.md) | [AGE-MOEA-2 (2022)](../methods/age-moea-2.md) |
| ✅ | [JANUS-AGE-MOEA-2-CMIX-CPT — Categorical Parameter Trees](janus-age-moea-2-cmix-cpt.md) | [AGE-MOEA-2-CPT](age-moea-2-cpt.md) + [JANUS-AGE-MOEA-2-CMIX (2026)](../methods/janus-age-moea-2-cmix.md) |

## See also

[Methods](../methods/index.md) · [Samplers](../samplers.md) · [Search Spaces](../search-spaces.md) ·
[Multi-objective Optimization](../multi-objective.md) · the [API reference](../../api/index.md).
