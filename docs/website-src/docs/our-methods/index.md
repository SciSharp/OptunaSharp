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

**NSGA-III extensions.**

| | Method | Built on |
|---|---|---|
| ✅ | [NSGA-III-CPT — Categorical Parameter Trees](nsga-iii-cpt.md) | [NSGA-III (2014)](../methods/nsga-iii.md) |

## See also

[Methods](../methods/index.md) · [Samplers](../samplers.md) · [Search Spaces](../search-spaces.md) ·
[Multi-objective Optimization](../multi-objective.md) · the [API reference](../../api/index.md).
