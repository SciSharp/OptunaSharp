# Search Spaces

Because OptunaSharp is *define-by-run*, there is no static search-space declaration — the space is
whatever the objective happened to suggest. Different trials can suggest different parameters (a
conditional `if` branch, a variable number of layers), so the space is **dynamic**. Several samplers
need a stable, shared space to reason about; that is what search-space inference computes. OptunaSharp
ports `optuna.search_space` into **`OptunaSharp.SearchSpace`**.

## Intersection search space

The **intersection** search space is the set of parameters that appear in *every* completed trial,
with a distribution compatible across all of them. It is the safe common ground a relative sampler can
model jointly.

```csharp
using OptunaSharp.Collections;     // OrderedDict
using OptunaSharp.Distributions;   // BaseDistribution
using OptunaSharp.SearchSpace;

var searchSpace = new IntersectionSearchSpace();
OrderedDict<string, BaseDistribution> space = searchSpace.Calculate(study);

foreach (var (name, dist) in space)
    Console.WriteLine($"{name}: {dist}");
```

`IntersectionSearchSpace` can cache incrementally across calls within a run:

```csharp
public IntersectionSearchSpace(bool includePruned = false);
public OrderedDict<string, BaseDistribution> Calculate(Study study, bool useCache = false);
```

Set `includePruned: true` to also fold in parameters from pruned trials. Pass `useCache: true` to
reuse the result of previous calls and only fold in trials seen since.

For a one-shot computation over a specific list of trials, use the static helper:

```csharp
OrderedDict<string, BaseDistribution> space =
    SearchSpaceFunctions.IntersectionSearchSpaceOf(trials, includePruned: false);
```

The result preserves the order in which parameters were first seen (`OrderedDict`), matching Optuna.

## Grouped search space

When a sampler is configured to model parameters in independent groups (for example
`new TPESampler(group: true)`), OptunaSharp decomposes the space into maximal subsets of parameters
that always co-occur, so each group is modeled jointly and groups are modeled independently. That
decomposition is an internal detail driven by the sampler — you enable it through the sampler's
`group` option rather than calling it directly. See [Samplers](samplers.md).

## Relationship to Optuna

| Optuna (Python) | OptunaSharp |
|-----------------|-------------|
| `optuna.search_space.IntersectionSearchSpace()` | `new IntersectionSearchSpace()` |
| `.calculate(study)` | `.Calculate(study)` |
| `intersection_search_space(trials)` | `SearchSpaceFunctions.IntersectionSearchSpaceOf(trials)` |

## See also

[Distributions](distributions.md) · [Samplers](samplers.md) · [Study and Trials](study-and-trials.md) ·
the [API reference](../api/index.md).
