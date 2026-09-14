# Key Features

A short tour of what makes OptunaSharp — and Optuna, the framework it ports — productive. Where
[Getting Started](../getting-started.md) installs it and the [Guides](../study-and-trials.md) are the
full reference, these five pages showcase the framework's headline features, each a direct port of the
matching page in Optuna's
[Key Features tutorial](https://optuna.readthedocs.io/en/stable/tutorial/index.html#key-features)
(`optuna/tutorial/10_key_features/`).

## The features

| # | Feature | In short |
|---|---------|----------|
| 1 | [Lightweight, versatile, and platform-agnostic architecture](first.md) | A first optimization, the Study/Trial/Parameter vocabulary, and why Core runs anywhere .NET runs |
| 2 | [Pythonic search space](search-space.md) | Define search spaces with ordinary C# — conditionals and loops, define-by-run |
| 3 | [Efficient optimization algorithms](efficient-algorithms.md) | The sampler and pruner suites, how to switch them, and which to pick |
| 4 | [Easy parallelization](parallelization.md) | Multi-thread, multi-process, and multi-node optimization |
| 5 | [Quick visualization](visualization.md) | Plot the optimization history, importances, contours, and more |

## Where to go next

Once the tour makes sense, the [Guides](../study-and-trials.md) cover each subsystem in depth and the
[Recipes](../recipes/index.md) are task-focused how-tos (persisting a study, writing a custom sampler,
ablation sweeps, and more).

## Relationship to Optuna

Each page corresponds one-to-one with a file in Optuna's
[`tutorial/10_key_features/`](https://github.com/optuna/optuna/tree/master/tutorial/10_key_features).
The C# port keeps the same narrative; under a fixed seed the trial sequence is bit-for-bit identical to
Python wherever the path is deterministic.

## See also

[Getting Started](../getting-started.md) · [Study and Trials](../study-and-trials.md) ·
[Recipes](../recipes/index.md) · the [API reference](../../api/index.md).
