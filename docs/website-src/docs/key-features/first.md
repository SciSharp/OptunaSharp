# Lightweight, versatile, and platform-agnostic architecture

OptunaSharp's core is small and has one dependency ([NumSharp](../getting-started.md#the-numsharp-reference)).
That keeps it easy to drop into any project and quick to get from "hello world" to a real tuning run.

## A quadratic-function example

Optuna is usually used to tune hyperparameters, but the mechanics are clearest on a plain function —
let's minimize `(x - 2)²`. By convention the function to optimize is named `Objective`; it asks the
trial for each parameter with a `Suggest*` call, then returns the value to minimize:

```csharp
using OptunaSharp;

double Objective(Trial trial)
{
    double x = trial.SuggestFloat("x", -10, 10);   // uniform in [-10, 10]
    return (x - 2) * (x - 2);
}

var study = Optuna.CreateStudy();
study.Optimize(Objective, nTrials: 100);

var bestParams = study.BestParams;
double foundX = (double)bestParams["x"]!;
Console.WriteLine($"Found x: {foundX}, (x - 2)^2: {(foundX - 2) * (foundX - 2)}");
```

Each call of `Objective` is one **trial**; OptunaSharp instantiates a `Trial` and evaluates the
function with a new `x`. After 100 trials the `x` it found is close to the optimum, `2`.

> For machine learning, `Objective` would instead return a model's loss or (negated) accuracy.

## The study object

Three terms:

- **Trial** — a single call of the objective function.
- **Study** — an optimization session: a set of trials.
- **Parameter** — a variable being optimized, such as `x` above.

`Optuna.CreateStudy()` returns a [`Study`](../study-and-trials.md), which exposes the results:

```csharp
IReadOnlyDictionary<string, object?> best = study.BestParams;   // {"x": ~2.0}
double bestValue          = study.BestValue;                     // ~0.0
FrozenTrial bestTrial     = study.BestTrial;
List<FrozenTrial> trials  = study.Trials;                        // every trial
int nTrials               = study.Trials.Count;

foreach (FrozenTrial t in study.Trials.Take(2))                  // first two
    Console.WriteLine(t);
```

Calling `Optimize` again **continues** the same study — the new trials are appended:

```csharp
study.Optimize(Objective, nTrials: 100);
Console.WriteLine(study.Trials.Count);   // now 200
```

## Platform-agnostic by design

`OptunaSharp` (Core) is **pure managed code** — NumSharp for numerics, with the native LAPACK/BLAS
kernels supplied by `NumSharp.Interop.OpenBLAS`. It multi-targets **.NET 8** and **.NET 10** and runs
anywhere .NET runs: Windows, Linux, macOS, containers, CI. Anything that needs a foreign runtime
(Python, Torch, cloud SDKs, a database driver) lives in a **separate satellite package**, so a Core
dependency never drags in machinery you didn't ask for. See the [Integrations](../../integrations/index.md)
for those optional packages.

## Relationship to Optuna

| Optuna (Python) | OptunaSharp |
|-----------------|-------------|
| `optuna.create_study()` | `Optuna.CreateStudy()` |
| `study.optimize(objective, n_trials=100)` | `study.Optimize(Objective, nTrials: 100)` |
| `study.best_params["x"]` | `study.BestParams["x"]` |
| `study.best_value` / `study.best_trial` | `study.BestValue` / `study.BestTrial` |
| `study.trials` / `len(study.trials)` | `study.Trials` / `study.Trials.Count` |

## See also

[Getting Started](../getting-started.md) · [Study and Trials](../study-and-trials.md) ·
[Pythonic search space](search-space.md) · the [API reference](../../api/index.md).
