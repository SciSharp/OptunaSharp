# Multi-objective Optimization

Many problems trade off several goals at once — accuracy against latency, return against risk. A
**multi-objective** study optimizes a vector of objectives and returns the **Pareto front**: the set
of trials where no objective can be improved without worsening another. OptunaSharp has no separate
multi-objective module; it is the same [`Study`](study-and-trials.md) API with more than one direction,
mirroring Optuna.

## Defining multiple objectives

Pass `directions` (plural) to `CreateStudy` and return an `IReadOnlyList<double>` from the objective —
one value per direction, in order.

```csharp
using OptunaSharp;

var study = Optuna.CreateStudy(directions: new[]
{
    StudyDirection.Minimize,   // objective 0: model size
    StudyDirection.Maximize,   // objective 1: accuracy
});

study.Optimize(trial =>
{
    long units = trial.SuggestInt("units", 16, 256);
    double drop = trial.SuggestFloat("dropout", 0.0, 0.5);

    double size = ModelSize(units);
    double acc  = Train(units, drop);
    return new[] { size, acc };     // IReadOnlyList<double>, aligned with directions
}, nTrials: 200);
```

The default sampler for a multi-objective study is `NSGAIISampler`.

## Reading the Pareto front

Single-objective conveniences (`BestValue`, `BestParams`, `BestTrial`) don't apply — there is no
single best. Use `BestTrials`, the non-dominated set:

```csharp
foreach (FrozenTrial t in study.BestTrials)
    Console.WriteLine($"#{t.Number}: [{string.Join(", ", t.Values!)}]");

IReadOnlyList<StudyDirection> dirs = study.Directions;   // the objective directions
```

Each `FrozenTrial.Values` holds that trial's objective vector (versus `Value` for single-objective).

## Choosing a sampler

- **`NSGAIISampler`** (default) — non-dominated sorting with crowding distance; strong for 2–3
  objectives.
- **`NSGAIIISampler`** — reference-point based; better for many objectives.
- **`TPESampler`** — drives MOTPE automatically when the study has multiple directions.
- **`GPSampler`** — Gaussian-process multi-objective acquisition (log-EHVI and its constrained/batched
  variants) for expensive objectives.

See [Samplers](samplers.md) for construction and options.

## Naming and constraints

Label the objectives for readable reports and visualizations:

```csharp
study.SetMetricNames(new[] { "model_size", "accuracy" });
```

For constrained multi-objective optimization, supply a `constraintsFunc` to the sampler (supported by
`NSGAIISampler`, `NSGAIIISampler`, `TPESampler`, and `GPSampler`); a trial is feasible when every
returned constraint value is ≤ 0.

## Visualizing trade-offs

Two [visualization](visualization.md) builders target multi-objective studies:

- `ParetoFront` → `ParetoFrontInfo` — the front in objective space.
- `HypervolumeHistory` → `HypervolumeHistoryInfo` — dominated hypervolume over time, relative to a
  reference point, a standard convergence measure.

## Relationship to Optuna

| Optuna (Python) | OptunaSharp |
|-----------------|-------------|
| `create_study(directions=["minimize", "maximize"])` | `Optuna.CreateStudy(directions: new[] { StudyDirection.Minimize, StudyDirection.Maximize })` |
| `return [size, acc]` | `return new[] { size, acc }` |
| `study.best_trials` | `study.BestTrials` |
| `study.set_metric_names([...])` | `study.SetMetricNames(new[] { ... })` |

**Parity.** The hypervolume computation and the hypervolume subset selection (HSSP) that breaks ties in
multi-objective TPE match Optuna bit for bit, zero-contribution ties with 3 or more objectives included: the
2- and 3-objective volumes use the same BLAS dot products NumPy uses (through NumSharp), so the tiny
residuals that decide those ties are Optuna's own. In OptunaSharp 1.0.0 those ties could resolve to a
different, equally good point.

## See also

[Study and Trials](study-and-trials.md) · [Samplers](samplers.md) · [Visualization](visualization.md) ·
the [API reference](../api/index.md).
