# Study and Trials

A **study** is one optimization session; a **trial** is one evaluation of your objective. Together
they implement Optuna's *define-by-run* model: the objective *suggests* hyperparameters as it runs,
and the sampler draws them from the search space it has inferred so far. OptunaSharp ports
`optuna.study` and `optuna.trial`; the entry points live in the root **`OptunaSharp`** namespace.

## Your first study

```csharp
using OptunaSharp;

var study = Optuna.CreateStudy(direction: StudyDirection.Minimize);

study.Optimize(trial =>
{
    double x = trial.SuggestFloat("x", -10, 10);
    return (x - 2) * (x - 2);
}, nTrials: 100);

Console.WriteLine(study.BestValue);            // ~0
Console.WriteLine(study.BestParams["x"]);      // ~2
```

The objective is a `Func<Trial, double>`. Its return value is the number to minimize (or maximize).

## Creating a study

```csharp
public static Study CreateStudy(
    BaseStorage? storage = null,        // default: InMemoryStorage
    BaseSampler? sampler = null,        // default: TPESampler (single-obj) / NSGAIISampler (multi-obj)
    BasePruner? pruner = null,          // default: MedianPruner
    string? studyName = null,           // default: an auto-generated "no-name-…"
    StudyDirection? direction = null,   // Minimize (default) or Maximize
    bool loadIfExists = false,          // resume instead of throwing if the name exists
    IEnumerable<StudyDirection>? directions = null);   // multi-objective (see below)
```

Related module functions on `Optuna`: `LoadStudy`, `DeleteStudy`, `CopyStudy`,
`GetAllStudySummaries`, and `GetAllStudyNames`. Pass a persistent [storage](storage.md) plus
`studyName` and `loadIfExists: true` to make a run resumable. `GetAllStudySummaries` returns
`StudySummary` records — a study's name, direction(s), best trial, trial count, timestamps, and
attributes — a lightweight snapshot that never loads every trial.

## Suggesting hyperparameters

Inside the objective, ask the trial for each hyperparameter. The sampler decides the value; you get a
concrete number or choice back immediately.

```csharp
double lr    = trial.SuggestFloat("lr", 1e-5, 1e-1, log: true);   // log-uniform
double drop  = trial.SuggestFloat("dropout", 0.0, 0.5, step: 0.1); // stepped
long   units = trial.SuggestInt("units", 16, 256, log: true);
long   layers = trial.SuggestInt("layers", 1, 5);                  // step defaults to 1

// Categorical — a typed overload returns the element type directly:
string optimizer = trial.SuggestCategorical("optimizer", new[] { "adam", "sgd", "rmsprop" });
bool   useBn     = trial.SuggestCategorical("batch_norm", new[] { true, false });
```

A C# `enum` drives a categorical directly — its members become the string choices and the value comes
back typed (see [Pythonic search space](key-features/search-space.md)):

```csharp
enum Optimizer { Adam, Sgd, RmsProp }

// inside the objective — over every declared member:
Optimizer optimizer = trial.SuggestCategorical<Optimizer>("optimizer");
```

The full signatures (on the shared `BaseTrial` base):

```csharp
double SuggestFloat(string name, double low, double high, double? step = null, bool log = false);
long   SuggestInt(string name, long low, long high, long step = 1, bool log = false);
CategoricalChoice SuggestCategorical(string name, IEnumerable<CategoricalChoice> choices);
// plus typed convenience overloads for string / bool / int / long / double sequences,
// and SuggestCategorical<TEnum>(name) / (name, subset) for enums
```

`step` and `log` are mutually exclusive for floats, exactly as in Optuna. Categorical values are
modeled with [`CategoricalChoice`](distributions.md), which implements Python equality semantics
(`true == 1`, `1.0 == 1`, `NaN != NaN`).

## Ask-and-tell

For loops you control yourself — a training framework's own loop, batched evaluation, or
human-in-the-loop tuning — use ask/tell instead of `Optimize`:

```csharp
for (int i = 0; i < 100; i++)
{
    Trial trial = study.Ask();
    double x = trial.SuggestFloat("x", -10, 10);
    study.Tell(trial, (x - 2) * (x - 2));
}
```

`Tell` has overloads taking a `Trial` or a trial number, a single `double` or an
`IReadOnlyList<double>` (multi-objective), and an explicit `TrialState` (e.g. to record a failure).
`Ask` optionally takes `fixedDistributions` to pin part of the search space.

## Reading results

```csharp
double best              = study.BestValue;    // single-objective
var    bestParams        = study.BestParams;   // IReadOnlyDictionary<string, object?>
FrozenTrial bestTrial    = study.BestTrial;
List<FrozenTrial> all    = study.Trials;
```

Filter by state with `GetTrials(deepcopy: true, states: new[] { TrialState.Complete })`. `TrialState`
is `Running`, `Complete`, `Pruned`, `Fail`, or `Waiting`; `state.IsFinished()` is a handy extension.

## Pruning unpromising trials

Report intermediate values during training and let the [pruner](pruners.md) stop hopeless trials
early:

```csharp
study.Optimize(trial =>
{
    for (int step = 0; step < 100; step++)
    {
        double loss = TrainOneEpoch(trial, step);
        trial.Report(loss, step);
        if (trial.ShouldPrune())
            throw new TrialPruned();
    }
    return FinalScore(trial);
}, nTrials: 50);
```

`Optimize` catches `TrialPruned` and records the trial as `Pruned`.

## Attributes, enqueueing, and manual trials

```csharp
study.SetUserAttr("dataset", "cifar10");           // arbitrary metadata on the study
trial.SetUserAttr("fold", 3);                       // …or on a trial

study.EnqueueTrial(new() { ["x"] = 0.0 });          // evaluate specific params next
study.AddTrial(FrozenTrial.CreateTrial(             // inject an already-evaluated trial
    value: 0.5, @params: new() { ["x"] = 1.5 }));
```

## Callbacks and trial limits

`Optimize` takes a `callbacks` list of `Action<Study, FrozenTrial>` run after every trial — for logging,
early exit, or bookkeeping. `MaxTrialsCallback` caps a study at a **total** number of trials in chosen
states across every worker, independent of how many `nTrials` any single `Optimize` call requested
(useful when several processes share one storage):

```csharp
using OptunaSharp;

study.Optimize(Objective, nTrials: 1000, callbacks: new[]
{
    new MaxTrialsCallback(100, new[] { TrialState.Complete }).Invoke,   // stop once 100 have completed
});
```

## Trial types

- **`Trial`** — the live handle passed to your objective; its `Suggest*` calls draw from the sampler.
- **`FrozenTrial`** — an immutable snapshot in `study.Trials`; exposes `Number`, `State`, `Value`/
  `Values`, `Params`, `UserAttrs`, `Duration`, `IntermediateValues`, and `DeepCopy()`. Build one with
  the static `FrozenTrial.CreateTrial(...)` for tests or `AddTrial`.
- **`FixedTrial`** — returns preset parameters, for unit-testing an objective without a study:
  `new FixedTrial(new() { ["x"] = 2.0 })`.

## Exporting trials

`study.TrialsDataframe()` returns a `TrialsTable` (the analog of Optuna's pandas
`trials_dataframe()`) with a column per parameter, value, attribute, and timestamp.

## Multi-objective studies

Pass `directions` instead of `direction` to optimize several objectives at once; the objective then
returns an `IReadOnlyList<double>`. See [Multi-objective Optimization](multi-objective.md).

## Relationship to Optuna

| Optuna (Python) | OptunaSharp |
|-----------------|-------------|
| `optuna.create_study(direction="minimize")` | `Optuna.CreateStudy(direction: StudyDirection.Minimize)` |
| `study.optimize(objective, n_trials=100)` | `study.Optimize(objective, nTrials: 100)` |
| `trial.suggest_float("x", -10, 10, log=True)` | `trial.SuggestFloat("x", -10, 10, log: true)` |
| `trial.suggest_int("n", 1, 5)` | `trial.SuggestInt("n", 1, 5)` |
| `trial.suggest_categorical("k", ["a","b"])` | `trial.SuggestCategorical("k", new[] { "a", "b" })` |
| `study.best_params` / `study.best_value` | `study.BestParams` / `study.BestValue` |
| `study.ask()` / `study.tell(t, v)` | `study.Ask()` / `study.Tell(t, v)` |
| `raise optuna.TrialPruned()` | `throw new TrialPruned()` |

Under a fixed seed, OptunaSharp reproduces Optuna's trial sequence **bit-for-bit** wherever the path
is deterministic — the guarantee the port is built around.

## See also

[Distributions](distributions.md) · [Samplers](samplers.md) · [Pruners](pruners.md) ·
[Storage](storage.md) · [Multi-objective Optimization](multi-objective.md) ·
the [API reference](../api/index.md).
