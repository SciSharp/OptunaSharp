# Exceptions

OptunaSharp ports Optuna's exception types into the root **`OptunaSharp`** namespace. They fall into two
groups: Optuna's own exceptions, all deriving from `OptunaError`, and Python's built-in exceptions,
which OptunaSharp raises as the matching **NumSharp / .NET** types rather than redefining — so typed
error handling stays consistent across the SciSharp STACK.

## Optuna-specific exceptions

Every Optuna-specific error derives from `OptunaError` (itself a plain `Exception`), so a single
`catch (OptunaError)` covers them all.

| Exception | Raised when | Optuna |
|-----------|-------------|--------|
| `OptunaError` | Base class for all Optuna-specific errors. | `optuna.exceptions.OptunaError` |
| `TrialPruned` | Thrown by your objective to mark the trial pruned (after `ShouldPrune`). `Optimize` catches it and records the trial as `Pruned`. | `optuna.TrialPruned` |
| `CLIUsageError` | The command-line interface received an invalid configuration. | `optuna.exceptions.CLIUsageError` |
| `StorageInternalError` | A storage backend operation failed. | `optuna.exceptions.StorageInternalError` |
| `DuplicatedStudyError` | Creating a study whose name already exists (without `loadIfExists: true`). | `optuna.exceptions.DuplicatedStudyError` |
| `UpdateFinishedTrialError` | Writing to a trial that has already finished. | `optuna.exceptions.UpdateFinishedTrialError` |

`TrialPruned` is the one you raise yourself; the rest come from OptunaSharp:

```csharp
using OptunaSharp;

study.Optimize(trial =>
{
    for (int step = 0; step < 100; step++)
    {
        trial.Report(TrainOneEpoch(trial, step), step);
        if (trial.ShouldPrune())
            throw new TrialPruned();     // Optimize records this trial as Pruned
    }
    return Validate(trial);
}, nTrials: 50);
```

## Python built-ins map to NumSharp / .NET types

Optuna also raises several plain Python built-in exceptions — `ValueError`, `TypeError`, `KeyError`,
`RuntimeError`, `NotImplementedError`. OptunaSharp does **not** redefine these; it throws the existing
NumSharp or .NET type instead, so an argument rejected by the same rule as Python raises the same kind
of error you already catch elsewhere in a NumSharp program.

| Python (Optuna) | OptunaSharp | Example |
|-----------------|-------------|---------|
| `ValueError` | `NumSharp.ValueError` | `SuggestFloat` with `low > high`; an unsupported storage URL |
| `TypeError` | `NumSharp.TypeError` | a distribution handed the wrong value type |
| `KeyError` | `NumSharp.KeyError` | loading a study name that doesn't exist |
| `RuntimeError` | `NumSharp.RuntimeError` | an invalid internal state |
| `NotImplementedError` | `System.NotImplementedException` | `trial.Report` in a multi-objective study |

```csharp
using OptunaSharp;
using NumSharp;   // for ValueError, KeyError, …

try
{
    var study = Optuna.LoadStudy(storage: storage, studyName: "does-not-exist");
}
catch (KeyError)   // NumSharp.KeyError — the Python KeyError analog
{
    // no such study in this storage
}
```

## Relationship to Optuna

| Optuna (Python) | OptunaSharp |
|-----------------|-------------|
| `raise optuna.TrialPruned()` | `throw new TrialPruned()` |
| `except optuna.exceptions.OptunaError` | `catch (OptunaError)` |
| `except optuna.exceptions.DuplicatedStudyError` | `catch (DuplicatedStudyError)` |
| `raise ValueError(...)` | `throw new NumSharp.ValueError(...)` |
| `raise NotImplementedError(...)` | `throw new NotImplementedException(...)` |

**Divergence.** Python's `UpdateFinishedTrialError` inherits from **both** `OptunaError` and
`RuntimeError`; C# has single inheritance, so OptunaSharp keeps the `OptunaError` base only. Catch it by
its own type or by `OptunaError`.

## See also

[Study and Trials](study-and-trials.md) · [Pruners](pruners.md) — where `TrialPruned` is used ·
[Storage](storage.md) — where the storage errors surface · the [API reference](../api/index.md).
