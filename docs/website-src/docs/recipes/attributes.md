# User attributes

Annotate a study or an individual trial with your own metadata — a dataset name, the git commit, a
secondary metric you want to keep alongside the objective. Attributes are key/value pairs; the key is
a string and the value should be JSON-serializable so it survives a round-trip through persistent
[storage](../storage.md).

## Adding attributes to a study

`Study.SetUserAttr` registers one attribute; `Study.UserAttrs` reads them back.

```csharp
using OptunaSharp;
using OptunaSharp.Storages;

using var storage = new RDBStorage("sqlite:///example.db");
var study = Optuna.CreateStudy(storage: storage);

study.SetUserAttr("contributors", new[] { "Akiba", "Sano" });
study.SetUserAttr("dataset", "MNIST");

foreach (var (key, value) in study.UserAttrs)
    Console.WriteLine($"{key} = {value}");
// contributors = [Akiba, Sano]
// dataset = MNIST
```

A [`StudySummary`](../study-and-trials.md), which you get from `Optuna.GetAllStudySummaries`, also
carries the study's attributes — handy for listing studies without loading each one:

```csharp
foreach (StudySummary summary in Optuna.GetAllStudySummaries(storage))
    Console.WriteLine(summary.UserAttrs["dataset"]);
```

> The [`optunasharp study set-user-attr`](cli.md) command sets a study attribute from the shell.

## Adding attributes to a trial

A trial has the same method. Set attributes from inside the objective to record per-trial context —
here, a secondary metric computed alongside the objective:

```csharp
double Objective(Trial trial)
{
    double svcC = trial.SuggestFloat("svc_c", 1e-10, 1e10, log: true);
    double accuracy = TrainAndScore(svcC);        // your model here

    trial.SetUserAttr("accuracy", accuracy);       // keep the raw accuracy…
    return 1.0 - accuracy;                          // …while minimizing error
}

study.Optimize(Objective, nTrials: 1);

// Read it back from the stored trial:
Console.WriteLine(study.Trials[0].UserAttrs["accuracy"]);
```

The attribute is attached to that single `FrozenTrial`, not to the study.

## Relationship to Optuna

| Optuna (Python) | OptunaSharp |
|-----------------|-------------|
| `study.set_user_attr("dataset", "MNIST")` | `study.SetUserAttr("dataset", "MNIST")` |
| `study.user_attrs` | `study.UserAttrs` |
| `optuna.get_all_study_summaries("sqlite:///example.db")` | `Optuna.GetAllStudySummaries(storage)` |
| `trial.set_user_attr("accuracy", acc)` | `trial.SetUserAttr("accuracy", acc)` |
| `study.trials[0].user_attrs` | `study.Trials[0].UserAttrs` |

> Note the one API-shape difference: `GetAllStudySummaries` takes a **storage instance**, not a URL
> string. Construct the `RDBStorage` and pass it.

## See also

[Study and Trials](../study-and-trials.md) · [Storage](../storage.md) ·
[Command-line interface](cli.md) · the [API reference](../../api/index.md).
