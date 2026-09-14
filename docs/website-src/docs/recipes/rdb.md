# Saving/Resuming a Study with an RDB backend

A relational-database (RDB) backend makes a study **persistent**: you can save it, resume it later,
and inspect the history of every trial. It is also what lets several processes optimize the same
study at once. This recipe uses **SQLite**, a single-file database that needs no server; install
**`OptunaSharp.Storage.Sqlite`** to get `RDBStorage`.

> SQLite is one host only. For PostgreSQL or MySQL use `OptunaSharp.Storage.EFCore`; to share a
> study across machines, put it behind the [gRPC proxy](../storage.md#grpc-storage-proxy-distributed-cross-language)
> or use [journal-file storage](journal-storage.md). See the [Storage guide](../storage.md).

## New study

Create a persistent study by passing a storage. The SQLite file `example.db` is created
automatically and a new study record is written into it.

```csharp
using OptunaSharp;
using OptunaSharp.Storages;

using var storage = new RDBStorage("sqlite:///example.db");
var study = Optuna.CreateStudy(storage: storage, studyName: "example-study");

study.Optimize(trial =>
{
    double x = trial.SuggestFloat("x", -10, 10);
    return (x - 2) * (x - 2);
}, nTrials: 3);
```

`studyName` is the unique identifier you resume by. Because the same schema Optuna writes is used,
**Python Optuna can open `example.db`** and vice versa.

## Resume study

To resume, construct a study with the *same* name and storage and pass `loadIfExists: true` — it
attaches to the existing record instead of throwing that the name is taken.

```csharp
using var storage = new RDBStorage("sqlite:///example.db");
var study = Optuna.CreateStudy(
    storage: storage, studyName: "example-study", loadIfExists: true);

study.Optimize(trial =>
{
    double x = trial.SuggestFloat("x", -10, 10);
    return (x - 2) * (x - 2);
}, nTrials: 3);   // these 3 trials are appended to the 3 already stored
```

The storage does **not** persist the sampler's or pruner's in-memory state. If you seeded the sampler
for reproducibility, construct a fresh seeded sampler and pass it to `CreateStudy` when you resume.

## Reading the stored history

Every property of a study reads through the storage, so a resumed study sees all prior trials:

```csharp
using var storage = new RDBStorage("sqlite:///example.db");
var study = Optuna.CreateStudy(
    storage: storage, studyName: "example-study", loadIfExists: true);

Console.WriteLine($"Best params: {study.BestParams["x"]}");
Console.WriteLine($"Best value:  {study.BestValue}");
Console.WriteLine($"Best trial:  {study.BestTrial}");
Console.WriteLine($"Trials:      {study.Trials.Count}");

// Tabular export (the analog of Optuna's trials_dataframe):
TrialsTable df = study.TrialsDataframe(attrs: new[] { "number", "value", "params", "state" });
```

`TrialsDataframe` returns a [`TrialsTable`](../study-and-trials.md#exporting-trials) with a column per
field — no pandas runtime required.

## Relationship to Optuna

| Optuna (Python) | OptunaSharp |
|-----------------|-------------|
| `optuna.create_study(study_name=…, storage="sqlite:///example.db")` | `Optuna.CreateStudy(studyName: …, storage: new RDBStorage("sqlite:///example.db"))` |
| `optuna.create_study(…, load_if_exists=True)` | `Optuna.CreateStudy(…, loadIfExists: true)` |
| `study.trials_dataframe(attrs=("number","value","params","state"))` | `study.TrialsDataframe(attrs: new[] { "number", "value", "params", "state" })` |
| `study.best_params` / `study.best_value` | `study.BestParams` / `study.BestValue` |

## See also

[Storage](../storage.md) · [Journal-file storage](journal-storage.md) ·
[Study and Trials](../study-and-trials.md) · the [SQLite integration guide](../../integrations/storage/sqlite.md).
