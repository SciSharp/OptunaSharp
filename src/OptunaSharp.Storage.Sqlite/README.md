# OptunaSharp.Storage.Sqlite

SQLite **RDB storage** for [OptunaSharp](https://github.com/SciSharp/OptunaSharp) — a faithful port
of `optuna.storages.RDBStorage` on top of
[Microsoft.Data.Sqlite](https://learn.microsoft.com/dotnet/standard/data/sqlite/). It persists a
study to a local `.db` file using **Optuna's exact schema** (alembic revision `v3.2.0.a`,
`schema_version = 12`), so **Python Optuna and OptunaSharp can open the same database file**.

Part of the [SciSharp STACK](https://github.com/SciSharp).

## Install

```bash
dotnet add package OptunaSharp.Storage.Sqlite
```

The native SQLite engine (`e_sqlite3`) is bundled by `Microsoft.Data.Sqlite` — no separate install.

## Usage

```csharp
using OptunaSharp;
using OptunaSharp.Storages;

using var storage = new RDBStorage("sqlite:///study.db");
var study = Optuna.CreateStudy(storage: storage, studyName: "quadratic");
study.Optimize(trial =>
{
    double x = trial.SuggestFloat("x", -100, 100);
    return x * x;
}, nTrials: 20);

Console.WriteLine($"{study.BestValue} at {study.BestParams["x"]}");
```

`Optuna.CreateStudy` wraps the `RDBStorage` in a read-through `CachedStorage` automatically — exactly
as Python's `optuna.storages.get_storage` returns `_CachedStorage(RDBStorage(...))`.

### URL forms

| URL | Database |
|-----|----------|
| `sqlite:///study.db` | file `study.db`, relative to the working directory |
| `sqlite:////data/study.db` | file `/data/study.db` (absolute) |
| `sqlite:///C:/data/study.db` | file `C:\data\study.db` (Windows absolute) |
| `sqlite:///:memory:` | in-memory database (lives as long as the `RDBStorage` instance) |

## Cross-language interoperability

A database created by this package can be reopened by Python Optuna, and vice versa:

```python
import optuna
study = optuna.load_study(study_name="quadratic", storage="sqlite:///study.db")
print(study.best_value, study.best_params)
```

Datetimes are stored as naive UTC (`YYYY-MM-DD HH:MM:SS.ffffff`), objective/intermediate `±inf`/`NaN`
are stored as a typed tag with a `NULL` value, distributions use Optuna's `distribution_json`, and
attributes use `json.dumps` — all byte-compatible with Optuna.

> **Concurrency.** As with Python Optuna, SQLite is **not recommended for parallel optimization**.
> This backend serializes all access through a single connection; for concurrent workers use the
> journal, Redis, or (later) gRPC backends.

## License

MIT — see the repository root.
