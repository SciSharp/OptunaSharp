# OptunaSharp.Storage.EFCore

**PostgreSQL and MySQL RDB storage** for [OptunaSharp](https://github.com/SciSharp/OptunaSharp) — the
[Entity Framework Core](https://learn.microsoft.com/ef/core/) port of `optuna.storages.RDBStorage` for
server databases. It persists a study using **Optuna's exact SQLAlchemy schema** (alembic revision
`v3.2.0.a`, `schema_version = 12`), so **Python Optuna and OptunaSharp can share the same
PostgreSQL/MySQL database**.

It is the EF Core sibling of the file-based [`OptunaSharp.Storage.Sqlite`](https://www.nuget.org/packages/OptunaSharp.Storage.Sqlite)
package. Part of the [SciSharp STACK](https://github.com/SciSharp).

## Install

```bash
dotnet add package OptunaSharp.Storage.EFCore
```

The two providers ship in the box — [Npgsql](https://www.npgsql.org/) for PostgreSQL and
[Pomelo](https://github.com/PomeloFoundation/Pomelo.EntityFrameworkCore.MySql) + MySqlConnector for
MySQL — so no separate provider install is needed.

## Usage

```csharp
using OptunaSharp;
using OptunaSharp.Storages.EFCore;

using var storage = new RDBStorage("postgresql://user:pass@localhost:5432/optuna");
// or: new RDBStorage("mysql://user:pass@localhost:3306/optuna");

var study = Optuna.CreateStudy(storage: storage, studyName: "quadratic");
study.Optimize(trial =>
{
    double x = trial.SuggestFloat("x", -100, 100);
    return x * x;
}, nTrials: 20);

Console.WriteLine($"{study.BestValue} at {study.BestParams["x"]}");
```

The URL scheme selects the provider (`postgresql[+driver]://…` → PostgreSQL, `mysql[+driver]://…` →
MySQL) — the same SQLAlchemy-style URL Python Optuna takes. `Optuna.CreateStudy` wraps the
`RDBStorage` in a read-through `CachedStorage` automatically, exactly as Python's
`optuna.storages.get_storage` returns `_CachedStorage(RDBStorage(...))`.

### URL forms

| URL | Backend |
|-----|---------|
| `postgresql://user:pass@host:5432/db` | PostgreSQL (Npgsql) |
| `postgresql+psycopg2://user:pass@host/db` | PostgreSQL (the `+driver` suffix is accepted and ignored) |
| `mysql://user:pass@host:3306/db` | MySQL (Pomelo) |
| `mysql+pymysql://user:pass@host/db` | MySQL (the `+driver` suffix is accepted and ignored) |

SQLite URLs (`sqlite:///…`) are rejected — use the `OptunaSharp.Storage.Sqlite` package for those.

## Configuration

`EFCoreRdbOptions` carries Optuna's `RDBStorage` knobs plus a connection-string escape hatch. Every
property is opt-in.

```csharp
using var storage = new RDBStorage("postgresql://user:pass@localhost:5432/optuna", new EFCoreRdbOptions
{
    SkipCompatibilityCheck = false,
    SkipTableCreation      = false,
    HeartbeatInterval      = 60,          // experimental
    GracePeriod            = 120,
    // any provider keyword the URL doesn't carry (SSL, pooling, timeouts, …):
    ConfigureConnectionString = b => b["SSL Mode"] = "Require",
});
```

Or bring your own native connection string — the "feel like the client, not Optuna" form:

```csharp
new RDBStorage(RdbProvider.PostgreSql,
    "Host=localhost;Port=5432;Username=u;Password=p;Database=optuna;Pooling=true;Maximum Pool Size=50");
```

## Cross-language interoperability

The schema this package creates is **byte-for-byte** what SQLAlchemy's `metadata.create_all` emits for
each dialect — native `ENUM` types (`studydirection`, `trialstate`, `trialvaluetype`,
`trialintermediatevaluetype`) + `SERIAL` + `TIMESTAMP WITHOUT TIME ZONE` on PostgreSQL, inline
`ENUM(...)` + `AUTO_INCREMENT` + `DATETIME` on MySQL — so a database created here can be reopened by
Python Optuna, and vice versa:

```python
import optuna
study = optuna.load_study(study_name="quadratic",
                          storage="postgresql://user:pass@localhost:5432/optuna")
print(study.best_value, study.best_params)
```

Every table is created with `IF NOT EXISTS` (PostgreSQL enum types are guarded by a catalog check), so
opening a database Python already created is a no-op. Datetimes are stored as naive UTC,
objective/intermediate `±inf`/`NaN` as a typed enum tag with a `NULL` value, distributions use Optuna's
`distribution_json`, and attributes use `json.dumps` — all byte-compatible with Optuna.

> **Concurrency.** Unlike SQLite, these servers support row-level locking, so the read-modify-write
> paths take a `SELECT … FOR UPDATE` lock exactly as Optuna does, and duplicate-key races are ignored
> the same way (`ignore_integrity_error`). A short-lived `DbContext` is used per operation.

## License

MIT — see the repository root.
