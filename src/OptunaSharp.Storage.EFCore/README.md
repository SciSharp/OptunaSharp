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
| `postgresql://user:pass@host:5432/db` | PostgreSQL (Npgsql); query = libpq parameters, as psycopg2 passes them |
| `postgresql+psycopg2://user:pass@host/db?sslmode=require` | PostgreSQL, the same |
| `postgresql+pg8000://user:pass@host/db` | PostgreSQL; query = pg8000's arguments |
| `mysql://user:pass@host:3306/db` | MySQL (Pomelo); query = MySQLdb's arguments |
| `mysql+pymysql://user:pass@host/db?ssl_ca=/ca.pem` | MySQL; query = PyMySQL's arguments |

The URL is read exactly as Python Optuna reads it: Optuna's `{SCHEMA_VERSION}` template, SQLAlchemy's URL
grammar and driver registry, the driver's `create_connect_args`, then the Python driver's own handling of
those arguments, with Python's error kinds and messages (`postgres://` is refused as SQLAlchemy refuses it;
a native .NET keyword such as `?Pooling=false` fails as libpq fails — set those with
`ConfigureConnectionString`). libpq's `pg_service.conf` service files, the URI form of `?dsn=`, and libpq's
validation of `require_auth`, the SCRAM keys and the TLS and protocol ranges work as in Python, and so do
MySQLdb's and PyMySQL's own argument checks. What MySqlConnector or Npgsql cannot express raises
`NotSupportedException`, but only after the Python driver's checks have passed. SQLite URLs (`sqlite:///…`) are
rejected — use the `OptunaSharp.Storage.Sqlite` package for those.

## Configuration — Optuna's API *and* the native EF Core / Npgsql / Pomelo surface

`EFCoreRdbOptions` carries Optuna's `RDBStorage` knobs *and* the full EF Core surface, so the storage
feels like the client you already know — not just like Optuna. Every property is opt-in.

```csharp
using var storage = new RDBStorage("postgresql://user:pass@localhost:5432/optuna", new EFCoreRdbOptions
{
    // Optuna's RDBStorage knobs
    SkipCompatibilityCheck = false,
    SkipTableCreation      = false,
    HeartbeatInterval      = 60,          // experimental
    GracePeriod            = 120,

    // any provider connection-string keyword the URL doesn't carry (SSL, pooling, timeouts, …):
    ConfigureConnectionString = b => ((NpgsqlConnectionStringBuilder)b).SslMode = SslMode.Require,

    // the native EF Core provider surface — resilient connections, command timeout, …:
    ConfigureNpgsql = o => o.EnableRetryOnFailure().CommandTimeout(30),
    ConfigureMySql  = o => o.EnableRetryOnFailure(),   // ignored on PostgreSQL
    MySqlServerVersion = new MySqlServerVersion(new Version(8, 0, 36)), // skip the auto-detect probe

    // the general EF Core surface — logging, interceptors, …:
    ConfigureDbContext = b => b.LogTo(Console.WriteLine).EnableSensitiveDataLogging(),
});
```

`EnableRetryOnFailure()` is safe: the storage runs its transactions inside EF Core's execution
strategy, so a retrying strategy never trips the "does not support user-initiated transactions" guard.

### Ways to construct — pick the one that feels most like your stack

```csharp
// 1) Optuna-faithful URL
new RDBStorage("postgresql://user:pass@host:5432/optuna");
new RDBStorage("mysql://user:pass@host:3306/optuna");

// 2) explicit provider + native ADO.NET connection string
new RDBStorage(RdbProvider.PostgreSql, "Host=host;Port=5432;Username=u;Password=p;Database=optuna;Maximum Pool Size=50");

// 3) a native connection-string builder — every Npgsql / MySqlConnector knob is yours
new RDBStorage(new NpgsqlConnectionStringBuilder { Host = "host", Database = "optuna", Pooling = true });
new RDBStorage(new MySqlConnectionStringBuilder { Server = "host", Database = "optuna" });

// 4) your own NpgsqlDataSource — full control of pooling, logging, type mappings, …
var dsb = new NpgsqlDataSourceBuilder(connString);
RDBStorage.MapOptunaEnums(dsb);         // required: maps Optuna's four native enum types
dsb.UseLoggerFactory(myLoggerFactory);
using var storage = new RDBStorage(dsb.Build(), ownsDataSource: true);
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
