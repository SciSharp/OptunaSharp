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

## Configuration — the full SQLite surface

Beyond Optuna's `RDBStorage(url, …)` API, `SqliteRdbOptions` exposes the **entire** SQLite
configuration surface — every `SqliteConnectionStringBuilder` option and the load-bearing runtime
`PRAGMA`s — with two escape hatches for anything not modelled. Every property is opt-in; a `null`
leaves SQLite's default in place (the exceptions are `BusyTimeoutMs`, which defaults to `30000`, and
`ForeignKeys`, which defaults to `false` to match Optuna).

```csharp
using var storage = new RDBStorage("sqlite:///study.db", new SqliteRdbOptions
{
    // PRAGMAs
    JournalMode  = SqliteJournalMode.Wal,      // better read/write concurrency
    Synchronous  = SqliteSynchronous.Normal,
    BusyTimeoutMs = 15_000,
    CacheSize    = -16_000,                     // 16 MiB page cache
    MmapSize     = 256L * 1024 * 1024,
    TempStore    = SqliteTempStore.Memory,
    ForeignKeys  = true,                        // safe with the explicit cascade in DeleteStudy
    AdditionalPragmas = new[] { "wal_autocheckpoint=1000" },

    // connection string
    Cache   = SqliteCacheMode.Shared,
    Pooling = true,

    // escape hatches
    ConfigureConnectionString = b => b.DefaultTimeout = 30,
    ConfigureConnection       = c => { /* register a collation, run setup SQL, … */ },
});
```

The five constructors cover every way to supply the connection:

```csharp
new RDBStorage("sqlite:///study.db");                              // Optuna-faithful
new RDBStorage("sqlite:///study.db", b => b.Cache = SqliteCacheMode.Shared);  // URL + native tweak
new RDBStorage("sqlite:///study.db", options);                     // URL + full options
new RDBStorage(new SqliteConnectionStringBuilder { DataSource = "study.db" }, options);  // native builder
new RDBStorage(myOpenSqliteConnection, options, ownsConnection: false);       // bring your own connection
```

With an injected `SqliteConnection` the storage feels like the client, not Optuna: its
connection-string is used as-is, `ownsConnection: false` (the default) keeps the lifetime yours, and
the `PRAGMA`/escape-hatch options still apply.

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
