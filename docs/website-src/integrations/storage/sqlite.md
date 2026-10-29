# SQLite

`OptunaSharp.Storage.Sqlite` stores studies in a local SQLite file using `Microsoft.Data.Sqlite`. Its public storage type is `OptunaSharp.Storages.RDBStorage`. Tables, distribution JSON, attributes, and value tags follow Python Optuna's RDB representation, so compatible Python and .NET applications can reopen the same file.

## Install and persist a study

Requires .NET 8 or .NET 10 and a writable database location. No database server is needed.

```bash
dotnet add package OptunaSharp.Storage.Sqlite
```

```csharp
using OptunaSharp;
using OptunaSharp.Storages;

using var storage = new RDBStorage("sqlite:///study.db");
var study = Optuna.CreateStudy(
    storage: storage, studyName: "quadratic", loadIfExists: true);

study.Optimize(trial =>
{
    double x = trial.SuggestFloat("x", -10, 10);
    return (x - 2) * (x - 2);
}, nTrials: 20);

Console.WriteLine(study.BestValue);
```

`loadIfExists: true` reuses the named study when you rerun this program. `Optuna.CreateStudy` automatically wraps the RDB backend in `CachedStorage`; retain and dispose the original `RDBStorage` when work finishes.

| URL | Meaning |
|---|---|
| `sqlite:///study.db` | File relative to the process's working directory (made absolute with Python's `os.path.abspath` rules) |
| `sqlite:////data/study.db` | Absolute Unix path `/data/study.db` |
| `sqlite:///C:/data/study.db` | Absolute Windows path |
| `sqlite:///:memory:` or `sqlite://` | In-memory database kept alive by the storage's connection |
| `sqlite:///study.db?timeout=10` | `study.db` with a 10-second busy timeout |
| `sqlite:///file:study.db?mode=ro&uri=true` | SQLite URI filename: `study.db`, read-only |
| `sqlite:///study-v{SCHEMA_VERSION}.db` | Optuna's template, filled before parsing: `study-v12.db` |

The URL is read exactly as Python Optuna reads it: Optuna's `{SCHEMA_VERSION}` template, then SQLAlchemy's URL grammar, then SQLAlchemy's pysqlite dialect. The query string is never part of the file name. pysqlite reads six query arguments and ignores every other one. `timeout` (seconds) is the busy timeout. `uri=true` treats the path as a SQLite URI filename and appends the remaining arguments as URI parameters. `isolation_level`, `detect_types`, `check_same_thread` and `cached_statements` are type-checked as Python checks them and otherwise have no effect. A bad value fails with Python's message, for example `could not convert string to float: 'abc'`.

A bare path such as `study.db` is not a URL, so it fails with SQLAlchemy's `ArgumentError` text (`Could not parse SQLAlchemy URL from given URL string`), as an `ArgumentException`. A host in a SQLite URL (`sqlite://host/x.db`) fails pysqlite's `Invalid SQLite URL` check. Without `?timeout=`, the busy timeout is `SqliteRdbOptions.BusyTimeoutMs` (30 s by default, chosen for concurrent workers) instead of Python's 5 s. An explicitly assigned `BusyTimeoutMs` takes precedence over the URL's `?timeout=`, as SQLAlchemy's `connect_args` would.

## Configure SQLite

`SqliteRdbOptions` groups connection-string options, PRAGMAs, schema checks, and heartbeats. The URL-plus-callback overload is convenient when only the connection string needs adjustment. For fuller configuration:

```csharp
using OptunaSharp.Storages;

using var storage = new RDBStorage("sqlite:///study.db", new SqliteRdbOptions
{
    JournalMode = SqliteJournalMode.Wal,
    BusyTimeoutMs = 30_000,
    DefaultTimeoutSeconds = 30,
    ForeignKeys = true,
    CacheSize = -8192,
    ConfigureConnectionString = builder => builder.Pooling = true,
});
```

`BusyTimeoutMs` defaults to 30,000. `ForeignKeys` defaults to `false` to match the upstream storage behavior; application code performs study deletion's cascade explicitly. The other nullable settings normally leave provider defaults in place. WAL is opt-in and does not make SQLite a row-locking server database.

The options also expose `Mode`, `Cache`, `Password`, `RecursiveTriggers`, `Synchronous`, `PageSize`, `MmapSize`, `TempStore`, `AutoVacuum`, and `SecureDelete`. A password needs an encryption-capable SQLite provider; setting the property alone does not supply encryption. Page-size and auto-vacuum configuration belong on a fresh database.

`AdditionalPragmas` runs after the typed PRAGMAs. `ConfigureConnection` runs on the open connection after PRAGMAs and before schema initialization, allowing custom functions, collations, or setup SQL.

## Connection ownership and concurrency

The storage accepts a URL, a URL with an options/callback argument, a `SqliteConnectionStringBuilder`, or an existing `SqliteConnection`:

```csharp
using Microsoft.Data.Sqlite;
using OptunaSharp.Storages;

using var connection = new SqliteConnection("Data Source=study.db");
using var storage = new RDBStorage(connection, new SqliteRdbOptions
{
    BusyTimeoutMs = 30_000,
});
```

An injected connection is opened if necessary and remains caller-owned unless `ownsConnection: true` is passed. Its existing connection string is used as-is: typed connection-string options and `ConfigureConnectionString` do not rewrite it. PRAGMAs and `ConfigureConnection` still run. URL and builder constructors create a connection that the storage disposes.

The storage is `IDisposable` and `IAsyncDisposable`, so both `using` and `await using` work, and DI scopes can dispose it either way. Microsoft.Data.Sqlite closes a connection synchronously even through `DisposeAsync()`, so the async form frees no thread here. It exists so that every OptunaSharp storage can be disposed the same way.

One connection remains open for the storage's lifetime, and all operations through that instance are serialized by a lock. That lock does not coordinate other processes or other storage instances. SQLite's own locking still governs access to the file. For many competing workers, consider [PostgreSQL / MySQL](ef-core.md) or a [Redis journal](redis/index.md). A gRPC server in front of SQLite retains SQLite's write constraints.

## Schema compatibility and upgrades

This implementation targets `schema_version = 12` and Alembic revision `v3.2.0.a`. These are schema identifiers, not the OptunaSharp package version. Normal construction creates missing tables and checks the revision.

The SQLite implementation provides `GetCurrentVersion()`, `GetHeadVersion()`, `GetAllVersions()`, and `Upgrade()`. Its migration chain covers known revisions from `v0.9.0.a` through `v3.2.0.a`. For an existing older database, open without creating current tables before running the upgrade:

```csharp
using OptunaSharp.Storages;

using var storage = new RDBStorage(
    "sqlite:///study.db",
    skipCompatibilityCheck: true,
    skipTableCreation: true);
storage.Upgrade();
Console.WriteLine(storage.GetCurrentVersion());
```

Back up the file and stop its writers before applying migrations. DDL is not one atomic transaction; the revision is advanced after each completed migration. An unrecognized revision is rejected. `SkipCompatibilityCheck` bypasses a check, not a migration.

## Heartbeats and Python interoperability

`HeartbeatInterval`, `GracePeriod`, and `HeartbeatStaleTrialCallback` configure experimental stale-trial handling. Intervals and grace periods must be positive; omitted grace defaults to twice the heartbeat interval. `FailedTrialCallback` is a deprecated alias, and specifying both callback properties is an error. Heartbeat recording is integrated with `Study.Optimize`; a manual ask/tell loop needs its own heartbeat lifecycle.

Python can load the same named study from the same file location:

```python
import optuna

study = optuna.load_study(
    study_name="quadratic", storage="sqlite:///study.db")
print(study.best_value)
```

Datetimes use naive UTC; infinite objective values and nonfinite intermediate values use Optuna's tagged database representation. Schema and value interoperability have dedicated tests, including migrations of Python-created legacy files. A schema-compatible shared file does not imply identical sampler random state across runtimes.

## Implementation and verification

- [RDB storage implementation](https://github.com/SciSharp/OptunaSharp/blob/master/src/OptunaSharp.Storage.Sqlite/Storages/Rdb/RDBStorage.cs) and [all SQLite options](https://github.com/SciSharp/OptunaSharp/blob/master/src/OptunaSharp.Storage.Sqlite/Storages/Rdb/SqliteRdbOptions.cs)
- [Migration chain](https://github.com/SciSharp/OptunaSharp/blob/master/src/OptunaSharp.Storage.Sqlite/Storages/Rdb/RDBStorage.Migrations.cs)
- [Storage lifecycle tests](https://github.com/SciSharp/OptunaSharp/blob/master/test/OptunaSharp.Tests.Storage/SqliteRdbStorageTests.cs) and [options/ownership tests](https://github.com/SciSharp/OptunaSharp/blob/master/test/OptunaSharp.Tests.Storage/SqliteRdbOptionsTests.cs)
- [Python interoperability](https://github.com/SciSharp/OptunaSharp/blob/master/test/OptunaSharp.Tests.Interop/SqliteRdbInteropParityTests.cs) and [migration interoperability](https://github.com/SciSharp/OptunaSharp/blob/master/test/OptunaSharp.Tests.Interop/SqliteRdbUpgradeInteropParityTests.cs)
