# PostgreSQL and MySQL through EF Core

`OptunaSharp.Storage.EFCore` implements `OptunaSharp.Storages.EFCore.RDBStorage` for PostgreSQL and MySQL. It uses Npgsql and Pomelo/MySqlConnector through Entity Framework Core, mapping Optuna's tables rather than introducing a separate application schema. Both database providers ship as dependencies of the package.

## Install and connect

Requires .NET 8 or .NET 10, a reachable PostgreSQL or MySQL server, an existing database, and credentials allowed to read/write its Optuna tables. Initial setup also needs permission to create tables and, for PostgreSQL, enum types.

```bash
dotnet add package OptunaSharp.Storage.EFCore
```

```csharp
using OptunaSharp;
using RdbStorage = OptunaSharp.Storages.EFCore.RDBStorage;

using var storage = new RdbStorage(
    "postgresql://user:password@localhost:5432/optuna");
// MySQL: new RdbStorage("mysql://user:password@localhost:3306/optuna")

var study = Optuna.CreateStudy(
    storage: storage, studyName: "quadratic", loadIfExists: true);
study.Optimize(trial =>
{
    double x = trial.SuggestFloat("x", -10, 10);
    return (x - 2) * (x - 2);
}, nTrials: 20);
```

The alias makes the EF Core type unambiguous when the SQLite package is also referenced. Study creation wraps this storage in `CachedStorage` automatically.

## URL and provider selection

| URL scheme | .NET provider |
|---|---|
| `postgresql://`, `postgres://` | Npgsql |
| `postgresql+psycopg2://` | Npgsql; the Python driver suffix is accepted and ignored |
| `mysql://`, `mariadb://` | Pomelo with MySqlConnector |
| `mysql+pymysql://` | Pomelo; the Python driver suffix is accepted and ignored |

SQLite URLs are rejected: use [the SQLite package](sqlite.md). Credentials and database names are URL-decoded. URL query pairs are passed to the native .NET connection-string builder, so they must be valid keywords for that provider; this is not arbitrary SQLAlchemy `engine_kwargs` translation.

MariaDB is routed through the MySQL provider, with provider version detection, but the repository's database interoperability fixtures target PostgreSQL 16 and MySQL 8.0. Those fixtures do not establish coverage of every MariaDB or server version.

## Native configuration

`EFCoreRdbOptions` carries storage options and callbacks for the native provider:

```csharp
using Microsoft.EntityFrameworkCore;
using Npgsql;
using OptunaSharp.Storages.EFCore;

var connection = new NpgsqlConnectionStringBuilder
{
    Host = "localhost",
    Port = 5432,
    Database = "optuna",
    Username = "user",
    Password = "password",
    Pooling = true,
    MaxPoolSize = 50,
};

using var storage = new RDBStorage(connection, new EFCoreRdbOptions
{
    ConfigureNpgsql = options => options.EnableRetryOnFailure().CommandTimeout(30),
    ConfigureDbContext = options => options.LogTo(Console.WriteLine),
});
```

Available construction paths are a SQLAlchemy-style URL, explicit `RdbProvider` plus native connection string, `NpgsqlConnectionStringBuilder`, `MySqlConnectionStringBuilder`, and an existing `NpgsqlDataSource`.

`ConfigureConnectionString` receives a native `DbConnectionStringBuilder` before connection creation. `ConfigureNpgsql` or `ConfigureMySql` runs inside the matching provider setup, followed by `ConfigureDbContext`. Only the callback for the selected provider is applied.

For MySQL, `MySqlServerVersion` avoids the extra connection used by `ServerVersion.AutoDetect`. Use the version corresponding to your deployment, for example:

```csharp
using System;
using Microsoft.EntityFrameworkCore;
using OptunaSharp.Storages.EFCore;

using var storage = new RDBStorage(
    "mysql://user:password@localhost:3306/optuna",
    new EFCoreRdbOptions
    {
        MySqlServerVersion = new MySqlServerVersion(new Version(8, 0, 36)),
        ConfigureMySql = options => options.EnableRetryOnFailure(),
    });
```

Mutating operations execute their transactions inside EF Core's configured execution strategy. This supports enabling provider retry-on-failure without the user-initiated transaction error. It does not promise exactly-once completion after an ambiguous network failure.

## PostgreSQL data-source ownership

When using a data source your application owns, map Optuna's four native enum types before building it:

```csharp
using Npgsql;
using RdbStorage = OptunaSharp.Storages.EFCore.RDBStorage;

var builder = new NpgsqlDataSourceBuilder(
    "Host=localhost;Database=optuna;Username=user;Password=password");
RdbStorage.MapOptunaEnums(builder);
using var dataSource = builder.Build();
using var storage = new RdbStorage(dataSource);
```

The injected data source is caller-owned by default; `ownsDataSource: true` transfers disposal to the storage. Because the data source is already built, `ConfigureConnectionString` cannot reconfigure it. The other constructors create and own the PostgreSQL data source. `Dispose()` releases owned PostgreSQL pooled connections; MySQL contexts use provider-managed connections per operation.

## Concurrency, heartbeats, and schema lifecycle

Each operation creates a short-lived `DbContext`. Trial-update paths use transactions and `SELECT ... FOR UPDATE`, and the implementation handles relevant duplicate-key races. It does not share one `DbContext` between workers.

There is a concurrency limitation in the current trial-creation path: it checks that the study exists, inserts the trial, and derives its number by counting earlier trial IDs, without locking the study row. Concurrent creators can therefore race when assigning per-study trial numbers. The existing database suite does not establish safe simultaneous trial creation. Serialize creation for a shared study until that path has been hardened and verified; server database support alone is not a blanket parallel-worker guarantee.

`HeartbeatInterval`, `GracePeriod`, and `HeartbeatStaleTrialCallback` enable experimental stale-trial handling with `Study.Optimize`. The default grace period is twice the heartbeat interval. `FailedTrialCallback` is the deprecated alias and cannot be supplied together with the current callback. Heartbeat tests cover stale detection and retry callbacks. Heartbeats are not forwarded by the [gRPC proxy](grpc/index.md).

This package targets `schema_version = 12` and Alembic revision `v3.2.0.a`. It creates missing tables and verifies the revision; PostgreSQL enum creation is guarded by catalog checks. `SkipTableCreation` assumes an existing schema. `SkipCompatibilityCheck` only disables revision validation.

Unlike the SQLite implementation, this package has no public `Upgrade()` migration runner. For a known older database, its compatibility error directs users to Python Optuna's `optuna storage upgrade --storage ...` command. Coordinate upgrades with database backups and stopped workers; do not treat EF Core `Migrate()` or bypassing the check as an Optuna schema migration.

## Python interoperability

Python applications require the corresponding Python database driver as well as Optuna. The same database and study name can be used from Python:

```python
import optuna

study = optuna.load_study(
    study_name="quadratic",
    storage="postgresql://user:password@localhost:5432/optuna")
print(study.best_value)
```

PostgreSQL uses Optuna's native enum types, generated IDs, and timestamps without time zone; MySQL uses inline enum types, auto-increment IDs, and datetimes. Stored datetimes are naive UTC. Distributions and attributes use Optuna-compatible JSON, while infinite objective values and nonfinite intermediate values use the schema's value tags.

The interop suite covers both directions against real PostgreSQL/MySQL servers. Those tests require their database containers and Python dependencies; the presence of a test is not a claim that an arbitrary deployment has been exercised.

## Implementation and verification

- [Storage implementation](https://github.com/SciSharp/OptunaSharp/blob/master/src/OptunaSharp.Storage.EFCore/Storages/Rdb/RDBStorage.cs), [options](https://github.com/SciSharp/OptunaSharp/blob/master/src/OptunaSharp.Storage.EFCore/Storages/Rdb/EFCoreRdbOptions.cs), and [schema](https://github.com/SciSharp/OptunaSharp/blob/master/src/OptunaSharp.Storage.EFCore/Storages/Rdb/RdbSchema.cs)
- [Database, configuration, and heartbeat tests](https://github.com/SciSharp/OptunaSharp/blob/master/test/OptunaSharp.Tests.Storage/EFCoreRdbStorageTests.cs)
- [URL parsing tests](https://github.com/SciSharp/OptunaSharp/blob/master/test/OptunaSharp.Tests.Storage/EFCoreRdbUrlTests.cs)
- [Python/PostgreSQL/MySQL interoperability](https://github.com/SciSharp/OptunaSharp/blob/master/test/OptunaSharp.Tests.Interop/EFCoreRdbInteropParityTests.cs)
