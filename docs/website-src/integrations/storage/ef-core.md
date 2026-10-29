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

| URL scheme | .NET provider | Arguments read as |
|---|---|---|
| `postgresql://`, `postgresql+psycopg2://` | Npgsql | libpq connection parameters, as psycopg2 passes them |
| `postgresql+pg8000://` | Npgsql | pg8000's `connect` arguments |
| `mysql://`, `mariadb://`, `mysql+mysqldb://` | Pomelo with MySqlConnector | MySQLdb (`mysqlclient`) `connect` arguments |
| `mysql+pymysql://` | Pomelo with MySqlConnector | PyMySQL `Connection` arguments |
| `mysql+mysqlconnector://`, `mysql+mariadbconnector://`, `mysql+cymysql://` | Pomelo with MySqlConnector | the common arguments (charset, timeouts, socket, SSL files) |

A URL is read exactly as Python Optuna reads it: Optuna's `{SCHEMA_VERSION}` template, SQLAlchemy's URL grammar and driver registry, the driver's `create_connect_args`, then the driver's own handling of those arguments, expressed with Npgsql or MySqlConnector. Credentials and database names are URL-decoded. The query string carries the Python driver's arguments, not .NET connection-string keywords. `?sslmode=require&connect_timeout=7&application_name=optuna` works for PostgreSQL, and so does a repeated `?host=` (SQLAlchemy's multi-host form). `?ssl_ca=/path/ca.pem` works for PyMySQL. A native keyword such as `?SSL Mode=Require` or `?Pooling=false` fails as libpq fails, with psycopg2's `invalid dsn: …` message as a `FormatException`. Set native keywords with `ConfigureConnectionString` instead.

What Python rejects is rejected with Python's error kind and message:

- `postgres://` fails with `Can't load plugin: sqlalchemy.dialects:postgres`. SQLAlchemy dropped that name.
- An unknown driver suffix fails the same way.
- A bad libpq value fails with libpq's message, for example `invalid sslmode value: "bogus"`. libpq's checks run in libpq 18's order, and an explicit empty value (`PGSSLMODE` exported empty, `sslmode=''` in a dsn) is invalid, as in libpq.
- A `?port=` reaches PyMySQL as text, so it fails with `port should be of type int`.
- An unknown MySQL argument fails with the driver's `TypeError`. For `mysql://` (MySQLdb), the arguments go through CPython's own argument parsing first: a repeated key is a tuple (`connect() argument 16 must be str, not tuple`), an integer beyond a C `int` raises the platform's `OverflowError`, and `ssl_mode` must be one of the five upper-case names (`Unknown ssl_mode specification` otherwise).
- PyMySQL's own failures on text values carry PyMySQL's messages: `?ssl=…` (`'str' object has no attribute 'get'`), an unknown `charset` (`'NoneType' object has no attribute 'encoding'`, while `UTF8MB4` and `Utf8` work), `?conv=…`, a password outside Latin-1 (`UnicodeEncodeError`, as PyMySQL encodes it as Latin-1), and a missing `ssl_ca` or `ssl_cert` file (`FileNotFoundException`, Python's `FileNotFoundError`).

PostgreSQL connection strings follow libpq end to end:

- **Connection string arguments.** A `?dsn=` (psycopg2) or `?conninfo=` (psycopg) may use libpq's `keyword=value` form or its URI form (`postgresql://user@host:port,host2/db?sslmode=require`). It is merged with the URL's other arguments the way the driver merges them.
- **Service files.** `?service=name`, or `PGSERVICE`, reads that group from `PGSERVICEFILE` or `~/.pg_service.conf` (`%APPDATA%/postgresql/.pg_service.conf` on Windows), then from `$PGSYSCONFDIR/pg_service.conf`. The group fills in what the URL leaves out, and libpq's errors apply to a missing group or a malformed file. libpq's compiled-in last resort, the `SYSCONFDIR` of each libpq build, is not read: set `PGSYSCONFDIR` to use a system-wide file.
- **Defaults.** `PGREQUIRESSL=1` means `sslmode=require`, and `sslrootcert=system` without an `sslmode` means `verify-full`.
- **Validation before refusal.** `require_auth`, the SCRAM keys, the TLS and protocol version ranges and `client_encoding` are validated as libpq validates them (every UTF-8 spelling of `client_encoding` is accepted). A NUL in a value is psycopg2's `embedded null character`.

An asyncio driver (`+asyncpg`, `+psycopg_async`, `+aiomysql`, `+asyncmy`) fails on SQLAlchemy's synchronous-engine greenlet error. Settings MySqlConnector or Npgsql cannot express raise `NotSupportedException` naming the argument, but only after the Python driver's own checks have passed, so a URL Python rejects fails with Python's error. Examples:

- PostgreSQL: `hostaddr`, `require_auth` (Npgsql cannot restrict authentication methods), the SCRAM keys, `keepalives_count`, `tcp_user_timeout`, a minimum protocol above 3.0.
- MySQL: a character set other than UTF-8 (MySqlConnector always talks `utf8mb4`), `init_command`, `sql_mode`, `client_flag` bits that change how SQL parses, an option file PyMySQL would read (a missing one is skipped, as `configparser` skips it).

A PyMySQL password with Latin-1 characters outside ASCII is sent as UTF-8, where PyMySQL sends Latin-1. SQLite URLs are rejected: use [the SQLite package](sqlite.md).

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

The storage is also `IAsyncDisposable`. `DisposeAsync()` closes an owned data source through Npgsql's asynchronous close, without holding a thread, and leaves the same things open as `Dispose()`. Use `await using` in async code. A DI scope disposed asynchronously, such as an ASP.NET Core request scope, calls it for you. Either method can run first; the second does nothing.

## Concurrency, heartbeats, and schema lifecycle

Each operation creates a short-lived `DbContext`. Trial-update paths use transactions and `SELECT ... FOR UPDATE`, and the implementation handles relevant duplicate-key races. It does not share one `DbContext` between workers.

Trial creation locks the study's row (`SELECT ... FOR UPDATE`) before it inserts the trial and counts the earlier ones for its number, as Optuna's `_create_new_trial` does. Concurrent creators, whether threads of one `Optimize(nJobs: …)`, separate storages or separate processes, therefore number their trials one after another. A deadlock or lock timeout retries the creation up to five times after a random sleep of up to two seconds, as Optuna does; other database errors fail at once. Until 2026-10-28 the lock was missing: eight workers left 73 distinct trials of 80 on PostgreSQL and 71 on MySQL, and eight separate storages 64 and 33, because creators took the same number. The storage suite now checks these topologies on both servers (`StorageConcurrencyTests`): one storage under `nJobs: 8`, eight separate storages on one study, eight storages popping enqueued trials, and sixteen storages creating one study name.

Many storages may open a fresh database at once, as a fleet of workers starting together does. On PostgreSQL the schema DDL runs in one transaction holding an advisory lock, so the later sessions wait and then find the schema in place. MySQL's `CREATE TABLE IF NOT EXISTS` is already atomic there. Optuna 5.0 fails this case on both servers: 15 of 16 storages raised a duplicate `pg_type` key on PostgreSQL and "Table 'studies' already exists" on MySQL.

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
