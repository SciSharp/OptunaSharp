# Storage integrations

Storage integrations connect OptunaSharp studies to SQLite, PostgreSQL, MySQL, Redis, or a remote gRPC storage server. They persist trial parameters, values, intermediate reports, and study metadata. Model files and other large outputs belong in an [artifact store](../artifacts/index.md).

## Choose the storage connection

| Integration | Package | Data location | Connection model |
|---|---|---|---|
| [SQLite](sqlite.md) | `OptunaSharp.Storage.Sqlite` | Local SQLite database | One connection, serialized operations |
| [PostgreSQL / MySQL through EF Core](ef-core.md) | `OptunaSharp.Storage.EFCore` | Server database using Optuna's schema | Short-lived contexts and trial-update row locks; trial-creation concurrency limitation |
| [StackExchange.Redis](redis/stackexchange-redis.md) | `OptunaSharp.Storage.StackExchange.Redis` | Redis journal keys | StackExchange.Redis multiplexer |
| [NRedisStack](redis/nredisstack.md) | `OptunaSharp.Storage.NRedisStack` | The same Redis journal format | NRedisStack's underlying multiplexer |
| [ServiceStack.Redis](redis/servicestack-redis.md) | `OptunaSharp.Storage.ServiceStack.Redis` | The same Redis journal format | ServiceStack client manager / pool |
| [Redis shared primitives](redis/index.md#shared-package) | `OptunaSharp.Storage.Redis` | No standalone storage | URL parser, key layout, and Lua script shared by the three adapters |
| [gRPC](grpc/index.md) | `OptunaSharp.Storage.Grpc` | A server's backing storage | Managed gRPC client and ASP.NET Core server |

These packages target .NET 8 and .NET 10. Install the integration package you use; its project references bring in the core `OptunaSharp` package. In-memory storage and the file journal are already part of core; see the [storage guide](../../docs/storage.md).

SQLite is convenient for local persistent studies. PostgreSQL and MySQL support transactional server storage, although the current EF Core trial-creation path has a [concurrency limitation](ef-core.md#concurrency-heartbeats-and-schema-lifecycle). Redis provides an operation journal that each worker replays locally. gRPC centralizes access to a backing storage through a server; it does not replace that storage's durability or concurrency behavior.

## Pass a storage object

Construct the integration's storage and pass it to `Optuna.CreateStudy` or `Optuna.LoadStudy`. Core's `StorageFunctions.GetStorage(string)` does not select installed database packages from a URL. The URL belongs in the integration constructor.

```csharp
using OptunaSharp;
using SqliteStorage = OptunaSharp.Storages.RDBStorage;

using var storage = new SqliteStorage("sqlite:///studies.db");
var study = Optuna.CreateStudy(
    storage: storage,
    studyName: "quadratic",
    loadIfExists: true);

study.Optimize(trial =>
{
    double x = trial.SuggestFloat("x", -10, 10);
    return (x - 2) * (x - 2);
}, nTrials: 20);
```

Both RDB implementations implement `ICachedStorageBackend`, so study creation wraps them in `CachedStorage` automatically. Redis backends must be wrapped in `JournalStorage`. `GrpcStorageProxy` already implements `BaseStorage` and maintains its own trial cache.

## Sharing data with Python

The RDB packages map Optuna's SQLAlchemy schema; Redis shares its journal operation format; gRPC shares its protobuf service contract. These are separate compatibility boundaries. Redis replay snapshots are runtime-specific, gRPC does not transport heartbeat operations, and schema upgrades differ between the two RDB packages. Each guide explains its boundary and links to the corresponding interoperability tests.

Storage shares study history, not the in-memory state of a sampler or a running objective. Configure each worker's sampler, pruner, and objective as part of the worker application.

Implementation: [storage selection and automatic cache wrapping](https://github.com/SciSharp/OptunaSharp/blob/master/src/OptunaSharp.Core/Storages/StorageFunctions.cs).
