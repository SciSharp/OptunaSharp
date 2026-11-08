# ServiceStack.Redis

`OptunaSharp.Storage.ServiceStack.Redis` provides `ServiceStackJournalRedisBackend` in `OptunaSharp.Storages.Journal`. It uses a ServiceStack `IRedisClientsManager` and borrows clients for journal commands. This adapter connects an existing ServiceStack-based application to the same [journal format](index.md) as the other Redis adapters.

## Install and use

Requires .NET 8 or .NET 10 and a reachable Redis server. The package's [dependency notice](https://github.com/SciSharp/OptunaSharp/blob/main/src/OptunaSharp.Storage.ServiceStack.Redis/README.md) identifies ServiceStack.Redis as commercially licensed with a request-limited free tier. Review the dependency's applicable terms for your application; OptunaSharp's own license does not replace them.

```bash
dotnet add package OptunaSharp.Storage.ServiceStack.Redis
```

```csharp
using OptunaSharp;
using OptunaSharp.Storages.Journal;
using ServiceStack.Redis;

using var manager = new RedisManagerPool("localhost:6379");
var backend = new ServiceStackJournalRedisBackend(manager, prefix: "experiments");
var study = Optuna.CreateStudy(
    storage: new JournalStorage(backend),
    studyName: "quadratic", loadIfExists: true);
study.Optimize(trial => Math.Pow(trial.SuggestFloat("x", -10, 10), 2), nTrials: 20);
```

The application owns the manager. The adapter returns borrowed `IRedisClient` instances after command use. Disposing the adapter does not dispose an injected manager. Keep the manager alive for the entire study workload.

## Constructors and endpoint settings

| First arguments | Behavior |
|---|---|
| `string url` | Parse the Redis URL and construct a `RedisManagerPool` |
| `string url, Action<ServiceStack.Redis.RedisEndpoint> configure` | Modify the native endpoint before pool creation |
| `ServiceStack.Redis.RedisEndpoint endpoint` | Construct a pool from the native endpoint |
| `IRedisClientsManager manager` | Borrow clients from your existing manager |

All overloads also accept `useCluster = false` and `prefix = ""`. `redis://` and `rediss://` URLs are supported. `unix://` throws `ValueError` because this adapter does not support Unix-domain sockets.

The native endpoint is distinct from `OptunaSharp.Storages.Journal.RedisEndpoint`, the neutral URL-parser result. Qualify or alias the ServiceStack type if both namespaces are imported:

```csharp
using OptunaSharp.Storages.Journal;
using ServiceStackEndpoint = ServiceStack.Redis.RedisEndpoint;

var backend = new ServiceStackJournalRedisBackend(
    new ServiceStackEndpoint("localhost", 6379) { Db = 1 },
    prefix: "experiments");
```

The endpoint and URL paths build a `RedisManagerPool` whose resolver serves the endpoint object itself, so every pooled client copies its host, port, TLS, database, username, password, client name and connect/send/receive timeouts. A URL's timeouts default to redis-py's 5 seconds. These paths do not expose a manager-configuration callback. Use an injected manager for pool selection, pool sizing, or application-managed connection behavior.

ServiceStack.Redis takes certificate callbacks only process-wide (`RedisConfig.CertificateValidationCallback`), speaks RESP2 only, authenticates only when a password is set, and turns revocation checking on whenever TLS versions are given. So a URL's `ssl_cert_reqs=none`, `ssl_check_hostname=false`, extra certificate authorities, client certificate, `ssl_min_version`, `protocol=3`, or a user name without a password raises `NotSupportedException`. Use the StackExchange or NRedisStack backend for those, or configure ServiceStack yourself. Leave the native endpoint's `NamespacePrefix` unset and use OptunaSharp's `prefix` argument to preserve the shared journal keys.

As with the other adapters, the backend is `IDisposable`. The URL and endpoint convenience constructors create an internal pool that the backend owns, and `Dispose()` closes it. An injected manager is never disposed by the backend.

## Commands, compatibility, and limits

The adapter uses `GetValue`, `SetValue`, `SetValueIfNotExists`, and `IncrementValue` for the raw journal strings. On standalone Redis, reads and appends use `ExecLua` with the shared scripts: one request per batch of up to 1,000 records and one per appended record, which also stretches a request-metered licence. Cluster mode allocates the counter and writes the record separately, and reads key by key. Snapshots use native byte `Get`/`Set` under `{prefix}:snapshot:optunasharp`, avoiding ServiceStack typed-object serialization.

`useCluster` selects that append algorithm; it does not provision a cluster or configure the manager's routing. Read the [Redis overview](index.md) for missing-record waits, prefix isolation, and the difference between interoperable logs and runtime-specific snapshots.

Implementation: [ServiceStackJournalRedisBackend](https://github.com/SciSharp/OptunaSharp/blob/main/src/OptunaSharp.Storage.ServiceStack.Redis/Storages/Journal/ServiceStackJournalRedisBackend.cs). Verification covers [pool/endpoint/callback constructors](https://github.com/SciSharp/OptunaSharp/blob/main/test/OptunaSharp.Tests.Storage/RedisBackendTests.cs), [cross-library bytes](https://github.com/SciSharp/OptunaSharp/blob/main/test/OptunaSharp.Tests.Storage/CrossLibraryParityTests.cs), and [Python read/write interoperability](https://github.com/SciSharp/OptunaSharp/blob/main/test/OptunaSharp.Tests.Interop/JournalRedisInteropTests.cs).
