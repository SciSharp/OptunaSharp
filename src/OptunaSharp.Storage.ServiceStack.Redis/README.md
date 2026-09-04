# OptunaSharp.Storage.ServiceStack.Redis

Redis journal-log backend for [**OptunaSharp**](https://github.com/SciSharp/OptunaSharp) on
[**ServiceStack.Redis**](https://github.com/ServiceStack/ServiceStack.Redis). Part of the
[SciSharp STACK](https://github.com/SciSharp).

`ServiceStackJournalRedisBackend` persists an Optuna study's operation log to a Redis server. It is
byte-compatible with Python Optuna's `JournalRedisBackend` and with the sibling
`StackExchange`/`NRedisStack` backends — a study written by any of them can be read by the others over
the same Redis.

## ⚠️ License

**ServiceStack.Redis is commercially licensed.** Its free tier is **request-quota-limited** (and
requires accepting ServiceStack's license terms). Unlike the MIT
`OptunaSharp.Storage.StackExchange.Redis` and `OptunaSharp.Storage.NRedisStack` backends, real
production use of this package needs a [ServiceStack license](https://servicestack.net). Choose this
package only if your application already standardizes on ServiceStack.Redis.

## Usage

```csharp
using OptunaSharp.Storages;
using OptunaSharp.Storages.Journal;

var backend = new ServiceStackJournalRedisBackend("redis://localhost:6379");
var study = Optuna.CreateStudy(storage: new JournalStorage(backend));
```

`redis://` and `rediss://` (TLS) URLs are accepted (unix sockets are not supported by
ServiceStack.Redis).

## Native ServiceStack.Redis configuration

The URL constructor mirrors Python Optuna, but you are not limited to it — drive ServiceStack.Redis on
its own terms. Four constructors, all taking the same `useCluster` / `prefix`:

```csharp
using ServiceStack.Redis;

// 1. redis-py-style URL (Optuna-identical)
new ServiceStackJournalRedisBackend("redis://localhost:6379");

// 2. keep the URL, tweak the derived ServiceStack RedisEndpoint (routing/auth: Ssl, Db, Password, …)
new ServiceStackJournalRedisBackend("redis://localhost:6379", ep => { ep.Ssl = true; ep.Db = 1; });

// 3. a ServiceStack RedisEndpoint you built yourself
new ServiceStackJournalRedisBackend(new RedisEndpoint("localhost", 6379) { Ssl = true, Db = 1 });

// 4. an IRedisClientsManager you already own — any manager type + pool config, shared / DI-managed
//    (you keep its lifetime). This is the full-control path with no configuration caveats.
new ServiceStackJournalRedisBackend(new RedisManagerPool("localhost:6379"));
```

The endpoint/callback constructors build a `RedisManagerPool` from the endpoint's connection string,
which reliably carries routing and auth (host, port, `Ssl`, `Db`, `Password`, `Username`). Two
ServiceStack.Redis specifics: timeouts/retry are set on the process-wide `RedisConfig` (its pooled
managers do not apply a per-endpoint `ConnectTimeout`), and do **not** set the endpoint's
`NamespacePrefix` — use OptunaSharp's `prefix` parameter instead. For anything beyond routing/auth,
inject a fully-built `IRedisClientsManager`.

The on-wire journal format is fixed regardless of how you connect, so the study stays byte-compatible
with Python Optuna and the sibling backends whichever constructor you use.

## License

OptunaSharp is MIT © SciSharp STACK Team — **but this package depends on the commercial
ServiceStack.Redis** (see above).
