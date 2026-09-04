# ServiceStack.Redis backend

`OptunaSharp.Storage.ServiceStack.Redis` is the Redis journal backend on
[**ServiceStack.Redis**](https://github.com/ServiceStack/ServiceStack.Redis). Choose it only when your
application already standardizes on ServiceStack.Redis.

> [!WARNING]
> **ServiceStack.Redis is commercially licensed.** Its free tier is request-quota-limited and requires
> accepting ServiceStack's license terms. Unlike the MIT [StackExchange.Redis](stackexchange-redis.md)
> and [NRedisStack](nredisstack.md) backends, production use of this package needs a
> [ServiceStack license](https://servicestack.net). If you do not already depend on ServiceStack,
> prefer one of the MIT backends.

## Install

```bash
dotnet add package OptunaSharp.Storage.ServiceStack.Redis
```

## Usage

```csharp
using OptunaSharp.Storages;
using OptunaSharp.Storages.Journal;

var backend = new ServiceStackJournalRedisBackend("redis://localhost:6379");
var study = Optuna.CreateStudy(storage: new JournalStorage(backend));

study.Optimize(trial =>
{
    var x = trial.SuggestFloat("x", -10, 10);
    return (x - 2) * (x - 2);
}, nTrials: 20);
```

The URL constructor is `ServiceStackJournalRedisBackend(url, useCluster = false, prefix = "")`.
`redis://` and `rediss://` (TLS) URLs are accepted; **`unix://` sockets are not supported** by
ServiceStack.Redis. See the [Redis overview](index.md) for the URL format, cluster mode, and key
prefixes.

## Native ServiceStack.Redis configuration

A `redis://` URL cannot express ServiceStack.Redis' endpoint settings or your choice of client
manager, so three more constructors let you use the client directly — all also taking `useCluster` and
`prefix`:

```csharp
using ServiceStack.Redis;

// FULL CONTROL — hand in any IRedisClientsManager you built: RedisManagerPool,
// PooledRedisClientManager, BasicRedisClientManager, with any pool config; shared / DI-managed.
// You keep its lifetime. This is the path with no configuration caveats.
new ServiceStackJournalRedisBackend(new RedisManagerPool("localhost:6379"));

// a ServiceStack RedisEndpoint you built yourself (routing/auth: host, port, Ssl, Db, Password, User)
new ServiceStackJournalRedisBackend(new RedisEndpoint("localhost", 6379) { Ssl = true, Db = 1 });

// keep the URL, tweak the derived RedisEndpoint before the pool is built
new ServiceStackJournalRedisBackend("redis://localhost:6379", ep => { ep.Ssl = true; ep.Db = 1; });
```

The `IRedisClientsManager` constructor is the full escape hatch: manager type, pool size and Redis
config are entirely yours. The stored journal bytes are the same whichever constructor you use, so
byte-compatibility with Python Optuna and the sibling backends is unaffected.

> [!NOTE]
> The endpoint / callback constructors build a `RedisManagerPool` from the endpoint's connection
> string ([`RedisEndpoint.ToString()`](https://reference.servicestack.net/api/ServiceStack.Redis/RedisEndpoint/)),
> which reliably carries **routing and auth** (host, port, `Ssl`, `Db`, `Password`, `Username`). Two
> caveats specific to ServiceStack.Redis:
>
> - **Timeouts / retry are configured globally, not per endpoint.** ServiceStack's pooled managers do
>   not apply the endpoint's `ConnectTimeout` (a
>   [known quirk](https://docs.servicestack.net/redis/client-managers)); set the process-wide
>   `RedisConfig` defaults instead (e.g. `RedisConfig.DefaultConnectTimeout`,
>   `RedisConfig.DefaultRetryTimeout`), or build and inject your own manager.
> - **Do not set the endpoint's `NamespacePrefix`** — use OptunaSharp's `prefix` parameter for key
>   namespacing. `NamespacePrefix` would risk diverging from the shared journal key layout and breaking
>   cross-backend / Python interoperability.

## Byte-compatibility

Logs are stored through ServiceStack.Redis' pooled `IRedisClient` (`GetValue` / `SetValue` /
`SetValueIfNotExists` / `IncrementValue`, and native `Eval` for the atomic append) using the shared
key layout and Optuna's `json.dumps` encoding — so a study is interchangeable with Python Optuna and
with the [StackExchange.Redis](stackexchange-redis.md) and [NRedisStack](nredisstack.md) backends.

## License

OptunaSharp itself is MIT, but **this package depends on the commercial ServiceStack.Redis** (see the
warning above).
