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

The constructor is `ServiceStackJournalRedisBackend(url, useCluster = false, prefix = "")`.
`redis://` and `rediss://` (TLS) URLs are accepted; **`unix://` sockets are not supported** by
ServiceStack.Redis. See the [Redis overview](index.md) for the URL format, cluster mode, and key
prefixes.

## Byte-compatibility

Logs are stored through ServiceStack.Redis' pooled `IRedisClient` (`GetValue` / `SetValue` /
`SetValueIfNotExists` / `IncrementValue`, and native `Eval` for the atomic append) using the shared
key layout and Optuna's `json.dumps` encoding — so a study is interchangeable with Python Optuna and
with the [StackExchange.Redis](stackexchange-redis.md) and [NRedisStack](nredisstack.md) backends.

## License

OptunaSharp itself is MIT, but **this package depends on the commercial ServiceStack.Redis** (see the
warning above).
