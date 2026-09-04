# StackExchange.Redis backend

`OptunaSharp.Storage.StackExchange.Redis` is the Redis journal backend on
[**StackExchange.Redis**](https://github.com/StackExchange/StackExchange.Redis) — the
de-facto-standard .NET Redis client (MIT). If you have no particular preference, use this one.

## Install

```bash
dotnet add package OptunaSharp.Storage.StackExchange.Redis
```

## Usage

```csharp
using OptunaSharp.Storages;
using OptunaSharp.Storages.Journal;

var backend = new StackExchangeJournalRedisBackend("redis://localhost:6379");
var study = Optuna.CreateStudy(storage: new JournalStorage(backend));

study.Optimize(trial =>
{
    var x = trial.SuggestFloat("x", -10, 10);
    return (x - 2) * (x - 2);
}, nTrials: 20);
```

The constructor is `StackExchangeJournalRedisBackend(url, useCluster = false, prefix = "")`.
`redis://`, `rediss://` (TLS) and `unix://` URLs are accepted. See the
[Redis overview](index.md) for the URL format, cluster mode, and key prefixes.

## Byte-compatibility

Logs are stored through StackExchange.Redis' `IDatabase` (`StringGet` / `StringSet` /
`StringIncrement` / `ScriptEvaluate`) using the shared key layout and Optuna's `json.dumps` encoding,
so a study is interchangeable with Python Optuna and with the
[NRedisStack](nredisstack.md) and [ServiceStack.Redis](servicestack-redis.md) backends.

## License

MIT — both OptunaSharp and StackExchange.Redis.
