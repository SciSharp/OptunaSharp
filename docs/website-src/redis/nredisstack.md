# NRedisStack backend

`OptunaSharp.Storage.NRedisStack` is the Redis journal backend on
[**NRedisStack**](https://github.com/redis/NRedisStack) — Redis's official .NET client (MIT). Choose
it when your application already standardizes on NRedisStack.

## Install

```bash
dotnet add package OptunaSharp.Storage.NRedisStack
```

## Usage

```csharp
using OptunaSharp.Storages;
using OptunaSharp.Storages.Journal;

var backend = new NRedisStackJournalRedisBackend("redis://localhost:6379");
var study = Optuna.CreateStudy(storage: new JournalStorage(backend));

study.Optimize(trial =>
{
    var x = trial.SuggestFloat("x", -10, 10);
    return (x - 2) * (x - 2);
}, nTrials: 20);
```

The constructor is `NRedisStackJournalRedisBackend(url, useCluster = false, prefix = "")`. See the
[Redis overview](index.md) for the URL format, cluster mode, and key prefixes.

## About NRedisStack

NRedisStack layers the Redis Stack module commands (JSON, Search, TimeSeries, Bloom) on top of
StackExchange.Redis. A journal backend needs only plain `GET` / `SET` / `INCR` / `EVAL`, which
NRedisStack serves through the underlying StackExchange.Redis `IDatabase` — so this backend is
byte-identical to the [StackExchange.Redis](stackexchange-redis.md) one and fully interoperable with it
and with Python Optuna.

> [!NOTE]
> Storing each log as a RedisJSON document (NRedisStack's headline feature) is deliberately **not**
> used: it would change the stored bytes and break interoperability with Python Optuna and the sibling
> backends.

## License

MIT — both OptunaSharp and NRedisStack.
