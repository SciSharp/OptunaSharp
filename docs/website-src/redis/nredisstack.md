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

The URL constructor is `NRedisStackJournalRedisBackend(url, useCluster = false, prefix = "")`. See the
[Redis overview](index.md) for the URL format, cluster mode, and key prefixes.

## Native configuration

Because NRedisStack builds on StackExchange.Redis, its connection is configured through the same
`ConfigurationOptions` / `IConnectionMultiplexer` types. Three more constructors (all also taking
`useCluster` and `prefix`) let you use the client directly:

```csharp
// keep the URL, tweak the derived ConfigurationOptions
new NRedisStackJournalRedisBackend("redis://localhost:6379", o => { o.ConnectTimeout = 5000; o.ClientName = "my-app"; });

// a ConfigurationOptions you built yourself
new NRedisStackJournalRedisBackend(new ConfigurationOptions { EndPoints = { "localhost:6379" }, Ssl = true });

// reuse a multiplexer you already own — for example the very connection on which you use NRedisStack's
// module APIs (JSON / Search / TimeSeries / Bloom) elsewhere. You keep its lifetime.
new NRedisStackJournalRedisBackend(myConnectionMultiplexer);
```

The journal still stores plain strings (see the note below); reusing your NRedisStack connection only
shares the transport, not the module data model, so byte-compatibility is unaffected.

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
