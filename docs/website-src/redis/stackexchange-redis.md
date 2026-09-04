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

The URL constructor is `StackExchangeJournalRedisBackend(url, useCluster = false, prefix = "")`.
`redis://`, `rediss://` (TLS) and `unix://` URLs are accepted. See the
[Redis overview](index.md) for the URL format, cluster mode, and key prefixes.

## Native StackExchange.Redis configuration

A `redis://` URL cannot express most of StackExchange.Redis' settings, so three more constructors let
you use the client directly — all also take `useCluster` and `prefix`:

```csharp
// keep the URL, tweak the derived ConfigurationOptions before connecting
new StackExchangeJournalRedisBackend("redis://localhost:6379", o =>
{
    o.ConnectTimeout        = 5000;
    o.SyncTimeout           = 5000;
    o.ConnectRetry          = 3;
    o.ClientName            = "my-app";
    o.ReconnectRetryPolicy  = new ExponentialRetry(5000);
});

// build the ConfigurationOptions yourself (multiple endpoints, TLS certificates, CommandMap, …)
var options = new ConfigurationOptions
{
    EndPoints = { "redis-a:6379", "redis-b:6379" },
    Ssl = true,
    ClientName = "my-app",
};
new StackExchangeJournalRedisBackend(options);

// reuse a connection you already own (pooled / shared across the app / DI-managed) —
// the backend never disposes a multiplexer it did not create
IConnectionMultiplexer mux = ConnectionMultiplexer.Connect("localhost:6379");
new StackExchangeJournalRedisBackend(mux);
```

Every StackExchange.Redis knob is reachable this way. The stored journal bytes are the same whichever
constructor you use, so byte-compatibility with Python Optuna and the sibling backends is unaffected.

## Byte-compatibility

Logs are stored through StackExchange.Redis' `IDatabase` (`StringGet` / `StringSet` /
`StringIncrement` / `ScriptEvaluate`) using the shared key layout and Optuna's `json.dumps` encoding,
so a study is interchangeable with Python Optuna and with the
[NRedisStack](nredisstack.md) and [ServiceStack.Redis](servicestack-redis.md) backends.

## License

MIT — both OptunaSharp and StackExchange.Redis.
