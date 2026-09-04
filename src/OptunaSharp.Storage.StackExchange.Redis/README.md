# OptunaSharp.Storage.StackExchange.Redis

Redis journal-log backend for [**OptunaSharp**](https://github.com/SciSharp/OptunaSharp) on
[**StackExchange.Redis**](https://github.com/StackExchange/StackExchange.Redis) — the de-facto-standard
.NET Redis client (MIT). Part of the [SciSharp STACK](https://github.com/SciSharp).

`StackExchangeJournalRedisBackend` persists an Optuna study's operation log to a Redis server
(standalone or cluster). It is byte-compatible with Python Optuna's
`optuna.storages.journal.JournalRedisBackend` and with the sibling `NRedisStack`/`ServiceStack`
backends — a study written by any of them can be read by the others over the same Redis.

## Usage

```csharp
using OptunaSharp.Storages;
using OptunaSharp.Storages.Journal;

var backend = new StackExchangeJournalRedisBackend("redis://localhost:6379");
var study = Optuna.CreateStudy(storage: new JournalStorage(backend));
study.Optimize(t => { var x = t.SuggestFloat("x", -10, 10); return (x - 2) * (x - 2); }, nTrials: 20);
```

`redis://`, `rediss://` (TLS) and `unix://` URLs are accepted (optional `[user]:[pass]@`, `/db`,
`?db=`). For a cluster, pass `useCluster: true`.

## Native StackExchange.Redis configuration

The URL constructor mirrors Python Optuna, but you are not limited to it — drive StackExchange.Redis
on its own terms. Four constructors, all taking the same `useCluster` / `prefix`:

```csharp
// 1. redis-py-style URL (Optuna-identical)
new StackExchangeJournalRedisBackend("redis://localhost:6379");

// 2. keep the URL, tweak the derived ConfigurationOptions
new StackExchangeJournalRedisBackend("redis://localhost:6379", o =>
{
    o.ConnectTimeout = 5000;
    o.ClientName     = "my-app";
    o.ConnectRetry   = 3;
});

// 3. a ConfigurationOptions you built yourself (timeouts, retry policy, TLS certs, CommandMap, …)
var options = new ConfigurationOptions { EndPoints = { "localhost:6379" }, Ssl = true, ClientName = "my-app" };
new StackExchangeJournalRedisBackend(options);

// 4. an IConnectionMultiplexer you already own (pooled / shared / DI-managed — you keep its lifetime)
new StackExchangeJournalRedisBackend(existingMultiplexer);
```

The on-wire journal format is fixed regardless of how you connect, so the study stays byte-compatible
with Python Optuna and the sibling backends whichever constructor you use.

## License

MIT © SciSharp STACK Team. (StackExchange.Redis is MIT.)
