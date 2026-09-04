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
`?db=`). For a cluster, pass `useCluster: true`. An internal constructor accepts an existing
`IConnectionMultiplexer` to reuse a shared connection.

## License

MIT © SciSharp STACK Team. (StackExchange.Redis is MIT.)
