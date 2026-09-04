# OptunaSharp.Storage.NRedisStack

Redis journal-log backend for [**OptunaSharp**](https://github.com/SciSharp/OptunaSharp) on
[**NRedisStack**](https://github.com/redis/NRedisStack) — Redis's official .NET client (MIT). Part of
the [SciSharp STACK](https://github.com/SciSharp).

`NRedisStackJournalRedisBackend` persists an Optuna study's operation log to a Redis server. It is
byte-compatible with Python Optuna's `JournalRedisBackend` and with the sibling
`StackExchange`/`ServiceStack` backends — a study written by any of them can be read by the others over
the same Redis.

## Note on NRedisStack

NRedisStack layers the Redis Stack module commands (JSON, Search, TimeSeries, Bloom) on top of
StackExchange.Redis. A journal backend needs only plain `GET`/`SET`/`INCR`/`EVAL`, so this backend
uses the underlying StackExchange.Redis `IDatabase` surface (NRedisStack's modules do not apply, and
storing the log as a RedisJSON document would change the stored bytes and break cross-language
interop). Choose this package when your app already standardizes on NRedisStack.

## Usage

```csharp
using OptunaSharp.Storages;
using OptunaSharp.Storages.Journal;

var backend = new NRedisStackJournalRedisBackend("redis://localhost:6379");
var study = Optuna.CreateStudy(storage: new JournalStorage(backend));
```

## License

MIT © SciSharp STACK Team. (NRedisStack is MIT.)
