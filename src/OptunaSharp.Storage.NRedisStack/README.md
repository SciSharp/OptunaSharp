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

## Native configuration

The URL constructor mirrors Python Optuna, but you are not limited to it. Four constructors, all
taking the same `useCluster` / `prefix`:

```csharp
// 1. redis-py-style URL (Optuna-identical)
new NRedisStackJournalRedisBackend("redis://localhost:6379");

// 2. keep the URL, tweak the derived ConfigurationOptions
new NRedisStackJournalRedisBackend("redis://localhost:6379", o => { o.ConnectTimeout = 5000; o.ClientName = "my-app"; });

// 3. a ConfigurationOptions you built yourself
new NRedisStackJournalRedisBackend(new ConfigurationOptions { EndPoints = { "localhost:6379" }, Ssl = true });

// 4. an IConnectionMultiplexer you already own — e.g. the same connection you use for NRedisStack's
//    module APIs (JSON/Search/…) elsewhere in your app; you keep its lifetime
new NRedisStackJournalRedisBackend(existingMultiplexer);
```

The on-wire journal format is fixed regardless of how you connect, so the study stays byte-compatible
with Python Optuna and the sibling backends whichever constructor you use.

## License

MIT © SciSharp STACK Team. (NRedisStack is MIT.)
