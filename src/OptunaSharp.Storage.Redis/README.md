# OptunaSharp.Storage.Redis

Redis journal-log backend for [**OptunaSharp**](https://github.com/SciSharp/OptunaSharp) — a .NET
port of [Optuna](https://github.com/optuna/optuna), the define-by-run hyperparameter-optimization
framework, and part of the [SciSharp STACK](https://github.com/SciSharp).

`JournalRedisBackend` persists an Optuna study's operation log to a Redis server (standalone or
cluster). It is a faithful port of Optuna's `optuna.storages.journal.JournalRedisBackend` and is
**byte-compatible with Python Optuna** on the wire — the same `{prefix}:log_number` counter,
`{prefix}:log:{n}` entries, and `{prefix}:snapshot` blob — so a study written to Redis by Python
Optuna can be read by OptunaSharp and vice versa.

Built on [StackExchange.Redis](https://github.com/StackExchange/StackExchange.Redis).

## Why a separate package?

OptunaSharp Core stays pure managed (NumSharp only). Anything that needs a foreign runtime — here, a
Redis server via StackExchange.Redis — ships as its own package, mirroring Optuna's own optional
dependencies. Install this only if you want the Redis backend.

## Install

```bash
dotnet add package OptunaSharp.Storage.Redis
```

## Usage

```csharp
using OptunaSharp.Storages;
using OptunaSharp.Storages.Journal;

// A journal storage whose backend is a Redis server.
var backend = new JournalRedisBackend("redis://localhost:6379");
var storage = new JournalStorage(backend);

var study = Optuna.CreateStudy(storage: storage);
study.Optimize(trial => {
    var x = trial.SuggestFloat("x", -10, 10);
    return (x - 2) * (x - 2);
}, nTrials: 20);
```

### Connection URL

`redis://`, `rediss://` (TLS) and `unix://` are accepted, with an optional
`[username]:[password]@` and `/db` (or `?db=`):

```
redis://localhost:6379
redis://:password@localhost:6379/1
rediss://user:password@redis.example.com:6380
unix:///var/run/redis/redis.sock?db=0
```

### Cluster

For a Redis **cluster**, pass `useCluster: true`. On a standalone server (the default) an append is a
single atomic `EVAL` (increment-then-set) that keeps the preserved logs consistent; a cluster cannot
slot-route the keys a script derives, so it falls back to two non-atomic commands — under network
errors the preserved logs may be inconsistent.

```csharp
var backend = new JournalRedisBackend("redis://node1:6379", useCluster: true, prefix: "team-a");
```

## Status

Experimental, tracking Optuna's own `@experimental_class("3.1.0")` marking (constructing the backend
raises an `ExperimentalWarning`). The interface can change in the future.

## License

MIT © SciSharp STACK Team.
