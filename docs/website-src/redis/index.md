# Redis Storage

OptunaSharp can persist an Optuna **study's journal** to a **Redis** server, so multiple processes —
and Python Optuna itself — can share one study over a network. It is the .NET port of Optuna's
[`optuna.storages.journal.JournalRedisBackend`](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.storages.journal.JournalRedisBackend.html).

## Choose your Redis client

The Redis backend ships as **three interchangeable packages**, one per .NET Redis client. They all
write **byte-identical** data to the same keys, so a study written by any of them can be read by the
others — and by Python Optuna over the same server.

| Package | Redis client | License | Choose it when… |
|---|---|---|---|
| [OptunaSharp.Storage.StackExchange.Redis](stackexchange-redis.md) | StackExchange.Redis | MIT | You have no preference — the de-facto-standard client |
| [OptunaSharp.Storage.NRedisStack](nredisstack.md) | NRedisStack | MIT | Your app standardizes on Redis's official client |
| [OptunaSharp.Storage.ServiceStack.Redis](servicestack-redis.md) | ServiceStack.Redis | ⚠️ Commercial | Your app already uses ServiceStack.Redis |

All three depend on a small shared package, **`OptunaSharp.Storage.Redis`**, which holds only the
client-agnostic pieces — the `redis://` URL parser and the key/log format — and which you never
reference directly.

> [!TIP]
> If you are unsure, use **[StackExchange.Redis](stackexchange-redis.md)** — MIT-licensed and the most
> widely used .NET Redis client.

## Quick start

Install one of the packages:

```bash
dotnet add package OptunaSharp.Storage.StackExchange.Redis
```

Then wrap the backend in a `JournalStorage`:

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

Each backend's type is named after its client — `StackExchangeJournalRedisBackend`,
`NRedisStackJournalRedisBackend`, `ServiceStackJournalRedisBackend` — but all three take the same
constructor arguments and behave identically. To switch clients you change the package and the class
name; nothing else.

## Connection URL

All three accept a `redis-py`-style URL:

```
redis://localhost:6379
redis://:password@localhost:6379/1
rediss://user:password@redis.example.com:6380     # TLS
unix:///var/run/redis/redis.sock                  # StackExchange / NRedisStack only
```

An optional `[username]:[password]@`, a `/db` index, and `?db=` / `?username=` / `?password=` query
parameters are supported (query parameters take precedence over the URL components).

## Standalone vs cluster

By default a standalone server is assumed and each log append is a **single atomic Lua `EVAL`**
(increment-then-set), which keeps the preserved journal consistent. For a Redis **cluster**, pass
`useCluster: true` — the append becomes a non-atomic `INCR` + `SET` (a cluster cannot slot-route the
keys a script derives), so the journal can become inconsistent under network errors.

```csharp
var backend = new StackExchangeJournalRedisBackend("redis://node1:6379", useCluster: true, prefix: "team-a");
```

The `prefix` namespaces the keys — `{prefix}:log_number`, `{prefix}:log:{n}`, `{prefix}:snapshot` — so
many studies (or users) can share one server without colliding.

## Interoperability

The on-wire format is **byte-compatible with Python Optuna's `JournalRedisBackend`**: every log is
stored as `json.dumps(log)` under the shared keys. A study can be written by Python and continued in
C# (with any of the three clients) and vice versa, over the same Redis. The three C# backends are
likewise fully interchangeable with one another — this is verified end to end by a cross-library
parity test (every library reads every other library's data byte-identically) and a Python ↔ C#
interop test.

## Status

Experimental, tracking Optuna's own `@experimental_class("3.1.0")` marking — constructing a backend
raises an `ExperimentalWarning`. The interface may change in a future release.
