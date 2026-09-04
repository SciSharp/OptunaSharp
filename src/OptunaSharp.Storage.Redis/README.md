# OptunaSharp.Storage.Redis

Shared, **client-agnostic** primitives for the OptunaSharp Redis journal backends — with **no Redis
client dependency of its own**. Part of [OptunaSharp](https://github.com/SciSharp/OptunaSharp) (a .NET
port of Optuna) and the [SciSharp STACK](https://github.com/SciSharp).

## What's here

- **`RedisEndpoint` + `RedisUrl.Parse(url)`** — parse a `redis-py`-style URL (`redis://`, `rediss://`
  TLS, `unix://`, with `[user]:[pass]@`, `/db`, `?db=`/`?username=`/`?password=`) into a neutral
  endpoint. Each backend builds its own client from this.
- **`RedisJournalProtocol`** — the Redis key layout (`{prefix}:log_number`, `{prefix}:log:{n}`,
  `{prefix}:snapshot`), the atomic-append Lua script, and the experimental-warning text.

Sharing these constants (rather than an implementation abstraction) is exactly what keeps the three
**independent** client backends byte-identical against the same Redis, and byte-compatible with Python
Optuna's `JournalRedisBackend`:

- **`OptunaSharp.Storage.StackExchange.Redis`** — on StackExchange.Redis (the de-facto standard, MIT)
- **`OptunaSharp.Storage.NRedisStack`** — on NRedisStack (Redis's official client, MIT)
- **`OptunaSharp.Storage.ServiceStack.Redis`** — on ServiceStack.Redis (commercial license)

Reference one of those to actually use Redis storage; this package holds only the shared protocol.

## License

MIT © SciSharp STACK Team.
