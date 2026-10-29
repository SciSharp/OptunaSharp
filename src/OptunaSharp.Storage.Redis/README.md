# OptunaSharp.Storage.Redis

Shared, **client-agnostic** primitives for the OptunaSharp Redis journal backends — with **no Redis
client dependency of its own**. Part of [OptunaSharp](https://github.com/SciSharp/OptunaSharp) (a .NET
port of Optuna) and the [SciSharp STACK](https://github.com/SciSharp).

## What's here

- **`RedisEndpoint` + `RedisUrl.Parse(url)`** — parse a Redis URL (`redis://`, `rediss://` TLS,
  `unix://`) exactly as redis-py 8's `Redis.from_url` parses it: user information, host, port, the
  path or `?db=` database, and the connection options (`socket_timeout` / `socket_connect_timeout`,
  5 seconds by default, `client_name`, `lib_name`, `protocol`, and the `ssl_*` TLS settings), with
  redis-py's error kinds and messages for what it rejects. Each backend builds its own client from
  this neutral endpoint.
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
