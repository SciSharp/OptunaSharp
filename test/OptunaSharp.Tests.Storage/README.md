# OptunaSharp.Tests.Storage.Redis

Tests for the Redis journal backend — `OptunaSharp.Storage.Redis` (`JournalRedisBackend`). Two layers.

## Unit tests (default — no external dependencies)

The `redis://` → `ConfigurationOptions` URL adapter (`RedisUrlTests`): `redis://` / `rediss://` (TLS) /
`unix://` schemes, `[username]:[password]@` (percent-decoded), `/db` vs `?db=` precedence, and the
verbatim redis-py scheme error. No server needed.

```bash
dotnet test test/OptunaSharp.Tests.Storage.Redis -c Release --filter "TestCategory!=Integration"
```

## Integration tests (`Integration` category — dockerized Redis)

Full backend coverage against a **real, credential-free Redis** — no account or config needed:
append/read (middle offset, beyond-max, empty, multi-call counter continuity), the counter/key
layout, the standalone atomic-Lua-`EVAL` vs cluster `INCR`+`SET` branches (and that they store
**identical bytes**), **exact byte-compatibility with Python `json.dumps`** (default spaced
separators), the corrupt-last-log read rule, snapshots (save/load/absent/overwrite/binary), prefix
isolation, the empty-prefix default, the deprecated `JournalRedisStorage` alias, **atomic concurrent
appends**, and a full `Study` replay through `JournalStorage`.

| Under test | Server | Image |
|---|---|---|
| `JournalRedisBackend` | Redis | `redis:7-alpine` |

```bash
# 1. start Redis
docker compose -f test/OptunaSharp.Tests.Storage.Redis/docker-compose.yml up -d

# 2. run the integration tests
dotnet test test/OptunaSharp.Tests.Storage.Redis -c Release --filter TestCategory=Integration

# 3. stop Redis
docker compose -f test/OptunaSharp.Tests.Storage.Redis/docker-compose.yml down -v
```

When Redis is not reachable the integration tests **self-skip** (`Assert.Inconclusive`), so a plain
`dotnet test` without Docker is safe. Set `OPTUNASHARP_REDIS_REQUIRED=1` to turn an absent server into
a hard failure instead (for CI once wired). Override the endpoint with `OPTUNASHARP_REDIS_URL` (default
`redis://localhost:6379`).

The **cross-language** gate — Python Optuna ↔ Redis ↔ OptunaSharp, a study written by one side and
read back by the other over the same keys — lives in
`OptunaSharp.Tests.Interop/JournalRedisInteropTests` and uses this same server (plus
`pip install redis` in the interop Python).
