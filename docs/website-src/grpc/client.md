# Client — `GrpcStorageProxy`

`GrpcStorageProxy` is a `BaseStorage` whose every operation is a gRPC call to a
[`GrpcProxyServer`](server.md). Pass it to `Optuna.CreateStudy` / `Optuna.LoadStudy` like any other
storage. It is the port of Optuna's
[`optuna.storages.GrpcStorageProxy`](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.storages.GrpcStorageProxy.html).

```csharp
using OptunaSharp;
using OptunaSharp.Storages.Grpc;

using var storage = new GrpcStorageProxy(host: "localhost", port: 13000);
storage.WaitServerReady(TimeSpan.FromSeconds(10));

var study = Optuna.CreateStudy(storage: storage, studyName: "dist", loadIfExists: true);
```

Many clients can point at one server and optimize the **same** study concurrently — the
distributed-worker pattern the proxy exists for.

## Connecting

Four constructors take you from "just host and port" (Optuna-identical) to "here is the channel I
already built". The richer overloads expose the whole `Grpc.Net.Client` surface; the on-wire protobuf is
fixed by `api.proto` regardless of how you connect, so cross-language byte-compatibility always holds.

```csharp
// 1. host + port over insecure cleartext HTTP/2 (h2c) — the Optuna-identical entry point.
new GrpcStorageProxy("localhost", 13000);

// 2. host + port, then tweak the derived GrpcChannelOptions before the channel opens
//    (message sizes, a retry ServiceConfig, a custom HttpHandler, a LoggerFactory, …).
new GrpcStorageProxy("localhost", 13000, o =>
{
    o.MaxSendMessageSize = 32 * 1024 * 1024;
    o.ServiceConfig      = new ServiceConfig { /* MethodConfig with a RetryPolicy */ };
});

// 3. any address + fully-specified GrpcChannelOptions — use an https:// address with TLS credentials
//    for a secure channel.
new GrpcStorageProxy("https://server-host:443", new GrpcChannelOptions
{
    Credentials = new SslCredentials(),
});

// 4. bind to a GrpcChannel you built and own — the full escape hatch (pooling, TLS, interceptors,
//    a channel shared across your app or managed by DI). The proxy never disposes it.
var channel = GrpcChannel.ForAddress("http://server-host:13000");
new GrpcStorageProxy(channel);
```

Constructors 1–3 create the channel and own it (it is disposed with the proxy). Constructor 4 borrows a
channel you own — the proxy will not dispose it.

## Waiting for the server

`WaitServerReady(timeout)` blocks until the channel connects, or throws `TimeoutException` after
`timeout` (wait indefinitely when omitted):

```csharp
storage.WaitServerReady(TimeSpan.FromSeconds(10));
```

## The client-side trial cache

Like Optuna's proxy, `GrpcStorageProxy` keeps an incremental cache of a study's trials. Each
`GetAllTrials` requests only the trials that are **new** (id greater than the last finished one) or
**still unfinished** (whose state may have advanced), and merges them into the cache — so re-reading a
growing study is cheap and does not re-transfer finished trials. `DeleteStudy` invalidates that study's
cache.

## Error handling

Storage failures on the server come back as gRPC status codes and are re-raised as the matching typed
exception, exactly as Optuna does:

| Situation | Exception |
|---|---|
| No such study / trial | `KeyError` |
| Duplicate study name | `DuplicatedStudyError` |
| Write to a finished trial | `UpdateFinishedTrialError` |
| Invalid argument (e.g. incompatible distribution) | `ValueError` |

## Lifetime

`GrpcStorageProxy` is `IDisposable`. `Dispose()` (or `Close()`) shuts down the channel **only if the
proxy created it** — an injected channel (constructor 4) is left for you to dispose. Use a `using`
declaration for the common case:

```csharp
using var storage = new GrpcStorageProxy("localhost", 13000);
```

## Status

Experimental (Optuna v4.2.0) — the constructor raises an `ExperimentalWarning`.

## See also

- [Server — `GrpcProxyServer`](server.md)
- [Cross-language (Python ↔ C#)](cross-language.md)
- [gRPC overview](index.md)
