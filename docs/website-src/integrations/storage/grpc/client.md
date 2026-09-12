# gRPC client

`GrpcStorageProxy` implements `BaseStorage` over a managed `GrpcChannel`. Pass it directly to `Optuna.CreateStudy` or `Optuna.LoadStudy` to use a [gRPC storage server](server.md).

## Connect and load a study

Install `OptunaSharp.Storage.Grpc` in a .NET 8 or .NET 10 application:

```csharp
using OptunaSharp;
using OptunaSharp.Storages.Grpc;

using var storage = new GrpcStorageProxy("localhost", 13000);
storage.WaitServerReady(TimeSpan.FromSeconds(10));

var study = Optuna.CreateStudy(
    storage: storage, studyName: "distributed", loadIfExists: true);
study.Optimize(trial => Math.Pow(trial.SuggestFloat("x", -10, 10), 2), nTrials: 20);
```

`WaitServerReady(timeout)` waits for channel connectivity and throws `TimeoutException` if it does not become ready within the limit. With no argument it can wait indefinitely. Channel readiness does not validate a particular study or replace a real storage operation. Storage methods use synchronous generated RPC calls; the public API does not provide a per-operation cancellation-token or deadline argument.

## Four channel constructors

| Constructor | Connection | Ownership |
|---|---|---|
| `GrpcStorageProxy(string host = "localhost", int port = 13000)` | Cleartext HTTP/2 | Proxy owns channel |
| `GrpcStorageProxy(string host, int port, Action<GrpcChannelOptions> configureChannel)` | Modify the default cleartext channel options | Proxy owns channel |
| `GrpcStorageProxy(string address, GrpcChannelOptions channelOptions)` | Explicit address and native options | Proxy owns channel |
| `GrpcStorageProxy(GrpcChannel channel)` | Existing channel | Caller owns channel |

The host/port paths set insecure credentials and unlimited receive size before applying the optional callback. The address/options path uses exactly the options supplied; it does not apply those convenience defaults.

```csharp
using Grpc.Net.Client;
using OptunaSharp.Storages.Grpc;

using var channel = GrpcChannel.ForAddress("https://storage.example.com", new GrpcChannelOptions
{
    MaxReceiveMessageSize = 32 * 1024 * 1024,
    MaxSendMessageSize = 8 * 1024 * 1024,
});
using var storage = new GrpcStorageProxy(channel);
```

This HTTPS example assumes a matching TLS endpoint and a certificate trusted by the client. The channel can also carry a custom HTTP handler, logging factory, or service configuration. Configure both client and server message limits for the expected study size.

Channel retry configuration does not make all storage mutations idempotent. In particular, creating a trial after an ambiguous response needs care: blindly retrying a successful write can create another trial. The proxy does not add an idempotency key or an exactly-once transaction protocol.

## Cache behavior

Each proxy holds an incremental cache keyed by study. A refresh requests trial IDs greater than the greatest finished ID it has observed, plus the IDs of unfinished trials. The response is merged locally and returned in trial-number order, with state filtering applied in the client cache. Finished trials are retained, while running and waiting trials are refreshed.

`GetAllTrials` returns deep copies by default; `deepcopy: false` exposes the cached trial objects. Avoid modifying those objects. `DeleteStudy` clears its cache entry, and a `NOT_FOUND` response while refreshing also removes it. Cached state is process-local and not written to disk.

The cache lock serializes refreshes within one proxy. Different workers can use separate proxies. Their safe concurrent writes still depend on the server's backing storage.

## Error mapping

Expected storage errors are translated at the corresponding RPC boundaries:

| Server status | Client exception | Typical cause |
|---|---|---|
| `NOT_FOUND` | `NumSharp.KeyError` | Missing study or trial |
| `ALREADY_EXISTS` | `OptunaSharp.DuplicatedStudyError` | Duplicate study name |
| `FAILED_PRECONDITION` | `OptunaSharp.UpdateFinishedTrialError` | Mutation of a finished trial |
| `INVALID_ARGUMENT` | `NumSharp.ValueError` | Invalid parameter/distribution operation |

These are per-operation mappings, not a conversion of every possible server failure. Unhandled transport failures remain gRPC `RpcException`s. Inspect the status when diagnosing unavailable endpoints or message-size failures.

## Lifetime and limitations

`Dispose()` and `Close()` dispose only a channel created by the proxy. An injected channel stays available for other users and must be disposed by its owner. Keep the proxy/channel alive until optimization and any pending storage access finish.

The proxy does not expose heartbeat operations, execute objective code remotely, or send artifact file contents. It emits an experimental warning at construction. See [cross-language behavior](cross-language.md) before combining runtime versions or transport configurations.

Implementation: [client and cache](https://github.com/SciSharp/OptunaSharp/blob/master/src/OptunaSharp.Storage.Grpc/Storages/Grpc/GrpcStorageProxy.cs). Tests cover [channel configuration, ownership, cache refresh, and errors](https://github.com/SciSharp/OptunaSharp/blob/master/test/OptunaSharp.Tests.Storage/GrpcStorageProxyTests.cs), and [multiple clients sharing one server](https://github.com/SciSharp/OptunaSharp/blob/master/test/OptunaSharp.Tests.Interop/GrpcLocalhostInteropTests.cs).
