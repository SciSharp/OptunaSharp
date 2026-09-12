# gRPC storage proxy

`OptunaSharp.Storage.Grpc` lets remote C# and Python workers access a storage through one gRPC service. It includes the `GrpcStorageProxy` client, a Kestrel-based `GrpcProxyServer`, and extensions for adding the service to an existing ASP.NET Core application. All are in `OptunaSharp.Storages.Grpc`.

```text
C# worker     GrpcStorageProxy ─┐
                              ├─ HTTP/2 → StorageService → backing storage
Python worker GrpcStorageProxy ┘              server         SQLite / EF Core
```

The objective, sampler, and pruner run in each worker. The server handles study and trial storage operations; it does not train models, execute objective code, or schedule workers.

## Install

```bash
dotnet add package OptunaSharp.Storage.Grpc
```

The package targets .NET 8 and .NET 10. It uses `Grpc.Net.Client`, `Grpc.AspNetCore`, and Google.Protobuf, and references the `Microsoft.AspNetCore.App` shared framework because client and server ship together. Install the backing storage package separately in the server project. Python workers need Optuna's gRPC API and `grpcio`.

## Run a local server

For a local persistent example, add the SQLite package to the server project:

```bash
dotnet add package OptunaSharp.Storage.Sqlite
```

```csharp
using OptunaSharp.Storages;
using OptunaSharp.Storages.Grpc;

using var backend = new RDBStorage("sqlite:///optuna.db");
GrpcProxyServer.RunGrpcProxyServer(
    StorageFunctions.GetStorage(backend),
    host: "localhost", port: 13000);
```

This call blocks until server shutdown. `StorageFunctions.GetStorage(backend)` wraps RDB storage in the server's `CachedStorage`. The server also accepts an EF Core RDB backend; read that backend's [concurrency limits](../ef-core.md#concurrency-heartbeats-and-schema-lifecycle). SQLite retains its serialized-connection behavior behind the proxy.

## Connect a worker

```csharp
using OptunaSharp;
using OptunaSharp.Storages.Grpc;

using var storage = new GrpcStorageProxy("localhost", 13000);
storage.WaitServerReady(TimeSpan.FromSeconds(10));

var study = Optuna.CreateStudy(
    storage: storage, studyName: "distributed", loadIfExists: true);
study.Optimize(trial =>
{
    double x = trial.SuggestFloat("x", -10, 10);
    return (x - 2) * (x - 2);
}, nTrials: 20);
```

Other workers connect to the same endpoint and study name. They see the same history, while each `Optimize` call has its own trial budget and objective process. A remote worker uses the server's reachable host name in place of `localhost`; the server must also listen on an appropriate network interface.

## Contract, caching, and supported scope

The service contract is based on Optuna's `api.proto`: package, service, field numbers, and messages are preserved. A `csharp_namespace` option changes generated C# names without changing the wire contract. Storage attributes, distributions, and timestamps are encoded to match the upstream representation.

`GrpcStorageProxy` caches finished trials and requests new or unfinished trials on refresh. The server still obtains the backing storage's trial list before applying the request's incremental filter. Client caching reduces transmitted results; it does not by itself eliminate server-side enumeration.

The documented backing-storage path is RDB storage. `InMemoryStorage` is exercised by localhost and language-interoperability tests, but loses state when the server exits. Although the API takes `BaseStorage` and does not reject journals at construction, the package's server contract marks `JournalStorage` unsupported. Use Redis journals directly through their adapters instead of presenting them as a supported gRPC backend.

The protobuf service has no heartbeat RPCs, and the proxy does not implement `IBaseHeartbeat`. Setting a heartbeat interval on the server's RDB backend does not make remote workers record heartbeats. The transport also does not upload model artifacts; use an artifact integration for file contents.

The client and blocking server runner emit experimental warnings, following the upstream API introduced in Optuna 4.2.0. Defaults use cleartext HTTP/2 with no application authentication configured. The [server guide](server.md) describes host configuration and the [client guide](client.md) covers channel ownership and message limits.

## Read next

- [Client configuration, errors, and lifetime](client.md)
- [Server hosting, backing storage, and endpoint configuration](server.md)
- [Python and C# interoperability](cross-language.md)

Implementation: [gRPC package and dependencies](https://github.com/SciSharp/OptunaSharp/blob/master/src/OptunaSharp.Storage.Grpc/OptunaSharp.Storage.Grpc.csproj), [protobuf contract](https://github.com/SciSharp/OptunaSharp/blob/master/src/OptunaSharp.Storage.Grpc/Protos/api.proto), and [server service](https://github.com/SciSharp/OptunaSharp/blob/master/src/OptunaSharp.Storage.Grpc/Storages/Grpc/OptunaStorageProxyService.cs).
