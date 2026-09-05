# gRPC Storage Proxy

OptunaSharp can front **any** storage with a **gRPC proxy**, so workers on different machines — and in
different languages — share one study over the network. A lightweight **client**
([`GrpcStorageProxy`](client.md)) forwards every storage operation to a **server**
([`GrpcProxyServer`](server.md)) that wraps a real backend (typically the SQLite `RDBStorage`).

It is the .NET port of Optuna's
[`optuna.storages.GrpcStorageProxy`](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.storages.GrpcStorageProxy.html)
and
[`optuna.storages.run_grpc_proxy_server`](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.storages.run_grpc_proxy_server.html),
and ships as the package **`OptunaSharp.Storage.Grpc`** (namespace `OptunaSharp.Storages.Grpc`).

```
   worker 1 ─┐
   worker 2 ─┤  GrpcStorageProxy ──HTTP/2──▶  GrpcProxyServer ──▶  BaseStorage
   worker N ─┘     (client)                     (server)          (RDBStorage / …)
   (C# and/or Python)
```

## Install

```bash
dotnet add package OptunaSharp.Storage.Grpc
```

The **client** rides `Grpc.Net.Client`; the **server** self-hosts on `Grpc.AspNetCore` (Kestrel,
cleartext HTTP/2), so the package references the ASP.NET Core shared framework.

## Quick start

Run a **server** in front of a backend (wrap an RDB backend with `StorageFunctions.GetStorage` so the
server keeps an in-process cache, exactly as Optuna advises):

```csharp
using OptunaSharp.Storages;
using OptunaSharp.Storages.Grpc;

var backend = new RDBStorage("sqlite:///optuna.db");            // OptunaSharp.Storage.Sqlite
GrpcProxyServer.RunGrpcProxyServer(StorageFunctions.GetStorage(backend), host: "0.0.0.0", port: 13000);
```

Point any number of **clients** at it — `GrpcStorageProxy` is a `BaseStorage`, so it drops straight into
`Optuna.CreateStudy` / `Optuna.LoadStudy`:

```csharp
using OptunaSharp;
using OptunaSharp.Storages.Grpc;

using var storage = new GrpcStorageProxy(host: "server-host", port: 13000);
storage.WaitServerReady(TimeSpan.FromSeconds(10));

var study = Optuna.CreateStudy(storage: storage, studyName: "dist", loadIfExists: true);
study.Optimize(trial =>
{
    double x = trial.SuggestFloat("x", -10, 10);
    return (x - 2) * (x - 2);
}, nTrials: 100);
```

## How it works

- **The wire contract is Optuna's own `api.proto`, vendored verbatim.** The `StorageService` gRPC
  service mirrors the `BaseStorage` surface (create/delete study, set attrs, create trial, set
  param/state/values/intermediates, get trial(s), …). Because both runtimes are generated from the same
  contract, an OptunaSharp client and a Python `run_grpc_proxy_server` interoperate on the wire — see
  [Cross-language](cross-language.md).
- **Client-side cache.** Like Optuna's, the client keeps an incremental trial cache: each
  `GetAllTrials` fetches only the trials that are new or still unfinished, so re-reading a growing study
  stays cheap.
- **Exact error mapping.** Storage exceptions travel as gRPC status codes and are re-raised as the
  matching typed exception on the other side (`KeyError`, `DuplicatedStudyError`,
  `UpdateFinishedTrialError`, `ValueError`) — in both directions.

## Configuration at a glance

Beyond Optuna's host/port, the proxy exposes the full Grpc.Net.Client and Grpc.AspNetCore surface (TLS,
interceptors, timeouts, message sizes, an injected channel, or an existing ASP.NET Core host). The
on-wire protobuf is fixed by `api.proto`, so cross-language byte-compatibility holds however you connect.
See [Client](client.md) and [Server](server.md).

## Status

Experimental, tracking Optuna's own `@experimental` marking (added in Optuna v4.2.0) — constructing the
client or running the server raises an `ExperimentalWarning`. The interface may change in a future
release.

> Like Optuna, the gRPC proxy does **not** support `JournalStorage`; put the SQLite `RDBStorage` behind
> the server. The default transport is insecure (cleartext HTTP/2), matching Optuna — keep it behind your
> own network boundary, or configure TLS (see [Client](client.md) / [Server](server.md)).

## See also

- [Client — `GrpcStorageProxy`](client.md)
- [Server — `GrpcProxyServer`](server.md)
- [Cross-language (Python ↔ C#)](cross-language.md)
- [Storage guide](../docs/storage.md) — how a study uses its storage, and the other backends.
