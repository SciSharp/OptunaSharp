# OptunaSharp.Storage.Grpc

gRPC **storage proxy** for [OptunaSharp](https://www.nuget.org/packages/OptunaSharp) — a faithful port
of Optuna's `optuna.storages.GrpcStorageProxy` (the client) and `optuna.storages.run_grpc_proxy_server`
(the server). It lets many workers share one storage over the network: the workers talk to a lightweight
gRPC **client** that forwards every storage operation to a **server** fronting a real backend (typically
the SQLite `RDBStorage`).

Because this package is generated from **Optuna's `api.proto`, verbatim**, an OptunaSharp client and a
**Python Optuna** `run_grpc_proxy_server` interoperate on the wire — and vice versa. This is verified
live, both directions, in the interop suite.

Part of the [SciSharp STACK](https://github.com/SciSharp).

> **Experimental** (as in Optuna, since v4.2.0): the interface can change.

## Install

```bash
dotnet add package OptunaSharp.Storage.Grpc
```

The **client** (`GrpcStorageProxy`) rides `Grpc.Net.Client`. The **server** (`GrpcProxyServer`) self-hosts
on `Grpc.AspNetCore` (Kestrel, cleartext HTTP/2), so the package references the ASP.NET Core shared
framework.

## Client — `GrpcStorageProxy`

```csharp
using OptunaSharp;
using OptunaSharp.Storages.Grpc;

using var storage = new GrpcStorageProxy(host: "localhost", port: 13000);
storage.WaitServerReady(TimeSpan.FromSeconds(10));

var study = Optuna.CreateStudy(storage: storage);
study.Optimize(t =>
{
    double x = t.SuggestFloat("x", -10, 10);
    return (x - 2) * (x - 2);
}, nTrials: 100);
```

`GrpcStorageProxy` is a `BaseStorage` — pass it to `Optuna.CreateStudy` / `Optuna.LoadStudy` like any
other storage. It keeps a **client-side trial cache** (exactly like Optuna's): each `GetAllTrials` fetches
only the trials that are new or still unfinished, so a growing study stays cheap to re-read.

## Server — `GrpcProxyServer`

Run a server fronting any storage. Wrap an RDB backend in `StorageFunctions.GetStorage` (as Optuna's
docstring advises) so the server keeps an in-process cache:

```csharp
using OptunaSharp.Storages;
using OptunaSharp.Storages.Grpc;

// var backend = new RDBStorage("sqlite:////path/to/optuna.db");   // OptunaSharp.Storage.Sqlite
var storage = StorageFunctions.GetStorage(backend);
GrpcProxyServer.RunGrpcProxyServer(storage, host: "localhost", port: 13000); // blocks
```

Need a non-blocking server (tests, embedding)? `GrpcProxyServer.MakeServer(...)` returns a startable
`WebApplication`:

```csharp
var app = GrpcProxyServer.MakeServer(storage, "localhost", 13000);
await app.StartAsync();
// ... app.StopAsync() when done ...
```

## Configuring the connection (full Grpc.Net.Client / Grpc.AspNetCore surface)

Beyond Optuna's host/port, the proxy lets you drive Grpc.Net.Client and Grpc.AspNetCore on their own
terms — the on-wire protobuf is fixed by `api.proto`, so cross-language byte-compatibility holds however
you connect.

**Client** — four constructors, from "just host/port" to "here is the channel I already built":

```csharp
new GrpcStorageProxy("localhost", 13000);                       // 1. Optuna-identical (insecure h2c)
new GrpcStorageProxy("localhost", 13000, o => o.MaxSendMessageSize = null);  // 2. tweak the derived options
new GrpcStorageProxy("https://host:443", new GrpcChannelOptions // 3. any address + full options (TLS, ...)
{
    Credentials = new SslCredentials(),
    ServiceConfig = new ServiceConfig { /* retry policy */ },
});
var channel = GrpcChannel.ForAddress("http://host:13000");      // 4. inject a channel you own
new GrpcStorageProxy(channel);                                   //    (the proxy never disposes it)
```

**Server** — `MakeServer` / `RunGrpcProxyServer` take optional `configureGrpc` (the
`GrpcServiceOptions`: interceptors, `EnableDetailedErrors`, compression, message sizes) and
`configureBuilder` (full `WebApplicationBuilder` access: TLS endpoints, authentication, your own
services and logging):

```csharp
GrpcProxyServer.RunGrpcProxyServer(storage, "0.0.0.0", 13000,
    configureGrpc: o => o.Interceptors.Add<MyInterceptor>(),
    configureBuilder: b => b.WebHost.ConfigureKestrel(k => { /* add a TLS endpoint */ }));
```

**Bring your own host** — already run an ASP.NET Core app? Register the proxy into it instead of the
self-contained host, and it inherits your Kestrel, TLS, authentication, interceptors, and other
endpoints:

```csharp
var builder = WebApplication.CreateBuilder(args);
builder.Services.AddOptunaStorageProxy(StorageFunctions.GetStorage(myBackend));
var app = builder.Build();
app.MapOptunaStorageProxy();       // chain .RequireAuthorization(), etc.
app.Run();
```

## Cross-language (Python ↔ C#)

Both directions work because the wire contract is Optuna's own `api.proto`:

```python
# Python server, OptunaSharp (C#) clients connect to it
from optuna.storages import run_grpc_proxy_server, get_storage
run_grpc_proxy_server(get_storage("sqlite:///optuna.db"), host="0.0.0.0", port=13000)
```

```python
# Python client against an OptunaSharp (C#) server
from optuna.storages import GrpcStorageProxy
storage = GrpcStorageProxy(host="csharp-host", port=13000)
```

Errors map through gRPC status codes exactly as Optuna does, so a missing study raises `KeyError`, a
duplicate name raises `DuplicatedStudyError`, a write to a finished trial raises
`UpdateFinishedTrialError`, and an invalid argument raises `ValueError` — on either side of the wire.

> Like Optuna, the gRPC proxy does **not** support `JournalStorage`; use the SQLite `RDBStorage`
> (`OptunaSharp.Storage.Sqlite`) behind the server.

## Notes

- The only deviation from Optuna's `api.proto` is a wire-neutral `option csharp_namespace` (so the
  generated DTOs don't squat the global `Optuna` namespace); the package/message/field/service contract
  is byte-identical, so cross-language wire-compatibility is unchanged.
- Insecure (cleartext HTTP/2) transport, matching Optuna's `grpc.insecure_channel` /
  `add_insecure_port`. Put it behind your own network boundary.

## License

MIT — see [LICENSE](https://github.com/SciSharp/OptunaSharp/blob/master/LICENSE).
