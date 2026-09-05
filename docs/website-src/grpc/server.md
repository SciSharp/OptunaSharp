# Server — `GrpcProxyServer`

`GrpcProxyServer` runs a gRPC server in front of any `BaseStorage`, so remote clients — OptunaSharp's
[`GrpcStorageProxy`](client.md) or Python Optuna's `GrpcStorageProxy` — can share it. It is the port of
Optuna's
[`optuna.storages.run_grpc_proxy_server`](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.storages.run_grpc_proxy_server.html).
Where Optuna hosts on grpcio's synchronous server, OptunaSharp hosts on **Grpc.AspNetCore** (Kestrel +
HTTP/2), serving cleartext HTTP/2 (h2c) to match grpc's `add_insecure_port`.

## Which storage to put behind it

Pass a **cache-fronted** storage, not a bare `RDBStorage`, exactly as Optuna's docstring advises — wrap
the backend in `StorageFunctions.GetStorage` so the server keeps an in-process cache and avoids a
round-trip to the database on every read:

```csharp
using OptunaSharp.Storages;

var backend = new RDBStorage("sqlite:///optuna.db");           // OptunaSharp.Storage.Sqlite
var storage = StorageFunctions.GetStorage(backend);            // -> CachedStorage(RDBStorage)
```

> Like Optuna, the gRPC proxy does **not** support `JournalStorage`; use the SQLite `RDBStorage`.

## Running the server

`RunGrpcProxyServer` starts the server and **blocks** until the process is shut down:

```csharp
using OptunaSharp.Storages.Grpc;

GrpcProxyServer.RunGrpcProxyServer(storage, host: "0.0.0.0", port: 13000);   // blocks
```

Need a **non-blocking** server (tests, or embedding it)? `MakeServer` returns a startable
`WebApplication`:

```csharp
var app = GrpcProxyServer.MakeServer(storage, "localhost", 13000);
await app.StartAsync();
// ... work ...
await app.StopAsync();
```

`host` accepts `localhost`, `0.0.0.0`, `::`, an IP literal, or a DNS name.

## Configuring gRPC and the host

Both entry points take two optional hooks:

- **`configureGrpc`** — the server's `GrpcServiceOptions`: interceptors, `EnableDetailedErrors`,
  compression providers, message-size limits.
- **`configureBuilder`** — full access to the `WebApplicationBuilder` before it is built: add a TLS
  endpoint, authentication, your own services, or logging.

```csharp
GrpcProxyServer.RunGrpcProxyServer(storage, "0.0.0.0", 13000,
    configureGrpc: o =>
    {
        o.Interceptors.Add<MyLoggingInterceptor>();
        o.EnableDetailedErrors = true;
    },
    configureBuilder: b =>
    {
        b.WebHost.ConfigureKestrel(k => { /* add an https endpoint with a certificate */ });
    });
```

## Bring your own host

Already run an ASP.NET Core app? Register the proxy **into it** instead of the self-contained host — it
then inherits your Kestrel, TLS, authentication, interceptors, and other endpoints. Two extension
methods do it:

```csharp
using OptunaSharp.Storages;
using OptunaSharp.Storages.Grpc;

var builder = WebApplication.CreateBuilder(args);
builder.Services.AddOptunaStorageProxy(StorageFunctions.GetStorage(myBackend));

var app = builder.Build();
app.MapOptunaStorageProxy();          // returns a builder — chain .RequireAuthorization(), etc.
app.Run();
```

`AddOptunaStorageProxy(storage)` registers the backing storage and the gRPC services (returning the
`IGrpcServerBuilder` for further configuration); `MapOptunaStorageProxy()` maps the proxy service onto
the app's endpoints. The service type itself, `OptunaStorageProxyService`, is public if you prefer to
wire it by hand.

## Output and logging

`RunGrpcProxyServer` logs `Server started at {host}:{port}` and `Listening...` through OptunaSharp's
logging seam (the `optuna` logger), matching Optuna's `server.py`. The ASP.NET Core / Kestrel logging
pipeline is silenced by default; re-add providers via `configureBuilder` if you want it.

## Status

Experimental (Optuna v4.2.0) — `RunGrpcProxyServer` raises an `ExperimentalWarning`.

## See also

- [Client — `GrpcStorageProxy`](client.md)
- [Cross-language (Python ↔ C#)](cross-language.md)
- [gRPC overview](index.md)
