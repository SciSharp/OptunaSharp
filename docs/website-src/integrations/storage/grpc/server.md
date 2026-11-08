# gRPC server

`GrpcProxyServer` exposes an existing storage through `OptunaStorageProxyService`. It hosts on ASP.NET Core/Kestrel and serves the same protobuf storage operations as Python Optuna's gRPC server.

## Backing storage and prerequisites

Install `OptunaSharp.Storage.Grpc` and the backing-storage package in the server application. Both .NET 8 and .NET 10 are supported; the gRPC package references the ASP.NET Core shared framework.

```bash
dotnet add package OptunaSharp.Storage.Grpc
dotnet add package OptunaSharp.Storage.Sqlite
```

Wrap an RDB storage with `StorageFunctions.GetStorage` before hosting it. That creates the server's `CachedStorage`; the client has a separate incremental cache.

```csharp
using OptunaSharp.Storages;
using OptunaSharp.Storages.Grpc;

using var backend = new RDBStorage("sqlite:///optuna.db");
GrpcProxyServer.RunGrpcProxyServer(
    StorageFunctions.GetStorage(backend),
    host: "localhost", port: 13000);
```

`RunGrpcProxyServer` starts listening and blocks until shutdown. It emits an experimental warning and logs `Server started at {host}:{port}` and `Listening...` through OptunaSharp's logger.

An [EF Core RDB storage](../ef-core.md) can also be passed, retaining that implementation's concurrency limitations. In-memory backing storage is useful in tests but does not survive process termination. The server's documented contract excludes `JournalStorage`, even though its `BaseStorage` argument does not enforce that restriction in the constructor.

## Control start, stop, and disposal

`MakeServer` returns a `WebApplication` without starting it. Manage both the application and the database lifetime explicitly:

```csharp
using Microsoft.Extensions.Hosting;
using OptunaSharp.Storages;
using OptunaSharp.Storages.Grpc;

using var backend = new RDBStorage("sqlite:///optuna.db");
await using var app = GrpcProxyServer.MakeServer(
    StorageFunctions.GetStorage(backend), "localhost", 13000);
await app.StartAsync();
await app.WaitForShutdownAsync();
```

For an embedded service or test, call `StopAsync()` when finished, then dispose the app. The provided backend remains application-owned; registering an existing storage instance is not a transfer of database-resource ownership. `CachedStorage` is not the object to dispose in place of the underlying RDB storage.

`host` accepts `localhost`, `0.0.0.0`, `::`, IP literals, or DNS names. `localhost` binds IPv4 loopback; `0.0.0.0` binds all IPv4 interfaces. Choose the listening interface to match your worker network.

## Server configuration hooks

Both entry points accept:

- `configureGrpc`: modify `GrpcServiceOptions` for interceptors, compression, error detail, and message sizes.
- `configureBuilder`: configure the `WebApplicationBuilder` before it builds, including services, Kestrel, logging, and authentication components.

The convenience host configures an HTTP/2-only cleartext listener, unlimited receive/send message sizes, and an ASP.NET logging pipeline with providers cleared. The gRPC callback runs after those message-size defaults.

```csharp
using OptunaSharp.Storages;
using OptunaSharp.Storages.Grpc;

using var backend = new RDBStorage("sqlite:///optuna.db");
await using var app = GrpcProxyServer.MakeServer(
    StorageFunctions.GetStorage(backend),
    configureGrpc: options =>
    {
        options.MaxReceiveMessageSize = 8 * 1024 * 1024;
        options.MaxSendMessageSize = 32 * 1024 * 1024;
    });
await app.StartAsync();
// Keep the application alive for the required workload.
await app.StopAsync();
```

Adding an HTTPS listener through the builder does not remove the already configured cleartext listener. Use an application-owned host when you need explicit endpoint selection, authentication middleware, or endpoint authorization.

## Add the service to an existing ASP.NET Core host

Create a project using `Microsoft.NET.Sdk.Web`, install the packages above, and configure the HTTP/2 endpoint in the host:

```csharp
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Server.Kestrel.Core;
using OptunaSharp.Storages;
using OptunaSharp.Storages.Grpc;

using var backend = new RDBStorage("sqlite:///optuna.db");
var builder = WebApplication.CreateBuilder(args);
builder.WebHost.ConfigureKestrel(options =>
    options.ListenLocalhost(13000, endpoint => endpoint.Protocols = HttpProtocols.Http2));
builder.Services.AddOptunaStorageProxy(StorageFunctions.GetStorage(backend));

var app = builder.Build();
app.MapOptunaStorageProxy();
await app.RunAsync();
```

This complete local example uses cleartext HTTP/2. In a deployed application, configure the desired TLS listener and client trust together. `MapOptunaStorageProxy()` returns an endpoint convention builder, so an application with authentication and authorization configured can add `.RequireAuthorization()`.

`AddOptunaStorageProxy` uses `TryAddSingleton` for `BaseStorage`: an existing `BaseStorage` registration takes precedence. It also calls ordinary `AddGrpc()`, which does **not** apply `MakeServer`'s unlimited message-size defaults or its logging changes. Configure the returned gRPC builder or your host's gRPC options explicitly when needed.

## Runtime limits

The service invokes synchronous backend operations and relies on that backend's locking. It does not place every request behind a global service lock. An incremental `GetTrials` response is filtered after reading the backend's trial list, so the client cache saves wire traffic without removing all server-side enumeration.

No heartbeat RPC is available. Enabling heartbeats on the backing RDB instance does not automatically record heartbeats for objectives running in clients. Durability, schema compatibility, and write contention continue to belong to the backing storage.

Implementation: [server factory](https://github.com/SciSharp/OptunaSharp/blob/main/src/OptunaSharp.Storage.Grpc/Storages/Grpc/GrpcProxyServer.cs), [ASP.NET Core registration](https://github.com/SciSharp/OptunaSharp/blob/main/src/OptunaSharp.Storage.Grpc/Storages/Grpc/GrpcProxyServiceExtensions.cs), and [service operations](https://github.com/SciSharp/OptunaSharp/blob/main/src/OptunaSharp.Storage.Grpc/Storages/Grpc/OptunaStorageProxyService.cs). [Server tests](https://github.com/SciSharp/OptunaSharp/blob/main/test/OptunaSharp.Tests.Storage/GrpcStorageProxyTests.cs) cover real localhost connections, interceptor configuration, and embedding in an existing host.
