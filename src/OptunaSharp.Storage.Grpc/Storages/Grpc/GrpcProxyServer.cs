using System;
using System.Linq;
using System.Net;
using System.Net.Sockets;
using Grpc.AspNetCore.Server;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Server.Kestrel.Core;     // HttpProtocols
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using OptunaSharp.Internal;                          // OptunaWarnings, WarningCategory
using OptunaSharp.Logging;                           // OptunaLogging
using OptunaSharp.Storages;                          // BaseStorage

namespace OptunaSharp.Storages.Grpc;

/// <summary>
///     The gRPC proxy server — the port of <c>optuna/storages/_grpc/server.py</c> (<c>make_server</c> +
///     <c>run_grpc_proxy_server</c>, experimental since Optuna v4.2.0). It fronts a real
///     <see cref="BaseStorage" /> with the <see cref="OptunaStorageProxyService" /> so remote clients
///     (OptunaSharp's <see cref="GrpcStorageProxy" /> or Python Optuna's <c>GrpcStorageProxy</c>) can
///     share one storage over the network.
///
///     <para>Where Optuna hosts on grpcio's synchronous server, OptunaSharp hosts on
///     <b>Grpc.AspNetCore</b> (Kestrel + HTTP/2), serving cleartext HTTP/2 (h2c) to match grpc's
///     <c>add_insecure_port</c>. Pass a cache-fronted storage (e.g. <c>StorageFunctions.GetStorage(rdb)</c>)
///     rather than a bare <c>RDBStorage</c>, exactly as Optuna's docstring advises, so the server keeps an
///     in-process cache.</para>
/// </summary>
public static class GrpcProxyServer
{
    private static readonly Logger _logger = OptunaLogging.GetLogger("optuna");

    /// <summary>
    ///     Build (but do not start) a gRPC server for <paramref name="storage" /> — the port of
    ///     <c>make_server</c>. The returned <see cref="WebApplication" /> is startable
    ///     (<see cref="HostingAbstractionsHostExtensions.Start" /> / <c>StartAsync</c>) and
    ///     <c>IAsyncDisposable</c>; callers that want the blocking behaviour should use
    ///     <see cref="RunGrpcProxyServer" /> instead.
    /// </summary>
    /// <param name="storage">The storage to proxy.</param>
    /// <param name="host">Hostname to listen on (<c>localhost</c>, <c>0.0.0.0</c>, an IP, or a DNS name).</param>
    /// <param name="port">Port to listen on.</param>
    /// <param name="configureGrpc">Optional: tweak the server's <see cref="GrpcServiceOptions" /> after the
    ///   defaults (interceptors, <c>EnableDetailedErrors</c>, compression, max message sizes, ...).</param>
    /// <param name="configureBuilder">Optional: full access to the <see cref="WebApplicationBuilder" />
    ///   before <c>Build()</c> — add TLS endpoints / authentication / your own services / logging, etc. For
    ///   embedding the proxy in an app you already host, prefer <see cref="GrpcProxyServiceExtensions" />.</param>
    public static WebApplication MakeServer(
        BaseStorage storage, string host = "localhost", int port = 13000,
        Action<GrpcServiceOptions>? configureGrpc = null,
        Action<WebApplicationBuilder>? configureBuilder = null)
    {
        var builder = WebApplication.CreateSlimBuilder();
        // Kestrel and gRPC log to the ASP.NET logging pipeline; keep it silent so the only output is the
        // two lines run_grpc_proxy_server prints on Optuna's own logging seam (matching server.py).
        builder.Logging.ClearProviders();

        builder.WebHost.ConfigureKestrel(options =>
        {
            // Cleartext HTTP/2 (h2c) — the analog of grpc.server(...).add_insecure_port(f"{host}:{port}").
            options.Listen(ResolveEndpoint(host, port), listen => listen.Protocols = HttpProtocols.Http2);
        });

        builder.Services.AddGrpc(options =>
        {
            // Optuna sets grpc.max_receive_message_length = -1 (unlimited) on the client; mirror it on the
            // server so a large GetTrials reply (a big study) is never truncated in either direction.
            options.MaxReceiveMessageSize = null;
            options.MaxSendMessageSize = null;
            configureGrpc?.Invoke(options);
        });
        builder.Services.AddSingleton(storage);

        // Full escape hatch: let the caller shape the host (TLS, auth, extra endpoints/services, logging)
        // on top of the default insecure-h2c server before it is built.
        configureBuilder?.Invoke(builder);

        var app = builder.Build();
        app.MapGrpcService<OptunaStorageProxyService>();
        return app;
    }

    /// <summary>
    ///     Run a gRPC server for the given storage, host, and port — the port of
    ///     <c>run_grpc_proxy_server</c>. Blocks until the process is shut down.
    /// </summary>
    /// <param name="storage">A storage object to proxy.</param>
    /// <param name="host">Hostname to listen on.</param>
    /// <param name="port">Port to listen on.</param>
    /// <remarks>
    ///     Like Optuna, the gRPC storage proxy does not support <c>JournalStorage</c>; use the SQLite
    ///     <c>RDBStorage</c> (wrapped via <c>StorageFunctions.GetStorage</c>) instead.
    /// </remarks>
    public static void RunGrpcProxyServer(
        BaseStorage storage, string host = "localhost", int port = 13000,
        Action<GrpcServiceOptions>? configureGrpc = null,
        Action<WebApplicationBuilder>? configureBuilder = null)
    {
        OptunaWarnings.Warn(
            GrpcProto.ExperimentalMessage(nameof(RunGrpcProxyServer), "4.2.0"),
            WarningCategory.ExperimentalWarning);

        var app = MakeServer(storage, host, port, configureGrpc, configureBuilder);
        app.Start();
        _logger.Info($"Server started at {host}:{port}");
        _logger.Info("Listening...");
        app.WaitForShutdown();
    }

    /// <summary>Resolve <paramref name="host" /> (a name, an IP, <c>0.0.0.0</c>, or <c>::</c>) to a bind endpoint.</summary>
    private static IPEndPoint ResolveEndpoint(string host, int port)
    {
        IPAddress address;
        if (string.IsNullOrEmpty(host) || host == "0.0.0.0")
            address = IPAddress.Any;
        else if (host is "localhost")
            address = IPAddress.Loopback;
        else if (host is "::" or "[::]")
            address = IPAddress.IPv6Any;
        else if (IPAddress.TryParse(host, out var parsed))
            address = parsed;
        else
        {
            var addresses = Dns.GetHostAddresses(host);
            address = addresses.FirstOrDefault(a => a.AddressFamily == AddressFamily.InterNetwork)
                      ?? addresses.FirstOrDefault()
                      ?? IPAddress.Any;
        }
        return new IPEndPoint(address, port);
    }
}
