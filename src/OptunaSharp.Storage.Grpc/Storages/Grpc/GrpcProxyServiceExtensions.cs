using Grpc.AspNetCore.Server;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Routing;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.DependencyInjection.Extensions;
using OptunaSharp.Storages;

namespace OptunaSharp.Storages.Grpc;

/// <summary>
///     Register the Optuna storage-proxy gRPC service into an <b>existing</b> ASP.NET Core app — the
///     "bring your own host" path, for when you already run a web server and want to add the storage
///     proxy to it (with your own Kestrel/TLS/authentication/interceptors/other endpoints) instead of the
///     self-contained <see cref="GrpcProxyServer" />.
/// </summary>
/// <example>
///     <code>
///     var builder = WebApplication.CreateBuilder(args);
///     builder.Services.AddOptunaStorageProxy(StorageFunctions.GetStorage(myBackend));
///     var app = builder.Build();
///     app.MapOptunaStorageProxy();          // optionally .RequireAuthorization(), etc.
///     app.Run();
///     </code>
/// </example>
public static class GrpcProxyServiceExtensions
{
    /// <summary>
    ///     Register the backing <paramref name="storage" /> and the gRPC server services so the proxy
    ///     service can be mapped. Returns the <see cref="IGrpcServerBuilder" /> for further gRPC
    ///     configuration (e.g. <c>.AddServiceOptions&lt;OptunaStorageProxyService&gt;(...)</c>).
    /// </summary>
    /// <param name="services">The app's service collection.</param>
    /// <param name="storage">The storage the proxy service will front. Wrap an RDB backend with
    ///   <c>StorageFunctions.GetStorage</c> first so the server keeps an in-process cache.</param>
    public static IGrpcServerBuilder AddOptunaStorageProxy(this IServiceCollection services, BaseStorage storage)
    {
        services.TryAddSingleton(storage);
        return services.AddGrpc();
    }

    /// <summary>
    ///     Map the Optuna storage-proxy service onto the app's endpoints. Chain the returned builder to
    ///     add conventions (for example <c>.RequireAuthorization()</c>).
    /// </summary>
    /// <param name="endpoints">The app's endpoint route builder.</param>
    public static GrpcServiceEndpointConventionBuilder MapOptunaStorageProxy(this IEndpointRouteBuilder endpoints)
        => endpoints.MapGrpcService<OptunaStorageProxyService>();
}
