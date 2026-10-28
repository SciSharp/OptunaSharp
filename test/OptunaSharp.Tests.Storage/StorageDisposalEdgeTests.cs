using System;
using System.Data;
using System.Net;
using System.Net.Sockets;
using System.Threading.Tasks;
using AwesomeAssertions;
using Microsoft.AspNetCore.Builder;
using Microsoft.Data.Sqlite;
using Microsoft.Extensions.DependencyInjection;
using Npgsql;
using OptunaSharp;
using OptunaSharp.Storages;
using OptunaSharp.Storages.EFCore;
using OptunaSharp.Storages.EFCore.Extensions;
using OptunaSharp.Storages.Extensions;
using OptunaSharp.Storages.Grpc;
using OptunaSharp.Storages.Grpc.Extensions;
using EFCoreRdbStorage = OptunaSharp.Storages.EFCore.RDBStorage;
using SqliteRdbStorage = OptunaSharp.Storages.RDBStorage;

namespace OptunaSharp.Tests.Storage;

/// <summary>
///     A storage registered with a scoped (or transient) lifetime is disposed by the container twice: once through its
///     concrete registration, and once through the <see cref="BaseStorage" /> forwarder, whose factory returns the same
///     instance. A scope disposed asynchronously (an ASP.NET Core request, <c>AsyncServiceScope</c>) does it through
///     <c>DisposeAsync</c>, a synchronous one through <c>Dispose</c>. Both rounds must succeed and close what the storage
///     owns exactly once. The Redis backends' twin of these cases is in <see cref="RedisJournalConcurrencyTests" />.
/// </summary>
[TestClass]
public class StorageDisposalEdgeTests
{
    /// <summary>
    ///     A scoped SQLite storage built over a connection it owns: the scope resolves it through both registrations,
    ///     disposes it twice, and the connection ends closed with no error from the second disposal.
    /// </summary>
    /// <param name="async">Whether the scope is disposed asynchronously.</param>
    /// <returns>The asynchronous test.</returns>
    [TestMethod]
    [DataRow(false)]
    [DataRow(true)]
    public async Task ScopedSqliteStorage_IsDisposedThroughBothRegistrations_AndClosesItsConnection(bool async)
    {
        var conn = new SqliteConnection("Data Source=:memory:");
        conn.Open();
        await using var provider = new ServiceCollection()
            .AddSqliteStorage(_ => new SqliteRdbStorage(conn, ownsConnection: true), ServiceLifetime.Scoped)
            .BuildServiceProvider();

        await InScope(provider, async, sp =>
        {
            var storage = sp.GetRequiredService<SqliteRdbStorage>();
            sp.GetRequiredService<BaseStorage>().Should().BeSameAs(storage, "the forwarder hands out the scope's instance");
            Optuna.CreateStudy(storage: storage, studyName: "di").Optimize(t => t.SuggestFloat("x", 0, 1), nTrials: 2);
        });

        conn.State.Should().Be(ConnectionState.Closed, "the scope disposed the storage, which owned the connection");
    }

    /// <summary>
    ///     A scoped gRPC proxy, which owns the channel it opened: both disposals succeed, and the proxy is closed
    ///     afterwards (a call throws <see cref="ObjectDisposedException" />), while the server keeps serving others.
    /// </summary>
    /// <param name="async">Whether the scope is disposed asynchronously.</param>
    /// <returns>The asynchronous test.</returns>
    [TestMethod]
    [DataRow(false)]
    [DataRow(true)]
    public async Task ScopedGrpcProxy_IsDisposedThroughBothRegistrations_AndClosesItsChannel(bool async)
    {
        var listener = new TcpListener(IPAddress.Loopback, 0);
        listener.Start();
        int port = ((IPEndPoint)listener.LocalEndpoint).Port;
        listener.Stop();
        WebApplication app = GrpcProxyServer.MakeServer(new InMemoryStorage(), "localhost", port);
        await app.StartAsync();
        try
        {
            await using var provider = new ServiceCollection()
                .AddOptunaGrpcStorageProxyClient("localhost", port, ServiceLifetime.Scoped)
                .BuildServiceProvider();
            GrpcStorageProxy proxy = null!;

            await InScope(provider, async, sp =>
            {
                proxy = sp.GetRequiredService<GrpcStorageProxy>();
                sp.GetRequiredService<BaseStorage>().Should().BeSameAs(proxy);
                proxy.WaitServerReady(TimeSpan.FromSeconds(15));
                Optuna.CreateStudy(storage: proxy, studyName: "di").Optimize(t => t.SuggestFloat("x", 0, 1), nTrials: 2);
            });

            proxy.Invoking(p => p.GetAllStudies()).Should().Throw<ObjectDisposedException>("the scope closed the proxy's own channel");
            using var other = new GrpcStorageProxy("localhost", port);
            other.WaitServerReady(TimeSpan.FromSeconds(15));
            other.GetAllStudies().Should().ContainSingle(s => s.StudyName == "di", "closing one client leaves the server serving");
        }
        finally
        {
            await app.StopAsync();
            await app.DisposeAsync();
        }
    }

    /// <summary>
    ///     A scoped EF Core storage over a PostgreSQL data source it owns: both disposals succeed and the data source is
    ///     disposed once (Npgsql ignores the second). Needs the PostgreSQL container.
    /// </summary>
    /// <param name="async">Whether the scope is disposed asynchronously.</param>
    /// <returns>The asynchronous test.</returns>
    [TestMethod]
    [TestCategory("Integration")]
    [DataRow(false)]
    [DataRow(true)]
    public async Task ScopedEFCoreStorage_IsDisposedThroughBothRegistrations_AndClosesAnOwnedDataSource(bool async)
    {
        EFCoreDbSession.EnsureAvailable(RdbProvider.PostgreSql);
        EFCoreDbSession.Reset(RdbProvider.PostgreSql);
        var builder = new NpgsqlDataSourceBuilder(EFCoreDbSession.ConnectionString(RdbProvider.PostgreSql));
        EFCoreRdbStorage.MapOptunaEnums(builder);
        NpgsqlDataSource dataSource = builder.Build();
        await using var provider = new ServiceCollection()
            .AddEFCoreStorage(_ => new EFCoreRdbStorage(dataSource, ownsDataSource: true), ServiceLifetime.Scoped)
            .BuildServiceProvider();

        await InScope(provider, async, sp =>
        {
            var storage = sp.GetRequiredService<EFCoreRdbStorage>();
            sp.GetRequiredService<BaseStorage>().Should().BeSameAs(storage);
            Optuna.CreateStudy(storage: storage, studyName: "di").Optimize(t => t.SuggestFloat("x", 0, 1), nTrials: 2);
        });

        dataSource.Invoking(ds => ds.OpenConnection()).Should().Throw<ObjectDisposedException>(
            "the scope disposed the storage, which owned the data source");
    }

    /// <summary>
    ///     Runs <paramref name="body" /> in a new scope of <paramref name="provider" />, then disposes the scope the
    ///     requested way.
    /// </summary>
    /// <param name="provider">The root container.</param>
    /// <param name="async">Dispose the scope with <c>DisposeAsync</c> (true) or <c>Dispose</c> (false).</param>
    /// <param name="body">The work done with the scope's services.</param>
    /// <returns>A task that completes once the scope is disposed.</returns>
    private static async Task InScope(IServiceProvider provider, bool async, Action<IServiceProvider> body)
    {
        if (async)
        {
            await using var scope = provider.CreateAsyncScope();
            body(scope.ServiceProvider);
        }
        else
        {
            using var scope = provider.CreateScope();
            body(scope.ServiceProvider);
        }
    }
}
