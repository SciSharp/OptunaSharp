using System;
using System.Collections.Concurrent;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using AwesomeAssertions;
using Microsoft.Extensions.DependencyInjection;
using OptunaSharp.Collections;
using OptunaSharp.Pruners;
using OptunaSharp.Samplers;
using OptunaSharp.Storages;
using OptunaSharp.Storages.Journal;
using OptunaSharp.Storages.Journal.Extensions;
using ServiceStack.Redis;

namespace OptunaSharp.Tests.Storage;

/// <summary>
///     Concurrency and connection ownership of the three Redis journal backends against a real Redis: several
///     storages optimizing one study at once, and backends that close what they opened (scoped DI registrations
///     used to leak one connection per scope). Every case self-skips when Redis is unreachable.
/// </summary>
/// <remarks>
///     ServiceStack's free tier allows 6,000 requests per hour per process, counted client-side. The study case
///     below costs it about 600; keep any addition here small, since the whole suite shares that budget.
/// </remarks>
[TestClass]
[TestCategory("Integration")]
public class RedisJournalConcurrencyTests
{
    /// <summary>
    ///     Three workers, each with its own <see cref="JournalStorage" /> over its own backend, optimize one study at
    ///     once (the in-process shape of three processes): none fails, and a fresh replay is a perfect study.
    /// </summary>
    /// <param name="backendName">One of <see cref="RedisSession.Backends" />.</param>
    [TestMethod]
    [DataRow("StackExchange")]
    [DataRow("NRedisStack")]
    [DataRow("ServiceStack")]
    public void SeparateStorages_OptimizingOneStudyAtOnce_ProduceAPerfectStudy(string backendName)
    {
        RedisSession.EnsureReachable();
        var create = RedisSession.Backends.Single(b => b.Name == backendName).Create;
        string prefix = RedisSession.UniquePrefix();
        Optuna.CreateStudy(storage: new JournalStorage(create(prefix, false)), studyName: "s");

        var errors = new ConcurrentQueue<Exception>();
        using var barrier = new Barrier(3);
        var threads = Enumerable.Range(0, 3).Select(w => new Thread(() =>
        {
            barrier.SignalAndWait();
            try
            {
                var study = Optuna.LoadStudy("s", new JournalStorage(create(prefix, false)),
                    sampler: new RandomSampler(10 + w), pruner: new NopPruner());
                study.Optimize(t =>
                {
                    double x = t.SuggestFloat("x", -1, 1);
                    long k = t.SuggestInt("k", 0, 9);
                    t.Report(x, 0);
                    t.Report(x + k, 1);
                    t.SetUserAttr("worker", w);
                    return x + k;
                }, nTrials: 3);
            }
            catch (Exception e)
            {
                errors.Enqueue(e);
            }
        })).ToList();
        threads.ForEach(t => t.Start());
        threads.ForEach(t => t.Join());

        errors.Should().BeEmpty();
        var trials = Optuna.LoadStudy("s", new JournalStorage(create(prefix, false))).Trials;
        trials.Select(t => t.Number).Should().Equal(Enumerable.Range(0, 9));
        trials.Should().OnlyContain(t => t.State == TrialState.Complete
            && t.IntermediateValues.Count == 2 && t.UserAttrs.ContainsKey("worker"));
    }

    /// <summary>
    ///     <c>nJobs</c> at four times the thread pool's minimum on one shared storage: every trial completes.
    ///     StackExchange.Redis completes even synchronous calls on the .NET thread pool, and n_jobs workers used to
    ///     be pool threads blocked on the storage, so above the pool minimum the client starved and timed out
    ///     (<c>nJobs: 96</c> died after 88 trials). ServiceStack is left out: it uses its own sockets, and the trial
    ///     count here would not fit its free tier on a large machine.
    /// </summary>
    /// <param name="backendName"><c>StackExchange</c> or <c>NRedisStack</c>.</param>
    [TestMethod]
    [DataRow("StackExchange")]
    [DataRow("NRedisStack")]
    public void NJobs_FarAboveThePoolMinimum_DoesNotStarveTheRedisClient(string backendName)
    {
        RedisSession.EnsureReachable();
        ThreadPool.GetMinThreads(out int minWorkers, out _);
        int nJobs = Math.Max(16, 4 * minWorkers);
        var create = RedisSession.Backends.Single(b => b.Name == backendName).Create;
        var study = Optuna.CreateStudy(storage: new JournalStorage(create(RedisSession.UniquePrefix(), false)),
            sampler: new RandomSampler(3), pruner: new NopPruner());

        study.Optimize(t =>
        {
            double x = t.SuggestFloat("x", 0, 1);
            t.Report(x, 0);
            return x;
        }, nTrials: 2 * nJobs, nJobs: nJobs);

        study.Trials.Should().HaveCount(2 * nJobs).And.OnlyContain(t => t.State == TrialState.Complete);
    }

    /// <summary>Disposing a backend twice is harmless: DI may reach it through more than one registration.</summary>
    /// <param name="backendName">One of <see cref="RedisSession.Backends" />.</param>
    [TestMethod]
    [DataRow("StackExchange")]
    [DataRow("NRedisStack")]
    [DataRow("ServiceStack")]
    public void Dispose_CanBeCalledTwice(string backendName)
    {
        RedisSession.EnsureReachable();
        var backend = (IDisposable)RedisSession.Backends.Single(b => b.Name == backendName).Create(RedisSession.UniquePrefix(), false);

        backend.Dispose();
        backend.Invoking(b => b.Dispose()).Should().NotThrow();
    }

    /// <summary>
    ///     A StackExchange-based backend closes the connection it opened from a URL, so later journal calls fail with
    ///     <see cref="ObjectDisposedException" />, and leaves an injected multiplexer open.
    /// </summary>
    /// <param name="backendName"><c>StackExchange</c> or <c>NRedisStack</c>; both own a <c>ConnectionMultiplexer</c>.</param>
    [TestMethod]
    [DataRow("StackExchange")]
    [DataRow("NRedisStack")]
    public void Dispose_ClosesAnOwnedConnection_ButNotAnInjectedOne(string backendName)
    {
        RedisSession.EnsureReachable();
        var owned = RedisSession.Backends.Single(b => b.Name == backendName).Create(RedisSession.UniquePrefix(), false);
        owned.AppendLogs(new[] { new OrderedDict<string, object?> { ["op"] = 0L } });

        ((IDisposable)owned).Dispose();

        owned.Invoking(b => b.ReadLogs(0).ToList()).Should().Throw<ObjectDisposedException>();

        using var mux = RedisSession.RawConnection();
        IDisposable injected = backendName == "StackExchange"
            ? new StackExchangeJournalRedisBackend(mux, prefix: RedisSession.UniquePrefix())
            : new NRedisStackJournalRedisBackend(mux, prefix: RedisSession.UniquePrefix());
        injected.Dispose();
        mux.IsConnected.Should().BeTrue("a backend never closes a client it was given");
        mux.GetDatabase().Ping();
    }

    /// <summary>A ServiceStack backend leaves an injected client manager open when disposed.</summary>
    [TestMethod]
    public void ServiceStack_Dispose_LeavesAnInjectedManagerOpen()
    {
        RedisSession.EnsureReachable();
        using IRedisClientsManager manager = ServiceStackJournalRedisBackend.CreateManager(RedisUrl.Parse(RedisSession.Url));
        var backend = new ServiceStackJournalRedisBackend(manager, prefix: RedisSession.UniquePrefix());

        backend.Dispose();

        using IRedisClient client = manager.GetClient();
        client.Ping().Should().BeTrue("a backend never closes a client manager it was given");
    }

    /// <summary>
    ///     A scoped DI registration disposes the backend it built when the scope ends, closing its connection (40
    ///     scopes used to leave 40 connections open), while a pre-built backend registered as an instance survives
    ///     the container.
    /// </summary>
    /// <param name="backendName"><c>StackExchange</c> or <c>NRedisStack</c>.</param>
    [TestMethod]
    [DataRow("StackExchange")]
    [DataRow("NRedisStack")]
    public void DiScope_DisposesTheBackendItBuilt_ButNotAPrebuiltOne(string backendName)
    {
        RedisSession.EnsureReachable();
        string prefix = RedisSession.UniquePrefix();
        bool se = backendName == "StackExchange";
        var services = new ServiceCollection();
        if (se)
            services.AddStackExchangeRedisJournalStorage(RedisSession.Url, prefix: prefix, lifetime: ServiceLifetime.Scoped);
        else
            services.AddNRedisStackJournalStorage(RedisSession.Url, prefix: prefix, lifetime: ServiceLifetime.Scoped);

        BaseJournalBackend scopedBackend;
        using (var provider = services.BuildServiceProvider())
        {
            using (var scope = provider.CreateScope())
            {
                _ = scope.ServiceProvider.GetRequiredService<BaseStorage>().GetAllStudies();
                scopedBackend = se
                    ? scope.ServiceProvider.GetRequiredService<StackExchangeJournalRedisBackend>()
                    : scope.ServiceProvider.GetRequiredService<NRedisStackJournalRedisBackend>();
            }

            scopedBackend.Invoking(b => b.ReadLogs(0).ToList()).Should().Throw<ObjectDisposedException>(
                "the scope disposed the backend it built, and with it the connection");
        }

        var prebuilt = RedisSession.Backends.Single(b => b.Name == backendName).Create(prefix, false);
        var prebuiltServices = new ServiceCollection();
        if (se)
            prebuiltServices.AddStackExchangeRedisJournalStorage((StackExchangeJournalRedisBackend)prebuilt);
        else
            prebuiltServices.AddNRedisStackJournalStorage((NRedisStackJournalRedisBackend)prebuilt);
        using (var provider = prebuiltServices.BuildServiceProvider())
            _ = provider.GetRequiredService<BaseStorage>().GetAllStudies();

        prebuilt.Invoking(b => b.ReadLogs(0).ToList()).Should().NotThrow("a container never disposes an instance it was handed");
        ((IDisposable)prebuilt).Dispose();
    }

    /// <summary>
    ///     <c>DisposeAsync</c> closes the connection a StackExchange-based backend opened, so later journal calls fail
    ///     with <see cref="ObjectDisposedException" />, and an <c>await using</c> leaves an injected multiplexer open:
    ///     the asynchronous twin of <see cref="Dispose_ClosesAnOwnedConnection_ButNotAnInjectedOne" />.
    /// </summary>
    /// <param name="backendName"><c>StackExchange</c> or <c>NRedisStack</c>; both own a <c>ConnectionMultiplexer</c>.</param>
    /// <returns>The asynchronous test.</returns>
    [TestMethod]
    [DataRow("StackExchange")]
    [DataRow("NRedisStack")]
    public async Task DisposeAsync_ClosesAnOwnedConnection_ButNotAnInjectedOne(string backendName)
    {
        RedisSession.EnsureReachable();
        var owned = RedisSession.Backends.Single(b => b.Name == backendName).Create(RedisSession.UniquePrefix(), false);
        owned.AppendLogs(new[] { new OrderedDict<string, object?> { ["op"] = 0L } });

        await ((IAsyncDisposable)owned).DisposeAsync();

        owned.Invoking(b => b.ReadLogs(0).ToList()).Should().Throw<ObjectDisposedException>();

        using var mux = RedisSession.RawConnection();
        IAsyncDisposable injected = backendName == "StackExchange"
            ? new StackExchangeJournalRedisBackend(mux, prefix: RedisSession.UniquePrefix())
            : new NRedisStackJournalRedisBackend(mux, prefix: RedisSession.UniquePrefix());
        await injected.DisposeAsync();
        mux.IsConnected.Should().BeTrue("a backend never closes a client it was given");
        mux.GetDatabase().Ping();
    }

    /// <summary>
    ///     <c>Dispose</c> and <c>DisposeAsync</c> share one once-only flag: whichever runs first closes, and the other,
    ///     in either order, does nothing. A container can reach one backend through two registrations, and a caller
    ///     can mix <c>using</c> and <c>await using</c>.
    /// </summary>
    /// <param name="backendName">One of <see cref="RedisSession.Backends" />.</param>
    /// <returns>The asynchronous test.</returns>
    [TestMethod]
    [DataRow("StackExchange")]
    [DataRow("NRedisStack")]
    [DataRow("ServiceStack")]
    public async Task DisposeAndDisposeAsync_CloseOnce_InEitherOrder(string backendName)
    {
        RedisSession.EnsureReachable();
        var create = RedisSession.Backends.Single(b => b.Name == backendName).Create;

        var syncFirst = create(RedisSession.UniquePrefix(), false);
        ((IDisposable)syncFirst).Dispose();
        await syncFirst.Awaiting(b => ((IAsyncDisposable)b).DisposeAsync().AsTask()).Should().NotThrowAsync();

        var asyncFirst = create(RedisSession.UniquePrefix(), false);
        await ((IAsyncDisposable)asyncFirst).DisposeAsync();
        asyncFirst.Invoking(b => ((IDisposable)b).Dispose()).Should().NotThrow();
        await asyncFirst.Awaiting(b => ((IAsyncDisposable)b).DisposeAsync().AsTask()).Should().NotThrowAsync();
    }

    /// <summary>
    ///     <c>Dispose</c> and <c>DisposeAsync</c> racing on two threads, as a host shutting down along two paths at once:
    ///     neither throws, and afterwards the backend is closed. The once-only flag is an interlocked exchange, so
    ///     exactly one of the two runs the close. Ten rounds per client, a few requests each, well inside ServiceStack's
    ///     hourly budget.
    /// </summary>
    /// <param name="backendName">One of <see cref="RedisSession.Backends" />.</param>
    [TestMethod]
    [DataRow("StackExchange")]
    [DataRow("NRedisStack")]
    [DataRow("ServiceStack")]
    public void DisposeAndDisposeAsync_RacingOnTwoThreads_NeitherThrows_AndTheBackendIsClosed(string backendName)
    {
        RedisSession.EnsureReachable();
        var create = RedisSession.Backends.Single(b => b.Name == backendName).Create;

        for (int round = 0; round < 10; round++)
        {
            var backend = create(RedisSession.UniquePrefix(), false);
            backend.AppendLogs(new[] { new OrderedDict<string, object?> { ["op"] = 0L } });
            using var barrier = new Barrier(2);
            var errors = new ConcurrentQueue<Exception>();
            var closers = new Action[]
            {
                () => ((IDisposable)backend).Dispose(),
                () => ((IAsyncDisposable)backend).DisposeAsync().AsTask().GetAwaiter().GetResult(),
            }.Select(close => new Thread(() =>
            {
                barrier.SignalAndWait();
                try
                {
                    close();
                }
                catch (Exception e)
                {
                    errors.Enqueue(e);
                }
            })).ToList();
            closers.ForEach(t => t.Start());
            closers.ForEach(t => t.Join());

            errors.Should().BeEmpty($"round {round}: two disposals at once must not fail");
            backend.Invoking(b => b.ReadLogs(0).ToList()).Should().Throw<ObjectDisposedException>(
                $"round {round}: one of the two disposals closed the owned connection");
        }
    }

    /// <summary>A ServiceStack backend leaves an injected client manager open when disposed asynchronously.</summary>
    /// <returns>The asynchronous test.</returns>
    [TestMethod]
    public async Task ServiceStack_DisposeAsync_LeavesAnInjectedManagerOpen()
    {
        RedisSession.EnsureReachable();
        using IRedisClientsManager manager = ServiceStackJournalRedisBackend.CreateManager(RedisUrl.Parse(RedisSession.Url));
        var backend = new ServiceStackJournalRedisBackend(manager, prefix: RedisSession.UniquePrefix());

        await backend.DisposeAsync();

        using IRedisClient client = manager.GetClient();
        client.Ping().Should().BeTrue("a backend never closes a client manager it was given");
    }

    /// <summary>
    ///     An asynchronously disposed DI scope (as ASP.NET Core disposes its request scopes) closes the backend it
    ///     built, and so does an asynchronously disposed container for a singleton. Implementing
    ///     <see cref="IAsyncDisposable" /> must not cost the synchronous path: a synchronously disposed scope still
    ///     disposes it (<see cref="DiScope_DisposesTheBackendItBuilt_ButNotAPrebuiltOne" />), where a type with only
    ///     <c>DisposeAsync</c> would make that scope throw.
    /// </summary>
    /// <param name="backendName"><c>StackExchange</c> or <c>NRedisStack</c>.</param>
    /// <returns>The asynchronous test.</returns>
    [TestMethod]
    [DataRow("StackExchange")]
    [DataRow("NRedisStack")]
    public async Task AsyncDiScope_ClosesTheBackendItBuilt(string backendName)
    {
        RedisSession.EnsureReachable();
        bool se = backendName == "StackExchange";

        // A container whose backend comes from the URL overload, so the container built it and owns its connection.
        ServiceProvider Build(ServiceLifetime lifetime)
        {
            var services = new ServiceCollection();
            string prefix = RedisSession.UniquePrefix();
            if (se)
                services.AddStackExchangeRedisJournalStorage(RedisSession.Url, prefix: prefix, lifetime: lifetime);
            else
                services.AddNRedisStackJournalStorage(RedisSession.Url, prefix: prefix, lifetime: lifetime);
            return services.BuildServiceProvider();
        }

        // Uses the storage once (so the connection is open), then returns the backend instance the provider holds.
        BaseJournalBackend Resolve(IServiceProvider sp)
        {
            _ = sp.GetRequiredService<BaseStorage>().GetAllStudies();
            return se
                ? sp.GetRequiredService<StackExchangeJournalRedisBackend>()
                : sp.GetRequiredService<NRedisStackJournalRedisBackend>();
        }

        await using (var scopedProvider = Build(ServiceLifetime.Scoped))
        {
            BaseJournalBackend scoped;
            await using (var scope = scopedProvider.CreateAsyncScope())
                scoped = Resolve(scope.ServiceProvider);
            scoped.Invoking(b => b.ReadLogs(0).ToList()).Should().Throw<ObjectDisposedException>(
                "the asynchronously disposed scope closed the backend it built");
        }

        var singletonProvider = Build(ServiceLifetime.Singleton);
        BaseJournalBackend singleton = Resolve(singletonProvider);
        await singletonProvider.DisposeAsync();
        singleton.Invoking(b => b.ReadLogs(0).ToList()).Should().Throw<ObjectDisposedException>(
            "the asynchronously disposed container closed its singleton backend");
    }
}
