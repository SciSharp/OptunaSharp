using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net;
using System.Net.Sockets;
using System.Threading;
using AwesomeAssertions;
using Microsoft.AspNetCore.Builder;
using Microsoft.Data.Sqlite;
using OptunaSharp;
using OptunaSharp.Collections;
using OptunaSharp.Samplers;
using OptunaSharp.Storages;
using OptunaSharp.Storages.EFCore;
using OptunaSharp.Storages.Grpc;
using EFCoreRdbStorage = OptunaSharp.Storages.EFCore.RDBStorage;
using SqliteRdbStorage = OptunaSharp.Storages.RDBStorage;

namespace OptunaSharp.Tests.Storage;

/// <summary>
///     The topologies the journal concurrency suite covers, on the other shared storages: one storage used by every
///     worker of <c>Optimize(nJobs)</c>; separate storages, each with its own connection (the in-process shape of
///     separate processes), optimizing one study at once; enqueued trials popped by separate storages; and separate
///     storages racing to create one study. Every result is judged on a fresh storage's view of the study: as many
///     trials as the workers ran, numbered from 0 without a gap, all COMPLETE with every report and attribute, since a
///     run that throws nothing can still leave a broken study behind.
/// </summary>
/// <remarks>
///     Optuna 5.0 on SQLite, same scenarios: the shared storage and the separate storages produced perfect studies;
///     the enqueued trials did not, because Python's state change WAITING → RUNNING relies on <c>SELECT … FOR UPDATE</c>,
///     which SQLite ignores. Two workers popped one trial, the second one's tell failed with "Cannot tell a COMPLETE
///     trial." or <c>UpdateFinishedTrialError</c>, and the dead workers left 7 of 24 enqueued trials WAITING. On
///     PostgreSQL and MySQL the row lock makes the pop atomic in both languages.
/// </remarks>
public abstract class StorageConcurrencyTestsBase
{
    private const int Steps = 2;

    /// <summary>
    ///     Makes the test database empty and reachable, or marks the test inconclusive when its server is unavailable
    ///     (each subclass's service gate).
    /// </summary>
    protected abstract void ResetDatabase();

    /// <summary>Opens a new storage with a connection (or channel) of its own over the test database.</summary>
    /// <returns>A storage the caller disposes when it implements <see cref="IDisposable" />.</returns>
    protected abstract BaseStorage OpenStorage();

    /// <summary>Clears the database before each test.</summary>
    [TestInitialize]
    public void SetUp() => ResetDatabase();

    /// <summary>
    ///     Eight workers of one <c>Optimize(nJobs: 8)</c> share one storage: the study is perfect. The storage serializes
    ///     its own connection's use; nothing else may interleave two workers' writes of one trial.
    /// </summary>
    [TestMethod]
    public void SharedStorage_UnderNJobs_ProducesAPerfectStudy()
    {
        var storage = OpenStorage();
        try
        {
            var study = Optuna.CreateStudy(storage: storage, studyName: "shared", sampler: new RandomSampler(1));
            study.Optimize(t => Objective(t, "w"), nTrials: 80, nJobs: 8);
        }
        finally
        {
            (storage as IDisposable)?.Dispose();
        }

        ShouldBePerfect(LoadTrials("shared"), 80);
    }

    /// <summary>
    ///     Eight workers, each with its own storage and connection, optimize one study at once: nobody fails and the
    ///     study is perfect. The database alone arbitrates between them, as between processes.
    /// </summary>
    [TestMethod]
    public void SeparateStorages_OptimizingOneStudy_ProduceAPerfectStudy()
    {
        WithStorage(storage => Optuna.CreateStudy(storage: storage, studyName: "separate"));

        var errors = RunConcurrently(8, w => WithStorage(storage =>
            Optuna.LoadStudy("separate", storage, sampler: new RandomSampler(100 + w))
                .Optimize(t => Objective(t, "w" + w), nTrials: 10)));

        errors.Should().BeEmpty();
        ShouldBePerfect(LoadTrials("separate"), 80);
    }

    /// <summary>
    ///     24 enqueued trials, popped by eight workers with separate storages: each runs exactly once, none is left
    ///     WAITING, and no worker fails on a trial another one already finished. The change WAITING → RUNNING must
    ///     admit one worker per trial.
    /// </summary>
    [TestMethod]
    public void EnqueuedTrials_PoppedBySeparateStorages_RunExactlyOnce()
    {
        WithStorage(storage =>
        {
            var seed = Optuna.CreateStudy(storage: storage, studyName: "enqueued");
            for (int i = 0; i < 24; i++)
                seed.EnqueueTrial(new OrderedDict<string, object?> { ["x"] = (double)i, ["k"] = (long)i });
        });

        var errors = RunConcurrently(8, w => WithStorage(storage =>
            Optuna.LoadStudy("enqueued", storage, sampler: new RandomSampler(100 + w))
                .Optimize(t => Objective(t, "w" + w), nTrials: 3)));

        errors.Should().BeEmpty("no two workers may pop the same trial");
        var trials = LoadTrials("enqueued");
        ShouldBePerfect(trials, 24);
        trials.Select(t => Convert.ToInt64(t.Params["k"])).Order().Should().Equal(Enumerable.Range(0, 24).Select(i => (long)i));
    }

    /// <summary>
    ///     Sixteen separate storages create one study name at once: with <c>loadIfExists</c> every caller gets the one
    ///     study; without it exactly one creates it and the other fifteen get <see cref="DuplicatedStudyError" />.
    /// </summary>
    [TestMethod]
    public void CreateStudy_RacingOnOneName_CreatesItOnce()
    {
        var ids = new ConcurrentBag<int>();
        var shared = RunConcurrently(16, _ => WithStorage(storage =>
            ids.Add(Optuna.CreateStudy(storage: storage, studyName: "shared", loadIfExists: true).StudyId)));
        shared.Should().BeEmpty();
        ids.Distinct().Should().ContainSingle("every caller loads the one study");

        int created = 0, duplicates = 0;
        var exclusive = RunConcurrently(16, _ => WithStorage(storage =>
        {
            try
            {
                Optuna.CreateStudy(storage: storage, studyName: "exclusive");
                Interlocked.Increment(ref created);
            }
            catch (DuplicatedStudyError)
            {
                Interlocked.Increment(ref duplicates);
            }
        }));
        exclusive.Should().BeEmpty();
        created.Should().Be(1);
        duplicates.Should().Be(15);
        WithStorage(storage => storage.GetAllStudies().Select(s => s.StudyName).Should().BeEquivalentTo(new[] { "shared", "exclusive" }));
    }

    /// <summary>Runs <paramref name="body" /> with a new storage and disposes the storage afterwards.</summary>
    /// <param name="body">The work using the storage.</param>
    private void WithStorage(Action<BaseStorage> body)
    {
        var storage = OpenStorage();
        try
        {
            body(storage);
        }
        finally
        {
            (storage as IDisposable)?.Dispose();
        }
    }

    /// <summary>Loads the study through a fresh storage and returns its trials.</summary>
    /// <param name="name">The study name.</param>
    /// <returns>Deep copies of the trials in number order.</returns>
    private List<FrozenTrial> LoadTrials(string name)
    {
        List<FrozenTrial> trials = null!;
        WithStorage(storage => trials = Optuna.LoadStudy(name, storage).Trials);
        return trials;
    }

    /// <summary>Two parameters, <see cref="Steps" /> reports and one user attribute per trial.</summary>
    /// <param name="t">The running trial.</param>
    /// <param name="worker">Recorded as the <c>worker</c> user attribute, proving the trial ran to its end.</param>
    /// <returns>A value depending on both parameters.</returns>
    private static double Objective(Trial t, string worker)
    {
        double x = t.SuggestFloat("x", -10, 10);
        long k = t.SuggestInt("k", 0, 1000);
        for (int step = 0; step < Steps; step++)
            t.Report(x * x + step, step);
        t.SetUserAttr("worker", worker);
        return x * x + k * 1e-3;
    }

    /// <summary>The "perfect study" invariant: contiguous numbers from 0, all complete, every step and attribute present.</summary>
    /// <param name="trials">Trials from a fresh storage.</param>
    /// <param name="expected">The number of trials the workers ran.</param>
    private static void ShouldBePerfect(List<FrozenTrial> trials, int expected)
    {
        trials.Should().HaveCount(expected);
        trials.Select(t => t.Number).Should().Equal(Enumerable.Range(0, expected));
        trials.Should().OnlyContain(t => t.State == TrialState.Complete, "no trial may be left RUNNING, WAITING or FAIL");
        trials.Should().OnlyContain(t => t.IntermediateValues.Count == Steps);
        trials.Should().OnlyContain(t => t.UserAttrs.ContainsKey("worker"));
    }

    /// <summary>Starts every worker on its own thread, releases them together, and collects what they threw.</summary>
    /// <param name="workers">The number of worker threads.</param>
    /// <param name="body">The work of one worker, given its index.</param>
    /// <returns>Every exception a worker let escape.</returns>
    private static List<Exception> RunConcurrently(int workers, Action<int> body)
    {
        var errors = new ConcurrentQueue<Exception>();
        using var barrier = new Barrier(workers);
        var threads = Enumerable.Range(0, workers).Select(w => new Thread(() =>
        {
            barrier.SignalAndWait();
            try
            {
                body(w);
            }
            catch (Exception e)
            {
                errors.Enqueue(e);
            }
        })).ToList();
        threads.ForEach(t => t.Start());
        threads.ForEach(t => t.Join());
        return errors.ToList();
    }
}

/// <summary>
///     The concurrency topologies on a SQLite file, each storage with its own connection. Needs no service.
/// </summary>
/// <remarks>
///     [Misaligned] for <see cref="StorageConcurrencyTestsBase.EnqueuedTrials_PoppedBySeparateStorages_RunExactlyOnce" />:
///     Optuna 5.0 pops one enqueued trial twice here (see the base class), OptunaSharp's transaction admits one worker.
/// </remarks>
[TestClass]
public sealed class SqliteStorageConcurrencyTests : StorageConcurrencyTestsBase
{
    private string _path = null!;

    /// <summary>Starts every test on a new database file.</summary>
    protected override void ResetDatabase()
        => _path = Path.Combine(Path.GetTempPath(), $"optunasharp-rdb-cc-{Guid.NewGuid():N}.db").Replace('\\', '/');

    /// <summary>Opens a storage with its own connection to the test's file.</summary>
    /// <returns>A new SQLite storage the caller disposes.</returns>
    protected override BaseStorage OpenStorage() => new SqliteRdbStorage("sqlite:///" + _path);

    /// <summary>Deletes the database file and SQLite's side files once every pooled connection is closed.</summary>
    [TestCleanup]
    public void DeleteDatabase()
    {
        SqliteConnection.ClearAllPools();
        foreach (string f in new[] { _path, _path + "-wal", _path + "-shm", _path + "-journal" })
        {
            try
            {
                File.Delete(f);
            }
            catch (IOException)
            {
            }
        }
    }
}

/// <summary>The concurrency topologies on a live PostgreSQL server (Testcontainers), each storage with its own data source lease.</summary>
[TestClass]
public sealed class PostgreSqlStorageConcurrencyTests : StorageConcurrencyTestsBase
{
    /// <summary>Requires the PostgreSQL container and drops every Optuna table.</summary>
    protected override void ResetDatabase()
    {
        EFCoreDbSession.EnsureAvailable(RdbProvider.PostgreSql);
        EFCoreDbSession.Reset(RdbProvider.PostgreSql);
    }

    /// <summary>Opens a storage on the container's database.</summary>
    /// <returns>A new EF Core storage the caller disposes.</returns>
    protected override BaseStorage OpenStorage()
        => new EFCoreRdbStorage(RdbProvider.PostgreSql, EFCoreDbSession.ConnectionString(RdbProvider.PostgreSql));
}

/// <summary>The concurrency topologies on a live MySQL server (Testcontainers).</summary>
[TestClass]
public sealed class MySqlStorageConcurrencyTests : StorageConcurrencyTestsBase
{
    /// <summary>Requires the MySQL container and drops every Optuna table.</summary>
    protected override void ResetDatabase()
    {
        EFCoreDbSession.EnsureAvailable(RdbProvider.MySql);
        EFCoreDbSession.Reset(RdbProvider.MySql);
    }

    /// <summary>Opens a storage on the container's database.</summary>
    /// <returns>A new EF Core storage the caller disposes.</returns>
    protected override BaseStorage OpenStorage()
        => new EFCoreRdbStorage(RdbProvider.MySql, EFCoreDbSession.ConnectionString(RdbProvider.MySql));
}

/// <summary>
///     The concurrency topologies through the gRPC storage proxy: one in-process server fronting an in-memory storage,
///     each worker with a proxy and channel of its own, as separate client processes would have.
/// </summary>
[TestClass]
public sealed class GrpcStorageConcurrencyTests : StorageConcurrencyTestsBase
{
    private WebApplication? _app;
    private int _port;

    /// <summary>Starts a new server over a fresh in-memory storage on a free loopback port.</summary>
    protected override void ResetDatabase()
    {
        var listener = new TcpListener(IPAddress.Loopback, 0);
        listener.Start();
        _port = ((IPEndPoint)listener.LocalEndpoint).Port;
        listener.Stop();
        _app = GrpcProxyServer.MakeServer(new InMemoryStorage(), "localhost", _port);
        _app.StartAsync().GetAwaiter().GetResult();
    }

    /// <summary>Opens a proxy with its own channel and waits until the server answers.</summary>
    /// <returns>A new proxy the caller disposes.</returns>
    protected override BaseStorage OpenStorage()
    {
        var proxy = new GrpcStorageProxy("localhost", _port);
        proxy.WaitServerReady(TimeSpan.FromSeconds(15));
        return proxy;
    }

    /// <summary>Stops the server.</summary>
    [TestCleanup]
    public void StopServer()
    {
        if (_app is null)
            return;
        _app.StopAsync().GetAwaiter().GetResult();
        _app.DisposeAsync().AsTask().GetAwaiter().GetResult();
    }
}
