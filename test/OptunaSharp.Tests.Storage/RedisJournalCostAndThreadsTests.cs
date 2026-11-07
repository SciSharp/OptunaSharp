using System;
using System.Collections.Concurrent;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using AwesomeAssertions;
using OptunaSharp.Samplers;
using OptunaSharp.Storages.Journal;
using StackExchange.Redis;
using StackExchange.Redis.Profiling;

namespace OptunaSharp.Tests.Storage;

/// <summary>
///     Plan 10's Redis items against a real Redis: what a trial costs in client round trips and how snapshots are
///     taken (item 2.2), and StackExchange.Redis's <c>DedicatedThreads</c> feature flag, which the backends turn on
///     before they connect (item 2.3). Every case self-skips when Redis is unreachable.
/// </summary>
/// <remarks>
///     The feature flag is process-wide, so this class never runs in parallel with another, and every test puts the
///     flag back as it found it. StackExchange.Redis 2.x, which the net8.0 build uses, has no such flag: the cases about
///     the flag itself report Inconclusive there, saying so, while the many-callers case runs on both.
/// </remarks>
[TestClass]
[TestCategory("Integration")]
[DoNotParallelize]
public class RedisJournalCostAndThreadsTests
{
    private const string Flag = "DedicatedThreads";

    private bool _flagBefore;

    /// <summary>Remembers the process-wide flag, which several tests change.</summary>
    [TestInitialize]
    public void RememberFlag() => _flagBefore = ConnectionMultiplexer.GetFeatureFlag(Flag);

    /// <summary>Puts the process-wide flag back.</summary>
    [TestCleanup]
    public void RestoreFlag() => ConnectionMultiplexer.SetFeatureFlag(Flag, _flagBefore);

    /// <summary>
    ///     Reports Inconclusive when the loaded StackExchange.Redis has no <c>DedicatedThreads</c> flag (2.x ignores the
    ///     unknown name, so the flag reads back <c>false</c> after being set).
    /// </summary>
    private static void RequireTheFlag()
    {
        ConnectionMultiplexer.SetFeatureFlag(Flag, true);
        if (!ConnectionMultiplexer.GetFeatureFlag(Flag))
            Assert.Inconclusive(
                $"StackExchange.Redis {typeof(ConnectionMultiplexer).Assembly.GetName().Version} has no DedicatedThreads " +
                "feature flag (2.x); the backends' dedicatedThreads argument has no effect there.");
    }

    // ------------------------------------------------------------------ round trips (item 2.2)

    /// <summary>
    ///     A trial with two parameters (ask, two suggests, tell) on a warm standalone journal costs 12 client commands,
    ///     every one a script call: 8 syncs through the read script and 4 appends through the append script. Before
    ///     item 2.2 the same trial cost 20: 12 <c>GET</c>s, 4 <c>SETNX</c>es and 4 appends (measured 2026-11-07).
    ///     Counted with StackExchange.Redis's profiler, that is client calls, not the server's command statistics, which
    ///     also count the commands a script runs.
    /// </summary>
    [TestMethod]
    public void StandaloneTrial_CostsTwelveClientCommands_EveryOneAScriptCall()
    {
        RedisSession.EnsureReachable();
        using var mux = RedisSession.RawConnection();
        var session = new ProfilingSession();
        mux.RegisterProfiler(() => session);
        var study = Optuna.CreateStudy(
            storage: new JournalStorage(new StackExchangeJournalRedisBackend(mux, prefix: RedisSession.UniquePrefix())),
            sampler: new RandomSampler(0));

        void OneTrial()
        {
            var trial = study.Ask();
            double x = trial.SuggestFloat("x", 0, 1);
            long k = trial.SuggestInt("k", 0, 9);
            study.Tell(trial, x + k);
        }

        OneTrial();   // the first one also creates the study's records and warms the client's script cache
        session.FinishProfiling();

        OneTrial();
        var commands = session.FinishProfiling().Select(c => c.Command).ToList();

        commands.Should().HaveCount(12);
        commands.Should().OnlyContain(c => c == "EVAL" || c == "EVALSHA", "no GET or SETNX remains on a standalone journal");
    }

    // ------------------------------------------------------------------ snapshots (item 2.2)

    /// <summary>
    ///     With <c>snapshotInterval: 5</c> the storage saves a snapshot when it creates trial 5, under OptunaSharp's own
    ///     key only, and a new storage over the journal restores every trial whole.
    /// </summary>
    [TestMethod]
    public void JournalStorage_SnapshotsAtTheConfiguredInterval_UnderItsOwnKeyOnly()
    {
        RedisSession.EnsureReachable();
        string prefix = RedisSession.UniquePrefix();
        var study = Optuna.CreateStudy(
            storage: new JournalStorage(new StackExchangeJournalRedisBackend(RedisSession.Url, prefix: prefix), snapshotInterval: 5),
            sampler: new RandomSampler(0));
        var db = RedisSession.RawConnection().GetDatabase();

        study.Optimize(t => t.SuggestFloat("x", 0, 1) + t.SuggestInt("k", 0, 9), nTrials: 5);
        db.KeyExists(RedisJournalProtocol.OwnSnapshotKey(prefix)).Should().BeFalse("trials 0 to 4 are below the interval");

        study.Optimize(t => t.SuggestFloat("x", 0, 1) + t.SuggestInt("k", 0, 9), nTrials: 4);
        db.KeyExists(RedisJournalProtocol.OwnSnapshotKey(prefix)).Should().BeTrue("creating trial 5 saves a snapshot");
        db.KeyExists(RedisJournalProtocol.SnapshotKey(prefix)).Should().BeFalse("Python Optuna's snapshot key is never written");

        var reopened = Optuna.LoadStudy(study.StudyName,
            new JournalStorage(new StackExchangeJournalRedisBackend(RedisSession.Url, prefix: prefix))).Trials;
        reopened.Select(t => (t.Number, t.State, t.Value, t.Params["x"], t.Params["k"])).Should().Equal(
            study.Trials.Select(t => (t.Number, t.State, t.Value, t.Params["x"], t.Params["k"])));
    }

    /// <summary>
    ///     With no snapshot of its own, a backend looks at Python's key by its first byte (<c>GETRANGE 0 0</c>) and stops
    ///     there when it holds Python's pickle. A C# worker that only reads a Python study, whose own key stays empty,
    ///     therefore never downloads Python's pickle, which grows with the study. Counted with the client's profiler.
    /// </summary>
    [TestMethod]
    public void LoadSnapshot_WithOnlyPythonsPickle_ReadsItsFirstByte_NotThePickle()
    {
        RedisSession.EnsureReachable();
        using var mux = RedisSession.RawConnection();
        string prefix = RedisSession.UniquePrefix();
        // Pickle protocol 5 starts 0x80 0x05; the body's size stands in for a large study's replay state.
        var pickle = new byte[256 * 1024];
        pickle[0] = 0x80;
        pickle[1] = 0x05;
        mux.GetDatabase().StringSet(RedisJournalProtocol.SnapshotKey(prefix), pickle);
        var session = new ProfilingSession();
        mux.RegisterProfiler(() => session);
        var backend = new StackExchangeJournalRedisBackend(mux, prefix: prefix);

        backend.LoadSnapshot().Should().BeNull("Python's pickle is not OptunaSharp's snapshot");

        session.FinishProfiling().Select(c => c.Command).Should().Equal(new[] { "GET", "GETRANGE" },
            "the own key is read, then only the first byte of Python's");
    }

    // ------------------------------------------------------------------ DedicatedThreads (item 2.3)

    /// <summary>The URL and options constructors of both StackExchange.Redis-based backends turn the flag on by default.</summary>
    /// <param name="backendName"><c>StackExchange</c> or <c>NRedisStack</c>.</param>
    /// <param name="fromOptions">Whether to use the <see cref="ConfigurationOptions" /> constructor instead of the URL one.</param>
    [TestMethod]
    [DataRow("StackExchange", false)]
    [DataRow("StackExchange", true)]
    [DataRow("NRedisStack", false)]
    [DataRow("NRedisStack", true)]
    public void OwningConstructors_TurnTheFlagOn_ByDefault(string backendName, bool fromOptions)
    {
        RedisSession.EnsureReachable();
        RequireTheFlag();
        ConnectionMultiplexer.SetFeatureFlag(Flag, false);

        using var backend = Build(backendName, fromOptions, dedicatedThreads: null);

        ConnectionMultiplexer.GetFeatureFlag(Flag).Should().BeTrue();
    }

    /// <summary><c>dedicatedThreads: false</c> leaves the process-wide flag as it is: it never turns it off.</summary>
    /// <param name="backendName"><c>StackExchange</c> or <c>NRedisStack</c>.</param>
    [TestMethod]
    [DataRow("StackExchange")]
    [DataRow("NRedisStack")]
    public void DedicatedThreadsFalse_LeavesTheFlagAsItIs(string backendName)
    {
        RedisSession.EnsureReachable();
        RequireTheFlag();

        ConnectionMultiplexer.SetFeatureFlag(Flag, false);
        using (Build(backendName, fromOptions: false, dedicatedThreads: false))
            ConnectionMultiplexer.GetFeatureFlag(Flag).Should().BeFalse();

        ConnectionMultiplexer.SetFeatureFlag(Flag, true);
        using (Build(backendName, fromOptions: false, dedicatedThreads: false))
            ConnectionMultiplexer.GetFeatureFlag(Flag).Should().BeTrue();
    }

    /// <summary>A backend over a multiplexer the caller built never touches the flag.</summary>
    [TestMethod]
    public void InjectedMultiplexer_LeavesTheFlagAlone()
    {
        RedisSession.EnsureReachable();
        RequireTheFlag();
        ConnectionMultiplexer.SetFeatureFlag(Flag, false);
        using var mux = RedisSession.RawConnection();

        using (new StackExchangeJournalRedisBackend(mux))
        using (new NRedisStackJournalRedisBackend(mux))
            ConnectionMultiplexer.GetFeatureFlag(Flag).Should().BeFalse();
    }

    /// <summary>
    ///     The host scenario of THREADING.md §7.2: more thread-pool callers than the pool's minimum ask and tell on one
    ///     journal at once, no <c>nJobs</c>, and no trial is lost.
    /// </summary>
    /// <param name="backendName"><c>StackExchange</c> or <c>NRedisStack</c>.</param>
    /// <remarks>
    ///     <para>
    ///         On StackExchange.Redis 3.x (net10.0) the backends' default turns the flag on, no call times out and every
    ///         trial is told; without the flag, 64 callers on 32 cores lost 6 of 640 trials (4
    ///         <c>RedisTimeoutException</c>s, a <c>RedisConnectionException</c> and a <c>TaskCanceledException</c>,
    ///         measured 2026-11-07).
    ///     </para>
    ///     <para>
    ///         StackExchange.Redis 2.x (net8.0) has no flag, and whether its synchronous calls survive parked callers
    ///         depends on the pool growing before the 5 s <c>syncTimeout</c>. The first measurement passed (640 trials in
    ///         37.6 s), but on a machine busy with other work this case timed out in 1 of 3 runs on the code before plan
    ///         10's item 2.7 and in 2 of 3 after it (2026-11-07): luck, not a guarantee. On 2.x the case therefore pins the
    ///         remedy §7.2 gives net8.0 hosts, a pool minimum above the callers, and restores the minimum afterwards. The
    ///         class does not run in parallel with others, so the raised minimum reaches no other test.
    ///     </para>
    /// </remarks>
    [TestMethod]
    [DataRow("StackExchange")]
    [DataRow("NRedisStack")]
    public void ManyThreadPoolCallers_AskingAndTelling_LoseNoTrial(string backendName)
    {
        RedisSession.EnsureReachable();
        // The test leaves the flag to the backend's owning constructor, the path a host takes; on 3.x it must be on by
        // the time the callers start, on 2.x it does not exist.
        ConnectionMultiplexer.SetFeatureFlag(Flag, false);
        bool hasFlag = typeof(ConnectionMultiplexer).Assembly.GetName().Version!.Major >= 3;
        ThreadPool.GetMinThreads(out int minWorkers, out int minCompletionPorts);
        int callers = 2 * minWorkers + 8;
        const int trialsPerCaller = 3;
        if (!hasFlag)
            ThreadPool.SetMinThreads(callers + 16, minCompletionPorts);
        try
        {
            var create = RedisSession.Backends.Single(b => b.Name == backendName).Create;
            var study = Optuna.CreateStudy(storage: new JournalStorage(create(RedisSession.UniquePrefix(), false)),
                sampler: new RandomSampler(0));
            if (hasFlag)
                ConnectionMultiplexer.GetFeatureFlag(Flag).Should().BeTrue("the owning constructor turns the flag on by default");

            var errors = new ConcurrentQueue<Exception>();
            var callerTasks = Enumerable.Range(0, callers).Select(_ => Task.Run(() =>
            {
                for (int i = 0; i < trialsPerCaller; i++)
                {
                    try
                    {
                        var trial = study.Ask();
                        study.Tell(trial, trial.SuggestFloat("x", 0, 1));
                    }
                    catch (Exception e)
                    {
                        errors.Enqueue(e);
                    }
                }
            })).ToArray();
            Task.WaitAll(callerTasks);

            errors.Should().BeEmpty();
            study.Trials.Should().HaveCount(callers * trialsPerCaller).And.OnlyContain(t => t.State == TrialState.Complete);
        }
        finally
        {
            // Back to the minimum the test found, so the cases after it see the pool the process started with.
            if (!hasFlag)
                ThreadPool.SetMinThreads(minWorkers, minCompletionPorts);
        }
    }

    /// <summary>Builds a StackExchange.Redis-based backend through one of its owning constructors.</summary>
    /// <param name="backendName"><c>StackExchange</c> or <c>NRedisStack</c>.</param>
    /// <param name="fromOptions">Whether to use the <see cref="ConfigurationOptions" /> constructor instead of the URL one.</param>
    /// <param name="dedicatedThreads">The argument to pass, or <c>null</c> to leave it at its default.</param>
    /// <returns>The backend; dispose it to close its connection.</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="backendName" /> is neither of the two.</exception>
    private static IDisposable Build(string backendName, bool fromOptions, bool? dedicatedThreads)
    {
        var options = StackExchangeJournalRedisBackend.ToConfiguration(RedisUrl.Parse(RedisSession.Url));
        string prefix = RedisSession.UniquePrefix();
        return (backendName, fromOptions, dedicatedThreads) switch
        {
            ("StackExchange", false, null) => new StackExchangeJournalRedisBackend(RedisSession.Url, prefix: prefix),
            ("StackExchange", false, { } d) => new StackExchangeJournalRedisBackend(RedisSession.Url, prefix: prefix, dedicatedThreads: d),
            ("StackExchange", true, null) => new StackExchangeJournalRedisBackend(options, prefix: prefix),
            ("StackExchange", true, { } d) => new StackExchangeJournalRedisBackend(options, prefix: prefix, dedicatedThreads: d),
            ("NRedisStack", false, null) => new NRedisStackJournalRedisBackend(RedisSession.Url, prefix: prefix),
            ("NRedisStack", false, { } d) => new NRedisStackJournalRedisBackend(RedisSession.Url, prefix: prefix, dedicatedThreads: d),
            ("NRedisStack", true, null) => new NRedisStackJournalRedisBackend(options, prefix: prefix),
            ("NRedisStack", true, { } d) => new NRedisStackJournalRedisBackend(options, prefix: prefix, dedicatedThreads: d),
            _ => throw new ArgumentOutOfRangeException(nameof(backendName), backendName, "StackExchange or NRedisStack."),
        };
    }
}
