using System;
using System.Linq;
using System.Threading;
using AwesomeAssertions;
using OptunaSharp.Samplers;
using OptunaSharp.Storages.Journal;
using Python.Runtime;
using StackExchange.Redis;

namespace OptunaSharp.Tests.Interop;

/// <summary>
///     Cross-language Redis persistence, live: Python Optuna writes a study to Redis → OptunaSharp
///     reads and continues it → Python reads the C#-written operations back. Same Redis keys, both
///     directions — run against <b>each</b> of the three C# client backends (StackExchange /
///     NRedisStack / ServiceStack), proving the on-wire encoding is interchangeable across the language
///     boundary for every library (the cross-library C#↔C# parity lives in
///     <c>OptunaSharp.Tests.Storage.CrossLibraryParityTests</c>).
///
///     Requires a reachable Redis AND redis-py in the interop Python; self-skips (Inconclusive)
///     otherwise, like the <c>RequiresPython</c> gate.
/// </summary>
[TestClass]
public class JournalRedisInteropTests
{
    /// <summary>The Redis every case uses (the Storage suite's compose service).</summary>
    internal const string Url = "redis://127.0.0.1:6379";
    private const string ProbeConfig = "127.0.0.1:6379,connectTimeout=1500,connectRetry=0,abortConnect=true";

    private string _prefix = null!;

    [TestInitialize]
    public void SetUp() => _prefix = $"optunasharp-interop:{Guid.NewGuid():N}";

    [TestMethod]
    [TestCategory("RequiresPython")]
    [TestCategory("RequiresRedis")]
    [TestCategory("Integration")]
    public void StackExchange_Python_RoundTrip()
        => RunParity(prefix => new StackExchangeJournalRedisBackend(Url, prefix: prefix));

    [TestMethod]
    [TestCategory("RequiresPython")]
    [TestCategory("RequiresRedis")]
    [TestCategory("Integration")]
    public void NRedisStack_Python_RoundTrip()
        => RunParity(prefix => new NRedisStackJournalRedisBackend(Url, prefix: prefix));

    [TestMethod]
    [TestCategory("RequiresPython")]
    [TestCategory("RequiresRedis")]
    [TestCategory("Integration")]
    public void ServiceStack_Python_RoundTrip()
        => RunParity(prefix => new ServiceStackJournalRedisBackend(Url, prefix: prefix));

    /// <summary>Python writes → the given C# backend reads &amp; continues → Python reads the C# ops back.</summary>
    private void RunParity(Func<string, BaseJournalBackend> createBackend)
    {
        PythonSession.EnsureOrInconclusive();
        EnsureRedisOrInconclusive();

        // --- 1. Python writes 5 trials + a user attr into Redis. -------------------------
        double pythonBest;
        using (Py.GIL())
        {
            using PyModule scope = Py.CreateScope();
            scope.Exec(
                "import optuna\n" +
                "optuna.logging.set_verbosity(optuna.logging.WARNING)\n" +
                $"backend = optuna.storages.journal.JournalRedisBackend('{Url}', prefix='{_prefix}')\n" +
                "storage = optuna.storages.JournalStorage(backend)\n" +
                "study = optuna.create_study(study_name='interop', storage=storage, " +
                "sampler=optuna.samplers.RandomSampler(seed=1))\n" +
                "def objective(t):\n" +
                "    x = t.suggest_float('x', -10, 10)\n" +
                "    n = t.suggest_int('n', 1, 5)\n" +
                "    t.report(x * 0.5, 0)\n" +
                "    return (x - 2) ** 2 + n\n" +
                "study.optimize(objective, n_trials=5)\n" +
                "study.set_user_attr('writer', 'python')\n" +
                "best = float(study.best_value)\n");
            pythonBest = scope.Get("best").As<double>();
        }

        // --- 2. OptunaSharp reads the study from Redis and continues it (via this backend). ----
        var storage = new JournalStorage(createBackend(_prefix));
        var study = Optuna.LoadStudy("interop", storage, sampler: new RandomSampler(seed: 7));

        study.Trials.Should().HaveCount(5, "all Python trials must replay from Redis");
        study.BestValue.Should().Be(pythonBest, "the replayed best value must be bit-identical");
        study.UserAttrs["writer"].Should().Be("python");
        study.Trials[0].IntermediateValues.Should().ContainKey(0);

        study.Optimize(t =>
        {
            double x = t.SuggestFloat("x", -10, 10);
            long n = t.SuggestInt("n", 1, 5);
            return Math.Pow(x - 2, 2) + n;
        }, nTrials: 3);
        study.SetUserAttr("continued_by", "csharp");
        double csharpBest = study.BestValue;
        study.Trials.Should().HaveCount(8);

        // --- 3. Python reads the C#-continued study back from the same Redis keys. -------
        using (Py.GIL())
        {
            using PyModule scope = Py.CreateScope();
            scope.Exec(
                "import optuna, math\n" +
                "optuna.logging.set_verbosity(optuna.logging.WARNING)\n" +
                $"backend = optuna.storages.journal.JournalRedisBackend('{Url}', prefix='{_prefix}')\n" +
                "storage = optuna.storages.JournalStorage(backend)\n" +
                "study = optuna.load_study(study_name='interop', storage=storage)\n" +
                "n_trials = len(study.trials)\n" +
                "best = float(study.best_value)\n" +
                "attrs = dict(study.user_attrs)\n" +
                "t5_x = float(study.trials[5].params['x'])\n" +
                "all_finite = all(t.value is not None and math.isfinite(t.value) for t in study.trials)\n");

            scope.Get("n_trials").As<int>().Should().Be(8, "Python must see the C# trials");
            scope.Get("best").As<double>().Should().Be(csharpBest, "best value agrees bit-exactly across languages");
            scope.Get("t5_x").As<double>().Should().Be((double)study.Trials[5].Params["x"]!,
                "a C#-written param must replay bit-identically in Python");
            using var attrs = scope.Get("attrs");
            attrs["writer"].As<string>().Should().Be("python");
            attrs["continued_by"].As<string>().Should().Be("csharp");
            scope.Get("all_finite").As<bool>().Should().BeTrue();
        }
    }

    /// <summary>
    ///     Plan 10, item 2.2: on one Redis journal each language keeps its own replay snapshot. Python pickles its state
    ///     under <c>{prefix}:snapshot</c>; OptunaSharp writes its binary v2 format (item 2.7) under
    ///     <c>{prefix}:snapshot:optunasharp</c>. Neither
    ///     overwrites the other's, and a cold start of either language restores its own snapshot without the
    ///     <c>Failed to restore `JournalStorageReplayResult`.</c> warning both logged while they shared one key (and then
    ///     replayed the whole journal).
    /// </summary>
    /// <remarks>
    ///     C# runs trials 0–6 at an interval of 5 (its snapshot at trial 5), Python continues with trials 7–13 at its
    ///     <c>SNAPSHOT_INTERVAL</c> patched from 100 to 5 (its snapshot at trial 10; the module global is put back
    ///     afterwards), then each language opens a new storage over the journal and must see all 14 trials.
    /// </remarks>
    [TestMethod]
    [TestCategory("RequiresPython")]
    [TestCategory("RequiresRedis")]
    [TestCategory("Integration")]
    public void PythonAndCSharp_OnOneRedisJournal_EachKeepsItsOwnSnapshot()
    {
        PythonSession.EnsureOrInconclusive();
        EnsureRedisOrInconclusive();

        var csharpWarnings = new System.Collections.Concurrent.ConcurrentQueue<string>();
        var handler = new RestoreWarningHandler(csharpWarnings);
        OptunaSharp.Logging.OptunaLogging.AddHandler(handler);
        try
        {
            // --- 1. C# creates the study: trials 0-6, a snapshot at trial 5 under its own key only. -------------
            var writer = new JournalStorage(new StackExchangeJournalRedisBackend(Url, prefix: _prefix), snapshotInterval: 5);
            var study = Optuna.CreateStudy(studyName: "snapshots", storage: writer, sampler: new RandomSampler(seed: 3));
            study.Optimize(t => t.SuggestFloat("x", -10, 10), nTrials: 7);

            using var mux = ConnectionMultiplexer.Connect(ProbeConfig);
            var db = mux.GetDatabase();
            byte[] ownAfterCSharp = (byte[])db.StringGet(RedisJournalProtocol.OwnSnapshotKey(_prefix))!;
            JournalStorageReplayResult.IsSnapshotV2(ownAfterCSharp).Should().BeTrue("OptunaSharp's snapshot is its binary v2 format");
            db.KeyExists(RedisJournalProtocol.SnapshotKey(_prefix)).Should().BeFalse("OptunaSharp never writes Python's key");

            // --- 2. Python continues (trials 7-13, its snapshot at trial 10), then cold-starts over the journal. ---
            double[] pythonValues;
            using (Py.GIL())
            {
                using PyModule scope = Py.CreateScope();
                scope.Exec(
                    "import logging, pickle, optuna\n" +
                    "import optuna.storages.journal._storage as js\n" +
                    "optuna.logging.set_verbosity(optuna.logging.WARNING)\n" +
                    "messages = []\n" +
                    "class Collect(logging.Handler):\n" +
                    "    def emit(self, record):\n" +
                    "        messages.append(record.getMessage())\n" +
                    "handler = Collect()\n" +
                    "logger = logging.getLogger('optuna.storages.journal._storage')\n" +
                    "logger.addHandler(handler)\n" +
                    "saved_interval = js.SNAPSHOT_INTERVAL\n" +
                    "js.SNAPSHOT_INTERVAL = 5\n" +
                    "try:\n" +
                    $"    backend = optuna.storages.journal.JournalRedisBackend('{Url}', prefix='{_prefix}')\n" +
                    "    study = optuna.load_study(study_name='snapshots', storage=optuna.storages.JournalStorage(backend),\n" +
                    "                              sampler=optuna.samplers.RandomSampler(seed=4))\n" +
                    "    study.optimize(lambda t: t.suggest_float('x', -10, 10), n_trials=7)\n" +
                    "    snapshot = backend.load_snapshot()\n" +
                    "    is_pickle = snapshot is not None and snapshot[:1] == b'\\x80'\n" +
                    "    pickled_trials = len(pickle.loads(snapshot).get_all_trials(0, None))\n" +
                    $"    cold_backend = optuna.storages.journal.JournalRedisBackend('{Url}', prefix='{_prefix}')\n" +
                    "    cold = optuna.load_study(study_name='snapshots', storage=optuna.storages.JournalStorage(cold_backend))\n" +
                    "    cold_values = [float(t.value) for t in cold.trials]\n" +
                    "finally:\n" +
                    "    js.SNAPSHOT_INTERVAL = saved_interval\n" +
                    "    logger.removeHandler(handler)\n" +
                    "restore_failures = sum('Failed to restore' in m for m in messages)\n");

                scope.Get("is_pickle").As<bool>().Should().BeTrue("Python's key holds Python's pickle");
                scope.Get("pickled_trials").As<int>().Should().Be(11, "Python saved its state when it created trial 10");
                scope.Get("restore_failures").As<int>().Should().Be(0, "Python's cold start restores its own pickle");
                using var coldValues = scope.Get("cold_values");
                pythonValues = Enumerable.Range(0, 14).Select(i => coldValues[i].As<double>()).ToArray();
            }

            // --- 3. C#'s own snapshot is untouched, and a C# cold start restores it and sees all 14 trials. --------
            // Outside the GIL: the C# side never needs Python, and the interop discipline keeps the two apart.
            ((byte[])db.StringGet(RedisJournalProtocol.OwnSnapshotKey(_prefix))!).Should().Equal(ownAfterCSharp,
                "Python never writes OptunaSharp's key");
            var cold = Optuna.LoadStudy("snapshots",
                new JournalStorage(new StackExchangeJournalRedisBackend(Url, prefix: _prefix), snapshotInterval: 5));
            cold.Trials.Select(t => t.Value!.Value).Should().Equal(pythonValues,
                "both languages replay the same 14 trials, bit for bit");

            csharpWarnings.Should().BeEmpty("OptunaSharp's cold start restores its own v2 snapshot");
        }
        finally
        {
            OptunaSharp.Logging.OptunaLogging.RemoveHandler(handler);
        }
    }

    /// <summary>Collects OptunaSharp's journal-restore failure warnings, the C# twin of Python's log handler above.</summary>
    /// <param name="sink">Receives each warning's message.</param>
    private sealed class RestoreWarningHandler(System.Collections.Concurrent.ConcurrentQueue<string> sink)
        : OptunaSharp.Logging.ILogHandler
    {
        /// <summary>Keeps the record when it is a journal snapshot that failed to restore.</summary>
        /// <param name="record">The log record.</param>
        public void Emit(in OptunaSharp.Logging.LogRecord record)
        {
            if (record.Message.Contains("Failed to restore"))
                sink.Enqueue(record.Message);
        }
    }

    /// <summary>
    ///     Why no Redis server can be used in this process, or <see langword="null" /> when one answered a PING. Probed
    ///     once, by the first test that needs Redis, since a server does not appear or vanish during a test run.
    /// </summary>
    /// <remarks>
    ///     Probing per test made every Redis case on a machine without Redis wait out its own refused connect. A refused
    ///     loopback connect returns at once on Linux, but Windows retries the SYN for about 2 s before failing, so on the
    ///     Windows runner (no Redis service) each skipped case cost about 1.8 s.
    /// </remarks>
    private static readonly Lazy<string?> RedisUnavailableReason = new(ProbeRedis, LazyThreadSafetyMode.ExecutionAndPublication);

    /// <summary>Connects to <see cref="Url" /> and pings it; the result is cached by <see cref="RedisUnavailableReason" />.</summary>
    /// <returns><see langword="null" /> when the server answered; otherwise the skip message naming the failure.</returns>
    private static string? ProbeRedis()
    {
        try
        {
            using var mux = ConnectionMultiplexer.Connect(ProbeConfig);
            _ = mux.GetDatabase().Ping();
            return null;
        }
        catch (Exception e)
        {
            // Any failure (refused, timed out, protocol) means the same thing to these tests: no usable server.
            return $"No Redis reachable at {Url}; skipping. Start one " +
                   $"(e.g. `docker run --rm -p 6379:6379 redis:7-alpine`). ({e.GetType().Name})";
        }
    }

    /// <summary>Skip (Inconclusive) unless a Redis server is reachable AND redis-py is importable.</summary>
    /// <remarks>
    ///     Shared with the Redis fleet of <see cref="JournalMixedFleetInteropTests" />. The server probe runs once per
    ///     process (<see cref="RedisUnavailableReason" />); the redis-py check is a cheap import lookup and runs each time.
    /// </remarks>
    /// <exception cref="AssertInconclusiveException">No Redis server answered, or redis-py is not installed.</exception>
    internal static void EnsureRedisOrInconclusive()
    {
        if (RedisUnavailableReason.Value is { } reason)
            Assert.Inconclusive(reason);

        using (Py.GIL())
        {
            using PyModule scope = Py.CreateScope();
            scope.Exec("import importlib.util as _u\nhas_redis = _u.find_spec('redis') is not None\n");
            if (!scope.Get("has_redis").As<bool>())
                Assert.Inconclusive("redis-py is not installed in the interop Python (pip install redis).");
        }
    }
}
