using System;
using System.Linq;
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

    /// <summary>Skip (Inconclusive) unless a Redis server is reachable AND redis-py is importable.</summary>
    /// <remarks>Shared with the Redis fleet of <see cref="JournalMixedFleetInteropTests" />.</remarks>
    internal static void EnsureRedisOrInconclusive()
    {
        try
        {
            using var mux = ConnectionMultiplexer.Connect(ProbeConfig);
            _ = mux.GetDatabase().Ping();
        }
        catch (Exception e)
        {
            Assert.Inconclusive(
                $"No Redis reachable at {Url}; skipping. Start one " +
                $"(e.g. `docker run --rm -p 6379:6379 redis:7-alpine`). ({e.GetType().Name})");
        }

        using (Py.GIL())
        {
            using PyModule scope = Py.CreateScope();
            scope.Exec("import importlib.util as _u\nhas_redis = _u.find_spec('redis') is not None\n");
            if (!scope.Get("has_redis").As<bool>())
                Assert.Inconclusive("redis-py is not installed in the interop Python (pip install redis).");
        }
    }
}
