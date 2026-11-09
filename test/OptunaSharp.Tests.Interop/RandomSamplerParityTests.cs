using System;
using System.Collections.Generic;
using System.Linq;
using AwesomeAssertions;
using OptunaSharp.Collections;
using OptunaSharp.Samplers;
using Python.Runtime;

namespace OptunaSharp.Tests.Interop;

/// <summary>
///     LIVE bit-parity: OptunaSharp's RandomSampler against real Optuna running in-process, for
///     multiple seeds over a mixed search space (float, log-float, stepped float, int, log-int,
///     stepped int, categorical). Bit-identical is the bar — this is the plan §2 RNG-discipline
///     payoff, checked against whatever Optuna version is installed rather than pinned constants.
/// </summary>
[TestClass]
public class RandomSamplerParityTests
{
    private const string ObjectiveSource =
        "import optuna, json\n" +
        "optuna.logging.set_verbosity(optuna.logging.WARNING)\n" +
        "def objective(trial):\n" +
        "    x = trial.suggest_float('x', -10, 10)\n" +
        "    y = trial.suggest_float('y', 1e-5, 1e-1, log=True)\n" +
        "    z = trial.suggest_float('z', 0.1, 0.9, step=0.2)\n" +
        "    n = trial.suggest_int('n', 1, 100)\n" +
        "    m = trial.suggest_int('m', 1, 128, log=True)\n" +
        "    k = trial.suggest_int('k', 0, 20, step=5)\n" +
        "    c = trial.suggest_categorical('c', ['adam', 'sgd', 'rmsprop'])\n" +
        "    return x * x + y + z + n + m + k + (1 if c == 'adam' else 0)\n";

    private static List<(double X, double Y, double Z, long N, long M, long K, string C, double Value)>
        RunPython(int seed, int nTrials)
    {
        var rows = new List<(double, double, double, long, long, long, string, double)>();
        using (Py.GIL())
        {
            using PyModule scope = Py.CreateScope();
            scope.Exec(ObjectiveSource);
            scope.Exec(
                $"study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed={seed}))\n" +
                $"study.optimize(objective, n_trials={nTrials})\n" +
                "rows = [[t.params['x'], t.params['y'], t.params['z'], t.params['n'], t.params['m']," +
                " t.params['k'], t.params['c'], t.value] for t in study.trials]\n");
            using var pyRows = scope.Get("rows");
            long count = pyRows.Length();
            for (int i = 0; i < count; i++)
            {
                using var row = pyRows[i];
                rows.Add((
                    row[0].As<double>(), row[1].As<double>(), row[2].As<double>(),
                    row[3].As<long>(), row[4].As<long>(), row[5].As<long>(),
                    row[6].As<string>(), row[7].As<double>()));
            }
        }
        return rows;
    }

    private static Study RunCSharp(int seed, int nTrials)
    {
        var study = Optuna.CreateStudy(sampler: new RandomSampler(seed));
        study.Optimize(t =>
        {
            double x = t.SuggestFloat("x", -10, 10);
            double y = t.SuggestFloat("y", 1e-5, 1e-1, log: true);
            double z = t.SuggestFloat("z", 0.1, 0.9, step: 0.2);
            long n = t.SuggestInt("n", 1, 100);
            long m = t.SuggestInt("m", 1, 128, log: true);
            long k = t.SuggestInt("k", 0, 20, step: 5);
            string c = t.SuggestCategorical("c", new[] { "adam", "sgd", "rmsprop" });
            return x * x + y + z + n + m + k + (c == "adam" ? 1 : 0);
        }, nTrials: nTrials);
        return study;
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(0)]
    [DataRow(42)]
    [DataRow(123)]
    [DataRow(20260830)]
    public void MixedSearchSpace_IsBitIdentical_ToLiveOptuna(int seed)
    {
        // Asserts bit-identical suggestions vs live Optuna — host-pinned to x64 (arm64 libm drifts ~1 ULP).
        PythonSession.EnsureBitExactHostOrInconclusive();

        const int nTrials = 10;
        // Compute the C# side first, then open the GIL for the Python run + comparison.
        var csharp = RunCSharp(seed, nTrials);
        var python = RunPython(seed, nTrials);

        python.Should().HaveCount(nTrials);
        for (int i = 0; i < nTrials; i++)
        {
            var p = csharp.Trials[i].Params;
            ((double)p["x"]!).Should().Be(python[i].X, $"seed {seed} trial {i} x must be bit-identical");
            ((double)p["y"]!).Should().Be(python[i].Y, $"seed {seed} trial {i} y");
            ((double)p["z"]!).Should().Be(python[i].Z, $"seed {seed} trial {i} z");
            p["n"].Should().Be(python[i].N, $"seed {seed} trial {i} n");
            p["m"].Should().Be(python[i].M, $"seed {seed} trial {i} m");
            p["k"].Should().Be(python[i].K, $"seed {seed} trial {i} k");
            p["c"].Should().Be(python[i].C, $"seed {seed} trial {i} c");
            csharp.Trials[i].Value.Should().Be(python[i].Value, $"seed {seed} trial {i} objective value");
        }
    }

    /// <summary>
    ///     Plan 14: <see cref="RandomSampler" /> now draws the transform's bound rows one at a time through
    ///     <c>NumPyDraws.Uniform</c> instead of NumPy's array form (and creates no NumSharp arrays). This pins a long run over
    ///     every distribution kind to live Optuna: one-hot categoricals with two choices and with five choices of mixed types
    ///     (str, int, float, None), a log int spanning 1..2^40, a stepped int with negative bounds and a float spanning
    ///     ±1e300, 300 trials per seed, every value compared through Python's <c>repr</c> (exact for floats).
    /// </summary>
    /// <param name="seed">The seed both samplers start from.</param>
    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(0)]
    [DataRow(31337)]
    public void LongRun_AllDistributionKinds_IsBitIdentical_ToLiveOptuna(int seed)
    {
        PythonSession.EnsureBitExactHostOrInconclusive();
        const int nTrials = 300;
        string[] names = { "a", "b", "c", "d", "e", "f", "g", "h", "w" };

        var study = Optuna.CreateStudy(sampler: new RandomSampler(seed));
        study.Optimize(t =>
        {
            double a = t.SuggestFloat("a", -10, 10);
            t.SuggestFloat("b", 1e-5, 1e-1, log: true);
            t.SuggestFloat("c", 0.1, 0.9, step: 0.2);
            t.SuggestInt("d", 1, 100);
            t.SuggestInt("e", 1, 1L << 40, log: true);
            t.SuggestInt("f", -20, 20, step: 5);
            t.SuggestCategorical("g", new[] { true, false });
            t.SuggestCategorical("h", new OptunaSharp.Distributions.CategoricalChoice[]
                { "x", 1L, 2.5, OptunaSharp.Distributions.CategoricalChoice.None, "y" });
            t.SuggestFloat("w", -1e300, 1e300);
            return a;
        }, nTrials: nTrials);
        var csharp = study.Trials
            .Select(t => string.Join(" ", names.Select(n => ReprOf(t.Params[n]))))
            .ToList();

        var python = new List<string>();
        using (Py.GIL())
        {
            using PyModule scope = Py.CreateScope();
            scope.Exec(
                "import optuna\n" +
                "optuna.logging.set_verbosity(optuna.logging.WARNING)\n" +
                "def objective(t):\n" +
                "    a = t.suggest_float('a', -10, 10)\n" +
                "    t.suggest_float('b', 1e-5, 1e-1, log=True)\n" +
                "    t.suggest_float('c', 0.1, 0.9, step=0.2)\n" +
                "    t.suggest_int('d', 1, 100)\n" +
                "    t.suggest_int('e', 1, 2**40, log=True)\n" +
                "    t.suggest_int('f', -20, 20, step=5)\n" +
                "    t.suggest_categorical('g', [True, False])\n" +
                "    t.suggest_categorical('h', ['x', 1, 2.5, None, 'y'])\n" +
                "    t.suggest_float('w', -1e300, 1e300)\n" +
                "    return a\n" +
                $"study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed={seed}))\n" +
                $"study.optimize(objective, n_trials={nTrials})\n" +
                $"rows = [' '.join(repr(t.params[n]) for n in {PyList(names)}) for t in study.trials]\n");
            using var rows = scope.Get("rows");
            long count = rows.Length();
            for (int i = 0; i < count; i++)
            {
                using var row = rows[i];
                python.Add(row.As<string>());
            }
        }

        python.Should().HaveCount(nTrials);
        for (int i = 0; i < nTrials; i++)
            csharp[i].Should().Be(python[i], $"seed {seed} trial {i} (a b c d e f g h w)");
    }

    /// <summary>Python's <c>repr</c> of a suggested external value, for the value kinds the long run suggests.</summary>
    /// <param name="value">A trial's external parameter value.</param>
    /// <returns>The text Python's <c>repr</c> prints for the same value.</returns>
    /// <exception cref="NotSupportedException">The value is of a kind the long run never suggests.</exception>
    private static string ReprOf(object? value) => value switch
    {
        null => "None",
        bool b => b ? "True" : "False",
        long l => l.ToString(System.Globalization.CultureInfo.InvariantCulture),
        double d => OptunaSharp.Internal.PythonRepr.FloatRepr(d),
        string s => "'" + s + "'",
        _ => throw new NotSupportedException(value.GetType().Name),
    };

    /// <summary>A Python list literal of the given names, for embedding in a script.</summary>
    /// <param name="names">Parameter names without quotes or backslashes.</param>
    /// <returns>The list literal, for example <c>['a', 'b']</c>.</returns>
    private static string PyList(IEnumerable<string> names) => "[" + string.Join(", ", names.Select(n => "'" + n + "'")) + "]";

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void DistributionJson_IsByteIdentical_ToLiveOptuna()
    {
        PythonSession.EnsureOrInconclusive();

        // (C# json, python expression) pairs — serialized forms must match byte-for-byte.
        var pairs = new (string CSharp, string Python)[]
        {
            (OptunaSharp.Distributions.Distribution.ToJson(new OptunaSharp.Distributions.FloatDistribution(1, 10)),
                "optuna.distributions.FloatDistribution(1, 10)"),
            (OptunaSharp.Distributions.Distribution.ToJson(new OptunaSharp.Distributions.FloatDistribution(1e-5, 1e-1, log: true)),
                "optuna.distributions.FloatDistribution(1e-5, 1e-1, log=True)"),
            (OptunaSharp.Distributions.Distribution.ToJson(new OptunaSharp.Distributions.FloatDistribution(0.1, 1.0, step: 0.1)),
                "optuna.distributions.FloatDistribution(0.1, 1.0, step=0.1)"),
            (OptunaSharp.Distributions.Distribution.ToJson(new OptunaSharp.Distributions.IntDistribution(1, 100, log: true)),
                "optuna.distributions.IntDistribution(1, 100, log=True)"),
            (OptunaSharp.Distributions.Distribution.ToJson(new OptunaSharp.Distributions.CategoricalDistribution(
                    new OptunaSharp.Distributions.CategoricalChoice[] { OptunaSharp.Distributions.CategoricalChoice.None, true, 1L, 2.5, "x" })),
                "optuna.distributions.CategoricalDistribution([None, True, 1, 2.5, 'x'])"),
        };

        using (Py.GIL())
        {
            using PyModule scope = Py.CreateScope();
            scope.Exec("import optuna\nfrom optuna.distributions import distribution_to_json\n");
            foreach (var (csharpJson, pythonExpr) in pairs)
            {
                scope.Exec($"j = distribution_to_json({pythonExpr})");
                string pythonJson = scope.Get("j").As<string>();
                csharpJson.Should().Be(pythonJson, $"for {pythonExpr}");
            }
        }
    }
}
