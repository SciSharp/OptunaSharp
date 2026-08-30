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
        PythonSession.EnsureOrInconclusive();

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
