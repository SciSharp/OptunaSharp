using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json;
using AwesomeAssertions;
using OptunaSharp.Interop.Pymoo;
using OptunaSharp.Samplers;
using OptunaSharp.Storages;
using Python.Runtime;

namespace OptunaSharp.Tests.Interop;

/// <summary>
///     Live parity for the native <see cref="PymooSampler" /> against the <b>real pymoo</b> it drives.
///     OptunaSharp runs a multi-objective study (its <see cref="PymooSampler" /> reconstructs and drives
///     genuine pymoo algorithms via pythonnet); the assertions come in two tiers:
///
///     <list type="bullet">
///         <item><b>Behavioral</b> — every algorithm in the <see cref="PymooAlgorithm" /> zoo completes a
///         seeded 2-objective study, produces a non-trivial Pareto front, and is reproducible under a
///         fixed seed.</item>
///         <item><b>Value-parity</b> — for a plain unit-cube float space, the bridge's per-trial
///         trajectory matches a direct pymoo <c>ask</c>/<c>tell</c> replay <b>seeded with the same
///         generation 0</b> (the one lossy seam — the high-boundary clip of Optuna's half-open float
///         interval — is applied on the Python side too, so the match is tight). This isolates "does the
///         bridge faithfully drive pymoo" from the deliberate generation-0 RNG difference vs a monolithic
///         <c>pymoo.minimize</c> (which the bridge does not claim to reproduce bit-for-bit).</item>
///     </list>
///
///     <para>Self-skips (Inconclusive) when <c>pymoo</c> is not installed in the interop interpreter.</para>
/// </summary>
[TestClass]
public class PymooSamplerParityTests : OptunaInteropTestBase
{
    /// <summary>PymooSampler's ctor imports pymoo; probe it up front and self-skip if it is missing.</summary>
    private void RequirePymooOrInconclusive()
    {
        PythonSession.EnsureOrInconclusive();
        try
        {
            PyExec("import pymoo\n");
        }
        catch (PythonException)
        {
            Assert.Inconclusive("pymoo is not installed in the interop interpreter (`pip install pymoo`).");
        }
    }

    /// <summary>ZDT1 restricted to two variables in the unit cube — a smooth, conflicting 2-objective
    /// trade-off with a broad Pareto front (identical C# / Python arithmetic: sqrt only).</summary>
    private static IReadOnlyList<double> Zdt1(double x0, double x1)
    {
        double f1 = x0;
        double g = 1.0 + 9.0 * x1;
        double f2 = g * (1.0 - Math.Sqrt(f1 / g));
        return new[] { f1, f2 };
    }

    private static PymooSampler MakeSampler(PymooAlgorithm algorithm, int populationSize, int seed)
    {
        // One options object serves every family — each algorithm reads only the knobs it needs
        // (reference directions for NSGA-III/U-NSGA-III/RVEA/C-TAEA; reference points for R-NSGA-II/III).
        var options = new PymooAlgorithmOptions
        {
            NumberOfPartitions = 12,
            ReferencePoints = new[] { new[] { 0.3, 0.7 }, new[] { 0.7, 0.3 } },
        };
        return new PymooSampler(algorithm, populationSize: populationSize, seed: seed, options: options);
    }

    // ---- behavioral: the whole zoo completes + spreads a front + reproduces -----------------------

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(PymooAlgorithm.Nsga2)]
    [DataRow(PymooAlgorithm.Nsga3)]
    [DataRow(PymooAlgorithm.RNsga2)]
    [DataRow(PymooAlgorithm.RNsga3)]
    [DataRow(PymooAlgorithm.UNsga3)]
    [DataRow(PymooAlgorithm.AgeMoea)]
    [DataRow(PymooAlgorithm.AgeMoea2)]
    [DataRow(PymooAlgorithm.Rvea)]
    [DataRow(PymooAlgorithm.SmsEmoa)]
    [DataRow(PymooAlgorithm.Spea2)]
    [DataRow(PymooAlgorithm.Ctaea)]
    public void Algorithm_CompletesAndSpreadsAFront(PymooAlgorithm algorithm)
    {
        RequirePymooOrInconclusive();

        var study = Optuna.CreateStudy(
            storage: new InMemoryStorage(),
            directions: new[] { StudyDirection.Minimize, StudyDirection.Minimize },
            sampler: MakeSampler(algorithm, populationSize: 12, seed: 1));
        study.Optimize((Func<Trial, IReadOnlyList<double>>)(t =>
        {
            double x0 = t.SuggestFloat("x0", 0, 1);
            double x1 = t.SuggestFloat("x1", 0, 1);
            return Zdt1(x0, x1);
        }), nTrials: 80);

        // Every trial finished, each with the two objective values.
        study.Trials.Should().OnlyContain(t => t.State == TrialState.Complete);
        study.Trials.Should().OnlyContain(t => t.Values!.Count == 2);

        // A non-trivial, spread Pareto front (ZDT1 after several pymoo generations).
        var front = study.BestTrials;
        front.Count.Should().BeGreaterThanOrEqualTo(2, $"{algorithm} should find a spread front");
        var f0 = front.Select(t => t.Values![0]).ToList();
        (f0.Max() - f0.Min()).Should().BeGreaterThan(0.0, "the front must span a range of the first objective");
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(PymooAlgorithm.Nsga2)]
    [DataRow(PymooAlgorithm.SmsEmoa)]
    [DataRow(PymooAlgorithm.Nsga3)]
    public void SameSeed_IsReproducible(PymooAlgorithm algorithm)
    {
        RequirePymooOrInconclusive();

        List<double> Run()
        {
            var study = Optuna.CreateStudy(
                directions: new[] { StudyDirection.Minimize, StudyDirection.Minimize },
                sampler: MakeSampler(algorithm, populationSize: 10, seed: 5));
            study.Optimize((Func<Trial, IReadOnlyList<double>>)(t =>
            {
                double x0 = t.SuggestFloat("x0", 0, 1);
                double x1 = t.SuggestFloat("x1", 0, 1);
                return Zdt1(x0, x1);
            }), nTrials: 50);
            return study.Trials.OrderBy(t => t.Number).SelectMany(t => t.Values!).ToList();
        }

        Run().Should().Equal(Run());
    }

    // ---- value-parity: the bridge reproduces a direct pymoo ask/tell replay (same seed + gen 0) -----

    private const string PymooReplaySource =
        "import numpy as np, json\n" +
        "from pymoo.core.problem import Problem\n" +
        "from pymoo.algorithms.moo.nsga2 import NSGA2\n" +
        "def _zdt1(X):\n" +
        "    X = np.atleast_2d(X)\n" +
        "    f1 = X[:, 0]; g = 1.0 + 9.0 * X[:, 1]; f2 = g * (1.0 - np.sqrt(f1 / g))\n" +
        "    return np.column_stack([f1, f2])\n" +
        "_gen0 = np.array(json.loads(gen0_json), dtype=float)\n" +
        "_P = _gen0.shape[0]\n" +
        "_HI = np.nextafter(1.0, 0.0)\n" +   // Optuna's half-open float interval: suggest_float never returns `high`
        "_a = NSGA2(pop_size=_P, sampling=_gen0)\n" +
        "_a.setup(Problem(n_var=2, n_obj=2, xl=np.zeros(2), xu=np.ones(2)), termination=('n_gen', 10**9), seed=SEED, verbose=False)\n" +
        "_params = []; _values = []\n" +
        "for _g in range(NGEN):\n" +
        "    _pop = _a.ask()\n" +
        "    _Xc = np.minimum(_pop.get('X'), _HI)\n" +          // match the C# untransform high-clip
        "    _F = _zdt1(_Xc)\n" +
        "    for _k in range(_P):\n" +
        "        _params.append([float(_Xc[_k, 0]), float(_Xc[_k, 1])])\n" +
        "        _values.append([float(_F[_k, 0]), float(_F[_k, 1])])\n" +
        "    _pop.set('F', _F)\n" +
        "    _a.tell(infills=_pop)\n" +
        "result_json = json.dumps({'params': _params, 'values': _values})\n";

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(1, 8, 6)]
    [DataRow(42, 10, 5)]
    public void Nsga2_ValueParity_MatchesLivePymooReplay(int seed, int populationSize, int nGenerations)
    {
        RequirePymooOrInconclusive();

        int nTrials = populationSize * nGenerations;
        var study = Optuna.CreateStudy(
            storage: new InMemoryStorage(),
            directions: new[] { StudyDirection.Minimize, StudyDirection.Minimize },
            sampler: new PymooSampler(PymooAlgorithm.Nsga2, populationSize: populationSize, seed: seed));
        study.Optimize((Func<Trial, IReadOnlyList<double>>)(t =>
        {
            double x0 = t.SuggestFloat("x0", 0, 1);
            double x1 = t.SuggestFloat("x1", 0, 1);
            return Zdt1(x0, x1);
        }), nTrials: nTrials);

        var csTrials = study.Trials.OrderBy(t => t.Number).ToList();

        // Generation 0 is OptunaSharp's random startup; feed it to a direct pymoo run as its initial
        // population so the two trajectories share the exact same starting point.
        double[][] gen0 = csTrials.Take(populationSize)
            .Select(t => new[] { Convert.ToDouble(t.Params["x0"]), Convert.ToDouble(t.Params["x1"]) })
            .ToArray();

        PySet("gen0_json", JsonSerializer.Serialize(gen0));
        PyExec($"SEED = {seed}\nNGEN = {nGenerations}\n");
        PyExec(PymooReplaySource);

        (double[][] pyParams, double[][] pyValues) = ParseReplay(PyStr("result_json"));

        pyParams.Length.Should().Be(nTrials);
        for (int i = 0; i < nTrials; i++)
        {
            double csX0 = Convert.ToDouble(csTrials[i].Params["x0"]);
            double csX1 = Convert.ToDouble(csTrials[i].Params["x1"]);
            csX0.Should().BeApproximately(pyParams[i][0], 1e-9, $"trial {i} x0");
            csX1.Should().BeApproximately(pyParams[i][1], 1e-9, $"trial {i} x1");

            csTrials[i].Values![0].Should().BeApproximately(pyValues[i][0], 1e-9, $"trial {i} f0");
            csTrials[i].Values![1].Should().BeApproximately(pyValues[i][1], 1e-9, $"trial {i} f1");
        }
    }

    // ---- maximize: the objective-sign-flip path (pymoo minimizes; MAXIMIZE negates) ----------------

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void Maximize_CompletesAndSpreadsAFront()
    {
        RequirePymooOrInconclusive();

        // Both objectives maximized: the bridge flips each sign before telling pymoo. The negated ZDT1
        // still has a broad trade-off front.
        var study = Optuna.CreateStudy(
            directions: new[] { StudyDirection.Maximize, StudyDirection.Maximize },
            sampler: new PymooSampler(PymooAlgorithm.Nsga2, populationSize: 12, seed: 3));
        study.Optimize((Func<Trial, IReadOnlyList<double>>)(t =>
        {
            double x0 = t.SuggestFloat("x0", 0, 1);
            double x1 = t.SuggestFloat("x1", 0, 1);
            var zdt = Zdt1(x0, x1);
            return new[] { -zdt[0], -zdt[1] };   // maximize the negated objectives
        }), nTrials: 72);

        study.Trials.Should().OnlyContain(t => t.State == TrialState.Complete);
        study.BestTrials.Count.Should().BeGreaterThanOrEqualTo(2);
    }

    private static (double[][] Params, double[][] Values) ParseReplay(string json)
    {
        using JsonDocument doc = JsonDocument.Parse(json);
        double[][] Read(string key) => doc.RootElement.GetProperty(key).EnumerateArray()
            .Select(row => row.EnumerateArray().Select(v => v.GetDouble()).ToArray())
            .ToArray();
        return (Read("params"), Read("values"));
    }
}
