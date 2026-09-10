using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json;
using AwesomeAssertions;
using OptunaSharp.Distributions;
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

    // A direct pymoo NSGA-II ask/tell run over a plain 2-var unit-cube ZDT1, seeded with the SAME
    // generation 0 the bridge started from, with the SAME operators, and (optionally) the SAME
    // g0 = x0 - 0.5 constraint. It mirrors the bridge exactly — including the Optuna half-open-interval
    // high-clip (min(x, nextafter(1,0))) applied before evaluating — so a faithful bridge reproduces it.
    private const string PymooReplaySource =
        "import numpy as np, json\n" +
        "from pymoo.core.problem import Problem\n" +
        "from pymoo.algorithms.moo.nsga2 import NSGA2\n" +
        "from pymoo.operators.crossover.sbx import SBX\n" +
        "from pymoo.operators.mutation.pm import PM\n" +
        "def _zdt1(X):\n" +
        "    X = np.atleast_2d(X)\n" +
        "    f1 = X[:, 0]; g = 1.0 + 9.0 * X[:, 1]; f2 = g * (1.0 - np.sqrt(f1 / g))\n" +
        "    return np.column_stack([f1, f2])\n" +
        "_gen0 = np.array(json.loads(gen0_json), dtype=float)\n" +
        "_P = _gen0.shape[0]\n" +
        "_HI = np.nextafter(1.0, 0.0)\n" +
        "_kw = dict(pop_size=_P, sampling=_gen0)\n" +
        "if CX_ETA is not None or CX_PROB is not None:\n" +
        "    _ck = {}\n" +
        "    if CX_ETA is not None: _ck['eta'] = CX_ETA\n" +
        "    if CX_PROB is not None: _ck['prob'] = CX_PROB\n" +
        "    _kw['crossover'] = SBX(**_ck)\n" +
        "if MUT_ETA is not None or MUT_PROB is not None:\n" +
        "    _mk = {}\n" +
        "    if MUT_ETA is not None: _mk['eta'] = MUT_ETA\n" +
        "    if MUT_PROB is not None: _mk['prob'] = MUT_PROB\n" +
        "    _kw['mutation'] = PM(**_mk)\n" +
        "_a = NSGA2(**_kw)\n" +
        "_a.setup(Problem(n_var=2, n_obj=2, n_ieq_constr=(1 if CONSTRAINED else 0), xl=np.zeros(2), xu=np.ones(2)), termination=('n_gen', 10**9), seed=SEED, verbose=False)\n" +
        "_params = []; _values = []\n" +
        "for _g in range(NGEN):\n" +
        "    _pop = _a.ask()\n" +
        "    _Xc = np.minimum(_pop.get('X'), _HI)\n" +
        "    _F = _zdt1(_Xc)\n" +
        "    for _k in range(_P):\n" +
        "        _params.append([float(_Xc[_k, 0]), float(_Xc[_k, 1])])\n" +
        "        _values.append([float(_F[_k, 0]), float(_F[_k, 1])])\n" +
        "    _pop.set('F', _F)\n" +
        "    if CONSTRAINED:\n" +
        "        _pop.set('G', (_Xc[:, 0] - 0.5).reshape(-1, 1))\n" +
        "    _a.tell(infills=_pop)\n" +
        "result_json = json.dumps({'params': _params, 'values': _values})\n";

    private static string PyNum(double? v) => v is { } d ? d.ToString("R", System.Globalization.CultureInfo.InvariantCulture) : "None";

    /// <summary>Run the direct pymoo replay seeded with <paramref name="gen0" />, returning its per-trial
    /// (params, objectives) so the C# study's trajectory can be diffed against it.</summary>
    private (double[][] Params, double[][] Values) RunReplay(
        double[][] gen0, int seed, int nGenerations,
        double? cxEta = null, double? cxProb = null, double? mutEta = null, double? mutProb = null,
        bool constrained = false)
    {
        PySet("gen0_json", JsonSerializer.Serialize(gen0));
        PyExec($"SEED = {seed}\nNGEN = {nGenerations}\n" +
               $"CX_ETA = {PyNum(cxEta)}\nCX_PROB = {PyNum(cxProb)}\n" +
               $"MUT_ETA = {PyNum(mutEta)}\nMUT_PROB = {PyNum(mutProb)}\n" +
               $"CONSTRAINED = {(constrained ? "True" : "False")}\n");
        PyExec(PymooReplaySource);
        return ParseReplay(PyStr("result_json"));
    }

    private static double[][] Gen0Of(IReadOnlyList<FrozenTrial> csTrials, int populationSize)
        => csTrials.Take(populationSize)
            .Select(t => new[] { Convert.ToDouble(t.Params["x0"]), Convert.ToDouble(t.Params["x1"]) })
            .ToArray();

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
        (double[][] pyParams, double[][] pyValues) = RunReplay(Gen0Of(csTrials, populationSize), seed, nGenerations);

        AssertTrajectoryMatches(csTrials, pyParams, pyValues, nTrials);
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void CustomOperators_ValueParity_MatchesLivePymooReplay()
    {
        RequirePymooOrInconclusive();

        const int seed = 3, P = 10, G = 6;
        var options = new PymooAlgorithmOptions
        {
            CrossoverEta = 30, CrossoverProbability = 0.7, MutationEta = 10, MutationProbability = 0.2,
        };
        var study = Optuna.CreateStudy(
            storage: new InMemoryStorage(),
            directions: new[] { StudyDirection.Minimize, StudyDirection.Minimize },
            sampler: new PymooSampler(PymooAlgorithm.Nsga2, populationSize: P, seed: seed, options: options));
        study.Optimize((Func<Trial, IReadOnlyList<double>>)(t =>
            Zdt1(t.SuggestFloat("x0", 0, 1), t.SuggestFloat("x1", 0, 1))), nTrials: P * G);

        var csTrials = study.Trials.OrderBy(t => t.Number).ToList();
        (double[][] pyParams, double[][] pyValues) =
            RunReplay(Gen0Of(csTrials, P), seed, G, cxEta: 30, cxProb: 0.7, mutEta: 10, mutProb: 0.2);

        AssertTrajectoryMatches(csTrials, pyParams, pyValues, P * G);
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void Constrained_ValueParity_AndRespectsFeasibility()
    {
        RequirePymooOrInconclusive();

        const int seed = 7, P = 10, G = 6;
        // g0 = x0 - 0.5 (feasible when x0 <= 0.5) — pymoo's G <= 0 convention == Optuna's constraint sign.
        var study = Optuna.CreateStudy(
            storage: new InMemoryStorage(),
            directions: new[] { StudyDirection.Minimize, StudyDirection.Minimize },
            sampler: new PymooSampler(
                PymooAlgorithm.Nsga2, populationSize: P, seed: seed,
                constraintsFunc: t => new[] { Convert.ToDouble(t.Params["x0"]) - 0.5 }));
        study.Optimize((Func<Trial, IReadOnlyList<double>>)(t =>
            Zdt1(t.SuggestFloat("x0", 0, 1), t.SuggestFloat("x1", 0, 1))), nTrials: P * G);

        var csTrials = study.Trials.OrderBy(t => t.Number).ToList();
        (double[][] pyParams, double[][] pyValues) = RunReplay(Gen0Of(csTrials, P), seed, G, constrained: true);
        AssertTrajectoryMatches(csTrials, pyParams, pyValues, P * G);

        // The final Pareto front must be (essentially all) feasible — the constraint steered the search.
        double feasible = study.BestTrials.Count(t => Convert.ToDouble(t.Params["x0"]) <= 0.5 + 1e-9);
        (feasible / study.BestTrials.Count).Should().BeGreaterThanOrEqualTo(0.9, "the front should be feasible");
    }

    private static void AssertTrajectoryMatches(
        IReadOnlyList<FrozenTrial> csTrials, double[][] pyParams, double[][] pyValues, int nTrials)
    {
        pyParams.Length.Should().Be(nTrials);
        for (int i = 0; i < nTrials; i++)
        {
            Convert.ToDouble(csTrials[i].Params["x0"]).Should().BeApproximately(pyParams[i][0], 1e-9, $"trial {i} x0");
            Convert.ToDouble(csTrials[i].Params["x1"]).Should().BeApproximately(pyParams[i][1], 1e-9, $"trial {i} x1");
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

    // ---- edge cases: objective count, Optuna's full parameter space, dynamic space, validation --------

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void SingleObjective_CompletesAndImprovesOnTheFirstGeneration()
    {
        RequirePymooOrInconclusive();

        // A MOEA degenerates to a GA on one objective — the bridge still drives it (n_obj = 1).
        var study = Optuna.CreateStudy(
            storage: new InMemoryStorage(),
            sampler: new PymooSampler(PymooAlgorithm.Nsga2, populationSize: 12, seed: 1));
        study.Optimize(t =>
        {
            double a = t.SuggestFloat("a", 0, 1);
            double b = t.SuggestFloat("b", 0, 1);
            double c = t.SuggestFloat("c", 0, 1);
            return a * a + b * b + c * c;   // sphere, minimum 0 at the origin
        }, nTrials: 96);

        study.Trials.Should().OnlyContain(t => t.State == TrialState.Complete);
        var ordered = study.Trials.OrderBy(t => t.Number).ToList();
        double gen0Best = ordered.Take(12).Min(t => t.Value!.Value);
        double overallBest = ordered.Min(t => t.Value!.Value);
        overallBest.Should().BeLessThan(gen0Best, "pymoo should improve on the random first generation");
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void ThreeObjective_CompletesAndSpreadsAFront()
    {
        RequirePymooOrInconclusive();

        var study = Optuna.CreateStudy(
            directions: new[] { StudyDirection.Minimize, StudyDirection.Minimize, StudyDirection.Minimize },
            sampler: new PymooSampler(PymooAlgorithm.Nsga2, populationSize: 20, seed: 2));
        study.Optimize((Func<Trial, IReadOnlyList<double>>)(t =>
        {
            double x0 = t.SuggestFloat("x0", 0, 1);
            double x1 = t.SuggestFloat("x1", 0, 1);
            // conflicting trade-off: minimizing the first two objectives pushes the third up.
            return new[] { x0, x1, (1 - x0) + (1 - x1) };
        }), nTrials: 120);

        study.Trials.Should().OnlyContain(t => t.State == TrialState.Complete);
        study.Trials.Should().OnlyContain(t => t.Values!.Count == 3);
        study.BestTrials.Count.Should().BeGreaterThanOrEqualTo(3);
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void IntLogStepParams_RespectBoundsAndReproduce()
    {
        RequirePymooOrInconclusive();

        List<(long I, long J, long K, double F)> Run()
        {
            var study = Optuna.CreateStudy(
                directions: new[] { StudyDirection.Minimize, StudyDirection.Minimize },
                sampler: new PymooSampler(PymooAlgorithm.Nsga2, populationSize: 10, seed: 5));
            study.Optimize((Func<Trial, IReadOnlyList<double>>)(t =>
            {
                long i = t.SuggestInt("i", 0, 20);
                long j = t.SuggestInt("j", 1, 100, log: true);
                long k = t.SuggestInt("k", -10, 10, step: 2);
                double f = t.SuggestFloat("f", 0, 10, step: 0.5);
                return new[] { (double)(i + j), k + f };
            }), nTrials: 50);
            return study.Trials.OrderBy(t => t.Number)
                .Select(t => (Convert.ToInt64(t.Params["i"]), Convert.ToInt64(t.Params["j"]),
                              Convert.ToInt64(t.Params["k"]), Convert.ToDouble(t.Params["f"])))
                .ToList();
        }

        var r1 = Run();
        r1.Should().OnlyContain(x => x.I >= 0 && x.I <= 20);
        r1.Should().OnlyContain(x => x.J >= 1 && x.J <= 100);
        r1.Should().OnlyContain(x => x.K >= -10 && x.K <= 10 && (x.K + 10) % 2 == 0);
        r1.Should().OnlyContain(x => x.F >= 0 && x.F <= 10 && Math.Abs(x.F / 0.5 - Math.Round(x.F / 0.5)) < 1e-9);
        r1.Should().Equal(Run(), "the same seed must reproduce the whole int/log/step trajectory");
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void CategoricalParams_StayInChoicesAndReproduce()
    {
        RequirePymooOrInconclusive();

        var choices = new CategoricalChoice[] { "a", "b", "c", "d" };
        List<string> Run()
        {
            var study = Optuna.CreateStudy(
                directions: new[] { StudyDirection.Minimize, StudyDirection.Minimize },
                sampler: new PymooSampler(PymooAlgorithm.Nsga2, populationSize: 10, seed: 4));
            study.Optimize((Func<Trial, IReadOnlyList<double>>)(t =>
            {
                double x = t.SuggestFloat("x", 0, 1);
                t.SuggestCategorical("c", choices);   // exercises the one-hot transform + argmax untransform
                return new[] { x, 1 - x };
            }), nTrials: 50);
            return study.Trials.OrderBy(t => t.Number).Select(t => t.Params["c"]!.ToString()!).ToList();
        }

        var r1 = Run();
        r1.Should().OnlyContain(v => v == "a" || v == "b" || v == "c" || v == "d");
        r1.Should().Equal(Run(), "the same seed must reproduce the categorical choices");
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void DynamicSearchSpace_ConditionalParamFallsBackToIndependent()
    {
        RequirePymooOrInconclusive();

        // `y` is only suggested when x > 0.5, so it is NOT in the intersection search space — the bridge
        // must sample it independently (never feed it to pymoo) and still complete every trial.
        var study = Optuna.CreateStudy(
            directions: new[] { StudyDirection.Minimize, StudyDirection.Minimize },
            sampler: new PymooSampler(PymooAlgorithm.Nsga2, populationSize: 10, seed: 8));
        study.Optimize((Func<Trial, IReadOnlyList<double>>)(t =>
        {
            double x = t.SuggestFloat("x", 0, 1);
            double extra = x > 0.5 ? t.SuggestFloat("y", 0, 1) : 0.0;
            return new[] { x, (1 - x) + extra };
        }), nTrials: 60);

        study.Trials.Should().OnlyContain(t => t.State == TrialState.Complete);
        study.Trials.Count(t => t.Params.ContainsKey("y")).Should().BeGreaterThan(0, "the conditional param must appear");
        study.BestTrials.Count.Should().BeGreaterThanOrEqualTo(2);
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void EliminateDuplicatesFalse_CompletesAndReproduces()
    {
        RequirePymooOrInconclusive();

        List<double> Run()
        {
            var study = Optuna.CreateStudy(
                directions: new[] { StudyDirection.Minimize, StudyDirection.Minimize },
                sampler: new PymooSampler(
                    PymooAlgorithm.Nsga2, populationSize: 10, seed: 6,
                    options: new PymooAlgorithmOptions { EliminateDuplicates = false }));
            study.Optimize((Func<Trial, IReadOnlyList<double>>)(t =>
                Zdt1(t.SuggestFloat("x0", 0, 1), t.SuggestFloat("x1", 0, 1))), nTrials: 50);
            return study.Trials.OrderBy(t => t.Number).SelectMany(t => t.Values!).ToList();
        }

        Run().Should().Equal(Run());
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void ReferencePointAlgorithm_WithoutReferencePoints_Throws()
    {
        RequirePymooOrInconclusive();

        // R-NSGA-II requires reference points; running without them raises a clear ValueError.
        var study = Optuna.CreateStudy(
            directions: new[] { StudyDirection.Minimize, StudyDirection.Minimize },
            sampler: new PymooSampler(PymooAlgorithm.RNsga2, populationSize: 10, seed: 1));
        Action act = () => study.Optimize((Func<Trial, IReadOnlyList<double>>)(t =>
            Zdt1(t.SuggestFloat("x0", 0, 1), t.SuggestFloat("x1", 0, 1))), nTrials: 30);
        act.Should().Throw<NumSharp.ValueError>().WithMessage("*requires reference points*");
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
