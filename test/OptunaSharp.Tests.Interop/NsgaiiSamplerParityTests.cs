using System;
using System.Collections.Generic;
using OptunaSharp.Distributions;
using OptunaSharp.Samplers;
using OptunaSharp.Samplers.Nsgaii;
using OptunaSharp.Storages;

namespace OptunaSharp.Tests.Interop;

/// <summary>
///     Full-state bit-parity of <see cref="NSGAIISampler" /> against live Optuna (PORTING_PLAN Phase
///     2.13). The default path (uniform crossover, no mutation) copies parent values verbatim and
///     resamples mutated-away parameters through the independent MT19937 <see cref="RandomSampler" />,
///     and the arithmetic crossovers (BLX-α/SBX/vSBX/SPX) are bit-exact — so every trial's
///     params/values/distributions plus the per-trial generation and the study parent-cache trial-id
///     lists agree bit-for-bit. UNDX (≥2 numerical parameters) is bit-exact too <b>when the OpenBLAS
///     backend is present</b>: its PSL basis now rides <c>np.linalg.qr</c> (LAPACK), whose column signs
///     are byte-identical to NumPy's, so <see cref="UndxCrossover_FullState_IsBitIdentical" /> is
///     host-pinned to x64 (the managed Householder fallback used on a no-native build sign-diverges —
///     see <see cref="UNDXCrossover" />).
/// </summary>
[TestClass]
public class NsgaiiSamplerParityTests : OptunaInteropTestBase
{
    private void AssertFullParity(Study csStudy, string pyCreateAndRun)
    {
        var csState = StudySnapshot.Capture(csStudy);
        PyExec(pyCreateAndRun);
        var pyState = PyStudyState("study");
        StudySnapshot.AssertParity(pyState, csState, DatetimeComparison.NullnessOnly);
    }

    /// <summary>
    ///     Pre-release review item H3, live: one sampler instance reused for a second study, the review's repro.
    ///     Population 10, 120 trials on a first in-memory study, then 25 on a second; both have study id 0. The
    ///     second study's first parent lookup used to throw <c>ArgumentOutOfRangeException: The view holds 1
    ///     trials</c>. Optuna's generation bookkeeping is a stateless scan of storage, so its second study starts at
    ///     generation 0 with its own parents, while the sampler's random stream and its intersection search-space
    ///     cache carry over on both sides. Both studies' full state, per-trial generations and parent caches
    ///     included, must be bit-identical to live Optuna's.
    /// </summary>
    [TestMethod]
    [TestCategory("RequiresPython")]
    public void ReusedForASecondStudy_BothStudies_AreBitIdentical()
    {
        PythonSession.EnsureOrInconclusive();

        var sampler = new NSGAIISampler(populationSize: 10, seed: 1);
        var directions = new[] { StudyDirection.Minimize, StudyDirection.Minimize };
        IReadOnlyList<double> Objective(Trial t) => new[] { t.SuggestFloat("x", 0, 1), 1 - t.SuggestFloat("y", 0, 1) };
        var csFirst = Optuna.CreateStudy(
            storage: new InMemoryStorage(), studyName: "nsga", sampler: sampler, directions: directions);
        csFirst.Optimize(Objective, nTrials: 120);
        var csSecond = Optuna.CreateStudy(
            storage: new InMemoryStorage(), studyName: "nsga", sampler: sampler, directions: directions);
        csSecond.Optimize(Objective, nTrials: 25);
        var csFirstState = StudySnapshot.Capture(csFirst);
        var csSecondState = StudySnapshot.Capture(csSecond);

        PyExec(
            "sampler = optuna.samplers.NSGAIISampler(population_size=10, seed=1)\n" +
            "def _obj(t):\n" +
            "    return (t.suggest_float('x', 0, 1), 1 - t.suggest_float('y', 0, 1))\n" +
            "study_first = optuna.create_study(study_name='nsga', directions=['minimize', 'minimize'], sampler=sampler)\n" +
            "study_first.optimize(_obj, n_trials=120)\n" +
            "study = optuna.create_study(study_name='nsga', directions=['minimize', 'minimize'], sampler=sampler)\n" +
            "study.optimize(_obj, n_trials=25)\n");
        StudySnapshot.AssertParity(PyStudyState("study_first"), csFirstState, DatetimeComparison.NullnessOnly);
        StudySnapshot.AssertParity(PyStudyState("study"), csSecondState, DatetimeComparison.NullnessOnly);
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(0)]
    [DataRow(42)]
    public void DefaultSingleObjective_FullState_IsBitIdentical(int seed)
    {
        PythonSession.EnsureOrInconclusive();

        // Small population forces several generations of crossover + mutation-resampling.
        var csStudy = Optuna.CreateStudy(
            storage: new InMemoryStorage(), studyName: "nsga",
            sampler: new NSGAIISampler(populationSize: 5, seed: seed), direction: StudyDirection.Minimize);
        csStudy.Optimize(t =>
        {
            double x = t.SuggestFloat("x", -5, 5);
            long y = t.SuggestInt("y", 0, 10);
            t.SuggestCategorical("z", new CategoricalChoice[] { "a", "b", "c" });
            double lx = t.SuggestFloat("lx", 1e-3, 1e3, log: true);
            return Math.Pow(x - 2, 2) + y + lx;
        }, nTrials: 20);

        AssertFullParity(csStudy,
            "study = optuna.create_study(study_name='nsga', direction='minimize', " +
            $"sampler=optuna.samplers.NSGAIISampler(population_size=5, seed={seed}))\n" +
            "def _obj(t):\n" +
            "    x = t.suggest_float('x', -5, 5)\n" +
            "    y = t.suggest_int('y', 0, 10)\n" +
            "    t.suggest_categorical('z', ['a', 'b', 'c'])\n" +
            "    lx = t.suggest_float('lx', 1e-3, 1e3, log=True)\n" +
            "    return (x - 2) ** 2 + y + lx\n" +
            "study.optimize(_obj, n_trials=20)\n");
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(0)]
    [DataRow(7)]
    public void DefaultMultiObjective_FullState_IsBitIdentical(int seed)
    {
        PythonSession.EnsureOrInconclusive();

        var csStudy = Optuna.CreateStudy(
            storage: new InMemoryStorage(), studyName: "nsga",
            sampler: new NSGAIISampler(populationSize: 5, seed: seed),
            directions: new[] { StudyDirection.Minimize, StudyDirection.Maximize });
        csStudy.Optimize(t =>
        {
            double x = t.SuggestFloat("x", -5, 5);
            double y = t.SuggestFloat("y", -5, 5);
            return (IReadOnlyList<double>)new[] { x * x + y, -(Math.Pow(x - 2, 2)) - y };
        }, nTrials: 20);

        AssertFullParity(csStudy,
            "study = optuna.create_study(study_name='nsga', directions=['minimize', 'maximize'], " +
            $"sampler=optuna.samplers.NSGAIISampler(population_size=5, seed={seed}))\n" +
            "def _obj(t):\n" +
            "    x = t.suggest_float('x', -5, 5)\n" +
            "    y = t.suggest_float('y', -5, 5)\n" +
            "    return (x * x + y, -((x - 2) ** 2) - y)\n" +
            "study.optimize(_obj, n_trials=20)\n");
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow("uniform", 1)]
    [DataRow("blx", 1)]
    [DataRow("sbx", 2)]
    [DataRow("vsbx", 1)]
    [DataRow("spx", 3)]
    public void Crossover_FullState_IsBitIdentical(string kind, int seed)
    {
        PythonSession.EnsureOrInconclusive();

        BaseCrossover csCrossover = kind switch
        {
            "uniform" => new UniformCrossover(),
            "blx" => new BLXAlphaCrossover(),
            "sbx" => new SBXCrossover(),
            "vsbx" => new VSBXCrossover(),
            "spx" => new SPXCrossover(),
            _ => throw new ArgumentException(kind),
        };
        string pyCrossover = kind switch
        {
            "uniform" => "UniformCrossover()",
            "blx" => "BLXAlphaCrossover()",
            "sbx" => "SBXCrossover()",
            "vsbx" => "VSBXCrossover()",
            "spx" => "SPXCrossover()",
            _ => throw new ArgumentException(kind),
        };

        // pop=6 covers SPX's n_parents=3; MO exercises the eta=20 branch of SBX/vSBX.
        var csStudy = Optuna.CreateStudy(
            storage: new InMemoryStorage(), studyName: "nsga",
            sampler: new NSGAIISampler(populationSize: 6, seed: seed, crossover: csCrossover),
            directions: new[] { StudyDirection.Minimize, StudyDirection.Maximize });
        csStudy.Optimize(t =>
        {
            double a = t.SuggestFloat("a", -5, 5);
            double b = t.SuggestFloat("b", -5, 5);
            double c = t.SuggestFloat("c", -5, 5);
            return (IReadOnlyList<double>)new[] { a * a + b * b + c * c, a - b + c };
        }, nTrials: 24);

        AssertFullParity(csStudy,
            "study = optuna.create_study(study_name='nsga', directions=['minimize', 'maximize'], " +
            $"sampler=optuna.samplers.NSGAIISampler(population_size=6, seed={seed}, " +
            $"crossover=optuna.samplers.nsgaii.{pyCrossover}))\n" +
            "def _obj(t):\n" +
            "    a = t.suggest_float('a', -5, 5)\n" +
            "    b = t.suggest_float('b', -5, 5)\n" +
            "    c = t.suggest_float('c', -5, 5)\n" +
            "    return (a * a + b * b + c * c, a - b + c)\n" +
            "study.optimize(_obj, n_trials=24)\n");
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(0)]
    [DataRow(7)]
    [DataRow(42)]
    public void UndxCrossover_FullState_IsBitIdentical(int seed)
    {
        // UNDX's orthonormal PSL basis rides np.linalg.qr (LAPACK) — bit-exact vs Optuna only where
        // OpenBLAS's LAPACK column signs match NumPy's, i.e. host-pinned to x64 like the linalg /
        // AGE-MOEA-2 parity tiers. (Without OpenBLAS the managed Householder fallback sign-diverges;
        // that path makes no byte-parity claim, so it is out of scope for this gate.) pop=6 covers
        // UNDX's n_parents=3; three float params exercise the ≥2-numerical basis that used to diverge.
        PythonSession.EnsureBitExactHostOrInconclusive();

        var csStudy = Optuna.CreateStudy(
            storage: new InMemoryStorage(), studyName: "nsga",
            sampler: new NSGAIISampler(populationSize: 6, seed: seed, crossover: new UNDXCrossover()),
            directions: new[] { StudyDirection.Minimize, StudyDirection.Maximize });
        csStudy.Optimize(t =>
        {
            double a = t.SuggestFloat("a", -5, 5);
            double b = t.SuggestFloat("b", -5, 5);
            double c = t.SuggestFloat("c", -5, 5);
            return (IReadOnlyList<double>)new[] { a * a + b * b + c * c, a - b + c };
        }, nTrials: 24);

        AssertFullParity(csStudy,
            "study = optuna.create_study(study_name='nsga', directions=['minimize', 'maximize'], " +
            $"sampler=optuna.samplers.NSGAIISampler(population_size=6, seed={seed}, " +
            "crossover=optuna.samplers.nsgaii.UNDXCrossover()))\n" +
            "def _obj(t):\n" +
            "    a = t.suggest_float('a', -5, 5)\n" +
            "    b = t.suggest_float('b', -5, 5)\n" +
            "    c = t.suggest_float('c', -5, 5)\n" +
            "    return (a * a + b * b + c * c, a - b + c)\n" +
            "study.optimize(_obj, n_trials=24)\n");
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(0)]
    [DataRow(5)]
    public void PolynomialMutation_FullState_IsBitIdentical(int seed)
    {
        PythonSession.EnsureOrInconclusive();

        var csStudy = Optuna.CreateStudy(
            storage: new InMemoryStorage(), studyName: "nsga",
            sampler: new NSGAIISampler(
                populationSize: 5, seed: seed, mutation: new PolynomialMutation(), mutationProb: 0.5),
            direction: StudyDirection.Minimize);
        csStudy.Optimize(t =>
        {
            double a = t.SuggestFloat("a", -5, 5);
            long b = t.SuggestInt("b", 0, 20);
            return a * a + b;
        }, nTrials: 20);

        AssertFullParity(csStudy,
            "study = optuna.create_study(study_name='nsga', direction='minimize', " +
            $"sampler=optuna.samplers.NSGAIISampler(population_size=5, seed={seed}, " +
            "mutation=optuna.samplers.nsgaii.PolynomialMutation(), mutation_prob=0.5))\n" +
            "def _obj(t):\n" +
            "    a = t.suggest_float('a', -5, 5)\n" +
            "    b = t.suggest_int('b', 0, 20)\n" +
            "    return a * a + b\n" +
            "study.optimize(_obj, n_trials=20)\n");
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(0)]
    [DataRow(11)]
    public void ConstrainedSetConstraint_FullState_IsBitIdentical(int seed)
    {
        PythonSession.EnsureOrInconclusive();

        var csStudy = Optuna.CreateStudy(
            storage: new InMemoryStorage(), studyName: "nsga",
            sampler: new NSGAIISampler(populationSize: 5, seed: seed),
            directions: new[] { StudyDirection.Minimize, StudyDirection.Maximize });
        csStudy.Optimize(t =>
        {
            double x = t.SuggestFloat("x", -5, 5);
            double y = t.SuggestFloat("y", -5, 5);
            t.SetConstraint("c0", x + y - 2.0);
            return (IReadOnlyList<double>)new[] { x * x + y * y, x - y };
        }, nTrials: 24);

        AssertFullParity(csStudy,
            "study = optuna.create_study(study_name='nsga', directions=['minimize', 'maximize'], " +
            $"sampler=optuna.samplers.NSGAIISampler(population_size=5, seed={seed}))\n" +
            "def _obj(t):\n" +
            "    x = t.suggest_float('x', -5, 5)\n" +
            "    y = t.suggest_float('y', -5, 5)\n" +
            "    t.set_constraint('c0', x + y - 2.0)\n" +
            "    return (x * x + y * y, x - y)\n" +
            "study.optimize(_obj, n_trials=24)\n");
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(3)]
    public void ConstrainedConstraintsFunc_FullState_IsBitIdentical(int seed)
    {
        PythonSession.EnsureOrInconclusive();

        // The deprecated constraints_func path stores an old-format list constraint in system attrs.
        var csStudy = Optuna.CreateStudy(
            storage: new InMemoryStorage(), studyName: "nsga",
            sampler: new NSGAIISampler(
                populationSize: 5, seed: seed,
                constraintsFunc: t => new[] { Convert.ToDouble(t.Params["x"]) + Convert.ToDouble(t.Params["y"]) - 1.0 }),
            directions: new[] { StudyDirection.Minimize, StudyDirection.Maximize });
        csStudy.Optimize(t =>
        {
            double x = t.SuggestFloat("x", -5, 5);
            double y = t.SuggestFloat("y", -5, 5);
            return (IReadOnlyList<double>)new[] { x * x + y * y, x - y };
        }, nTrials: 20);

        AssertFullParity(csStudy,
            "def _con(t):\n" +
            "    return [t.params['x'] + t.params['y'] - 1.0]\n" +
            "study = optuna.create_study(study_name='nsga', directions=['minimize', 'maximize'], " +
            $"sampler=optuna.samplers.NSGAIISampler(population_size=5, seed={seed}, constraints_func=_con))\n" +
            "def _obj(t):\n" +
            "    x = t.suggest_float('x', -5, 5)\n" +
            "    y = t.suggest_float('y', -5, 5)\n" +
            "    return (x * x + y * y, x - y)\n" +
            "study.optimize(_obj, n_trials=20)\n");
    }

    /// <summary>
    ///     Plan 13 gate (docs/plans/13-rustuna-derived-performance.md, phase 0): the default NSGA-II over 40 float
    ///     parameters for 2,000 trials (40 generations of 50), the workload whose suggestions item S4 now writes in place.
    ///     Every trial's full state, generations and parent caches included, is bit-identical to live Optuna's.
    /// </summary>
    [TestMethod]
    [TestCategory("RequiresPython")]
    [TestCategory("Slow")]
    public void LongRun_40Parameters_2000Trials_FullState_IsBitIdentical()
    {
        PythonSession.EnsureOrInconclusive();

        var csStudy = Optuna.CreateStudy(
            storage: new InMemoryStorage(), studyName: "nsga", sampler: new NSGAIISampler(seed: 0),
            directions: new[] { StudyDirection.Minimize, StudyDirection.Minimize });
        csStudy.Optimize(t =>
        {
            double a = 0, b = 0;
            for (int i = 0; i < 40; i++)
            {
                double v = t.SuggestFloat($"x{i}", -10, 10);
                a += v * v;
                b += (v - 2) * (v - 2);
            }
            return new[] { a, b };
        }, nTrials: 2000);

        AssertFullParity(csStudy,
            "study = optuna.create_study(study_name='nsga', directions=['minimize', 'minimize'], " +
            "sampler=optuna.samplers.NSGAIISampler(seed=0))\n" +
            "def _obj(t):\n" +
            "    a = 0.0\n" +
            "    b = 0.0\n" +
            "    for i in range(40):\n" +
            "        v = t.suggest_float(f'x{i}', -10, 10)\n" +
            "        a += v * v\n" +
            "        b += (v - 2) * (v - 2)\n" +
            "    return a, b\n" +
            "study.optimize(_obj, n_trials=2000)\n");
    }
}
