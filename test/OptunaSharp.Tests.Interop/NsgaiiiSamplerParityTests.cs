using System;
using System.Collections.Generic;
using OptunaSharp.Distributions;
using OptunaSharp.Samplers;
using OptunaSharp.Samplers.Nsgaii;
using OptunaSharp.Storages;

namespace OptunaSharp.Tests.Interop;

/// <summary>
///     Full-state bit-parity of <see cref="NSGAIIISampler" /> against live Optuna (PORTING_PLAN Phase
///     2.14). NSGA-III reuses the NSGA-II genetic machinery and swaps in reference-point niching. The
///     whole niching pipeline (filter-inf, ideal-point normalization, ASF extreme points, reference-point
///     association, and the survivor <c>rng.shuffle</c>) is bit-exact — the hyperplane-fit
///     <c>matrix_rank</c>/<c>solve</c> route through NumSharp's OpenBLAS LAPACK backend (byte-identical to
///     NumPy), resolving the earlier ~1-ULP managed-LU drift — so every trial's params/values/distributions
///     plus the per-trial generation and the study parent-cache trial-id lists agree bit-for-bit here.
/// </summary>
[TestClass]
public class NsgaiiiSamplerParityTests : OptunaInteropTestBase
{
    private void AssertFullParity(Study csStudy, string pyCreateAndRun)
    {
        var csState = StudySnapshot.Capture(csStudy);
        PyExec(pyCreateAndRun);
        var pyState = PyStudyState("study");
        StudySnapshot.AssertParity(pyState, csState, DatetimeComparison.NullnessOnly);
    }

    /// <summary>
    ///     Pre-release review item H3, live, for NSGA-III: one sampler instance reused for a second in-memory study
    ///     (both have study id 0). Population 10, 120 trials, then 25. The second study's first parent lookup used to
    ///     throw <c>ArgumentOutOfRangeException: The view holds 1 trials</c>. Optuna starts the second study at
    ///     generation 0 with its own parents, while the sampler's random stream (which reference-point niching also
    ///     draws from) and its intersection search-space cache carry over. Both studies' full state, generations and
    ///     parent caches included, must be bit-identical to live Optuna's.
    /// </summary>
    [TestMethod]
    [TestCategory("RequiresPython")]
    public void ReusedForASecondStudy_BothStudies_AreBitIdentical()
    {
        PythonSession.EnsureOrInconclusive();

        var sampler = new NSGAIIISampler(populationSize: 10, seed: 1);
        var directions = new[] { StudyDirection.Minimize, StudyDirection.Minimize };
        IReadOnlyList<double> Objective(Trial t) => new[] { t.SuggestFloat("x", 0, 1), 1 - t.SuggestFloat("y", 0, 1) };
        var csFirst = Optuna.CreateStudy(
            storage: new InMemoryStorage(), studyName: "nsga3", sampler: sampler, directions: directions);
        csFirst.Optimize(Objective, nTrials: 120);
        var csSecond = Optuna.CreateStudy(
            storage: new InMemoryStorage(), studyName: "nsga3", sampler: sampler, directions: directions);
        csSecond.Optimize(Objective, nTrials: 25);
        var csFirstState = StudySnapshot.Capture(csFirst);
        var csSecondState = StudySnapshot.Capture(csSecond);

        PyExec(
            "sampler = optuna.samplers.NSGAIIISampler(population_size=10, seed=1)\n" +
            "def _obj(t):\n" +
            "    return (t.suggest_float('x', 0, 1), 1 - t.suggest_float('y', 0, 1))\n" +
            "study_first = optuna.create_study(study_name='nsga3', directions=['minimize', 'minimize'], sampler=sampler)\n" +
            "study_first.optimize(_obj, n_trials=120)\n" +
            "study = optuna.create_study(study_name='nsga3', directions=['minimize', 'minimize'], sampler=sampler)\n" +
            "study.optimize(_obj, n_trials=25)\n");
        StudySnapshot.AssertParity(PyStudyState("study_first"), csFirstState, DatetimeComparison.NullnessOnly);
        StudySnapshot.AssertParity(PyStudyState("study"), csSecondState, DatetimeComparison.NullnessOnly);
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(0)]
    [DataRow(7)]
    public void DefaultTwoObjective_FullState_IsBitIdentical(int seed)
    {
        PythonSession.EnsureOrInconclusive();

        // Small population forces the borderline front to overflow every generation -> the niching
        // pipeline runs on a 2x2 hyperplane fit.
        var csStudy = Optuna.CreateStudy(
            storage: new InMemoryStorage(), studyName: "nsga3",
            sampler: new NSGAIIISampler(populationSize: 5, seed: seed),
            directions: new[] { StudyDirection.Minimize, StudyDirection.Maximize });
        csStudy.Optimize(t =>
        {
            double x = t.SuggestFloat("x", -5, 5);
            double y = t.SuggestFloat("y", -5, 5);
            return (IReadOnlyList<double>)new[] { x * x + y, -(Math.Pow(x - 2, 2)) - y };
        }, nTrials: 20);

        AssertFullParity(csStudy,
            "study = optuna.create_study(study_name='nsga3', directions=['minimize', 'maximize'], " +
            $"sampler=optuna.samplers.NSGAIIISampler(population_size=5, seed={seed}))\n" +
            "def _obj(t):\n" +
            "    x = t.suggest_float('x', -5, 5)\n" +
            "    y = t.suggest_float('y', -5, 5)\n" +
            "    return (x * x + y, -((x - 2) ** 2) - y)\n" +
            "study.optimize(_obj, n_trials=20)\n");
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(0)]
    [DataRow(3)]
    public void DefaultThreeObjective_FullState_IsBitIdentical(int seed)
    {
        PythonSession.EnsureOrInconclusive();

        // Three objectives is NSGA-III's target regime: a 3x3 hyperplane fit and a 10-point default
        // reference set. pop=6, 24 trials -> four generations of niching.
        var csStudy = Optuna.CreateStudy(
            storage: new InMemoryStorage(), studyName: "nsga3",
            sampler: new NSGAIIISampler(populationSize: 6, seed: seed),
            directions: new[] { StudyDirection.Minimize, StudyDirection.Minimize, StudyDirection.Maximize });
        csStudy.Optimize(t =>
        {
            double x = t.SuggestFloat("x", -5, 5);
            double y = t.SuggestFloat("y", -5, 5);
            double z = t.SuggestFloat("z", -5, 5);
            return (IReadOnlyList<double>)new[] { x * x + y, y * y + z, -(x + y + z) };
        }, nTrials: 24);

        AssertFullParity(csStudy,
            "study = optuna.create_study(study_name='nsga3', directions=['minimize', 'minimize', 'maximize'], " +
            $"sampler=optuna.samplers.NSGAIIISampler(population_size=6, seed={seed}))\n" +
            "def _obj(t):\n" +
            "    x = t.suggest_float('x', -5, 5)\n" +
            "    y = t.suggest_float('y', -5, 5)\n" +
            "    z = t.suggest_float('z', -5, 5)\n" +
            "    return (x * x + y, y * y + z, -(x + y + z))\n" +
            "study.optimize(_obj, n_trials=24)\n");
    }

    /// <summary>
    ///     The many-objective regime NSGA-III exists for: 8 and 10 minimized objectives (sums of squares around a
    ///     distinct centre each, polynomial only, so both languages evaluate them alike), population 20, 200 trials,
    ///     the default reference points (120 and 220 of them). From eight objectives on, NumPy's
    ///     <c>np.linalg.norm(..., axis=2)</c> in <c>_associate_individuals_with_reference_points</c> sums each row
    ///     pairwise; the left-to-right sum the port used until 2026-11-11 moved the reference-line distances by an ulp
    ///     and changed the survivors in both rows below: the trajectory left Optuna's at trial 80 (8 objectives, seed
    ///     4) and at trial 100 (10 objectives, seed 0); 6 of 8 seeds at 8 objectives happened to stay identical.
    /// </summary>
    /// <param name="nObjectives">The number of objectives (each over <c>nObjectives + 2</c> float parameters).</param>
    /// <param name="seed">The sampler seed.</param>
    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(8, 4)]
    [DataRow(10, 0)]
    public void ManyObjectives_FullState_IsBitIdentical(int nObjectives, int seed)
    {
        PythonSession.EnsureOrInconclusive();

        int nParams = nObjectives + 2;
        var csStudy = Optuna.CreateStudy(
            storage: new InMemoryStorage(), studyName: "nsga3",
            sampler: new NSGAIIISampler(populationSize: 20, seed: seed),
            directions: System.Linq.Enumerable.Repeat(StudyDirection.Minimize, nObjectives));
        csStudy.Optimize(t =>
        {
            var x = new double[nParams];
            for (int j = 0; j < nParams; j++)
                x[j] = t.SuggestFloat($"x{j:D2}", 0.0, 1.0);
            var f = new double[nObjectives];
            for (int m = 0; m < nObjectives; m++)
            {
                double s = 0.0;
                for (int j = 0; j < nParams; j++)
                {
                    double d = x[j] - ((m * 7 + j * 3) % 10) / 10.0;
                    s += d * d;
                }
                f[m] = s;
            }
            return (IReadOnlyList<double>)f;
        }, nTrials: 200);

        AssertFullParity(csStudy,
            "study = optuna.create_study(study_name='nsga3', " +
            $"directions=['minimize'] * {nObjectives}, " +
            $"sampler=optuna.samplers.NSGAIIISampler(population_size=20, seed={seed}))\n" +
            "def _obj(t):\n" +
            $"    x = [t.suggest_float(f'x{{j:02d}}', 0.0, 1.0) for j in range({nParams})]\n" +
            "    f = []\n" +
            $"    for m in range({nObjectives}):\n" +
            "        s = 0.0\n" +
            $"        for j in range({nParams}):\n" +
            "            d = x[j] - ((m * 7 + j * 3) % 10) / 10.0\n" +
            "            s += d * d\n" +
            "        f.append(s)\n" +
            "    return f\n" +
            "study.optimize(_obj, n_trials=200)\n");
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow("uniform", 1)]
    [DataRow("blx", 1)]
    [DataRow("sbx", 2)]
    [DataRow("spx", 3)]
    public void Crossover_FullState_IsBitIdentical(string kind, int seed)
    {
        PythonSession.EnsureOrInconclusive();

        BaseCrossover csCrossover = kind switch
        {
            "uniform" => new UniformCrossover(),
            "blx" => new BLXAlphaCrossover(),
            "sbx" => new SBXCrossover(),
            "spx" => new SPXCrossover(),
            _ => throw new ArgumentException(kind),
        };
        string pyCrossover = kind switch
        {
            "uniform" => "UniformCrossover()",
            "blx" => "BLXAlphaCrossover()",
            "sbx" => "SBXCrossover()",
            "spx" => "SPXCrossover()",
            _ => throw new ArgumentException(kind),
        };

        var csStudy = Optuna.CreateStudy(
            storage: new InMemoryStorage(), studyName: "nsga3",
            sampler: new NSGAIIISampler(populationSize: 6, seed: seed, crossover: csCrossover),
            directions: new[] { StudyDirection.Minimize, StudyDirection.Maximize });
        csStudy.Optimize(t =>
        {
            double a = t.SuggestFloat("a", -5, 5);
            double b = t.SuggestFloat("b", -5, 5);
            double c = t.SuggestFloat("c", -5, 5);
            return (IReadOnlyList<double>)new[] { a * a + b * b + c * c, a - b + c };
        }, nTrials: 24);

        AssertFullParity(csStudy,
            "study = optuna.create_study(study_name='nsga3', directions=['minimize', 'maximize'], " +
            $"sampler=optuna.samplers.NSGAIIISampler(population_size=6, seed={seed}, " +
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
    public void CustomReferencePoints_FullState_IsBitIdentical(int seed)
    {
        PythonSession.EnsureOrInconclusive();

        // A user-supplied (non-default) reference set drives the association step.
        var referencePoints = new double[,] { { 1.0, 0.0 }, { 0.5, 0.5 }, { 0.0, 1.0 } };
        var csStudy = Optuna.CreateStudy(
            storage: new InMemoryStorage(), studyName: "nsga3",
            sampler: new NSGAIIISampler(populationSize: 5, seed: seed, referencePoints: referencePoints),
            directions: new[] { StudyDirection.Minimize, StudyDirection.Maximize });
        csStudy.Optimize(t =>
        {
            double x = t.SuggestFloat("x", -5, 5);
            double y = t.SuggestFloat("y", -5, 5);
            return (IReadOnlyList<double>)new[] { x * x + y, -y };
        }, nTrials: 20);

        AssertFullParity(csStudy,
            "import numpy as np\n" +
            "rp = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])\n" +
            "study = optuna.create_study(study_name='nsga3', directions=['minimize', 'maximize'], " +
            $"sampler=optuna.samplers.NSGAIIISampler(population_size=5, seed={seed}, reference_points=rp))\n" +
            "def _obj(t):\n" +
            "    x = t.suggest_float('x', -5, 5)\n" +
            "    y = t.suggest_float('y', -5, 5)\n" +
            "    return (x * x + y, -y)\n" +
            "study.optimize(_obj, n_trials=20)\n");
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(0)]
    public void PolynomialMutation_FullState_IsBitIdentical(int seed)
    {
        PythonSession.EnsureOrInconclusive();

        var csStudy = Optuna.CreateStudy(
            storage: new InMemoryStorage(), studyName: "nsga3",
            sampler: new NSGAIIISampler(
                populationSize: 5, seed: seed, mutation: new PolynomialMutation(), mutationProb: 0.5),
            directions: new[] { StudyDirection.Minimize, StudyDirection.Maximize });
        csStudy.Optimize(t =>
        {
            double a = t.SuggestFloat("a", -5, 5);
            long b = t.SuggestInt("b", 0, 20);
            return (IReadOnlyList<double>)new[] { a * a + b, (double)-b };
        }, nTrials: 20);

        AssertFullParity(csStudy,
            "study = optuna.create_study(study_name='nsga3', directions=['minimize', 'maximize'], " +
            $"sampler=optuna.samplers.NSGAIIISampler(population_size=5, seed={seed}, " +
            "mutation=optuna.samplers.nsgaii.PolynomialMutation(), mutation_prob=0.5))\n" +
            "def _obj(t):\n" +
            "    a = t.suggest_float('a', -5, 5)\n" +
            "    b = t.suggest_int('b', 0, 20)\n" +
            "    return (a * a + b, -b)\n" +
            "study.optimize(_obj, n_trials=20)\n");
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(0)]
    public void ConstrainedSetConstraint_FullState_IsBitIdentical(int seed)
    {
        PythonSession.EnsureOrInconclusive();

        var csStudy = Optuna.CreateStudy(
            storage: new InMemoryStorage(), studyName: "nsga3",
            sampler: new NSGAIIISampler(populationSize: 5, seed: seed),
            directions: new[] { StudyDirection.Minimize, StudyDirection.Maximize });
        csStudy.Optimize(t =>
        {
            double x = t.SuggestFloat("x", -5, 5);
            double y = t.SuggestFloat("y", -5, 5);
            t.SetConstraint("c0", x + y - 2.0);
            return (IReadOnlyList<double>)new[] { x * x + y * y, x - y };
        }, nTrials: 24);

        AssertFullParity(csStudy,
            "study = optuna.create_study(study_name='nsga3', directions=['minimize', 'maximize'], " +
            $"sampler=optuna.samplers.NSGAIIISampler(population_size=5, seed={seed}))\n" +
            "def _obj(t):\n" +
            "    x = t.suggest_float('x', -5, 5)\n" +
            "    y = t.suggest_float('y', -5, 5)\n" +
            "    t.set_constraint('c0', x + y - 2.0)\n" +
            "    return (x * x + y * y, x - y)\n" +
            "study.optimize(_obj, n_trials=24)\n");
    }
}
