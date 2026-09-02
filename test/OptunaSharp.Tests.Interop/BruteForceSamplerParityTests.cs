using OptunaSharp.Distributions;
using OptunaSharp.Samplers;
using OptunaSharp.Storages;

namespace OptunaSharp.Tests.Interop;

/// <summary>
///     Full-state bit-parity of BruteForceSampler against live Optuna (the vendored 5.0.0.dev
///     reference). The exploration order is fixed by the seed through <c>sample_child</c>'s
///     <c>rng.choice(p=weights)</c>, and the sampler self-stops at exhaustion, so the entire study
///     state — every trial's params, <c>params_internal</c>, distribution JSON, values, and order —
///     must agree with Python's. Covers dynamic (define-by-run) spaces, plain grids, the
///     <c>avoid_premature_stop</c> criterion, and the NaN/±inf categorical quirk (<c>_is_nan</c>).
/// </summary>
[TestClass]
public class BruteForceSamplerParityTests : OptunaInteropTestBase
{
    private void AssertFullParity(Study csStudy, string pyCreateAndRun)
    {
        var csState = StudySnapshot.Capture(csStudy);
        PyExec(pyCreateAndRun);
        var pyState = PyStudyState("study");
        StudySnapshot.AssertParity(pyState, csState, DatetimeComparison.NullnessOnly);
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(0)]
    [DataRow(1)]
    [DataRow(42)]
    [DataRow(7)]
    public void ConditionalDynamicSpace_FullState_IsBitIdentical(int seed)
    {
        PythonSession.EnsureOrInconclusive();

        var csStudy = Optuna.CreateStudy(
            storage: new InMemoryStorage(), sampler: new BruteForceSampler(seed: seed), studyName: "bf");
        csStudy.Optimize(t =>
        {
            long a = t.SuggestInt("a", 0, 2);
            if (a == 0)
                return a + t.SuggestFloat("b", -1.0, 1.0, step: 0.5);
            if (a == 1)
            {
                var c = t.SuggestCategorical("c", new CategoricalChoice[] { "x", "y", CategoricalChoice.None });
                return (c.Value as string) == "x" ? a + 1 : a - 1;
            }

            return a * 2;
        });

        AssertFullParity(csStudy,
            $"study = optuna.create_study(study_name='bf', sampler=optuna.samplers.BruteForceSampler(seed={seed}))\n" +
            "def _obj(t):\n" +
            "    a = t.suggest_int('a', 0, 2)\n" +
            "    if a == 0:\n" +
            "        return a + t.suggest_float('b', -1.0, 1.0, step=0.5)\n" +
            "    if a == 1:\n" +
            "        c = t.suggest_categorical('c', ['x', 'y', None])\n" +
            "        return a + 1 if c == 'x' else a - 1\n" +
            "    return a * 2\n" +
            "study.optimize(_obj)\n");
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(0)]
    [DataRow(42)]
    public void IntGrid_FullState_IsBitIdentical(int seed)
    {
        PythonSession.EnsureOrInconclusive();

        var csStudy = Optuna.CreateStudy(
            storage: new InMemoryStorage(), sampler: new BruteForceSampler(seed: seed), studyName: "bf");
        csStudy.Optimize(t => (double)(t.SuggestInt("x", 0, 2) + t.SuggestInt("y", 0, 2)));

        AssertFullParity(csStudy,
            $"study = optuna.create_study(study_name='bf', sampler=optuna.samplers.BruteForceSampler(seed={seed}))\n" +
            "study.optimize(lambda t: float(t.suggest_int('x', 0, 2) + t.suggest_int('y', 0, 2)))\n");
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(0)]
    [DataRow(42)]
    public void AvoidPrematureStop_FullState_IsBitIdentical(int seed)
    {
        PythonSession.EnsureOrInconclusive();

        var csStudy = Optuna.CreateStudy(
            storage: new InMemoryStorage(),
            sampler: new BruteForceSampler(seed: seed, avoidPrematureStop: true),
            studyName: "bf");
        csStudy.Optimize(t => (double)(t.SuggestInt("x", 0, 1) + t.SuggestInt("y", 0, 1)));

        AssertFullParity(csStudy,
            $"study = optuna.create_study(study_name='bf', sampler=optuna.samplers.BruteForceSampler(" +
            $"seed={seed}, avoid_premature_stop=True))\n" +
            "study.optimize(lambda t: float(t.suggest_int('x', 0, 1) + t.suggest_int('y', 0, 1)))\n");
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void WeirdNanInfNoneChoices_FullState_IsBitIdentical()
    {
        PythonSession.EnsureOrInconclusive();

        // [inf, -inf, nan, None] over three params exercises the _is_nan subset filter that lets the
        // sampler recognize NaN-containing trials as visited (4**3 = 64 unique combinations).
        var weird = new CategoricalChoice[]
            { double.PositiveInfinity, double.NegativeInfinity, double.NaN, CategoricalChoice.None };
        var csStudy = Optuna.CreateStudy(
            storage: new InMemoryStorage(), sampler: new BruteForceSampler(seed: 0), studyName: "bf");
        csStudy.Optimize(t =>
        {
            for (int i = 0; i < 3; i++)
                t.SuggestCategorical($"c{i}", weird);
            return 0.0;
        });

        AssertFullParity(csStudy,
            "study = optuna.create_study(study_name='bf', sampler=optuna.samplers.BruteForceSampler(seed=0))\n" +
            "weird = [float('inf'), -float('inf'), float('nan'), None]\n" +
            "def _obj(t):\n" +
            "    for i in range(3):\n" +
            "        t.suggest_categorical(f'c{i}', weird)\n" +
            "    return 0.0\n" +
            "study.optimize(_obj)\n");
    }
}
