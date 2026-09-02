using System.Collections.Generic;
using OptunaSharp.Distributions;
using OptunaSharp.Samplers;
using OptunaSharp.Storages;

namespace OptunaSharp.Tests.Interop;

/// <summary>
///     Full-state bit-parity of GridSampler against live Optuna. The seeded shuffle of the grid
///     fixes the exact evaluation order, and GridSampler stops itself at exhaustion, so the whole
///     study state — every trial's params, the <c>search_space</c> and <c>grid_id</c> system
///     attrs (byte-for-byte), distributions and values — must agree with Python's.
/// </summary>
[TestClass]
public class GridSamplerParityTests : OptunaInteropTestBase
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
    [DataRow(20260902)]
    public void MixedFloatIntGrid_FullState_IsBitIdentical(int seed)
    {
        PythonSession.EnsureOrInconclusive();

        var searchSpace = new Dictionary<string, IReadOnlyList<CategoricalChoice>>
        {
            ["x"] = new CategoricalChoice[] { -50, 0, 50 },
            ["y"] = new CategoricalChoice[] { -99, 0, 99 },
        };
        var csStudy = Optuna.CreateStudy(
            storage: new InMemoryStorage(), sampler: new GridSampler(searchSpace, seed: seed), studyName: "grid");
        csStudy.Optimize(t =>
        {
            double x = t.SuggestFloat("x", -100, 100);
            long y = t.SuggestInt("y", -100, 100);
            return x * x + y * y;
        }, nTrials: 20);

        AssertFullParity(csStudy,
            $"study = optuna.create_study(study_name='grid', sampler=optuna.samplers.GridSampler(" +
            $"{{'x': [-50, 0, 50], 'y': [-99, 0, 99]}}, seed={seed}))\n" +
            "def _obj(t):\n" +
            "    x = t.suggest_float('x', -100, 100)\n" +
            "    y = t.suggest_int('y', -100, 100)\n" +
            "    return x * x + y * y\n" +
            "study.optimize(_obj, n_trials=20)\n");
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(0)]
    [DataRow(7)]
    public void CategoricalAndFloatGrid_FullState_IsBitIdentical(int seed)
    {
        PythonSession.EnsureOrInconclusive();

        var searchSpace = new Dictionary<string, IReadOnlyList<CategoricalChoice>>
        {
            ["opt"] = new CategoricalChoice[] { "adam", "sgd", "rmsprop" },
            ["drop"] = new CategoricalChoice[] { 0.0, 0.5 },
        };
        var csStudy = Optuna.CreateStudy(
            storage: new InMemoryStorage(), sampler: new GridSampler(searchSpace, seed: seed), studyName: "grid");
        csStudy.Optimize(t =>
        {
            string opt = t.SuggestCategorical("opt", new CategoricalChoice[] { "adam", "sgd", "rmsprop" }).Value as string ?? "";
            double drop = t.SuggestFloat("drop", 0.0, 1.0);
            return drop + (opt == "adam" ? 1.0 : 0.0);
        }, nTrials: 20);

        AssertFullParity(csStudy,
            $"study = optuna.create_study(study_name='grid', sampler=optuna.samplers.GridSampler(" +
            $"{{'opt': ['adam', 'sgd', 'rmsprop'], 'drop': [0.0, 0.5]}}, seed={seed}))\n" +
            "def _obj(t):\n" +
            "    opt = t.suggest_categorical('opt', ['adam', 'sgd', 'rmsprop'])\n" +
            "    drop = t.suggest_float('drop', 0.0, 1.0)\n" +
            "    return drop + (1.0 if opt == 'adam' else 0.0)\n" +
            "study.optimize(_obj, n_trials=20)\n");
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void OffGridQuantizedValue_StillMatches_State()
    {
        PythonSession.EnsureOrInconclusive();

        // GridSampler ignores the step= quantization: it samples the raw grid values (-0.5, 0.5)
        // even for suggest_float(step=1), warning on both sides. State parity must still hold.
        var searchSpace = new Dictionary<string, IReadOnlyList<CategoricalChoice>>
        {
            ["x"] = new CategoricalChoice[] { -0.5, 0.5 },
        };
        var csStudy = Optuna.CreateStudy(
            storage: new InMemoryStorage(), sampler: new GridSampler(searchSpace, seed: 0), studyName: "grid");
        csStudy.Optimize(t => t.SuggestFloat("x", -5, 5, step: 1), nTrials: 5);

        AssertFullParity(csStudy,
            "study = optuna.create_study(study_name='grid', sampler=optuna.samplers.GridSampler(" +
            "{'x': [-0.5, 0.5]}, seed=0))\n" +
            "study.optimize(lambda t: t.suggest_float('x', -5, 5, step=1), n_trials=5)\n");
    }
}
