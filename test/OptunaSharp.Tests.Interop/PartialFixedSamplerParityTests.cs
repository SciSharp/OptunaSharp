using System.Collections.Generic;
using OptunaSharp.Samplers;
using OptunaSharp.Storages;

namespace OptunaSharp.Tests.Interop;

/// <summary>
///     Full-state bit-parity of PartialFixedSampler against live Optuna. Phase 1 runs a base
///     RandomSampler; then the best <c>y</c> is fixed and the study re-optimizes, with the SAME
///     base sampler's RNG continuing for the unfixed <c>x</c>. The complete two-phase study state
///     must agree with Python's, which proves both the delegation and the RNG continuation.
/// </summary>
[TestClass]
public class PartialFixedSamplerParityTests : OptunaInteropTestBase
{
    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(0)]
    [DataRow(5)]
    [DataRow(42)]
    public void FixedYReoptimize_FullState_IsBitIdentical(int seed)
    {
        PythonSession.EnsureOrInconclusive();

        // --- OptunaSharp side. -----------------------------------------------------------
        var csStudy = Optuna.CreateStudy(
            storage: new InMemoryStorage(), sampler: new RandomSampler(seed), studyName: "pf");
        csStudy.Optimize(Objective, nTrials: 6);
        var fixedParams = new Dictionary<string, object?> { ["y"] = csStudy.BestParams["y"] };
        csStudy.Sampler = new PartialFixedSampler(fixedParams, csStudy.Sampler);
        csStudy.Optimize(Objective, nTrials: 6);
        var csState = StudySnapshot.Capture(csStudy);

        // --- live Optuna side (same two-phase workload). ---------------------------------
        PyExec(
            "def _obj(t):\n" +
            "    x = t.suggest_float('x', -1, 1)\n" +
            "    y = t.suggest_int('y', -1, 1)\n" +
            "    return x * x + y\n" +
            $"study = optuna.create_study(study_name='pf', sampler=optuna.samplers.RandomSampler(seed={seed}))\n" +
            "study.optimize(_obj, n_trials=6)\n" +
            "study.sampler = optuna.samplers.PartialFixedSampler({'y': study.best_params['y']}, study.sampler)\n" +
            "study.optimize(_obj, n_trials=6)\n");
        var pyState = PyStudyState("study");

        StudySnapshot.AssertParity(pyState, csState, DatetimeComparison.NullnessOnly);
    }

    private static double Objective(Trial t)
    {
        double x = t.SuggestFloat("x", -1, 1);
        long y = t.SuggestInt("y", -1, 1);
        return x * x + y;
    }
}
