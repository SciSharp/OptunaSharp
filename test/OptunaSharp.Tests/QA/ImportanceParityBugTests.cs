using System;
using AwesomeAssertions;
using OptunaSharp.Importance;
using OptunaSharp.Importance.Shap;
using OptunaSharp.Samplers;

namespace OptunaSharp.Tests.QA;

/// <summary>
///     Open parity bugs in the importance evaluators, found by the 2026-11-02 parity QA run
///     (<c>docs/qa/PARITY_QA_2026-11-02.md</c>). Expected values are Optuna 5.0.0 + optuna-integration 5.0.0 +
///     shap 0.52.0 + scikit-learn 1.8.0 results for the same study; each test fails today and passes once fixed,
///     at which point its <c>[OpenBugs]</c> attribute should be removed.
/// </summary>
[TestClass]
public class ImportanceParityBugTests
{
    /// <summary>
    ///     The objective of the QA-24 study: a stepped float, a log int and a stepped int, with every ninth trial
    ///     pruned and every eleventh failed, identical to the Python objective the expected values come from.
    /// </summary>
    /// <param name="t">The running trial; its three parameters are suggested in a fixed order.</param>
    /// <returns><c>(x - 0.3)^2 + 0.2 * ln(j) + 0.01 * s</c>.</returns>
    /// <exception cref="TrialPruned">For trial numbers congruent to 4 modulo 9, to put pruned trials in the study.</exception>
    /// <exception cref="InvalidOperationException">For trial numbers congruent to 7 modulo 11; the test catches it so those trials are stored as failed.</exception>
    private static double SteppedObjective(Trial t)
    {
        double x = t.SuggestFloat("x", 0, 1, step: 0.05);
        long j = t.SuggestInt("j", 1, 1000, log: true);
        long s = t.SuggestInt("s", 0, 30, step: 3);
        if (t.Number % 9 == 4)
            throw new TrialPruned();
        if (t.Number % 11 == 7)
            throw new InvalidOperationException("fail");
        return Math.Pow(x - 0.3, 2) + Math.Log(j) * 0.2 + s * 0.01;
    }

    /// <summary>
    ///     QA-24: SHAP importances must evaluate tree paths on <c>float32(X)</c>, as shap's <c>TreeExplainer</c>
    ///     does for scikit-learn forests; the port traverses with the original float64 values, so a parameter
    ///     whose float32 rounding lands on a split threshold takes the other branch.
    /// </summary>
    /// <remarks>
    ///     shap sets <c>input_dtype = np.float32</c> for scikit-learn trees and casts <c>X</c> before the C
    ///     extension runs; the port's comment says the opposite ("SHAP evaluates paths against the ORIGINAL float64
    ///     X"). Stepped float parameters make it common: the sampler returns <c>0.35000000000000003</c>, whose
    ///     float32 is <c>0.3499999940395355</c>, and a tree can split exactly there. In this study (Random seed 42,
    ///     80 trials) one of the 64 trees is visited differently for six rows and the importances differ by about
    ///     5e-4 relative: Python gives <c>j=0.8293157756066445, s=0.08730275399872063, x=0.08338147039463491</c>,
    ///     the port <c>j=0.8285727626755199, s=0.0873554365265051, x=0.08407180079797498</c>. The forest itself is
    ///     node-for-node identical (fANOVA on the same study is bit-exact), so the fix is confined to the
    ///     traversal.
    /// </remarks>
    [TestMethod]
    [OpenBugs]
    public void Shapley_SteppedFloatOnSplitThreshold_TraversesFloat32LikeShap()
    {
        var study = Optuna.CreateStudy(sampler: new RandomSampler(seed: 42));
        study.Optimize(SteppedObjective, nTrials: 80, @catch: new[] { typeof(InvalidOperationException) });

        var importances = ImportanceFunctions.GetParamImportances(study, new ShapleyImportanceEvaluator(seed: 0));

        importances.Keys.Should().Equal("j", "s", "x");
        BitConverter.DoubleToInt64Bits(importances["j"]).Should().Be(unchecked((long)0x3fea89c13cc9327cUL));
        BitConverter.DoubleToInt64Bits(importances["s"]).Should().Be(unchecked((long)0x3fb6597929467605UL));
        BitConverter.DoubleToInt64Bits(importances["x"]).Should().Be(unchecked((long)0x3fb5587cf06ff61dUL));
    }

    /// <summary>
    ///     QA-28: PED-ANOVA, the default evaluator of <c>get_param_importances</c>, must handle a numerical parameter
    ///     whose grid has 2^31 - 1 or more steps; the port's grid-size arithmetic overflows <see cref="int" /> and
    ///     the evaluation throws <see cref="OverflowException" />.
    /// </summary>
    /// <remarks>
    ///     <c>_count_numerical_param_in_grid</c> computes <c>min(round((high - low) / step) + 1, n_steps)</c> in
    ///     Python ints, which is just <c>n_steps</c> (50) for a wide range. The port writes
    ///     <c>Math.Min((int)Math.Round(...) + 1, nSteps)</c>: the cast saturates at <see cref="int.MaxValue" />
    ///     (net10.0; <see cref="int.MinValue" /> on net8.0), the <c>+ 1</c> wraps negative, and allocating the
    ///     negative-length count array throws "Arithmetic operation resulted in an overflow." The common
    ///     <c>suggest_int("random_state", 0, 2**31 - 1)</c> and <c>suggest_int("seed", 0, 2**32 - 1)</c> trigger it,
    ///     as do <c>suggest_int("n", 0, 10**10)</c> and <c>suggest_float("x", 0, 1, step=1e-10)</c>; so does
    ///     <c>plot_param_importances</c>, which uses the same evaluator. With high <c>2**31 - 2</c> the port
    ///     agrees with Python within the documented PED-ANOVA tolerance (about 10 ULP), so the comparison here
    ///     uses that tolerance instead of bits. Python (Random seed 0, 40 trials): <c>y=0.9953777359648283,
    ///     seed=0.004622264035171709</c>.
    /// </remarks>
    [TestMethod]
    [OpenBugs]
    public void PedAnova_IntRangeOf2Pow31Steps_DoesNotOverflow()
    {
        var study = Optuna.CreateStudy(sampler: new RandomSampler(seed: 0));
        study.Optimize(t => t.SuggestInt("seed", 0, int.MaxValue) * 1e-12 + t.SuggestFloat("y", 0, 1), nTrials: 40);

        var importances = ImportanceFunctions.GetParamImportances(study);

        importances.Keys.Should().Equal("y", "seed");
        importances["y"].Should().BeApproximately(0.9953777359648283, 1e-12);
        importances["seed"].Should().BeApproximately(0.004622264035171709, 1e-12);
    }
}
