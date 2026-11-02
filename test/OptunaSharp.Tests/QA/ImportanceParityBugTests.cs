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
}
