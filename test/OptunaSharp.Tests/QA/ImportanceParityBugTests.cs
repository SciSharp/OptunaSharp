using System;
using System.Linq;
using AwesomeAssertions;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
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

    /// <summary>
    ///     QA-40: the MDI evaluator must return NaN importances when the forest's trees split but their impurity
    ///     decreases sum to zero, as scikit-learn's <c>RandomForestRegressor.feature_importances_</c> does
    ///     (<c>all_importances / np.sum(all_importances)</c> with no guard); the port returns zeros, which
    ///     <c>get_param_importances</c> then normalizes to a uniform <c>1 / n</c>.
    /// </summary>
    /// <remarks>
    ///     With objective values near <c>-1e12</c> and a spread of about 29, the MSE criterion's
    ///     <c>sq_sum / n - mean^2</c> cancels: sklearn stores root impurity <c>2^27</c> and leaf impurities <c>0</c> and
    ///     <c>2^28</c>, so each splitting tree's decrease <c>2 * 2^27 - 0 - 2^28</c> is exactly 0. The port's CART
    ///     reproduces those trees bit for bit; only <c>Mdi.FeatureImportances</c> differs, through its
    ///     <c>if (sum != 0.0)</c> guard. Optuna 5.0.0 + scikit-learn 1.8.0 return
    ///     <c>{'b': nan, 'a': nan}</c> (normalized or not); the port returns <c>{b: 0.5, a: 0.5}</c>
    ///     (<c>0.0</c> each unnormalized). Found by the D75 random-study fuzz (1 of 600 studies).
    /// </remarks>
    [TestMethod]
    [OpenBugs]
    public void Mdi_ZeroTotalImpurityDecrease_ReturnsNaNLikeSklearn()
    {
        var study = Optuna.CreateStudy();
        var distributions = new OrderedDict<string, BaseDistribution>
        {
            ["a"] = new IntDistribution(1, 7),
            ["b"] = new IntDistribution(3, 5),
        };
        study.AddTrial(FrozenTrial.CreateTrial(value: -999995999937.6326,
            @params: new OrderedDict<string, object?> { ["a"] = 4L, ["b"] = 5L }, distributions: distributions));
        study.AddTrial(FrozenTrial.CreateTrial(value: -999995999966.4989,
            @params: new OrderedDict<string, object?> { ["a"] = 2L, ["b"] = 3L }, distributions: distributions));

        var importances = ImportanceFunctions.GetParamImportances(study, new MeanDecreaseImpurityImportanceEvaluator(seed: 0));

        importances.Keys.Should().BeEquivalentTo("a", "b");
        importances.Values.Should().OnlyContain(v => double.IsNaN(v));
    }

    /// <summary>
    ///     QA-52: <c>get_param_importances</c> must normalize PED-ANOVA's importances with CPython's builtin
    ///     <c>sum()</c>, which adds plain Python floats with Neumaier compensation since 3.12; the port adds them one by
    ///     one, so the normalized importances differ by an ULP in about a third of studies even where the raw ones
    ///     match bit for bit.
    /// </summary>
    /// <remarks>
    ///     <c>importance/__init__.py</c> computes <c>s = sum(res.values())</c>. PED-ANOVA returns <c>float</c>s, so the
    ///     sum takes CPython's compensated float loop; fANOVA and MDI return <c>np.float64</c> (a float subclass), which
    ///     takes the generic, sequential loop — the port's sequential sum is right for those two only. On 60 random
    ///     categorical-only studies (no <c>erf</c>, so PED-ANOVA's raw values are bit-identical) the normalized values
    ///     differ in exactly the 22 where Neumaier and sequential sums disagree. This study is the first of them:
    ///     Optuna 5.0.0 normalizes to the bits below, the port to <c>…34f9</c> and <c>…6eaa</c>. The same builtin
    ///     <c>sum()</c> is ported sequentially elsewhere (constraint violations in NSGA-II and TPE, Wilcoxon averages);
    ///     found by the D100 probe.
    /// </remarks>
    [TestMethod]
    [OpenBugs]
    public void PedAnova_NormalizedImportances_SumLikeCPythonBuiltinSum()
    {
        var choices = new[] { "a", "b", "c", "d" };
        var study = Optuna.CreateStudy(sampler: new RandomSampler(seed: 0));
        study.Optimize(t =>
        {
            double s = 0.0;
            for (int i = 0; i < 3; i++)
                s += (1 + i) * Array.IndexOf(choices, t.SuggestCategorical($"p{i}", choices));
            return s;
        }, nTrials: 40);

        var importances = ImportanceFunctions.GetParamImportances(study, new OptunaSharp.Importance.PedAnova.PedAnovaImportanceEvaluator());

        importances.Select(kv => $"{kv.Key}={BitConverter.DoubleToInt64Bits(kv.Value):x16}")
            .Should().Equal("p1=3fe017a96f8a34fa", "p2=3fd5484746ab5eb8", "p0=3fc510cbb4806eab");
    }
}
