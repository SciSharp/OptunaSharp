using System;
using System.Linq;
using System.Text;
using AwesomeAssertions;
using OptunaSharp.Collections;
using OptunaSharp.Internal;
using OptunaSharp.Pruners;

namespace OptunaSharp.Tests.QA;

/// <summary>
///     Open parity bugs in the pruners, found by the 2026-11-02 parity QA run
///     (<c>docs/qa/PARITY_QA_2026-11-02.md</c>). Expected values come from Optuna 5.0.0 with scipy 1.16.3 on
///     CPython 3.12; each test fails today and passes once the divergence is fixed, at which point its
///     <c>[OpenBugs]</c> attribute should be removed.
/// </summary>
[TestClass]
public class PrunerParityBugTests
{
    /// <summary>The 60 paired differences of the decision-flip scenario (scipy auto p = 0.0988, exact p = 0.1002).</summary>
    private static readonly double[] FlipDiffs =
    {
        0.147, -0.3656, 0.1353, 0.2789, 1.2407, -0.7052, -2.0175, 3.0403, 1.0218, 0.2965, 2.7017, -0.8999, -0.5469,
        -0.1918, 0.7459, -0.214, 1.0411, -0.1055, 1.564, 0.72, -1.4292, 0.2151, -0.9085, 0.1059, 0.2439, 2.5138,
        0.0053, -1.3908, 0.3502, -0.9404, 2.0932, 0.9575, 1.833, -0.1677, -0.8231, 0.5315, -1.1615, -0.7464,
        -0.5375, 0.0854, -1.3143, -1.8631, 1.6847, -0.3201, 0.0453, 0.8368, 0.6786, -0.4328, -0.2206, 1.8744,
        1.5147, 0.9976, 1.9319, -1.6778, 0.3765, -0.0396, -1.7129, 0.5913, 0.905, 0.5134,
    };

    /// <summary>
    ///     A deterministic sequence of <paramref name="n" /> distinct, nonzero differences, identical to the
    ///     Python expression the expected p-values were computed from.
    /// </summary>
    /// <param name="n">Number of differences; every absolute value is distinct, so scipy sees no ties or zeros.</param>
    /// <returns><c>((i * 7919) % 1000 - 450) / 1000.0 + 1e-7 * i</c> for <c>i</c> in <c>0..n-1</c>.</returns>
    private static double[] DistinctDiffs(int n)
        => Enumerable.Range(0, n).Select(i => ((i * 7919) % 1000 - 450) / 1000.0 + 1e-7 * i).ToArray();

    /// <summary>
    ///     QA-14: with more than 50 paired steps scipy's <c>method="auto"</c> switches to the normal
    ///     approximation before it even looks for ties; the port keeps using the exact distribution, so its
    ///     p-values differ from the third significant digit on.
    /// </summary>
    /// <remarks>
    ///     scipy 1.16.3, <c>_wilcoxon_iv</c>: <c>if method == "auto" and d.shape[-1] &gt; 50: method =
    ///     "asymptotic"</c>. <c>SignedRankTest.PValueZsplit</c> only checks "no ties and no zeros" and goes exact
    ///     at any length. Measured: n = 60 distinct differences give scipy <c>0.0012107564896459532</c>, the
    ///     port <c>0.0010342998730745664</c>; for this test's sequence scipy gives <c>0.14123347510360462</c>.
    ///     Continuous losses rarely tie, so every Wilcoxon-pruned run longer than 50 steps is affected.
    /// </remarks>
    [TestMethod]
    [OpenBugs]
    public void Wilcoxon_MoreThan50Steps_UsesScipyNormalApproximation()
    {
        double p = SignedRankTest.PValueZsplit(DistinctDiffs(60), SignedRankTest.Alternative.Greater);

        p.Should().BeApproximately(0.14123347510360462, 1e-12, "scipy.stats.wilcoxon(d, alternative='greater', zero_method='zsplit').pvalue");
    }

    /// <summary>
    ///     QA-14 (overflow): from about 1,024 steps the port's exact distribution overflows <c>2^n</c> in
    ///     doubles and returns NaN, so the pruner never prunes; scipy returns a normal-approximation p-value.
    /// </summary>
    /// <remarks>
    ///     For 1,100 distinct differences scipy gives <c>4.571585047815242e-08</c> (a certain prune) and the port
    ///     NaN (<c>p &lt; threshold</c> is false). The exact path is also O(n³): about 0.7 s per
    ///     <c>should_prune</c> at this length, paid at every step of every trial.
    /// </remarks>
    [TestMethod]
    [OpenBugs]
    public void Wilcoxon_1100Steps_PValueIsFiniteLikeScipy()
    {
        double p = SignedRankTest.PValueZsplit(DistinctDiffs(1100), SignedRankTest.Alternative.Greater);

        double.IsNaN(p).Should().BeFalse();
        p.Should().BeApproximately(4.571585047815242e-08, 1e-18);
    }

    /// <summary>
    ///     QA-14 (decision level): the same study must prune at the same steps as Optuna's
    ///     <c>WilcoxonPruner</c>; past 50 steps the port's exact p-values flip decisions near the threshold.
    /// </summary>
    /// <remarks>
    ///     A best trial reports 1.0 at steps 0–59; the current trial reports <c>1.0 + d[i]</c> at step
    ///     <c>i</c> and asks <c>should_prune()</c> after each report. Optuna 5.0.0 prunes at the steps marked
    ///     <c>P</c>; at step 59 scipy's p is 0.0988 (prune) where the exact p the port computes is 0.1002.
    /// </remarks>
    [TestMethod]
    [OpenBugs]
    public void WilcoxonPruner_60Steps_PrunesAtPythonsSteps()
    {
        var study = Optuna.CreateStudy(pruner: new WilcoxonPruner(pThreshold: 0.1));
        var bestValues = new OrderedDict<long, double>();
        for (int i = 0; i < FlipDiffs.Length; i++)
            bestValues[i] = 1.0;
        study.AddTrial(FrozenTrial.CreateTrial(value: 1.0, intermediateValues: bestValues));

        var trial = study.Ask();
        var decisions = new StringBuilder();
        for (int i = 0; i < FlipDiffs.Length; i++)
        {
            trial.Report(1.0 + FlipDiffs[i], i);
            decisions.Append(trial.ShouldPrune() ? 'P' : '.');
        }

        decisions.ToString().Should().Be("...................P...........PPP.P................P......P");
    }
}
