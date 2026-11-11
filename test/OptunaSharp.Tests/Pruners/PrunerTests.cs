using System;
using System.Collections.Generic;
using System.Linq;
using AwesomeAssertions;
using NumSharp;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Internal;
using OptunaSharp.Pruners;
using OptunaSharp.Samplers;

namespace OptunaSharp.Tests.Pruners;

/// <summary>
///     Pruner behavior over hand-constructed trial histories, plus the scipy-pinned Wilcoxon
///     p-values and binascii-pinned CRC-32 (Hyperband bracket routing).
/// </summary>
[TestClass]
public class PrunerTests
{
    private static Study StudyWithHistories(
        BasePruner pruner, StudyDirection direction, params double[][] histories)
    {
        var study = Optuna.CreateStudy(
            sampler: new RandomSampler(1), pruner: pruner, direction: direction);
        foreach (var history in histories)
        {
            var intermediate = new OrderedDict<long, double>();
            for (int i = 0; i < history.Length; i++)
                intermediate[i] = history[i];
            study.AddTrial(FrozenTrial.CreateTrial(
                value: history[^1],
                intermediateValues: intermediate));
        }
        return study;
    }

    private static FrozenTrial RunningTrial(Study study, params (int Step, double Value)[] reports)
    {
        var trial = study.Ask();
        trial.SuggestFloat("x", 0, 1);
        foreach (var (step, value) in reports)
            trial.Report(value, step);
        return TellHelpersProbe(study, trial);
    }

    private static FrozenTrial TellHelpersProbe(Study study, Trial trial)
        => study.Storage.GetTrial(trial.TrialId);

    // ------------------------------------------------------------------ Nop / Threshold

    [TestMethod]
    public void NopPruner_Never_Prunes()
    {
        var study = StudyWithHistories(new NopPruner(), StudyDirection.Minimize, new[] { 1.0 });
        var trial = FrozenTrial.CreateTrial(
            state: TrialState.Running,
            intermediateValues: new OrderedDict<long, double> { [0] = double.NaN });
        trial.SetDatetimeStart(DateTime.Now);
        new NopPruner().Prune(study, trial).Should().BeFalse();
    }

    [TestMethod]
    public void ThresholdPruner_Prunes_On_Bounds_And_NaN()
    {
        var study = Optuna.CreateStudy(sampler: new RandomSampler(1), pruner: new NopPruner());
        FrozenTrial Reporting(params (int, double)[] values)
        {
            var iv = new OrderedDict<long, double>();
            foreach (var (s, v) in values)
                iv[s] = v;
            var t = FrozenTrial.CreateTrial(state: TrialState.Running, intermediateValues: iv);
            return t;
        }

        var upper = new ThresholdPruner(upper: 1.0);
        upper.Prune(study, Reporting((0, 0.5))).Should().BeFalse();
        upper.Prune(study, Reporting((0, 0.5), (1, 1.2))).Should().BeTrue();

        var lower = new ThresholdPruner(lower: 0.0);
        lower.Prune(study, Reporting((0, 0.1))).Should().BeFalse();
        lower.Prune(study, Reporting((0, 0.1), (1, -1.0))).Should().BeTrue();

        new ThresholdPruner(upper: 5.0).Prune(study, Reporting((0, double.NaN))).Should().BeTrue();

        var neither = () => new ThresholdPruner();
        neither.Should().Throw<TypeError>().WithMessage("Either lower or upper must be specified.");
        var inverted = () => new ThresholdPruner(lower: 2.0, upper: 1.0);
        inverted.Should().Throw<ValueError>().WithMessage("lower should be smaller than upper.");
    }

    // ------------------------------------------------------------------ Percentile / Median

    [TestMethod]
    public void MedianPruner_Prunes_Below_Median()
    {
        // Minimize; completed histories at step 0: values 1, 2, 3 -> median 2.
        var study = StudyWithHistories(
            new MedianPruner(nStartupTrials: 1, nWarmupSteps: 0),
            StudyDirection.Minimize,
            new[] { 1.0 }, new[] { 2.0 }, new[] { 3.0 });

        var pruner = (MedianPruner)study.Pruner;
        FrozenTrial Probe(double v) => FrozenTrial.CreateTrial(
            state: TrialState.Running,
            intermediateValues: new OrderedDict<long, double> { [0] = v });

        pruner.Prune(study, Probe(1.5)).Should().BeFalse("1.5 <= median 2");
        pruner.Prune(study, Probe(2.0)).Should().BeFalse("equal to the median survives");
        pruner.Prune(study, Probe(2.5)).Should().BeTrue("2.5 > median 2");
        pruner.Prune(study, Probe(double.NaN)).Should().BeTrue("all-NaN intermediates prune");
    }

    [TestMethod]
    public void PercentilePruner_Validates_Arguments()
    {
        var pct = () => new PercentilePruner(101);
        pct.Should().Throw<ValueError>()
            .WithMessage("Percentile must be between 0 and 100 inclusive, but got percentile=101.0.");
        var startup = () => new PercentilePruner(50, nStartupTrials: -1);
        startup.Should().Throw<ValueError>()
            .WithMessage("Number of startup trials cannot be negative, but got n_startup_trials=-1.");
    }

    [TestMethod]
    public void NanPercentile_Matches_Numpy_Linear_Interpolation()
    {
        PercentilePruner.NanPercentile(new[] { 1.0, 2.0, 3.0, 4.0 }, 50).Should().Be(2.5);
        PercentilePruner.NanPercentile(new[] { 1.0, double.NaN, 3.0 }, 50).Should().Be(2.0);
        PercentilePruner.NanPercentile(new[] { 1.0, 2.0, 3.0, 4.0 }, 25).Should().Be(1.75);
    }

    /// <summary>
    ///     Pre-release review item H4: <see cref="PercentilePruner.NanPercentile" /> against values printed by
    ///     <c>numpy.nanpercentile</c> (numpy 2.4.2, <c>method="linear"</c>), compared bit for bit. The rows cover
    ///     both <c>_lerp</c> branches (<c>t &lt; 0.5</c> and the <c>b - (b - a) * (1 - t)</c> form from 0.5 on,
    ///     where the textbook formula is one ULP off), an index past the last element, NaN inputs, and the
    ///     infinite neighbours that make numpy return NaN even when the index lands exactly on a finite element.
    /// </summary>
    /// <remarks>
    ///     Row notes, in order: t = 0.5, where the textbook formula gives 0.019999999999999997; a NaN dropped first;
    ///     t = 0.25 (the <c>a + (b - a) * t</c> branch); three t ≥ 0.5 rows where the textbook formula gives
    ///     0.014999999999999902, 0.4760000000000001 and 0.2869999999999999; two ordinary rows; one value, where both
    ///     neighbours clamp to it; an index past the end, clamped to the last element; then the NaN rows: an index
    ///     landing on 2.0 whose next neighbour is inf, <c>inf * 0</c> at t = 0, <c>inf - inf</c> for a clamped
    ///     infinite last element, and single or paired infinities; then the two branches that keep an infinite
    ///     neighbour infinite instead (<c>1 - inf * 0.25</c> and <c>1 + inf * 0.25</c>); last, two rows whose result is a
    ///     zero whose sign NumPy's partition decides among tied <c>-0.0</c>/<c>0.0</c> neighbours (NumSharp's
    ///     <c>np.nanpercentile</c> partitions as NumPy does for these), and an all-NaN input. One tie pattern NumSharp 0.80.0
    ///     still resolves differently is pinned by <see cref="NanPercentile_TiedSignedZeros_KeepNumpysZeroSign" />.
    /// </remarks>
    /// <param name="values">The completed trials' reports at one step, NaNs included.</param>
    /// <param name="percentile">The percentile asked for, in [0, 100].</param>
    /// <param name="expected">numpy's result for the same arguments.</param>
    [TestMethod]
    [DataRow(new[] { 0.01, 0.03 }, 50.0, 0.02)]
    [DataRow(new[] { double.NaN, 0.01, 0.03 }, 50.0, 0.02)]
    [DataRow(new[] { 0.01, 0.03 }, 25.0, 0.015)]
    [DataRow(new[] { -0.85, -0.536, 0.566 }, 75.0, 0.015000000000000013)]
    [DataRow(new[] { 0.82, -0.9 }, 80.0, 0.47600000000000003)]
    [DataRow(new[] { -0.6, 0.3, 1.0, -0.35 }, 66.0, 0.287)]
    [DataRow(new[] { 0.1, 0.2, 0.4 }, 75.0, 0.30000000000000004)]
    [DataRow(new[] { 0.3, 0.7, 0.1 }, 60.0, 0.37999999999999995)]
    [DataRow(new[] { 5.0 }, 50.0, 5.0)]
    [DataRow(new[] { double.NegativeInfinity, 1.0 }, 100.0, 1.0)]
    [DataRow(new[] { 1.0, 2.0, double.PositiveInfinity }, 50.0, double.NaN)]
    [DataRow(new[] { 1.0, double.PositiveInfinity }, 0.0, double.NaN)]
    [DataRow(new[] { 1.0, double.PositiveInfinity }, 100.0, double.NaN)]
    [DataRow(new[] { double.PositiveInfinity }, 50.0, double.NaN)]
    [DataRow(new[] { double.NegativeInfinity }, 50.0, double.NaN)]
    [DataRow(new[] { double.NegativeInfinity, 1.0 }, 0.0, double.NaN)]
    [DataRow(new[] { double.NegativeInfinity, double.PositiveInfinity }, 50.0, double.NaN)]
    [DataRow(new[] { double.NegativeInfinity, 1.0 }, 75.0, double.NegativeInfinity)]
    [DataRow(new[] { 1.0, double.PositiveInfinity }, 25.0, double.PositiveInfinity)]
    [DataRow(new[] { -1.0, -0.0, 0.5, 0.0, -1.0, -1.0 }, 75.0, 0.0)]
    [DataRow(new[] { 0.0, -0.0, -0.0, -0.0, 1.0, 0.0 }, 50.0, -0.0)]
    [DataRow(new[] { double.NaN, double.NaN }, 50.0, double.NaN)]
    public void NanPercentile_IsBitIdenticalToNumpy(double[] values, double percentile, double expected)
    {
        double actual = PercentilePruner.NanPercentile(values, percentile);

        if (double.IsNaN(expected))
            double.IsNaN(actual).Should().BeTrue($"numpy returns nan for {string.Join(", ", values)} at q={percentile}");
        else
            BitConverter.DoubleToInt64Bits(actual).Should().Be(BitConverter.DoubleToInt64Bits(expected),
                $"numpy returns {expected:R} for {string.Join(", ", values)} at q={percentile}, not {actual:R}");
    }

    /// <summary>
    ///     QA-85: the median of <c>[0.0, -0.0, -0.0, -1.0]</c> is <c>-0.0</c> in NumPy 2.4.2: its partition leaves the two
    ///     <c>-0.0</c> as the interpolated neighbours. NumSharp 0.80.0's <c>np.nanpercentile</c> and <c>np.percentile</c>
    ///     return <c>+0.0</c> (they interpolate between <c>0.0</c> and <c>-0.0</c>, and <c>-0.0 - (-0.0 * 0.5)</c> is
    ///     <c>+0.0</c>), as did the sort-based port before 2026-11-11. An upstream NumSharp partition tie; no comparison can
    ///     observe the sign, so prune decisions are unaffected. Remove <c>[OpenBugs]</c> once NumSharp matches.
    /// </summary>
    [TestMethod]
    [OpenBugs]
    public void NanPercentile_TiedSignedZeros_KeepNumpysZeroSign()
    {
        double actual = PercentilePruner.NanPercentile(new[] { 0.0, -0.0, -0.0, -1.0 }, 50.0);

        BitConverter.DoubleToInt64Bits(actual).Should().Be(BitConverter.DoubleToInt64Bits(-0.0),
            $"numpy.nanpercentile([0.0, -0.0, -0.0, -1.0], 50) is -0.0, not {actual:R}");
    }

    /// <summary>
    ///     Pre-release review item H4 at the pruner level: a trial reporting exactly the interpolated cutoff (or
    ///     a value between numpy's cutoff and the textbook one) gets Optuna's decision, as printed by live Optuna
    ///     5.0.0 with numpy 2.4.2. Before the fix every row flipped: the review's repro (0.01 and 0.03 completed, a
    ///     report of 0.02) was pruned here and kept by Optuna, and an infinite report among the completed trials
    ///     gave a finite median that pruned a trial Optuna keeps (numpy's median of 1, 2 and inf is NaN).
    /// </summary>
    /// <param name="percentile">The pruner's percentile; 50 builds a <see cref="MedianPruner" />.</param>
    /// <param name="completedReports">The step-0 reports (and final values) of the completed trials, in order.</param>
    /// <param name="currentReport">The step-0 report of the trial being judged.</param>
    /// <param name="expected">Optuna's <c>should_prune()</c> for the same history.</param>
    [TestMethod]
    [DataRow(50.0, new[] { 0.01, 0.03 }, 0.02, false)]
    [DataRow(50.0, new[] { 1.0, 2.0, double.PositiveInfinity }, 3.0, false)]
    [DataRow(75.0, new[] { -0.85, -0.536, 0.566 }, 0.015000000000000013, false)]
    [DataRow(80.0, new[] { 0.82, -0.9 }, 0.4760000000000001, true)]
    public void PercentilePruner_AtTheInterpolatedCutoff_DecidesLikeOptuna(
        double percentile, double[] completedReports, double currentReport, bool expected)
    {
        int nStartupTrials = completedReports.Length;
        BasePruner pruner = percentile == 50.0
            ? new MedianPruner(nStartupTrials: nStartupTrials)
            : new PercentilePruner(percentile, nStartupTrials: nStartupTrials);
        var study = Optuna.CreateStudy(sampler: new RandomSampler(1), pruner: pruner);
        foreach (double report in completedReports)
        {
            var done = study.Ask();
            done.Report(report, 0);
            study.Tell(done, report);
        }

        var trial = study.Ask();
        trial.Report(currentReport, 0);

        trial.ShouldPrune().Should().Be(expected,
            $"Optuna's should_prune() for a report of {currentReport:R} after {string.Join(", ", completedReports)}");
    }

    // ------------------------------------------------------------------ Patient

    [TestMethod]
    public void PatientPruner_Waits_For_Patience_Window()
    {
        var study = Optuna.CreateStudy(
            sampler: new RandomSampler(1), pruner: new NopPruner(), direction: StudyDirection.Minimize);

        FrozenTrial Probe(params double[] values)
        {
            var iv = new OrderedDict<long, double>();
            for (int i = 0; i < values.Length; i++)
                iv[i] = values[i];
            return FrozenTrial.CreateTrial(state: TrialState.Running, intermediateValues: iv);
        }

        var pruner = new PatientPruner(null, patience: 1);
        pruner.Prune(study, Probe(10, 9)).Should().BeFalse("not enough steps yet");
        pruner.Prune(study, Probe(10, 9, 8)).Should().BeFalse("still improving");
        pruner.Prune(study, Probe(10, 8, 9, 8.5)).Should().BeTrue("no improvement within patience=1");
    }

    // ------------------------------------------------------------------ SuccessiveHalving

    [TestMethod]
    public void SuccessiveHalving_Static_Pieces()
    {
        SuccessiveHalvingPruner.CompletedRungKey(2).Should().Be("completed_rung_2");

        var trials = new List<FrozenTrial>
        {
            FrozenTrial.CreateTrial(value: 1.0, intermediateValues: new OrderedDict<long, double> { [250] = 1.0 }),
        };
        SuccessiveHalvingPruner.EstimateMinResource(trials).Should().Be(2, "250 // 100 = 2");
        SuccessiveHalvingPruner.EstimateMinResource(new List<FrozenTrial>()).Should().BeNull();

        // The first eta-1 trials: promoted only when the value equals the best.
        SuccessiveHalvingPruner.IsTrialPromotableToNextRung(
            1.0, new List<double> { 1.0, 2.0 }, 4, StudyDirection.Minimize).Should().BeTrue();
        SuccessiveHalvingPruner.IsTrialPromotableToNextRung(
            2.0, new List<double> { 1.0, 2.0 }, 4, StudyDirection.Minimize).Should().BeFalse();
        // With 8 competitors and eta=4: top-2 promote.
        SuccessiveHalvingPruner.IsTrialPromotableToNextRung(
            2.0, Enumerable.Range(1, 8).Select(v => (double)v).ToList(), 4, StudyDirection.Minimize)
            .Should().BeTrue();
        SuccessiveHalvingPruner.IsTrialPromotableToNextRung(
            3.0, Enumerable.Range(1, 8).Select(v => (double)v).ToList(), 4, StudyDirection.Minimize)
            .Should().BeFalse();
    }

    [TestMethod]
    public void SuccessiveHalving_EndToEnd_Prunes_Losers()
    {
        var study = Optuna.CreateStudy(
            sampler: new RandomSampler(7),
            pruner: new SuccessiveHalvingPruner(minResource: 1, reductionFactor: 2));
        int pruned = 0;
        study.Optimize(t =>
        {
            double x = t.SuggestFloat("x", 0, 1);
            for (int step = 0; step < 16; step++)
            {
                t.Report(x + step * 0.001, step);
                if (t.ShouldPrune())
                {
                    pruned++;
                    throw new TrialPruned();
                }
            }
            return x;
        }, nTrials: 12);

        study.Trials.Count(t => t.State == TrialState.Pruned).Should().Be(pruned);
        pruned.Should().BeGreaterThan(0, "ASHA must cut some of 12 random trials");
        study.Trials.Count(t => t.State == TrialState.Complete).Should().BeGreaterThan(0);
        // Rung attrs recorded on promoted trials.
        study.Trials.SelectMany(t => t.SystemAttrs.Keys)
            .Should().Contain(k => k.StartsWith("completed_rung_"));
    }

    // ------------------------------------------------------------------ Hyperband

    [TestMethod]
    public void Crc32_Matches_Python_Binascii()
    {
        // Pinned: binascii.crc32(b"...").
        Crc32.Compute(System.Text.Encoding.UTF8.GetBytes("study_0")).Should().Be(368277276u);
        Crc32.Compute(System.Text.Encoding.UTF8.GetBytes("my-study_15")).Should().Be(1706675147u);
        Crc32.Compute(System.Text.Encoding.UTF8.GetBytes("no-name-abc_7")).Should().Be(3149476025u);
    }

    [TestMethod]
    public void Hyperband_EndToEnd_Runs_And_Prunes()
    {
        var study = Optuna.CreateStudy(
            sampler: new RandomSampler(9),
            studyName: "hyperband-fixed-name",
            pruner: new HyperbandPruner(minResource: 1, maxResource: 27, reductionFactor: 3));
        study.Optimize(t =>
        {
            double x = t.SuggestFloat("x", 0, 1);
            for (int step = 0; step < 27; step++)
            {
                t.Report(x + step * 1e-4, step);
                if (t.ShouldPrune())
                    throw new TrialPruned();
            }
            return x;
        }, nTrials: 20);

        study.Trials.Should().HaveCount(20);
        study.Trials.Count(t => t.State == TrialState.Complete).Should().BeGreaterThan(0);
        study.Trials.Count(t => t.State == TrialState.Pruned).Should().BeGreaterThan(0);
    }

    [TestMethod]
    public void Hyperband_Validates_Bootstrap_With_Auto_MaxResource()
    {
        var act = () => new HyperbandPruner(bootstrapCount: 2);
        act.Should().Throw<ValueError>()
            .WithMessage("bootstrap_count > 0 and max_resource == 'auto' are mutually incompatible, bootstrap_count is 2");
    }

    // ------------------------------------------------------------------ Wilcoxon

    [TestMethod]
    public void SignedRankTest_Matches_Scipy_Pinned_PValues()
    {
        // Every (less, greater) pair below is repr(ss.wilcoxon(d, alternative=..., zero_method="zsplit").pvalue)
        // from scipy 1.16.3 — covering the exact, permutation and asymptotic method branches.
        var cases = new (double[] D, double Less, double Greater)[]
        {
            (new[] { 1.5, -2.25, 3.0, 4.5, -0.75, 2.0, 1.0, -3.5 }, 0.76953125, 0.2734375),
            (new[] { 1.0, 1.0, -1.0, 2.0, 3.0, -2.0 }, 0.828125, 0.265625),
            (new[] { 0.0, 1.0, -2.0, 3.0, 0.0, 4.0 }, 0.875, 0.1875),
            (Linspace(-3, 8, 20), 0.9963521957397461, 0.004154205322265625),
            (RoundedSin(30), 0.6482741845975598, 0.3517258154024402),
            (Enumerable.Repeat(1.0, 5).Concat(Enumerable.Repeat(-1.0, 9)).ToArray(),
                0.14252470370130638, 0.8574752962986936),
        };

        foreach (var (d, expectedLess, expectedGreater) in cases)
        {
            double less = SignedRankTest.PValueZsplit(d, SignedRankTest.Alternative.Less);
            double greater = SignedRankTest.PValueZsplit(d, SignedRankTest.Alternative.Greater);
            less.Should().BeApproximately(expectedLess, 1e-12, $"less p for [{string.Join(",", d.Take(4))}...]");
            greater.Should().BeApproximately(expectedGreater, 1e-12, $"greater p for [{string.Join(",", d.Take(4))}...]");
        }
    }

    private static double[] Linspace(double start, double stop, int num)
    {
        var result = new double[num];
        double step = (stop - start) / (num - 1);
        for (int i = 0; i < num; i++)
            result[i] = start + step * i;
        result[num - 1] = stop;
        return result;
    }

    private static double[] RoundedSin(int n)
        => Enumerable.Range(0, n).Select(i => Math.Round(Math.Sin(i) * 4, 1)).ToArray();

    [TestMethod]
    public void WilcoxonPruner_EndToEnd()
    {
        // Optuna's doc example shape: minimize mean loss over problem instances.
        var instances = Linspace(-1, 1, 50);
        var study = Optuna.CreateStudy(
            sampler: new RandomSampler(2), pruner: new WilcoxonPruner(pThreshold: 0.1));
        study.Optimize(t =>
        {
            double param = t.SuggestFloat("param", -1, 1);
            var results = new List<double>();
            for (int i = 0; i < instances.Length; i++)
            {
                double loss = Math.Pow(param - instances[i], 2);
                results.Add(loss);
                t.Report(loss, i);
                if (t.ShouldPrune())
                    return results.Average();  // The documented early-return workaround.
            }
            return results.Average();
        }, nTrials: 20);

        study.Trials.Should().HaveCount(20);
        study.Trials.Should().OnlyContain(t => t.State == TrialState.Complete);
        study.BestValue.Should().BeGreaterThanOrEqualTo(0);
    }

    [TestMethod]
    public void WilcoxonPruner_Validates_Arguments()
    {
        var p = () => new WilcoxonPruner(pThreshold: 1.5);
        p.Should().Throw<ValueError>().WithMessage("p_threshold must be between 0 and 1 but got 1.5.");
        var s = () => new WilcoxonPruner(nStartupSteps: -1);
        s.Should().Throw<ValueError>().WithMessage("n_startup_steps must be nonnegative but got -1.");
    }
}
