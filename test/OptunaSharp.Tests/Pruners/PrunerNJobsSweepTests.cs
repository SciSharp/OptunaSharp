using System;
using System.Linq;
using AwesomeAssertions;
using OptunaSharp.Pruners;
using OptunaSharp.Samplers;

namespace OptunaSharp.Tests.Pruners;

/// <summary>
///     Every Core pruner shared by the workers of <c>Optimize(nJobs: 8)</c>, each trial reporting ten steps and asking
///     whether to prune at every one: trials end COMPLETE or PRUNED, never FAIL or RUNNING, the numbering is
///     contiguous, and each pruner prunes as its rule implies. The rules are read from the same scenario on Optuna 5.0
///     (80 trials at <c>n_jobs=8</c>, three seeds): Nop and Patient (whose wrapped pruner is consulted only after a
///     stagnation this objective never shows) pruned nothing; Threshold pruned exactly the trials whose first report
///     exceeded its bound; Median, Percentile, Wilcoxon, Successive Halving and Hyperband pruned 50–76 of 80.
/// </summary>
/// <remarks>
///     Pruners decide from the history of other trials, which nJobs workers write while they read it. Every pruner call
///     runs under the pruner's monitor, as every sampler call runs under the sampler's; Hyperband, which builds its
///     brackets lazily, crashed Optimize before that (see <c>OptimizeConcurrencyTests</c>).
/// </remarks>
[TestClass]
public class PrunerNJobsSweepTests
{
    // The intermediate value starts 0.3 above x and falls to x by step 3, then stays: improving, then flat.
    private const int Steps = 10;

    /// <summary>Builds the named pruner as the Optuna 5.0 observation ran it.</summary>
    /// <param name="name">A key of the data rows below.</param>
    /// <returns>A new pruner instance.</returns>
    /// <exception cref="ArgumentException"><paramref name="name" /> names no pruner.</exception>
    private static BasePruner Make(string name) => name switch
    {
        "nop" => new NopPruner(),
        "median" => new MedianPruner(),
        "percentile" => new PercentilePruner(25.0),
        "patient" => new PatientPruner(new MedianPruner(), patience: 1),
        "threshold" => new ThresholdPruner(upper: 0.9),
        "wilcoxon" => new WilcoxonPruner(pThreshold: 0.1),
        "sha" => new SuccessiveHalvingPruner(),
        "hyperband" => new HyperbandPruner(maxResource: Steps),
        _ => throw new ArgumentException("unknown pruner " + name, nameof(name)),
    };

    /// <summary>
    ///     Runs 80 trials at <c>nJobs: 8</c> with the named pruner and checks the states it produced against its rule.
    /// </summary>
    /// <param name="name">The pruner under test (see <see cref="Make" />).</param>
    /// <exception cref="TrialPruned">Thrown inside the objective whenever the pruner says so; recorded as PRUNED.</exception>
    [TestMethod]
    [DataRow("nop")]
    [DataRow("median")]
    [DataRow("percentile")]
    [DataRow("patient")]
    [DataRow("threshold")]
    [DataRow("wilcoxon")]
    [DataRow("sha")]
    [DataRow("hyperband")]
    public void NJobs_Pruner_EndsEveryTrialCompleteOrPruned_AsItsRuleImplies(string name)
    {
        var study = Optuna.CreateStudy(sampler: new RandomSampler(50), pruner: Make(name));

        study.Optimize(t =>
        {
            double x = t.SuggestFloat("x", 0, 1);
            for (int step = 0; step < Steps; step++)
            {
                t.Report(x + Math.Max(0, 3 - step) * 0.1, step);
                if (t.ShouldPrune())
                    throw new TrialPruned();
            }
            return x;
        }, nTrials: 80, nJobs: 8);

        var trials = study.Trials;
        trials.Select(t => t.Number).Should().Equal(Enumerable.Range(0, 80));
        trials.Should().OnlyContain(t => t.State == TrialState.Complete || t.State == TrialState.Pruned);
        int pruned = trials.Count(t => t.State == TrialState.Pruned);
        switch (name)
        {
            case "nop":
            case "patient":
                pruned.Should().Be(0, "the objective never stagnates after an improvement, so Patient never consults Median");
                break;
            case "threshold":
                // A pure per-trial rule: the first report, x + 0.3, is the largest, so the trial is pruned at step 0
                // exactly when it exceeds the bound; the report's own expression is repeated so the rounding matches.
                trials.Should().OnlyContain(t => (t.State == TrialState.Pruned) == (Convert.ToDouble(t.Params["x"]) + Math.Max(0, 3 - 0) * 0.1 > 0.9));
                break;
            default:
                pruned.Should().BePositive("Optuna 5.0 pruned most of these trials with this pruner");
                break;
        }
        // A pruned trial keeps its last report as its value (Python's tell of a PRUNED trial), so none is left without one.
        trials.Where(t => t.State == TrialState.Pruned).Should().OnlyContain(t => t.Value != null);
    }
}
