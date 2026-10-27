using System;
using System.Collections.Generic;
using System.Linq;
using NumSharp;
using OptunaSharp.Internal;

namespace OptunaSharp.Pruners;

/// <summary>
///     Pruner based on the Wilcoxon signed-rank test.
///     Port of <c>optuna/pruners/_wilcoxon.py: WilcoxonPruner</c> (experimental since 3.6.0).
///
///     This pruner performs the Wilcoxon signed-rank test between the current trial and the
///     current best trial, and stops whenever the pruner is sure up to a given p-value that the
///     current trial is worse than the best one.
///
///     This pruner is effective for optimizing the mean/median of some (costly-to-evaluate)
///     performance scores over a set of problem instances. When you use this pruner, you must
///     call <c>Trial.Report(value, step)</c> for each step (instance id) with the evaluated
///     value; the instance id may not be in ascending order. Unlike other pruners, the reported
///     values need not converge to the real value.
///
///     This pruner cannot handle infinity or NaN values; trials containing those values are
///     never pruned. The scipy dependency is replaced by a self-contained signed-rank test
///     (plan 2.3), verified against scipy.
/// </summary>
/// <seealso href="https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.WilcoxonPruner.html">Official Optuna WilcoxonPruner reference.</seealso>
public class WilcoxonPruner : BasePruner
{
    private readonly int _nStartupSteps;
    private readonly double _pThreshold;

    /// <param name="pThreshold">
    ///     The p-value threshold for pruning, between 0 and 1. A trial will be pruned whenever
    ///     the pruner is sure up to the given p-value that the current trial is worse than the
    ///     best trial. The larger this value is, the more aggressive pruning will be performed.
    ///     Note that this pruner repeatedly performs statistical tests with increasing samples;
    ///     to get the nominal false-positive rate, specify the Pocock-corrected p-value.
    /// </param>
    /// <param name="nStartupSteps">
    ///     The number of steps before which no trials are pruned. Pruning starts only after you
    ///     have <c>nStartupSteps</c> steps of available observations for comparison between the
    ///     current trial and the best trial. Note the trial is not pruned at the first and
    ///     second steps even if set to 0 or 1, due to the lack of enough data.
    /// </param>
    /// <summary>Configures a paired signed-rank test against the current best trial.</summary>
    /// <exception cref="NumSharp.ValueError">The startup count is negative or the p-value threshold is outside [0, 1].</exception>
    public WilcoxonPruner(double pThreshold = 0.1, int nStartupSteps = 2)
    {
        if (nStartupSteps < 0)
            throw new ValueError($"n_startup_steps must be nonnegative but got {nStartupSteps}.");
        if (!(0.0 <= pThreshold && pThreshold <= 1.0))
            throw new ValueError($"p_threshold must be between 0 and 1 but got {PythonRepr.FloatRepr(pThreshold)}.");

        _nStartupSteps = nStartupSteps;
        _pThreshold = pThreshold;
    }

    /// <inheritdoc />
    /// <exception cref="NumSharp.RuntimeError">The study has multiple objectives and cannot supply a scalar best trial or direction.</exception>
    public override bool Prune(Study study, FrozenTrial trial)
    {
        if (trial.IntermediateValues.Count == 0)
            return false;

        var steps = trial.IntermediateValues.Keys.ToArray();
        var stepValues = steps.Select(s => trial.IntermediateValues[s]).ToArray();

        if (stepValues.Any(v => !double.IsFinite(v)))
        {
            OptunaWarnings.Warn(
                $"The intermediate values of the current trial (trial {trial.Number}) " +
                $"contain infinity/NaNs. WilcoxonPruner will not prune this trial.");
            return false;
        }

        FrozenTrial bestTrial;
        try
        {
            bestTrial = study.BestTrial;
        }
        catch (ValueError)
        {
            // Without a usable incumbent there is no paired reference; missing evidence keeps
            // the current evaluation running instead of turning into a pruning failure.
            return false;
        }

        if (bestTrial.IntermediateValues.Count == 0)
        {
            OptunaWarnings.Warn(
                "The best trial has no intermediate values so WilcoxonPruner cannot prune trials. " +
                "If you have added the best trial with Study.add_trial, please consider setting " +
                "intermediate_values argument.");
            return false;
        }

        var bestSteps = bestTrial.IntermediateValues.Keys.ToArray();
        var bestStepValues = bestSteps.Select(s => bestTrial.IntermediateValues[s]).ToArray();

        if (bestStepValues.Any(v => !double.IsFinite(v)))
        {
            OptunaWarnings.Warn(
                $"The intermediate values of the best trial (trial {bestTrial.Number}) " +
                $"contain infinity/NaNs. WilcoxonPruner will not prune the current trial.");
            return false;
        }

        // np.intersect1d(steps, best_steps, return_indices=True): first-occurrence indices of the
        // sorted common values (steps are dict keys, hence unique).
        var (idx1, idx2) = Intersect1d(steps, bestSteps);

        if (idx1.Count < stepValues.Length)
        {
            // This warning never fires if the "average_is_best" safety works, because the safety
            // ensures that the best trial always has the all steps.
            OptunaWarnings.Warn(
                "WilcoxonPruner finds steps existing in the current trial " +
                "but does not exist in the best trial. " +
                "Those values are ignored.");
        }

        var diffValues = new double[idx1.Count];
        for (int i = 0; i < idx1.Count; i++)
            diffValues[i] = stepValues[idx1[i]] - bestStepValues[idx2[i]];

        if (diffValues.Length < Math.Max(2, _nStartupSteps))
            return false;

        SignedRankTest.Alternative alt;
        bool averageIsBest;
        if (study.Direction == StudyDirection.Maximize)
        {
            alt = SignedRankTest.Alternative.Less;
            averageIsBest = bestStepValues.Sum() / bestStepValues.Length
                <= stepValues.Sum() / stepValues.Length;
        }
        else
        {
            alt = SignedRankTest.Alternative.Greater;
            averageIsBest = bestStepValues.Sum() / bestStepValues.Length
                >= stepValues.Sum() / stepValues.Length;
        }

        // We use zsplit to avoid the problem when all values are zero.
        double p = SignedRankTest.PValueZsplit(diffValues, alt);

        if (p < _pThreshold && averageIsBest)
        {
            // The test found the current trial is probably worse than the best trial, but the
            // value of the best trial was not better than the average of the current trial's
            // intermediate values. For safety, WilcoxonPruner concludes not to prune it for now.
            return false;
        }

        return p < _pThreshold;
    }

    /// <summary>Pairs equal instance identifiers in sorted order, using their first occurrence in each input.</summary>
    /// <param name="a">The current trial's steps (64-bit, like every trial step).</param>
    /// <param name="b">The best trial's steps.</param>
    /// <returns>Paired source indices ordered by their shared instance identifier.</returns>
    private static (List<int> Idx1, List<int> Idx2) Intersect1d(long[] a, long[] b)
    {
        var bIndex = new Dictionary<long, int>();
        for (int i = 0; i < b.Length; i++)
        {
            if (!bIndex.ContainsKey(b[i]))
                bIndex[b[i]] = i;
        }

        var common = new SortedSet<long>(a.Where(bIndex.ContainsKey));
        var idx1 = new List<int>(common.Count);
        var idx2 = new List<int>(common.Count);
        var aIndex = new Dictionary<long, int>();
        for (int i = 0; i < a.Length; i++)
        {
            if (!aIndex.ContainsKey(a[i]))
                aIndex[a[i]] = i;
        }
        foreach (long value in common)
        {
            idx1.Add(aIndex[value]);
            idx2.Add(bIndex[value]);
        }
        return (idx1, idx2);
    }
}
