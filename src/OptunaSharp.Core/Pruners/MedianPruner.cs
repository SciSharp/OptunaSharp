namespace OptunaSharp.Pruners;

/// <summary>
///     Pruner using the median stopping rule. Port of <c>optuna/pruners/_median.py: MedianPruner</c>.
///
///     Prune if the trial's best intermediate result is worse than median of intermediate results
///     of previous trials at the same step. It stops unpromising trials early based on the
///     intermediate results compared against the median of previous completed trials.
///
///     The pruner handles NaN values in the following manner:
///     1. If all intermediate values of the current trial are NaN, the trial will be pruned.
///     2. During the median calculation across completed trials, NaN values are ignored.
///        Only valid numeric values are considered.
/// </summary>
public class MedianPruner : PercentilePruner
{
    /// <param name="nStartupTrials">Pruning is disabled until the given number of trials finish in the same study.</param>
    /// <param name="nWarmupSteps">
    ///     Pruning is disabled while the current step is less than <c>nWarmupSteps</c>; the
    ///     earliest a trial can be pruned is at <c>step == nWarmupSteps</c>.
    /// </param>
    /// <param name="intervalSteps">
    ///     Interval in number of steps between the pruning checks, offset by the warmup steps.
    /// </param>
    /// <param name="nMinTrials">
    ///     Minimum number of reported trial results at a step to judge whether to prune.
    /// </param>
    public MedianPruner(
        int nStartupTrials = 5,
        int nWarmupSteps = 0,
        int intervalSteps = 1,
        int nMinTrials = 1)
        : base(50.0, nStartupTrials, nWarmupSteps, intervalSteps, nMinTrials)
    {
    }
}
