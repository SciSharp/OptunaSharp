using System;
using NumSharp;

namespace OptunaSharp.Pruners;

/// <summary>
///     Pruner to detect outlying metrics of the trials.
///     Port of <c>optuna/pruners/_threshold.py: ThresholdPruner</c>.
///
///     Prune if a metric exceeds upper threshold, falls behind lower threshold or reaches
///     <c>NaN</c>.
/// </summary>
public class ThresholdPruner : BasePruner
{
    private readonly double _lower;
    private readonly double _upper;
    private readonly int _nWarmupSteps;
    private readonly int _intervalSteps;

    /// <param name="lower">
    ///     A minimum value which determines whether pruner prunes or not. If an intermediate
    ///     value is smaller than lower, it prunes.
    /// </param>
    /// <param name="upper">
    ///     A maximum value which determines whether pruner prunes or not. If an intermediate
    ///     value is larger than upper, it prunes.
    /// </param>
    /// <param name="nWarmupSteps">Pruning is disabled if the step is less than the given number of warmup steps.</param>
    /// <param name="intervalSteps">
    ///     Interval in number of steps between the pruning checks, offset by the warmup steps.
    ///     If no value has been reported at the time of a pruning check, that particular check
    ///     will be postponed until a value is reported. Value must be at least 1.
    /// </param>
    public ThresholdPruner(
        double? lower = null,
        double? upper = null,
        int nWarmupSteps = 0,
        int intervalSteps = 1)
    {
        if (lower is null && upper is null)
            throw new TypeError("Either lower or upper must be specified.");

        double lowerValue = lower ?? double.NegativeInfinity;
        double upperValue = upper ?? double.PositiveInfinity;

        if (lowerValue > upperValue)
            throw new ValueError("lower should be smaller than upper.");
        if (nWarmupSteps < 0)
            throw new ValueError($"Number of warmup steps cannot be negative but got {nWarmupSteps}.");
        if (intervalSteps < 1)
            throw new ValueError($"Pruning interval steps must be at least 1 but got {intervalSteps}.");

        _lower = lowerValue;
        _upper = upperValue;
        _nWarmupSteps = nWarmupSteps;
        _intervalSteps = intervalSteps;
    }

    public override bool Prune(Study study, FrozenTrial trial)
    {
        int? stepNullable = trial.LastStep;
        if (stepNullable is null)
            return false;
        int step = stepNullable.Value;

        int nWarmupSteps = _nWarmupSteps;
        if (step < nWarmupSteps)
            return false;

        if (!PercentilePruner.IsFirstInIntervalStep(
                step, trial.IntermediateValues.Keys, nWarmupSteps, _intervalSteps))
        {
            return false;
        }

        double latestValue = trial.IntermediateValues[step];
        if (double.IsNaN(latestValue))
            return true;

        if (latestValue < _lower)
            return true;

        if (latestValue > _upper)
            return true;

        return false;
    }
}
