using System;
using System.Collections.Generic;
using System.Linq;
using NumSharp;

namespace OptunaSharp.Pruners;

/// <summary>
///     Pruner to keep the specified percentile of the trials.
///     Port of <c>optuna/pruners/_percentile.py: PercentilePruner</c>.
///
///     Prune if the best intermediate value is in the bottom percentile among trials at the same
///     step.
/// </summary>
public class PercentilePruner : BasePruner
{
    private readonly double _percentile;
    private readonly int _nStartupTrials;
    private readonly int _nWarmupSteps;
    private readonly int _intervalSteps;
    private readonly int _nMinTrials;

    /// <param name="percentile">
    ///     Percentile which must be between 0 and 100 inclusive (e.g., when given 25.0, top of
    ///     25th percentile trials are kept).
    /// </param>
    /// <param name="nStartupTrials">Pruning is disabled until the given number of trials finish in the same study.</param>
    /// <param name="nWarmupSteps">
    ///     Pruning is disabled while the current step is less than <c>nWarmupSteps</c>; the
    ///     earliest a trial can be pruned is at <c>step == nWarmupSteps</c>.
    /// </param>
    /// <param name="intervalSteps">
    ///     Interval in number of steps between the pruning checks, offset by the warmup steps.
    ///     If no value has been reported at the time of a pruning check, that particular check
    ///     will be postponed until a value is reported. Value must be at least 1.
    /// </param>
    /// <param name="nMinTrials">
    ///     Minimum number of reported trial results at a step to judge whether to prune. If the
    ///     number of reported intermediate values from all trials at the current step is less
    ///     than <c>nMinTrials</c>, the trial will not be pruned.
    /// </param>
    public PercentilePruner(
        double percentile,
        int nStartupTrials = 5,
        int nWarmupSteps = 0,
        int intervalSteps = 1,
        int nMinTrials = 1)
    {
        if (!(0.0 <= percentile && percentile <= 100))
            throw new ValueError(
                $"Percentile must be between 0 and 100 inclusive, but got percentile={Internal.PythonRepr.FloatRepr(percentile)}.");
        if (nStartupTrials < 0)
            throw new ValueError(
                $"Number of startup trials cannot be negative, but got n_startup_trials={nStartupTrials}.");
        if (nWarmupSteps < 0)
            throw new ValueError(
                $"Number of warmup steps cannot be negative, but got n_warmup_steps={nWarmupSteps}.");
        if (intervalSteps < 1)
            throw new ValueError(
                $"Pruning interval steps must be at least 1, but got interval_steps={intervalSteps}.");
        if (nMinTrials < 1)
            throw new ValueError(
                $"Number of trials for pruning must be at least 1, but got n_min_trials={nMinTrials}.");

        _percentile = percentile;
        _nStartupTrials = nStartupTrials;
        _nWarmupSteps = nWarmupSteps;
        _intervalSteps = intervalSteps;
        _nMinTrials = nMinTrials;
    }

    public override bool Prune(Study study, FrozenTrial trial)
    {
        var completedTrials = study.GetTrials(deepcopy: false, states: new[] { TrialState.Complete });
        int nTrials = completedTrials.Count;

        if (nTrials == 0)
            return false;

        if (nTrials < _nStartupTrials)
            return false;

        int? stepNullable = trial.LastStep;
        if (stepNullable is null)
            return false;
        int step = stepNullable.Value;

        int nWarmupSteps = _nWarmupSteps;
        if (step < nWarmupSteps)
            return false;

        if (!IsFirstInIntervalStep(step, trial.IntermediateValues.Keys, nWarmupSteps, _intervalSteps))
            return false;

        var direction = study.Direction;
        double bestIntermediateResult = GetBestIntermediateResultOverSteps(trial, direction);
        if (double.IsNaN(bestIntermediateResult))
            return true;

        double p = GetPercentileIntermediateResultOverTrials(
            completedTrials, direction, step, _percentile, _nMinTrials);
        if (double.IsNaN(p))
            return false;

        if (direction == StudyDirection.Maximize)
            return bestIntermediateResult < p;
        return bestIntermediateResult > p;
    }

    /// <summary>Python <c>_get_best_intermediate_result_over_steps</c> (np.nanmax / np.nanmin).</summary>
    internal static double GetBestIntermediateResultOverSteps(FrozenTrial trial, StudyDirection direction)
    {
        double best = double.NaN;
        foreach (double v in trial.IntermediateValues.Values)
        {
            if (double.IsNaN(v))
                continue;
            if (double.IsNaN(best))
                best = v;
            else
                best = direction == StudyDirection.Maximize ? Math.Max(best, v) : Math.Min(best, v);
        }
        return best;
    }

    /// <summary>Python <c>_get_percentile_intermediate_result_over_trials</c>.</summary>
    internal static double GetPercentileIntermediateResultOverTrials(
        IReadOnlyList<FrozenTrial> completedTrials,
        StudyDirection direction,
        int step,
        double percentile,
        int nMinTrials)
    {
        if (completedTrials.Count == 0)
            throw new ValueError("No trials have been completed.");

        var intermediateValues = new List<double>();
        foreach (var t in completedTrials)
        {
            if (t.IntermediateValues.TryGetValue(step, out double v))
                intermediateValues.Add(v);
        }

        if (intermediateValues.Count < nMinTrials)
            return double.NaN;

        if (direction == StudyDirection.Maximize)
            percentile = 100 - percentile;

        return NanPercentile(intermediateValues, percentile);
    }

    /// <summary>NumPy <c>np.nanpercentile(values, q)</c> with the default linear interpolation.</summary>
    internal static double NanPercentile(IEnumerable<double> values, double percentile)
    {
        var data = values.Where(v => !double.IsNaN(v)).ToArray();
        if (data.Length == 0)
            return double.NaN;
        Array.Sort(data);

        double rank = percentile / 100.0 * (data.Length - 1);
        int lower = (int)Math.Floor(rank);
        int upper = (int)Math.Ceiling(rank);
        if (lower == upper)
            return data[lower];
        return data[lower] + (data[upper] - data[lower]) * (rank - lower);
    }

    /// <summary>Python <c>_is_first_in_interval_step</c>.</summary>
    internal static bool IsFirstInIntervalStep(
        int step, IEnumerable<int> intermediateSteps, int nWarmupSteps, int intervalSteps)
    {
        int nearestLowerPruningStep =
            (step - nWarmupSteps) / intervalSteps * intervalSteps + nWarmupSteps;
        System.Diagnostics.Debug.Assert(nearestLowerPruningStep >= 0);

        // `intermediateSteps` may not be sorted so we must go through all elements.
        int secondLastStep = -1;
        foreach (int s in intermediateSteps)
        {
            if (s > secondLastStep && s != step)
                secondLastStep = s;
        }

        return secondLastStep < nearestLowerPruningStep;
    }
}
