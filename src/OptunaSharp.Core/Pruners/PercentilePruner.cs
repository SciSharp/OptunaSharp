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
/// <seealso href="https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.PercentilePruner.html">Official Optuna PercentilePruner reference.</seealso>
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
    /// <summary>Configures the reference percentile and evidence required before comparing trials.</summary>
    /// <exception cref="NumSharp.ValueError">The percentile is outside [0, 100], a startup/warmup count is negative, or an interval/minimum count is below one.</exception>
    public PercentilePruner(
        double percentile,
        int nStartupTrials = 5,
        int nWarmupSteps = 0,
        int intervalSteps = 1,
        int nMinTrials = 1)
    {
        // A percentile and a nonempty check interval must be meaningful before any trial history
        // is observed; retaining these guards also preserves Optuna's eager validation contract.
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

    /// <inheritdoc />
    /// <exception cref="NumSharp.RuntimeError">The study has multiple objectives and the pruning rule requests its scalar direction.</exception>
    public override bool Prune(Study study, FrozenTrial trial)
    {
        var completedTrials = study.GetTrials(deepcopy: false, states: new[] { TrialState.Complete });
        int nTrials = completedTrials.Count;

        // Startup and warmup gates prevent comparisons before sufficient reference evidence exists.
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
        // A trial with no numeric progress can be stopped, while a missing reference cutoff below
        // means there is no evidence against an otherwise usable current trial.
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
    /// <param name="trial">Trial snapshot being examined; copy its contents before modifying them.</param>
    /// <param name="direction">Whether objective values are minimized or maximized.</param>
    /// <returns>The best non-NaN report for the direction, or NaN when none exists.</returns>
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
    /// <param name="completedTrials">Completed trial snapshots used as historical observations.</param>
    /// <param name="direction">Whether objective values are minimized or maximized.</param>
    /// <param name="step">Positive spacing between adjacent quantized numerical values.</param>
    /// <param name="percentile">Percentage defining the retained reference cutoff, in [0, 100].</param>
    /// <param name="nMinTrials">Minimum reference report count required before computing a cutoff.</param>
    /// <returns>The oriented percentile cutoff, or NaN when reference reports are insufficient.</returns>
    /// <exception cref="NumSharp.ValueError">The completed-trial collection is empty.</exception>
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
    /// <param name="values">Numerical observations consumed in their supplied order.</param>
    /// <param name="percentile">Percentage defining the retained reference cutoff, in [0, 100].</param>
    /// <returns>The linearly interpolated percentile after excluding NaNs, or NaN for no usable data.</returns>
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
    /// <param name="step">Positive spacing between adjacent quantized numerical values.</param>
    /// <param name="intermediateSteps">Previously reported steps, which need not be sorted.</param>
    /// <param name="nWarmupSteps">Resource-step offset before pruning checks begin.</param>
    /// <param name="intervalSteps">Positive number of steps between scheduled pruning checks.</param>
    /// <returns>True if no earlier report has already triggered the current scheduled interval.</returns>
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
