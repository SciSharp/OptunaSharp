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

    /// <summary>
    ///     NumPy <c>np.nanpercentile(values, q)</c> with the default <c>method="linear"</c>, ported operation for
    ///     operation from numpy's <c>_quantile</c> (<c>_get_indexes</c>, <c>_get_gamma</c>, <c>_lerp</c>) so the
    ///     cutoff, and therefore every prune decision, is bit-identical to Optuna's.
    /// </summary>
    /// <remarks>
    ///     <para>Two numpy details decide the result, and a textbook <c>a + (b - a) * t</c> gets both wrong:</para>
    ///     <list type="bullet">
    ///         <item><description><c>_lerp</c> computes <c>b - (b - a) * (1 - t)</c> when <c>t &gt;= 0.5</c>. The two
    ///         formulas differ in the last bit for ordinary data: the median of 0.01 and 0.03 is 0.02 here and in
    ///         numpy but 0.019999999999999997 with the textbook formula, which flips the decision for a trial that
    ///         reported exactly 0.02 (pre-release review item H4).</description></item>
    ///         <item><description>The two neighbours are always combined, even when the virtual index lands exactly
    ///         on an element (<c>t == 0</c>) or at or past the last one (numpy clamps both neighbours to the last
    ///         element and measures <c>t</c> from index -1). An infinite neighbour therefore makes the result NaN
    ///         (<c>inf - inf</c>, <c>inf * 0</c>) or infinite, depending on the branch: the median of
    ///         <c>[1, 2, inf]</c> is NaN, not 2, so the pruner keeps the trial exactly as Optuna does, while the
    ///         median of <c>[-inf, 1]</c> is <c>-inf</c>.</description></item>
    ///     </list>
    ///     <para>NaN inputs are dropped first, as <c>nanpercentile</c> does; the order of the remaining values does
    ///     not matter because they are sorted. Only the sign of a zero result can differ from numpy (its partition
    ///     may order <c>-0.0</c> and <c>0.0</c> differently), which no comparison can observe.</para>
    /// </remarks>
    /// <param name="values">Numerical observations consumed in their supplied order.</param>
    /// <param name="percentile">Percentage defining the retained reference cutoff, in [0, 100].</param>
    /// <returns>
    ///     The interpolated percentile of the non-NaN values, or NaN when every value is NaN or there are none. With an
    ///     infinite value among the two interpolated neighbours the result is NaN or infinite, exactly as numpy's
    ///     (see remarks).
    /// </returns>
    internal static double NanPercentile(IEnumerable<double> values, double percentile)
    {
        var data = values.Where(v => !double.IsNaN(v)).ToArray();
        if (data.Length == 0)
            return double.NaN;
        Array.Sort(data);

        int n = data.Length;
        // np.true_divide(q, 100), then the 'linear' method's virtual index (n - 1) * q, both in float64.
        double quantile = percentile / 100.0;
        double virtualIndex = (n - 1) * quantile;

        // _get_indexes: the floor and the element after it. An index at or past the last element takes the last
        // element for both neighbours (numpy stores -1); a negative one takes the first.
        double previousIndex = Math.Floor(virtualIndex);
        double nextIndex = previousIndex + 1;
        if (virtualIndex >= n - 1)
        {
            previousIndex = -1;
            nextIndex = -1;
        }
        if (virtualIndex < 0)
        {
            previousIndex = 0;
            nextIndex = 0;
        }

        double previous = data[previousIndex < 0 ? n - 1 : (int)previousIndex];
        double next = data[nextIndex < 0 ? n - 1 : (int)nextIndex];

        // _get_gamma measures from the clamped index: a clamped -1 gives gamma = virtualIndex + 1 (>= 1), which
        // sends _lerp down its `b - diff * (1 - t)` branch with diff == 0, or NaN when the last element is infinite.
        double gamma = virtualIndex - previousIndex;
        return Lerp(previous, next, gamma);
    }

    /// <summary>
    ///     numpy's <c>_lerp(a, b, t)</c>: <c>a + (b - a) * t</c> below <c>t = 0.5</c> and <c>b - (b - a) * (1 - t)</c>
    ///     from 0.5 on. The second form returns <c>b</c> exactly at <c>t == 1</c>, and choosing between the forms the
    ///     way numpy does is what makes the median of two values bit-identical to numpy's.
    /// </summary>
    /// <param name="a">The lower neighbour (the value at the floor of the virtual index).</param>
    /// <param name="b">The upper neighbour.</param>
    /// <param name="t">The interpolation weight; numpy can pass values above 1 for a clamped index.</param>
    /// <returns>The interpolated value; NaN or infinite when either neighbour is infinite, as in numpy.</returns>
    private static double Lerp(double a, double b, double t)
    {
        double diff = b - a;
        return t >= 0.5 ? b - diff * (1 - t) : a + diff * t;
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
