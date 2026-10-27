using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using NumSharp;

namespace OptunaSharp.Pruners;

/// <summary>
///     Pruner using Asynchronous Successive Halving Algorithm.
///     Port of <c>optuna/pruners/_successive_halving.py: SuccessiveHalvingPruner</c>.
///
///     Successive Halving is a bandit-based algorithm to identify the best one among multiple
///     configurations. This class implements an asynchronous version of Successive Halving (ASHA).
///
///     Note that this class does not take care of the parameter for the maximum resource,
///     referred to as R in the paper. The maximum resource allocated to a trial is typically
///     limited inside the objective function.
/// </summary>
/// <seealso href="https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.SuccessiveHalvingPruner.html">Official Optuna SuccessiveHalvingPruner reference.</seealso>
public class SuccessiveHalvingPruner : BasePruner
{
    private long? _minResource;
    private readonly int _reductionFactor;
    private readonly int _minEarlyStoppingRate;
    private readonly int _bootstrapCount;

    /// <param name="minResource">
    ///     The minimum resource allocated to a trial (<c>r</c> in the paper), or <c>null</c> for
    ///     Optuna's <c>"auto"</c>: the value is determined based on a heuristic that looks at the
    ///     number of required steps for the first trial to complete.
    /// </param>
    /// <param name="reductionFactor">
    ///     Reduction factor of promotable trials (<c>eta</c> in the paper). At the completion
    ///     point of each rung, about <c>1/reductionFactor</c> trials will be promoted.
    /// </param>
    /// <param name="minEarlyStoppingRate">The minimum early-stopping rate (<c>s</c> in the paper).</param>
    /// <param name="bootstrapCount">
    ///     Minimum number of trials that need to complete a rung before any trial is considered
    ///     for promotion into the next rung.
    /// </param>
    /// <summary>Configures asynchronous rung promotion and its minimum resource budget.</summary>
    /// <exception cref="NumSharp.ValueError">The minimum resource is below one, the reduction factor is below two, an early-stopping/bootstrap count is negative, or positive bootstrap is combined with automatic minimum resource.</exception>
    public SuccessiveHalvingPruner(
        long? minResource = null,
        int reductionFactor = 4,
        int minEarlyStoppingRate = 0,
        int bootstrapCount = 0)
    {
        // Rung budgets must increase, and bootstrap needs an explicit first rung so that every
        // participant competes under the same resource schedule from its first report.
        if (minResource is not null && minResource < 1)
            throw new ValueError(
                $"The value of `min_resource` is {minResource}, " +
                "but must be either `min_resource >= 1` or 'auto'");

        if (reductionFactor < 2)
            throw new ValueError(
                $"The value of `reduction_factor` is {reductionFactor}, " +
                "but must be `reduction_factor >= 2`");

        if (minEarlyStoppingRate < 0)
            throw new ValueError(
                $"The value of `min_early_stopping_rate` is {minEarlyStoppingRate}, " +
                "but must be `min_early_stopping_rate >= 0`");

        if (bootstrapCount < 0)
            throw new ValueError(
                "The value of `bootstrap_count` is " +
                $"{bootstrapCount}, but must be `bootstrap_count >= 0`");

        if (bootstrapCount > 0 && minResource is null)
            throw new ValueError(
                "bootstrap_count > 0 and min_resource == 'auto' " +
                $"are mutually incompatible, bootstrap_count is {bootstrapCount}");

        _minResource = minResource;
        _reductionFactor = reductionFactor;
        _minEarlyStoppingRate = minEarlyStoppingRate;
        _bootstrapCount = bootstrapCount;
    }

    /// <inheritdoc />
    /// <exception cref="NumSharp.RuntimeError">The study has multiple objectives and the pruning rule requests its scalar direction.</exception>
    public override bool Prune(Study study, FrozenTrial trial)
    {
        long? stepNullable = trial.LastStep;
        if (stepNullable is null)
            return false;
        long step = stepNullable.Value;

        int rung = GetCurrentRung(trial);
        double value = trial.IntermediateValues[step];
        List<FrozenTrial>? trials = null;

        while (true)
        {
            if (_minResource is null)
            {
                trials ??= study.GetTrials(deepcopy: false);
                _minResource = EstimateMinResource(trials);
                if (_minResource is null)
                    return false;
            }

            long rungPromotionStep = _minResource.Value *
                (long)Math.Pow(_reductionFactor, _minEarlyStoppingRate + rung);
            if (step < rungPromotionStep)
                return false;

            if (double.IsNaN(value))
                return true;

            trials ??= study.GetTrials(deepcopy: false);

            string rungKey = CompletedRungKey(rung);

            // Publish the rung result before deciding promotion so later asynchronous trials see
            // this completed competition even when the current trial is about to be pruned.
            study.Storage.SetTrialSystemAttr(trial.TrialId, rungKey, value);

            var competing = GetCompetingValues(trials, value, rungKey);

            // 'competing' already includes the current trial.
            // Therefore, we need to use the '<=' operator here.
            if (competing.Count <= _bootstrapCount)
                return true;

            if (!IsTrialPromotableToNextRung(value, competing, _reductionFactor, study.Direction))
                return true;

            rung += 1;
        }
    }

    /// <summary>Python <c>_estimate_min_resource</c>.</summary>
    /// <param name="trials">Trial history available to the operation, in its existing order.</param>
    /// <returns>The heuristic minimum budget, or null until a completed trial supplies a last step.</returns>
    internal static long? EstimateMinResource(IReadOnlyList<FrozenTrial> trials)
    {
        var nSteps = trials
            .Where(t => t.State == TrialState.Complete && t.LastStep is not null)
            .Select(t => t.LastStep!.Value)
            .ToList();

        if (nSteps.Count == 0)
            return null;

        // Get the maximum number of steps and divide it by 100.
        long lastStep = nSteps.Max();
        return Math.Max(lastStep / 100, 1L);
    }

    /// <summary>Python <c>_get_current_rung</c> (takes O(log step) iterations).</summary>
    /// <param name="trial">Trial snapshot being examined; copy its contents before modifying them.</param>
    /// <returns>The first rung for which the trial has no stored completion value.</returns>
    internal static int GetCurrentRung(FrozenTrial trial)
    {
        int rung = 0;
        while (trial.SystemAttrsDict.ContainsKey(CompletedRungKey(rung)))
            rung += 1;
        return rung;
    }

    /// <summary>Python <c>_completed_rung_key</c>.</summary>
    /// <param name="rung">Zero-based successive-halving rung.</param>
    /// <returns>The storage attribute key for this rung.</returns>
    internal static string CompletedRungKey(int rung)
        => string.Format(CultureInfo.InvariantCulture, "completed_rung_{0}", rung);

    /// <summary>Python <c>_get_competing_values</c>.</summary>
    /// <param name="trials">Trial history available to the operation, in its existing order.</param>
    /// <param name="value">Current trial report competing for promotion at this rung.</param>
    /// <param name="rungKey">System-attribute key holding the completed rung value.</param>
    /// <returns>Stored values from the same rung followed by the current trial value.</returns>
    internal static List<double> GetCompetingValues(
        IReadOnlyList<FrozenTrial> trials, double value, string rungKey)
    {
        var competingValues = trials
            .Where(t => t.SystemAttrsDict.ContainsKey(rungKey))
            .Select(t => Convert.ToDouble(t.SystemAttrsDict[rungKey], CultureInfo.InvariantCulture))
            .ToList();
        competingValues.Add(value);
        return competingValues;
    }

    /// <summary>Python <c>_is_trial_promotable_to_next_rung</c>.</summary>
    /// <param name="value">Current trial report competing for promotion at this rung.</param>
    /// <param name="competingValues">Rung values including the current trial; sorting may reorder this list.</param>
    /// <param name="reductionFactor">Factor controlling the fraction of individuals promoted at each rung.</param>
    /// <param name="studyDirection">Whether objective values are minimized or maximized.</param>
    /// <returns>True when the value meets the direction-aware promotion cutoff.</returns>
    internal static bool IsTrialPromotableToNextRung(
        double value,
        List<double> competingValues,
        int reductionFactor,
        StudyDirection studyDirection)
    {
        int promotableIdx = (competingValues.Count / reductionFactor) - 1;

        if (promotableIdx == -1)
        {
            // Optuna does not support suspending or resuming ongoing trials. Therefore, for the
            // first `eta - 1` trials, this implementation instead promotes the trial if its value
            // is the smallest one among the competing values.
            promotableIdx = 0;
        }

        competingValues.Sort();
        if (studyDirection == StudyDirection.Maximize)
            return value >= competingValues[competingValues.Count - (promotableIdx + 1)];
        return value <= competingValues[promotableIdx];
    }
}
