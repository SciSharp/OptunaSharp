using System;
using System.Collections.Generic;
using System.Linq;
using NumSharp;
using OptunaSharp.Internal;
using OptunaSharp.Logging;
using OptunaSharp.Samplers;

namespace OptunaSharp.Visualization;

/// <summary>
///     The data backing a Pareto-front plot. Port of <c>_ParetoFrontInfo</c> in
///     <c>optuna/visualization/_pareto_front.py</c>. Each trial is paired with the target values used
///     for its axes (the objective values by default).
/// </summary>
/// <param name="NTargets">The number of displayed objective or custom target axes.</param>
/// <param name="TargetNames">The labels of the displayed target axes.</param>
/// <param name="BestTrialsWithValues">Feasible nondominated trials paired with their displayed target coordinates.</param>
/// <param name="NonBestTrialsWithValues">Feasible dominated trials paired with their displayed target coordinates.</param>
/// <param name="InfeasibleTrialsWithValues">Infeasible trials paired with their displayed target coordinates.</param>
/// <param name="AxisOrder">The target-column indices specifying the rendered axis order.</param>
/// <param name="IncludeDominatedTrials">Whether dominated feasible trials should be displayed.</param>
/// <param name="HasConstraints">Whether constraint values were available for classifying trials.</param>
public sealed record ParetoFrontInfo(
    int NTargets,
    IReadOnlyList<string> TargetNames,
    IReadOnlyList<(FrozenTrial Trial, IReadOnlyList<double> Values)> BestTrialsWithValues,
    IReadOnlyList<(FrozenTrial Trial, IReadOnlyList<double> Values)> NonBestTrialsWithValues,
    IReadOnlyList<(FrozenTrial Trial, IReadOnlyList<double> Values)> InfeasibleTrialsWithValues,
    IReadOnlyList<int> AxisOrder,
    bool IncludeDominatedTrials,
    bool HasConstraints);

/// <summary>
///     Data layer for <c>optuna.visualization.plot_pareto_front</c> — the pure
///     <c>_get_pareto_front_info</c> half of <c>_pareto_front.py</c>.
/// </summary>
public static class ParetoFront
{
    private static readonly Logger Logger =
        OptunaLogging.GetLogger("optuna.visualization._pareto_front");

    /// <summary>
    ///     Python <c>_get_pareto_front_info(study, target_names, include_dominated_trials,
    ///     constraints_func, targets)</c>.
    /// </summary>
    /// <param name="study">The study supplying trial history, objective directions, and storage.</param>
    /// <param name="targetNames">The displayed target-axis labels, or null to use objective names.</param>
    /// <param name="includeDominatedTrials">Whether feasible dominated trials should be included in the returned plot data.</param>
    /// <param name="constraintsFunc">A deprecated per-trial constraint selector; null uses recorded trial constraints.</param>
    /// <param name="targets">A selector for the displayed two or three target coordinates; null uses the trial objectives.</param>
    /// <returns>The feasible Pareto, feasible dominated, and infeasible trial groups with their plotted coordinates.</returns>
    /// <exception cref="NumSharp.ValueError">The plot does not have two or three targets, supplied target names have the wrong length, or an empty study with custom targets has no target names.</exception>
    public static ParetoFrontInfo GetInfo(
        Study study,
        IReadOnlyList<string>? targetNames = null,
        bool includeDominatedTrials = true,
        Func<FrozenTrial, IReadOnlyList<double>>? constraintsFunc = null,
        Func<FrozenTrial, IReadOnlyList<double>>? targets = null)
    {
        if (constraintsFunc is not null)
        {
            string msg = string.Format(Deprecations.Template, "`constraints_func`", "4.0.0", "6.0.0");
            OptunaWarnings.Warn(msg, WarningCategory.FutureWarning);
        }

        var feasibleTrials = new List<FrozenTrial>();
        var infeasibleTrials = new List<FrozenTrial>();
        bool hasConstraints = false;
        foreach (var trial in study.GetTrials(deepcopy: false, states: new[] { TrialState.Complete }))
        {
            if (constraintsFunc is not null)
            {
                // NOTE(nabenabe0928): This part is deprecated.
                hasConstraints = true;
                if (constraintsFunc(trial).All(x => x <= 0.0))
                    feasibleTrials.Add(trial);
                else
                    infeasibleTrials.Add(trial);
                continue;
            }

            var constraints = trial.Constraints;
            hasConstraints |= constraints.Count > 0;
            if (constraints.Values.All(x => x <= 0.0))
                feasibleTrials.Add(trial);
            else
                infeasibleTrials.Add(trial);
        }

        var bestTrials = MultiObjective.GetParetoFrontTrialsByTrials(feasibleTrials, study.Directions);
        List<FrozenTrial> nonBestTrials = includeDominatedTrials
            ? GetNonParetoFrontTrials(feasibleTrials, bestTrials)
            : new List<FrozenTrial>();

        if (bestTrials.Count == 0)
        {
            string whatTrial = hasConstraints ? "completed and feasible" : "completed";
            Logger.Warning($"Your study does not have any {whatTrial} trials. ");
        }

        var effectiveTargets = targets;
        if (effectiveTargets is null)
        {
            if (study.Directions.Count is 2 or 3)
                effectiveTargets = t => t.Values!;
            else
                throw new ValueError(
                    "`plot_pareto_front` function only supports 2 or 3 objective" +
                    " studies when using `targets` is `None`. Please use `targets`" +
                    " if your objective studies have more than 3 objectives.");
        }

        List<(FrozenTrial, IReadOnlyList<double>)> MakeTrialsWithValues(List<FrozenTrial> trials)
            => trials.Select(trial => (trial, (IReadOnlyList<double>)effectiveTargets(trial).ToList())).ToList();

        var bestTrialsWithValues = MakeTrialsWithValues(bestTrials);
        var nonBestTrialsWithValues = MakeTrialsWithValues(nonBestTrials);
        var infeasibleTrialsWithValues = MakeTrialsWithValues(infeasibleTrials);

        static int? InferNTargets(IReadOnlyList<(FrozenTrial, IReadOnlyList<double> Values)> t)
            => t.Count > 0 ? t[0].Values.Count : null;

        // Python `A or B`: fall through to B when A is None (or the falsy 0).
        int? nBest = InferNTargets(bestTrialsWithValues);
        int? nTargets = (nBest ?? 0) != 0 ? nBest : InferNTargets(infeasibleTrialsWithValues);
        if (nTargets is null)
        {
            if (targetNames is not null)
                nTargets = targetNames.Count;
            else if (targets is null)
                nTargets = study.Directions.Count;
            else
                throw new ValueError(
                    "If `targets` is specified for empty studies, `target_names` must be specified.");
        }

        if (nTargets is not (2 or 3))
            throw new ValueError(
                "`plot_pareto_front` function only supports 2 or 3 targets." +
                $" you used {nTargets} targets now.");

        IReadOnlyList<string> resolvedTargetNames;
        if (targetNames is null)
        {
            var metricNames = study.MetricNames;
            resolvedTargetNames = metricNames
                ?? Enumerable.Range(0, nTargets.Value).Select(i => $"Objective {i}").ToList();
        }
        else if (targetNames.Count != nTargets)
        {
            throw new ValueError($"The length of `target_names` is supposed to be {nTargets}.");
        }
        else
        {
            resolvedTargetNames = targetNames;
        }

        var axisOrder = Enumerable.Range(0, nTargets.Value).ToList();
        return new ParetoFrontInfo(
            nTargets.Value,
            resolvedTargetNames,
            bestTrialsWithValues,
            nonBestTrialsWithValues,
            infeasibleTrialsWithValues,
            axisOrder,
            includeDominatedTrials,
            hasConstraints);
    }

    /// <summary>Python <c>_get_non_pareto_front_trials(trials, pareto_trials)</c>.</summary>
    /// <param name="trials">Trial snapshots to inspect; their order is preserved unless the operation explicitly ranks them.</param>
    /// <param name="paretoTrials">The feasible nondominated trials to exclude from the dominated-trial result.</param>
    /// <returns>The feasible trials that are absent from the supplied Pareto front.</returns>
    private static List<FrozenTrial> GetNonParetoFrontTrials(
        List<FrozenTrial> trials, List<FrozenTrial> paretoTrials)
    {
        var paretoSet = new HashSet<FrozenTrial>(paretoTrials);
        return trials.Where(t => !paretoSet.Contains(t)).ToList();
    }
}
