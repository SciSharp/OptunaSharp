using System.Collections.Generic;
using NumSharp;
using OptunaSharp.Terminator;

namespace OptunaSharp.Visualization;

/// <summary>
///     The data backing one study's terminator-improvement plot. Port of <c>_ImprovementInfo</c> in
///     <c>optuna/visualization/_terminator_improvement.py</c>. <see cref="Errors" /> is <c>null</c> when
///     errors were not requested.
/// </summary>
/// <param name="TrialNumbers">The study-local trial numbers aligned with the value or estimate sequences.</param>
/// <param name="Improvements">The estimated remaining improvement aligned with the trial numbers.</param>
/// <param name="Errors">The statistical error estimates aligned with trial numbers, or null when not requested.</param>
public sealed record ImprovementInfo(
    IReadOnlyList<int> TrialNumbers,
    IReadOnlyList<double> Improvements,
    IReadOnlyList<double>? Errors);

/// <summary>
///     Data layer for <c>optuna.visualization.plot_terminator_improvement</c> — the pure
///     <c>_get_improvement_info</c> half of <c>_terminator_improvement.py</c> (the plotly
///     <c>_get_improvement_plot</c>/<c>_get_improvement_scatter</c> rendering is Tier X, PORTING_PLAN
///     5.2). Deferred from Phase 3.1 to 3.3 because it rides the GP improvement evaluators.
///
///     The improvement values ride the native GP stack (<see cref="RegretBoundEvaluator" /> by
///     default), so they are tolerance-exact vs Optuna, not bit-for-bit — the documented GP boundary.
/// </summary>
public static class TerminatorImprovement
{
    /// <summary>
    ///     Python <c>_get_improvement_info(study, get_error, improvement_evaluator, error_evaluator)</c>.
    ///     Walks the study's trials, and for every prefix that contains at least one completed trial,
    ///     evaluates the objective's improvement potential (and, when <paramref name="getError" />, the
    ///     error inherent in the objective).
    /// </summary>
    /// <param name="study">The study whose trials are evaluated for their improvement.</param>
    /// <param name="getError">Whether to also evaluate the error with <paramref name="errorEvaluator" />.</param>
    /// <param name="improvementEvaluator">
    ///     The improvement evaluator; defaults to <see cref="RegretBoundEvaluator" />.
    /// </param>
    /// <param name="errorEvaluator">
    ///     The error evaluator; defaults to <see cref="CrossValidationErrorEvaluator" /> (or
    ///     <see cref="StaticErrorEvaluator" /> with constant 0 when the improvement evaluator is a
    ///     <see cref="BestValueStagnationEvaluator" />).
    /// </param>
    /// <returns>The improvement and optional error estimates computed along the study's completed-trial history.</returns>
    /// <exception cref="NumSharp.ValueError">The study has multiple objectives.</exception>
    public static ImprovementInfo GetInfo(
        Study study,
        bool getError = false,
        BaseImprovementEvaluator? improvementEvaluator = null,
        BaseErrorEvaluator? errorEvaluator = null)
    {
        if (study.IsMultiObjective())
            throw new ValueError(
                "This function does not support multi-objective optimization study.");

        improvementEvaluator ??= new RegretBoundEvaluator();
        if (errorEvaluator == null)
        {
            errorEvaluator = improvementEvaluator is BestValueStagnationEvaluator
                ? new StaticErrorEvaluator(constant: 0)
                : new CrossValidationErrorEvaluator();
        }

        var trialNumbers = new List<int>();
        var completedTrials = new List<FrozenTrial>();
        var improvements = new List<double>();
        var errors = new List<double>();

        foreach (FrozenTrial trial in study.Trials)
        {
            if (trial.State == TrialState.Complete)
                completedTrials.Add(trial);

            if (completedTrials.Count == 0)
                continue;

            trialNumbers.Add(trial.Number);

            double improvement = improvementEvaluator.Evaluate(completedTrials, study.Direction);
            improvements.Add(improvement);

            if (getError)
            {
                double error = errorEvaluator.Evaluate(completedTrials, study.Direction);
                errors.Add(error);
            }
        }

        return errors.Count == 0
            ? new ImprovementInfo(trialNumbers, improvements, null)
            : new ImprovementInfo(trialNumbers, improvements, errors);
    }
}
