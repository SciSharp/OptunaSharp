using System;
using NumSharp;
using OptunaSharp.Internal;

namespace OptunaSharp.Terminator;

/// <summary>
///     The <c>optuna.terminator</c> module deprecation notice. Optuna emits it (as a
///     <c>FutureWarning</c>) from the entry points that construct or run the terminator mechanism —
///     <see cref="Terminator" />, <see cref="TerminatorCallback" />, and
///     <see cref="TerminatorFunctions.ReportCrossValidationScores" />.
/// </summary>
internal static class TerminatorModule
{
    /// <summary>
    ///     Python <c>_DEPRECATION_WARNING_MESSAGE</c> = <c>_DEPRECATION_WARNING_TEMPLATE.format(
    ///     name="`optuna.terminator` module", d_ver="4.9.0", r_ver="6.0.0")</c>.
    /// </summary>
    internal const string DeprecationWarningMessage =
        "`optuna.terminator` module has been deprecated in v4.9.0. This feature will be removed " +
        "in v6.0.0. See https://github.com/optuna/optuna/releases/tag/v4.9.0.";
}

/// <summary>
///     Base class for terminators. Port of <c>optuna/terminator/terminator.py: BaseTerminator</c>.
/// </summary>
public abstract class BaseTerminator
{
    /// <summary>Judge whether the study should be terminated based on the reported values.</summary>
    public abstract bool ShouldTerminate(Study study);
}

/// <summary>
///     Automatic stopping mechanism for Optuna studies.
///     Port of <c>optuna/terminator/terminator.py: Terminator</c>.
///
///     This class implements an automatic stopping mechanism for Optuna studies, aiming to prevent
///     unnecessary computation. The study is terminated when the statistical error, e.g.
///     cross-validation error, exceeds the room left for optimization.
///
///     For further information about the algorithm, please refer to the following paper:
///     - A. Makarova et al. Automatic termination for hyperparameter optimization.
///       (https://proceedings.mlr.press/v188/makarova22a.html).
/// </summary>
public class Terminator : BaseTerminator
{
    private readonly BaseImprovementEvaluator _improvementEvaluator;
    private readonly BaseErrorEvaluator _errorEvaluator;
    private readonly int _minNTrials;

    /// <param name="improvementEvaluator">
    ///     An evaluator object for assessing the room left for optimization. Defaults to a
    ///     <see cref="RegretBoundEvaluator" /> object.
    /// </param>
    /// <param name="errorEvaluator">
    ///     An evaluator for calculating the statistical error, e.g. cross-validation error.
    ///     Defaults to a <see cref="CrossValidationErrorEvaluator" /> object.
    /// </param>
    /// <param name="minNTrials">
    ///     The minimum number of trials before termination is considered. Defaults to <c>20</c>.
    /// </param>
    /// <exception cref="ValueError">If <paramref name="minNTrials" /> is not a positive integer.</exception>
    public Terminator(
        BaseImprovementEvaluator? improvementEvaluator = null,
        BaseErrorEvaluator? errorEvaluator = null,
        int minNTrials = ImprovementEvaluatorDefaults.DefaultMinNTrials)
    {
        OptunaWarnings.Warn(TerminatorModule.DeprecationWarningMessage, WarningCategory.FutureWarning);

        if (minNTrials <= 0)
            throw new ValueError("`min_n_trials` is expected to be a positive integer.");

        _improvementEvaluator = improvementEvaluator ?? new RegretBoundEvaluator();
        _errorEvaluator = errorEvaluator ?? InitializeErrorEvaluator();
        _minNTrials = minNTrials;
    }

    private BaseErrorEvaluator InitializeErrorEvaluator()
    {
        if (_improvementEvaluator is BestValueStagnationEvaluator)
            return new StaticErrorEvaluator(constant: 0);
        return new CrossValidationErrorEvaluator();
    }

    /// <summary>Judge whether the study should be terminated based on the reported values.</summary>
    public override bool ShouldTerminate(Study study)
    {
        var trials = study.GetTrials(deepcopy: true, states: new[] { TrialState.Complete });

        if (trials.Count < _minNTrials)
            return false;

        double improvement = _improvementEvaluator.Evaluate(
            trials: study.Trials,
            studyDirection: study.Direction);

        double error = _errorEvaluator.Evaluate(
            trials: study.Trials,
            studyDirection: study.Direction);

        bool shouldTerminate = improvement < error;
        return shouldTerminate;
    }
}
