using System;
using System.Collections.Generic;
using System.Linq;
using NumSharp;
using OptunaSharp.Samplers;

namespace OptunaSharp.Terminator;

/// <summary>Module-level constants of <c>optuna/terminator/improvement/evaluator.py</c>.</summary>
internal static class ImprovementEvaluatorDefaults
{
    /// <summary>Python <c>DEFAULT_TOP_TRIALS_RATIO</c>.</summary>
    internal const double DefaultTopTrialsRatio = 0.5;

    /// <summary>Python <c>DEFAULT_MIN_N_TRIALS</c>.</summary>
    internal const int DefaultMinNTrials = 20;
}

/// <summary>
///     Base class for improvement evaluators. Port of
///     <c>optuna/terminator/improvement/evaluator.py: BaseImprovementEvaluator</c>
///     (experimental since 3.2.0).
/// </summary>
public abstract class BaseImprovementEvaluator
{
    /// <summary>Evaluate the room left for optimization.</summary>
    public abstract double Evaluate(IReadOnlyList<FrozenTrial> trials, StudyDirection studyDirection);
}

/// <summary>
///     An improvement evaluator that estimates an upper bound on the regret with high-probability
///     confidence. Port of
///     <c>optuna/terminator/improvement/evaluator.py: RegretBoundEvaluator</c>
///     (experimental since 3.2.0).
///
///     NOT PORTED YET — this evaluator rides the native Gaussian-process stack, which arrives with
///     PORTING_PLAN Phase 3.3. Mirroring Optuna, it is the default <c>improvement_evaluator</c> of
///     <see cref="Terminator" />, so construction is benign (a study using an explicit non-GP
///     evaluator such as <see cref="BestValueStagnationEvaluator" /> works), but
///     <see cref="Evaluate" /> throws until the GP stack lands.
///
///     For further information about this evaluator, please refer to the following paper:
///     - Automatic Termination for Hyperparameter Optimization
///       (https://proceedings.mlr.press/v188/makarova22a.html).
/// </summary>
public class RegretBoundEvaluator : BaseImprovementEvaluator
{
    private readonly double _topTrialsRatio;
    private readonly int _minNTrials;
    private readonly int _optimizeNSamples;
    private readonly LazyRandomState _rng;

    /// <param name="topTrialsRatio">A ratio of top trials to be considered when estimating the regret.</param>
    /// <param name="minNTrials">A minimum number of complete trials to estimate the regret.</param>
    /// <param name="seed">Seed for random number generator.</param>
    public RegretBoundEvaluator(
        double topTrialsRatio = ImprovementEvaluatorDefaults.DefaultTopTrialsRatio,
        int minNTrials = ImprovementEvaluatorDefaults.DefaultMinNTrials,
        int? seed = null)
    {
        _topTrialsRatio = topTrialsRatio;
        _minNTrials = minNTrials;
        _optimizeNSamples = 2048;
        _rng = new LazyRandomState(seed);
    }

    public override double Evaluate(IReadOnlyList<FrozenTrial> trials, StudyDirection studyDirection)
        => throw new NotImplementedException(
            "RegretBoundEvaluator is not ported yet (PORTING_PLAN Phase 3.3 — native GP stack). " +
            "Use an explicit non-GP improvement evaluator (e.g. new BestValueStagnationEvaluator()) " +
            "with Terminator in the meantime.");
}

/// <summary>
///     Evaluates the stagnation period of the best value in an optimization process.
///     Port of <c>optuna/terminator/improvement/evaluator.py: BestValueStagnationEvaluator</c>
///     (experimental since 3.4.0).
///
///     This class is initialized with a maximum stagnation period
///     (<c>max_stagnation_trials</c>) and is designed to evaluate the remaining trials before
///     reaching this maximum period of allowed stagnation. If this remaining trials reach zero,
///     the trial terminates. Therefore, the default error evaluator is instantiated by
///     <c>StaticErrorEvaluator(constant: 0)</c>.
/// </summary>
public class BestValueStagnationEvaluator : BaseImprovementEvaluator
{
    private readonly int _maxStagnationTrials;

    /// <param name="maxStagnationTrials">The maximum number of trials allowed for stagnation.</param>
    public BestValueStagnationEvaluator(int maxStagnationTrials = 30)
    {
        if (maxStagnationTrials < 0)
            throw new ValueError("The maximum number of stagnant trials must not be negative.");

        _maxStagnationTrials = maxStagnationTrials;
    }

    public override double Evaluate(IReadOnlyList<FrozenTrial> trials, StudyDirection studyDirection)
    {
        ValidateInput(trials);
        bool isMaximizeDirection = studyDirection == StudyDirection.Maximize;
        var completeTrials = trials.Where(t => t.State == TrialState.Complete).ToList();
        int currentStep = completeTrials.Count - 1;

        int bestStep = 0;
        for (int i = 0; i < completeTrials.Count; i++)
        {
            double bestValue = completeTrials[bestStep].Value!.Value;
            double currentValue = completeTrials[i].Value!.Value;
            if (isMaximizeDirection && bestValue < currentValue)
                bestStep = i;
            else if (!isMaximizeDirection && bestValue > currentValue)
                bestStep = i;
        }

        return _maxStagnationTrials - (currentStep - bestStep);
    }

    private static void ValidateInput(IReadOnlyList<FrozenTrial> trials)
    {
        if (!trials.Any(t => t.State == TrialState.Complete))
            throw new ValueError(
                "Because no trial has been completed yet, the improvement cannot be evaluated.");
    }
}
