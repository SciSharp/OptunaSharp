using System;
using System.Collections.Generic;
using NumSharp;
using OptunaSharp.Samplers;

namespace OptunaSharp.Terminator;

/// <summary>
///     Evaluates a kind of regret, called the Expected Minimum Model Regret (EMMR).
///     Port of <c>optuna/terminator/improvement/emmr.py: EMMREvaluator</c>
///     (experimental since 4.0.0).
///
///     EMMR is an upper bound of "expected minimum simple regret" in the optimization process.
///     Expected minimum simple regret is a quantity that converges to zero only if the
///     optimization process has found the global optima.
///
///     NOT PORTED YET — this evaluator rides the native Gaussian-process stack, which arrives with
///     PORTING_PLAN Phase 3.3. Construction is benign (so it can be paired into a
///     <see cref="MedianErrorEvaluator" /> whose own validation runs), but <see cref="Evaluate" />
///     throws until the GP stack lands.
///
///     For further information about expected minimum simple regret and the algorithm, please
///     refer to the following paper:
///     - A stopping criterion for Bayesian optimization by the gap of expected minimum simple
///       regrets (https://proceedings.mlr.press/v206/ishibashi23a.html).
/// </summary>
public class EMMREvaluator : BaseImprovementEvaluator
{
    private readonly bool _deterministic;
    private readonly double _delta;
    private readonly LazyRandomState _rng;

    /// <summary>Python <c>self.min_n_trials</c> (a public attribute).</summary>
    public int MinNTrials;

    /// <param name="deterministicObjective">Whether the objective function is deterministic.</param>
    /// <param name="delta">A float number related to the criterion for termination.</param>
    /// <param name="minNTrials">A minimum number of complete trials to compute the criterion.</param>
    /// <param name="seed">A random seed for EMMREvaluator.</param>
    public EMMREvaluator(
        bool deterministicObjective = false,
        double delta = 0.1,
        int minNTrials = 2,
        int? seed = null)
    {
        if (minNTrials <= 1)
            throw new ValueError("`min_n_trials` is expected to be a finite integer more than one.");

        _deterministic = deterministicObjective;
        _delta = delta;
        MinNTrials = minNTrials;
        _rng = new LazyRandomState(seed);
    }

    public override double Evaluate(IReadOnlyList<FrozenTrial> trials, StudyDirection studyDirection)
        => throw new NotImplementedException(
            "EMMREvaluator is not ported yet (PORTING_PLAN Phase 3.3 — native GP stack). " +
            "Use an explicit non-GP improvement evaluator (e.g. new BestValueStagnationEvaluator()) " +
            "in the meantime.");
}
