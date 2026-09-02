using System;
using System.Collections.Generic;
using System.Linq;
using NumSharp;

namespace OptunaSharp.Terminator;

/// <summary>
///     An error evaluator that returns the ratio to the initial median.
///     Port of <c>optuna/terminator/median_erroreval.py: MedianErrorEvaluator</c>
///     (experimental since 4.0.0).
///
///     This error evaluator is introduced as a heuristic in the following paper:
///     - A stopping criterion for Bayesian optimization by the gap of expected minimum simple
///       regrets (https://proceedings.mlr.press/v206/ishibashi23a.html).
/// </summary>
public class MedianErrorEvaluator : BaseErrorEvaluator
{
    // Python `sys.float_info.min` — the smallest positive *normalized* double (2**-1022).
    private const double FloatInfoMin = 2.2250738585072014e-308;

    // Python `sys.float_info.max` — the largest finite double (== double.MaxValue).
    private const double FloatInfoMax = double.MaxValue;

    private readonly BaseImprovementEvaluator _pairedImprovementEvaluator;
    private readonly int _warmUpTrials;
    private readonly int _nInitialTrials;
    private readonly double _thresholdRatio;
    private double? _threshold;

    /// <param name="pairedImprovementEvaluator">
    ///     The <c>improvement_evaluator</c> instance which is set with this <c>error_evaluator</c>.
    /// </param>
    /// <param name="warmUpTrials">
    ///     The number of initial trials to be discarded before the calculation of the median.
    ///     In Optuna, the first 10 trials are often random sampling; <c>warmUpTrials</c> can
    ///     exclude them from the calculation.
    /// </param>
    /// <param name="nInitialTrials">
    ///     The number of initial trials considered in the calculation of the median after
    ///     <c>warmUpTrials</c>.
    /// </param>
    /// <param name="thresholdRatio">The ratio between the threshold and the initial median.</param>
    public MedianErrorEvaluator(
        BaseImprovementEvaluator pairedImprovementEvaluator,
        int warmUpTrials = 10,
        int nInitialTrials = 20,
        double thresholdRatio = 0.01)
    {
        if (warmUpTrials < 0)
            throw new ValueError("`warm_up_trials` is expected to be a non-negative integer.");
        if (nInitialTrials <= 0)
            throw new ValueError("`n_initial_trials` is expected to be a positive integer.");
        if (thresholdRatio <= 0.0 || !double.IsFinite(thresholdRatio))
            throw new ValueError("`threshold_ratio_to_initial_median` is expected to be a positive.");

        _pairedImprovementEvaluator = pairedImprovementEvaluator;
        _warmUpTrials = warmUpTrials;
        _nInitialTrials = nInitialTrials;
        _thresholdRatio = thresholdRatio;
        _threshold = null;
    }

    public override double Evaluate(IReadOnlyList<FrozenTrial> trials, StudyDirection studyDirection)
    {
        if (_threshold is not null)
            return _threshold.Value;

        var completeTrials = trials.Where(t => t.State == TrialState.Complete).ToList();
        if (completeTrials.Count < _warmUpTrials + _nInitialTrials)
        {
            // Do not terminate. It assumes that improvement must be non-negative.
            return -FloatInfoMin;
        }

        // trials.sort(key=lambda trial: trial.number) — numbers are unique, so this is a total order.
        completeTrials = completeTrials.OrderBy(t => t.Number).ToList();
        var criteria = new List<double>();
        for (int i = 1; i <= _nInitialTrials; i++)
        {
            // Python slice trials[warm_up : warm_up + i].
            var slice = completeTrials.GetRange(_warmUpTrials, i);
            criteria.Add(_pairedImprovementEvaluator.Evaluate(slice, studyDirection));
        }
        criteria.Sort();
        double threshold = criteria[criteria.Count / 2];
        threshold = Math.Min(FloatInfoMax, threshold * _thresholdRatio);
        _threshold = threshold;
        return threshold;
    }
}
