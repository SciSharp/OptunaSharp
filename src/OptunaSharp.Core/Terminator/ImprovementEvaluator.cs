using System;
using System.Collections.Generic;
using System.Linq;
using NumSharp;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Gp;
using OptunaSharp.Internal;
using OptunaSharp.Samplers;
using OptunaSharp.SearchSpace;

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
    /// <param name="trials">Trial snapshots to inspect; their order is preserved unless the operation explicitly ranks them.</param>
    /// <param name="studyDirection">Whether larger or smaller objective values are preferred.</param>
    /// <returns>The estimated remaining room for improvement under this evaluator's criterion.</returns>
    public abstract double Evaluate(IReadOnlyList<FrozenTrial> trials, StudyDirection studyDirection);
}

/// <summary>
///     An improvement evaluator that estimates an upper bound on the regret with high-probability
///     confidence. Port of
///     <c>optuna/terminator/improvement/evaluator.py: RegretBoundEvaluator</c>
///     (experimental since 3.2.0).
///
///     This evaluator evaluates the regret of the current best solution, defined as the difference
///     between the objective value of the best solution and of the global optimum. Specifically, it
///     calculates the upper bound on the regret based on the fact that the empirical estimator of the
///     objective function is bounded by lower and upper confidence bounds with high probability under
///     the Gaussian-process model assumption. It is the default <c>improvement_evaluator</c> of
///     <see cref="Terminator" />.
///
///     Rides the native GP stack (fit + posterior), so its output is tolerance-exact vs Optuna, not
///     bit-for-bit — the documented GP boundary (the posterior goes through the managed triangular
///     solve). The RNG draw for the search-space sampling matches Python's <c>LazyRandomState</c>
///     bit-for-bit, so the sampled point sequence is identical.
///
///     For further information about this evaluator, please refer to the following paper:
///     - Automatic Termination for Hyperparameter Optimization
///       (https://proceedings.mlr.press/v188/makarova22a.html).
/// </summary>
public class RegretBoundEvaluator : BaseImprovementEvaluator
{
    private readonly double _topTrialsRatio;
    private readonly int _minNTrials;
    private readonly double _minimumNoise;
    private readonly int _optimizeNSamples;
    private readonly LazyRandomState _rng;

    /// <param name="topTrialsRatio">A ratio of top trials to be considered when estimating the regret.</param>
    /// <param name="minNTrials">A minimum number of complete trials to estimate the regret.</param>
    /// <param name="seed">Seed for random number generator.</param>
    /// <summary>Configures the best-trial subset and GP settings used to bound remaining regret.</summary>
    public RegretBoundEvaluator(
        double topTrialsRatio = ImprovementEvaluatorDefaults.DefaultTopTrialsRatio,
        int minNTrials = ImprovementEvaluatorDefaults.DefaultMinNTrials,
        int? seed = null)
    {
        _topTrialsRatio = topTrialsRatio;
        _minNTrials = minNTrials;
        _minimumNoise = Prior.DefaultMinimumNoiseVar;
        _optimizeNSamples = 2048;
        _rng = new LazyRandomState(seed);
    }

    /// <summary>
    ///     <c>_get_top_n</c> — select the top-<c>n</c> trials by value. <c>top_n =
    ///     clip(int(n_trials·ratio), min_n_trials, n_trials)</c>; the threshold is the
    ///     <c>(n_trials − top_n)</c>-th order statistic (<c>np.partition</c>), and the mask keeps every
    ///     value at or above it.
    /// </summary>
    /// <param name="normalizedParams">Normalized coordinates in search-space order.</param>
    /// <param name="values">The ordered values to aggregate or format.</param>
    /// <returns>The parameter rows and objective values retained as the best observations for fitting the regret model.</returns>
    private (double[,] Params, double[] Values) GetTopN(double[,] normalizedParams, double[] values)
    {
        int nTrials = values.Length;
        int topN = Math.Min(Math.Max((int)(nTrials * _topTrialsRatio), _minNTrials), nTrials);

        // np.partition(values, kth)[kth] — the kth order statistic (0-indexed, ascending).
        int kth = nTrials - topN;
        var sorted = (double[])values.Clone();
        Array.Sort(sorted);
        double topNVal = sorted[kth];

        int dim = normalizedParams.GetLength(1);
        var idxs = new List<int>();
        for (int i = 0; i < nTrials; i++)
            if (values[i] >= topNVal)
                idxs.Add(i);

        var paramsOut = new double[idxs.Count, dim];
        var valuesOut = new double[idxs.Count];
        for (int r = 0; r < idxs.Count; r++)
        {
            valuesOut[r] = values[idxs[r]];
            for (int d = 0; d < dim; d++)
                paramsOut[r, d] = normalizedParams[idxs[r], d];
        }
        return (paramsOut, valuesOut);
    }

    /// <summary>Returns the estimated upper confidence bound on remaining regret, in objective-value units.</summary>
    /// <param name="trials">Trial snapshots to inspect; their order is preserved unless the operation explicitly ranks them.</param>
    /// <param name="studyDirection">Whether larger or smaller objective values are preferred.</param>
    /// <returns>The estimated upper confidence bound on remaining regret, in objective-value units.</returns>
    /// <exception cref="NumSharp.ValueError">There are no completed trials, or their common search space is empty.</exception>
    public override double Evaluate(IReadOnlyList<FrozenTrial> trials, StudyDirection studyDirection)
    {
        var optunaSearchSpace = SearchSpaceFunctions.IntersectionSearchSpaceOf(trials);
        ValidateInput(trials, optunaSearchSpace);

        var completeTrials = trials.Where(t => t.State == TrialState.Complete).ToList();

        // _gp module assumes that optimization direction is maximization.
        double sign = studyDirection == StudyDirection.Minimize ? -1.0 : 1.0;
        var values = new double[completeTrials.Count];
        for (int i = 0; i < completeTrials.Count; i++)
            values[i] = completeTrials[i].Value!.Value * sign;

        var searchSpace = new GpSearchSpace(optunaSearchSpace);
        double[,] normalizedParams = searchSpace.GetNormalizedParams(
            completeTrials.Select(t => t.Params).ToList());
        (double[,] normalizedTopNParams, double[] topNValues) = GetTopN(normalizedParams, values);

        double topNValuesMean = MathUtil.Mean(topNValues);
        double topNValuesStd = Math.Max(1e-10, MathUtil.Std(topNValues));
        var standardizedTopNValues = new double[topNValues.Length];
        for (int i = 0; i < topNValues.Length; i++)
            standardizedTopNValues[i] = (topNValues[i] - topNValuesMean) / topNValuesStd;

        GPRegressor gpr = GPRegressor.FitKernelParams(
            normalizedTopNParams, standardizedTopNValues, searchSpace.IsCategorical,
            _minimumNoise, deterministicObjective: false, gprCache: null);

        // Optuna's evaluate() leaves self._optimize_n_samples unused and relies on the default 2048;
        // since it is hardcoded to 2048 in __init__ (no constructor parameter exposes it), threading
        // the field through is behaviorally identical and keeps the field meaningful.
        double standardizedRegretBound = ComputeStandardizedRegretBound(
            gpr, searchSpace, normalizedTopNParams, standardizedTopNValues,
            optimizeNSamples: _optimizeNSamples, rng: _rng.Rng);
        return standardizedRegretBound * topNValuesStd;  // regret bound
    }

    /// <summary>Checks the completed-trial and common-search-space prerequisites of the regret model.</summary>
    /// <param name="trials">Trial snapshots to inspect; their order is preserved unless the operation explicitly ranks them.</param>
    /// <param name="searchSpace">The parameter distributions in the column order used by the transform or model.</param>
    /// <exception cref="NumSharp.ValueError">There are no completed trials, or their common search space is empty.</exception>
    private static void ValidateInput(
        IReadOnlyList<FrozenTrial> trials, OrderedDict<string, BaseDistribution> searchSpace)
    {
        if (!trials.Any(t => t.State == TrialState.Complete))
            throw new ValueError(
                "Because no trial has been completed yet, the regret bound cannot be evaluated.");

        if (searchSpace.Count == 0)
            throw new ValueError(
                "The intersection search space is empty. This condition is not supported by "
                + $"{nameof(RegretBoundEvaluator)}.");
    }

    /// <summary>
    ///     <c>_get_beta</c> — the confidence-width coefficient (Srinivas et al. 2010, scaled down by 5
    ///     as in the paper's experiments).
    /// </summary>
    /// <param name="nParams">The number of modeled parameters.</param>
    /// <param name="nTrials">The number of observed trials used in the confidence-width calculation.</param>
    /// <param name="delta">The probability of exceeding the confidence bound.</param>
    /// <returns>The confidence-width coefficient computed from parameter count, trial count, and confidence level.</returns>
    internal static double GetBeta(int nParams, int nTrials, double delta = 0.1)
    {
        // beta = 2 * log(n_params * n_trials**2 * pi**2 / 6 / delta), then /= 5.
        double beta = 2.0 * Math.Log(
            (double)((long)nParams * nTrials * nTrials) * Math.Pow(Math.PI, 2.0) / 6.0 / delta);
        beta /= 5.0;
        return beta;
    }

    /// <summary>
    ///     <c>_compute_standardized_regret_bound</c> — <c>max(UCB) − max(LCB)</c> in standardized space.
    ///     The <c>_gp</c> module maximizes, so (vs the paper): max(UCB) over the search space minus
    ///     max(LCB) over the top trials. UCB over the search space is the larger of the UCB at the top
    ///     trials and the UCB found by random search sampling (<see cref="OptimSample" />).
    /// </summary>
    /// <param name="gpr">The Gaussian-process model supplying posterior predictions and kernel length scales.</param>
    /// <param name="searchSpace">The normalized coordinate layout, bounds, and discrete choices used by the acquisition optimizer.</param>
    /// <param name="normalizedTopNParams">The normalized parameter rows retained as the best observations for the regret bound.</param>
    /// <param name="standardizedTopNValues">The standardized objective values corresponding to the retained parameter rows.</param>
    /// <param name="delta">The probability of exceeding the confidence bound.</param>
    /// <param name="optimizeNSamples">The number of candidate samples used when optimizing the regret-bound acquisition functions.</param>
    /// <param name="rng">The random generator whose state is advanced when drawing samples.</param>
    /// <returns>The upper confidence bound on regret in standardized objective units.</returns>
    internal static double ComputeStandardizedRegretBound(
        GPRegressor gpr, GpSearchSpace searchSpace, double[,] normalizedTopNParams,
        double[] standardizedTopNValues, double delta = 0.1, int optimizeNSamples = 2048,
        NumPyRandom? rng = null)
    {
        int nTrials = normalizedTopNParams.GetLength(0);
        int nParams = normalizedTopNParams.GetLength(1);

        double beta = GetBeta(nParams, nTrials, delta);

        var ucbAcqf = new UCB(gpr, searchSpace, beta);
        // UCB over the search space. (Original: LCB over the search space.)
        double ucbOverTopN = Max(ucbAcqf.EvalAcqfNoGrad(normalizedTopNParams));
        double sampledUcb = OptimSample.OptimizeAcqfSample(ucbAcqf, optimizeNSamples, rng!).Value;
        double standardizedUcbValue = Math.Max(ucbOverTopN, sampledUcb);

        var lcbAcqf = new LCB(gpr, searchSpace, beta);
        // LCB over the top trials. (Original: UCB over the top trials.)
        double standardizedLcbValue = Max(lcbAcqf.EvalAcqfNoGrad(normalizedTopNParams));

        // max(UCB) - max(LCB). (Original: min(UCB) - min(LCB).)
        return standardizedUcbValue - standardizedLcbValue;  // standardized regret bound
    }

    /// <summary>Returns the largest value in the supplied sequence.</summary>
    /// <param name="a">The first numerical operand; its dimensions must agree with the operation described in the summary.</param>
    /// <returns>The largest value in the supplied sequence.</returns>
    private static double Max(double[] a)
    {
        double m = a[0];
        for (int i = 1; i < a.Length; i++)
            if (a[i] > m)
                m = a[i];
        return m;
    }
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
    /// <summary>Sets the allowed count of trials without a new best objective value.</summary>
    /// <exception cref="NumSharp.ValueError">The permitted stagnation count is negative.</exception>
    public BestValueStagnationEvaluator(int maxStagnationTrials = 30)
    {
        if (maxStagnationTrials < 0)
            throw new ValueError("The maximum number of stagnant trials must not be negative.");

        _maxStagnationTrials = maxStagnationTrials;
    }

    /// <summary>Returns the remaining allowed stagnation count before the configured threshold is exceeded.</summary>
    /// <param name="trials">Trial snapshots to inspect; their order is preserved unless the operation explicitly ranks them.</param>
    /// <param name="studyDirection">Whether larger or smaller objective values are preferred.</param>
    /// <returns>The remaining allowed stagnation count before the configured threshold is exceeded.</returns>
    /// <exception cref="NumSharp.ValueError">The trial list contains no completed trial.</exception>
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

    /// <summary>Ensures that the evaluator has completed objective values from which to measure stagnation.</summary>
    /// <param name="trials">Trial snapshots to inspect; their order is preserved unless the operation explicitly ranks them.</param>
    /// <exception cref="NumSharp.ValueError">The trial list contains no completed trial.</exception>
    private static void ValidateInput(IReadOnlyList<FrozenTrial> trials)
    {
        if (!trials.Any(t => t.State == TrialState.Complete))
            throw new ValueError(
                "Because no trial has been completed yet, the improvement cannot be evaluated.");
    }
}
