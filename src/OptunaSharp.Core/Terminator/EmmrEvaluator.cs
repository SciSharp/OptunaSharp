using System;
using System.Collections.Generic;
using System.Linq;
using NumSharp;
using OptunaSharp.Gp;
using OptunaSharp.Internal;
using OptunaSharp.Samplers;
using OptunaSharp.SearchSpace;

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
///     Rides the native GP stack (two kernel fits + posteriors), so its output is tolerance-exact vs
///     Optuna, not bit-for-bit — the documented GP boundary. The <c>scipy.stats.norm.cdf</c> term
///     additionally rides <see cref="Erf" /> (the erf/ucrt tolerance class).
///
///     For further information about expected minimum simple regret and the algorithm, please
///     refer to the following paper:
///     - A stopping criterion for Bayesian optimization by the gap of expected minimum simple
///       regrets (https://proceedings.mlr.press/v206/ishibashi23a.html).
/// </summary>
public class EMMREvaluator : BaseImprovementEvaluator
{
    /// <summary>Python <c>MARGIN_FOR_NUMARICAL_STABILITY</c>.</summary>
    private const double MarginForNumericalStability = 0.1;

    // Python `sys.float_info.min` — the smallest positive *normalized* double (2**-1022).
    private const double FloatInfoMin = 2.2250738585072014e-308;

    // Python `sys.float_info.max` — the largest finite double (== double.MaxValue).
    private const double FloatInfoMax = double.MaxValue;

    private static readonly double NormPdfC = Math.Sqrt(2.0 * Math.PI);

    private readonly bool _deterministic;
    private readonly double _delta;
    private readonly LazyRandomState _rng;

    /// <summary>Python <c>self.min_n_trials</c> (a public attribute).</summary>
    public int MinNTrials;

    /// <param name="deterministicObjective">Whether the objective function is deterministic.</param>
    /// <param name="delta">A float number related to the criterion for termination.</param>
    /// <param name="minNTrials">A minimum number of complete trials to compute the criterion.</param>
    /// <param name="seed">A random seed for EMMREvaluator.</param>
    /// <summary>Configures the minimum data requirement and deterministic/noisy GP assumptions for EMMR evaluation.</summary>
    /// <exception cref="NumSharp.ValueError">The minimum trial count is less than two.</exception>
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

    /// <summary>Returns the expected minimum model regret estimate, or a conservative large value before enough trials exist.</summary>
    /// <param name="trials">Trial snapshots to inspect; their order is preserved unless the operation explicitly ranks them.</param>
    /// <param name="studyDirection">Whether larger or smaller objective values are preferred.</param>
    /// <returns>The expected minimum model regret estimate, or a conservative large value before enough trials exist.</returns>
    public override double Evaluate(IReadOnlyList<FrozenTrial> trials, StudyDirection studyDirection)
    {
        var optunaSearchSpace = SearchSpaceFunctions.IntersectionSearchSpaceOf(trials);
        var completeTrials = trials.Where(t => t.State == TrialState.Complete).ToList();

        if (completeTrials.Count < MinNTrials)
            return FloatInfoMax * MarginForNumericalStability;  // Do not terminate.

        var searchSpace = new GpSearchSpace(optunaSearchSpace);
        double[,] normalizedParams = searchSpace.GetNormalizedParams(
            completeTrials.Select(t => (IReadOnlyDictionary<string, object?>)t.ParamsDict).ToList());
        if (searchSpace.Dim == 0)
        {
            OptunaWarnings.Warn(
                $"{nameof(EMMREvaluator)} cannot consider any search space."
                + "Termination will never occur in this study.",
                WarningCategory.UserWarning);
            return FloatInfoMax * MarginForNumericalStability;  // Do not terminate.
        }

        int lenTrials = completeTrials.Count;

        // _gp module assumes that optimization direction is maximization.
        double sign = studyDirection == StudyDirection.Minimize ? -1.0 : 1.0;
        var scoreVals = new double[lenTrials];
        for (int i = 0; i < lenTrials; i++)
            scoreVals[i] = completeTrials[i].Value!.Value * sign;
        scoreVals = WarnAndConvertInf(scoreVals);
        double scoreMean = MathUtil.Mean(scoreVals);
        double scoreStd = Math.Max(FloatInfoMin, MathUtil.Std(scoreVals));
        var standardizedScoreVals = new double[lenTrials];
        for (int i = 0; i < lenTrials; i++)
            standardizedScoreVals[i] = (scoreVals[i] - scoreMean) / scoreStd;

        double[,] normalizedParamsHead = HeadRows(normalizedParams, lenTrials - 1);
        double[] standardizedHead = standardizedScoreVals[..(lenTrials - 1)];

        GPRegressor gprT1 = GPRegressor.FitKernelParams(  // Fit kernel with up to (t-1)-th observation
            normalizedParamsHead, standardizedHead, searchSpace.IsCategorical,
            Prior.DefaultMinimumNoiseVar, deterministicObjective: _deterministic, gprCache: null);

        GPRegressor gprT = GPRegressor.FitKernelParams(  // Fit kernel with up to t-th observation
            normalizedParams, standardizedScoreVals, searchSpace.IsCategorical,
            Prior.DefaultMinimumNoiseVar, deterministicObjective: _deterministic, gprCache: gprT1);

        int thetaTStarIndex = ArgMax(standardizedScoreVals, lenTrials);
        int thetaT1StarIndex = ArgMax(standardizedScoreVals, lenTrials - 1);
        double[] thetaTStar = Row(normalizedParams, thetaTStarIndex);
        double[] thetaT1Star = Row(normalizedParams, thetaT1StarIndex);
        double covTBetweenThetaTStarAndThetaT1Star = ComputeGpPosteriorCovTwoThetas(
            normalizedParams, gprT, thetaTStarIndex, thetaT1StarIndex);

        // Use gpr_t instead of gpr_t1 because KL Div. requires the same prior for both posterior.
        // cf. Sec. 4.4 of https://proceedings.mlr.press/v206/ishibashi23a/ishibashi23a.pdf
        (double muT1ThetaTWithNuT, double varianceT1ThetaTWithNuT) =
            ComputeGpPosterior(Row(normalizedParams, lenTrials - 1), gprT);
        (_, double varianceTThetaT1Star) = ComputeGpPosterior(thetaT1Star, gprT);
        (double muTThetaTStar, double varianceTThetaTStar) = ComputeGpPosterior(thetaTStar, gprT);
        (double muT1ThetaT1Star, _) = ComputeGpPosterior(thetaT1Star, gprT1);

        double yT = standardizedScoreVals[lenTrials - 1];
        double kappaT1 = RegretBoundEvaluator.ComputeStandardizedRegretBound(
            gprT1, searchSpace, normalizedParamsHead, standardizedHead, _delta, rng: _rng.Rng);

        double theorem1DeltaMuTStar = muT1ThetaT1Star - muTThetaTStar;

        double alg1DeltaRTildeTTerm1 = theorem1DeltaMuTStar;

        double theorem1V = Math.Sqrt(
            Math.Max(
                1e-10,
                varianceTThetaTStar
                - 2.0 * covTBetweenThetaTStarAndThetaT1Star
                + varianceTThetaT1Star));
        double theorem1G = (muTThetaTStar - muT1ThetaT1Star) / theorem1V;

        double alg1DeltaRTildeTTerm2 = theorem1V * NormPdf(theorem1G);
        double alg1DeltaRTildeTTerm3 = theorem1V * theorem1G * Erf.NormCdf(theorem1G);

        double lambda = Math.Pow(Prior.DefaultMinimumNoiseVar, -1.0);
        double lambdaInv = Math.Pow(lambda, -1.0);
        double eq4RhsTerm1 = 0.5 * Math.Log(1.0 + lambda * varianceT1ThetaTWithNuT);
        double eq4RhsTerm2 =
            -0.5 * varianceT1ThetaTWithNuT / (varianceT1ThetaTWithNuT + lambdaInv);
        double eq4RhsTerm3 =
            0.5
            * varianceT1ThetaTWithNuT
            * Math.Pow(yT - muT1ThetaTWithNuT, 2.0)
            / Math.Pow(varianceT1ThetaTWithNuT + lambdaInv, 2.0);

        double alg1DeltaRTildeTTerm4 =
            kappaT1 * Math.Sqrt(0.5 * (eq4RhsTerm1 + eq4RhsTerm2 + eq4RhsTerm3));

        return Math.Min(
            FloatInfoMax * 0.5,
            alg1DeltaRTildeTTerm1
            + alg1DeltaRTildeTTerm2
            + alg1DeltaRTildeTTerm3
            + alg1DeltaRTildeTTerm4);
    }

    /// <summary><c>_compute_gp_posterior</c> — the (scalar) posterior mean and variance at one point.</summary>
    /// <param name="xParams">The normalized coordinates at which to evaluate the GP posterior.</param>
    /// <param name="gpr">The Gaussian-process model supplying posterior predictions and kernel length scales.</param>
    /// <returns>The model's mean and variance at the selected input row.</returns>
    private static (double Mean, double Var) ComputeGpPosterior(double[] xParams, GPRegressor gpr)
    {
        var row = new double[1, xParams.Length];
        for (int d = 0; d < xParams.Length; d++)
            row[0, d] = xParams[d];
        (double[] mean, double[] var) = gpr.Posterior(row);
        return (mean[0], var[0]);
    }

    /// <summary>
    ///     <c>_compute_gp_posterior_cov_two_thetas</c> — <c>Cov(f(θ₁), f(θ₂))</c> under the GP posterior
    ///     (the variance when <paramref name="theta1Index" /> == <paramref name="theta2Index" />).
    /// </summary>
    /// <param name="normalizedParams">Normalized coordinates in search-space order.</param>
    /// <param name="gpr">The Gaussian-process model supplying posterior predictions and kernel length scales.</param>
    /// <param name="theta1Index">The first trial-row index whose posterior covariance is requested.</param>
    /// <param name="theta2Index">The second trial-row index whose posterior covariance is requested.</param>
    /// <returns>The joint posterior covariance between the two selected input rows.</returns>
    private static double ComputeGpPosteriorCovTwoThetas(
        double[,] normalizedParams, GPRegressor gpr, int theta1Index, int theta2Index)
    {
        if (theta1Index == theta2Index)
            return ComputeGpPosterior(Row(normalizedParams, theta1Index), gpr).Var;

        int dim = normalizedParams.GetLength(1);
        var pair = new double[2, dim];
        for (int d = 0; d < dim; d++)
        {
            pair[0, d] = normalizedParams[theta1Index, d];
            pair[1, d] = normalizedParams[theta2Index, d];
        }
        (_, double[,] covar) = gpr.PosteriorJoint(pair);
        return covar[0, 1];
    }

    /// <summary>scipy's <c>norm.pdf</c> — the standard-normal density <c>exp(−x²/2)/√(2π)</c>.</summary>
    /// <param name="x">The scalar argument.</param>
    /// <returns>The standard-normal probability density at the supplied argument.</returns>
    private static double NormPdf(double x) => Math.Exp(-Math.Pow(x, 2.0) / 2.0) / NormPdfC;

    /// <summary><c>np.argmax</c> over the first <paramref name="count" /> entries (first maximum).</summary>
    /// <param name="a">The first numerical operand; its dimensions must agree with the operation described in the summary.</param>
    /// <param name="count">The number of leading values to include in the search.</param>
    /// <returns>The index of the first largest value.</returns>
    private static int ArgMax(double[] a, int count)
    {
        int best = 0;
        for (int i = 1; i < count; i++)
            if (a[i] > a[best])
                best = i;
        return best;
    }

    /// <summary>Returns a copy of the requested matrix row.</summary>
    /// <param name="mat">The matrix whose rows are copied.</param>
    /// <param name="r">The zero-based matrix row index.</param>
    /// <returns>A copy of the requested matrix row.</returns>
    private static double[] Row(double[,] mat, int r)
    {
        int dim = mat.GetLength(1);
        var row = new double[dim];
        for (int d = 0; d < dim; d++)
            row[d] = mat[r, d];
        return row;
    }

    /// <summary>Returns a new matrix containing the requested leading rows.</summary>
    /// <param name="mat">The matrix whose rows are copied.</param>
    /// <param name="rows">The number of matrix rows to allocate or copy.</param>
    /// <returns>A new matrix containing the requested leading rows.</returns>
    private static double[,] HeadRows(double[,] mat, int rows)
    {
        int dim = mat.GetLength(1);
        var head = new double[rows, dim];
        for (int r = 0; r < rows; r++)
            for (int d = 0; d < dim; d++)
                head[r, d] = mat[r, d];
        return head;
    }

    /// <summary>
    ///     <c>gp.warn_and_convert_inf</c> for a 1-D score vector — clip each non-finite entry to the
    ///     finite min/max (or 0 when the whole vector is non-finite). Fast-paths the all-finite case.
    /// </summary>
    /// <param name="values">The ordered values to aggregate or format.</param>
    /// <returns>A copy with infinite values replaced by finite extrema after emitting the diagnostic.</returns>
    private static double[] WarnAndConvertInf(double[] values)
    {
        bool allFinite = values.All(double.IsFinite);
        if (allFinite)
            return values;

        OptunaWarnings.Warn(
            "Clip non-finite values to the min/max finite values for GP fittings.",
            WarningCategory.UserWarning);

        double lo = double.PositiveInfinity, hi = double.NegativeInfinity;
        bool anyFinite = false;
        foreach (double v in values)
            if (double.IsFinite(v))
            {
                anyFinite = true;
                lo = Math.Min(lo, v);
                hi = Math.Max(hi, v);
            }
        if (!anyFinite) { lo = 0.0; hi = 0.0; }

        var result = new double[values.Length];
        for (int i = 0; i < values.Length; i++)
            result[i] = Math.Min(Math.Max(values[i], lo), hi);
        return result;
    }
}
