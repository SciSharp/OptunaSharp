using System;
using System.Collections.Generic;
using OptunaSharp.Internal;

namespace OptunaSharp.Gp;

/// <summary>
///     GP acquisition-function primitives — port of the log-expected-improvement helpers in
///     <c>optuna/_gp/acqf.py</c> (<c>standard_logei</c> / <c>logei</c>). LogEI is the acquisition
///     function <see cref="OptunaSharp.Samplers.GPSampler" /> maximizes for single-objective optimization; these are the
///     numerically stable log forms from "Unexpected Improvements to Expected Improvement"
///     (arxiv 2310.20708). They ride <see cref="Erf" /> (erfc/erfcx) → the erf/ucrt tolerance class.
/// </summary>
internal static class Acqf
{
    private static readonly double SqrtHalf = Math.Sqrt(0.5);
    private static readonly double InvSqrt2Pi = 1.0 / Math.Sqrt(2.0 * Math.PI);
    private static readonly double SqrtHalfPi = Math.Sqrt(0.5 * Math.PI);
    private static readonly double LogSqrt2Pi = Math.Log(Math.Sqrt(2.0 * Math.PI));
    private static readonly double SqrtPi = Math.Sqrt(Math.PI);

    /// <summary>
    ///     <c>standard_logei(z) = log E_{x~N(0,1)}[max(0, x + z)]</c>. The stable log-EI: the primary
    ///     branch is <c>log(z·Φ(z) + φ(z))</c>; for <c>z &lt; -25</c> the <c>erfcx</c> form avoids the
    ///     <c>0/0</c> as the improvement vanishes (Eq. 9 of arxiv 2310.20708).
    /// </summary>
    /// <param name="z">The standardized normal argument.</param>
    /// <returns>The log expected positive improvement for the standardized normal argument.</returns>
    internal static double StandardLogEi(double z)
    {
        if (z < -25.0)
        {
            double w = -SqrtHalf * z;
            return -0.5 * z * z - LogSqrt2Pi + Math.Log(1.0 + SqrtHalfPi * z * Erf.Erfcx(w));
        }
        double zHalf = 0.5 * z;
        return Math.Log(zHalf * Erf.Erfc(-SqrtHalf * z) + Math.Exp(-zHalf * z) * InvSqrt2Pi);
    }

    /// <summary>
    ///     <see cref="StandardLogEi" /> and its derivative <c>d/dz</c> — the analytic replacement for
    ///     autograd through <c>standard_logei</c>. In the primary branch the derivative collapses to
    ///     <c>Φ(z)/EI(z)</c> (the sufficient-decrease cancellation); the <c>z &lt; -25</c> branch
    ///     differentiates the <c>erfcx</c> form directly.
    /// </summary>
    /// <param name="z">The standardized normal argument.</param>
    /// <returns>The log expected improvement and its derivative with respect to the standardized argument.</returns>
    internal static (double Value, double Grad) StandardLogEiAndGrad(double z)
    {
        if (z < -25.0)
        {
            double w = -SqrtHalf * z;
            double ecx = Erf.Erfcx(w);
            double g = 1.0 + SqrtHalfPi * z * ecx;
            double value = -0.5 * z * z - LogSqrt2Pi + Math.Log(g);
            // g'(z) = √(π/2)·[(1 + 2w²)·erfcx(w) − 2w/√π], with w = −√½·z.
            double dg = SqrtHalfPi * ((1.0 + 2.0 * w * w) * ecx - 2.0 * w / SqrtPi);
            return (value, -z + dg / g);
        }
        double zHalf = 0.5 * z;
        double erfcNeg = Erf.Erfc(-SqrtHalf * z);                        // 2·Φ(z)
        double denom = zHalf * erfcNeg + Math.Exp(-zHalf * z) * InvSqrt2Pi;  // EI(z)
        return (Math.Log(denom), 0.5 * erfcNeg / denom);                // Φ(z)/EI(z)
    }

    /// <summary><c>logei(mean, var, f0) = standard_logei((mean − f0)/σ) + log σ</c>, σ = √var.</summary>
    /// <param name="mean">The predictive mean in the standardized objective coordinate system.</param>
    /// <param name="var">The nonnegative predictive variance.</param>
    /// <param name="f0">The incumbent objective threshold in the maximization convention.</param>
    /// <returns>The logarithm of expected improvement above the incumbent threshold.</returns>
    internal static double LogEi(double mean, double var, double f0)
    {
        double sigma = Math.Sqrt(var);
        return StandardLogEi((mean - f0) / sigma) + Math.Log(sigma);
    }

    /// <summary>
    ///     <c>_compute_mean_max_utility(log_util_vals)</c> — the q-batch reduction: the max over the
    ///     q-batch axis (<c>amax</c>, dim=-1) then the log-mean-exp over the fixed-sample axis
    ///     (<c>logsumexp − log(n_qmc)</c>). <paramref name="logUtil" /> is <c>(n_qmc, q_batch)</c>; the
    ///     result is scalar.
    /// </summary>
    /// <param name="logUtil">Per-sample log utilities with shape (QMC samples, batch points).</param>
    /// <returns>The log mean of the per-sample maximum utility across batch points.</returns>
    internal static double MeanMaxUtility(double[,] logUtil)
    {
        int nQmc = logUtil.GetLength(0);
        int q = logUtil.GetLength(1);
        var maxVal = new double[nQmc];
        double lseMax = double.NegativeInfinity;
        for (int s = 0; s < nQmc; s++)
        {
            double m = double.NegativeInfinity;
            for (int j = 0; j < q; j++)
                if (logUtil[s, j] > m) m = logUtil[s, j];
            maxVal[s] = m;
            if (m > lseMax) lseMax = m;
        }
        double sum = 0.0;
        for (int s = 0; s < nQmc; s++)
            sum += Math.Exp(maxVal[s] - lseMax);
        return lseMax + Math.Log(sum) - Math.Log(nQmc);
    }

    /// <summary>
    ///     <see cref="MeanMaxUtility" /> plus its gradient w.r.t. the query point. The q-batch acqfs share
    ///     the invariant that <b>only the last (query) column</b> of <paramref name="logUtil" /> depends
    ///     on <c>x</c> (the running-fantasy columns are constant), with derivative
    ///     <paramref name="dQuery" /> (<c>n_qmc × dim</c>). The gradient flows through the <c>amax</c>
    ///     selection (per-sample argmax) and the <c>logsumexp</c> softmax — nonzero only where the query
    ///     column wins that sample's max.
    /// </summary>
    /// <param name="logUtil">Per-sample log utilities with shape (QMC samples, batch points).</param>
    /// <param name="dQuery">The derivative of the query-point log utility with shape (QMC samples, dimensions).</param>
    /// <returns>The log mean maximum utility and its gradient with respect to the query coordinates.</returns>
    internal static (double Value, double[] Grad) MeanMaxUtilityWithGrad(double[,] logUtil, double[,] dQuery)
    {
        int nQmc = logUtil.GetLength(0);
        int q = logUtil.GetLength(1);
        int queryCol = q - 1;
        int dim = dQuery.GetLength(1);

        var maxVal = new double[nQmc];
        var argmax = new int[nQmc];
        double lseMax = double.NegativeInfinity;
        for (int s = 0; s < nQmc; s++)
        {
            double m = double.NegativeInfinity;
            int am = 0;
            for (int j = 0; j < q; j++)
                if (logUtil[s, j] > m) { m = logUtil[s, j]; am = j; }
            maxVal[s] = m; argmax[s] = am;
            if (m > lseMax) lseMax = m;
        }
        double expSum = 0.0;
        for (int s = 0; s < nQmc; s++)
            expSum += Math.Exp(maxVal[s] - lseMax);
        double value = lseMax + Math.Log(expSum) - Math.Log(nQmc);

        var grad = new double[dim];
        for (int s = 0; s < nQmc; s++)
        {
            if (argmax[s] != queryCol)
                continue;   // gradient flows only when the query column is this sample's max
            double softmax = Math.Exp(maxVal[s] - lseMax) / expSum;
            for (int d = 0; d < dim; d++)
                grad[d] += softmax * dQuery[s, d];
        }
        return (value, grad);
    }

    /// <summary>
    ///     <c>torch.special.logsumexp</c> over a vector — <c>m + log Σ exp(v − m)</c> with <c>m = max v</c>,
    ///     so large terms cannot overflow. An all-<c>−∞</c> (or empty) input returns <c>−∞</c> instead of
    ///     the <c>NaN</c> the naive shift would produce, matching torch.
    /// </summary>
    /// <param name="vals">The terms to combine; NaN entries propagate to the result.</param>
    /// <returns>The logarithm of the sum of the exponentials of <paramref name="vals" />.</returns>
    internal static double LogSumExp(double[] vals)
    {
        double m = double.NegativeInfinity;
        foreach (double v in vals) if (v > m) m = v;
        if (double.IsNegativeInfinity(m)) return m;
        double sum = 0.0;
        foreach (double v in vals) sum += Math.Exp(v - m);
        return m + Math.Log(sum);
    }

    /// <summary>
    ///     The log of the mean of exponentials, <c>logsumexp(v) − log(len(v))</c> — the Monte-Carlo
    ///     expectation over QMC draws that the q-batch acquisition functions take in log space.
    /// </summary>
    /// <param name="vals">The per-draw log utilities; must be non-empty for the mean to be defined.</param>
    /// <returns>The logarithm of the arithmetic mean of the exponentials of <paramref name="vals" />.</returns>
    internal static double LogMeanExp(double[] vals) => LogSumExp(vals) - Math.Log(vals.Length);
}

/// <summary>
///     Per-QMC-draw Pareto baselines and the log hypervolume improvement over them — port of
///     <c>acqf._PerSampleParetoBaseline</c> (Optuna 5.0.0, #6804). For every QMC draw it unions the
///     baseline observations with that draw's fantasy observations (which a caller may have filtered
///     beforehand — <see cref="QLogCEHVI" /> keeps only fantasies sampled as feasible), decomposes the
///     non-dominated region into boxes, and scores a candidate's per-draw log HVI against those boxes.
///     Shared by <see cref="QLogEHVI" /> and <see cref="QLogCEHVI" />.
/// </summary>
/// <remarks>
///     <para><b>Padding is part of the value.</b> Optuna stacks the per-draw box lists with
///     <c>pad_sequence</c>: missing boxes get lower bound <c>0</c> and width <c>EPS</c>. A padded box
///     clamps every coordinate to exactly <c>EPS</c>, so each one adds the constant
///     <c>EPS^n_objectives</c> to that draw's HVI sum (and nothing to its gradient). The padding is
///     reproduced here rather than skipped, because draws with very different box counts — routine once
///     fantasies are feasibility-filtered — would otherwise disagree with Optuna in low-HVI regions.</para>
///     <para>The reference point comes from the baseline observations alone, exactly as in Optuna, so a
///     fantasy worse than the reference point simply drops out of that draw's decomposition.</para>
/// </remarks>
internal sealed class PerSampleParetoBaseline
{
    private const double Eps = 1e-12;   // _EPS in acqf.py: keeps log() and the gradient finite

    private readonly int _nObjectives;
    private readonly double[][,] _lowerBounds;   // per QMC draw: (n_boxes_max, n_obj), padded with 0
    private readonly double[][,] _intervals;     // per QMC draw: (n_boxes_max, n_obj), clamped ≥ EPS, padded with EPS

    /// <summary>Decomposes, for every QMC draw, the region dominated by the baseline plus that draw's fantasies.</summary>
    /// <param name="yBaseline">The baseline observations, one row per trial and one column per objective, in the maximization convention.</param>
    /// <param name="fantasySamplesPerQmc">
    ///     One fantasy matrix per QMC draw, each <c>(n_fantasies_for_draw, n_objectives)</c>; a draw may have
    ///     zero rows, in which case its baseline is <paramref name="yBaseline" /> alone.
    /// </param>
    internal PerSampleParetoBaseline(double[,] yBaseline, IReadOnlyList<double[,]> fantasySamplesPerQmc)
    {
        _nObjectives = yBaseline.GetLength(1);
        double[] refPoint = LogEHVI.GetReferencePoint(yBaseline);
        int nQmc = fantasySamplesPerQmc.Count;

        var lowers = new double[nQmc][,];
        var uppers = new double[nQmc][,];
        int nBoxesMax = 0;
        for (int s = 0; s < nQmc; s++)
        {
            double[,] fantasy = fantasySamplesPerQmc[s];
            double[,] y = fantasy.GetLength(0) > 0 ? ConcatRows(yBaseline, fantasy) : yBaseline;
            (lowers[s], uppers[s]) = LogEHVI.GetNonDominatedBoxBoundsMax(y, refPoint);
            nBoxesMax = Math.Max(nBoxesMax, lowers[s].GetLength(0));
        }

        // pad_sequence(batch_first=True): every draw gets n_boxes_max rows; the padding rows keep the
        // default lower bound 0.0 and the interval padding_value _EPS (see the remarks for why it matters).
        _lowerBounds = new double[nQmc][,];
        _intervals = new double[nQmc][,];
        for (int s = 0; s < nQmc; s++)
        {
            int nBoxes = lowers[s].GetLength(0);
            var lower = new double[nBoxesMax, _nObjectives];
            var interval = new double[nBoxesMax, _nObjectives];
            for (int b = 0; b < nBoxesMax; b++)
                for (int o = 0; o < _nObjectives; o++)
                {
                    if (b < nBoxes)
                    {
                        lower[b, o] = lowers[s][b, o];
                        interval[b, o] = Math.Max(Eps, uppers[s][b, o] - lowers[s][b, o]);
                    }
                    else
                    {
                        interval[b, o] = Eps;
                    }
                }
            _lowerBounds[s] = lower;
            _intervals[s] = interval;
        }
    }

    /// <summary>The number of QMC draws this baseline was built for.</summary>
    internal int NQmc => _lowerBounds.Length;

    /// <summary>
    ///     <c>compute_per_sample_log_hvi(Y_post)</c> — for each QMC draw <c>s</c>, the log HVI
    ///     <c>logsumexp_b Σ_o log(clamp(Y_post[s,o] − lower[s,b,o], EPS, interval[s,b,o]))</c>.
    /// </summary>
    /// <param name="yPost">The candidate's per-draw objective samples, shape <c>(n_qmc, n_objectives)</c>.</param>
    /// <returns>The per-draw log hypervolume improvement, shape <c>(n_qmc,)</c>.</returns>
    internal double[] ComputePerSampleLogHvi(double[,] yPost) => PerSampleLogHvi(yPost, out _);

    /// <summary>
    ///     <see cref="ComputePerSampleLogHvi" /> plus its gradient with respect to the candidate point —
    ///     the analytic replacement for autograd: a softmax over each draw's boxes times the two-sided
    ///     clamp mask times <c>1/raw</c> times the objective samples' own gradient.
    /// </summary>
    /// <param name="yPost">The candidate's per-draw objective samples, shape <c>(n_qmc, n_objectives)</c>.</param>
    /// <param name="dYPost">Per objective, the <c>(n_qmc, dim)</c> derivative of its samples with respect to the candidate point.</param>
    /// <returns>The per-draw log HVI and its <c>(n_qmc, dim)</c> gradient.</returns>
    internal (double[] LogHvi, double[,] DLogHvi) ComputePerSampleLogHviWithGrad(double[,] yPost, double[][,] dYPost)
    {
        double[] hvi = PerSampleLogHvi(yPost, out double[][] termsPerS);
        int nQmc = hvi.Length;
        int dim = dYPost[0].GetLength(1);
        var dHvi = new double[nQmc, dim];
        for (int s = 0; s < nQmc; s++)
        {
            double[,] lower = _lowerBounds[s];
            double[,] interval = _intervals[s];
            double[] term = termsPerS[s];
            for (int b = 0; b < term.Length; b++)
            {
                double w = Math.Exp(term[b] - hvi[s]);   // this box's share of the draw's logsumexp
                if (w == 0.0) continue;
                for (int o = 0; o < _nObjectives; o++)
                {
                    double raw = yPost[s, o] - lower[b, o];
                    // Two-sided clamp: the gradient exists only strictly inside (EPS, interval). A padded
                    // box has interval == EPS, so it never contributes here, exactly like autograd.
                    if (!(raw > Eps && raw < interval[b, o]))
                        continue;
                    double factor = w / raw;
                    double[,] ds = dYPost[o];
                    for (int d = 0; d < dim; d++)
                        dHvi[s, d] += factor * ds[s, d];
                }
            }
        }
        return (hvi, dHvi);
    }

    /// <summary>
    ///     The per-draw log HVI and the per-box log terms it reduces (kept for the gradient's box softmax).
    /// </summary>
    /// <param name="yPost">The candidate's per-draw objective samples, shape <c>(n_qmc, n_objectives)</c>.</param>
    /// <param name="termsPerS">Receives, per draw, the <c>Σ_o log(clamped diff)</c> term of every (padded) box.</param>
    /// <returns>The per-draw log hypervolume improvement.</returns>
    private double[] PerSampleLogHvi(double[,] yPost, out double[][] termsPerS)
    {
        int nQmc = _lowerBounds.Length;
        var hvi = new double[nQmc];
        termsPerS = new double[nQmc][];
        for (int s = 0; s < nQmc; s++)
        {
            double[,] lower = _lowerBounds[s];
            double[,] interval = _intervals[s];
            int nBoxes = lower.GetLength(0);
            var term = new double[nBoxes];
            for (int b = 0; b < nBoxes; b++)
            {
                double sumLog = 0.0;
                for (int o = 0; o < _nObjectives; o++)
                {
                    double raw = yPost[s, o] - lower[b, o];
                    double iv = interval[b, o];
                    // clamp_(min=EPS, max=interval); a NaN sample stays NaN, as in torch.
                    double diff = raw < Eps ? Eps : (raw > iv ? iv : raw);
                    sumLog += Math.Log(diff);
                }
                term[b] = sumLog;
            }
            hvi[s] = Acqf.LogSumExp(term);
            termsPerS[s] = term;
        }
        return hvi;
    }

    /// <summary>Stacks two row-major matrices vertically (<c>torch.cat([top, bottom], dim=0)</c>).</summary>
    /// <param name="top">The rows placed first; its column count defines the result's width.</param>
    /// <param name="bottom">The rows appended after <paramref name="top" />; must have the same column count.</param>
    /// <returns>A new matrix holding the rows of <paramref name="top" /> followed by those of <paramref name="bottom" />.</returns>
    private static double[,] ConcatRows(double[,] top, double[,] bottom)
    {
        int nTop = top.GetLength(0), nBottom = bottom.GetLength(0), cols = top.GetLength(1);
        var result = new double[nTop + nBottom, cols];
        for (int i = 0; i < nTop; i++)
            for (int j = 0; j < cols; j++)
                result[i, j] = top[i, j];
        for (int i = 0; i < nBottom; i++)
            for (int j = 0; j < cols; j++)
                result[nTop + i, j] = bottom[i, j];
        return result;
    }
}

/// <summary>
///     The common surface the acqf optimizer (<see cref="OptimMixed" />) drives — port of
///     <c>acqf.BaseAcquisitionFunc</c>. It exposes the kernel <see cref="LengthScales" /> (used for the
///     gradient-ascent preconditioning), the normalized <see cref="SearchSpace" />, and the three
///     evaluation entry points: batched <c>eval_acqf_no_grad</c>, its single-point form (the discrete
///     line/exhaustive search), and value+gradient <c>eval_acqf_with_grad</c> (the gradient ascent).
/// </summary>
internal interface IAcquisitionFunc
{
    /// <summary>The normalized parameter space over which this acquisition function is optimized.</summary>
    GpSearchSpace SearchSpace { get; }
    /// <summary>The kernel length scale of each coordinate, used to precondition gradient steps.</summary>
    double[] LengthScales { get; }

    /// <summary><c>eval_acqf_no_grad</c> over a batch <c>(B, dim)</c>.</summary>
    /// <param name="x">Normalized input coordinates in search-space order; rectangular matrices contain one point per row.</param>
    /// <returns>The acquisition value at the supplied point, or one value per input row for a batch.</returns>
    double[] EvalAcqfNoGrad(double[,] x);

    /// <summary><c>eval_acqf_no_grad</c> at a single point <c>(dim,)</c> — the discrete searches.</summary>
    /// <param name="x">Normalized input coordinates in search-space order; rectangular matrices contain one point per row.</param>
    /// <returns>The acquisition value at the supplied point, or one value per input row for a batch.</returns>
    double EvalAcqfNoGrad(double[] x);

    /// <summary><c>eval_acqf_with_grad</c> — value + ∂/∂x at a single point <c>(dim,)</c>.</summary>
    /// <param name="x">Normalized input coordinates in search-space order; rectangular matrices contain one point per row.</param>
    /// <returns>The acquisition value and its gradient in normalized parameter coordinates.</returns>
    (double Value, double[] Grad) EvalAcqfWithGrad(double[] x);
}

/// <summary>
///     The log-expected-improvement acquisition function — port of <c>acqf.LogEI</c>. It scores a
///     normalized point by the log expected improvement of the GP posterior over the current best
///     value <c>threshold</c>. <see cref="EvalAcqfNoGrad(double[,])" /> evaluates a batch (used for the
///     preliminary QMC samples and the discrete search); <see cref="EvalAcqfWithGrad" /> returns the
///     value and the hand-derived gradient (the autograd replacement the acqf optimizer's gradient
///     ascent needs). When <c>threshold</c> is <c>−∞</c> (no feasible trial) the acqf is identically 0.
/// </summary>
internal sealed class LogEI : IAcquisitionFunc
{
    private readonly GPRegressor _gpr;
    private readonly double _threshold;
    private readonly double _stabilizingNoise;

    /// <summary>Configures log expected improvement from the fitted model, search space, and supplied thresholds or sampling settings.</summary>
    /// <param name="gpr">The Gaussian-process model supplying posterior predictions and kernel length scales.</param>
    /// <param name="searchSpace">The normalized coordinate layout, bounds, and discrete choices used by the acquisition optimizer.</param>
    /// <param name="threshold">The incumbent objective value that a candidate must improve upon, in the model's maximization convention.</param>
    /// <param name="stabilizingNoise">A small variance increment used to avoid singular predictive standard deviations.</param>
    internal LogEI(GPRegressor gpr, GpSearchSpace searchSpace, double threshold, double stabilizingNoise = 1e-12)
    {
        _gpr = gpr;
        _threshold = threshold;
        _stabilizingNoise = stabilizingNoise;
        SearchSpace = searchSpace;
        LengthScales = gpr.LengthScales;
    }

    /// <summary>The normalized parameter space over which this acquisition function is optimized.</summary>
    public GpSearchSpace SearchSpace { get; }
    /// <summary>The kernel length scale of each coordinate, used to precondition gradient steps.</summary>
    public double[] LengthScales { get; }

    /// <summary><c>eval_acqf_no_grad</c> — LogEI at each row of <paramref name="x" /> (<c>(B, dim)</c>).</summary>
    /// <param name="x">Normalized input coordinates in search-space order; rectangular matrices contain one point per row.</param>
    /// <returns>The acquisition value at the supplied point, or one value per input row for a batch.</returns>
    public double[] EvalAcqfNoGrad(double[,] x)
    {
        int b = x.GetLength(0);
        var result = new double[b];
        if (double.IsNegativeInfinity(_threshold))
            return result;  // all zeros
        (double[] mean, double[] var) = _gpr.Posterior(x);
        for (int i = 0; i < b; i++)
            result[i] = Acqf.LogEi(mean[i], var[i] + _stabilizingNoise, _threshold);
        return result;
    }

    /// <summary><c>eval_acqf_no_grad</c> at a single point — the 0-d form used by the discrete searches.</summary>
    /// <param name="x">Normalized input coordinates in search-space order; rectangular matrices contain one point per row.</param>
    /// <returns>The acquisition value at the supplied point, or one value per input row for a batch.</returns>
    public double EvalAcqfNoGrad(double[] x)
    {
        if (double.IsNegativeInfinity(_threshold))
            return 0.0;
        int dim = x.Length;
        var row = new double[1, dim];
        for (int d = 0; d < dim; d++)
            row[0, d] = x[d];
        (double[] mean, double[] var) = _gpr.Posterior(row);
        return Acqf.LogEi(mean[0], var[0] + _stabilizingNoise, _threshold);
    }

    /// <summary><c>eval_acqf_with_grad</c> — LogEI and ∂LogEI/∂x at a single point <paramref name="x" /> (<c>(dim,)</c>).</summary>
    /// <param name="x">Normalized input coordinates in search-space order; rectangular matrices contain one point per row.</param>
    /// <returns>The acquisition value and its gradient in normalized parameter coordinates.</returns>
    public (double Value, double[] Grad) EvalAcqfWithGrad(double[] x)
    {
        int dim = x.Length;
        if (double.IsNegativeInfinity(_threshold))
            return (0.0, new double[dim]);

        (double mean, double var, double[] dMean, double[] dVar, bool _) = _gpr.PosteriorWithGrad(x);
        double sigma = Math.Sqrt(var + _stabilizingNoise);
        double z = (mean - _threshold) / sigma;
        (double logeiVal, double dlogeiDz) = Acqf.StandardLogEiAndGrad(z);
        double value = logeiVal + Math.Log(sigma);

        var grad = new double[dim];
        for (int d = 0; d < dim; d++)
        {
            double dSigma = dVar[d] / (2.0 * sigma);
            double dz = dMean[d] / sigma - z * dSigma / sigma;
            grad[d] = dlogeiDz * dz + dSigma / sigma;
        }
        return (value, grad);
    }
}

/// <summary>
///     The q-batch log-expected-improvement acquisition function — port of <c>acqf.qLogEI</c>, the acqf
///     <see cref="OptunaSharp.Samplers.GPSampler" /> maximizes for single-objective optimization <b>with running trials</b>
///     (PORTING_PLAN Phase 3.2 part 3e-6). It conditions on the running fantasies via a
///     <see cref="ConditionalGPRegressor" /> and, per QMC draw, forms the joint-posterior sample vector
///     over the running trials plus the query point, takes the improvement <c>log(y − threshold)</c>,
///     and reduces with <see cref="Acqf.MeanMaxUtility" /> (max over the q-batch, log-mean over the
///     samples). See Eq. (10) of arxiv 2310.20708.
///
///     <para><b>Gradient:</b> the fantasy columns are constant, so only the query column carries an
///     x-gradient (via <see cref="ConditionalGPRegressor.SampleJointPosteriorWithGrad" />); it flows
///     through the <c>amax</c> selection (the per-sample argmax) and the <c>logsumexp</c> softmax — the
///     hand-derived replacement for autograd. When <c>threshold</c> is <c>−∞</c> the acqf is identically
///     0 (matching Optuna's all-zeros utility).</para>
/// </summary>
internal sealed class QLogEI : IAcquisitionFunc
{
    private const double Eps = 1e-12;

    private readonly ConditionalGPRegressor _condGpr;
    private readonly double _threshold;

    /// <summary>Configures batch log expected improvement from the fitted model, search space, and supplied thresholds or sampling settings.</summary>
    /// <param name="gpr">The Gaussian-process model supplying posterior predictions and kernel length scales.</param>
    /// <param name="searchSpace">The normalized coordinate layout, bounds, and discrete choices used by the acquisition optimizer.</param>
    /// <param name="threshold">The incumbent objective value that a candidate must improve upon, in the model's maximization convention.</param>
    /// <param name="nQmcSamples">The number of fixed quasi-Monte Carlo draws used to approximate posterior expectations.</param>
    /// <param name="qmcSeed">The seed used to scramble the quasi-Monte Carlo normal samples.</param>
    /// <param name="normalizedParamsOfRunningTrials">The normalized parameter rows of running trials to include as pending candidates.</param>
    /// <param name="stabilizingNoise">A small variance increment used to avoid singular predictive standard deviations.</param>
    internal QLogEI(
        GPRegressor gpr, GpSearchSpace searchSpace, double threshold,
        int nQmcSamples, int qmcSeed, double[,] normalizedParamsOfRunningTrials,
        double stabilizingNoise = 1e-12)
    {
        _threshold = threshold;
        _condGpr = new ConditionalGPRegressor(
            gpr, normalizedParamsOfRunningTrials, nQmcSamples, qmcSeed, stabilizingNoise);
        SearchSpace = searchSpace;
        LengthScales = gpr.LengthScales;
    }

    /// <summary>The normalized parameter space over which this acquisition function is optimized.</summary>
    public GpSearchSpace SearchSpace { get; }
    /// <summary>The kernel length scale of each coordinate, used to precondition gradient steps.</summary>
    public double[] LengthScales { get; }

    /// <summary><c>compute_per_sample_log_utility</c> — <c>log(clamp_min(y_post − threshold, EPS))</c>, shape <c>(n_qmc, r+1)</c>.</summary>
    /// <param name="yPost">Posterior objective draws with rows for QMC samples and columns for pending/query points.</param>
    /// <returns>A matrix of log utilities with one row per posterior sample and one column per batch point.</returns>
    internal double[,] PerSampleLogUtility(double[,] yPost)
    {
        int nQmc = yPost.GetLength(0), q = yPost.GetLength(1);
        var lu = new double[nQmc, q];
        for (int s = 0; s < nQmc; s++)
            for (int j = 0; j < q; j++)
            {
                double v = yPost[s, j] - _threshold;
                lu[s, j] = Math.Log(v < Eps ? Eps : v);
            }
        return lu;
    }

    /// <summary>The per-sample log utility at a single query point (used by <see cref="QLogCEI" /> to compose feasible improvement).</summary>
    /// <param name="x">Normalized input coordinates in search-space order; rectangular matrices contain one point per row.</param>
    /// <returns>The fixed-sample log utilities of pending points and the query point.</returns>
    internal double[,] ComputePerSampleLogUtility(double[] x)
    {
        if (double.IsNegativeInfinity(_threshold))
            return new double[_condGpr.NQmc, _condGpr.NRunning + 1];   // all zeros
        return PerSampleLogUtility(_condGpr.SampleJointPosterior(x));
    }

    /// <summary>The per-sample log utility and its x-gradient (the gradient flows through the query column only).</summary>
    /// <param name="x">Normalized input coordinates in search-space order; rectangular matrices contain one point per row.</param>
    /// <returns>The per-sample log utilities and query-coordinate derivatives.</returns>
    internal (double[,] LogUtil, double[,] DQuery) ComputePerSampleLogUtilityWithGrad(double[] x)
    {
        if (double.IsNegativeInfinity(_threshold))
            return (new double[_condGpr.NQmc, _condGpr.NRunning + 1], new double[_condGpr.NQmc, x.Length]);
        (double[,] yPost, double[,] dSamples) = _condGpr.SampleJointPosteriorWithGrad(x);
        return (PerSampleLogUtility(yPost), QueryLogUtilGrad(yPost, dSamples));
    }

    /// <summary>
    ///     <c>∂ log_util[s, query]/∂x_d = mask[s]·(1/(y_post[s,query] − threshold))·∂samples[s]/∂x_d</c>
    ///     (the <c>clamp_min</c> mask zeroes it where the improvement ≤ EPS); the fantasy columns are 0.
    /// </summary>
    /// <param name="yPost">Posterior objective draws with rows for QMC samples and columns for pending/query points.</param>
    /// <param name="dSamples">The derivative of query posterior samples with respect to normalized coordinates.</param>
    /// <returns>The derivative of each query sample's log utility in normalized coordinate order.</returns>
    private double[,] QueryLogUtilGrad(double[,] yPost, double[,] dSamples)
    {
        int nQmc = yPost.GetLength(0);
        int queryCol = yPost.GetLength(1) - 1;
        int dim = dSamples.GetLength(1);
        var dq = new double[nQmc, dim];
        for (int s = 0; s < nQmc; s++)
        {
            double v = yPost[s, queryCol] - _threshold;
            if (!(v > Eps))
                continue;   // clamp_min → zero gradient
            double inv = 1.0 / v;
            for (int d = 0; d < dim; d++)
                dq[s, d] = inv * dSamples[s, d];
        }
        return dq;
    }

    /// <summary><c>eval_acqf_no_grad</c> over a batch <c>(B, dim)</c>.</summary>
    /// <param name="x">Normalized input coordinates in search-space order; rectangular matrices contain one point per row.</param>
    /// <returns>The acquisition value at the supplied point, or one value per input row for a batch.</returns>
    public double[] EvalAcqfNoGrad(double[,] x)
    {
        int b = x.GetLength(0), dim = x.GetLength(1);
        var result = new double[b];
        var row = new double[dim];
        for (int r = 0; r < b; r++)
        {
            for (int d = 0; d < dim; d++) row[d] = x[r, d];
            result[r] = EvalAcqfNoGrad(row);
        }
        return result;
    }

    /// <summary><c>eval_acqf_no_grad</c> at a single point.</summary>
    /// <param name="x">Normalized input coordinates in search-space order; rectangular matrices contain one point per row.</param>
    /// <returns>The acquisition value at the supplied point, or one value per input row for a batch.</returns>
    public double EvalAcqfNoGrad(double[] x) => Acqf.MeanMaxUtility(ComputePerSampleLogUtility(x));

    /// <summary><c>eval_acqf_with_grad</c> — value and the hand-derived gradient through amax/logsumexp.</summary>
    /// <param name="x">Normalized input coordinates in search-space order; rectangular matrices contain one point per row.</param>
    /// <returns>The acquisition value and its gradient in normalized parameter coordinates.</returns>
    public (double Value, double[] Grad) EvalAcqfWithGrad(double[] x)
    {
        (double[,] logUtil, double[,] dQuery) = ComputePerSampleLogUtilityWithGrad(x);
        return Acqf.MeanMaxUtilityWithGrad(logUtil, dQuery);
    }
}

/// <summary>
///     The q-batch log probability-of-improvement acquisition function — port of <c>acqf.qLogPI</c>. It
///     is the running-trial analogue of <see cref="LogPI" />: <see cref="OptunaSharp.Samplers.GPSampler" /> uses it, summed
///     over the constraint GPs, as the feasibility term of <see cref="QLogCEI" /> (single-objective
///     constrained optimization with running trials, PORTING_PLAN Phase 3.2 part 3e-6). Per QMC draw it
///     forms the joint-posterior sample over the running trials plus the query point
///     (<see cref="ConditionalGPRegressor" />), applies the smooth feasibility indicator
///     <c>logsigmoid((y − threshold)/τ)</c> (τ = 1e-2), and reduces with <see cref="Acqf.MeanMaxUtility" />.
///
///     <para>Since Optuna 5.0.0 it also serves <see cref="QLogCEHVI" /> (multi-objective constrained
///     with running trials) through two lower-level entry points: the candidate-only feasibility
///     (<see cref="ComputeCandidateLogUtility" />) and the per-draw feasibility of the running fantasies
///     (<see cref="IsFantasyAboveThreshold" />), which decides which fantasies enter the Pareto baseline.</para>
/// </summary>
internal sealed class QLogPI : IAcquisitionFunc
{
    private readonly ConditionalGPRegressor _condGpr;
    private readonly double _threshold;
    private readonly double _tau;

    /// <summary>Configures batch log probability of improvement from the fitted model, search space, and supplied thresholds or sampling settings.</summary>
    /// <param name="gpr">The Gaussian-process model supplying posterior predictions and kernel length scales.</param>
    /// <param name="searchSpace">The normalized coordinate layout, bounds, and discrete choices used by the acquisition optimizer.</param>
    /// <param name="threshold">The incumbent objective value that a candidate must improve upon, in the model's maximization convention.</param>
    /// <param name="nQmcSamples">The number of fixed quasi-Monte Carlo draws used to approximate posterior expectations.</param>
    /// <param name="qmcSeed">The seed used to scramble the quasi-Monte Carlo normal samples.</param>
    /// <param name="normalizedParamsOfRunningTrials">The normalized parameter rows of running trials to include as pending candidates.</param>
    /// <param name="stabilizingNoise">A small variance increment used to avoid singular predictive standard deviations.</param>
    /// <param name="tau">The positive temperature controlling the smooth probability-of-improvement approximation.</param>
    internal QLogPI(
        GPRegressor gpr, GpSearchSpace searchSpace, double threshold, int nQmcSamples, int qmcSeed,
        double[,] normalizedParamsOfRunningTrials, double stabilizingNoise = 1e-12, double tau = 1e-2)
    {
        _threshold = threshold;
        _tau = tau;
        _condGpr = new ConditionalGPRegressor(
            gpr, normalizedParamsOfRunningTrials, nQmcSamples, qmcSeed, stabilizingNoise);
        SearchSpace = searchSpace;
        LengthScales = gpr.LengthScales;
    }

    /// <summary>The normalized parameter space over which this acquisition function is optimized.</summary>
    public GpSearchSpace SearchSpace { get; }
    /// <summary>The kernel length scale of each coordinate, used to precondition gradient steps.</summary>
    public double[] LengthScales { get; }

    /// <summary>Returns a matrix of log utilities with one row per posterior sample and one column per batch point.</summary>
    /// <param name="yPost">Posterior objective draws with rows for QMC samples and columns for pending/query points.</param>
    /// <returns>A matrix of log utilities with one row per posterior sample and one column per batch point.</returns>
    private double[,] PerSampleLogUtility(double[,] yPost)
    {
        int nQmc = yPost.GetLength(0), q = yPost.GetLength(1);
        var lu = new double[nQmc, q];
        for (int s = 0; s < nQmc; s++)
            for (int j = 0; j < q; j++)
                lu[s, j] = LogSigmoid((yPost[s, j] - _threshold) / _tau);
        return lu;
    }

    /// <summary><c>compute_per_sample_log_utility(x)</c> — <c>logsigmoid((y_post − threshold)/τ)</c>, shape <c>(n_qmc, r+1)</c>.</summary>
    /// <param name="x">Normalized input coordinates in search-space order; rectangular matrices contain one point per row.</param>
    /// <returns>The fixed-sample log utilities of pending points and the query point.</returns>
    internal double[,] ComputePerSampleLogUtility(double[] x)
        => PerSampleLogUtility(_condGpr.SampleJointPosterior(x));

    /// <summary>The per-sample log utility and its x-gradient (through the query column only).</summary>
    /// <param name="x">Normalized input coordinates in search-space order; rectangular matrices contain one point per row.</param>
    /// <returns>The per-sample log utilities and query-coordinate derivatives.</returns>
    internal (double[,] LogUtil, double[,] DQuery) ComputePerSampleLogUtilityWithGrad(double[] x)
    {
        (double[,] yPost, double[,] dSamples) = _condGpr.SampleJointPosteriorWithGrad(x);
        int nQmc = yPost.GetLength(0);
        int queryCol = yPost.GetLength(1) - 1;
        int dim = dSamples.GetLength(1);
        var dq = new double[nQmc, dim];
        for (int s = 0; s < nQmc; s++)
        {
            // d logsigmoid(arg)/d y_post = sigmoid(−arg)/τ = 1/((1 + e^arg)·τ).
            double arg = (yPost[s, queryCol] - _threshold) / _tau;
            double dlog = 1.0 / (1.0 + Math.Exp(arg)) / _tau;
            for (int d = 0; d < dim; d++)
                dq[s, d] = dlog * dSamples[s, d];
        }
        return (PerSampleLogUtility(yPost), dq);
    }

    /// <summary>
    ///     <c>compute_candidate_log_utility(x)</c> (Optuna 5.0.0, #6804) — the smooth feasibility
    ///     <c>logsigmoid((y − threshold)/τ)</c> of the CANDIDATE column only, one value per QMC draw.
    ///     <see cref="QLogCEHVI" /> uses it instead of <see cref="ComputePerSampleLogUtility" /> because the
    ///     running trials' feasibility is already folded into its per-draw Pareto baseline.
    /// </summary>
    /// <param name="x">The candidate point in normalized search-space coordinates.</param>
    /// <returns>The per-draw candidate log feasibility, shape <c>(n_qmc,)</c>.</returns>
    internal double[] ComputeCandidateLogUtility(double[] x)
    {
        double[] yPost = _condGpr.SampleQuerySamples(x);
        var lu = new double[yPost.Length];
        for (int s = 0; s < yPost.Length; s++)
            lu[s] = LogSigmoid((yPost[s] - _threshold) / _tau);
        return lu;
    }

    /// <summary>
    ///     <see cref="ComputeCandidateLogUtility" /> plus its gradient with respect to the candidate point:
    ///     <c>sigmoid(−arg)/τ · ∂y/∂x</c> per draw (the analytic replacement for autograd).
    /// </summary>
    /// <param name="x">The candidate point in normalized search-space coordinates.</param>
    /// <returns>The per-draw candidate log feasibility and its <c>(n_qmc, dim)</c> gradient.</returns>
    internal (double[] LogUtil, double[,] DLogUtil) ComputeCandidateLogUtilityWithGrad(double[] x)
    {
        (double[] yPost, double[,] dSamples) = _condGpr.SampleQueryWithGrad(x);
        int nQmc = yPost.Length;
        int dim = dSamples.GetLength(1);
        var lu = new double[nQmc];
        var dlu = new double[nQmc, dim];
        for (int s = 0; s < nQmc; s++)
        {
            double arg = (yPost[s] - _threshold) / _tau;
            lu[s] = LogSigmoid(arg);
            // d logsigmoid(arg)/d y = sigmoid(−arg)/τ = 1/((1 + e^arg)·τ).
            double dlog = 1.0 / (1.0 + Math.Exp(arg)) / _tau;
            for (int d = 0; d < dim; d++)
                dlu[s, d] = dlog * dSamples[s, d];
        }
        return (lu, dlu);
    }

    /// <summary>
    ///     <c>is_fantasy_above_threshold()</c> (Optuna 5.0.0, #6804) — whether each running trial's fantasy
    ///     satisfies this constraint in each QMC draw. The comparison is inclusive (<c>&gt;=</c>), exactly
    ///     like Optuna, so a fantasy sitting on the threshold counts as feasible.
    /// </summary>
    /// <returns>A <c>(n_qmc, n_running)</c> mask; <c>true</c> where the fantasy is at or above the threshold.</returns>
    internal bool[,] IsFantasyAboveThreshold()
    {
        double[,] fantasy = _condGpr.GetFantasySamples();
        int nQmc = fantasy.GetLength(0), nRunning = fantasy.GetLength(1);
        var mask = new bool[nQmc, nRunning];
        for (int s = 0; s < nQmc; s++)
            for (int r = 0; r < nRunning; r++)
                mask[s, r] = fantasy[s, r] >= _threshold;
        return mask;
    }

    /// <summary><c>eval_acqf_no_grad</c> over a batch <c>(B, dim)</c>.</summary>
    /// <param name="x">Normalized input coordinates in search-space order; rectangular matrices contain one point per row.</param>
    /// <returns>The acquisition value at the supplied point, or one value per input row for a batch.</returns>
    public double[] EvalAcqfNoGrad(double[,] x)
    {
        int b = x.GetLength(0), dim = x.GetLength(1);
        var result = new double[b];
        var row = new double[dim];
        for (int r = 0; r < b; r++)
        {
            for (int d = 0; d < dim; d++) row[d] = x[r, d];
            result[r] = EvalAcqfNoGrad(row);
        }
        return result;
    }

    /// <summary><c>eval_acqf_no_grad</c> at a single point.</summary>
    /// <param name="x">Normalized input coordinates in search-space order; rectangular matrices contain one point per row.</param>
    /// <returns>The acquisition value at the supplied point, or one value per input row for a batch.</returns>
    public double EvalAcqfNoGrad(double[] x) => Acqf.MeanMaxUtility(ComputePerSampleLogUtility(x));

    /// <summary><c>eval_acqf_with_grad</c> — value and the hand-derived gradient through amax/logsumexp.</summary>
    /// <param name="x">Normalized input coordinates in search-space order; rectangular matrices contain one point per row.</param>
    /// <returns>The acquisition value and its gradient in normalized parameter coordinates.</returns>
    public (double Value, double[] Grad) EvalAcqfWithGrad(double[] x)
    {
        (double[,] logUtil, double[,] dQuery) = ComputePerSampleLogUtilityWithGrad(x);
        return Acqf.MeanMaxUtilityWithGrad(logUtil, dQuery);
    }

    /// <summary><c>torch.nn.functional.logsigmoid(v) = min(0,v) − log1p(e^(−|v|))</c> — the numerically stable log σ.</summary>
    /// <param name="v">The values used by the numerical reduction or transformation.</param>
    /// <returns>The logarithm of the logistic sigmoid, evaluated without unnecessary exponential overflow.</returns>
    private static double LogSigmoid(double v)
        => v >= 0 ? -MathUtil.Log1p(Math.Exp(-v)) : v - MathUtil.Log1p(Math.Exp(v));
}

/// <summary>
///     The q-batch log constrained-expected-improvement acquisition function — port of
///     <c>acqf.qLogCEI</c>, the acqf <see cref="OptunaSharp.Samplers.GPSampler" /> maximizes for single-objective
///     <b>constrained</b> optimization <b>with running trials</b> (PORTING_PLAN Phase 3.2 part 3e-6). It
///     composes the per-QMC-sample log feasible improvement — the <see cref="QLogEI" /> objective utility
///     plus the summed <see cref="QLogPI" /> constraint utilities — and reduces the sum with
///     <see cref="Acqf.MeanMaxUtility" /> (so the max-over-q-batch is taken on the joint feasible
///     improvement, not per acqf). Each constraint <see cref="QLogPI" /> uses the QMC seed
///     <c>qmc_seed + i + 1</c> (its own fantasies), matching Optuna. When there is no feasible trial the
///     objective threshold is <c>−∞</c> and its utility is 0, leaving only the constraint terms.
/// </summary>
internal sealed class QLogCEI : IAcquisitionFunc
{
    private readonly QLogEI _acqf;
    private readonly QLogPI[] _constraintsAcqfList;

    /// <summary>Configures batch log constrained expected improvement from the fitted model, search space, and supplied thresholds or sampling settings.</summary>
    /// <param name="gpr">The Gaussian-process model supplying posterior predictions and kernel length scales.</param>
    /// <param name="searchSpace">The normalized coordinate layout, bounds, and discrete choices used by the acquisition optimizer.</param>
    /// <param name="threshold">The incumbent objective value that a candidate must improve upon, in the model's maximization convention.</param>
    /// <param name="nQmcSamples">The number of fixed quasi-Monte Carlo draws used to approximate posterior expectations.</param>
    /// <param name="qmcSeed">The seed used to scramble the quasi-Monte Carlo normal samples.</param>
    /// <param name="constraintsGprList">One Gaussian-process model per modeled constraint.</param>
    /// <param name="constraintsThresholdList">The feasibility threshold for each constraint model, in matching order.</param>
    /// <param name="normalizedParamsOfRunningTrials">The normalized parameter rows of running trials to include as pending candidates.</param>
    /// <param name="stabilizingNoise">A small variance increment used to avoid singular predictive standard deviations.</param>
    /// <exception cref="System.ArgumentException">The constraint-model and threshold lists are empty or have different lengths.</exception>
    internal QLogCEI(
        GPRegressor gpr, GpSearchSpace searchSpace, double threshold, int nQmcSamples, int qmcSeed,
        IReadOnlyList<GPRegressor> constraintsGprList, IReadOnlyList<double> constraintsThresholdList,
        double[,] normalizedParamsOfRunningTrials, double stabilizingNoise = 1e-12)
    {
        if (constraintsGprList.Count != constraintsThresholdList.Count || constraintsGprList.Count == 0)
            throw new ArgumentException("constraints_gpr_list and constraints_threshold_list must be non-empty and equal-length.");

        _acqf = new QLogEI(
            gpr, searchSpace, threshold, nQmcSamples, qmcSeed, normalizedParamsOfRunningTrials, stabilizingNoise);
        _constraintsAcqfList = new QLogPI[constraintsGprList.Count];
        for (int i = 0; i < constraintsGprList.Count; i++)
            _constraintsAcqfList[i] = new QLogPI(
                constraintsGprList[i], searchSpace, constraintsThresholdList[i], nQmcSamples, qmcSeed + i + 1,
                normalizedParamsOfRunningTrials, stabilizingNoise);

        SearchSpace = searchSpace;
        LengthScales = gpr.LengthScales;
    }

    /// <summary>The normalized parameter space over which this acquisition function is optimized.</summary>
    public GpSearchSpace SearchSpace { get; }
    /// <summary>The kernel length scale of each coordinate, used to precondition gradient steps.</summary>
    public double[] LengthScales { get; }

    /// <summary><c>eval_acqf_no_grad</c> over a batch <c>(B, dim)</c>.</summary>
    /// <param name="x">Normalized input coordinates in search-space order; rectangular matrices contain one point per row.</param>
    /// <returns>The acquisition value at the supplied point, or one value per input row for a batch.</returns>
    public double[] EvalAcqfNoGrad(double[,] x)
    {
        int b = x.GetLength(0), dim = x.GetLength(1);
        var result = new double[b];
        var row = new double[dim];
        for (int r = 0; r < b; r++)
        {
            for (int d = 0; d < dim; d++) row[d] = x[r, d];
            result[r] = EvalAcqfNoGrad(row);
        }
        return result;
    }

    /// <summary><c>eval_acqf_no_grad</c> at a single point.</summary>
    /// <param name="x">Normalized input coordinates in search-space order; rectangular matrices contain one point per row.</param>
    /// <returns>The acquisition value at the supplied point, or one value per input row for a batch.</returns>
    public double EvalAcqfNoGrad(double[] x)
    {
        double[,] logUtil = _acqf.ComputePerSampleLogUtility(x);
        foreach (QLogPI c in _constraintsAcqfList)
            AddInPlace(logUtil, c.ComputePerSampleLogUtility(x));
        return Acqf.MeanMaxUtility(logUtil);
    }

    /// <summary><c>eval_acqf_with_grad</c> — value and gradient of the composed feasible-improvement utility.</summary>
    /// <param name="x">Normalized input coordinates in search-space order; rectangular matrices contain one point per row.</param>
    /// <returns>The acquisition value and its gradient in normalized parameter coordinates.</returns>
    public (double Value, double[] Grad) EvalAcqfWithGrad(double[] x)
    {
        (double[,] logUtil, double[,] dQuery) = _acqf.ComputePerSampleLogUtilityWithGrad(x);
        foreach (QLogPI c in _constraintsAcqfList)
        {
            (double[,] lu, double[,] dq) = c.ComputePerSampleLogUtilityWithGrad(x);
            AddInPlace(logUtil, lu);
            AddInPlace(dQuery, dq);
        }
        return Acqf.MeanMaxUtilityWithGrad(logUtil, dQuery);
    }

    /// <summary>Adds matching log-utility arrays so independent objective and feasibility factors combine multiplicatively.</summary>
    /// <param name="a">The first numerical operand; its dimensions must agree with the operation described in the summary.</param>
    /// <param name="b">The second numerical operand; its dimensions must agree with the first operand.</param>
    private static void AddInPlace(double[,] a, double[,] b)
    {
        int rows = a.GetLength(0), cols = a.GetLength(1);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                a[i, j] += b[i, j];
    }
}

/// <summary>
///     The log probability-of-improvement acquisition function — port of <c>acqf.LogPI</c>. It scores
///     a point by <c>log Φ((mean − threshold)/σ)</c> (the log of the feasible probability of the GP
///     posterior over <c>threshold</c>). <see cref="OptunaSharp.Samplers.GPSampler" /> uses it, summed over the constraint
///     GPs, as the feasibility term of <see cref="LogCEI" /> (single-objective constrained
///     optimization) and of <see cref="LogCEHVI" /> (multi-objective constrained). It rides
///     <c>log_ndtr</c> → the erf/ucrt tolerance class.
///
///     <para><b>Running trials (Kriging-Believer):</b> when
///     <c>normalizedParamsOfRunningTrials</c> is given, the running points are appended to the GP with
///     their posterior mean as the fantasy objective value (<see cref="GPRegressor.AppendRunningData" />),
///     so the feasible-probability accounts for the trials already in flight (Optuna's
///     <c>append_running_data</c> path; PORTING_PLAN Phase 3.2 part 3e-6). <b>Footgun:</b> this mutates
///     the supplied GP. Optuna 5.0.0 keeps the parameter but no longer passes it from the sampler — the
///     multi-objective constrained running case moved to <see cref="QLogCEHVI" /> (#6804) — so in the
///     sampler every <see cref="LogPI" /> is now built without running trials.</para>
/// </summary>
internal sealed class LogPI : IAcquisitionFunc
{
    private readonly GPRegressor _gpr;
    private readonly double _threshold;
    private readonly double _stabilizingNoise;

    /// <summary>Configures log probability of improvement from the fitted model, search space, and supplied thresholds or sampling settings.</summary>
    /// <param name="gpr">The Gaussian-process model supplying posterior predictions and kernel length scales.</param>
    /// <param name="searchSpace">The normalized coordinate layout, bounds, and discrete choices used by the acquisition optimizer.</param>
    /// <param name="threshold">The incumbent objective value that a candidate must improve upon, in the model's maximization convention.</param>
    /// <param name="normalizedParamsOfRunningTrials">The normalized parameter rows of running trials to include as pending candidates.</param>
    /// <param name="stabilizingNoise">A small variance increment used to avoid singular predictive standard deviations.</param>
    internal LogPI(
        GPRegressor gpr, GpSearchSpace searchSpace, double threshold,
        double[,]? normalizedParamsOfRunningTrials = null, double stabilizingNoise = 1e-12)
    {
        _gpr = gpr;
        _threshold = threshold;
        _stabilizingNoise = stabilizingNoise;

        if (normalizedParamsOfRunningTrials != null)
            // Kriging Believer: append the running points with their posterior mean as the fantasy
            // objective, mutating the constraint GP in place (matching Optuna's LogPI).
            _gpr.AppendRunningData(
                normalizedParamsOfRunningTrials, _gpr.Posterior(normalizedParamsOfRunningTrials).Mean);

        SearchSpace = searchSpace;
        LengthScales = gpr.LengthScales;
    }

    /// <summary>The normalized parameter space over which this acquisition function is optimized.</summary>
    public GpSearchSpace SearchSpace { get; }
    /// <summary>The kernel length scale of each coordinate, used to precondition gradient steps.</summary>
    public double[] LengthScales { get; }

    /// <summary><c>eval_acqf_no_grad</c> — <c>log_ndtr((mean − threshold)/σ)</c> over a batch <c>(B, dim)</c>.</summary>
    /// <param name="x">Normalized input coordinates in search-space order; rectangular matrices contain one point per row.</param>
    /// <returns>The acquisition value at the supplied point, or one value per input row for a batch.</returns>
    public double[] EvalAcqfNoGrad(double[,] x)
    {
        int b = x.GetLength(0);
        (double[] mean, double[] var) = _gpr.Posterior(x);
        var u = new double[b];
        for (int i = 0; i < b; i++)
            u[i] = (mean[i] - _threshold) / Math.Sqrt(var[i] + _stabilizingNoise);
        return TruncNorm.LogNdtr(u);
    }

    /// <summary><c>eval_acqf_no_grad</c> at a single point — the discrete searches.</summary>
    /// <param name="x">Normalized input coordinates in search-space order; rectangular matrices contain one point per row.</param>
    /// <returns>The acquisition value at the supplied point, or one value per input row for a batch.</returns>
    public double EvalAcqfNoGrad(double[] x)
    {
        int dim = x.Length;
        var row = new double[1, dim];
        for (int d = 0; d < dim; d++)
            row[0, d] = x[d];
        return EvalAcqfNoGrad(row)[0];
    }

    /// <summary>
    ///     <c>eval_acqf_with_grad</c> — value and the hand-derived gradient. With
    ///     <c>u = (mean − threshold)/σ</c>, <c>d log_ndtr(u)/du = φ(u)/Φ(u) =
    ///     exp(norm_logpdf(u) − log_ndtr(u))</c>, chained through the posterior gradient
    ///     (<c>∂u/∂x = ∂mean/∂x / σ − u·∂σ/∂x / σ</c>, <c>∂σ/∂x = ∂var/∂x / (2σ)</c>).
    /// </summary>
    /// <param name="x">Normalized input coordinates in search-space order; rectangular matrices contain one point per row.</param>
    /// <returns>The acquisition value and its gradient in normalized parameter coordinates.</returns>
    public (double Value, double[] Grad) EvalAcqfWithGrad(double[] x)
    {
        int dim = x.Length;
        (double mean, double var, double[] dMean, double[] dVar, bool _) = _gpr.PosteriorWithGrad(x);
        double sigma = Math.Sqrt(var + _stabilizingNoise);
        double u = (mean - _threshold) / sigma;
        double value = TruncNorm.LogNdtr(new[] { u })[0];
        double dLogNdtr = Math.Exp(TruncNorm.NormLogPdf(u) - value);   // φ(u)/Φ(u)

        var grad = new double[dim];
        for (int d = 0; d < dim; d++)
        {
            double dSigma = dVar[d] / (2.0 * sigma);
            double du = dMean[d] / sigma - u * dSigma / sigma;
            grad[d] = dLogNdtr * du;
        }
        return (value, grad);
    }
}

/// <summary>
///     The log constrained-expected-improvement acquisition function — port of <c>acqf.LogCEI</c>, the
///     acqf <see cref="OptunaSharp.Samplers.GPSampler" /> maximizes for single-objective <b>constrained</b> optimization.
///     It assumes independence between the objective and each constraint, so the log feasible
///     improvement is the objective log-EI plus the summed log feasible-probabilities:
///     <c>LogCEI(x) = LogEI(x) + Σ_c LogPI_c(x)</c>. (When there is no feasible trial the objective
///     <c>threshold</c> is <c>−∞</c> and the LogEI term is 0, leaving only the constraint terms.)
///
///     <para><b>Scope:</b> the no-running-trials path — <c>qLogCEI</c> (with running trials) is
///     deferred to the q-batch part.</para>
/// </summary>
internal sealed class LogCEI : IAcquisitionFunc
{
    private readonly LogEI _acqf;
    private readonly LogPI[] _constraintsAcqfList;

    /// <summary>Configures log constrained expected improvement from the fitted model, search space, and supplied thresholds or sampling settings.</summary>
    /// <param name="gpr">The Gaussian-process model supplying posterior predictions and kernel length scales.</param>
    /// <param name="searchSpace">The normalized coordinate layout, bounds, and discrete choices used by the acquisition optimizer.</param>
    /// <param name="threshold">The incumbent objective value that a candidate must improve upon, in the model's maximization convention.</param>
    /// <param name="constraintsGprList">One Gaussian-process model per modeled constraint.</param>
    /// <param name="constraintsThresholdList">The feasibility threshold for each constraint model, in matching order.</param>
    /// <param name="stabilizingNoise">A small variance increment used to avoid singular predictive standard deviations.</param>
    /// <exception cref="System.ArgumentException">The constraint-model and threshold lists are empty or have different lengths.</exception>
    internal LogCEI(
        GPRegressor gpr, GpSearchSpace searchSpace, double threshold,
        IReadOnlyList<GPRegressor> constraintsGprList, IReadOnlyList<double> constraintsThresholdList,
        double stabilizingNoise = 1e-12)
    {
        if (constraintsGprList.Count != constraintsThresholdList.Count || constraintsGprList.Count == 0)
            throw new ArgumentException("constraints_gpr_list and constraints_threshold_list must be non-empty and equal-length.");

        _acqf = new LogEI(gpr, searchSpace, threshold, stabilizingNoise);
        _constraintsAcqfList = new LogPI[constraintsGprList.Count];
        for (int i = 0; i < constraintsGprList.Count; i++)
            _constraintsAcqfList[i] = new LogPI(
                constraintsGprList[i], searchSpace, constraintsThresholdList[i], null, stabilizingNoise);

        SearchSpace = searchSpace;
        LengthScales = gpr.LengthScales;
    }

    /// <summary>The normalized parameter space over which this acquisition function is optimized.</summary>
    public GpSearchSpace SearchSpace { get; }
    /// <summary>The kernel length scale of each coordinate, used to precondition gradient steps.</summary>
    public double[] LengthScales { get; }

    /// <summary><c>eval_acqf_no_grad</c> — <c>LogEI(x) + Σ LogPI_c(x)</c> over a batch.</summary>
    /// <param name="x">Normalized input coordinates in search-space order; rectangular matrices contain one point per row.</param>
    /// <returns>The acquisition value at the supplied point, or one value per input row for a batch.</returns>
    public double[] EvalAcqfNoGrad(double[,] x)
    {
        double[] result = _acqf.EvalAcqfNoGrad(x);
        foreach (LogPI c in _constraintsAcqfList)
        {
            double[] cv = c.EvalAcqfNoGrad(x);
            for (int i = 0; i < result.Length; i++)
                result[i] += cv[i];
        }
        return result;
    }

    /// <summary><c>eval_acqf_no_grad</c> at a single point.</summary>
    /// <param name="x">Normalized input coordinates in search-space order; rectangular matrices contain one point per row.</param>
    /// <returns>The acquisition value at the supplied point, or one value per input row for a batch.</returns>
    public double EvalAcqfNoGrad(double[] x)
    {
        double v = _acqf.EvalAcqfNoGrad(x);
        foreach (LogPI c in _constraintsAcqfList)
            v += c.EvalAcqfNoGrad(x);
        return v;
    }

    /// <summary><c>eval_acqf_with_grad</c> — the summed value and gradient of the objective + constraint terms.</summary>
    /// <param name="x">Normalized input coordinates in search-space order; rectangular matrices contain one point per row.</param>
    /// <returns>The acquisition value and its gradient in normalized parameter coordinates.</returns>
    public (double Value, double[] Grad) EvalAcqfWithGrad(double[] x)
    {
        (double value, double[] grad) = _acqf.EvalAcqfWithGrad(x);
        foreach (LogPI c in _constraintsAcqfList)
        {
            (double cv, double[] cg) = c.EvalAcqfWithGrad(x);
            value += cv;
            for (int d = 0; d < grad.Length; d++)
                grad[d] += cg[d];
        }
        return (value, grad);
    }
}

/// <summary>
///     The log expected-hypervolume-improvement acquisition function — port of <c>acqf.LogEHVI</c>, the
///     acqf <see cref="OptunaSharp.Samplers.GPSampler" /> maximizes for <b>multi-objective</b> optimization (PORTING_PLAN
///     Phase 3.2 part 3e-4). It is a Sobol'-QMC Monte-Carlo estimate (Eq. (1) of arxiv 2006.05078) of
///     the expected improvement in dominated hypervolume: the non-dominated region below the current
///     Pareto front (and dominated by a reference point) is decomposed into boxes
///     (<see cref="Hypervolume.GetNonDominatedBoxBounds" />), and for a fixed set of standard-normal
///     samples (<see cref="Qmc.SampleFromNormalSobol" />) the log expected HVI is a log-sum-exp over
///     boxes and samples.
///
///     <para><b>Deterministic and differentiable:</b> the QMC samples are fixed at construction, so the
///     acqf is a smooth function of <c>x</c> optimizable by the gradient ascent. The hand-derived
///     gradient (the autograd replacement) chains the box-clamped <c>log</c>/<c>logsumexp</c> softmax
///     weights through each objective's posterior mean/variance gradient. Tolerance-exact vs Optuna (the
///     posterior rides the managed triangular solve; the QMC samples match torch to ~1e-15) — the
///     documented GP boundary, not bit-for-bit.</para>
///
///     <para><b>Running trials (Kriging-Believer):</b> when
///     <c>normalizedParamsOfRunningTrials</c> is given, each objective GP is extended with the running
///     points (their posterior mean as the fantasy value) via
///     <see cref="GPRegressor.AppendRunningData" />, so the posterior accounts for the trials already in
///     flight (Optuna's <c>append_running_data</c> path, PORTING_PLAN Phase 3.2 part 3e-6). The box
///     decomposition still uses the completed-trial <c>Y_train</c>. Until Optuna 5.0.0 the
///     MO-constrained <see cref="LogCEHVI" /> used this path for running trials; 5.0.0 (#6804) routes
///     that case to <see cref="QLogCEHVI" />, so the sampler no longer passes running trials here (the
///     parameter stays, as in Optuna's class, and it mutates the supplied GPs when used).</para>
/// </summary>
internal sealed class LogEHVI : IAcquisitionFunc
{
    private const double Eps = 1e-12;

    private readonly GPRegressor[] _gprList;
    private readonly double _stabilizingNoise;
    private readonly double[,] _fixedSamples;   // (n_qmc, n_obj) standard-normal QMC base samples
    private readonly double[,] _lowerBounds;    // (n_boxes, n_obj) box lower bounds (maximization space)
    private readonly double[,] _intervals;      // (n_boxes, n_obj) box widths, clamped ≥ EPS
    private readonly int _nObjectives;
    private readonly int _nQmc;
    private readonly int _nBoxes;

    /// <summary>Configures log expected hypervolume improvement from the fitted model, search space, and supplied thresholds or sampling settings.</summary>
    /// <param name="gprList">One Gaussian-process model per objective, in objective order.</param>
    /// <param name="searchSpace">The normalized coordinate layout, bounds, and discrete choices used by the acquisition optimizer.</param>
    /// <param name="yTrain">The observed objective values used to train or condition the model.</param>
    /// <param name="nQmcSamples">The number of fixed quasi-Monte Carlo draws used to approximate posterior expectations.</param>
    /// <param name="qmcSeed">The seed used to scramble the quasi-Monte Carlo normal samples.</param>
    /// <param name="normalizedParamsOfRunningTrials">The normalized parameter rows of running trials to include as pending candidates.</param>
    /// <param name="stabilizingNoise">A small variance increment used to avoid singular predictive standard deviations.</param>
    internal LogEHVI(
        GPRegressor[] gprList, GpSearchSpace searchSpace, double[,] yTrain,
        int nQmcSamples, int qmcSeed, double[,]? normalizedParamsOfRunningTrials = null,
        double stabilizingNoise = 1e-12)
    {
        _gprList = gprList;
        _stabilizingNoise = stabilizingNoise;
        _nObjectives = yTrain.GetLength(1);

        if (normalizedParamsOfRunningTrials != null)
            // Kriging Believer: append the running points with each objective's posterior mean as the
            // fantasy value, mutating the objective GPs in place (matching Optuna's LogEHVI).
            foreach (GPRegressor gpr in gprList)
                gpr.AppendRunningData(
                    normalizedParamsOfRunningTrials, gpr.Posterior(normalizedParamsOfRunningTrials).Mean);

        _fixedSamples = Qmc.SampleFromNormalSobol(_nObjectives, nQmcSamples, qmcSeed);
        _nQmc = _fixedSamples.GetLength(0);

        double[] refPoint = GetReferencePoint(yTrain);
        (double[,] lower, double[,] upper) = GetNonDominatedBoxBoundsMax(yTrain, refPoint);
        _lowerBounds = lower;
        _nBoxes = lower.GetLength(0);
        _intervals = new double[_nBoxes, _nObjectives];
        for (int b = 0; b < _nBoxes; b++)
            for (int i = 0; i < _nObjectives; i++)
                _intervals[b, i] = Math.Max(Eps, upper[b, i] - lower[b, i]);

        // length_scales = mean over objectives of each GP's per-parameter length scales (preconditioning).
        SearchSpace = searchSpace;
        LengthScales = MeanLengthScales(_gprList, searchSpace.Dim);
    }

    /// <summary>The normalized parameter space over which this acquisition function is optimized.</summary>
    public GpSearchSpace SearchSpace { get; }
    /// <summary>The kernel length scale of each coordinate, used to precondition gradient steps.</summary>
    public double[] LengthScales { get; }

    /// <summary><c>eval_acqf_no_grad</c> over a batch <c>(B, dim)</c>.</summary>
    /// <param name="x">Normalized input coordinates in search-space order; rectangular matrices contain one point per row.</param>
    /// <returns>The acquisition value at the supplied point, or one value per input row for a batch.</returns>
    public double[] EvalAcqfNoGrad(double[,] x)
    {
        int b = x.GetLength(0);
        int dim = x.GetLength(1);
        var result = new double[b];
        var row = new double[dim];
        for (int r = 0; r < b; r++)
        {
            for (int d = 0; d < dim; d++)
                row[d] = x[r, d];
            result[r] = EvalAcqfNoGrad(row);
        }
        return result;
    }

    /// <summary><c>eval_acqf_no_grad</c> at a single point.</summary>
    /// <param name="x">Normalized input coordinates in search-space order; rectangular matrices contain one point per row.</param>
    /// <returns>The acquisition value at the supplied point, or one value per input row for a batch.</returns>
    public double EvalAcqfNoGrad(double[] x)
    {
        // Y_post[s, i] = mean_i + stdev_i * fixed_samples[s, i].
        var yPost = new double[_nQmc, _nObjectives];
        for (int i = 0; i < _nObjectives; i++)
        {
            (double mean, double var) = PosteriorAt(_gprList[i], x);
            double stdev = Math.Sqrt(var + _stabilizingNoise);
            for (int s = 0; s < _nQmc; s++)
                yPost[s, i] = mean + stdev * _fixedSamples[s, i];
        }
        return LogEhviFromYPost(yPost);
    }

    /// <summary>
    ///     <c>eval_acqf_with_grad</c> — value and the hand-derived gradient. For each sample <c>s</c> and
    ///     box <c>b</c>, <c>L[s,b] = Σ_i log(clamp(Y_post[s,i] − lower[b,i], EPS, interval[b,i]))</c>;
    ///     the value is <c>logsumexp_{s,b} L − log(n_qmc)</c>, and the gradient weights each term by the
    ///     softmax <c>exp(L[s,b] − value − log n_qmc)</c>... (see body) chained through
    ///     <c>∂Y_post/∂x = ∂mean/∂x + fixed·∂stdev/∂x</c>, with the clamp zeroing the out-of-range terms.
    /// </summary>
    /// <param name="x">Normalized input coordinates in search-space order; rectangular matrices contain one point per row.</param>
    /// <returns>The acquisition value and its gradient in normalized parameter coordinates.</returns>
    public (double Value, double[] Grad) EvalAcqfWithGrad(double[] x)
    {
        int dim = x.Length;

        // Per-objective posterior mean/var and their gradients.
        var mean = new double[_nObjectives];
        var stdev = new double[_nObjectives];
        var dMean = new double[_nObjectives][];
        var dStdev = new double[_nObjectives][];
        for (int i = 0; i < _nObjectives; i++)
        {
            (double m, double v, double[] dm, double[] dv, bool _) = _gprList[i].PosteriorWithGrad(x);
            double sd = Math.Sqrt(v + _stabilizingNoise);
            mean[i] = m;
            stdev[i] = sd;
            dMean[i] = dm;
            var ds = new double[dim];
            for (int d = 0; d < dim; d++)
                ds[d] = dv[d] / (2.0 * sd);   // ∂stdev/∂x = ∂var/∂x / (2·stdev)
            dStdev[i] = ds;
        }

        // Forward: L[s,b] and the logsumexp.
        var yPost = new double[_nQmc, _nObjectives];
        for (int i = 0; i < _nObjectives; i++)
            for (int s = 0; s < _nQmc; s++)
                yPost[s, i] = mean[i] + stdev[i] * _fixedSamples[s, i];
        double logNQmc = Math.Log(_nQmc);
        var l = new double[_nQmc * _nBoxes];
        var clampMask = new bool[_nQmc, _nBoxes, _nObjectives];
        var invDiff = new double[_nQmc, _nBoxes, _nObjectives];
        int flat = 0;
        double maxL = double.NegativeInfinity;
        for (int s = 0; s < _nQmc; s++)
            for (int b = 0; b < _nBoxes; b++)
            {
                double sum = 0.0;
                for (int i = 0; i < _nObjectives; i++)
                {
                    double raw = yPost[s, i] - _lowerBounds[b, i];
                    double interval = _intervals[b, i];
                    double diff = raw < Eps ? Eps : (raw > interval ? interval : raw);
                    bool inRange = raw > Eps && raw < interval;   // clamp_ zeroes the gradient at both bounds
                    clampMask[s, b, i] = inRange;
                    invDiff[s, b, i] = 1.0 / diff;
                    sum += Math.Log(diff);
                }
                l[flat++] = sum;
                if (sum > maxL) maxL = sum;
            }

        double expSum = 0.0;
        for (int k = 0; k < l.Length; k++)
            expSum += Math.Exp(l[k] - maxL);
        double value = maxL + Math.Log(expSum) - logNQmc;

        // Backward: grad[d] = Σ_{s,b} softmax(L)[s,b] · Σ_i mask · invDiff · (dMean_i[d] + fixed[s,i]·dStdev_i[d]).
        var grad = new double[dim];
        flat = 0;
        for (int s = 0; s < _nQmc; s++)
            for (int b = 0; b < _nBoxes; b++)
            {
                double w = Math.Exp(l[flat++] - maxL) / expSum;   // softmax over (s,b)
                if (w == 0.0)
                    continue;
                for (int i = 0; i < _nObjectives; i++)
                {
                    if (!clampMask[s, b, i])
                        continue;
                    double factor = w * invDiff[s, b, i];
                    double fixedSi = _fixedSamples[s, i];
                    double[] dm = dMean[i];
                    double[] ds = dStdev[i];
                    for (int d = 0; d < dim; d++)
                        grad[d] += factor * (dm[d] + fixedSi * ds[d]);
                }
            }
        return (value, grad);
    }

    /// <summary><c>logehvi</c> — the log-sum-exp over boxes/samples of the box-clamped log volumes, given <c>Y_post</c> (n_qmc, n_obj).</summary>
    /// <param name="yPost">Posterior objective draws with rows for QMC samples and columns for pending/query points.</param>
    /// <returns>The logarithm of the mean hypervolume improvement over posterior samples.</returns>
    private double LogEhviFromYPost(double[,] yPost)
    {
        double logNQmc = Math.Log(_nQmc);
        var l = new double[_nQmc * _nBoxes];
        int flat = 0;
        double maxL = double.NegativeInfinity;
        for (int s = 0; s < _nQmc; s++)
            for (int b = 0; b < _nBoxes; b++)
            {
                double sum = 0.0;
                for (int i = 0; i < _nObjectives; i++)
                {
                    double raw = yPost[s, i] - _lowerBounds[b, i];
                    double interval = _intervals[b, i];
                    double diff = raw < Eps ? Eps : (raw > interval ? interval : raw);
                    sum += Math.Log(diff);
                }
                l[flat++] = sum;
                if (sum > maxL) maxL = sum;
            }
        double expSum = 0.0;
        for (int k = 0; k < l.Length; k++)
            expSum += Math.Exp(l[k] - maxL);
        return maxL + Math.Log(expSum) - logNQmc;
    }

    /// <summary>Returns the scalar posterior mean and variance at the normalized query point.</summary>
    /// <param name="gpr">The Gaussian-process model supplying posterior predictions and kernel length scales.</param>
    /// <param name="x">Normalized input coordinates in search-space order; rectangular matrices contain one point per row.</param>
    /// <returns>The scalar posterior mean and variance at the normalized query point.</returns>
    private static (double Mean, double Var) PosteriorAt(GPRegressor gpr, double[] x)
    {
        int dim = x.Length;
        var row = new double[1, dim];
        for (int d = 0; d < dim; d++)
            row[0, d] = x[d];
        (double[] mean, double[] var) = gpr.Posterior(row);
        return (mean[0], var[0]);
    }

    /// <summary>
    ///     <c>acqf._get_reference_point(Y)</c> — <c>Y</c> is to be maximized, so <c>loss = −Y</c> is
    ///     minimized: <c>ref = nextafter(max(1.1·m, 0.9·m), +∞)</c> with <c>m = max_row(−Y)</c>.
    /// </summary>
    /// <param name="yTrain">The observed objective values used to train or condition the model.</param>
    /// <returns>A reference vector worse than the observed objective extremes in the operation's coordinate convention.</returns>
    internal static double[] GetReferencePoint(double[,] yTrain)
    {
        int n = yTrain.GetLength(0);
        int m = yTrain.GetLength(1);
        var refPoint = new double[m];
        for (int j = 0; j < m; j++)
        {
            double mx = double.NegativeInfinity;
            for (int r = 0; r < n; r++)
            {
                double loss = -yTrain[r, j];
                if (loss > mx) mx = loss;
            }
            refPoint[j] = Internal.MathUtil.NextAfter(Math.Max(1.1 * mx, 0.9 * mx), double.PositiveInfinity);
        }
        return refPoint;
    }

    /// <summary>
    ///     <c>acqf._get_non_dominated_box_bounds(Y, ref_point)</c> — in maximization space: filter the
    ///     losses that dominate the reference point, take their Pareto front, decompose, and flip the
    ///     signs back (<c>lower = −ubs</c>, <c>upper = −lbs</c>).
    /// </summary>
    /// <param name="yTrain">The observed objective values used to train or condition the model.</param>
    /// <param name="refPoint">The reference objective vector bounding the dominated region.</param>
    /// <returns>Lower and upper corner matrices for nondominated boxes in maximization coordinates.</returns>
    internal static (double[,] Lower, double[,] Upper) GetNonDominatedBoxBoundsMax(double[,] yTrain, double[] refPoint)
    {
        int n = yTrain.GetLength(0);
        int m = yTrain.GetLength(1);
        // loss_vals = -Y; keep rows with all(loss < ref_point).
        var kept = new System.Collections.Generic.List<double[]>();
        for (int r = 0; r < n; r++)
        {
            var loss = new double[m];
            bool all = true;
            for (int j = 0; j < m; j++)
            {
                loss[j] = -yTrain[r, j];
                if (!(loss[j] < refPoint[j])) all = false;
            }
            if (all) kept.Add(loss);
        }
        double[][] lossVals = kept.ToArray();
        bool[] onFront = Internal.MultiObjective.IsParetoFront(lossVals, assumeUniqueLexsorted: false);
        var paretoSols = new System.Collections.Generic.List<double[]>();
        for (int r = 0; r < lossVals.Length; r++)
            if (onFront[r]) paretoSols.Add(lossVals[r]);

        (double[][] lbs, double[][] ubs) =
            Internal.Hypervolume.GetNonDominatedBoxBounds(paretoSols.ToArray(), refPoint);

        int nBoxes = lbs.Length;
        var lower = new double[nBoxes, m];
        var upper = new double[nBoxes, m];
        for (int b = 0; b < nBoxes; b++)
            for (int j = 0; j < m; j++)
            {
                lower[b, j] = -ubs[b][j];   // flip back to maximization
                upper[b, j] = -lbs[b][j];
            }
        return (lower, upper);
    }

    /// <summary>Mean over the objective GPs of each per-parameter length scale (the acqf-optimizer preconditioning).</summary>
    /// <param name="gprList">One Gaussian-process model per objective, in objective order.</param>
    /// <param name="dim">The number of coordinates in each point.</param>
    /// <returns>The mean kernel length scale across objective models for each coordinate.</returns>
    internal static double[] MeanLengthScales(GPRegressor[] gprList, int dim)
    {
        var ls = new double[dim];
        for (int d = 0; d < dim; d++)
        {
            double s = 0.0;
            foreach (GPRegressor gpr in gprList)
                s += gpr.LengthScales[d];
            ls[d] = s / gprList.Length;
        }
        return ls;
    }
}

/// <summary>
///     The q-batch log expected-hypervolume-improvement acquisition function — port of
///     <c>acqf.qLogEHVI</c>, the acqf <see cref="OptunaSharp.Samplers.GPSampler" /> maximizes for <b>multi-objective</b>
///     optimization <b>with running trials</b> (PORTING_PLAN Phase 3.2 part 3e-6). Unlike
///     <see cref="LogEHVI" /> (independent per-objective sampling + one box decomposition), each QMC
///     draw gets its <b>own</b> box decomposition — computed from that draw's running fantasies
///     (<see cref="ListConditionalGPRegressor.GetFantasySamples" />) appended to <c>Y_train</c> — and the
///     per-objective conditional query sample (<see cref="ListConditionalGPRegressor.SampleCandidatePosterior" />).
///     Per draw the log HVI is a <c>logsumexp</c> over that draw's boxes; the acqf is the log-mean-exp
///     of those per-sample HVIs (Eq. (3) of arxiv 2006.05078). One conditional GP per objective, each
///     seeded <c>qmc_seed + i</c>.
///
///     <para><b>Structure (Optuna 5.0.0, #6804):</b> the conditional regressors live in a
///     <see cref="ListConditionalGPRegressor" /> and the per-draw decomposition in a
///     <see cref="PerSampleParetoBaseline" />, both shared with <see cref="QLogCEHVI" />. Upstream's
///     refactor left the values unchanged. On the C# side the shared baseline also brings in Optuna's
///     <c>pad_sequence</c> padding of the per-draw box lists (every missing box adds
///     <c>EPS^n_objectives</c> to that draw's HVI), which the earlier per-draw layout here had skipped;
///     the effect is only visible where a draw's real HVI is itself near <c>EPS^n_objectives</c>.</para>
///
///     <para><b>Gradient</b> (the autograd replacement): the fantasies (hence the boxes) are constant,
///     so the gradient flows only through the query samples; it chains a nested softmax (over the boxes
///     within a sample, then over the samples) times the two-sided box-clamp mask times each objective's
///     conditional-sample gradient.</para>
/// </summary>
internal sealed class QLogEHVI : IAcquisitionFunc
{
    private readonly ListConditionalGPRegressor _condGpr;
    private readonly PerSampleParetoBaseline _baseline;

    /// <summary>Configures batch log expected hypervolume improvement from the fitted model, search space, and supplied thresholds or sampling settings.</summary>
    /// <param name="gprList">One Gaussian-process model per objective, in objective order.</param>
    /// <param name="searchSpace">The normalized coordinate layout, bounds, and discrete choices used by the acquisition optimizer.</param>
    /// <param name="yTrain">The observed objective values used to train or condition the model.</param>
    /// <param name="nQmcSamples">The number of fixed quasi-Monte Carlo draws used to approximate posterior expectations.</param>
    /// <param name="qmcSeed">The seed used to scramble the quasi-Monte Carlo normal samples.</param>
    /// <param name="normalizedParamsOfRunningTrials">The normalized parameter rows of running trials to include as pending candidates.</param>
    /// <param name="stabilizingNoise">A small variance increment used to avoid singular predictive standard deviations.</param>
    internal QLogEHVI(
        GPRegressor[] gprList, GpSearchSpace searchSpace, double[,] yTrain, int nQmcSamples, int qmcSeed,
        double[,] normalizedParamsOfRunningTrials, double stabilizingNoise = 1e-12)
    {
        _condGpr = new ListConditionalGPRegressor(
            gprList, normalizedParamsOfRunningTrials, nQmcSamples, qmcSeed, stabilizingNoise);
        // A per-QMC-draw box decomposition over Y_train ∪ {this draw's fantasies}, one shared ref point.
        _baseline = new PerSampleParetoBaseline(yTrain, _condGpr.GetFantasySamples());

        SearchSpace = searchSpace;
        LengthScales = LogEHVI.MeanLengthScales(gprList, searchSpace.Dim);
    }

    /// <summary>The normalized parameter space over which this acquisition function is optimized.</summary>
    public GpSearchSpace SearchSpace { get; }
    /// <summary>The kernel length scale of each coordinate, used to precondition gradient steps.</summary>
    public double[] LengthScales { get; }

    /// <summary>
    ///     <c>compute_per_sample_log_utility(x)</c> — the candidate's log hypervolume improvement over each
    ///     QMC draw's baseline, shape <c>(n_qmc,)</c>.
    /// </summary>
    /// <param name="x">The candidate point in normalized search-space coordinates.</param>
    /// <returns>The per-draw log HVI of the candidate.</returns>
    internal double[] ComputePerSampleLogUtility(double[] x)
        => _baseline.ComputePerSampleLogHvi(_condGpr.SampleCandidatePosterior(x));

    /// <summary><c>eval_acqf_no_grad</c> over a batch <c>(B, dim)</c>.</summary>
    /// <param name="x">Normalized input coordinates in search-space order; rectangular matrices contain one point per row.</param>
    /// <returns>The acquisition value at the supplied point, or one value per input row for a batch.</returns>
    public double[] EvalAcqfNoGrad(double[,] x)
    {
        int b = x.GetLength(0), dim = x.GetLength(1);
        var result = new double[b];
        var row = new double[dim];
        for (int r = 0; r < b; r++)
        {
            for (int d = 0; d < dim; d++) row[d] = x[r, d];
            result[r] = EvalAcqfNoGrad(row);
        }
        return result;
    }

    /// <summary><c>eval_acqf_no_grad</c> at a single point.</summary>
    /// <param name="x">Normalized input coordinates in search-space order; rectangular matrices contain one point per row.</param>
    /// <returns>The acquisition value at the supplied point, or one value per input row for a batch.</returns>
    public double EvalAcqfNoGrad(double[] x) => Acqf.LogMeanExp(ComputePerSampleLogUtility(x));

    /// <summary><c>eval_acqf_with_grad</c> — value and the hand-derived nested-softmax gradient.</summary>
    /// <param name="x">Normalized input coordinates in search-space order; rectangular matrices contain one point per row.</param>
    /// <returns>The acquisition value and its gradient in normalized parameter coordinates.</returns>
    public (double Value, double[] Grad) EvalAcqfWithGrad(double[] x)
    {
        (double[,] samples, double[][,] dSamples) = _condGpr.SampleCandidatePosteriorWithGrad(x);
        (double[] hvi, double[,] dHvi) = _baseline.ComputePerSampleLogHviWithGrad(samples, dSamples);
        double logsumexpS = Acqf.LogSumExp(hvi);
        double value = logsumexpS - Math.Log(hvi.Length);

        // grad[d] = Σ_s softmax_s(hvi) · ∂hvi_s/∂x_d — the outer (over-draws) softmax of the nested pair;
        // the inner softmax over each draw's boxes is already inside ∂hvi_s/∂x_d.
        int dim = x.Length;
        var grad = new double[dim];
        for (int s = 0; s < hvi.Length; s++)
        {
            double w = Math.Exp(hvi[s] - logsumexpS);
            if (w == 0.0) continue;
            for (int d = 0; d < dim; d++)
                grad[d] += w * dHvi[s, d];
        }
        return (value, grad);
    }
}

/// <summary>
///     The log constrained-expected-hypervolume-improvement acquisition function — port of
///     <c>acqf.LogCEHVI</c>, the acqf <see cref="OptunaSharp.Samplers.GPSampler" /> maximizes for <b>multi-objective
///     constrained</b> optimization (PORTING_PLAN Phase 3.2 part 3e-5). Assuming independence between
///     the objectives and each constraint, it is the log expected hypervolume improvement over the
///     <b>feasible</b> trials plus the summed log feasible-probabilities:
///     <c>LogCEHVI(x) = LogEHVI_feasible(x) + Σ_c LogPI_c(x)</c>. When there is no feasible trial the
///     <see cref="LogEHVI" /> term is dropped and only the constraint terms remain (the optimizer then
///     drives toward feasibility).
///
///     <para><b>Sequential only (Optuna 5.0.0, #6804):</b> this acqf no longer takes running trials.
///     Until Optuna 5.0.0 the running points were forwarded here and appended as Kriging-Believer
///     fantasies; 5.0.0 removed that parameter and <see cref="OptunaSharp.Samplers.GPSampler" /> now
///     scores the multi-objective constrained running case with the q-batch <see cref="QLogCEHVI" />.
///     The inner <see cref="LogEHVI" /> and <see cref="LogPI" /> are built without running trials (both
///     still accept them, exactly like their Python counterparts, but nothing in the sampler passes
///     them any more).</para>
/// </summary>
internal sealed class LogCEHVI : IAcquisitionFunc
{
    private readonly LogEHVI? _acqf;
    private readonly LogPI[] _constraintsAcqfList;

    /// <summary>Configures log constrained expected hypervolume improvement from the fitted model, search space, and supplied thresholds or sampling settings.</summary>
    /// <param name="gprList">One Gaussian-process model per objective, in objective order.</param>
    /// <param name="searchSpace">The normalized coordinate layout, bounds, and discrete choices used by the acquisition optimizer.</param>
    /// <param name="yFeasible">The objective rows of feasible completed observations, or <c>null</c> when no completed trial is feasible.</param>
    /// <param name="nQmcSamples">The number of fixed quasi-Monte Carlo draws used to approximate posterior expectations.</param>
    /// <param name="qmcSeed">The seed used to scramble the quasi-Monte Carlo normal samples.</param>
    /// <param name="constraintsGprList">One Gaussian-process model per modeled constraint.</param>
    /// <param name="constraintsThresholdList">The feasibility threshold for each constraint model, in matching order.</param>
    /// <param name="stabilizingNoise">A small variance increment used to avoid singular predictive standard deviations.</param>
    /// <exception cref="System.ArgumentException">The constraint-model and threshold lists are empty or have different lengths.</exception>
    internal LogCEHVI(
        GPRegressor[] gprList, GpSearchSpace searchSpace, double[,]? yFeasible,
        int nQmcSamples, int qmcSeed,
        IReadOnlyList<GPRegressor> constraintsGprList, IReadOnlyList<double> constraintsThresholdList,
        double stabilizingNoise = 1e-12)
    {
        if (constraintsGprList.Count != constraintsThresholdList.Count || constraintsGprList.Count == 0)
            throw new ArgumentException("constraints_gpr_list and constraints_threshold_list must be non-empty and equal-length.");

        // Optuna 5.0.0 passes None for the running trials to both inner acqfs: no Kriging-Believer
        // fantasies are appended, so the constraint and objective GPs are left unmutated.
        _acqf = yFeasible != null
            ? new LogEHVI(gprList, searchSpace, yFeasible, nQmcSamples, qmcSeed,
                null, stabilizingNoise)
            : null;
        _constraintsAcqfList = new LogPI[constraintsGprList.Count];
        for (int i = 0; i < constraintsGprList.Count; i++)
            _constraintsAcqfList[i] = new LogPI(
                constraintsGprList[i], searchSpace, constraintsThresholdList[i],
                null, stabilizingNoise);

        SearchSpace = searchSpace;
        LengthScales = LogEHVI.MeanLengthScales(gprList, searchSpace.Dim);
    }

    /// <summary>The normalized parameter space over which this acquisition function is optimized.</summary>
    public GpSearchSpace SearchSpace { get; }
    /// <summary>The kernel length scale of each coordinate, used to precondition gradient steps.</summary>
    public double[] LengthScales { get; }

    /// <summary><c>eval_acqf_no_grad</c> — <c>Σ LogPI_c(x)</c> (+ <c>LogEHVI(x)</c> when there is a feasible term) over a batch.</summary>
    /// <param name="x">Normalized input coordinates in search-space order; rectangular matrices contain one point per row.</param>
    /// <returns>The acquisition value at the supplied point, or one value per input row for a batch.</returns>
    public double[] EvalAcqfNoGrad(double[,] x)
    {
        int b = x.GetLength(0);
        var result = new double[b];
        foreach (LogPI c in _constraintsAcqfList)
        {
            double[] cv = c.EvalAcqfNoGrad(x);
            for (int i = 0; i < b; i++)
                result[i] += cv[i];
        }
        if (_acqf != null)
        {
            double[] ev = _acqf.EvalAcqfNoGrad(x);
            for (int i = 0; i < b; i++)
                result[i] += ev[i];
        }
        return result;
    }

    /// <summary><c>eval_acqf_no_grad</c> at a single point.</summary>
    /// <param name="x">Normalized input coordinates in search-space order; rectangular matrices contain one point per row.</param>
    /// <returns>The acquisition value at the supplied point, or one value per input row for a batch.</returns>
    public double EvalAcqfNoGrad(double[] x)
    {
        double v = 0.0;
        foreach (LogPI c in _constraintsAcqfList)
            v += c.EvalAcqfNoGrad(x);
        if (_acqf != null)
            v += _acqf.EvalAcqfNoGrad(x);
        return v;
    }

    /// <summary><c>eval_acqf_with_grad</c> — the summed value and gradient of the constraint terms (and the feasible EHVI term when present).</summary>
    /// <param name="x">Normalized input coordinates in search-space order; rectangular matrices contain one point per row.</param>
    /// <returns>The acquisition value and its gradient in normalized parameter coordinates.</returns>
    public (double Value, double[] Grad) EvalAcqfWithGrad(double[] x)
    {
        double value = 0.0;
        var grad = new double[x.Length];
        foreach (LogPI c in _constraintsAcqfList)
        {
            (double cv, double[] cg) = c.EvalAcqfWithGrad(x);
            value += cv;
            for (int d = 0; d < grad.Length; d++)
                grad[d] += cg[d];
        }
        if (_acqf != null)
        {
            (double ev, double[] eg) = _acqf.EvalAcqfWithGrad(x);
            value += ev;
            for (int d = 0; d < grad.Length; d++)
                grad[d] += eg[d];
        }
        return (value, grad);
    }
}

/// <summary>
///     The q-batch constrained log expected-hypervolume-improvement acquisition function — port of
///     <c>acqf.qLogCEHVI</c> (added in Optuna 5.0.0 by #6804), the acqf
///     <see cref="OptunaSharp.Samplers.GPSampler" /> maximizes for <b>multi-objective constrained</b>
///     optimization <b>with running trials</b>. It replaces the Kriging-Believer <see cref="LogCEHVI" />
///     that handled this case before 5.0.0: rather than appending the running points' posterior means
///     as if they had been observed, it integrates over QMC fantasies of both the objectives and the
///     constraints at those points, so a running trial only shapes the baseline in the draws where it is
///     sampled as feasible.
///
///     <para><b>With a feasible baseline</b> (<c>yFeasible</c> is not null), per QMC draw <c>s</c>:
///     constraint <c>c</c> is a <see cref="QLogPI" /> seeded <c>qmcSeed + nObjectives + c</c>; a running
///     trial's objective fantasy joins draw <c>s</c>'s Pareto baseline only when every constraint's
///     fantasy for it is at or above its threshold; the baseline is <c>yFeasible</c> plus those feasible
///     fantasies (<see cref="PerSampleParetoBaseline" />, objective <c>o</c> seeded <c>qmcSeed + o</c>);
///     and the candidate scores
///     <c>L_s(x) = Σ_c logsigmoid((y_c,s(x) − t_c)/τ) + logHVI_s(Y_s(x))</c>. The acqf is
///     <c>logsumexp_s L_s − log S</c>.</para>
///
///     <para><b>With no feasible trial</b> (<c>yFeasible</c> is null): without an HVI term the tower
///     property reduces the expectation over conditional feasibility (given the running fantasies) to
///     the marginal feasibility probability, so the acqf is just <c>Σ_c LogPI_c(x)</c> — plain
///     <see cref="LogPI" /> terms with no running-trial conditioning at all.</para>
///
///     <para><b>Gradient</b> (the autograd replacement): the fantasies, the feasibility mask and the
///     boxes are constants of the candidate, so the gradient is the softmax over draws of
///     <c>Σ_c sigmoid(−arg_c)/τ · ∂y_c/∂x + ∂logHVI_s/∂x</c>. Tolerance, not bit-exact — the documented
///     GP boundary (managed triangular solves; QMC samples matched to torch ~1e-15).</para>
/// </summary>
internal sealed class QLogCEHVI : IAcquisitionFunc
{
    private readonly QLogPI[] _constraintsAcqfList;           // used only with a feasible baseline
    private readonly LogPI[] _marginalConstraintsAcqfList;    // used only when no trial is feasible
    private readonly ListConditionalGPRegressor? _condGpr;    // null exactly when _baseline is null
    private readonly PerSampleParetoBaseline? _baseline;

    /// <summary>Configures the q-batch constrained log EHVI over the running trials, choosing the feasible-baseline or marginal-feasibility form from <paramref name="yFeasible" />.</summary>
    /// <param name="gprList">One Gaussian-process model per objective, in objective order, fitted on completed trials only.</param>
    /// <param name="searchSpace">The normalized coordinate layout, bounds, and discrete choices used by the acquisition optimizer.</param>
    /// <param name="yFeasible">The objective rows of feasible completed observations, or <c>null</c> when no completed trial is feasible.</param>
    /// <param name="nQmcSamples">The number of fixed quasi-Monte Carlo draws used to approximate posterior expectations.</param>
    /// <param name="qmcSeed">The base QMC seed: objective <c>o</c> uses <c>qmcSeed + o</c>, constraint <c>c</c> uses <c>qmcSeed + nObjectives + c</c>.</param>
    /// <param name="constraintsGprList">One Gaussian-process model per modeled constraint.</param>
    /// <param name="constraintsThresholdList">The feasibility threshold for each constraint model, in matching order.</param>
    /// <param name="normalizedParamsOfRunningTrials">The normalized parameter rows of the running trials the fantasies are drawn at.</param>
    /// <param name="stabilizingNoise">A small variance increment used to avoid singular predictive standard deviations.</param>
    /// <param name="tau">The positive temperature of the smooth feasibility indicator in each constraint <see cref="QLogPI" />.</param>
    /// <exception cref="System.ArgumentException">The constraint-model and threshold lists are empty or have different lengths.</exception>
    internal QLogCEHVI(
        GPRegressor[] gprList, GpSearchSpace searchSpace, double[,]? yFeasible, int nQmcSamples, int qmcSeed,
        IReadOnlyList<GPRegressor> constraintsGprList, IReadOnlyList<double> constraintsThresholdList,
        double[,] normalizedParamsOfRunningTrials, double stabilizingNoise = 1e-12, double tau = 1e-2)
    {
        if (constraintsGprList.Count != constraintsThresholdList.Count || constraintsGprList.Count == 0)
            throw new ArgumentException("constraints_gpr_list and constraints_threshold_list must be non-empty and equal-length.");

        int nConstraints = constraintsGprList.Count;
        if (yFeasible == null)
        {
            // Marginal feasibility only: plain LogPI without running trials (see the class remarks).
            _constraintsAcqfList = Array.Empty<QLogPI>();
            _marginalConstraintsAcqfList = new LogPI[nConstraints];
            for (int c = 0; c < nConstraints; c++)
                _marginalConstraintsAcqfList[c] = new LogPI(
                    constraintsGprList[c], searchSpace, constraintsThresholdList[c], null, stabilizingNoise);
        }
        else
        {
            _marginalConstraintsAcqfList = Array.Empty<LogPI>();
            _constraintsAcqfList = new QLogPI[nConstraints];
            for (int c = 0; c < nConstraints; c++)
                _constraintsAcqfList[c] = new QLogPI(
                    constraintsGprList[c], searchSpace, constraintsThresholdList[c], nQmcSamples,
                    qmcSeed + gprList.Length + c, normalizedParamsOfRunningTrials, stabilizingNoise, tau);

            // is_fantasy_feasible[s, r] = torch.all(stack(is_fantasy_above_threshold per constraint), -1):
            // a running trial is feasible in draw s only if EVERY constraint's fantasy clears its threshold.
            bool[,] isFantasyFeasible = _constraintsAcqfList[0].IsFantasyAboveThreshold();
            for (int c = 1; c < nConstraints; c++)
            {
                bool[,] above = _constraintsAcqfList[c].IsFantasyAboveThreshold();
                for (int s = 0; s < isFantasyFeasible.GetLength(0); s++)
                    for (int r = 0; r < isFantasyFeasible.GetLength(1); r++)
                        isFantasyFeasible[s, r] &= above[s, r];
            }

            _condGpr = new ListConditionalGPRegressor(
                gprList, normalizedParamsOfRunningTrials, nQmcSamples, qmcSeed, stabilizingNoise);
            // Both the constraint masks and the objective fantasies come from the same nQmcSamples, so
            // draw s of one lines up with draw s of the other (Python zips them).
            double[][,] fantasies = _condGpr.GetFantasySamples();
            var feasibleFantasies = new double[fantasies.Length][,];
            for (int s = 0; s < fantasies.Length; s++)
                feasibleFantasies[s] = FeasibleRows(fantasies[s], isFantasyFeasible, s);
            _baseline = new PerSampleParetoBaseline(yFeasible, feasibleFantasies);
        }

        SearchSpace = searchSpace;
        LengthScales = LogEHVI.MeanLengthScales(gprList, searchSpace.Dim);
    }

    /// <summary>The normalized parameter space over which this acquisition function is optimized.</summary>
    public GpSearchSpace SearchSpace { get; }
    /// <summary>The kernel length scale of each coordinate (the mean over the objectives), used to precondition gradient steps.</summary>
    public double[] LengthScales { get; }

    /// <summary><c>eval_acqf_no_grad</c> over a batch <c>(B, dim)</c>.</summary>
    /// <param name="x">Normalized input coordinates in search-space order; rectangular matrices contain one point per row.</param>
    /// <returns>The acquisition value at the supplied point, or one value per input row for a batch.</returns>
    public double[] EvalAcqfNoGrad(double[,] x)
    {
        int b = x.GetLength(0), dim = x.GetLength(1);
        var result = new double[b];
        var row = new double[dim];
        for (int r = 0; r < b; r++)
        {
            for (int d = 0; d < dim; d++) row[d] = x[r, d];
            result[r] = EvalAcqfNoGrad(row);
        }
        return result;
    }

    /// <summary><c>eval_acqf_no_grad</c> at a single point.</summary>
    /// <param name="x">Normalized input coordinates in search-space order; rectangular matrices contain one point per row.</param>
    /// <returns>The acquisition value at the supplied point, or one value per input row for a batch.</returns>
    public double EvalAcqfNoGrad(double[] x)
    {
        if (_baseline == null)
        {
            // torch.stack([LogPI_c(x) ...]).sum(dim=0)
            double v = 0.0;
            foreach (LogPI c in _marginalConstraintsAcqfList)
                v += c.EvalAcqfNoGrad(x);
            return v;
        }

        // log_feasible_improvement = constraint_log_feasibility + per-draw log HVI (that order, as in Python).
        double[] logFeasibleImprovement = ConstraintLogFeasibility(x);
        double[] hvi = _baseline.ComputePerSampleLogHvi(_condGpr!.SampleCandidatePosterior(x));
        for (int s = 0; s < hvi.Length; s++)
            logFeasibleImprovement[s] += hvi[s];
        return Acqf.LogMeanExp(logFeasibleImprovement);
    }

    /// <summary><c>eval_acqf_with_grad</c> — value and the hand-derived gradient (softmax over draws of the per-draw gradient).</summary>
    /// <param name="x">Normalized input coordinates in search-space order; rectangular matrices contain one point per row.</param>
    /// <returns>The acquisition value and its gradient in normalized parameter coordinates.</returns>
    public (double Value, double[] Grad) EvalAcqfWithGrad(double[] x)
    {
        int dim = x.Length;
        var grad = new double[dim];
        if (_baseline == null)
        {
            double v = 0.0;
            foreach (LogPI c in _marginalConstraintsAcqfList)
            {
                (double cv, double[] cg) = c.EvalAcqfWithGrad(x);
                v += cv;
                for (int d = 0; d < dim; d++)
                    grad[d] += cg[d];
            }
            return (v, grad);
        }

        // Σ_c candidate log feasibility and its gradient, summed in constraint order.
        (double[] cons, double[,] dCons) = _constraintsAcqfList[0].ComputeCandidateLogUtilityWithGrad(x);
        for (int c = 1; c < _constraintsAcqfList.Length; c++)
        {
            (double[] lu, double[,] dlu) = _constraintsAcqfList[c].ComputeCandidateLogUtilityWithGrad(x);
            for (int s = 0; s < lu.Length; s++)
            {
                cons[s] += lu[s];
                for (int d = 0; d < dim; d++)
                    dCons[s, d] += dlu[s, d];
            }
        }

        (double[,] samples, double[][,] dSamples) = _condGpr!.SampleCandidatePosteriorWithGrad(x);
        (double[] hvi, double[,] dHvi) = _baseline.ComputePerSampleLogHviWithGrad(samples, dSamples);

        var logFeasibleImprovement = new double[hvi.Length];
        for (int s = 0; s < hvi.Length; s++)
            logFeasibleImprovement[s] = cons[s] + hvi[s];
        double lse = Acqf.LogSumExp(logFeasibleImprovement);
        double value = lse - Math.Log(logFeasibleImprovement.Length);

        // ∂/∂x of logsumexp_s L_s = Σ_s softmax_s(L) · ∂L_s/∂x, with ∂L_s = ∂cons_s + ∂logHVI_s.
        for (int s = 0; s < hvi.Length; s++)
        {
            double w = Math.Exp(logFeasibleImprovement[s] - lse);
            if (w == 0.0) continue;
            for (int d = 0; d < dim; d++)
                grad[d] += w * (dCons[s, d] + dHvi[s, d]);
        }
        return (value, grad);
    }

    /// <summary>
    ///     <c>sum(acqf.compute_candidate_log_utility(x) for acqf in constraints)</c> — the candidate's
    ///     summed smooth log feasibility per QMC draw, accumulated in constraint order.
    /// </summary>
    /// <param name="x">The candidate point in normalized search-space coordinates.</param>
    /// <returns>A fresh <c>(n_qmc,)</c> array the caller may modify.</returns>
    private double[] ConstraintLogFeasibility(double[] x)
    {
        double[] cons = _constraintsAcqfList[0].ComputeCandidateLogUtility(x);
        for (int c = 1; c < _constraintsAcqfList.Length; c++)
        {
            double[] lu = _constraintsAcqfList[c].ComputeCandidateLogUtility(x);
            for (int s = 0; s < lu.Length; s++)
                cons[s] += lu[s];
        }
        return cons;
    }

    /// <summary>
    ///     <c>fantasy[is_feasible]</c> for one QMC draw — the running trials' objective fantasies whose
    ///     constraints were all sampled as feasible in that draw, in running-trial order.
    /// </summary>
    /// <param name="fantasy">The draw's <c>(n_running, n_objectives)</c> objective fantasies.</param>
    /// <param name="isFantasyFeasible">The <c>(n_qmc, n_running)</c> joint feasibility mask.</param>
    /// <param name="s">The QMC draw whose row of the mask applies.</param>
    /// <returns>A new <c>(n_feasible, n_objectives)</c> matrix; zero rows when no fantasy is feasible.</returns>
    private static double[,] FeasibleRows(double[,] fantasy, bool[,] isFantasyFeasible, int s)
    {
        int nRunning = fantasy.GetLength(0), nObjectives = fantasy.GetLength(1);
        int count = 0;
        for (int r = 0; r < nRunning; r++)
            if (isFantasyFeasible[s, r]) count++;
        var rows = new double[count, nObjectives];
        int k = 0;
        for (int r = 0; r < nRunning; r++)
        {
            if (!isFantasyFeasible[s, r]) continue;
            for (int o = 0; o < nObjectives; o++)
                rows[k, o] = fantasy[r, o];
            k++;
        }
        return rows;
    }
}

/// <summary>
///     The upper confidence bound acquisition function — port of <c>acqf.UCB</c>. <c>mean + √(β·var)</c>
///     over the GP posterior. Consumed by <see cref="OptunaSharp.Terminator.RegretBoundEvaluator" />
///     (PORTING_PLAN Phase 3.3): the standardized regret bound is <c>max(UCB) − max(LCB)</c>. Rides the
///     managed posterior (triangular solve) → the documented GP tolerance boundary, not bit-exact.
/// </summary>
internal sealed class UCB : IAcquisitionFunc
{
    private readonly GPRegressor _gpr;
    private readonly double _beta;

    /// <summary>Configures the upper confidence bound from the fitted model, search space, and supplied thresholds or sampling settings.</summary>
    /// <param name="gpr">The Gaussian-process model supplying posterior predictions and kernel length scales.</param>
    /// <param name="searchSpace">The normalized coordinate layout, bounds, and discrete choices used by the acquisition optimizer.</param>
    /// <param name="beta">The nonnegative confidence-width coefficient.</param>
    internal UCB(GPRegressor gpr, GpSearchSpace searchSpace, double beta)
    {
        _gpr = gpr;
        _beta = beta;
        SearchSpace = searchSpace;
        LengthScales = gpr.LengthScales;
    }

    /// <summary>The normalized parameter space over which this acquisition function is optimized.</summary>
    public GpSearchSpace SearchSpace { get; }
    /// <summary>The kernel length scale of each coordinate, used to precondition gradient steps.</summary>
    public double[] LengthScales { get; }

    /// <summary><c>eval_acqf_no_grad</c> — <c>mean + √(β·var)</c> at each row of <paramref name="x" /> (<c>(B, dim)</c>).</summary>
    /// <param name="x">Normalized input coordinates in search-space order; rectangular matrices contain one point per row.</param>
    /// <returns>The acquisition value at the supplied point, or one value per input row for a batch.</returns>
    public double[] EvalAcqfNoGrad(double[,] x)
    {
        int b = x.GetLength(0);
        (double[] mean, double[] var) = _gpr.Posterior(x);
        var result = new double[b];
        for (int i = 0; i < b; i++)
            result[i] = mean[i] + Math.Sqrt(_beta * var[i]);
        return result;
    }

    /// <summary><c>eval_acqf_no_grad</c> at a single point.</summary>
    /// <param name="x">Normalized input coordinates in search-space order; rectangular matrices contain one point per row.</param>
    /// <returns>The acquisition value at the supplied point, or one value per input row for a batch.</returns>
    public double EvalAcqfNoGrad(double[] x)
    {
        int dim = x.Length;
        var row = new double[1, dim];
        for (int d = 0; d < dim; d++)
            row[0, d] = x[d];
        (double[] mean, double[] var) = _gpr.Posterior(row);
        return mean[0] + Math.Sqrt(_beta * var[0]);
    }

    /// <summary><c>eval_acqf_with_grad</c> — value and ∂UCB/∂x (autograd's inherited path; unused by the terminator).</summary>
    /// <param name="x">Normalized input coordinates in search-space order; rectangular matrices contain one point per row.</param>
    /// <returns>The acquisition value and its gradient in normalized parameter coordinates.</returns>
    public (double Value, double[] Grad) EvalAcqfWithGrad(double[] x)
    {
        (double mean, double var, double[] dMean, double[] dVar, bool _) = _gpr.PosteriorWithGrad(x);
        double sq = Math.Sqrt(_beta * var);
        double factor = _beta / (2.0 * sq);
        var grad = new double[x.Length];
        for (int d = 0; d < x.Length; d++)
            grad[d] = dMean[d] + factor * dVar[d];
        return (mean + sq, grad);
    }
}

/// <summary>
///     The lower confidence bound acquisition function — port of <c>acqf.LCB</c>. <c>mean − √(β·var)</c>
///     over the GP posterior. Consumed by <see cref="OptunaSharp.Terminator.RegretBoundEvaluator" />
///     (PORTING_PLAN Phase 3.3). Rides the managed posterior → the documented GP tolerance boundary.
/// </summary>
internal sealed class LCB : IAcquisitionFunc
{
    private readonly GPRegressor _gpr;
    private readonly double _beta;

    /// <summary>Configures the lower confidence bound from the fitted model, search space, and supplied thresholds or sampling settings.</summary>
    /// <param name="gpr">The Gaussian-process model supplying posterior predictions and kernel length scales.</param>
    /// <param name="searchSpace">The normalized coordinate layout, bounds, and discrete choices used by the acquisition optimizer.</param>
    /// <param name="beta">The nonnegative confidence-width coefficient.</param>
    internal LCB(GPRegressor gpr, GpSearchSpace searchSpace, double beta)
    {
        _gpr = gpr;
        _beta = beta;
        SearchSpace = searchSpace;
        LengthScales = gpr.LengthScales;
    }

    /// <summary>The normalized parameter space over which this acquisition function is optimized.</summary>
    public GpSearchSpace SearchSpace { get; }
    /// <summary>The kernel length scale of each coordinate, used to precondition gradient steps.</summary>
    public double[] LengthScales { get; }

    /// <summary><c>eval_acqf_no_grad</c> — <c>mean − √(β·var)</c> at each row of <paramref name="x" /> (<c>(B, dim)</c>).</summary>
    /// <param name="x">Normalized input coordinates in search-space order; rectangular matrices contain one point per row.</param>
    /// <returns>The acquisition value at the supplied point, or one value per input row for a batch.</returns>
    public double[] EvalAcqfNoGrad(double[,] x)
    {
        int b = x.GetLength(0);
        (double[] mean, double[] var) = _gpr.Posterior(x);
        var result = new double[b];
        for (int i = 0; i < b; i++)
            result[i] = mean[i] - Math.Sqrt(_beta * var[i]);
        return result;
    }

    /// <summary><c>eval_acqf_no_grad</c> at a single point.</summary>
    /// <param name="x">Normalized input coordinates in search-space order; rectangular matrices contain one point per row.</param>
    /// <returns>The acquisition value at the supplied point, or one value per input row for a batch.</returns>
    public double EvalAcqfNoGrad(double[] x)
    {
        int dim = x.Length;
        var row = new double[1, dim];
        for (int d = 0; d < dim; d++)
            row[0, d] = x[d];
        (double[] mean, double[] var) = _gpr.Posterior(row);
        return mean[0] - Math.Sqrt(_beta * var[0]);
    }

    /// <summary><c>eval_acqf_with_grad</c> — value and ∂LCB/∂x (autograd's inherited path; unused by the terminator).</summary>
    /// <param name="x">Normalized input coordinates in search-space order; rectangular matrices contain one point per row.</param>
    /// <returns>The acquisition value and its gradient in normalized parameter coordinates.</returns>
    public (double Value, double[] Grad) EvalAcqfWithGrad(double[] x)
    {
        (double mean, double var, double[] dMean, double[] dVar, bool _) = _gpr.PosteriorWithGrad(x);
        double sq = Math.Sqrt(_beta * var);
        double factor = _beta / (2.0 * sq);
        var grad = new double[x.Length];
        for (int d = 0; d < x.Length; d++)
            grad[d] = dMean[d] - factor * dVar[d];
        return (mean - sq, grad);
    }
}
