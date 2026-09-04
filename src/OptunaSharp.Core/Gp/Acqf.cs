using System;
using System.Collections.Generic;
using OptunaSharp.Internal;

namespace OptunaSharp.Gp;

/// <summary>
///     GP acquisition-function primitives — port of the log-expected-improvement helpers in
///     <c>optuna/_gp/acqf.py</c> (<c>standard_logei</c> / <c>logei</c>). LogEI is the acquisition
///     function <see cref="GPSampler" /> maximizes for single-objective optimization; these are the
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
    GpSearchSpace SearchSpace { get; }
    double[] LengthScales { get; }

    /// <summary><c>eval_acqf_no_grad</c> over a batch <c>(B, dim)</c>.</summary>
    double[] EvalAcqfNoGrad(double[,] x);

    /// <summary><c>eval_acqf_no_grad</c> at a single point <c>(dim,)</c> — the discrete searches.</summary>
    double EvalAcqfNoGrad(double[] x);

    /// <summary><c>eval_acqf_with_grad</c> — value + ∂/∂x at a single point <c>(dim,)</c>.</summary>
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

    internal LogEI(GPRegressor gpr, GpSearchSpace searchSpace, double threshold, double stabilizingNoise = 1e-12)
    {
        _gpr = gpr;
        _threshold = threshold;
        _stabilizingNoise = stabilizingNoise;
        SearchSpace = searchSpace;
        LengthScales = gpr.LengthScales;
    }

    public GpSearchSpace SearchSpace { get; }
    public double[] LengthScales { get; }

    /// <summary><c>eval_acqf_no_grad</c> — LogEI at each row of <paramref name="x" /> (<c>(B, dim)</c>).</summary>
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
///     <see cref="GPSampler" /> maximizes for single-objective optimization <b>with running trials</b>
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

    public GpSearchSpace SearchSpace { get; }
    public double[] LengthScales { get; }

    /// <summary><c>compute_per_sample_log_utility</c> — <c>log(clamp_min(y_post − threshold, EPS))</c>, shape <c>(n_qmc, r+1)</c>.</summary>
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
    internal double[,] ComputePerSampleLogUtility(double[] x)
    {
        if (double.IsNegativeInfinity(_threshold))
            return new double[_condGpr.NQmc, _condGpr.NRunning + 1];   // all zeros
        return PerSampleLogUtility(_condGpr.SampleJointPosterior(x));
    }

    /// <summary>The per-sample log utility and its x-gradient (the gradient flows through the query column only).</summary>
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
    public double EvalAcqfNoGrad(double[] x) => Acqf.MeanMaxUtility(ComputePerSampleLogUtility(x));

    /// <summary><c>eval_acqf_with_grad</c> — value and the hand-derived gradient through amax/logsumexp.</summary>
    public (double Value, double[] Grad) EvalAcqfWithGrad(double[] x)
    {
        (double[,] logUtil, double[,] dQuery) = ComputePerSampleLogUtilityWithGrad(x);
        return Acqf.MeanMaxUtilityWithGrad(logUtil, dQuery);
    }
}

/// <summary>
///     The q-batch log probability-of-improvement acquisition function — port of <c>acqf.qLogPI</c>. It
///     is the running-trial analogue of <see cref="LogPI" />: <see cref="GPSampler" /> uses it, summed
///     over the constraint GPs, as the feasibility term of <see cref="QLogCEI" /> (single-objective
///     constrained optimization with running trials, PORTING_PLAN Phase 3.2 part 3e-6). Per QMC draw it
///     forms the joint-posterior sample over the running trials plus the query point
///     (<see cref="ConditionalGPRegressor" />), applies the smooth feasibility indicator
///     <c>logsigmoid((y − threshold)/τ)</c> (τ = 1e-2), and reduces with <see cref="Acqf.MeanMaxUtility" />.
/// </summary>
internal sealed class QLogPI : IAcquisitionFunc
{
    private readonly ConditionalGPRegressor _condGpr;
    private readonly double _threshold;
    private readonly double _tau;

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

    public GpSearchSpace SearchSpace { get; }
    public double[] LengthScales { get; }

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
    internal double[,] ComputePerSampleLogUtility(double[] x)
        => PerSampleLogUtility(_condGpr.SampleJointPosterior(x));

    /// <summary>The per-sample log utility and its x-gradient (through the query column only).</summary>
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

    /// <summary><c>eval_acqf_no_grad</c> over a batch <c>(B, dim)</c>.</summary>
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
    public double EvalAcqfNoGrad(double[] x) => Acqf.MeanMaxUtility(ComputePerSampleLogUtility(x));

    /// <summary><c>eval_acqf_with_grad</c> — value and the hand-derived gradient through amax/logsumexp.</summary>
    public (double Value, double[] Grad) EvalAcqfWithGrad(double[] x)
    {
        (double[,] logUtil, double[,] dQuery) = ComputePerSampleLogUtilityWithGrad(x);
        return Acqf.MeanMaxUtilityWithGrad(logUtil, dQuery);
    }

    /// <summary><c>torch.nn.functional.logsigmoid(v) = min(0,v) − log1p(e^(−|v|))</c> — the numerically stable log σ.</summary>
    private static double LogSigmoid(double v)
        => v >= 0 ? -MathUtil.Log1p(Math.Exp(-v)) : v - MathUtil.Log1p(Math.Exp(v));
}

/// <summary>
///     The q-batch log constrained-expected-improvement acquisition function — port of
///     <c>acqf.qLogCEI</c>, the acqf <see cref="GPSampler" /> maximizes for single-objective
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

    public GpSearchSpace SearchSpace { get; }
    public double[] LengthScales { get; }

    /// <summary><c>eval_acqf_no_grad</c> over a batch <c>(B, dim)</c>.</summary>
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
    public double EvalAcqfNoGrad(double[] x)
    {
        double[,] logUtil = _acqf.ComputePerSampleLogUtility(x);
        foreach (QLogPI c in _constraintsAcqfList)
            AddInPlace(logUtil, c.ComputePerSampleLogUtility(x));
        return Acqf.MeanMaxUtility(logUtil);
    }

    /// <summary><c>eval_acqf_with_grad</c> — value and gradient of the composed feasible-improvement utility.</summary>
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
///     posterior over <c>threshold</c>). <see cref="GPSampler" /> uses it, summed over the constraint
///     GPs, as the feasibility term of <see cref="LogCEI" /> (single-objective constrained
///     optimization) and of <see cref="LogCEHVI" /> (multi-objective constrained). It rides
///     <c>log_ndtr</c> → the erf/ucrt tolerance class.
///
///     <para><b>Running trials (Kriging-Believer):</b> when
///     <c>normalizedParamsOfRunningTrials</c> is given, the running points are appended to the GP with
///     their posterior mean as the fantasy objective value (<see cref="GPRegressor.AppendRunningData" />),
///     so the feasible-probability accounts for the trials already in flight (Optuna's
///     <c>append_running_data</c> path; PORTING_PLAN Phase 3.2 part 3e-6).</para>
/// </summary>
internal sealed class LogPI : IAcquisitionFunc
{
    private readonly GPRegressor _gpr;
    private readonly double _threshold;
    private readonly double _stabilizingNoise;

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

    public GpSearchSpace SearchSpace { get; }
    public double[] LengthScales { get; }

    /// <summary><c>eval_acqf_no_grad</c> — <c>log_ndtr((mean − threshold)/σ)</c> over a batch <c>(B, dim)</c>.</summary>
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
///     acqf <see cref="GPSampler" /> maximizes for single-objective <b>constrained</b> optimization.
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

    public GpSearchSpace SearchSpace { get; }
    public double[] LengthScales { get; }

    /// <summary><c>eval_acqf_no_grad</c> — <c>LogEI(x) + Σ LogPI_c(x)</c> over a batch.</summary>
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
    public double EvalAcqfNoGrad(double[] x)
    {
        double v = _acqf.EvalAcqfNoGrad(x);
        foreach (LogPI c in _constraintsAcqfList)
            v += c.EvalAcqfNoGrad(x);
        return v;
    }

    /// <summary><c>eval_acqf_with_grad</c> — the summed value and gradient of the objective + constraint terms.</summary>
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
///     acqf <see cref="GPSampler" /> maximizes for <b>multi-objective</b> optimization (PORTING_PLAN
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
///     flight (Optuna's <c>append_running_data</c> path; used by the MO-constrained
///     <see cref="LogCEHVI" /> with running trials, PORTING_PLAN Phase 3.2 part 3e-6). The box
///     decomposition still uses the completed-trial <c>Y_train</c>.</para>
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

    public GpSearchSpace SearchSpace { get; }
    public double[] LengthScales { get; }

    /// <summary><c>eval_acqf_no_grad</c> over a batch <c>(B, dim)</c>.</summary>
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
///     <c>acqf.qLogEHVI</c>, the acqf <see cref="GPSampler" /> maximizes for <b>multi-objective</b>
///     optimization <b>with running trials</b> (PORTING_PLAN Phase 3.2 part 3e-6). Unlike
///     <see cref="LogEHVI" /> (independent per-objective sampling + one box decomposition), each QMC
///     draw gets its <b>own</b> box decomposition — computed from that draw's running fantasies
///     (<see cref="ConditionalGPRegressor.GetFantasySamples" />) appended to <c>Y_train</c> — and the
///     per-objective conditional query sample (<see cref="ConditionalGPRegressor.SampleQuerySamples" />).
///     Per draw the log HVI is a <c>logsumexp</c> over that draw's boxes; the acqf is the log-mean-exp
///     of those per-sample HVIs (Eq. (3) of arxiv 2006.05078). One conditional GP per objective, each
///     seeded <c>qmc_seed + i</c>.
///
///     <para><b>Gradient</b> (the autograd replacement): the fantasies (hence the boxes) are constant,
///     so the gradient flows only through the query samples; it chains a nested softmax (over the boxes
///     within a sample, then over the samples) times the two-sided box-clamp mask times each objective's
///     conditional-sample gradient.</para>
/// </summary>
internal sealed class QLogEHVI : IAcquisitionFunc
{
    private const double Eps = 1e-12;

    private readonly ConditionalGPRegressor[] _condGprList;
    private readonly int _nObjectives;
    private readonly int _nQmc;
    private readonly double[][,] _lowerBounds;   // per QMC draw: (n_boxes_s, n_obj) box lower bounds
    private readonly double[][,] _intervals;     // per QMC draw: (n_boxes_s, n_obj) box widths, clamped ≥ EPS

    internal QLogEHVI(
        GPRegressor[] gprList, GpSearchSpace searchSpace, double[,] yTrain, int nQmcSamples, int qmcSeed,
        double[,] normalizedParamsOfRunningTrials, double stabilizingNoise = 1e-12)
    {
        _nObjectives = yTrain.GetLength(1);
        _condGprList = new ConditionalGPRegressor[_nObjectives];
        for (int o = 0; o < _nObjectives; o++)
            _condGprList[o] = new ConditionalGPRegressor(
                gprList[o], normalizedParamsOfRunningTrials, nQmcSamples, qmcSeed + o, stabilizingNoise);
        _nQmc = _condGprList[0].NQmc;
        int nRunning = _condGprList[0].NRunning;
        int nTrain = yTrain.GetLength(0);

        // A per-QMC-draw box decomposition over Y_train ∪ {this draw's fantasies}, one shared ref point.
        double[] refPoint = LogEHVI.GetReferencePoint(yTrain);
        var fantasies = new double[_nObjectives][,];   // fantasies[o] = (n_qmc, n_running)
        for (int o = 0; o < _nObjectives; o++)
            fantasies[o] = _condGprList[o].GetFantasySamples();

        _lowerBounds = new double[_nQmc][,];
        _intervals = new double[_nQmc][,];
        for (int s = 0; s < _nQmc; s++)
        {
            var yFantasy = new double[nTrain + nRunning, _nObjectives];
            for (int t = 0; t < nTrain; t++)
                for (int o = 0; o < _nObjectives; o++)
                    yFantasy[t, o] = yTrain[t, o];
            for (int i = 0; i < nRunning; i++)
                for (int o = 0; o < _nObjectives; o++)
                    yFantasy[nTrain + i, o] = fantasies[o][s, i];
            (double[,] lower, double[,] upper) = LogEHVI.GetNonDominatedBoxBoundsMax(yFantasy, refPoint);
            int nBoxes = lower.GetLength(0);
            var interval = new double[nBoxes, _nObjectives];
            for (int b = 0; b < nBoxes; b++)
                for (int o = 0; o < _nObjectives; o++)
                    interval[b, o] = Math.Max(Eps, upper[b, o] - lower[b, o]);
            _lowerBounds[s] = lower;
            _intervals[s] = interval;
        }

        SearchSpace = searchSpace;
        LengthScales = LogEHVI.MeanLengthScales(gprList, searchSpace.Dim);
    }

    public GpSearchSpace SearchSpace { get; }
    public double[] LengthScales { get; }

    /// <summary><c>eval_acqf_no_grad</c> over a batch <c>(B, dim)</c>.</summary>
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
    public double EvalAcqfNoGrad(double[] x)
    {
        var samplesByObj = new double[_nObjectives][];
        for (int o = 0; o < _nObjectives; o++)
            samplesByObj[o] = _condGprList[o].SampleQuerySamples(x);
        double[] hvi = PerSampleLogHvi(samplesByObj, out _);
        return LogMeanExp(hvi);
    }

    /// <summary><c>eval_acqf_with_grad</c> — value and the hand-derived nested-softmax gradient.</summary>
    public (double Value, double[] Grad) EvalAcqfWithGrad(double[] x)
    {
        int dim = x.Length;
        var samplesByObj = new double[_nObjectives][];
        var dSamplesByObj = new double[_nObjectives][,];
        for (int o = 0; o < _nObjectives; o++)
        {
            (double[] samp, double[,] dsamp) = _condGprList[o].SampleQueryWithGrad(x);
            samplesByObj[o] = samp;
            dSamplesByObj[o] = dsamp;
        }

        double[] hvi = PerSampleLogHvi(samplesByObj, out double[][] termsPerS);
        double logsumexpS = LogSumExp(hvi);
        double value = logsumexpS - Math.Log(_nQmc);

        // grad[d] = Σ_{s,b} exp(term[s][b] − logsumexpS) · Σ_o mask[s,b,o]·(1/raw)·∂samples[o][s]/∂x_d.
        var grad = new double[dim];
        for (int s = 0; s < _nQmc; s++)
        {
            double[,] lower = _lowerBounds[s];
            double[,] interval = _intervals[s];
            int nBoxes = lower.GetLength(0);
            double[] term = termsPerS[s];
            for (int b = 0; b < nBoxes; b++)
            {
                double w = Math.Exp(term[b] - logsumexpS);
                if (w == 0.0) continue;
                for (int o = 0; o < _nObjectives; o++)
                {
                    double raw = samplesByObj[o][s] - lower[b, o];
                    if (!(raw > Eps && raw < interval[b, o]))
                        continue;   // two-sided clamp → gradient only in the open interval
                    double factor = w / raw;
                    double[,] ds = dSamplesByObj[o];
                    for (int d = 0; d < dim; d++)
                        grad[d] += factor * ds[s, d];
                }
            }
        }
        return (value, grad);
    }

    /// <summary>
    ///     <c>_per_sample_log_hvi</c> — for each QMC draw <c>s</c>, the log HVI
    ///     <c>logsumexp_b Σ_o log(clamp(samples[o][s] − lower[s,b,o], EPS, interval[s,b,o]))</c>. Also
    ///     returns the per-draw box terms (for the gradient's box softmax).
    /// </summary>
    private double[] PerSampleLogHvi(double[][] samplesByObj, out double[][] termsPerS)
    {
        var hvi = new double[_nQmc];
        termsPerS = new double[_nQmc][];
        for (int s = 0; s < _nQmc; s++)
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
                    double raw = samplesByObj[o][s] - lower[b, o];
                    double iv = interval[b, o];
                    double diff = raw < Eps ? Eps : (raw > iv ? iv : raw);
                    sumLog += Math.Log(diff);
                }
                term[b] = sumLog;
            }
            hvi[s] = LogSumExp(term);
            termsPerS[s] = term;
        }
        return hvi;
    }

    private static double LogSumExp(double[] vals)
    {
        double m = double.NegativeInfinity;
        foreach (double v in vals) if (v > m) m = v;
        if (double.IsNegativeInfinity(m)) return m;
        double sum = 0.0;
        foreach (double v in vals) sum += Math.Exp(v - m);
        return m + Math.Log(sum);
    }

    private static double LogMeanExp(double[] vals) => LogSumExp(vals) - Math.Log(vals.Length);
}

/// <summary>
///     The log constrained-expected-hypervolume-improvement acquisition function — port of
///     <c>acqf.LogCEHVI</c>, the acqf <see cref="GPSampler" /> maximizes for <b>multi-objective
///     constrained</b> optimization (PORTING_PLAN Phase 3.2 part 3e-5). Assuming independence between
///     the objectives and each constraint, it is the log expected hypervolume improvement over the
///     <b>feasible</b> trials plus the summed log feasible-probabilities:
///     <c>LogCEHVI(x) = LogEHVI_feasible(x) + Σ_c LogPI_c(x)</c>. When there is no feasible trial the
///     <see cref="LogEHVI" /> term is dropped and only the constraint terms remain (the optimizer then
///     drives toward feasibility).
///
///     <para><b>Running trials (Kriging-Believer):</b> when <c>normalizedParamsOfRunningTrials</c> is
///     given it is forwarded to both the feasible <see cref="LogEHVI" /> and each constraint
///     <see cref="LogPI" />, which append the running points as fantasies. Unlike the single-objective
///     path this is <b>not</b> a q-batch (Monte-Carlo) acqf — the MO-constrained running path stays the
///     Kriging-Believer <see cref="LogCEHVI" /> (PORTING_PLAN Phase 3.2 part 3e-6).</para>
/// </summary>
internal sealed class LogCEHVI : IAcquisitionFunc
{
    private readonly LogEHVI? _acqf;
    private readonly LogPI[] _constraintsAcqfList;

    internal LogCEHVI(
        GPRegressor[] gprList, GpSearchSpace searchSpace, double[,]? yFeasible,
        int nQmcSamples, int qmcSeed,
        IReadOnlyList<GPRegressor> constraintsGprList, IReadOnlyList<double> constraintsThresholdList,
        double[,]? normalizedParamsOfRunningTrials = null, double stabilizingNoise = 1e-12)
    {
        if (constraintsGprList.Count != constraintsThresholdList.Count || constraintsGprList.Count == 0)
            throw new ArgumentException("constraints_gpr_list and constraints_threshold_list must be non-empty and equal-length.");

        _acqf = yFeasible != null
            ? new LogEHVI(gprList, searchSpace, yFeasible, nQmcSamples, qmcSeed,
                normalizedParamsOfRunningTrials, stabilizingNoise)
            : null;
        _constraintsAcqfList = new LogPI[constraintsGprList.Count];
        for (int i = 0; i < constraintsGprList.Count; i++)
            _constraintsAcqfList[i] = new LogPI(
                constraintsGprList[i], searchSpace, constraintsThresholdList[i],
                normalizedParamsOfRunningTrials, stabilizingNoise);

        SearchSpace = searchSpace;
        LengthScales = LogEHVI.MeanLengthScales(gprList, searchSpace.Dim);
    }

    public GpSearchSpace SearchSpace { get; }
    public double[] LengthScales { get; }

    /// <summary><c>eval_acqf_no_grad</c> — <c>Σ LogPI_c(x)</c> (+ <c>LogEHVI(x)</c> when there is a feasible term) over a batch.</summary>
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
///     The upper confidence bound acquisition function — port of <c>acqf.UCB</c>. <c>mean + √(β·var)</c>
///     over the GP posterior. Consumed by <see cref="OptunaSharp.Terminator.RegretBoundEvaluator" />
///     (PORTING_PLAN Phase 3.3): the standardized regret bound is <c>max(UCB) − max(LCB)</c>. Rides the
///     managed posterior (triangular solve) → the documented GP tolerance boundary, not bit-exact.
/// </summary>
internal sealed class UCB : IAcquisitionFunc
{
    private readonly GPRegressor _gpr;
    private readonly double _beta;

    internal UCB(GPRegressor gpr, GpSearchSpace searchSpace, double beta)
    {
        _gpr = gpr;
        _beta = beta;
        SearchSpace = searchSpace;
        LengthScales = gpr.LengthScales;
    }

    public GpSearchSpace SearchSpace { get; }
    public double[] LengthScales { get; }

    /// <summary><c>eval_acqf_no_grad</c> — <c>mean + √(β·var)</c> at each row of <paramref name="x" /> (<c>(B, dim)</c>).</summary>
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

    internal LCB(GPRegressor gpr, GpSearchSpace searchSpace, double beta)
    {
        _gpr = gpr;
        _beta = beta;
        SearchSpace = searchSpace;
        LengthScales = gpr.LengthScales;
    }

    public GpSearchSpace SearchSpace { get; }
    public double[] LengthScales { get; }

    /// <summary><c>eval_acqf_no_grad</c> — <c>mean − √(β·var)</c> at each row of <paramref name="x" /> (<c>(B, dim)</c>).</summary>
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
