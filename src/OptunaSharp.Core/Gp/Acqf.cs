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
    private static double[] GetReferencePoint(double[,] yTrain)
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
    private static (double[,] Lower, double[,] Upper) GetNonDominatedBoxBoundsMax(double[,] yTrain, double[] refPoint)
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
