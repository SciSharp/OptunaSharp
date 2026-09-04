using System;
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
