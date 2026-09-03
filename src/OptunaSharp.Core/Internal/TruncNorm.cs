using System;
using System.Collections.Generic;

namespace OptunaSharp.Internal;

/// <summary>
///     Partial port of <c>optuna/samplers/_tpe/_truncnorm.py</c> (itself lifted from SciPy) — the
///     log-Gaussian-probability-mass machinery that <c>PedAnovaImportanceEvaluator</c> depends on
///     via <c>_MixtureOfProductDistribution.log_pdf</c>. The sampling half of that module (<c>ppf</c>
///     / <c>rvs</c> / <c>_ndtri_exp</c>) lands with the TPE sampler in Phase 2.8.
///
///     <para><b>[Misaligned] erf/erfc.</b> <c>_ndtr</c>/<c>_ndtr_single</c> call Optuna's
///     <c>math.erf</c>/<c>math.erfc</c>, which on CPython resolve to the platform C library
///     (<c>ucrt</c> on Windows). No pure-managed <c>erf</c> reproduces that libm bit-for-bit —
///     <see cref="Erf" /> ports FreeBSD's <c>s_erf.c</c>, which differs from <c>ucrt</c> by ~1-3 ULP.
///     Consequently the truncated-normal density (and any importance value that rides on it) matches
///     Python only to a few ULP, not bit-for-bit. This is the same class of system-libm divergence
///     the plan already excuses for LAPACK-backed linear algebra; the categorical path, the trial
///     selection, and everything not routed through <c>erf</c> stay bit-exact. Optuna's own ≥2000-
///     element vectorized <c>erf</c> approximation IS reproduced bit-for-bit by
///     <see cref="Erf.ComputeArray" />.</para>
/// </summary>
internal static class TruncNorm
{
    private static readonly double Sqrt2 = Math.Sqrt(2.0);

    // Python: _norm_pdf_C = sqrt(2*pi); _norm_pdf_logC = log(_norm_pdf_C).
    internal static readonly double NormPdfLogC = Math.Log(Math.Sqrt(2.0 * Math.PI));

    // Python: _ndtri_exp_approx_C = sqrt(3) / pi.
    private static readonly double NdtriExpApproxC = Math.Sqrt(3.0) / Math.PI;

    // sys.float_info.epsilon (machine epsilon for float64), NOT double.Epsilon (smallest subnormal).
    private const double MachineEpsilon = 2.220446049250313e-16;

    /// <summary>Python <c>_ndtr_single(a)</c> — the scalar normal CDF used inside <c>_log_ndtr</c>.</summary>
    private static double NdtrSingle(double a)
    {
        double x = a / Sqrt2;
        double y;
        if (x < -1.0 / Sqrt2)
            y = 0.5 * Erf.Erfc(-x);
        else if (x < 1.0 / Sqrt2)
            y = 0.5 + 0.5 * Erf.Compute(x);
        else
            y = 1.0 - 0.5 * Erf.Erfc(x);
        return y;
    }

    /// <summary>Python <c>_ndtr(a)</c> — the vectorized normal CDF (uses the <c>_erf.py</c> variant).</summary>
    private static double[] Ndtr(double[] a)
    {
        var scaled = new double[a.Length];
        for (int i = 0; i < a.Length; i++)
            scaled[i] = a[i] / Sqrt2;
        var erf = Erf.ComputeArray(scaled);
        var result = new double[a.Length];
        for (int i = 0; i < a.Length; i++)
            result[i] = 0.5 + 0.5 * erf[i];
        return result;
    }

    /// <summary>Python <c>_log_ndtr_single(a)</c>.</summary>
    private static double LogNdtrSingle(double a)
    {
        if (a > 6.0)
            return -NdtrSingle(-a);
        if (a > -20.0)
            return Math.Log(NdtrSingle(a));

        double logLhs = -0.5 * a * a - Math.Log(-a) - 0.5 * Math.Log(2.0 * Math.PI);
        double lastTotal = 0.0;
        double rightHandSide = 1.0;
        double numerator = 1.0;
        double denomFactor = 1.0;
        double denomCons = 1.0 / (a * a);
        int sign = 1;
        int i = 0;

        while (Math.Abs(lastTotal - rightHandSide) > MachineEpsilon)
        {
            i += 1;
            lastTotal = rightHandSide;
            sign = -sign;
            denomFactor *= denomCons;
            numerator *= 2 * i - 1;
            rightHandSide += sign * numerator * denomFactor;
        }

        return logLhs + Math.Log(rightHandSide);
    }

    /// <summary>Python <c>_log_diff(log_p, log_q) = log_p + log1p(-exp(log_q - log_p))</c>.</summary>
    private static double LogDiff(double logP, double logQ) => logP + MathUtil.Log1p(-Math.Exp(logQ - logP));

    /// <summary>
    ///     Python <c>_log_gauss_mass(a, b)</c> — log of the Gaussian probability mass in the
    ///     interval <c>[a, b]</c>. Element-wise over 1-D arrays.
    /// </summary>
    internal static double[] LogGaussMass(double[] a, double[] b)
    {
        int n = a.Length;
        var outArr = new double[n];

        // case_left = b <= 0; case_right = a > 0; case_central = ~(left | right). NumPy assigns
        // left, then right (overwriting), then central — so a > 0 wins the left/right overlap.
        var centralIdx = new System.Collections.Generic.List<int>();
        for (int i = 0; i < n; i++)
        {
            if (!(a[i] > 0.0) && !(b[i] <= 0.0))
                centralIdx.Add(i);
        }

        if (centralIdx.Count > 0)
        {
            var ac = new double[centralIdx.Count];
            var negB = new double[centralIdx.Count];
            for (int k = 0; k < centralIdx.Count; k++)
            {
                ac[k] = a[centralIdx[k]];
                negB[k] = -b[centralIdx[k]];
            }
            var ndtrAc = Ndtr(ac);
            var ndtrNegB = Ndtr(negB);
            for (int k = 0; k < centralIdx.Count; k++)
                outArr[centralIdx[k]] = MathUtil.Log1p(-ndtrAc[k] - ndtrNegB[k]);
        }

        for (int i = 0; i < n; i++)
        {
            if (a[i] > 0.0)
                // mass_case_right(a, b) = mass_case_left(-b, -a).
                outArr[i] = LogDiff(LogNdtrSingle(-a[i]), LogNdtrSingle(-b[i]));
            else if (b[i] <= 0.0)
                outArr[i] = LogDiff(LogNdtrSingle(b[i]), LogNdtrSingle(a[i]));
        }

        return outArr;
    }

    // ------------------------------------------------------------------------------------------
    // Sampling half of _truncnorm.py (arrives with TPE, PORTING_PLAN Phase 2.8).
    // ------------------------------------------------------------------------------------------

    /// <summary>Python <c>_log_ndtr(a)</c> — the vectorized <c>frompyfunc(_log_ndtr_single)</c>.</summary>
    internal static double[] LogNdtr(double[] a)
    {
        var result = new double[a.Length];
        for (int i = 0; i < a.Length; i++)
            result[i] = LogNdtrSingle(a[i]);
        return result;
    }

    /// <summary>Python <c>_norm_logpdf(x) = -(x**2) / 2 - _norm_pdf_logC</c> (scalar; array <c>**2</c> == <c>x*x</c>).</summary>
    internal static double NormLogPdf(double x) => -(x * x) / 2.0 - NormPdfLogC;

    /// <summary>
    ///     Python <c>_ndtri_exp(y)</c> — the Newton iteration returning <c>x</c> with
    ///     <c>y == log_ndtr(x)</c>. Operates element-wise but iterates globally (all elements step
    ///     together, break when every element's update satisfies <c>|dx| &lt; 1e-8 |x|</c>), exactly
    ///     as NumPy's whole-array loop does.
    /// </summary>
    internal static double[] NdtriExp(double[] y)
    {
        int n = y.Length;
        var flipped = new bool[n];
        var z = new double[n];
        for (int i = 0; i < n; i++)
        {
            z[i] = y[i];
            if (y[i] > -1e-2)
            {
                // Flip the sign near zero for stability: z = log(-expm1(y)).
                flipped[i] = true;
                z[i] = Math.Log(-MathUtil.Expm1(y[i]));
            }
        }

        var x = new double[n];
        for (int i = 0; i < n; i++)
        {
            if (z[i] < -5.0)
                x[i] = -Math.Sqrt(-2.0 * (z[i] + NormPdfLogC));
            else
                x[i] = -NdtriExpApproxC * Math.Log(MathUtil.Expm1(-z[i]));
        }

        for (int iter = 0; iter < 100; iter++)
        {
            var logNdtrX = LogNdtr(x);
            bool allConverged = true;
            for (int i = 0; i < n; i++)
            {
                double x2 = x[i] * x[i];
                double logNormPdfX = -0.5 * x2 - NormPdfLogC;
                // exp(log_ndtr_x - log_norm_pdf_x) instead of ndtr/norm_pdf for numerical stability.
                double dx = (logNdtrX[i] - z[i]) * Math.Exp(logNdtrX[i] - logNormPdfX);
                x[i] -= dx;
                if (!(Math.Abs(dx) < 1e-8 * Math.Abs(x[i])))
                    allConverged = false;
            }
            if (allConverged)
                break;
        }

        for (int i = 0; i < n; i++)
        {
            if (flipped[i])
                x[i] = -x[i];
        }
        return x;
    }

    /// <summary>
    ///     Python <c>_truncnorm.ppf(q, a, b)</c> — the percent-point function of the truncated
    ///     standard normal on <c>[a, b]</c>. Inputs are same-length flattened arrays (the caller
    ///     broadcasts). <c>case_left</c> (<c>a &lt; 0</c>) and <c>case_right</c> are computed on
    ///     disjoint subsets for numerical stability, matching Optuna.
    /// </summary>
    internal static double[] Ppf(double[] q, double[] a, double[] b)
    {
        int n = q.Length;
        var logMass = LogGaussMass(a, b);
        var outArr = new double[n];

        var leftIdx = new List<int>();
        var rightIdx = new List<int>();
        for (int i = 0; i < n; i++)
        {
            if (a[i] < 0.0)
                leftIdx.Add(i);
            else
                rightIdx.Add(i);
        }

        if (leftIdx.Count > 0)
        {
            // ppf_left: _ndtri_exp(_log_sum(_log_ndtr(a), log(q) + log_mass)).
            var aSub = new double[leftIdx.Count];
            for (int k = 0; k < leftIdx.Count; k++)
                aSub[k] = a[leftIdx[k]];
            var logNdtrA = LogNdtr(aSub);
            var arg = new double[leftIdx.Count];
            for (int k = 0; k < leftIdx.Count; k++)
                arg[k] = MathUtil.LogAddExp(logNdtrA[k], Math.Log(q[leftIdx[k]]) + logMass[leftIdx[k]]);
            var res = NdtriExp(arg);
            for (int k = 0; k < leftIdx.Count; k++)
                outArr[leftIdx[k]] = res[k];
        }

        if (rightIdx.Count > 0)
        {
            // ppf_right: -_ndtri_exp(_log_sum(_log_ndtr(-b), log1p(-q) + log_mass)).
            var negBSub = new double[rightIdx.Count];
            for (int k = 0; k < rightIdx.Count; k++)
                negBSub[k] = -b[rightIdx[k]];
            var logNdtrNegB = LogNdtr(negBSub);
            var arg = new double[rightIdx.Count];
            for (int k = 0; k < rightIdx.Count; k++)
                arg[k] = MathUtil.LogAddExp(logNdtrNegB[k], MathUtil.Log1p(-q[rightIdx[k]]) + logMass[rightIdx[k]]);
            var res = NdtriExp(arg);
            for (int k = 0; k < rightIdx.Count; k++)
                outArr[rightIdx[k]] = -res[k];
        }

        // out[q == 0] = a; out[q == 1] = b; out[a == b] = nan (applied sequentially, in this order).
        for (int i = 0; i < n; i++)
        {
            if (q[i] == 0.0)
                outArr[i] = a[i];
            if (q[i] == 1.0)
                outArr[i] = b[i];
            if (a[i] == b[i])
                outArr[i] = double.NaN;
        }
        return outArr;
    }
}
