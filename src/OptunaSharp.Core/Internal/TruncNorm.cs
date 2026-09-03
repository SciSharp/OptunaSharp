using System;

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
}
