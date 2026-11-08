using System;

namespace OptunaSharp.Internal;

/// <summary>
///     Port of FreeBSD's <c>s_erf.c</c> (the algorithm behind CPython's <c>math.erf</c> /
///     <c>math.erfc</c>) and of Optuna's vectorized approximation in
///     <c>optuna/samplers/_tpe/_erf.py</c>. Optuna ships its own erf so scipy isn't needed —
///     OptunaSharp mirrors that (plan 2.8), and the Wilcoxon pruner's normal CDF rides on it.
/// </summary>
internal static class Erf
{
    private const double Tiny = 1e-300;
    private const double Erx = 8.45062911510467529297e-01;

    // In the domain [0, 2**-28], only the first term in the power series expansion of erf(x)
    // is used. The magnitude of the first neglected terms is less than 2**-84.
    private const double Efx = 1.28379167095512586316e-01;
    private const double Efx8 = 1.02703333676410069053e+00;

    // Coefficients for approximation to erf on [0, 0.84375].
    private const double PP0 = 1.28379167095512558561e-01;
    private const double PP1 = -3.25042107247001499370e-01;
    private const double PP2 = -2.84817495755985104766e-02;
    private const double PP3 = -5.77027029648944159157e-03;
    private const double PP4 = -2.37630166566501626084e-05;
    private const double QQ1 = 3.97917223959155352819e-01;
    private const double QQ2 = 6.50222499887672944485e-02;
    private const double QQ3 = 5.08130628187576562776e-03;
    private const double QQ4 = 1.32494738004321644526e-04;
    private const double QQ5 = -3.96022827877536812320e-06;

    // Coefficients for approximation to erf in [0.84375, 1.25].
    private const double PA0 = -2.36211856075265944077e-03;
    private const double PA1 = 4.14856118683748331666e-01;
    private const double PA2 = -3.72207876035701323847e-01;
    private const double PA3 = 3.18346619901161753674e-01;
    private const double PA4 = -1.10894694282396677476e-01;
    private const double PA5 = 3.54783043256182359371e-02;
    private const double PA6 = -2.16637559486879084300e-03;
    private const double QA1 = 1.06420880400844228286e-01;
    private const double QA2 = 5.40397917702171048937e-01;
    private const double QA3 = 7.18286544141962662868e-02;
    private const double QA4 = 1.26171219808761642112e-01;
    private const double QA5 = 1.36370839120290507362e-02;
    private const double QA6 = 1.19844998467991074170e-02;

    // Coefficients for approximation to erfc in [1.25, 1/0.35].
    private const double RA0 = -9.86494403484714822705e-03;
    private const double RA1 = -6.93858572707181764372e-01;
    private const double RA2 = -1.05586262253232909814e+01;
    private const double RA3 = -6.23753324503260060396e+01;
    private const double RA4 = -1.62396669462573470355e+02;
    private const double RA5 = -1.84605092906711035994e+02;
    private const double RA6 = -8.12874355063065934246e+01;
    private const double RA7 = -9.81432934416914548592e+00;
    private const double SA1 = 1.96512716674392571292e+01;
    private const double SA2 = 1.37657754143519042600e+02;
    private const double SA3 = 4.34565877475229228821e+02;
    private const double SA4 = 6.45387271733267880336e+02;
    private const double SA5 = 4.29008140027567833386e+02;
    private const double SA6 = 1.08635005541779435134e+02;
    private const double SA7 = 6.57024977031928170135e+00;
    private const double SA8 = -6.04244152148580987438e-02;

    // Coefficients for approximation to erfc in [1/0.35, 28].
    private const double RB0 = -9.86494292470009928597e-03;
    private const double RB1 = -7.99283237680523006574e-01;
    private const double RB2 = -1.77579549177547519889e+01;
    private const double RB3 = -1.60636384855821916062e+02;
    private const double RB4 = -6.37566443368389627722e+02;
    private const double RB5 = -1.02509513161107724954e+03;
    private const double RB6 = -4.83519191608651397019e+02;
    private const double SB1 = 3.03380607434824582924e+01;
    private const double SB2 = 3.25792512996573918826e+02;
    private const double SB3 = 1.53672958608443695994e+03;
    private const double SB4 = 3.19985821950859553908e+03;
    private const double SB5 = 2.55305040643316442583e+03;
    private const double SB6 = 4.74528541206955367215e+02;
    private const double SB7 = -2.24409524465858183362e+01;

    /// <summary>Returns the signed high 32 bits of the IEEE-754 representation.</summary>
    /// <param name="x">The scalar argument.</param>
    /// <returns>The signed high 32 bits of the IEEE-754 representation.</returns>
    private static int HighWord(double x) => (int)(BitConverter.DoubleToInt64Bits(x) >> 32);

    /// <summary>Clear the low 32 bits of the mantissa (FreeBSD <c>SET_LOW_WORD(z, 0)</c>).</summary>
    /// <param name="x">The scalar argument.</param>
    /// <returns>The input with its low 32 representation bits cleared.</returns>
    private static double ClearLowWord(double x)
        => BitConverter.Int64BitsToDouble(BitConverter.DoubleToInt64Bits(x) & unchecked((long)0xFFFFFFFF00000000));

    /// <summary>The error function, matching C99 <c>erf</c> (FreeBSD <c>s_erf.c</c>).</summary>
    /// <param name="x">The scalar argument.</param>
    /// <returns>The error function evaluated at the supplied argument.</returns>
    internal static double Compute(double x)
    {
        int hx = HighWord(x);
        int ix = hx & 0x7fffffff;
        if (ix >= 0x7ff00000)
        {
            // erf(nan) = nan, erf(+-inf) = +-1.
            int i = ((uint)hx >> 31) != 0 ? -1 : 1;
            return i + 1.0 / x;
        }

        if (ix < 0x3feb0000)
        {
            // |x| < 0.84375
            if (ix < 0x3e300000)
            {
                // |x| < 2**-28
                if (ix < 0x00800000)
                {
                    // Avoid underflow.
                    return 0.125 * (8.0 * x + Efx8 * x);
                }
                return x + Efx * x;
            }
            double z0 = x * x;
            double r0 = PP0 + z0 * (PP1 + z0 * (PP2 + z0 * (PP3 + z0 * PP4)));
            double s0 = 1.0 + z0 * (QQ1 + z0 * (QQ2 + z0 * (QQ3 + z0 * (QQ4 + z0 * QQ5))));
            double y0 = r0 / s0;
            return x + x * y0;
        }
        if (ix < 0x3ff40000)
        {
            // 0.84375 <= |x| < 1.25
            double s1 = Math.Abs(x) - 1.0;
            double P = PA0 + s1 * (PA1 + s1 * (PA2 + s1 * (PA3 + s1 * (PA4 + s1 * (PA5 + s1 * PA6)))));
            double Q = 1.0 + s1 * (QA1 + s1 * (QA2 + s1 * (QA3 + s1 * (QA4 + s1 * (QA5 + s1 * QA6)))));
            if (hx >= 0)
                return Erx + P / Q;
            return -Erx - P / Q;
        }
        if (ix >= 0x40180000)
        {
            // inf > |x| >= 6
            if (hx >= 0)
                return 1.0 - Tiny;
            return Tiny - 1.0;
        }

        double xAbs = Math.Abs(x);
        double z = xAbs * xAbs;
        double s = 1.0 / z;
        double R, S;
        if (ix < 0x4006DB6E)
        {
            // |x| < 1/0.35
            R = RA0 + s * (RA1 + s * (RA2 + s * (RA3 + s * (RA4 + s * (RA5 + s * (RA6 + s * RA7))))));
            S = 1.0 + s * (SA1 + s * (SA2 + s * (SA3 + s * (SA4 + s * (SA5 + s * (SA6 + s * (SA7 + s * SA8)))))));
        }
        else
        {
            // |x| >= 1/0.35
            R = RB0 + s * (RB1 + s * (RB2 + s * (RB3 + s * (RB4 + s * (RB5 + s * RB6)))));
            S = 1.0 + s * (SB1 + s * (SB2 + s * (SB3 + s * (SB4 + s * (SB5 + s * (SB6 + s * SB7))))));
        }
        double zRounded = ClearLowWord(xAbs);
        double r = Math.Exp(-zRounded * zRounded - 0.5625) * Math.Exp((zRounded - xAbs) * (zRounded + xAbs) + R / S);
        if (hx >= 0)
            return 1.0 - r / xAbs;
        return r / xAbs - 1.0;
    }

    /// <summary>The complementary error function, matching C99 <c>erfc</c> (FreeBSD <c>s_erf.c</c>).</summary>
    /// <param name="x">The scalar argument.</param>
    /// <returns>The complementary error function evaluated at the supplied argument.</returns>
    internal static double Erfc(double x)
    {
        int hx = HighWord(x);
        int ix = hx & 0x7fffffff;
        if (ix >= 0x7ff00000)
        {
            // erfc(nan) = nan, erfc(+inf) = 0, erfc(-inf) = 2.
            int i = ((uint)hx >> 31) != 0 ? -1 : 1;
            return (double)(1 - i) + 1.0 / x;
        }

        if (ix < 0x3feb0000)
        {
            // |x| < 0.84375
            if (ix < 0x3c700000)
            {
                // |x| < 2**-56
                return 1.0 - x;
            }
            double z0 = x * x;
            double r0 = PP0 + z0 * (PP1 + z0 * (PP2 + z0 * (PP3 + z0 * PP4)));
            double s0 = 1.0 + z0 * (QQ1 + z0 * (QQ2 + z0 * (QQ3 + z0 * (QQ4 + z0 * QQ5))));
            double y0 = r0 / s0;
            if (hx < 0x3fd00000)
            {
                // x < 1/4
                return 1.0 - (x + x * y0);
            }
            double r1 = x * y0;
            r1 += x - 0.5;
            return 0.5 - r1;
        }
        if (ix < 0x3ff40000)
        {
            // 0.84375 <= |x| < 1.25
            double s1 = Math.Abs(x) - 1.0;
            double P = PA0 + s1 * (PA1 + s1 * (PA2 + s1 * (PA3 + s1 * (PA4 + s1 * (PA5 + s1 * PA6)))));
            double Q = 1.0 + s1 * (QA1 + s1 * (QA2 + s1 * (QA3 + s1 * (QA4 + s1 * (QA5 + s1 * QA6)))));
            if (hx >= 0)
                return 1.0 - Erx - P / Q;
            return 1.0 + Erx + P / Q;
        }

        if (ix < 0x403c0000)
        {
            // |x| < 28
            double xAbs = Math.Abs(x);
            double z = xAbs * xAbs;
            double s = 1.0 / z;
            double R, S;
            if (ix < 0x4006DB6D)
            {
                // |x| < 1/0.35 ~ 2.857143
                R = RA0 + s * (RA1 + s * (RA2 + s * (RA3 + s * (RA4 + s * (RA5 + s * (RA6 + s * RA7))))));
                S = 1.0 + s * (SA1 + s * (SA2 + s * (SA3 + s * (SA4 + s * (SA5 + s * (SA6 + s * (SA7 + s * SA8)))))));
            }
            else
            {
                // |x| >= 1/0.35
                if (hx < 0 && ix >= 0x40180000)
                {
                    // x < -6
                    return 2.0 - Tiny;
                }
                R = RB0 + s * (RB1 + s * (RB2 + s * (RB3 + s * (RB4 + s * (RB5 + s * RB6)))));
                S = 1.0 + s * (SB1 + s * (SB2 + s * (SB3 + s * (SB4 + s * (SB5 + s * (SB6 + s * SB7))))));
            }
            double zRounded = ClearLowWord(xAbs);
            double r = Math.Exp(-zRounded * zRounded - 0.5625) * Math.Exp((zRounded - xAbs) * (zRounded + xAbs) + R / S);
            if (hx > 0)
                return r / xAbs;
            return 2.0 - r / xAbs;
        }

        if (hx > 0)
            return Tiny * Tiny;  // Underflows to 0 with the raised inexact, like the C.
        return 2.0 - Tiny;
    }

    /// <summary>
    ///     Optuna's vectorized low-accuracy path (<c>_erf.py: erf</c>) used for arrays of 2000+
    ///     elements — reproduced so large-study TPE stays bit-comparable with Python. Smaller
    ///     inputs go through <see cref="Compute" /> exactly like Python calls <c>math.erf</c>.
    /// </summary>
    /// <param name="x">The numerical input values, in the order used by the operation.</param>
    /// <returns>The elementwise error-function values in input order.</returns>
    internal static double[] ComputeArray(double[] x)
    {
        var result = new double[x.Length];
        ComputeArray(x, result);
        return result;
    }

    /// <summary>
    ///     <see cref="ComputeArray(double[])" /> into a caller's buffer, for callers that reuse scratch memory (Rustuna
    ///     plan 13, item S6). The path is chosen by <paramref name="x" />'s length exactly as the array overload chooses
    ///     it by the array's length, so a caller must pass a span of the logical length, never a larger rented buffer.
    /// </summary>
    /// <param name="x">The numerical input values; its length selects the exact or the vectorized path.</param>
    /// <param name="result">Receives one value per element of <paramref name="x" />; at least as long, and not overlapping it.</param>
    /// <exception cref="IndexOutOfRangeException"><paramref name="result" /> is shorter than <paramref name="x" />.</exception>
    internal static void ComputeArray(ReadOnlySpan<double> x, Span<double> result)
    {
        if (x.Length < 2000)
        {
            for (int i = 0; i < x.Length; i++)
                result[i] = Compute(x[i]);
            return;
        }

        for (int i = 0; i < x.Length; i++)
        {
            double v = x[i];
            if (double.IsNaN(v))
            {
                result[i] = double.NaN;
                continue;
            }
            double a = Math.Abs(v);
            double outVal;
            if (a >= 6)
            {
                outVal = 1.0;
            }
            else if (a < Math.Pow(2, -28))
            {
                outVal = (1 + Efx) * a;
            }
            else if (a < 0.84375)
            {
                double z = a * a;
                double p = PP0 + z * (PP1 + z * (PP2 + z * (PP3 + z * PP4)));
                double q = 1.0 + z * (QQ1 + z * (QQ2 + z * (QQ3 + z * (QQ4 + z * QQ5))));
                outVal = a * (1 + p / q);
            }
            else if (a < 1.25)
            {
                double s = a - 1;
                double p = PA0 + s * (PA1 + s * (PA2 + s * (PA3 + s * (PA4 + s * (PA5 + s * PA6)))));
                double q = 1.0 + s * (QA1 + s * (QA2 + s * (QA3 + s * (QA4 + s * (QA5 + s * QA6)))));
                outVal = Erx + p / q;
            }
            else if (a < 1 / 0.35)
            {
                double z = a * a;
                double s = 1 / z;
                double r = RA0 + s * (RA1 + s * (RA2 + s * (RA3 + s * (RA4 + s * (RA5 + s * (RA6 + s * RA7))))));
                double sd = 1.0 + s * (SA1 + s * (SA2 + s * (SA3 + s * (SA4 + s * (SA5 + s * (SA6 + s * (SA7 + s * SA8)))))));
                outVal = 1 - Math.Exp(-z - 0.5625 + r / sd) / a;
            }
            else
            {
                double z = a * a;
                double s = 1 / z;
                double r = RB0 + s * (RB1 + s * (RB2 + s * (RB3 + s * (RB4 + s * (RB5 + s * RB6)))));
                double sd = 1.0 + s * (SB1 + s * (SB2 + s * (SB3 + s * (SB4 + s * (SB5 + s * (SB6 + s * SB7))))));
                outVal = 1 - Math.Exp(-z - 0.5625 + r / sd) / a;
            }
            result[i] = Math.Sign(v) * outVal;
        }
    }

    /// <summary>
    ///     Scaled complementary error function <c>erfcx(x) = exp(x²)·erfc(x)</c> — the numerically
    ///     stable form used by the GP LogEI acquisition function for large arguments (where a naive
    ///     <c>exp(x²)·erfc(x)</c> would overflow × underflow). For <c>|x| ≤ 25</c> the direct product
    ///     is safe (no overflow, no underflow); beyond that the asymptotic series
    ///     <c>1/(x√π)·Σ(-1)ᵏ(2k-1)!!/(2x²)ᵏ</c> is used. Rides the erf/ucrt tolerance class.
    /// </summary>
    /// <param name="x">The scalar argument.</param>
    /// <returns>The scaled complementary error function exp(x*x) * erfc(x).</returns>
    internal static double Erfcx(double x)
    {
        if (x < 0.0)
            return 2.0 * Math.Exp(x * x) - Erfcx(-x);  // erfcx(-x) = 2·exp(x²) − erfcx(x)
        if (x <= 25.0)
            return Math.Exp(x * x) * Erfc(x);
        // Asymptotic expansion for large x (convergent-enough here: successive terms shrink fast).
        double inv2x2 = 1.0 / (2.0 * x * x);
        double term = 1.0;
        double sum = 1.0;
        for (int k = 1; k <= 12; k++)
        {
            term *= -(2 * k - 1) * inv2x2;
            sum += term;
        }
        return sum / (x * Math.Sqrt(Math.PI));
    }

    /// <summary>Standard normal CDF: <c>0.5 * erfc(-z / sqrt(2))</c>.</summary>
    /// <param name="z">The standardized normal argument.</param>
    /// <returns>The lower-tail standard-normal probability at the supplied argument.</returns>
    internal static double NormCdf(double z) => 0.5 * Erfc(-z / Math.Sqrt(2.0));

    /// <summary>Standard normal survival function: <c>0.5 * erfc(z / sqrt(2))</c>.</summary>
    /// <param name="z">The standardized normal argument.</param>
    /// <returns>The upper-tail standard-normal probability at the supplied argument.</returns>
    internal static double NormSf(double z) => 0.5 * Erfc(z / Math.Sqrt(2.0));
}
