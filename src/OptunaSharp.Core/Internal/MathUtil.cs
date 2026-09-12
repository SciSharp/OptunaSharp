using System;
using System.Globalization;

namespace OptunaSharp.Internal;

/// <summary>Floating-point and reduction helpers preserving the numerical conventions expected by the Optuna port.</summary>
internal static class MathUtil
{
    /// <summary>
    ///     Python <c>f"{x:.3g}"</c> — 3 significant figures, C-style <c>%g</c>. .NET's <c>"G3"</c>
    ///     reproduces C <c>%g</c> exactly (same fixed/scientific threshold, same rounding, same
    ///     trailing-zero stripping, min-2-digit exponent) except for the uppercase <c>E</c>, so
    ///     lowercasing it makes the two byte-identical (verified against CPython's <c>format</c>).
    /// </summary>
    /// <param name="x">The scalar argument.</param>
    /// <returns>The invariant three-significant-figure representation with a lowercase exponent marker.</returns>
    internal static string FormatG3(double x)
        => x.ToString("G3", CultureInfo.InvariantCulture).Replace("E", "e");

    /// <summary>C99 / NumPy <c>nextafter(x, y)</c>: the next representable double after <paramref name="x" /> toward <paramref name="y" />.</summary>
    /// <param name="x">The starting floating-point value.</param>
    /// <param name="y">The value defining the direction in which to take one representable step.</param>
    /// <returns>The adjacent representable double toward the second value, or that value when equal.</returns>
    internal static double NextAfter(double x, double y)
    {
        if (double.IsNaN(x) || double.IsNaN(y))
            return x + y;
        if (x == y)
            return y;
        return y > x ? Math.BitIncrement(x) : Math.BitDecrement(x);
    }

    /// <summary>NumPy <c>np.clip(value, low, high)</c> for scalars (NaN propagates).</summary>
    /// <param name="value">The value to convert, format, or compare using the operation's stated convention.</param>
    /// <param name="low">The lower endpoint of the allowed range.</param>
    /// <param name="high">The upper endpoint of the allowed range.</param>
    /// <returns>The value clamped to the interval; NaN propagates.</returns>
    internal static double Clip(double value, double low, double high)
        => Math.Min(Math.Max(value, low), high);

    /// <summary>
    ///     C99 / NumPy <c>np.log1p(x)</c> — <c>log(1 + x)</c> accurate for small <c>x</c>. .NET has
    ///     no <c>Math.Log1p</c>; this correction-term form (<c>log(u) - ((u-1) - x)/u</c>, u = 1+x)
    ///     is bit-identical to NumPy's <c>log1p</c> because <see cref="Math.Log(double)" /> already
    ///     matches NumPy's <c>np.log</c> bit-for-bit (verified over 220k points, 0 ULP).
    /// </summary>
    /// <param name="x">The scalar argument.</param>
    /// <returns>The natural logarithm of one plus the argument, with correction near zero.</returns>
    internal static double Log1p(double x)
    {
        if (double.IsNaN(x))
            return double.NaN;
        double u = 1.0 + x;
        if (u == 1.0)
            return x;  // |x| below the rounding of 1 + x (incl. x == 0): log1p(x) == x.
        if (u <= 0.0)
            return Math.Log(u);  // x <= -1: -inf at x == -1, NaN below — same as np.log(1 + x).
        if (double.IsPositiveInfinity(u))
            return Math.Log(u);
        return Math.Log(u) - ((u - 1.0) - x) / u;
    }

    private static readonly double Loge2 = Math.Log(2.0);

    /// <summary>
    ///     C99 / NumPy <c>np.expm1(x)</c> — <c>exp(x) - 1</c> accurate for small <c>x</c>. .NET has
    ///     no <c>Math.Expm1</c>; this Kahan form (<c>(u-1)*x/log(u)</c>, u = exp(x)) is bit-identical
    ///     to NumPy's <c>expm1</c> because <see cref="Math.Exp(double)" />/<see cref="Math.Log(double)" />
    ///     already match NumPy's <c>np.exp</c>/<c>np.log</c> bit-for-bit (verified against Python over
    ///     the TPE truncnorm input range).
    /// </summary>
    /// <param name="x">The scalar argument.</param>
    /// <returns>The exponential of the argument minus one, with correction near zero.</returns>
    internal static double Expm1(double x)
    {
        if (double.IsNaN(x))
            return double.NaN;
        double u = Math.Exp(x);
        if (u == 1.0)
            return x;  // |x| below the rounding of exp(x): expm1(x) == x.
        double y = u - 1.0;
        if (y == -1.0)
            return -1.0;  // exp(x) underflowed to 0 (x very negative).
        if (double.IsPositiveInfinity(u))
            return u;  // overflow.
        return y * x / Math.Log(u);
    }

    /// <summary>
    ///     NumPy <c>np.logaddexp(a, b) = log(exp(a) + exp(b))</c>, computed stably. Reproduces
    ///     NumPy's <c>npy_logaddexp</c> exactly (the <c>a == b</c> branch adds <c>log(2)</c> and
    ///     handles the infinities); bit-identical to Python because it rides on the bit-exact
    ///     <see cref="Log1p" /> and <see cref="Math.Exp(double)" />.
    /// </summary>
    /// <param name="a">The logarithm of the first nonnegative addend.</param>
    /// <param name="b">The logarithm of the second nonnegative addend.</param>
    /// <returns>The natural logarithm of exp(a) + exp(b), computed without unnecessary overflow.</returns>
    internal static double LogAddExp(double a, double b)
    {
        if (a == b)
            return a + Loge2;  // Also covers a == b == ±inf.
        double tmp = a - b;
        if (tmp > 0)
            return a + Log1p(Math.Exp(-tmp));
        if (tmp <= 0)
            return b + Log1p(Math.Exp(tmp));
        return tmp;  // NaN input.
    }

    /// <summary>
    ///     NumPy's <c>np.add.reduce</c> pairwise summation for a contiguous float64 array — the
    ///     exact accumulation order of <c>ndarray.sum()</c>, so reductions stay bit-identical.
    /// </summary>
    /// <param name="a">The contiguous float64 values to sum in NumPy's accumulation order.</param>
    /// <returns>The sum using NumPy's contiguous float64 pairwise accumulation order.</returns>
    internal static double PairwiseSum(double[] a) => PairwiseSum(a, 0, a.Length);

    /// <inheritdoc cref="PairwiseSum(double[])" />
    /// <param name="a">The contiguous values containing the slice to sum.</param>
    /// <param name="offset">The zero-based inclusive start of the slice.</param>
    /// <param name="n">The number of values to sum from the offset; the slice must fit in the array.</param>
    /// <returns>The slice sum using NumPy's pairwise accumulation order.</returns>
    internal static double PairwiseSum(double[] a, int offset, int n)
    {
        if (n < 8)
        {
            double res = 0.0;
            for (int i = 0; i < n; i++)
                res += a[offset + i];
            return res;
        }

        if (n <= 128)
        {
            double r0 = a[offset], r1 = a[offset + 1], r2 = a[offset + 2], r3 = a[offset + 3];
            double r4 = a[offset + 4], r5 = a[offset + 5], r6 = a[offset + 6], r7 = a[offset + 7];
            int i;
            for (i = 8; i < n - n % 8; i += 8)
            {
                r0 += a[offset + i];
                r1 += a[offset + i + 1];
                r2 += a[offset + i + 2];
                r3 += a[offset + i + 3];
                r4 += a[offset + i + 4];
                r5 += a[offset + i + 5];
                r6 += a[offset + i + 6];
                r7 += a[offset + i + 7];
            }

            double res = (r0 + r1 + (r2 + r3)) + (r4 + r5 + (r6 + r7));
            for (; i < n; i++)
                res += a[offset + i];
            return res;
        }

        // Divide by two but avoid non-multiples of the unroll factor.
        int n2 = n / 2;
        n2 -= n2 % 8;
        return PairwiseSum(a, offset, n2) + PairwiseSum(a, offset + n2, n - n2);
    }

    /// <summary>
    ///     NumPy <c>np.linspace(start, stop, num)</c> (endpoint=True) — bit-identical to NumPy for
    ///     the common case: <c>y[i] = i * step + start</c> with <c>step = (stop - start) / (num - 1)</c>,
    ///     and the final element pinned to <paramref name="stop" /> exactly.
    /// </summary>
    /// <param name="start">The inclusive start of the interval or slice.</param>
    /// <param name="stop">The final value of the generated numerical interval.</param>
    /// <param name="num">The number of values in the output grid.</param>
    /// <returns>An evenly spaced grid including both endpoints when at least two values are requested.</returns>
    internal static double[] Linspace(double start, double stop, int num)
    {
        if (num <= 0)
            return Array.Empty<double>();
        if (num == 1)
            return new[] { start };

        int div = num - 1;
        double step = (stop - start) / div;
        var y = new double[num];
        for (int i = 0; i < num; i++)
            y[i] = i * step + start;  // arange(num) * step + start.
        y[^1] = stop;  // endpoint override.
        return y;
    }

    /// <summary>
    ///     NumPy <c>np.mean</c> over a float64 array — <c>PairwiseSum(a) / a.Length</c>, so the
    ///     accumulation order matches NumPy's <c>add.reduce</c> bit-for-bit. Empty → NaN (as NumPy).
    /// </summary>
    /// <param name="a">The observations whose arithmetic mean is computed.</param>
    /// <returns>The arithmetic mean of the input values.</returns>
    internal static double Mean(double[] a)
        => a.Length == 0 ? double.NaN : PairwiseSum(a) / a.Length;

    /// <summary>
    ///     NumPy <c>np.std</c> (population, <c>ddof=0</c>) over a float64 array. Reproduces NumPy's
    ///     algorithm exactly: <c>mean = sum/n</c>; residuals squared via <c>x*x</c> (NumPy array
    ///     <c>**2</c> is a multiply, not <see cref="Math.Pow(double,double)" />); variance is the
    ///     pairwise sum of those over <c>n</c>; then <see cref="Math.Sqrt(double)" />. Empty → NaN.
    /// </summary>
    /// <param name="a">The observations whose population standard deviation is computed.</param>
    /// <returns>The population standard deviation of the input values.</returns>
    internal static double Std(double[] a)
    {
        int n = a.Length;
        if (n == 0)
            return double.NaN;
        double mean = PairwiseSum(a) / n;
        var sq = new double[n];
        for (int i = 0; i < n; i++)
        {
            double d = a[i] - mean;
            sq[i] = d * d;
        }
        return Math.Sqrt(PairwiseSum(sq) / n);
    }
}
