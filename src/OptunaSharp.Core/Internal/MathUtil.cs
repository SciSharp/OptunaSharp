using System;
using System.Globalization;
using System.Runtime.CompilerServices;
using NumSharp.Utilities;

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
    ///     NumPy's float64 <c>np.log1p(x)</c> for one value, bit for bit and without allocating: it runs NumSharp's public
    ///     model of the UCRT routine NumPy calls (<see cref="NDLog1pExpm1Math.Log1p(double)" />), so a scalar call costs a
    ///     few flops, never a 0-d <c>NDArray</c> left to the finalizer.
    /// </summary>
    /// <param name="x">The argument; any double, including NaN and the infinities.</param>
    /// <returns>
    ///     <c>log(1 + x)</c>, accurate near zero: <c>x</c> itself when <c>1 + x</c> rounds to 1, <c>-inf</c> at
    ///     <c>x == -1</c>, the positive quiet NaN below <c>-1</c> (as the UCRT returns it), <c>+inf</c> at <c>+inf</c>,
    ///     and a NaN argument passed through with its payload.
    /// </returns>
    /// <remarks>
    ///     <para>
    ///         The model is Goldberg's correction over the UCRT's own <c>log</c>,
    ///         <c>u = 1 + x; log(u) - ((u - 1) - x) / u</c>, the form this helper used to inline. Measured against NumPy
    ///         2.4.2 on Windows (2026-11-11, 600,000 seeded inputs over (-1, 1), tiny magnitudes and [1, 1e6]): 100%
    ///         bit-identical, as was the former inline copy, so the switch to NumSharp changed no finite result.
    ///     </para>
    ///     <para>
    ///         The only observable difference from the former copy is a NaN's bits: a NaN argument now keeps its payload
    ///         instead of becoming <see cref="double.NaN" />, and <c>x &lt; -1</c> yields the positive quiet NaN where
    ///         <see cref="Math.Log(double)" /> of a negative number returned the negative one. Both are NaN to every
    ///         comparison; only code hashing raw bits could tell them apart.
    ///     </para>
    /// </remarks>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static double Log1p(double x) => NDLog1pExpm1Math.Log1p(x);

    /// <summary>
    ///     NumPy's float64 <c>np.expm1(x)</c> for one value, without allocating, through NumSharp's public model of the
    ///     UCRT routine NumPy calls (<see cref="NDLog1pExpm1Math.Expm1(double)" />): NumPy's exact bits outside
    ///     <c>[-ln 2, ln 1.5]</c>, within 2 ulp of them inside that band. Prefer it over any hand-written formula; none
    ///     reproduces the UCRT inside the band.
    /// </summary>
    /// <param name="x">The argument; any double, including NaN and the infinities.</param>
    /// <returns>
    ///     <c>exp(x) - 1</c>, accurate near zero: <c>-1</c> once <c>exp(x)</c> underflows (and for <c>-inf</c>),
    ///     <c>+inf</c> on overflow, a zero's sign kept, and a NaN argument passed through with its payload.
    /// </returns>
    /// <remarks>
    ///     <para>
    ///         <b>Why NumSharp's model and not Kahan's formula.</b> Outside the band the UCRT returns exactly
    ///         <c>exp(x) - 1</c>, which the model reproduces. Inside it the UCRT evaluates an undocumented polynomial
    ///         that is not correctly rounded and that no public formula reproduces, so the model runs Sun's
    ///         <c>s_expm1.c</c>, the closest known. This helper used to compute Kahan's <c>(u - 1) * x / log(u)</c>
    ///         with <c>u = exp(x)</c>, documented as bit-identical to NumPy; measured on 2026-11-11 against NumPy 2.4.2
    ///         on Windows (200,000 seeded inputs per range), it was not:
    ///     </para>
    ///     <list type="bullet">
    ///         <item><description>outside the band: Kahan 82.0% equal (worst 2 ulp); the model 100%;</description></item>
    ///         <item><description><c>(-0.01, 0]</c>, <see cref="TruncNorm.NdtriExp" />'s flip branch: Kahan 62.4% (worst 3 ulp); the model 69.6% (worst 2);</description></item>
    ///         <item><description><c>[0.01, 5]</c>, its initial-guess branch: Kahan 84.3% (worst 3 ulp); the model 96.7% (worst 2);</description></item>
    ///         <item><description>the whole band: Kahan 50.9% (worst 4 ulp); the model 58.2% (worst 3).</description></item>
    ///     </list>
    ///     <para>
    ///         <b>Consequence.</b> The switch moves TPE's truncated-normal quantiles (<see cref="TruncNorm.NdtriExp" />
    ///         seeds its Newton iteration with this value) closer to NumPy's; it also changes their last bits relative
    ///         to OptunaSharp builds before the switch, so a seeded TPE study can differ from an earlier release in the
    ///         low bits. TPE's live parity was already erf-tolerance parity (the documented <see cref="Erf" /> class), and
    ///         the residual in-band difference stays within that class.
    ///     </para>
    /// </remarks>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static double Expm1(double x) => NDLog1pExpm1Math.Expm1(x);

    /// <summary>
    ///     NumPy's float64 <c>np.logaddexp(a, b) = log(exp(a) + exp(b))</c> for one pair, bit for bit and without
    ///     allocating: NumSharp's public port of NumPy's <c>npy_logaddexp</c> statements
    ///     (<see cref="NDLogAddExpMath.LogAddExp(double, double)" />) over the same UCRT <c>log1p</c> model as
    ///     <see cref="Log1p" />. Use it wherever Optuna combines log-probabilities, including the cancellation regime
    ///     (<c>a ≈ b</c>), where a <c>log1p</c> one ulp off is amplified into many result ulps.
    /// </summary>
    /// <param name="a">The logarithm of the first nonnegative addend.</param>
    /// <param name="b">The logarithm of the second nonnegative addend.</param>
    /// <returns>
    ///     The natural logarithm of <c>exp(a) + exp(b)</c>, computed without overflow: <c>a + log(2)</c> when
    ///     <c>a == b</c> (same-signed infinities included), and NaN when either argument is NaN.
    /// </returns>
    /// <remarks>
    ///     The statements are the ones this helper used to inline (the <c>a == b</c> shortcut, then the larger argument
    ///     plus <c>log1p(exp(-|a - b|))</c>). Measured against NumPy 2.4.2 on Windows (2026-11-11, 400,000 seeded pairs,
    ///     near-equal and wide): 100% bit-identical, as was the former copy. NumSharp's own <c>np.logaddexp</c> ran an
    ///     fdlibm <c>log1p</c> one ulp off until its commit <c>16b0817a</c> (2026-11-05, release 0.80.0), which is why the
    ///     copy existed.
    /// </remarks>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static double LogAddExp(double a, double b) => NDLogAddExpMath.LogAddExp(a, b);

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
