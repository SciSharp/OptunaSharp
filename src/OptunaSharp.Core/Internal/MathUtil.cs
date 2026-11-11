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

    /// <summary>
    ///     NumPy's <c>np.isclose(a, b, rtol, atol, equal_nan)</c> for two Python floats, in NumPy's own statements:
    ///     <c>(|a - b| &lt;= atol + rtol * |b|) &amp; isfinite(b) | (a == b)</c>, plus "both NaN" when
    ///     <paramref name="equalNan" /> is set. Use it wherever Optuna calls <c>np.isclose</c> on scalars in a loop.
    /// </summary>
    /// <param name="a">The value tested (NumPy's <c>a</c>).</param>
    /// <param name="b">The reference value the relative tolerance scales with (NumPy's <c>b</c>; the test is not symmetric).</param>
    /// <param name="rtol">The relative tolerance; NumPy's default is <c>1e-5</c>.</param>
    /// <param name="atol">The absolute tolerance; NumPy's default is <c>1e-8</c>.</param>
    /// <param name="equalNan">Whether two NaNs compare as close.</param>
    /// <returns>
    ///     Whether <paramref name="a" /> is within tolerance of <paramref name="b" />. Two equal infinities are close
    ///     (<c>a == b</c>); a finite value is never close to an infinity, and two different infinities are not close.
    /// </returns>
    /// <remarks>
    ///     The <c>isfinite(b)</c> and <c>a == b</c> terms are what a bare <c>|a - b| &lt;= atol + rtol * |b|</c> misses:
    ///     with <c>b = inf</c> that inequality holds for every finite <c>a</c> (<c>inf &lt;= inf</c>), and with
    ///     <c>a = b = inf</c> it fails (<c>inf - inf</c> is NaN). NumSharp's <c>np.isclose</c> computes the same answers
    ///     (the unit tests compare the two on the edge cases), but a scalar call through it costs about 9 µs (two 0-d
    ///     arrays and a result) against a few nanoseconds here, and callers such as <c>Study.ShouldSkipEnqueue</c> compare
    ///     every parameter of every trial.
    /// </remarks>
    internal static bool IsClose(double a, double b, double rtol = 1e-5, double atol = 1e-8, bool equalNan = false)
    {
        // numpy.isclose: less_equal(abs(x - y), atol + rtol * abs(y)) & isfinite(y) | (x == y); the comparisons are false
        // for a NaN operand, as NumPy's are under errstate(invalid='ignore').
        bool close = Math.Abs(a - b) <= atol + rtol * Math.Abs(b) && double.IsFinite(b) || a == b;
        // if equal_nan: result |= isnan(x) & isnan(y)
        if (equalNan && double.IsNaN(a) && double.IsNaN(b))
            close = true;
        return close;
    }

    /// <summary>
    ///     NumPy <c>np.clip(value, low, high)</c> where the bounds are scalars — NumPy's CONST loop
    ///     (<c>_npy_clip_const_minmax_</c>): <c>value</c> is kept unless it lies strictly outside, so a tie keeps the
    ///     value's own zero sign. Use it where Optuna clips a scalar, a 0-d value, or an array against scalar bounds.
    /// </summary>
    /// <param name="value">The value to clip; a NaN stays NaN.</param>
    /// <param name="low">The lower bound; a NaN lower bound is the result.</param>
    /// <param name="high">The upper bound; a NaN upper bound is the result when <paramref name="low" /> is not NaN.</param>
    /// <returns>
    ///     <paramref name="low" /> when the value is below it, <paramref name="high" /> when above it, else the value
    ///     itself (<c>np.clip(-0.0, 0.0, 1.0)</c> is <c>-0.0</c>); the NaN bound when a bound is NaN.
    /// </returns>
    /// <remarks>
    ///     <para>
    ///         NumPy picks the loop per inner-loop call: with both bound strides 0 (scalar or 0-d bounds, a scalar value,
    ///         or a column against one-element bounds) it runs <c>x &lt; lo ? lo : x; x &gt; hi ? hi : x</c>, after
    ///         filling the call with a NaN bound when there is one; otherwise <see cref="ClipGeneral" />. The former
    ///         <c>Math.Min(Math.Max(x, low), high)</c> orders <c>-0.0</c> below <c>+0.0</c>, so a tie at a zero bound
    ///         took the other zero (QA-30: a stepped float with upper bound <c>-0.0</c> stored <c>-0.0</c> for its top
    ///         grid point where Optuna stores <c>0.0</c>).
    ///     </para>
    ///     <para>
    ///         NumSharp's <c>np.clip</c> makes the same decision (its port of NumPy's const-loop test) and gives the same
    ///         answers (the unit tests compare them); this scalar form avoids a NumSharp call per sampled value.
    ///     </para>
    /// </remarks>
    internal static double Clip(double value, double low, double high)
    {
        // "min_val and/or max_val are nans": the call is filled with the NaN bound, the lower one first.
        if (double.IsNaN(low))
            return low;
        if (double.IsNaN(high))
            return high;
        // The comparisons are false for a NaN value, which therefore passes through.
        if (value < low)
            value = low;
        if (value > high)
            value = high;
        return value;
    }

    /// <summary>
    ///     NumPy <c>np.clip</c>'s GENERAL loop for one element, <c>_NPY_MIN(_NPY_MAX(x, lo), hi)</c>: what an
    ///     inner-loop call with array bounds (a nonzero bound stride) computes. A tie takes the bound.
    /// </summary>
    /// <param name="value">The value to clip; a NaN stays NaN.</param>
    /// <param name="low">The lower bound for this element.</param>
    /// <param name="high">The upper bound for this element.</param>
    /// <returns>
    ///     <c>max = isnan(x) ? x : (x &gt; lo ? x : lo)</c>, then <c>isnan(max) ? max : (max &lt; hi ? max : hi)</c>: a value
    ///     equal to a bound comes back as the bound (<c>-0.0</c> against a <c>+0.0</c> bound is <c>+0.0</c>), and a NaN
    ///     bound is the result unless the value is NaN.
    /// </returns>
    /// <remarks>
    ///     TPE's discrete rounding, <c>np.clip(..., low_d, high_d)</c> over an (n, k) block with per-column bounds, runs
    ///     this loop when k &gt;= 2 (the inner loop walks the k columns, so the bounds have a stride); with one discrete
    ///     column the bounds broadcast with stride 0 and NumPy runs <see cref="Clip" />'s const loop instead (measured on
    ///     NumPy 2.4.2, 2026-11-11). NumSharp's <c>np.clip</c> agrees on both layouts.
    /// </remarks>
    internal static double ClipGeneral(double value, double low, double high)
    {
        // clip.cpp: _NPY_MAX(a, b) = npy_isnan(a) ? a : PyArray_MAX(a, b), PyArray_MAX(a, b) = a > b ? a : b.
        double max = double.IsNaN(value) ? value : (value > low ? value : low);
        // _NPY_MIN(a, b) = npy_isnan(a) ? a : PyArray_MIN(a, b), PyArray_MIN(a, b) = a < b ? a : b.
        return double.IsNaN(max) ? max : (max < high ? max : high);
    }

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
    ///     NumPy <c>np.linspace(start, stop, num)</c> (endpoint=True), computed by NumSharp's <c>np.linspace</c>, which ports
    ///     every branch NumPy takes, so the grid is NumPy's in every case, a step that underflows to zero included. Meant
    ///     for the visualization grids and ticks (a call or two per plot); hot loops keep their own copies.
    /// </summary>
    /// <param name="start">The inclusive start of the interval or slice.</param>
    /// <param name="stop">The final value of the generated numerical interval.</param>
    /// <param name="num">The number of values in the output grid; zero or less gives an empty grid.</param>
    /// <returns>
    ///     An evenly spaced grid: <c>i * step + start</c>, the last element exactly <paramref name="stop" /> when there are
    ///     two or more, <c>(i / (num - 1)) * (stop - start) + start</c> when the step underflows to zero, and for one
    ///     element <c>0 * (stop - start) + start</c> (NaN when an end is infinite).
    /// </returns>
    /// <remarks>
    ///     The former inline loop always took <c>i * step + start</c>. Over a subnormal range NumPy's step
    ///     <c>delta / (num - 1)</c> is zero and NumPy switches to scaling <c>i / div</c> by <c>delta</c> (gh-5437), so every
    ///     point but the last collapsed onto <paramref name="start" /> here: an EDF over values <c>0.0</c> and <c>5e-324</c>
    ///     stayed at 0.5 until its last point where Optuna's reaches 1.0 at point 50 (parity QA-54). A negative
    ///     <paramref name="num" /> keeps returning an empty grid instead of NumPy's <c>ValueError</c>, as before, because no
    ///     caller passes one.
    /// </remarks>
    internal static double[] Linspace(double start, double stop, int num)
    {
        if (num <= 0)
            return Array.Empty<double>();
        using NumSharp.NDArray grid = NumSharp.np.linspace(start, stop, num);
        return grid.ToArray<double>();
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
