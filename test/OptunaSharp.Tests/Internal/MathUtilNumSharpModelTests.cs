using System;
using System.IO;
using System.Text.Json;
using AwesomeAssertions;
using NumSharp.Utilities;
using OptunaSharp.Internal;

namespace OptunaSharp.Tests.Internal;

/// <summary>
///     Pins <see cref="MathUtil.Log1p" />, <see cref="MathUtil.Expm1" /> and <see cref="MathUtil.LogAddExp" /> to NumSharp's
///     public scalar models of the UCRT routines NumPy calls, and those to NumPy itself.
///     <list type="bullet">
///         <item><description>
///             On every OS: each helper is NumSharp's model bit for bit, so a future inline copy (the former Kahan
///             <c>expm1</c> was 62–84% equal to NumPy where TPE calls it) fails here first.
///         </description></item>
///         <item><description>
///             On Windows: the committed corpus <c>oracle/numpy_scalar_math.json</c> (NumPy 2.4.2, win-amd64, recorded by
///             <c>oracle/gen_numpy_scalar_math_oracle.py</c>) is replayed with no Python. log1p and logaddexp must match
///             bit for bit everywhere; expm1 bit for bit outside <c>[-ln 2, ln 1.5]</c> and within 2 ulp inside it, the
///             documented bound of the model (the UCRT's in-band polynomial is undocumented).
///         </description></item>
///     </list>
///     The corpus is Windows-only because NumPy's float64 log1p/expm1 call the platform libm (glibc or SVML loops on
///     Linux, Apple's libm on macOS), so its bits describe Windows NumPy only, which is what NumSharp's models reproduce.
/// </summary>
[TestClass]
public class MathUtilNumSharpModelTests
{
    /// <summary>Seeded inputs spanning every regime of the three functions, shared by the delegation pins.</summary>
    /// <returns>About 30,000 doubles: specials, tiny and huge magnitudes, both sides of expm1's band, and (-1, 1).</returns>
    private static double[] DelegationInputs()
    {
        var specials = new[]
        {
            0.0, -0.0, -1.0, 1.0, double.PositiveInfinity, double.NegativeInfinity, double.Epsilon, -double.Epsilon,
            1e-300, -1e-300, -Math.Log(2.0), Math.Log(1.5), -745.2, 709.7, 710.0, -1000.0,
        };
        // A seeded System.Random only picks test inputs; no sampler path reads it, so the RNG rule does not apply.
        var rng = new Random(20261111);
        var values = new double[specials.Length + 30_000];
        specials.CopyTo(values, 0);
        for (int i = specials.Length; i < values.Length; i++)
        {
            double sign = rng.Next(2) == 0 ? -1.0 : 1.0;
            values[i] = (i % 3) switch
            {
                0 => rng.NextDouble() * 2.0 - 1.0,                       // (-1, 1): log1p's core and expm1's band.
                1 => sign * Math.Pow(10.0, rng.NextDouble() * 600 - 300),  // tiny to huge magnitudes, both signs.
                _ => rng.NextDouble() * 80.0 - 40.0,                     // expm1 well outside the band, both sides.
            };
        }
        return values;
    }

    /// <summary>Loads the committed NumPy corpus copied next to the test assembly.</summary>
    /// <returns>The corpus root: <c>log1p</c>, <c>expm1</c> and <c>logaddexp</c> arrays of hex bit patterns.</returns>
    /// <exception cref="FileNotFoundException">The corpus was not copied to the output directory.</exception>
    private static JsonElement LoadCorpus()
    {
        string path = Path.Combine(AppContext.BaseDirectory, "oracle", "numpy_scalar_math.json");
        return JsonDocument.Parse(File.ReadAllText(path)).RootElement.Clone();
    }

    /// <summary>Decodes a corpus entry: 16 hex digits of an IEEE-754 bit pattern, sign of zero and NaN payload kept.</summary>
    /// <param name="element">A JSON string holding the bit pattern.</param>
    /// <returns>The exact double the pattern encodes.</returns>
    private static double FromBits(JsonElement element)
        => BitConverter.Int64BitsToDouble(unchecked((long)Convert.ToUInt64(element.GetString(), 16)));

    /// <summary>
    ///     The distance between two doubles in units in the last place, counted along the ordered bit patterns, so 0 means
    ///     bit-identical (NaN to NaN only when the bits agree) and +0 to -0 counts as one step.
    /// </summary>
    /// <param name="a">The first value.</param>
    /// <param name="b">The second value.</param>
    /// <returns>The number of representable doubles between them; <see cref="long.MaxValue" /> when exactly one is NaN.</returns>
    private static long UlpDistance(double a, double b)
    {
        long ia = BitConverter.DoubleToInt64Bits(a), ib = BitConverter.DoubleToInt64Bits(b);
        if (ia == ib)
            return 0;
        if (double.IsNaN(a) || double.IsNaN(b))
            return long.MaxValue;
        // Map the sign-magnitude patterns onto one monotonic integer line, negatives below positives.
        if (ia < 0)
            ia = long.MinValue - ia;
        if (ib < 0)
            ib = long.MinValue - ib;
        return Math.Abs(ia - ib);
    }

    /// <summary>
    ///     Skips (Inconclusive) off Windows: the corpus records Windows NumPy, whose log1p/expm1 come from the UCRT. CI
    ///     already excludes the <c>WindowsOnly</c> category elsewhere; this keeps an unfiltered local run green too.
    /// </summary>
    /// <exception cref="AssertInconclusiveException">The test runs on a platform other than Windows.</exception>
    private static void RequireWindows()
    {
        if (!OperatingSystem.IsWindows())
            Assert.Inconclusive("The NumPy corpus is a win-amd64 pin: NumPy's log1p/expm1 call the platform libm.");
    }

    /// <summary>
    ///     <see cref="MathUtil.Log1p" /> and <see cref="MathUtil.Expm1" /> return NumSharp's models bit for bit over 30,000
    ///     seeded inputs and the special values, on every OS.
    /// </summary>
    [TestMethod]
    public void Log1pAndExpm1_AreNumSharpsModels_BitForBit()
    {
        foreach (double x in DelegationInputs())
        {
            BitConverter.DoubleToInt64Bits(MathUtil.Log1p(x)).Should()
                .Be(BitConverter.DoubleToInt64Bits(NDLog1pExpm1Math.Log1p(x)), $"log1p({x:R})");
            BitConverter.DoubleToInt64Bits(MathUtil.Expm1(x)).Should()
                .Be(BitConverter.DoubleToInt64Bits(NDLog1pExpm1Math.Expm1(x)), $"expm1({x:R})");
        }
    }

    /// <summary>
    ///     <see cref="MathUtil.LogAddExp" /> returns NumSharp's <c>npy_logaddexp</c> port bit for bit over 15,000 seeded
    ///     pairs (near-equal ones included, the cancellation regime) and the infinities, on every OS.
    /// </summary>
    [TestMethod]
    public void LogAddExp_IsNumSharpsModel_BitForBit()
    {
        double[] values = DelegationInputs();
        for (int i = 0; i + 1 < values.Length; i += 2)
        {
            double a = values[i];
            // Every other pair is near-equal: b = a(1 + 1e-12) exercises log1p of exp(-tiny).
            double b = (i / 2) % 2 == 0 ? values[i + 1] : a * (1 + 1e-12);
            BitConverter.DoubleToInt64Bits(MathUtil.LogAddExp(a, b)).Should()
                .Be(BitConverter.DoubleToInt64Bits(NDLogAddExpMath.LogAddExp(a, b)), $"logaddexp({a:R}, {b:R})");
        }
        MathUtil.LogAddExp(double.PositiveInfinity, double.PositiveInfinity).Should().Be(double.PositiveInfinity);
        MathUtil.LogAddExp(double.NegativeInfinity, double.NegativeInfinity).Should().Be(double.NegativeInfinity);
        MathUtil.LogAddExp(double.NegativeInfinity, -3.5).Should().Be(-3.5);
    }

    /// <summary>
    ///     <see cref="MathUtil.Log1p" /> reproduces Windows NumPy's <c>np.log1p</c> bit for bit over the 163-input corpus:
    ///     zeros of both signs, <c>-1</c>, the infinities, subnormals, (-1, 1), just above -1, and up to 1e300.
    /// </summary>
    [TestMethod]
    [WindowsOnly]
    public void Log1p_ReplaysNumPyCorpus_BitForBit()
    {
        RequireWindows();
        foreach (JsonElement entry in LoadCorpus().GetProperty("log1p").EnumerateArray())
        {
            double x = FromBits(entry.GetProperty("x")), expected = FromBits(entry.GetProperty("y"));
            UlpDistance(MathUtil.Log1p(x), expected).Should().Be(0, $"log1p({x:R}) must be NumPy's {expected:R}");
        }
    }

    /// <summary>
    ///     <see cref="MathUtil.LogAddExp" /> reproduces Windows NumPy's <c>np.logaddexp</c> bit for bit over the 131-pair
    ///     corpus: near-equal pairs down to a 1e-17 gap, wide pairs, equal pairs and the infinities.
    /// </summary>
    [TestMethod]
    [WindowsOnly]
    public void LogAddExp_ReplaysNumPyCorpus_BitForBit()
    {
        RequireWindows();
        foreach (JsonElement entry in LoadCorpus().GetProperty("logaddexp").EnumerateArray())
        {
            double a = FromBits(entry.GetProperty("a")), b = FromBits(entry.GetProperty("b"));
            double expected = FromBits(entry.GetProperty("y"));
            UlpDistance(MathUtil.LogAddExp(a, b), expected).Should().Be(0, $"logaddexp({a:R}, {b:R}) must be NumPy's {expected:R}");
        }
    }

    /// <summary>
    ///     <see cref="MathUtil.Expm1" /> reproduces Windows NumPy's <c>np.expm1</c> bit for bit outside
    ///     <c>[-ln 2, ln 1.5]</c> (where the UCRT returns <c>exp(x) - 1</c>) and within 2 ulp inside it, over the 233-input
    ///     corpus, which includes both ranges <see cref="TruncNorm.NdtriExp" /> calls it on. The former Kahan form failed
    ///     both halves: 2 ulp off outside the band, up to 4 inside.
    /// </summary>
    [TestMethod]
    [WindowsOnly]
    public void Expm1_ReplaysNumPyCorpus_ExactOutsideTheBand_WithinTwoUlpInside()
    {
        RequireWindows();
        JsonElement corpus = LoadCorpus();
        double low = FromBits(corpus.GetProperty("band").GetProperty("low"));
        double high = FromBits(corpus.GetProperty("band").GetProperty("high"));
        int inBand = 0, inBandExact = 0;
        foreach (JsonElement entry in corpus.GetProperty("expm1").EnumerateArray())
        {
            double x = FromBits(entry.GetProperty("x")), expected = FromBits(entry.GetProperty("y"));
            long ulps = UlpDistance(MathUtil.Expm1(x), expected);
            if (x < low || x > high)
            {
                ulps.Should().Be(0, $"expm1({x:R}) lies outside the band, where NumPy's bits are exp(x) - 1");
            }
            else
            {
                inBand++;
                if (ulps == 0)
                    inBandExact++;
                ulps.Should().BeLessThanOrEqualTo(2, $"expm1({x:R}) lies inside the band, within 2 ulp of the UCRT polynomial");
            }
        }
        // The band must actually have been exercised, or the bound above proved nothing.
        inBand.Should().BeGreaterThan(100);
        inBandExact.Should().BeGreaterThan(0);
    }
}
