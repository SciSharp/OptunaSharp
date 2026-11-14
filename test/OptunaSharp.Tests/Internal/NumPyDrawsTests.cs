using System;
using System.Collections.Generic;
using AwesomeAssertions;
using NumSharp;
using OptunaSharp.Internal;

namespace OptunaSharp.Tests.Internal;

/// <summary>
///     Plan 14: pins every <see cref="NumPyDraws" /> helper to the NumSharp call it replaces. Each test drives two
///     generators seeded alike, one through the helper and one through the NumSharp call the sampler code made before plan
///     14, and asserts the values are bit-identical and the streams stay in step afterwards (the next draws agree), so the
///     helper consumes exactly the words the call consumed. NumSharp's own calls are pinned to NumPy by its oracle; the
///     samplers built on the helpers are pinned to live Optuna by the interop parity suites.
/// </summary>
[TestClass]
public class NumPyDrawsTests
{
    /// <summary>Two generators seeded alike: the left is drawn through <see cref="NumPyDraws" />, the right through NumSharp.</summary>
    /// <param name="seed">The seed both generators start from.</param>
    /// <returns>The helper-side and the NumSharp-side generator.</returns>
    private static (NumPyRandom Helper, NumPyRandom NumSharp) Pair(int seed)
        => (np.random.RandomState(seed), np.random.RandomState(seed));

    /// <summary>Asserts both generators continue with the same words: a long run of plain uniforms must agree bit for bit.</summary>
    /// <param name="helper">The generator drawn through the helpers.</param>
    /// <param name="reference">The generator drawn through NumSharp.</param>
    private static void AssertInStep(NumPyRandom helper, NumPyRandom reference)
    {
        using NDArray expected = reference.random_sample(64L);
        double[] want = expected.ToArray<double>();
        for (int i = 0; i < want.Length; i++)
            BitConverter.DoubleToInt64Bits(NumPyDraws.Rand(helper)).Should().Be(BitConverter.DoubleToInt64Bits(want[i]),
                $"the streams must stay in step (draw {i} after the compared calls)");
    }

    /// <summary>A thousand scalar uniforms equal NumSharp's <c>rand()</c> bit for bit, and the streams stay in step.</summary>
    /// <param name="seed">The seed both generators start from.</param>
    [TestMethod]
    [DataRow(0)]
    [DataRow(42)]
    [DataRow(2147483646)]
    public void Rand_Scalar_MatchesNumSharpRand_AndKeepsTheStreamInStep(int seed)
    {
        var (helper, reference) = Pair(seed);
        for (int i = 0; i < 1000; i++)
        {
            using NDArray expected = reference.rand();
            BitConverter.DoubleToInt64Bits(NumPyDraws.Rand(helper)).Should().Be(BitConverter.DoubleToInt64Bits((double)expected));
        }
        AssertInStep(helper, reference);
    }

    /// <summary>A vector of uniforms equals NumSharp's <c>rand(n)</c> element by element, for empty, single and long vectors.</summary>
    /// <param name="n">The number of draws.</param>
    [TestMethod]
    [DataRow(0)]
    [DataRow(1)]
    [DataRow(7)]
    [DataRow(1000)]
    public void Rand_Vector_MatchesNumSharpRand(int n)
    {
        var (helper, reference) = Pair(123);
        using NDArray expected = reference.rand(n);
        double[] actual = NumPyDraws.Rand(helper, n);
        actual.Should().HaveCount(n);
        double[] want = expected.ToArray<double>();
        for (int i = 0; i < n; i++)
            BitConverter.DoubleToInt64Bits(actual[i]).Should().Be(BitConverter.DoubleToInt64Bits(want[i]), $"value {i} of {n}");
        AssertInStep(helper, reference);
    }

    /// <summary>
    ///     A vector of uniforms equals NumSharp's <c>random_sample(n)</c>, at TPE's sizes (24 candidates; 40 parameters × 24
    ///     candidates).
    /// </summary>
    /// <param name="n">The number of draws.</param>
    [TestMethod]
    [DataRow(0L)]
    [DataRow(24L)]
    [DataRow(960L)]
    public void RandomSample_MatchesNumSharpRandomSample(long n)
    {
        var (helper, reference) = Pair(20260830);
        using NDArray expected = reference.random_sample(n);
        double[] actual = NumPyDraws.RandomSample(helper, n);
        double[] want = expected.ToArray<double>();
        actual.Should().HaveCount(want.Length);
        for (int i = 0; i < want.Length; i++)
            BitConverter.DoubleToInt64Bits(actual[i]).Should().Be(BitConverter.DoubleToInt64Bits(want[i]), $"value {i}");
        AssertInStep(helper, reference);
    }

    /// <summary>A negative count raises NumPy's ValueError before any draw, so the stream does not move.</summary>
    [TestMethod]
    public void RandomSample_Negative_RaisesValueError_AndDrawsNothing()
    {
        var (helper, reference) = Pair(5);
        Action act = () => NumPyDraws.RandomSample(helper, -1);
        act.Should().Throw<ValueError>();
        AssertInStep(helper, reference);
    }

    /// <summary>
    ///     The bounds RandomSampler meets: plain, log-transformed and step-widened float and int bounds, one-hot categorical
    ///     rows, a zero-width range, reversed bounds and very wide ones.
    /// </summary>
    /// <returns>Pairs of (low, high).</returns>
    private static IEnumerable<(double Low, double High)> Bounds() => new[]
    {
        (-10.0, 10.0), (Math.Log(1e-5), Math.Log(1e-1)), (0.1 - 0.1, 0.9 + 0.1), (0.5, 100.5), (Math.Log(0.5), Math.Log(128.5)),
        (-2.5, 22.5), (0.0, 1.0), (3.25, 3.25), (5.0, -5.0), (-1e300, 1e300), (1e-300, 2e-300), (-0.0, 0.0),
    };

    /// <summary>Scalar uniforms over every bound shape RandomSampler meets equal NumSharp's scalar <c>uniform(low, high)</c>.</summary>
    /// <param name="seed">The seed both generators start from.</param>
    [TestMethod]
    [DataRow(0)]
    [DataRow(99)]
    public void Uniform_MatchesNumSharpScalarUniform(int seed)
    {
        var (helper, reference) = Pair(seed);
        for (int round = 0; round < 50; round++)
        {
            foreach (var (low, high) in Bounds())
            {
                using NDArray expected = reference.uniform(low, high);
                BitConverter.DoubleToInt64Bits(NumPyDraws.Uniform(helper, low, high))
                    .Should().Be(BitConverter.DoubleToInt64Bits((double)expected), $"uniform({low}, {high}), round {round}");
            }
        }
        AssertInStep(helper, reference);
    }

    /// <summary>
    ///     Drawing a bounds column row by row equals NumSharp's array form <c>uniform(bounds[:, 0], bounds[:, 1])</c>, which
    ///     RandomSampler used before plan 14: four one-hot rows of a categorical, and the single row of a numerical parameter.
    /// </summary>
    [TestMethod]
    public void Uniform_RowByRow_MatchesNumSharpArrayUniform_OverABoundsColumn()
    {
        double[,] categorical = { { 0, 1 }, { 0, 1 }, { 0, 1 }, { 0, 1 } };
        double[,] numerical = { { Math.Log(1e-3), Math.Log(10.0) } };
        foreach (double[,] bounds in new[] { categorical, numerical })
        {
            var (helper, reference) = Pair(77);
            for (int round = 0; round < 100; round++)
            {
                int rows = bounds.GetLength(0);
                var lows = new double[rows];
                var highs = new double[rows];
                for (int r = 0; r < rows; r++)
                {
                    lows[r] = bounds[r, 0];
                    highs[r] = bounds[r, 1];
                }
                using NDArray lowArray = np.array(lows);
                using NDArray highArray = np.array(highs);
                using NDArray expected = reference.uniform(lowArray, highArray);
                double[] want = expected.ToArray<double>();
                for (int r = 0; r < rows; r++)
                    BitConverter.DoubleToInt64Bits(NumPyDraws.Uniform(helper, bounds[r, 0], bounds[r, 1]))
                        .Should().Be(BitConverter.DoubleToInt64Bits(want[r]), $"row {r} of {rows}, round {round}");
            }
            AssertInStep(helper, reference);
        }
    }

    /// <summary>
    ///     A non-finite range raises NumPy's <c>OverflowError</c> (NumSharp's Python class since 0.81.0, which was .NET's
    ///     <see cref="OverflowException" /> in 0.80.0) with NumPy's text before any draw, as NumSharp's call does, and the
    ///     stream does not move.
    /// </summary>
    /// <param name="low">The lower bound.</param>
    /// <param name="high">The upper bound.</param>
    [TestMethod]
    [DataRow(double.MinValue, double.MaxValue)]
    [DataRow(0.0, double.PositiveInfinity)]
    [DataRow(double.NaN, 1.0)]
    public void Uniform_NonFiniteRange_RaisesOverflow_AndDrawsNothing(double low, double high)
    {
        var (helper, reference) = Pair(3);
        Action numSharp = () => reference.uniform(low, high).Dispose();
        numSharp.Should().ThrowExactly<OverflowError>().WithMessage("Range exceeds valid bounds");
        Action act = () => NumPyDraws.Uniform(helper, low, high);
        act.Should().ThrowExactly<OverflowError>().WithMessage("Range exceeds valid bounds");
        AssertInStep(helper, reference);
    }

    /// <summary>Integer draws equal NumSharp's <c>randint(low, high)</c>, including the seed ranges CMA-ES and GP draw from.</summary>
    /// <param name="low">The smallest value that can be drawn.</param>
    /// <param name="high">One past the largest value that can be drawn.</param>
    [TestMethod]
    [DataRow(0L, 2L)]
    [DataRow(0L, 50L)]
    [DataRow(1L, 65536L)]
    [DataRow(1L, 2147483646L)]
    public void RandInt_MatchesNumSharpRandint(long low, long high)
    {
        var (helper, reference) = Pair(11);
        for (int i = 0; i < 500; i++)
        {
            using NDArray expected = reference.randint(low, high);
            NumPyDraws.RandInt(helper, low, high).Should().Be((long)expected, $"draw {i}");
        }
        AssertInStep(helper, reference);
    }

    /// <summary>
    ///     Gaussians equal NumSharp's <c>normal</c>, scalar and vector, including the polar method's cached second value:
    ///     odd vector sizes, scalars and plain uniforms are interleaved so a cached Gaussian must be handed out on both sides.
    /// </summary>
    [TestMethod]
    public void Normal_Scalar_And_Vector_MatchNumSharp_IncludingTheCachedGaussian()
    {
        var (helper, reference) = Pair(42);
        for (int round = 0; round < 40; round++)
        {
            int n = 1 + 2 * (round % 5);
            using NDArray expectedVector = reference.normal(0.5, 0.35, new Shape(n));
            double[] actual = NumPyDraws.Normal(helper, 0.5, 0.35, n);
            double[] want = expectedVector.ToArray<double>();
            for (int i = 0; i < n; i++)
                BitConverter.DoubleToInt64Bits(actual[i]).Should().Be(BitConverter.DoubleToInt64Bits(want[i]), $"vector draw {i}, round {round}");

            using NDArray expectedScalar = reference.normal(-1.0, 4.0);
            BitConverter.DoubleToInt64Bits(NumPyDraws.Normal(helper, -1.0, 4.0))
                .Should().Be(BitConverter.DoubleToInt64Bits((double)expectedScalar), $"scalar draw, round {round}");

            using NDArray expectedUniform = reference.rand();
            BitConverter.DoubleToInt64Bits(NumPyDraws.Rand(helper))
                .Should().Be(BitConverter.DoubleToInt64Bits((double)expectedUniform), $"uniform between normals, round {round}");
        }
        AssertInStep(helper, reference);
    }

    /// <summary>
    ///     Every helper in the order NSGA-II's child generation uses them (a crossover decision, a parent tournament, a
    ///     crossover vector, a resample, per-gene decisions), repeated, equals the same sequence of NumSharp calls.
    /// </summary>
    [TestMethod]
    public void MixedSequence_OfEveryHelper_MatchesTheNumSharpSequence()
    {
        var (helper, reference) = Pair(2026);
        for (int round = 0; round < 200; round++)
        {
            using (NDArray d = reference.rand())
                NumPyDraws.Rand(helper).Should().Be((double)d);
            using (NDArray d = reference.randint(0, 20))
                NumPyDraws.RandInt(helper, 0, 20).Should().Be((long)d);
            using (NDArray d = reference.rand(8))
                NumPyDraws.Rand(helper, 8).Should().Equal(d.ToArray<double>());
            using (NDArray d = reference.uniform(-5.0, 5.0))
                NumPyDraws.Uniform(helper, -5.0, 5.0).Should().Be((double)d);
            using (NDArray d = reference.random_sample(3L))
                NumPyDraws.RandomSample(helper, 3).Should().Equal(d.ToArray<double>());
        }
        AssertInStep(helper, reference);
    }
}
