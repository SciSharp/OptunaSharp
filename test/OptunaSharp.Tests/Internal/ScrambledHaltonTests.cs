using System;
using AwesomeAssertions;
using NumSharp;
using OptunaSharp.Internal;

namespace OptunaSharp.Tests.Internal;

/// <summary>
///     Plan 15: <see cref="ScrambledHalton" /> draws a sequence's digit permutations once into an immutable engine that
///     QMCSampler keeps across trials, instead of re-drawing them for every point. These tests pin the engine to the
///     per-call algorithm it replaced, written out here, by bits. Parity with scipy itself is covered by
///     <c>QmcSamplerTests.ScrambledHalton*</c> and the live <c>QmcSamplerParityTests</c>.
/// </summary>
[TestClass]
public class ScrambledHaltonTests
{
    /// <summary>
    ///     The engine's points equal the replaced per-call algorithm's for several dimensions, seeds and indexes,
    ///     including indexes with many base-b digits.
    /// </summary>
    [TestMethod]
    public void EnginePoint_EqualsThePerCallAlgorithm()
    {
        foreach (var (dim, seed) in new[] { (1, 0), (3, 42), (10, 0), (20, 2024) })
        {
            ScrambledHalton.Engine engine = ScrambledHalton.Create(dim, seed);
            foreach (ulong n in new ulong[] { 0, 1, 2, 3, 7, 100, 4095, 4096, 99_991, 1_000_003, 123_456_789 })
            {
                double[] expected = ReplacedPoint(dim, n, seed);
                double[] actual = engine.Point(n);
                for (int d = 0; d < dim; d++)
                    BitConverter.DoubleToInt64Bits(actual[d]).Should().Be(BitConverter.DoubleToInt64Bits(expected[d]),
                        $"dim={dim}, seed={seed}, n={n}, coordinate {d}");
            }
        }
    }

    /// <summary>The static per-call entry point still answers what the engine answers.</summary>
    [TestMethod]
    public void StaticPoint_EqualsEnginePoint()
    {
        ScrambledHalton.Engine engine = ScrambledHalton.Create(6, 17);
        for (ulong n = 0; n < 50; n++)
            ScrambledHalton.Point(6, n, 17).Should().Equal(engine.Point(n));
    }

    /// <summary>
    ///     The algorithm <see cref="ScrambledHalton.Point" /> used before plan 15, kept verbatim as the reference:
    ///     the permutations are re-drawn from <c>default_rng(seed)</c> on every call.
    /// </summary>
    /// <param name="dim">The number of coordinates in each point.</param>
    /// <param name="n">The zero-based sequence index.</param>
    /// <param name="seed">The scramble seed.</param>
    /// <returns>The scrambled Halton point at <paramref name="n" />.</returns>
    private static double[] ReplacedPoint(int dim, ulong n, int seed)
    {
        int[] primes = QuasiMonteCarlo.HaltonPrimes(dim);
        Generator rng = np.random.default_rng(seed);

        var point = new double[dim];
        for (int d = 0; d < dim; d++)
        {
            int b = primes[d];
            int count = (int)Math.Ceiling(54.0 / Math.Log2(b)) - 1;
            var perms = new long[count][];
            for (int j = 0; j < count; j++)
            {
                NDArray row = np.arange((long)b);
                rng.shuffle(row);
                perms[j] = row.ToArray<long>();
            }

            double r = 0.0;
            double b2r = 1.0 / b;
            ulong quotient = n;
            for (int j = 0; j < count; j++)
            {
                int remainder = (int)(quotient % (ulong)b);
                remainder = (int)perms[j][remainder];
                r += remainder * b2r;
                b2r /= b;
                quotient /= (ulong)b;
            }
            point[d] = r;
        }
        return point;
    }
}
