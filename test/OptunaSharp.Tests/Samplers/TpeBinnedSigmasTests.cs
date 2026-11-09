using System;
using System.Collections.Generic;
using AwesomeAssertions;
using OptunaSharp.Internal;
using OptunaSharp.Samplers.Tpe;

namespace OptunaSharp.Tests.Samplers;

/// <summary>
///     Plan 14: TPE's kernel widths come from <see cref="ParzenEstimator.BinnedRawSigmas" /> whenever it can prove the result
///     equals Optuna's sort-based <c>compute_sigmas</c> (<see cref="ParzenEstimator.SortedRawSigmas" />). These tests compare
///     the two paths bit for bit, after the shared clip, over many generated center sets: uniform, tied (integer grids,
///     repeated values), clustered, on and next to bin edges, at the domain ends, tiny and huge domains, with and without
///     endpoints; and they check the binned path really runs on ordinary inputs (a fallback-only implementation would pass
///     the equality tests vacuously).
/// </summary>
[TestClass]
public class TpeBinnedSigmasTests
{
    /// <summary>The clip <see cref="ParzenEstimator.ComputeSigmas" /> applies to both paths' unclipped widths.</summary>
    /// <param name="raw">Unclipped widths; clipped in place.</param>
    /// <param name="minsigma">The magic-clip floor.</param>
    /// <param name="maxsigma">The domain width.</param>
    /// <returns>The same array, clipped.</returns>
    private static double[] Clip(double[] raw, double minsigma, double maxsigma)
    {
        for (int i = 0; i < raw.Length; i++)
            raw[i] = MathUtil.Clip(raw[i], minsigma, maxsigma);
        return raw;
    }

    /// <summary>
    ///     Compares both paths for one center set: when the binned path answers, its clipped widths must equal the sort's
    ///     bit for bit.
    /// </summary>
    /// <param name="mus">The centers.</param>
    /// <param name="low">The domain's lower end.</param>
    /// <param name="high">The domain's upper end.</param>
    /// <param name="considerEndpoints">Whether the endpoints count as neighbors.</param>
    /// <param name="label">Text identifying the case in a failure message.</param>
    /// <returns>Whether the binned path answered (false when it fell back).</returns>
    private static bool CompareOnce(double[] mus, double low, double high, bool considerEndpoints, string label)
    {
        int n = mus.Length;
        double divisor = Math.Min(100.0, 1.0 + (n + 1));
        double minsigma = (high - low) / divisor;
        double maxsigma = high - low;
        double[] expected = Clip(ParzenEstimator.SortedRawSigmas(mus, low, high, considerEndpoints), minsigma, maxsigma);
        double[]? binned = ParzenEstimator.BinnedRawSigmas(mus, low, high, minsigma, (int)divisor, considerEndpoints);
        if (binned is null)
            return false;
        Clip(binned, minsigma, maxsigma);
        for (int i = 0; i < n; i++)
        {
            if (BitConverter.DoubleToInt64Bits(binned[i]) != BitConverter.DoubleToInt64Bits(expected[i]))
                Assert.Fail($"{label}: width {i} of {n} is {binned[i]:R} binned, {expected[i]:R} sorted (center {mus[i]:R})");
        }
        return true;
    }

    /// <summary>The domains TPE meets: plain floats, log-transformed bounds, step-widened ints, unit cubes, tiny and huge spans.</summary>
    /// <returns>(low, high) pairs.</returns>
    private static IEnumerable<(double Low, double High)> Domains() => new[]
    {
        (-10.0, 10.0), (0.0, 1.0), (Math.Log(1e-5), Math.Log(1e-1)), (0.5, 100.5), (-0.5, 20.5), (Math.Log(0.5), Math.Log(128.5)),
        (-1e300, 1e300), (1e-300, 3e-300), (123456.789, 123456.791), (-3.0, -2.999999),
    };

    /// <summary>
    ///     Uniform, tied, clustered, edge and endpoint center sets of many sizes over every domain, both endpoint settings:
    ///     whenever the binned path answers it equals the sort, and on uniform inputs it answers almost always.
    /// </summary>
    /// <param name="seed">The generator seed for the center sets.</param>
    [TestMethod]
    [DataRow(1)]
    [DataRow(2)]
    [DataRow(3)]
    public void BinnedWidths_EqualSortedWidths_BitForBit(int seed)
    {
        var rng = new Random(seed);
        int uniformCases = 0, uniformAnswered = 0, answered = 0, cases = 0;
        int[] sizes = { 2, 3, 4, 5, 10, 25, 26, 97, 98, 99, 150, 500, 975, 1500 };
        foreach (var (low, high) in Domains())
        {
            foreach (int n in sizes)
            {
                double divisor = Math.Min(100.0, 1.0 + (n + 1));
                double binWidth = (high - low) / divisor;
                for (int kind = 0; kind < 7; kind++)
                {
                    var mus = new double[n];
                    for (int i = 0; i < n; i++)
                    {
                        double u = rng.NextDouble();
                        mus[i] = kind switch
                        {
                            // Uniform over the domain.
                            0 => low + (high - low) * u,
                            // A coarse grid: many exact ties, like integer or stepped parameters.
                            1 => low + (high - low) * (rng.Next(0, 9) / 8.0),
                            // Three tight clusters: wide gaps between, tiny gaps within.
                            2 => low + (high - low) * (0.1 + 0.4 * rng.Next(0, 3)) + binWidth * 1e-3 * u,
                            // Exactly on bin edges.
                            3 => Math.Min(high, low + binWidth * rng.Next(0, (int)divisor + 1)),
                            // One ulp either side of a bin edge.
                            4 => Math.Clamp(Nudge(low + binWidth * rng.Next(1, (int)divisor), rng), low, high),
                            // The domain ends, often repeated.
                            5 => rng.Next(0, 3) switch { 0 => low, 1 => high, _ => low + (high - low) * u },
                            // Two values a bin width apart, repeated: neighbor distances at the floor.
                            _ => low + binWidth * (rng.Next(0, 2) + 0.5 * (n % 2)),
                        };
                    }
                    foreach (bool considerEndpoints in new[] { false, true })
                    {
                        cases++;
                        bool ok = CompareOnce(mus, low, high, considerEndpoints, $"seed {seed} domain [{low:R}, {high:R}] n {n} kind {kind} endpoints {considerEndpoints}");
                        if (ok)
                            answered++;
                        if (kind == 0)
                        {
                            uniformCases++;
                            if (ok)
                                uniformAnswered++;
                        }
                    }
                }
            }
        }

        // A fallback-only implementation would pass the comparisons above vacuously: the fast path must actually run.
        uniformAnswered.Should().BeGreaterThan(uniformCases * 9 / 10, "uniform centers almost never need the sort");
        answered.Should().BeGreaterThan(cases / 2);
    }

    /// <summary>Moves a value by one or two ulps up or down.</summary>
    /// <param name="value">The value to move.</param>
    /// <param name="rng">Chooses the direction and the distance.</param>
    /// <returns>The neighboring double.</returns>
    private static double Nudge(double value, Random rng)
    {
        double moved = rng.Next(0, 2) == 0 ? Math.BitDecrement(value) : Math.BitIncrement(value);
        return rng.Next(0, 2) == 0 ? moved : (moved > value ? Math.BitIncrement(moved) : Math.BitDecrement(moved));
    }

    /// <summary>
    ///     The cases the binned path must hand back to the sort: fewer than two centers, a center outside the domain or NaN,
    ///     a degenerate domain, and a bin whose computed span exceeds the floor.
    /// </summary>
    [TestMethod]
    public void BinnedWidths_FallBack_WhereTheProofDoesNotHold()
    {
        ParzenEstimator.BinnedRawSigmas(new[] { 0.5 }, 0.0, 1.0, 1.0 / 3.0, 3, false).Should().BeNull("one center");
        ParzenEstimator.BinnedRawSigmas(new[] { 0.5, 1.5 }, 0.0, 1.0, 0.25, 4, false).Should().BeNull("a center above the domain");
        ParzenEstimator.BinnedRawSigmas(new[] { -0.1, 0.5 }, 0.0, 1.0, 0.25, 4, false).Should().BeNull("a center below the domain");
        ParzenEstimator.BinnedRawSigmas(new[] { double.NaN, 0.5 }, 0.0, 1.0, 0.25, 4, false).Should().BeNull("NaN");
        ParzenEstimator.BinnedRawSigmas(new[] { 1.0, 1.0 }, 1.0, 1.0, 0.0, 4, false).Should().BeNull("an empty domain");
        // A bin can span a little more than its width only through rounding (or, as here, when the bins do not cover the
        // domain and the last bin takes the rest); the guard refuses any span over the floor. Bins of width 0.1 but only
        // four of them: 0.35 and 0.9 both land in the last bin, 0.55 apart.
        ParzenEstimator.BinnedRawSigmas(new[] { 0.0, 0.35, 0.9 }, 0.0, 1.0, 0.1, 4, false).Should().BeNull("a bin spanning more than the floor");
    }

    /// <summary>
    ///     End to end through <see cref="ParzenEstimator.ComputeSigmas" />: the widths for a 975-center history (the size
    ///     of a 1,000-trial run's "above" group) equal the sort-based computation, with and without the magic clip (without
    ///     it the binned path is never used).
    /// </summary>
    /// <param name="considerMagicClip">Whether the magic clip is on (the only case the bins apply to).</param>
    /// <param name="considerEndpoints">Whether the endpoints count as neighbors.</param>
    [TestMethod]
    [DataRow(true, false)]
    [DataRow(true, true)]
    [DataRow(false, false)]
    public void ComputeSigmas_EqualsTheSortBasedComputation(bool considerMagicClip, bool considerEndpoints)
    {
        var rng = new Random(975);
        var mus = new double[975];
        for (int i = 0; i < mus.Length; i++)
            mus[i] = -5.0 + 10.0 * rng.NextDouble();
        var parameters = new ParzenEstimatorParameters(1.0, considerMagicClip, considerEndpoints, _ => Array.Empty<double>(), false);
        double[] actual = ParzenEstimator.ComputeSigmas(mus, -5.0, 5.0, mus.Length, parameters);

        double minsigma = considerMagicClip ? 10.0 / Math.Min(100.0, 1.0 + (mus.Length + 1)) : 1e-12;
        double[] expected = Clip(ParzenEstimator.SortedRawSigmas(mus, -5.0, 5.0, considerEndpoints), minsigma, 10.0);
        for (int i = 0; i < mus.Length; i++)
            BitConverter.DoubleToInt64Bits(actual[i]).Should().Be(BitConverter.DoubleToInt64Bits(expected[i]), $"width {i}");
    }
}
