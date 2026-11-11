using System;
using System.Collections.Generic;
using System.Linq;
using AwesomeAssertions;
using OptunaSharp.Internal;
using OptunaSharp.Samplers.Tpe;

namespace OptunaSharp.Tests.Samplers;

/// <summary>
///     Plan 15 (docs/plans/15-performance-pass-3.md), step B: TPE's log density finds the unique candidate values, kernels
///     and cell bounds of a stepped coordinate by hashing instead of sorting with LINQ, and adds the continuous dimensions
///     with each candidate's sums held in registers and unchecked vector loads. Every bit must stay. Each test compares
///     the new code with the code it replaced, written out here: the discrete branch over the sorted dedups, the sorted
///     dedup itself, and the scalar accumulation the vector path reproduces.
/// </summary>
[TestClass]
public class TpeDiscreteKernelTests
{
    // ------------------------------------------------------------------ the discrete log-pdf

    /// <summary>
    ///     Mixtures of stepped dimensions (integers, log integers and stepped floats), from one kernel to four hundred,
    ///     with kernels that share centers and widths (as clipped TPE kernels do) and candidates that repeat, against the
    ///     replaced branch over sorted dedups: every candidate's log density has the same bits.
    /// </summary>
    /// <param name="seed">Seeds the random mixtures.</param>
    [TestMethod]
    [DataRow(1)]
    [DataRow(2)]
    [DataRow(3)]
    public void DiscreteLogPdf_HashedDedups_MatchTheSortedCodeToTheBit(int seed)
    {
        var rng = new Random(seed);
        foreach (int nKernels in new[] { 1, 2, 5, 26, 101, 400 })
        {
            foreach (int nDims in new[] { 1, 3 })
            {
                var mixture = RandomDiscreteMixture(rng, nKernels, nDims, distinctSigmas: rng.Next(2) == 0);
                AssertSameBits(mixture, RandomDiscreteCandidates(rng, mixture, 24));
            }
        }
    }

    /// <summary>
    ///     The number of unique central cell bounds picks the <c>erf</c> path (exact below 2,000, Optuna's vectorized
    ///     approximation from there), so the hashed dedup must find exactly as many as the sorted one. One mixture whose
    ///     candidate-by-kernel grid has more than 2,000 unique central pairs and one with fewer, both checked to be on
    ///     their side of the switch with the sorted dedup, against the replaced branch. A cell is central only when a
    ///     kernel's center lies inside it, so the large case centers its 2,400 kernels on the candidates' values and gives
    ///     each its own width.
    /// </summary>
    [TestMethod]
    public void DiscreteLogPdf_BothSidesOfTheErfSwitch_MatchTheSortedCodeToTheBit()
    {
        var rng = new Random(41);
        const int nLarge = 2400;
        var mu = new double[nLarge];
        var sigma = new double[nLarge];
        var weights = new double[nLarge];
        for (int k = 0; k < nLarge; k++)
        {
            mu[k] = -48 + 4 * (k % 24);
            sigma[k] = 0.5 + rng.NextDouble() * 20;
            weights[k] = 1.0 / nLarge;
        }
        var large = new MixtureOfProductDistribution(
            weights, new List<BatchedDistributions> { new BatchedDiscreteTruncNormDistributions(mu, sigma, -50, 50, 1) });
        double[,] xLarge = DistinctCandidates(large, 24);
        CentralUniqueCount(large, xLarge).Should().BeGreaterThanOrEqualTo(2000, "this case must take the vectorized erf");
        AssertSameBits(large, xLarge);

        // 24 distinct candidates x 30 kernels: at most 720 unique pairs.
        var small = RandomDiscreteMixture(rng, 30, 1, distinctSigmas: true);
        double[,] xSmall = DistinctCandidates(small, 24);
        CentralUniqueCount(small, xSmall).Should().BeLessThan(2000, "this case must take the exact erf");
        AssertSameBits(small, xSmall);
    }

    // ------------------------------------------------------------------ the hashed dedup

    /// <summary>
    ///     <see cref="MixtureOfProductDistribution.UniqueInverse" /> groups pairs exactly as the sorted dedup it replaced:
    ///     the same number of groups, the same positions in each group (the two inverses differ only by a renumbering),
    ///     and the same representative bits, the group's first occurrence. The pairs come from a small pool so they repeat,
    ///     and the pool holds both zeros (equal, so grouped), NaN (equal to nothing, so a group of its own each time), and
    ///     both infinities.
    /// </summary>
    /// <param name="n">The number of pairs.</param>
    [TestMethod]
    [DataRow(0)]
    [DataRow(1)]
    [DataRow(2)]
    [DataRow(9)]
    [DataRow(300)]
    [DataRow(5000)]
    public void UniqueInverse_GroupsAsTheSortedDedup_KeepingFirstOccurrences(int n)
    {
        var rng = new Random(n + 3);
        double[] pool = { -0.0, 0.0, 1.5, -2.25, double.PositiveInfinity, double.NegativeInfinity, double.NaN, 3.0, 1e-300 };
        var a = new double[n];
        var b = new double[n];
        for (int i = 0; i < n; i++)
        {
            a[i] = pool[rng.Next(pool.Length)];
            b[i] = pool[rng.Next(pool.Length)];
        }

        var uniqueA = new double[n];
        var uniqueB = new double[n];
        var inverse = new int[n];
        int count = MixtureOfProductDistribution.UniqueInverse(a, b, uniqueA, uniqueB, inverse);
        var (sortedA, sortedB, sortedInverse) = SortedUniqueInverse2d(a, b);

        count.Should().Be(sortedA.Length);
        // The new numbering maps one-to-one onto the sorted numbering, and every position agrees with it.
        var toSorted = Enumerable.Repeat(-1, count).ToArray();
        for (int i = 0; i < n; i++)
        {
            if (toSorted[inverse[i]] < 0)
            {
                toSorted[inverse[i]] = sortedInverse[i];
                // The first position of a group is its representative: same bits in both dedups.
                BitConverter.DoubleToInt64Bits(uniqueA[inverse[i]]).Should().Be(BitConverter.DoubleToInt64Bits(a[i]), $"position {i}");
                BitConverter.DoubleToInt64Bits(uniqueB[inverse[i]]).Should().Be(BitConverter.DoubleToInt64Bits(b[i]), $"position {i}");
                BitConverter.DoubleToInt64Bits(sortedA[sortedInverse[i]]).Should().Be(BitConverter.DoubleToInt64Bits(a[i]), $"position {i}");
                BitConverter.DoubleToInt64Bits(sortedB[sortedInverse[i]]).Should().Be(BitConverter.DoubleToInt64Bits(b[i]), $"position {i}");
            }
            toSorted[inverse[i]].Should().Be(sortedInverse[i], $"position {i} is grouped as the sorted dedup groups it");
        }
        toSorted.Distinct().Count().Should().Be(count, "different groups stay different");
    }

    // ------------------------------------------------------------------ the continuous accumulation

    /// <summary>
    ///     <see cref="MixtureOfProductDistribution.AccumulateContinuous" />, on its vector branch (whole kernel blocks plus
    ///     the leftover kernels) and on its scalar branch (fewer kernels than a vector holds), against the per-candidate
    ///     scalar accumulation it replaced: every kernel count from one to three vectors and three more, one to ten
    ///     dimensions, candidates outside some kernels' intervals (<c>-inf</c> terms), empty intervals (NaN terms) and rows
    ///     that already hold values. Same bits, NaNs where there were NaNs.
    /// </summary>
    /// <param name="seed">Seeds the random grids.</param>
    [TestMethod]
    [DataRow(1)]
    [DataRow(2)]
    public void AccumulateContinuous_BothBranches_MatchTheReplacedScalarLoopToTheBit(int seed)
    {
        var rng = new Random(seed);
        int width = System.Numerics.Vector<double>.Count;
        foreach (int nCont in new[] { 1, 3, 10 })
        {
            for (int nKernels = 1; nKernels <= 3 * width + 3; nKernels++)
            {
                const int nSamples = 7;
                int gridSize = nCont * nKernels;
                var mu = new double[gridSize];
                var sigma = new double[gridSize];
                var a = new double[gridSize];
                var b = new double[gridSize];
                var logSigma = new double[gridSize];
                for (int g = 0; g < gridSize; g++)
                {
                    mu[g] = rng.NextDouble() * 4 - 2;
                    sigma[g] = 0.05 + rng.NextDouble();
                    double low = -3.0, high = rng.Next(12) == 0 ? -3.0 : 3.0;  // one interval in twelve is empty.
                    a[g] = (low - mu[g]) / sigma[g];
                    b[g] = (high - mu[g]) / sigma[g];
                    logSigma[g] = Math.Log(sigma[g]);
                }
                double[] logGaussMass = TruncNorm.LogGaussMass(a, b);
                var xCont = new double[nSamples * nCont];
                for (int i = 0; i < xCont.Length; i++)
                    xCont[i] = rng.NextDouble() * 8 - 4;  // some candidates fall outside [-3, 3].
                var start = new double[nSamples * nKernels];
                for (int i = 0; i < start.Length; i++)
                    start[i] = rng.NextDouble() * 10 - 5;

                var actual = (double[])start.Clone();
                MixtureOfProductDistribution.AccumulateContinuous(
                    xCont, nSamples, nCont, mu, sigma, a, b, logGaussMass, logSigma, actual, nKernels);
                double[] expected = ReferenceAccumulate(xCont, nSamples, nCont, mu, sigma, a, b, logGaussMass, logSigma, start, nKernels);

                for (int i = 0; i < expected.Length; i++)
                {
                    if (double.IsNaN(expected[i]))
                        double.IsNaN(actual[i]).Should().BeTrue($"element {i} was NaN before ({nKernels} kernels, {nCont} dims)");
                    else
                        BitConverter.DoubleToInt64Bits(actual[i]).Should().Be(BitConverter.DoubleToInt64Bits(expected[i]),
                            $"element {i}: {actual[i]:R} vs {expected[i]:R} ({nKernels} kernels, {nCont} dims)");
                }
            }
        }
    }

    /// <summary>
    ///     The vector branch reads and writes without bounds checks, so the method checks every length first: a grid one
    ///     element short, or a row buffer too short, is refused before anything is touched.
    /// </summary>
    [TestMethod]
    public void AccumulateContinuous_ShortBuffers_AreRefusedUpFront()
    {
        const int nSamples = 3, nCont = 2, nKernels = 9;
        var grid = new double[nCont * nKernels];
        var shortGrid = new double[nCont * nKernels - 1];
        var xCont = new double[nSamples * nCont];
        var rows = new double[nSamples * nKernels];

        Action shortSigma = () => MixtureOfProductDistribution.AccumulateContinuous(
            xCont, nSamples, nCont, grid, shortGrid, grid, grid, grid, grid, rows, nKernels);
        shortSigma.Should().Throw<ArgumentException>();

        Action shortRows = () => MixtureOfProductDistribution.AccumulateContinuous(
            xCont, nSamples, nCont, grid, grid, grid, grid, grid, grid, new double[nSamples * nKernels - 1], nKernels);
        shortRows.Should().Throw<ArgumentException>();

        Action shortX = () => MixtureOfProductDistribution.AccumulateContinuous(
            new double[nSamples * nCont - 1], nSamples, nCont, grid, grid, grid, grid, grid, grid, rows, nKernels);
        shortX.Should().Throw<ArgumentException>();
    }

    // ------------------------------------------------------------------ references: the replaced code

    /// <summary>
    ///     The accumulation as it was computed before plan 15: per candidate, a buffer of one sum per kernel starts at
    ///     +0.0, receives each dimension's term in dimension order, and is then added to the candidate's row.
    /// </summary>
    /// <param name="xCont">The candidates' coordinates, one row of <paramref name="nCont" /> per candidate.</param>
    /// <param name="nSamples">The number of candidates.</param>
    /// <param name="nCont">The number of dimensions.</param>
    /// <param name="mu">The kernels' centers, one row per dimension.</param>
    /// <param name="sigma">The kernels' widths, one row per dimension.</param>
    /// <param name="a">The standardized lower bounds, one row per dimension.</param>
    /// <param name="b">The standardized upper bounds, one row per dimension.</param>
    /// <param name="logGaussMass">The log masses between the bounds, one row per dimension.</param>
    /// <param name="logSigma">The kernels' log widths, one row per dimension.</param>
    /// <param name="start">The rows before the accumulation; not modified.</param>
    /// <param name="nKernels">The number of kernels.</param>
    /// <returns>The rows after the accumulation.</returns>
    private static double[] ReferenceAccumulate(
        double[] xCont, int nSamples, int nCont, double[] mu, double[] sigma, double[] a, double[] b,
        double[] logGaussMass, double[] logSigma, double[] start, int nKernels)
    {
        var rows = (double[])start.Clone();
        var sums = new double[nKernels];
        for (int s = 0; s < nSamples; s++)
        {
            Array.Clear(sums);
            for (int c = 0; c < nCont; c++)
            {
                for (int k = 0; k < nKernels; k++)
                {
                    int g = c * nKernels + k;
                    double xstd = (xCont[s * nCont + c] - mu[g]) / sigma[g];
                    double val = TruncNorm.NormLogPdf(xstd) - logGaussMass[g] - logSigma[g];
                    if (a[g] == b[g])
                        val = double.NaN;
                    else if (xstd < a[g] || xstd > b[g])
                        val = double.NegativeInfinity;
                    sums[k] += val;
                }
            }
            for (int k = 0; k < nKernels; k++)
                rows[s * nKernels + k] += sums[k];
        }
        return rows;
    }

    /// <summary>
    ///     The log-pdf of a mixture of stepped dimensions as computed before plan 15: the discrete branch over LINQ-sorted
    ///     dedups (<c>np.unique</c>, <c>_unique_inverse_2d</c>, <c>_log_gauss_mass_unique</c>), then the log mixture weights
    ///     and the log-sum-exp.
    /// </summary>
    /// <param name="mixture">A mixture of stepped dimensions only.</param>
    /// <param name="x">The candidates, one row each.</param>
    /// <returns>One log density per candidate.</returns>
    private static double[] ReferenceDiscreteLogPdf(MixtureOfProductDistribution mixture, double[,] x)
    {
        int nSamples = x.GetLength(0), nKernels = mixture.Weights.Length;
        var weighted = new double[nSamples * nKernels];
        for (int i = 0; i < mixture.Distributions.Count; i++)
        {
            var dtn = (BatchedNumericalDistributions)mixture.Distributions[i];
            bool isLog = dtn.IsLog;
            double step = dtn.Step;
            var xcol = new double[nSamples];
            for (int s = 0; s < nSamples; s++)
                xcol[s] = x[s, i];

            var (xiUniq, xiInv) = SortedUniqueInverse1d(xcol);
            var (muUniq, sigmaUniq, msInv) = SortedUniqueInverse2d(dtn.Mu, dtn.Sigma);
            int nXi = xiUniq.Length;
            int nMs = muUniq.Length;

            var a = new double[nXi * nMs];
            var b = new double[nXi * nMs];
            for (int p = 0; p < nXi; p++)
            {
                double left = isLog ? Math.Log(xiUniq[p] - step / 2.0) : xiUniq[p] - step / 2.0;
                double right = isLog ? Math.Log(xiUniq[p] + step / 2.0) : xiUniq[p] + step / 2.0;
                for (int qi = 0; qi < nMs; qi++)
                {
                    a[p * nMs + qi] = (left - muUniq[qi]) / sigmaUniq[qi];
                    b[p * nMs + qi] = (right - muUniq[qi]) / sigmaUniq[qi];
                }
            }

            var logMass = SortedLogGaussMassUnique(a, b);
            for (int s = 0; s < nSamples; s++)
            {
                int row = xiInv[s] * nMs;
                for (int k = 0; k < nKernels; k++)
                    weighted[s * nKernels + k] += logMass[row + msInv[k]];
            }

            var normLow = new double[nMs];
            var normHigh = new double[nMs];
            for (int qi = 0; qi < nMs; qi++)
            {
                normLow[qi] = (dtn.AdaptedLow - muUniq[qi]) / sigmaUniq[qi];
                normHigh[qi] = (dtn.AdaptedHigh - muUniq[qi]) / sigmaUniq[qi];
            }
            var norm = TruncNorm.LogGaussMass(normLow, normHigh);
            for (int s = 0; s < nSamples; s++)
            {
                for (int k = 0; k < nKernels; k++)
                    weighted[s * nKernels + k] -= norm[msInv[k]];
            }
        }

        for (int s = 0; s < nSamples; s++)
            for (int k = 0; k < nKernels; k++)
                weighted[s * nKernels + k] += Math.Log(mixture.Weights[k]);

        var result = new double[nSamples];
        var expBuf = new double[nKernels];
        for (int s = 0; s < nSamples; s++)
        {
            double max = double.NegativeInfinity;
            for (int k = 0; k < nKernels; k++)
                max = Math.Max(max, weighted[s * nKernels + k]);
            if (double.IsNegativeInfinity(max))
                max = 0.0;
            for (int k = 0; k < nKernels; k++)
                expBuf[k] = Math.Exp(weighted[s * nKernels + k] - max);
            result[s] = Math.Log(MathUtil.PairwiseSum(expBuf)) + max;
        }
        return result;
    }

    /// <summary>The replaced <c>_log_gauss_mass_unique</c>: the sorted dedup, the mass per unique pair, scattered back.</summary>
    /// <param name="a">The lower bounds.</param>
    /// <param name="b">The upper bounds.</param>
    /// <returns>The log mass of every input pair.</returns>
    private static double[] SortedLogGaussMassUnique(double[] a, double[] b)
    {
        var (aUniq, bUniq, inv) = SortedUniqueInverse2d(a, b);
        var lgmUniq = TruncNorm.LogGaussMass(aUniq, bUniq);
        var result = new double[a.Length];
        for (int j = 0; j < a.Length; j++)
            result[j] = lgmUniq[inv[j]];
        return result;
    }

    /// <summary>The replaced <c>np.unique(x, return_inverse=True)</c>: a stable LINQ sort, then neighbors compared with <c>!=</c>.</summary>
    /// <param name="x">The values.</param>
    /// <returns>The sorted unique values and each position's index among them.</returns>
    private static (double[] Uniq, int[] Inv) SortedUniqueInverse1d(double[] x)
    {
        int n = x.Length;
        var order = Enumerable.Range(0, n).OrderBy(i => x[i]).ToArray();
        var uniq = new List<double>();
        var inv = new int[n];
        for (int k = 0; k < n; k++)
        {
            int idx = order[k];
            if (k == 0 || x[idx] != x[order[k - 1]])
                uniq.Add(x[idx]);
            inv[idx] = uniq.Count - 1;
        }
        return (uniq.ToArray(), inv);
    }

    /// <summary>The replaced <c>_unique_inverse_2d(a, b)</c>: a stable LINQ sort by (a, b), then neighbors compared with <c>!=</c>.</summary>
    /// <param name="a">The pairs' first components.</param>
    /// <param name="b">The pairs' second components.</param>
    /// <returns>The sorted unique pairs and each position's index among them.</returns>
    private static (double[] AUniq, double[] BUniq, int[] Inv) SortedUniqueInverse2d(double[] a, double[] b)
    {
        int n = a.Length;
        var order = Enumerable.Range(0, n).OrderBy(i => a[i]).ThenBy(i => b[i]).ToArray();
        var aUniq = new List<double>();
        var bUniq = new List<double>();
        var inv = new int[n];
        for (int k = 0; k < n; k++)
        {
            int idx = order[k];
            int prev = order[k - 1 < 0 ? 0 : k - 1];
            if (k == 0 || a[idx] != a[prev] || b[idx] != b[prev])
            {
                aUniq.Add(a[idx]);
                bUniq.Add(b[idx]);
            }
            inv[idx] = aUniq.Count - 1;
        }
        return (aUniq.ToArray(), bUniq.ToArray(), inv);
    }

    // ------------------------------------------------------------------ helpers

    /// <summary>Compares the new <see cref="MixtureOfProductDistribution.LogPdf" /> with the replaced discrete code, bit for bit.</summary>
    /// <param name="mixture">A mixture of stepped dimensions only.</param>
    /// <param name="x">The candidates, one row each.</param>
    private static void AssertSameBits(MixtureOfProductDistribution mixture, double[,] x)
    {
        double[] actual = mixture.LogPdf(x);
        double[] expected = ReferenceDiscreteLogPdf(mixture, x);
        actual.Length.Should().Be(expected.Length);
        for (int s = 0; s < expected.Length; s++)
        {
            if (double.IsNaN(expected[s]))
                double.IsNaN(actual[s]).Should().BeTrue($"candidate {s} was NaN before");
            else
                BitConverter.DoubleToInt64Bits(actual[s]).Should().Be(BitConverter.DoubleToInt64Bits(expected[s]),
                    $"candidate {s}: {actual[s]:R} vs {expected[s]:R} ({mixture.Weights.Length} kernels, {x.GetLength(1)} dims)");
        }
    }

    /// <summary>
    ///     A random mixture of stepped dimensions, cycling through integers on [-50, 50], log integers on [1, 1000] and
    ///     floats on [0, 10] with step 0.25. Centers are values of the domain, so many repeat; widths repeat a common
    ///     clipped value unless <paramref name="distinctSigmas" />, as TPE's clipped kernels do.
    /// </summary>
    /// <param name="rng">The source of randomness.</param>
    /// <param name="nKernels">The number of kernels.</param>
    /// <param name="nDims">The number of dimensions.</param>
    /// <param name="distinctSigmas">Whether every kernel gets its own width.</param>
    /// <returns>The mixture.</returns>
    private static MixtureOfProductDistribution RandomDiscreteMixture(Random rng, int nKernels, int nDims, bool distinctSigmas)
    {
        var dims = new List<BatchedDistributions>();
        for (int d = 0; d < nDims; d++)
        {
            var mu = new double[nKernels];
            var sigma = new double[nKernels];
            switch (d % 3)
            {
                case 0:
                    for (int k = 0; k < nKernels; k++)
                    {
                        mu[k] = rng.Next(-50, 51);
                        sigma[k] = distinctSigmas ? 0.5 + rng.NextDouble() * 20 : (rng.Next(4) == 0 ? 7.0 : 1.01);
                    }
                    dims.Add(new BatchedDiscreteTruncNormDistributions(mu, sigma, -50, 50, 1));
                    break;
                case 1:
                    for (int k = 0; k < nKernels; k++)
                    {
                        mu[k] = Math.Log(rng.Next(1, 1001));
                        sigma[k] = distinctSigmas ? 0.05 + rng.NextDouble() * 2 : (rng.Next(4) == 0 ? 1.3 : 0.069);
                    }
                    dims.Add(new BatchedDiscreteTruncLogNormDistributions(mu, sigma, 1, 1000, 1));
                    break;
                default:
                    for (int k = 0; k < nKernels; k++)
                    {
                        mu[k] = rng.Next(0, 41) * 0.25;
                        sigma[k] = distinctSigmas ? 0.1 + rng.NextDouble() * 5 : (rng.Next(4) == 0 ? 2.5 : 0.1);
                    }
                    dims.Add(new BatchedDiscreteTruncNormDistributions(mu, sigma, 0, 10, 0.25));
                    break;
            }
        }
        var weights = new double[nKernels];
        double sum = 0;
        for (int k = 0; k < nKernels; k++)
            sum += weights[k] = 0.1 + rng.NextDouble();
        for (int k = 0; k < nKernels; k++)
            weights[k] /= sum;
        return new MixtureOfProductDistribution(weights, dims);
    }

    /// <summary>Candidates on each dimension's grid of values; with 24 draws from a narrow band, some repeat.</summary>
    /// <param name="rng">The source of randomness.</param>
    /// <param name="mixture">The mixture whose dimensions the candidates follow.</param>
    /// <param name="nSamples">The number of candidates.</param>
    /// <returns>The candidates, one row each.</returns>
    private static double[,] RandomDiscreteCandidates(Random rng, MixtureOfProductDistribution mixture, int nSamples)
    {
        int nDims = mixture.Distributions.Count;
        var x = new double[nSamples, nDims];
        for (int s = 0; s < nSamples; s++)
        {
            for (int d = 0; d < nDims; d++)
            {
                x[s, d] = (d % 3) switch
                {
                    0 => rng.Next(-12, 13),
                    1 => rng.Next(1, 30),
                    _ => rng.Next(10, 31) * 0.25,
                };
            }
        }
        return x;
    }

    /// <summary>Distinct candidates spread over the first dimension's domain.</summary>
    /// <param name="mixture">A one-dimensional integer mixture.</param>
    /// <param name="nSamples">The number of candidates.</param>
    /// <returns>The candidates, one row each.</returns>
    private static double[,] DistinctCandidates(MixtureOfProductDistribution mixture, int nSamples)
    {
        var x = new double[nSamples, mixture.Distributions.Count];
        for (int s = 0; s < nSamples; s++)
            x[s, 0] = -48 + 4 * s;
        return x;
    }

    /// <summary>
    ///     How many unique central cell bounds (<c>a &lt;= 0 &lt; b</c>) the sorted dedup finds for the first dimension:
    ///     the count that picks the <c>erf</c> path.
    /// </summary>
    /// <param name="mixture">A mixture whose first dimension is stepped and not log.</param>
    /// <param name="x">The candidates.</param>
    /// <returns>The number of unique central pairs.</returns>
    private static int CentralUniqueCount(MixtureOfProductDistribution mixture, double[,] x)
    {
        var dtn = (BatchedNumericalDistributions)mixture.Distributions[0];
        var xcol = Enumerable.Range(0, x.GetLength(0)).Select(s => x[s, 0]).ToArray();
        var (xiUniq, _) = SortedUniqueInverse1d(xcol);
        var (muUniq, sigmaUniq, _) = SortedUniqueInverse2d(dtn.Mu, dtn.Sigma);
        var a = new List<double>();
        var b = new List<double>();
        foreach (double xi in xiUniq)
        {
            for (int q = 0; q < muUniq.Length; q++)
            {
                a.Add((xi - dtn.Step / 2.0 - muUniq[q]) / sigmaUniq[q]);
                b.Add((xi + dtn.Step / 2.0 - muUniq[q]) / sigmaUniq[q]);
            }
        }
        var (aUniq, bUniq, _) = SortedUniqueInverse2d(a.ToArray(), b.ToArray());
        return Enumerable.Range(0, aUniq.Length).Count(i => !(aUniq[i] > 0.0) && !(bUniq[i] <= 0.0));
    }
}
