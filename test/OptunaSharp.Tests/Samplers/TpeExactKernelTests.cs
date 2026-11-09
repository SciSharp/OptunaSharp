using System;
using System.Collections.Generic;
using System.Linq;
using AwesomeAssertions;
using OptunaSharp.Internal;
using OptunaSharp.Samplers.Tpe;

namespace OptunaSharp.Tests.Samplers;

/// <summary>
///     Plan 13 (docs/plans/13-rustuna-derived-performance.md), item S6: the TPE scoring kernel computes per (kernel,
///     dimension) what used to be computed per candidate, runs its kernel loop on vectors, rents its grids, and sorts the
///     kernels' centers without LINQ, all keeping every bit. Each test compares the new code with a straightforward copy
///     of the code it replaced, bit for bit, on inputs chosen to reach every branch: every vector-remainder size, the
///     exact and the vectorized <c>erf</c> paths, log dimensions, candidates outside a kernel's interval, empty intervals,
///     ties and NaNs in the sort.
/// </summary>
[TestClass]
public class TpeExactKernelTests
{
    // ------------------------------------------------------------------ the continuous log-pdf

    /// <summary>
    ///     Mixtures of continuous kernels (plain and log), with every kernel count from 1 to three vectors and a few more
    ///     (so every remainder the scalar tail handles), against the replaced scalar code: every candidate's log density
    ///     has the same bits.
    /// </summary>
    /// <param name="seed">Seeds the random mixtures.</param>
    [TestMethod]
    [DataRow(1)]
    [DataRow(2)]
    [DataRow(3)]
    public void ContinuousLogPdf_EveryKernelCount_MatchesTheReplacedScalarCodeToTheBit(int seed)
    {
        var rng = new Random(seed);
        int maxKernels = 3 * System.Numerics.Vector<double>.Count + 5;
        foreach (int nCont in new[] { 1, 2, 3, 7, 40 })
        {
            for (int nKernels = 1; nKernels <= maxKernels; nKernels++)
                AssertSameBits(RandomMixture(rng, nKernels, nCont, logEvery: 3), RandomCandidates(rng, 24, nCont, 0.15));
        }
    }

    /// <summary>
    ///     A study-sized mixture (1,000 kernels, 3 and 40 dimensions): the bounds grid has thousands of central
    ///     elements, so <see cref="TruncNorm.LogGaussMass(double[], double[])" /> takes Optuna's vectorized <c>erf</c>,
    ///     reached now through rented buffers cut to the logical length. Same bits.
    /// </summary>
    [TestMethod]
    public void ContinuousLogPdf_LargeGrids_UseTheVectorizedErf_AndMatchToTheBit()
    {
        var rng = new Random(11);
        AssertSameBits(RandomMixture(rng, 1000, 3, logEvery: 2), RandomCandidates(rng, 24, 3, 0.1));
        AssertSameBits(RandomMixture(rng, 1000, 40, logEvery: 5), RandomCandidates(rng, 24, 40, 0.1));
    }

    /// <summary>
    ///     A dimension whose bounds coincide (a single-value distribution) gives every kernel an empty interval, so its
    ///     term is NaN, as <c>np.select</c> makes it; the new kernel yields the same NaN densities.
    /// </summary>
    [TestMethod]
    public void ContinuousLogPdf_AnEmptyInterval_IsNaN_AsBefore()
    {
        var rng = new Random(5);
        int nKernels = 2 * System.Numerics.Vector<double>.Count + 1;
        var dists = new List<BatchedNumericalDistributions>
        {
            RandomDimension(rng, nKernels, log: false),
            new BatchedTruncNormDistributions(Fill(nKernels, 0.5), Fill(nKernels, 0.25), 0.5, 0.5),
        };
        var mixture = new MixtureOfProductDistribution(RandomWeights(rng, nKernels), dists);
        double[,] x = RandomCandidates(rng, 6, 2, 0.0);
        for (int s = 0; s < 6; s++)
            x[s, 1] = 0.5;

        AssertSameBits(mixture, x);
        mixture.LogPdf(x).Should().OnlyContain(v => double.IsNaN(v));
    }

    /// <summary>Compares the new <see cref="MixtureOfProductDistribution.LogPdf" /> with the replaced code, bit for bit.</summary>
    /// <param name="mixture">A mixture of continuous dimensions only.</param>
    /// <param name="x">The candidates, one row each.</param>
    private static void AssertSameBits(MixtureOfProductDistribution mixture, double[,] x)
    {
        double[] actual = mixture.LogPdf(x);
        double[] expected = ReferenceLogPdf(mixture, x);
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
    ///     The log-pdf as it was computed before item S6, for mixtures of continuous dimensions: per candidate, per kernel,
    ///     per dimension, with <c>log(sigma)</c> taken in the innermost loop, the bounds grid kernel-major, then the log
    ///     weights per (candidate, kernel) and the log-sum-exp.
    /// </summary>
    /// <param name="mixture">A mixture of continuous dimensions only.</param>
    /// <param name="x">The candidates, one row each.</param>
    /// <returns>One log density per candidate.</returns>
    private static double[] ReferenceLogPdf(MixtureOfProductDistribution mixture, double[,] x)
    {
        int nSamples = x.GetLength(0), nKernels = mixture.Weights.Length, nCont = mixture.Distributions.Count;
        var weighted = new double[nSamples, nKernels];
        var dists = mixture.Distributions.Cast<BatchedNumericalDistributions>().ToList();

        var xCont = new double[nCont][];
        for (int c = 0; c < nCont; c++)
        {
            xCont[c] = new double[nSamples];
            for (int s = 0; s < nSamples; s++)
                xCont[c][s] = dists[c].IsLog ? Math.Log(x[s, c]) : x[s, c];
        }

        var aGrid = new double[nKernels * nCont];
        var bGrid = new double[nKernels * nCont];
        for (int k = 0; k < nKernels; k++)
        {
            for (int c = 0; c < nCont; c++)
            {
                aGrid[k * nCont + c] = (dists[c].AdaptedLow - dists[c].Mu[k]) / dists[c].Sigma[k];
                bGrid[k * nCont + c] = (dists[c].AdaptedHigh - dists[c].Mu[k]) / dists[c].Sigma[k];
            }
        }
        double[] logGaussMass = ReferenceLogGaussMass(aGrid, bGrid);

        for (int s = 0; s < nSamples; s++)
        {
            for (int k = 0; k < nKernels; k++)
            {
                double sum = 0.0;
                for (int c = 0; c < nCont; c++)
                {
                    double mu = dists[c].Mu[k], sigma = dists[c].Sigma[k];
                    int gi = k * nCont + c;
                    double xstd = (xCont[c][s] - mu) / sigma;
                    double val = TruncNorm.NormLogPdf(xstd) - logGaussMass[gi] - Math.Log(sigma);
                    if (aGrid[gi] == bGrid[gi])
                        val = double.NaN;
                    else if (xstd < aGrid[gi] || xstd > bGrid[gi])
                        val = double.NegativeInfinity;
                    sum += val;
                }
                weighted[s, k] += sum;
            }
        }

        for (int s = 0; s < nSamples; s++)
        {
            for (int k = 0; k < nKernels; k++)
                weighted[s, k] += Math.Log(mixture.Weights[k]);
        }

        var result = new double[nSamples];
        var expBuf = new double[nKernels];
        for (int s = 0; s < nSamples; s++)
        {
            double max = double.NegativeInfinity;
            for (int k = 0; k < nKernels; k++)
                max = Math.Max(max, weighted[s, k]);
            if (double.IsNegativeInfinity(max))
                max = 0.0;
            for (int k = 0; k < nKernels; k++)
                expBuf[k] = Math.Exp(weighted[s, k] - max);
            result[s] = Math.Log(MathUtil.PairwiseSum(expBuf)) + max;
        }
        return result;
    }

    // ------------------------------------------------------------------ the categorical log-pdf (plan 14)

    /// <summary>
    ///     Plan 14: a categorical dimension now takes the log of each (kernel, choice) weight once instead of once per
    ///     candidate and kernel. Mixtures of one to four categorical dimensions (two to seven choices, some weights zero, so
    ///     some terms are <c>-inf</c>), from one to forty kernels, against the per-candidate logs it replaced: every
    ///     candidate's log density has the same bits.
    /// </summary>
    /// <param name="seed">Seeds the random mixtures.</param>
    [TestMethod]
    [DataRow(1)]
    [DataRow(2)]
    public void CategoricalLogPdf_OneLogPerKernelAndChoice_MatchesThePerCandidateLogsToTheBit(int seed)
    {
        var rng = new Random(seed);
        foreach (int nDims in new[] { 1, 2, 4 })
        {
            for (int nKernels = 1; nKernels <= 40; nKernels += 3)
            {
                var dims = new List<BatchedDistributions>();
                var choices = new int[nDims];
                for (int d = 0; d < nDims; d++)
                {
                    choices[d] = rng.Next(2, 8);
                    var weights = new double[nKernels, choices[d]];
                    for (int k = 0; k < nKernels; k++)
                        for (int c = 0; c < choices[d]; c++)
                            weights[k, c] = rng.Next(0, 6) == 0 ? 0.0 : rng.NextDouble();
                    dims.Add(new BatchedCategoricalDistributions(weights));
                }
                var mixture = new MixtureOfProductDistribution(RandomWeights(rng, nKernels), dims);
                var x = new double[24, nDims];
                for (int s = 0; s < 24; s++)
                    for (int d = 0; d < nDims; d++)
                        x[s, d] = rng.Next(0, choices[d]);

                double[] actual = mixture.LogPdf(x);
                double[] expected = ReferenceCategoricalLogPdf(mixture, x);
                for (int s = 0; s < 24; s++)
                {
                    if (double.IsNaN(expected[s]))
                        double.IsNaN(actual[s]).Should().BeTrue($"candidate {s} was NaN before");
                    else
                        BitConverter.DoubleToInt64Bits(actual[s]).Should().Be(BitConverter.DoubleToInt64Bits(expected[s]),
                            $"candidate {s}: {nKernels} kernels, {nDims} categorical dims");
                }
            }
        }
    }

    /// <summary>
    ///     The log-pdf of a mixture of categorical dimensions as computed before plan 14: <c>log(weights[k, x])</c> taken
    ///     per candidate and kernel, then the log mixture weights and the log-sum-exp.
    /// </summary>
    /// <param name="mixture">A mixture of categorical dimensions only.</param>
    /// <param name="x">The candidates' choice indices, one row each.</param>
    /// <returns>One log density per candidate.</returns>
    private static double[] ReferenceCategoricalLogPdf(MixtureOfProductDistribution mixture, double[,] x)
    {
        int nSamples = x.GetLength(0), nKernels = mixture.Weights.Length;
        var weighted = new double[nSamples, nKernels];
        for (int d = 0; d < mixture.Distributions.Count; d++)
        {
            var cat = (BatchedCategoricalDistributions)mixture.Distributions[d];
            for (int s = 0; s < nSamples; s++)
            {
                long xi = (long)x[s, d];
                for (int k = 0; k < nKernels; k++)
                    weighted[s, k] += Math.Log(cat.Weights[k, xi]);
            }
        }
        for (int s = 0; s < nSamples; s++)
            for (int k = 0; k < nKernels; k++)
                weighted[s, k] += Math.Log(mixture.Weights[k]);

        var result = new double[nSamples];
        var expBuf = new double[nKernels];
        for (int s = 0; s < nSamples; s++)
        {
            double max = double.NegativeInfinity;
            for (int k = 0; k < nKernels; k++)
                max = Math.Max(max, weighted[s, k]);
            if (double.IsNegativeInfinity(max))
                max = 0.0;
            for (int k = 0; k < nKernels; k++)
                expBuf[k] = Math.Exp(weighted[s, k] - max);
            result[s] = Math.Log(MathUtil.PairwiseSum(expBuf)) + max;
        }
        return result;
    }

    // ------------------------------------------------------------------ the log Gaussian mass

    /// <summary>
    ///     <see cref="TruncNorm.LogGaussMass(double[], double[])" />, now over rented scratch memory, against the replaced
    ///     allocation-per-call code: the left, right and central cases, NaN bounds, and central counts on both sides of the
    ///     2,000-element switch to the vectorized <c>erf</c>.
    /// </summary>
    /// <param name="n">The number of intervals.</param>
    [TestMethod]
    [DataRow(1)]
    [DataRow(17)]
    [DataRow(1999)]
    [DataRow(2000)]
    [DataRow(2001)]
    [DataRow(6000)]
    public void LogGaussMass_OverRentedScratch_MatchesTheReplacedCodeToTheBit(int n)
    {
        var rng = new Random(n);
        var a = new double[n];
        var b = new double[n];
        for (int i = 0; i < n; i++)
        {
            // Mostly central intervals, so the central count crosses the switch with n; some left, some right.
            double u = rng.NextDouble();
            double lo = u < 0.8 ? -rng.NextDouble() * 4 : (u < 0.9 ? -8 - rng.NextDouble() * 20 : rng.NextDouble() * 6);
            a[i] = lo;
            b[i] = lo + rng.NextDouble() * 5;
        }
        if (n > 3)
        {
            a[1] = double.NaN;
            b[2] = double.NaN;
        }

        double[] actual = TruncNorm.LogGaussMass(a, b);
        double[] expected = ReferenceLogGaussMass(a, b);
        for (int i = 0; i < n; i++)
        {
            if (double.IsNaN(expected[i]))
                double.IsNaN(actual[i]).Should().BeTrue($"interval {i} was NaN before");
            else
                BitConverter.DoubleToInt64Bits(actual[i]).Should().Be(BitConverter.DoubleToInt64Bits(expected[i]), $"interval {i}");
        }
    }

    /// <summary>
    ///     The span overload of <see cref="Erf.ComputeArray(double[])" /> picks its path from the span's length, not from
    ///     the buffer behind it: 1,999 values inside a 5,000-element buffer take the exact path, as a 1,999-element array
    ///     does. A caller that passed a rented buffer whole would switch paths and change values; this pins the contract.
    /// </summary>
    [TestMethod]
    public void ErfComputeArray_OverASpan_ChoosesThePathByTheSpansLength()
    {
        var rng = new Random(7);
        var buffer = new double[5000];
        for (int i = 0; i < buffer.Length; i++)
            buffer[i] = (rng.NextDouble() - 0.5) * 8;

        var exact = new double[1999];
        Array.Copy(buffer, exact, 1999);
        double[] fromArray = Erf.ComputeArray(exact);
        var fromSpan = new double[1999];
        Erf.ComputeArray(new ReadOnlySpan<double>(buffer, 0, 1999), fromSpan);

        for (int i = 0; i < 1999; i++)
        {
            BitConverter.DoubleToInt64Bits(fromSpan[i]).Should().Be(BitConverter.DoubleToInt64Bits(fromArray[i]));
            BitConverter.DoubleToInt64Bits(fromSpan[i]).Should().Be(BitConverter.DoubleToInt64Bits(Erf.Compute(buffer[i])),
                "below 2,000 elements every value is the exact erf");
        }
    }

    /// <summary>
    ///     The log Gaussian mass as it was computed before item S6: the central intervals gathered into new arrays and
    ///     passed through the normal CDF, the tails through the scalar log CDF.
    /// </summary>
    /// <param name="a">The standardized lower bounds.</param>
    /// <param name="b">The standardized upper bounds.</param>
    /// <returns>One log mass per interval.</returns>
    private static double[] ReferenceLogGaussMass(double[] a, double[] b)
    {
        int n = a.Length;
        var result = new double[n];
        var central = new List<int>();
        for (int i = 0; i < n; i++)
        {
            if (!(a[i] > 0.0) && !(b[i] <= 0.0))
                central.Add(i);
        }
        if (central.Count > 0)
        {
            double[] ndtrA = ReferenceNdtr(central.Select(i => a[i]).ToArray());
            double[] ndtrNegB = ReferenceNdtr(central.Select(i => -b[i]).ToArray());
            for (int k = 0; k < central.Count; k++)
                result[central[k]] = MathUtil.Log1p(-ndtrA[k] - ndtrNegB[k]);
        }
        for (int i = 0; i < n; i++)
        {
            if (a[i] > 0.0)
                result[i] = LogDiff(LogNdtr(-a[i]), LogNdtr(-b[i]));
            else if (b[i] <= 0.0)
                result[i] = LogDiff(LogNdtr(b[i]), LogNdtr(a[i]));
        }
        return result;
    }

    /// <summary>The normal CDF as before: <c>0.5 + 0.5 * erf(a / sqrt(2))</c> over a new array.</summary>
    /// <param name="a">The arguments.</param>
    /// <returns>One probability per argument.</returns>
    private static double[] ReferenceNdtr(double[] a)
    {
        double sqrt2 = Math.Sqrt(2.0);
        double[] erf = Erf.ComputeArray(a.Select(v => v / sqrt2).ToArray());
        return erf.Select(e => 0.5 + 0.5 * e).ToArray();
    }

    /// <summary>The scalar log normal CDF Optuna's <c>_log_ndtr_single</c> computes.</summary>
    /// <param name="v">The argument.</param>
    /// <returns>Its log CDF.</returns>
    private static double LogNdtr(double v) => TruncNorm.LogNdtr(new[] { v })[0];

    /// <summary>Optuna's <c>_log_diff(log_p, log_q) = log_p + log1p(-exp(log_q - log_p))</c>.</summary>
    /// <param name="logP">The larger log term.</param>
    /// <param name="logQ">The log term to subtract.</param>
    /// <returns>The log of the difference.</returns>
    private static double LogDiff(double logP, double logQ) => logP + MathUtil.Log1p(-Math.Exp(logQ - logP));

    // ------------------------------------------------------------------ the stable argsort

    /// <summary>
    ///     <see cref="ParzenEstimator.StableArgsort" /> is exactly the permutation the replaced LINQ sort returned, over
    ///     arrays full of ties, NaNs, both zeros and both infinities, and the empty and one-element arrays.
    /// </summary>
    [TestMethod]
    public void StableArgsort_IsTheStableOrderByPermutation_WithTiesNaNsAndSignedZeros()
    {
        var rng = new Random(13);
        double[] specials = { double.NaN, 0.0, -0.0, double.PositiveInfinity, double.NegativeInfinity, 1.5, -1.5 };
        for (int round = 0; round < 300; round++)
        {
            int n = round % 50;
            var values = new double[n];
            for (int i = 0; i < n; i++)
            {
                double u = rng.NextDouble();
                values[i] = u < 0.3 ? specials[rng.Next(specials.Length)] : Math.Round(rng.NextDouble() * 6) - 3;
            }
            double[] before = (double[])values.Clone();

            int[] expected = Enumerable.Range(0, n).OrderBy(i => values[i]).ToArray();
            ParzenEstimator.StableArgsort(values).Should().Equal(expected, $"round {round}: [{string.Join(", ", values)}]");
            values.Select(BitConverter.DoubleToInt64Bits).Should().Equal(before.Select(BitConverter.DoubleToInt64Bits),
                "the input is not reordered");
        }
    }

    // ------------------------------------------------------------------ inputs

    /// <summary>A mixture of <paramref name="nCont" /> continuous dimensions with random kernels and weights.</summary>
    /// <param name="rng">The generator.</param>
    /// <param name="nKernels">Kernels per dimension, prior included.</param>
    /// <param name="nCont">Continuous dimensions.</param>
    /// <param name="logEvery">Every this-many-th dimension is a log dimension.</param>
    /// <returns>The mixture.</returns>
    private static MixtureOfProductDistribution RandomMixture(Random rng, int nKernels, int nCont, int logEvery)
    {
        var dists = new List<BatchedDistributions>();
        for (int c = 0; c < nCont; c++)
            dists.Add(RandomDimension(rng, nKernels, log: c % logEvery == logEvery - 1));
        return new MixtureOfProductDistribution(RandomWeights(rng, nKernels), dists);
    }

    /// <summary>One continuous dimension over [0.5, 3.5] (plain, or log so its kernels live in log space).</summary>
    /// <param name="rng">The generator.</param>
    /// <param name="nKernels">Kernels, prior included.</param>
    /// <param name="log">Whether the dimension is logarithmic.</param>
    /// <returns>The dimension's kernels.</returns>
    private static BatchedNumericalDistributions RandomDimension(Random rng, int nKernels, bool log)
    {
        double low = 0.5, high = 3.5;
        double lo = log ? Math.Log(low) : low, hi = log ? Math.Log(high) : high;
        var mu = new double[nKernels];
        var sigma = new double[nKernels];
        for (int k = 0; k < nKernels; k++)
        {
            mu[k] = lo + rng.NextDouble() * (hi - lo);
            sigma[k] = (hi - lo) * (0.01 + rng.NextDouble());
        }
        return log
            ? new BatchedTruncLogNormDistributions(mu, sigma, low, high)
            : new BatchedTruncNormDistributions(mu, sigma, low, high);
    }

    /// <summary>Positive weights summing to one.</summary>
    /// <param name="rng">The generator.</param>
    /// <param name="n">How many.</param>
    /// <returns>The weights.</returns>
    private static double[] RandomWeights(Random rng, int n)
    {
        var w = Enumerable.Range(0, n).Select(_ => 0.05 + rng.NextDouble()).ToArray();
        double sum = w.Sum();
        return w.Select(v => v / sum).ToArray();
    }

    /// <summary>Candidates in [0.5, 3.5], with a share outside it (their density is zero, -inf in log).</summary>
    /// <param name="rng">The generator.</param>
    /// <param name="nSamples">Candidates.</param>
    /// <param name="nCont">Dimensions.</param>
    /// <param name="outsideShare">The share of coordinates placed outside the bounds.</param>
    /// <returns>The candidates, one row each.</returns>
    private static double[,] RandomCandidates(Random rng, int nSamples, int nCont, double outsideShare)
    {
        var x = new double[nSamples, nCont];
        for (int s = 0; s < nSamples; s++)
        {
            for (int c = 0; c < nCont; c++)
            {
                double u = rng.NextDouble();
                x[s, c] = u < outsideShare ? (rng.Next(2) == 0 ? 0.4 : 3.6) : 0.5 + rng.NextDouble() * 3.0;
            }
        }
        return x;
    }

    /// <summary>An array of <paramref name="n" /> copies of <paramref name="value" />.</summary>
    /// <param name="n">The length.</param>
    /// <param name="value">The value.</param>
    /// <returns>The array.</returns>
    private static double[] Fill(int n, double value) => Enumerable.Repeat(value, n).ToArray();
}
