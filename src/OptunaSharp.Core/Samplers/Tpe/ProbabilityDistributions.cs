using System;
using System.Collections.Generic;
using System.Linq;
using NumSharp;
using OptunaSharp.Internal;

namespace OptunaSharp.Samplers.Tpe;

/// <summary>
///     Port of <c>optuna/samplers/_tpe/probability_distributions.py</c>: the batched distribution
///     kinds and the <c>_MixtureOfProductDistribution</c> that back both the TPE sampler (Phase 2.8)
///     and <c>PedAnovaImportanceEvaluator</c> (Phase 2.7).
///
///     <para><b>[Misaligned] erf/ucrt tolerance.</b> The numerical kinds route their density through
///     <see cref="TruncNorm" /> → <see cref="Erf" />, whose scalar path is FreeBSD <c>s_erf</c> — ~1-3
///     ULP from Python's platform <c>math.erf</c> (ucrt on Windows). So sampled numerical values and
///     their log-densities agree with Optuna to a few ULP, not bit-for-bit (the same class of
///     divergence documented on <see cref="TruncNorm" />). RNG draw counts and order are identical, so
///     the study trajectory stays in lock-step regardless.</para>
/// </summary>
internal abstract class BatchedDistributions
{
}

/// <summary>Python <c>_BatchedCategoricalDistributions</c>.</summary>
internal sealed class BatchedCategoricalDistributions : BatchedDistributions
{
    /// <summary>Per-kernel categorical weights, shape <c>(n_kernels, n_choices)</c>.</summary>
    internal double[,] Weights { get; }

    /// <summary>Retains the per-kernel probabilities for each categorical choice.</summary>
    /// <param name="weights">Per-kernel categorical probabilities, with kernels as rows and choices as columns.</param>
    internal BatchedCategoricalDistributions(double[,] weights) => Weights = weights;
}

/// <summary>
///     Common base of the four numerical batched kinds (truncated normal / log-normal, continuous /
///     discrete). Mirrors the shared <c>mu</c>/<c>sigma</c>/<c>low</c>/<c>high</c> fields and the
///     <c>adapted_low</c>/<c>adapted_high</c>/<c>is_log</c>/<c>step</c> properties in Optuna.
/// </summary>
internal abstract class BatchedNumericalDistributions : BatchedDistributions
{
    /// <summary>Kernel centers in transformed numerical coordinates.</summary>
    internal double[] Mu { get; }
    /// <summary>Per-kernel standard deviations in transformed numerical coordinates.</summary>
    internal double[] Sigma { get; }
    /// <summary>Lower endpoint of the represented numerical domain.</summary>
    internal double Low { get; }
    /// <summary>Upper endpoint of the represented numerical domain.</summary>
    internal double High { get; }

    /// <summary>Retains numerical kernel centers, bandwidths, and external truncation bounds.</summary>
    /// <param name="mu">Kernel centers in transformed numerical coordinates.</param>
    /// <param name="sigma">Kernel standard deviations in transformed coordinates.</param>
    /// <param name="low">Lower endpoint of the numerical domain.</param>
    /// <param name="high">Upper endpoint of the numerical domain.</param>
    protected BatchedNumericalDistributions(double[] mu, double[] sigma, double low, double high)
    {
        Mu = mu;
        Sigma = sigma;
        Low = low;
        High = high;
    }

    /// <summary>Lower truncation bound in kernel coordinates, expanded by half a bin for discrete domains.</summary>
    internal abstract double AdaptedLow { get; }
    /// <summary>Upper truncation bound in kernel coordinates, expanded by half a bin for discrete domains.</summary>
    internal abstract double AdaptedHigh { get; }
    /// <summary>Whether numerical kernels are represented in logarithmic coordinates.</summary>
    internal abstract bool IsLog { get; }

    /// <summary>Discretization step; <c>0.0</c> for the continuous kinds (Python's <c>step</c> property).</summary>
    internal abstract double Step { get; }
}

/// <summary>Python <c>_BatchedTruncNormDistributions</c>.</summary>
internal sealed class BatchedTruncNormDistributions : BatchedNumericalDistributions
{
    /// <summary>Creates continuous Gaussian kernels truncated directly in parameter space.</summary>
    /// <param name="mu">Kernel centers in transformed numerical coordinates.</param>
    /// <param name="sigma">Kernel standard deviations in transformed coordinates.</param>
    /// <param name="low">Lower endpoint of the numerical domain.</param>
    /// <param name="high">Upper endpoint of the numerical domain.</param>
    internal BatchedTruncNormDistributions(double[] mu, double[] sigma, double low, double high)
        : base(mu, sigma, low, high) { }

    /// <inheritdoc />
    internal override double AdaptedLow => Low;
    /// <inheritdoc />
    internal override double AdaptedHigh => High;
    /// <inheritdoc />
    internal override bool IsLog => false;
    /// <inheritdoc />
    internal override double Step => 0.0;
}

/// <summary>Python <c>_BatchedTruncLogNormDistributions</c>.</summary>
internal sealed class BatchedTruncLogNormDistributions : BatchedNumericalDistributions
{
    /// <summary>Creates continuous Gaussian kernels truncated in logarithmic parameter space.</summary>
    /// <param name="mu">Kernel centers in transformed numerical coordinates.</param>
    /// <param name="sigma">Kernel standard deviations in transformed coordinates.</param>
    /// <param name="low">Lower endpoint of the numerical domain.</param>
    /// <param name="high">Upper endpoint of the numerical domain.</param>
    internal BatchedTruncLogNormDistributions(double[] mu, double[] sigma, double low, double high)
        : base(mu, sigma, low, high) { }

    /// <inheritdoc />
    internal override double AdaptedLow => Math.Log(Low);
    /// <inheritdoc />
    internal override double AdaptedHigh => Math.Log(High);
    /// <inheritdoc />
    internal override bool IsLog => true;
    /// <inheritdoc />
    internal override double Step => 0.0;
}

/// <summary>Python <c>_BatchedDiscreteTruncNormDistributions</c>.</summary>
internal sealed class BatchedDiscreteTruncNormDistributions : BatchedNumericalDistributions
{
    private readonly double _step;

    /// <summary>Creates Gaussian kernels whose bins represent quantized numerical choices.</summary>
    /// <param name="mu">Kernel centers in transformed numerical coordinates.</param>
    /// <param name="sigma">Kernel standard deviations in transformed coordinates.</param>
    /// <param name="low">Lower endpoint of the numerical domain.</param>
    /// <param name="high">Upper endpoint of the numerical domain.</param>
    /// <param name="step">Positive spacing between adjacent quantized numerical values.</param>
    internal BatchedDiscreteTruncNormDistributions(double[] mu, double[] sigma, double low, double high, double step)
        : base(mu, sigma, low, high) => _step = step;

    /// <inheritdoc />
    internal override double AdaptedLow => Low - _step / 2.0;
    /// <inheritdoc />
    internal override double AdaptedHigh => High + _step / 2.0;
    /// <inheritdoc />
    internal override bool IsLog => false;
    /// <inheritdoc />
    internal override double Step => _step;
}

/// <summary>Python <c>_BatchedDiscreteTruncLogNormDistributions</c>.</summary>
internal sealed class BatchedDiscreteTruncLogNormDistributions : BatchedNumericalDistributions
{
    private readonly double _step;

    /// <summary>Creates logarithmic Gaussian kernels whose bins represent quantized numerical choices.</summary>
    /// <param name="mu">Kernel centers in transformed numerical coordinates.</param>
    /// <param name="sigma">Kernel standard deviations in transformed coordinates.</param>
    /// <param name="low">Lower endpoint of the numerical domain.</param>
    /// <param name="high">Upper endpoint of the numerical domain.</param>
    /// <param name="step">Positive spacing between adjacent quantized numerical values.</param>
    internal BatchedDiscreteTruncLogNormDistributions(double[] mu, double[] sigma, double low, double high, double step)
        : base(mu, sigma, low, high) => _step = step;

    /// <inheritdoc />
    internal override double AdaptedLow => Math.Log(Low - _step / 2.0);
    /// <inheritdoc />
    internal override double AdaptedHigh => Math.Log(High + _step / 2.0);
    /// <inheritdoc />
    internal override bool IsLog => true;
    /// <inheritdoc />
    internal override double Step => _step;
}

/// <summary>Python <c>_MixtureOfProductDistribution</c>.</summary>
internal sealed class MixtureOfProductDistribution
{
    /// <summary>
    ///     A bound below which <c>exp</c> is exactly <c>+0.0</c>: <c>exp(-746)</c> is about 0.42 of half the smallest
    ///     subnormal (2^-1075, about 2.5e-324), so a correctly rounded result is zero, and NumPy's and .NET's <c>exp</c>
    ///     return zero for every input below the underflow point (about -745.1332; glibc, ucrt and Apple's libm alike).
    ///     <see cref="LogPdf" /> stores <c>0.0</c> for such terms instead of calling <see cref="Math.Exp" />, whose underflow
    ///     path costs about three normal calls; <c>TpeExpUnderflowTests</c> pins that <see cref="Math.Exp" /> returns
    ///     <c>+0.0</c> below it on the platform that runs the tests.
    /// </summary>
    internal const double ExpIsZeroBelow = -746.0;

    /// <summary>Nonnegative mixture-component probabilities, in shared kernel order.</summary>
    internal double[] Weights { get; }
    /// <summary>Coordinate distributions sharing the same mixture-component index.</summary>
    internal IReadOnlyList<BatchedDistributions> Distributions { get; }

    /// <summary>Retains mixture weights and the coordinate distributions sharing each kernel index.</summary>
    /// <param name="weights">Nonnegative weights aligned with mixture components.</param>
    /// <param name="distributions">Coordinate distributions sharing the same mixture component ordering.</param>
    internal MixtureOfProductDistribution(double[] weights, IReadOnlyList<BatchedDistributions> distributions)
    {
        Weights = weights;
        Distributions = distributions;
    }

    /// <summary>
    ///     Python <c>_MixtureOfProductDistribution.sample(rng, batch_size)</c> — returns a
    ///     <c>(batch_size, n_dim)</c> array of samples in the original internal repr (log undone,
    ///     discrete grids snapped). The RNG draws are, in order: one <c>choice</c> over the mixture
    ///     weights (<c>batch_size</c> uniforms), one <c>rand(batch_size)</c> per categorical dim in
    ///     dim order, then a single <c>uniform(0, 1, (n_numerical, batch_size))</c> for all numerical
    ///     dims — matching Optuna byte-for-byte so the whole study stays RNG-aligned.
    /// </summary>
    /// <param name="rng">Random stream whose draw order is shared with the reference algorithm.</param>
    /// <param name="batchSize">Number of candidate vectors to draw from the mixture.</param>
    /// <returns>A sample-by-parameter matrix in internal distribution representation, with logarithms undone and discrete bins snapped.</returns>
    internal double[,] Sample(LazyRandomState rng, int batchSize)
    {
        int nDim = Distributions.Count;
        var activeIndices = Choice(rng, Weights, batchSize);
        var ret = new double[batchSize, nDim];

        var numericalDists = new List<BatchedNumericalDistributions>();
        var numericalInds = new List<int>();
        var discInds = new List<int>();
        var logInds = new List<int>();

        for (int i = 0; i < nDim; i++)
        {
            if (Distributions[i] is BatchedCategoricalDistributions cat)
            {
                // active_weights = weights[active_indices, :]; rnd_quantile = rng.rand(batch_size).
                var rndQuantile = NumPyDraws.Rand(rng.Rng, batchSize);
                int nChoices = cat.Weights.GetLength(1);
                for (int s = 0; s < batchSize; s++)
                {
                    int ai = activeIndices[s];
                    // cum_probs = cumsum(active_weights, axis=-1); cum_probs[:, -1] = 1;
                    // ret = sum(cum_probs < rnd_quantile[:, None], axis=-1).
                    double acc = 0.0;
                    int count = 0;
                    for (int c = 0; c < nChoices; c++)
                    {
                        acc += cat.Weights[ai, c];
                        double cp = c == nChoices - 1 ? 1.0 : acc;  // Avoid numerical errors.
                        if (cp < rndQuantile[s])
                            count++;
                    }
                    ret[s, i] = count;
                }
            }
            else
            {
                var num = (BatchedNumericalDistributions)Distributions[i];
                numericalDists.Add(num);
                numericalInds.Add(i);
                if (num.Step != 0.0)
                    discInds.Add(i);
                if (num.IsLog)
                    logInds.Add(i);
            }
        }

        if (numericalDists.Count > 0)
        {
            int nNum = numericalDists.Count;
            int total = nNum * batchSize;
            // _truncnorm.rvs draws random_state.uniform(0, 1, size=(n_num, batch)) in C order.
            var quantiles = NumPyDraws.RandomSample(rng.Rng, total);

            var aFlat = new double[total];
            var bFlat = new double[total];
            var loc = new double[total];
            var scale = new double[total];
            for (int d = 0; d < nNum; d++)
            {
                var nd = numericalDists[d];
                double adaptedLow = nd.AdaptedLow;
                double adaptedHigh = nd.AdaptedHigh;
                for (int s = 0; s < batchSize; s++)
                {
                    int idx = d * batchSize + s;
                    double mu = nd.Mu[activeIndices[s]];
                    double sigma = nd.Sigma[activeIndices[s]];
                    aFlat[idx] = (adaptedLow - mu) / sigma;
                    bFlat[idx] = (adaptedHigh - mu) / sigma;
                    loc[idx] = mu;
                    scale[idx] = sigma;
                }
            }

            var ppf = TruncNorm.Ppf(quantiles, aFlat, bFlat);
            for (int d = 0; d < nNum; d++)
            {
                int col = numericalInds[d];
                for (int s = 0; s < batchSize; s++)
                {
                    int idx = d * batchSize + s;
                    ret[s, col] = ppf[idx] * scale[idx] + loc[idx];
                }
            }

            // ret[:, log_inds] = np.exp(ret[:, log_inds]) — undo the log transform.
            foreach (int col in logInds)
            {
                for (int s = 0; s < batchSize; s++)
                    ret[s, col] = Math.Exp(ret[s, col]);
            }

            // ret[:, disc_inds] = clip(low + round((v - low) / step) * step, low, high).
            foreach (int col in discInds)
            {
                var nd = (BatchedNumericalDistributions)Distributions[col];
                double low = nd.Low, high = nd.High, step = nd.Step;
                for (int s = 0; s < batchSize; s++)
                {
                    double v = ret[s, col];
                    double rounded = low + Math.Round((v - low) / step, MidpointRounding.ToEven) * step;
                    ret[s, col] = MathUtil.Clip(rounded, low, high);
                }
            }
        }

        return ret;
    }

    /// <summary>
    ///     NumPy <c>rng.choice(len(weights), p=weights, size=size)</c>: cdf = cumsum(weights) (sequential),
    ///     cdf /= cdf[-1], then <c>cdf.searchsorted(random_sample(size), side='right')</c>. Replicated
    ///     directly because NumSharp's <c>choice</c> uses <c>side='left'</c> (see the BruteForce ledger).
    /// </summary>
    /// <param name="rng">Random stream whose draw order is shared with the reference algorithm.</param>
    /// <param name="weights">Nonnegative weights aligned with mixture components.</param>
    /// <param name="size">Number of candidate samples to draw.</param>
    /// <returns>Mixture-component indices sampled according to the supplied probabilities.</returns>
    private static int[] Choice(LazyRandomState rng, double[] weights, int size)
    {
        int n = weights.Length;
        var cdf = new double[n];
        double acc = 0.0;
        for (int i = 0; i < n; i++)
        {
            acc += weights[i];  // np.cumsum: sequential prefix sum.
            cdf[i] = acc;
        }
        double last = cdf[n - 1];
        for (int i = 0; i < n; i++)
            cdf[i] /= last;

        var u = NumPyDraws.RandomSample(rng.Rng, size);
        var result = new int[size];
        for (int j = 0; j < size; j++)
            result[j] = BisectRight(cdf, u[j]);
        return result;
    }

    /// <summary>Python <c>_MixtureOfProductDistribution.log_pdf(x)</c>, with <c>x</c> shape <c>(n_samples, dim)</c>.</summary>
    /// <param name="x">Input numerical coordinates or observation rows.</param>
    /// <returns>One mixture log probability density per sample row.</returns>
    /// <exception cref="System.InvalidOperationException">A coordinate uses an unknown batched-distribution implementation.</exception>
    internal double[] LogPdf(double[,] x)
    {
        int nSamples = x.GetLength(0);
        int dim = x.GetLength(1);
        int nKernels = Weights.Length;
        // weighted_log_pdf, (n_samples, n_kernels) row-major in a rented buffer (plan 13, item S6): at TPE's 24 candidates
        // and a thousand kernels it is 192 KB, a large-object-heap allocation per call, which kept background GCs busy
        // and made the sampler wait for them. Every element is cleared, then accumulated in the same order as before.
        double[] weightedLogPdf = System.Buffers.ArrayPool<double>.Shared.Rent(Math.Max(nSamples * nKernels, 1));
        try
        {
            Array.Clear(weightedLogPdf, 0, nSamples * nKernels);

            var contDists = new List<BatchedNumericalDistributions>();
            var contCols = new List<int>();

            for (int i = 0; i < dim; i++)
            {
                switch (Distributions[i])
                {
                    case BatchedCategoricalDistributions cat:
                    {
                        // xi = x[:, i, None, None].astype(int64); += log(take_along_axis(weights, xi))[..., 0].
                        for (int s = 0; s < nSamples; s++)
                        {
                            long xi = (long)x[s, i];
                            for (int k = 0; k < nKernels; k++)
                                weightedLogPdf[s * nKernels + k] += Math.Log(cat.Weights[k, xi]);
                        }
                        break;
                    }

                    case BatchedNumericalDistributions num when num.Step == 0.0:
                        // Continuous kind: collected and processed together after the loop.
                        contDists.Add(num);
                        contCols.Add(i);
                        break;

                    case BatchedNumericalDistributions dtn:
                    {
                        // Discrete kind (truncated normal or log-normal).
                        bool isLog = dtn.IsLog;
                        double step = dtn.Step;
                        var xcol = new double[nSamples];
                        for (int s = 0; s < nSamples; s++)
                            xcol[s] = x[s, i];

                        var (xiUniq, xiInv) = UniqueInverse1d(xcol);
                        var (muUniq, sigmaUniq, msInv) = UniqueInverse2d(dtn.Mu, dtn.Sigma);
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

                        var logMass = LogGaussMassUnique(a, b);  // shape (nXi, nMs), row-major.
                        for (int s = 0; s < nSamples; s++)
                        {
                            int row = xiInv[s] * nMs;
                            for (int k = 0; k < nKernels; k++)
                                weightedLogPdf[s * nKernels + k] += logMass[row + msInv[k]];
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
                                weightedLogPdf[s * nKernels + k] -= norm[msInv[k]];
                        }
                        break;
                    }

                    default:
                        throw new InvalidOperationException("Unknown batched distribution kind.");
                }
            }

            if (contDists.Count > 0)
                AddContinuousLogPdf(x, weightedLogPdf, contDists, contCols);

            // weighted_log_pdf += np.log(self.weights[np.newaxis]). Each kernel's log weight is the same for every sample,
            // so it is taken once per kernel (plan 13, item S6); Math.Log is deterministic, so the sums keep their bits.
            var logWeights = new double[nKernels];
            for (int k = 0; k < nKernels; k++)
                logWeights[k] = Math.Log(Weights[k]);
            for (int s = 0; s < nSamples; s++)
            {
                for (int k = 0; k < nKernels; k++)
                    weightedLogPdf[s * nKernels + k] += logWeights[k];
            }

            // np.log(np.exp(weighted_log_pdf - max_[:, None]).sum(axis=1)) + max_, with -inf max_ -> 0.
            var result = new double[nSamples];
            var expBuf = new double[nKernels];
            for (int s = 0; s < nSamples; s++)
            {
                double max = double.NegativeInfinity;
                for (int k = 0; k < nKernels; k++)
                    max = Math.Max(max, weightedLogPdf[s * nKernels + k]);
                if (double.IsNegativeInfinity(max))
                    max = 0.0;
                for (int k = 0; k < nKernels; k++)
                {
                    // exp(z) is exactly +0.0 for every z below the underflow point (about -745.13: the result is under half
                    // the smallest subnormal), so the call is skipped well below it (plan 14): kernels far from a candidate
                    // are most of a long history and each exp was a few nanoseconds of the hottest loop. The zero still
                    // goes into the buffer, so the pairwise sum sees the same elements; NaN fails the test and still
                    // reaches Math.Exp, as everything at or above the bound does.
                    double z = weightedLogPdf[s * nKernels + k] - max;
                    expBuf[k] = z < ExpIsZeroBelow ? 0.0 : Math.Exp(z);
                }
                result[s] = Math.Log(MathUtil.PairwiseSum(expBuf)) + max;
            }
            return result;
        }
        finally
        {
            System.Buffers.ArrayPool<double>.Shared.Return(weightedLogPdf);
        }
    }

    /// <summary>
    ///     The continuous branch of <c>log_pdf</c>: <c>_truncnorm.logpdf</c> over all continuous dims,
    ///     summed over dims and added to <paramref name="weightedLogPdf" />. The Gaussian-mass grid is
    ///     computed once over the <c>n_kernels × n_cont</c> standardized bounds, as NumPy does, so the
    ///     <see cref="Erf" /> array-size threshold is exercised identically.
    /// </summary>
    /// <param name="x">Input numerical coordinates or observation rows.</param>
    /// <param name="weightedLogPdf">
    ///     Log densities including each mixture component weight, (n_samples, n_kernels) row-major; at least
    ///     n_samples × n_kernels long (a rented buffer may be longer).
    /// </param>
    /// <param name="contDists">Continuous numerical coordinate distributions.</param>
    /// <param name="contCols">Original coordinate indices of continuous numerical distributions.</param>
    /// <remarks>
    ///     <para>
    ///         <b>Fused, same bits (Rustuna plan 13, item S6).</b> Everything that depends only on a (kernel, dim)
    ///         pair is computed once per call instead of once per candidate: the standardized bounds, their log
    ///         Gaussian mass and <c>log(sigma)</c>, which the per-candidate loop used to take 24 times per pair (more
    ///         than half of a 1,000-trial, 40-parameter TPE run). Each value is the same expression on the same
    ///         operands, and <c>Math.Log</c> is deterministic, so it has the same bits.
    ///     </para>
    ///     <para>
    ///         The grids are laid out by dimension (<c>[c * n_kernels + k]</c>) so the kernel loop is contiguous.
    ///         <see cref="TruncNorm.LogGaussMass(ReadOnlySpan{double}, ReadOnlySpan{double}, Span{double})" /> is elementwise, and the vectorized-<c>erf</c> switch depends only
    ///         on how many elements it gets (see <see cref="Erf.ComputeArray(ReadOnlySpan{double}, Span{double})" />), so evaluating the same elements in
    ///         another order changes no element.
    ///     </para>
    ///     <para>
    ///         <see cref="AccumulateDimension" /> adds one dimension's term to every kernel's sum, dimension after
    ///         dimension, so each kernel's sum is still <c>((0 + t0) + t1) + …</c> in dimension order; vector lanes
    ///         hold different kernels and never mix. .NET never contracts a multiply and an add into a fused
    ///         multiply-add on its own, so a lane computes exactly what the scalar loop computes.
    ///     </para>
    /// </remarks>
    private void AddContinuousLogPdf(
        double[,] x, double[] weightedLogPdf,
        List<BatchedNumericalDistributions> contDists, List<int> contCols)
    {
        int nSamples = x.GetLength(0);
        int nKernels = Weights.Length;
        int nCont = contDists.Count;
        int gridSize = nKernels * nCont;

        // Every buffer below is fully written before it is read, so none is zeroed first: zeroing the grids was a tenth
        // of a 1,000-trial, 40-parameter TPE run, and allocating them put four large-object-heap arrays per call on the
        // GC. The grids are rented, and every use goes through a span of the logical length: LogGaussMass and the
        // vectorized-erf switch depend on the length they are given, never on a rented buffer's size.
        // x_cont = log(x[:, col]) where is_log else x[:, col], one row per sample ([s * nCont + c]).
        var xCont = GC.AllocateUninitializedArray<double>(nSamples * nCont);
        for (int c = 0; c < nCont; c++)
        {
            bool isLog = contDists[c].IsLog;
            int col = contCols[c];
            for (int s = 0; s < nSamples; s++)
                xCont[s * nCont + c] = isLog ? Math.Log(x[s, col]) : x[s, col];
        }

        // Standardized bounds a, b, log(sigma) and the log Gaussian mass per (kernel, dim): one rented buffer cut into
        // four grids, each one row per dim over the kernels.
        double[] gridBuffer = System.Buffers.ArrayPool<double>.Shared.Rent(4 * gridSize);
        try
        {
            var aGrid = new Span<double>(gridBuffer, 0, gridSize);
            var bGrid = new Span<double>(gridBuffer, gridSize, gridSize);
            var logSigma = new Span<double>(gridBuffer, 2 * gridSize, gridSize);
            var logGaussMass = new Span<double>(gridBuffer, 3 * gridSize, gridSize);
            for (int c = 0; c < nCont; c++)
            {
                var nd = contDists[c];
                // AdaptedLow / AdaptedHigh recompute a logarithm per access on the log kinds: the same value every time.
                double adaptedLow = nd.AdaptedLow, adaptedHigh = nd.AdaptedHigh;
                double[] mu = nd.Mu, sigma = nd.Sigma;
                int row = c * nKernels;
                for (int k = 0; k < nKernels; k++)
                {
                    aGrid[row + k] = (adaptedLow - mu[k]) / sigma[k];
                    bGrid[row + k] = (adaptedHigh - mu[k]) / sigma[k];
                    logSigma[row + k] = Math.Log(sigma[k]);
                }
            }
            TruncNorm.LogGaussMass(aGrid, bGrid, logGaussMass);

            var sums = GC.AllocateUninitializedArray<double>(nKernels);
            for (int s = 0; s < nSamples; s++)
            {
                // Each kernel's sum starts at +0.0 for every candidate, as the scalar loop's `double sum = 0.0` did.
                Array.Clear(sums);
                for (int c = 0; c < nCont; c++)
                {
                    var nd = contDists[c];
                    int row = c * nKernels;
                    AccumulateDimension(
                        xCont[s * nCont + c], nd.Mu, nd.Sigma,
                        aGrid.Slice(row, nKernels), bGrid.Slice(row, nKernels),
                        logGaussMass.Slice(row, nKernels), logSigma.Slice(row, nKernels),
                        sums);
                }
                for (int k = 0; k < nKernels; k++)
                    weightedLogPdf[s * nKernels + k] += sums[k];
            }
        }
        finally
        {
            System.Buffers.ArrayPool<double>.Shared.Return(gridBuffer);
        }
    }

    /// <summary>
    ///     Adds one continuous dimension's truncated-normal log density at <paramref name="x" /> to every kernel's sum:
    ///     <c>(norm_logpdf((x - mu) / sigma) - log_gauss_mass) - log(sigma)</c>, NaN where the kernel's standardized
    ///     interval is empty (<c>a == b</c>) and <c>-inf</c> where <c>x</c> falls outside it, as
    ///     <c>np.select([a == b, (x &lt; a) | (x &gt; b)], [nan, -inf], default=out)</c>.
    /// </summary>
    /// <param name="x">The candidate's coordinate in this dimension, log-transformed already for the log kinds.</param>
    /// <param name="mu">The kernels' centers in this dimension.</param>
    /// <param name="sigma">The kernels' standard deviations in this dimension.</param>
    /// <param name="a">The kernels' standardized lower bounds.</param>
    /// <param name="b">The kernels' standardized upper bounds.</param>
    /// <param name="logGaussMass">The kernels' log Gaussian mass between <paramref name="a" /> and <paramref name="b" />.</param>
    /// <param name="logSigma">The kernels' <c>log(sigma)</c>.</param>
    /// <param name="sums">One running sum per kernel; this dimension's term is added to each.</param>
    /// <remarks>
    ///     The vector path computes every lane with the scalar path's operations in the scalar path's order: the same
    ///     division (never a multiplication by the reciprocal), the same subtraction order, the NaN and <c>-inf</c>
    ///     overrides applied last. The only freedom it takes is the sign of zero in <c>-(z * z)</c>, which the
    ///     subtraction of the normal's log constant right after removes. The tail that does not fill a vector takes
    ///     the scalar path.
    /// </remarks>
    private static void AccumulateDimension(
        double x, double[] mu, double[] sigma,
        ReadOnlySpan<double> a, ReadOnlySpan<double> b, ReadOnlySpan<double> logGaussMass, ReadOnlySpan<double> logSigma,
        double[] sums)
    {
        int n = sums.Length;
        int k = 0;
        if (System.Numerics.Vector.IsHardwareAccelerated && n >= System.Numerics.Vector<double>.Count)
        {
            int width = System.Numerics.Vector<double>.Count;
            var xv = new System.Numerics.Vector<double>(x);
            var two = new System.Numerics.Vector<double>(2.0);
            var logC = new System.Numerics.Vector<double>(TruncNorm.NormPdfLogC);
            var nan = new System.Numerics.Vector<double>(double.NaN);
            var negativeInfinity = new System.Numerics.Vector<double>(double.NegativeInfinity);
            for (; k <= n - width; k += width)
            {
                var muV = new System.Numerics.Vector<double>(mu, k);
                var sigmaV = new System.Numerics.Vector<double>(sigma, k);
                var aV = new System.Numerics.Vector<double>(a[k..]);
                var bV = new System.Numerics.Vector<double>(b[k..]);
                var xstd = (xv - muV) / sigmaV;
                var normLogPdf = -(xstd * xstd) / two - logC;
                var val = (normLogPdf - new System.Numerics.Vector<double>(logGaussMass[k..]))
                          - new System.Numerics.Vector<double>(logSigma[k..]);
                val = System.Numerics.Vector.ConditionalSelect(
                    System.Numerics.Vector.LessThan(xstd, aV) | System.Numerics.Vector.GreaterThan(xstd, bV),
                    negativeInfinity, val);
                val = System.Numerics.Vector.ConditionalSelect(System.Numerics.Vector.Equals(aV, bV), nan, val);
                (new System.Numerics.Vector<double>(sums, k) + val).CopyTo(sums, k);
            }
        }

        for (; k < n; k++)
        {
            double xstd = (x - mu[k]) / sigma[k];
            double val = TruncNorm.NormLogPdf(xstd) - logGaussMass[k] - logSigma[k];
            if (a[k] == b[k])
                val = double.NaN;
            else if (xstd < a[k] || xstd > b[k])
                val = double.NegativeInfinity;
            sums[k] += val;
        }
    }

    /// <summary>NumPy <c>searchsorted(cdf, v, side='right')</c> for an ascending array (bisect right).</summary>
    /// <param name="a">First numerical array operand.</param>
    /// <param name="v">Scalar operand used in the comparison or numerical transformation.</param>
    /// <returns>The insertion position after all entries equal to the target.</returns>
    private static int BisectRight(double[] a, double v)
    {
        int lo = 0, hi = a.Length;
        while (lo < hi)
        {
            int mid = (lo + hi) >> 1;
            if (v < a[mid])
                hi = mid;
            else
                lo = mid + 1;
        }
        return lo;
    }

    /// <summary>
    ///     Python <c>_log_gauss_mass_unique(a, b)</c>: dedup the raveled (a, b) pairs, evaluate
    ///     <c>_log_gauss_mass</c> once per unique pair, then scatter back. This is not merely an
    ///     optimization here — it fixes the array size passed to <c>erf</c>, which switches between
    ///     the exact and the vectorized approximation at 2000 elements (see <see cref="Erf" />).
    /// </summary>
    /// <param name="a">First numerical array operand.</param>
    /// <param name="b">Second numerical array operand.</param>
    /// <returns>Gaussian interval log masses expanded back to the original input order.</returns>
    private static double[] LogGaussMassUnique(double[] a, double[] b)
    {
        var (aUniq, bUniq, inv) = UniqueInverse2d(a, b);
        var lgmUniq = TruncNorm.LogGaussMass(aUniq, bUniq);
        var result = new double[a.Length];
        for (int j = 0; j < a.Length; j++)
            result[j] = lgmUniq[inv[j]];
        return result;
    }

    /// <summary>NumPy <c>np.unique(x, return_inverse=True)</c> for a 1-D array.</summary>
    /// <param name="x">Input numerical coordinates or observation rows.</param>
    /// <returns>Sorted unique values and inverse indices reconstructing the input.</returns>
    private static (double[] Uniq, int[] Inv) UniqueInverse1d(double[] x)
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

    /// <summary>
    ///     Python <c>_unique_inverse_2d(a, b)</c>: the unique (a, b) row pairs sorted
    ///     lexicographically, plus the inverse mapping original-index → unique-index.
    /// </summary>
    /// <param name="a">First numerical array operand.</param>
    /// <param name="b">Second numerical array operand.</param>
    /// <returns>Sorted unique paired values and inverse indices reconstructing the input pairs.</returns>
    private static (double[] AUniq, double[] BUniq, int[] Inv) UniqueInverse2d(double[] a, double[] b)
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
            // NumPy's `!=` treats -0.0 == 0.0 and NaN != NaN, matching C# `!=` on double.
            if (k == 0 || a[idx] != a[prev] || b[idx] != b[prev])
            {
                aUniq.Add(a[idx]);
                bUniq.Add(b[idx]);
            }
            inv[idx] = aUniq.Count - 1;
        }
        return (aUniq.ToArray(), bUniq.ToArray(), inv);
    }
}
