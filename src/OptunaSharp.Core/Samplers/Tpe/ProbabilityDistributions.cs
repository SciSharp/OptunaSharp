using System;
using System.Collections.Generic;
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
                        // Each candidate adds log(weights[k, xi]) for its choice xi. The log of every (kernel, choice)
                        // weight is taken once (plan 14; Rustuna 198e484 reduces the log calls too, with other numerics):
                        // one log per kernel and choice instead of one per candidate and kernel, 3,000 instead of 24,000
                        // for a three-choice parameter over a thousand kernels. Math.Log is deterministic, so each
                        // added term, and every sum, keeps its bits; only choices some candidate drew are computed.
                        int nChoices = cat.Weights.GetLength(1);
                        var logChoiceWeights = new double[nChoices][];
                        for (int s = 0; s < nSamples; s++)
                        {
                            long xi = (long)x[s, i];
                            double[]? column = logChoiceWeights[xi];
                            if (column is null)
                            {
                                column = new double[nKernels];
                                for (int k = 0; k < nKernels; k++)
                                    column[k] = Math.Log(cat.Weights[k, xi]);
                                logChoiceWeights[xi] = column;
                            }
                            for (int k = 0; k < nKernels; k++)
                                weightedLogPdf[s * nKernels + k] += column[k];
                        }
                        break;
                    }

                    case BatchedNumericalDistributions num when num.Step == 0.0:
                        // Continuous kind: collected and processed together after the loop.
                        contDists.Add(num);
                        contCols.Add(i);
                        break;

                    case BatchedNumericalDistributions dtn:
                        // Discrete kind (truncated normal or log-normal).
                        AddDiscreteLogPdf(x, i, dtn, weightedLogPdf);
                        break;

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
    ///         <see cref="AccumulateContinuous" /> then adds each candidate's sum over the dimensions to its row, every
    ///         sum still <c>((0 + t0) + t1) + …</c> in dimension order (see there for how the vector path keeps the bits).
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

        // Standardized bounds a, b, log(sigma), the log Gaussian mass, and the kernels' mu and sigma per (kernel, dim):
        // one rented buffer cut into six grids, each one row per dim over the kernels. mu and sigma are copied into rows
        // of their own so the accumulation reads every operand of a (kernel block, dim) at one offset.
        double[] gridBuffer = System.Buffers.ArrayPool<double>.Shared.Rent(6 * gridSize);
        try
        {
            var aGrid = new Span<double>(gridBuffer, 0, gridSize);
            var bGrid = new Span<double>(gridBuffer, gridSize, gridSize);
            var logSigma = new Span<double>(gridBuffer, 2 * gridSize, gridSize);
            var logGaussMass = new Span<double>(gridBuffer, 3 * gridSize, gridSize);
            var muGrid = new Span<double>(gridBuffer, 4 * gridSize, gridSize);
            var sigmaGrid = new Span<double>(gridBuffer, 5 * gridSize, gridSize);
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
                mu.AsSpan(0, nKernels).CopyTo(muGrid.Slice(row, nKernels));
                sigma.AsSpan(0, nKernels).CopyTo(sigmaGrid.Slice(row, nKernels));
            }
            TruncNorm.LogGaussMass(aGrid, bGrid, logGaussMass);

            AccumulateContinuous(
                xCont, nSamples, nCont, muGrid, sigmaGrid, aGrid, bGrid, logGaussMass, logSigma, weightedLogPdf, nKernels);
        }
        finally
        {
            System.Buffers.ArrayPool<double>.Shared.Return(gridBuffer);
        }
    }

    /// <summary>
    ///     Adds each candidate's continuous log density under each kernel to <paramref name="weightedLogPdf" />: the sum
    ///     over the continuous dimensions of <c>(norm_logpdf((x - mu) / sigma) - log_gauss_mass) - log(sigma)</c>, with
    ///     NaN where the kernel's standardized interval is empty (<c>a == b</c>) and <c>-inf</c> where <c>x</c> falls
    ///     outside it, as <c>np.select([a == b, (x &lt; a) | (x &gt; b)], [nan, -inf], default=out)</c> in
    ///     <c>_truncnorm.logpdf</c> followed by <c>.sum(axis=-1)</c>.
    /// </summary>
    /// <param name="xCont">
    ///     The candidates' continuous coordinates, log-transformed already for the log kinds, one row of
    ///     <paramref name="nCont" /> per candidate (<c>[s * n_cont + c]</c>); at least n_samples × n_cont long.
    /// </param>
    /// <param name="nSamples">The number of candidates.</param>
    /// <param name="nCont">The number of continuous dimensions.</param>
    /// <param name="muGrid">
    ///     The kernels' centers, one row of n_kernels per dimension (<c>[c * n_kernels + k]</c>); at least
    ///     n_cont × n_kernels long, like every grid.
    /// </param>
    /// <param name="sigmaGrid">The kernels' standard deviations, laid out as <paramref name="muGrid" />.</param>
    /// <param name="aGrid">The kernels' standardized lower bounds, laid out as <paramref name="muGrid" />.</param>
    /// <param name="bGrid">The kernels' standardized upper bounds, laid out as <paramref name="muGrid" />.</param>
    /// <param name="logGaussMass">The log Gaussian mass between the bounds, laid out as <paramref name="muGrid" />.</param>
    /// <param name="logSigma">The kernels' <c>log(sigma)</c>, laid out as <paramref name="muGrid" />.</param>
    /// <param name="weightedLogPdf">
    ///     (n_samples, n_kernels) row-major and at least that long; each candidate's sum per kernel is added in place.
    /// </param>
    /// <param name="nKernels">The number of kernels.</param>
    /// <exception cref="ArgumentException">
    ///     A grid, <paramref name="xCont" /> or <paramref name="weightedLogPdf" /> is shorter than the sizes imply; nothing
    ///     is read or written then.
    /// </exception>
    /// <remarks>
    ///     <para>
    ///         Each element gets exactly the scalar loop's operations: the sum starts at +0.0, receives the dimensions'
    ///         terms in dimension order, <c>((0 + t0) + t1) + …</c>, and is then added to the candidate's row
    ///         (<c>row + sum</c>). In the vector path a lane holds one kernel, so lanes never mix; the kernels after the
    ///         last full vector take the scalar path, the same kernels as before. The vector path computes every lane
    ///         with the scalar path's operations in the scalar path's order: the same division by sigma (never a
    ///         multiplication by its reciprocal), the same subtraction order, the NaN and <c>-inf</c> overrides applied
    ///         last. Its only freedom is the sign of zero in <c>-(z * z)</c>, which the subtraction of the normal's log
    ///         constant right after removes. .NET never contracts a multiply and an add into a fused multiply-add on its
    ///         own.
    ///     </para>
    ///     <para>
    ///         <b>Sums in registers, unchecked loads (plan 15).</b> The kernels are visited one vector-wide block at a
    ///         time, and each candidate's sums for the block stay in a register across the dimensions; before, a buffer
    ///         of n_kernels sums was cleared, read and written once per dimension and then added to the row, for every
    ///         candidate. Only the order in which different elements are computed changed. The loads go through
    ///         <see cref="System.Numerics.Vector.LoadUnsafe{T}(ref readonly T, nuint)" /> at one offset per (block, dimension),
    ///         which is why mu and sigma come as grid rows: with a span slice and a jagged-array lookup per operand, the
    ///         bounds checks were most of the loop's 60 instructions, and the unchecked loop runs 2.0 to 2.4 times
    ///         faster (same-process A/B, 100 to 3,000 kernels, same bits). Every length is checked once, up front, so no
    ///         load or store can leave its buffer. The halving in <c>-(z * z) / 2</c> is a multiplication by 0.5:
    ///         dividing by two and multiplying by a half give the correctly rounded value of the same real number (and
    ///         pass a NaN through alike), so the bits are equal, and the division by sigma that remains is the loop's
    ///         slowest instruction.
    ///     </para>
    /// </remarks>
    internal static void AccumulateContinuous(
        ReadOnlySpan<double> xCont, int nSamples, int nCont,
        ReadOnlySpan<double> muGrid, ReadOnlySpan<double> sigmaGrid,
        ReadOnlySpan<double> aGrid, ReadOnlySpan<double> bGrid, ReadOnlySpan<double> logGaussMass, ReadOnlySpan<double> logSigma,
        Span<double> weightedLogPdf, int nKernels)
    {
        // The unchecked loads below rely on these sizes; checking them here keeps every access inside its buffer.
        long gridSize = (long)nCont * nKernels;
        if (nSamples < 0 || nCont < 0 || nKernels < 0
            || xCont.Length < (long)nSamples * nCont || weightedLogPdf.Length < (long)nSamples * nKernels
            || muGrid.Length < gridSize || sigmaGrid.Length < gridSize || aGrid.Length < gridSize
            || bGrid.Length < gridSize || logGaussMass.Length < gridSize || logSigma.Length < gridSize)
            throw new ArgumentException("A continuous log-density buffer is shorter than its sizes imply.");

        if (System.Numerics.Vector.IsHardwareAccelerated && nKernels >= System.Numerics.Vector<double>.Count)
        {
            // Vector path: whole vector-wide kernel blocks here, then the kernels after the last whole block on the
            // scalar path, as before.
            int k = 0;
            int width = System.Numerics.Vector<double>.Count;
            var half = new System.Numerics.Vector<double>(0.5);
            var logC = new System.Numerics.Vector<double>(TruncNorm.NormPdfLogC);
            var nan = new System.Numerics.Vector<double>(double.NaN);
            var negativeInfinity = new System.Numerics.Vector<double>(double.NegativeInfinity);
            ref double x0 = ref System.Runtime.InteropServices.MemoryMarshal.GetReference(xCont);
            ref double mu0 = ref System.Runtime.InteropServices.MemoryMarshal.GetReference(muGrid);
            ref double sigma0 = ref System.Runtime.InteropServices.MemoryMarshal.GetReference(sigmaGrid);
            ref double a0 = ref System.Runtime.InteropServices.MemoryMarshal.GetReference(aGrid);
            ref double b0 = ref System.Runtime.InteropServices.MemoryMarshal.GetReference(bGrid);
            ref double logGaussMass0 = ref System.Runtime.InteropServices.MemoryMarshal.GetReference(logGaussMass);
            ref double logSigma0 = ref System.Runtime.InteropServices.MemoryMarshal.GetReference(logSigma);
            ref double row0 = ref System.Runtime.InteropServices.MemoryMarshal.GetReference(weightedLogPdf);
            for (; k <= nKernels - width; k += width)
            {
                for (int s = 0; s < nSamples; s++)
                {
                    // +0.0 in every lane, as the scalar loop's `double sum = 0.0`.
                    var sum = System.Numerics.Vector<double>.Zero;
                    nuint xRow = (nuint)s * (nuint)nCont;
                    // g walks this kernel block's column down the dimension rows: c * n_kernels + k.
                    nuint g = (nuint)k;
                    for (int c = 0; c < nCont; c++, g += (nuint)nKernels)
                    {
                        var aV = System.Numerics.Vector.LoadUnsafe(ref a0, g);
                        var bV = System.Numerics.Vector.LoadUnsafe(ref b0, g);
                        var xv = new System.Numerics.Vector<double>(System.Runtime.CompilerServices.Unsafe.Add(ref x0, xRow + (nuint)c));
                        var xstd = (xv - System.Numerics.Vector.LoadUnsafe(ref mu0, g)) / System.Numerics.Vector.LoadUnsafe(ref sigma0, g);
                        var normLogPdf = -(xstd * xstd) * half - logC;
                        var val = (normLogPdf - System.Numerics.Vector.LoadUnsafe(ref logGaussMass0, g))
                                  - System.Numerics.Vector.LoadUnsafe(ref logSigma0, g);
                        val = System.Numerics.Vector.ConditionalSelect(
                            System.Numerics.Vector.LessThan(xstd, aV) | System.Numerics.Vector.GreaterThan(xstd, bV),
                            negativeInfinity, val);
                        val = System.Numerics.Vector.ConditionalSelect(System.Numerics.Vector.Equals(aV, bV), nan, val);
                        sum += val;
                    }
                    nuint o = (nuint)s * (nuint)nKernels + (nuint)k;
                    System.Numerics.Vector.StoreUnsafe(System.Numerics.Vector.LoadUnsafe(ref row0, o) + sum, ref row0, o);
                }
            }
            AccumulateContinuousScalar(
                xCont, nSamples, nCont, muGrid, sigmaGrid, aGrid, bGrid, logGaussMass, logSigma, weightedLogPdf, nKernels, k);
        }
        else
        {
            // Scalar path: no vector hardware, or fewer kernels than one vector holds.
            AccumulateContinuousScalar(
                xCont, nSamples, nCont, muGrid, sigmaGrid, aGrid, bGrid, logGaussMass, logSigma, weightedLogPdf, nKernels, 0);
        }
    }

    /// <summary>
    ///     The scalar path of <see cref="AccumulateContinuous" /> for the kernels from <paramref name="firstKernel" /> on:
    ///     every kernel when there is no vector path, and the kernels after the vector path's last whole block otherwise.
    /// </summary>
    /// <param name="xCont">The candidates' continuous coordinates, laid out as for <see cref="AccumulateContinuous" />.</param>
    /// <param name="nSamples">The number of candidates.</param>
    /// <param name="nCont">The number of continuous dimensions.</param>
    /// <param name="muGrid">The kernels' centers, one row of n_kernels per dimension.</param>
    /// <param name="sigmaGrid">The kernels' standard deviations, laid out as <paramref name="muGrid" />.</param>
    /// <param name="aGrid">The kernels' standardized lower bounds, laid out as <paramref name="muGrid" />.</param>
    /// <param name="bGrid">The kernels' standardized upper bounds, laid out as <paramref name="muGrid" />.</param>
    /// <param name="logGaussMass">The log Gaussian mass between the bounds, laid out as <paramref name="muGrid" />.</param>
    /// <param name="logSigma">The kernels' <c>log(sigma)</c>, laid out as <paramref name="muGrid" />.</param>
    /// <param name="weightedLogPdf">(n_samples, n_kernels) row-major; each candidate's sum per kernel is added in place.</param>
    /// <param name="nKernels">The number of kernels.</param>
    /// <param name="firstKernel">The first kernel this path handles; the kernels before it are left alone.</param>
    /// <exception cref="IndexOutOfRangeException">A buffer is shorter than the sizes imply (the caller checks this first).</exception>
    /// <remarks>
    ///     The reference arithmetic the vector path reproduces lane by lane: the sum starts at +0.0, receives the
    ///     dimensions' terms in dimension order and is then added to the candidate's row. Every access is bounds-checked;
    ///     it handles at most a vector's width less one kernel per call on vector hardware.
    /// </remarks>
    private static void AccumulateContinuousScalar(
        ReadOnlySpan<double> xCont, int nSamples, int nCont,
        ReadOnlySpan<double> muGrid, ReadOnlySpan<double> sigmaGrid,
        ReadOnlySpan<double> aGrid, ReadOnlySpan<double> bGrid, ReadOnlySpan<double> logGaussMass, ReadOnlySpan<double> logSigma,
        Span<double> weightedLogPdf, int nKernels, int firstKernel)
    {
        for (int k = firstKernel; k < nKernels; k++)
        {
            for (int s = 0; s < nSamples; s++)
            {
                double sum = 0.0;
                for (int c = 0; c < nCont; c++)
                {
                    int g = c * nKernels + k;
                    double xstd = (xCont[s * nCont + c] - muGrid[g]) / sigmaGrid[g];
                    double val = TruncNorm.NormLogPdf(xstd) - logGaussMass[g] - logSigma[g];
                    if (aGrid[g] == bGrid[g])
                        val = double.NaN;
                    else if (xstd < aGrid[g] || xstd > bGrid[g])
                        val = double.NegativeInfinity;
                    sum += val;
                }
                weightedLogPdf[s * nKernels + k] += sum;
            }
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
    ///     The discrete branch of Python <c>_MixtureOfProductDistribution.log_pdf</c> for one stepped coordinate (a
    ///     discrete truncated normal or log-normal): adds the log Gaussian mass of each candidate's cell
    ///     <c>[x - step / 2, x + step / 2]</c> under each kernel to <paramref name="weightedLogPdf" />, then subtracts
    ///     each kernel's log mass over the adapted domain.
    /// </summary>
    /// <param name="x">The candidates, one row per sample and one column per coordinate.</param>
    /// <param name="column">This coordinate's column in <paramref name="x" />.</param>
    /// <param name="dtn">This coordinate's batched distributions; their step is positive.</param>
    /// <param name="weightedLogPdf">
    ///     Log densities, (n_samples, n_kernels) row-major and at least that long (a rented buffer may be longer); this
    ///     coordinate's terms are added in place.
    /// </param>
    /// <exception cref="IndexOutOfRangeException"><paramref name="weightedLogPdf" /> is shorter than n_samples × n_kernels.</exception>
    /// <remarks>
    ///     <para>
    ///         Python, for each stepped coordinate <c>i</c>:
    ///         <code>
    ///         xi_uniq, xi_inv = np.unique(x[:, i], return_inverse=True)
    ///         mu_uniq, sigma_uniq, mu_sigma_inv = _unique_inverse_2d(d.mu, d.sigma)
    ///         left = np.log(xi_uniq - step / 2) if is_log else (xi_uniq - step / 2)
    ///         right = np.log(xi_uniq + step / 2) if is_log else (xi_uniq + step / 2)
    ///         weighted_log_pdf += _log_gauss_mass_unique(
    ///             (left[:, np.newaxis] - mu_uniq) / sigma_uniq,
    ///             (right[:, np.newaxis] - mu_uniq) / sigma_uniq,
    ///         )[np.ix_(xi_inv, mu_sigma_inv)]
    ///         weighted_log_pdf -= _truncnorm._log_gauss_mass(
    ///             (d.adapted_low - mu_uniq) / sigma_uniq, (d.adapted_high - mu_uniq) / sigma_uniq
    ///         )[mu_sigma_inv]
    ///         </code>
    ///         The dedup inside <c>_log_gauss_mass_unique</c> is not merely an optimization: it fixes the array size
    ///         passed to <c>erf</c>, which switches between the exact and the vectorized approximation at 2000 elements
    ///         (see <see cref="Erf" />).
    ///     </para>
    ///     <para>
    ///         <b>Hashed dedups, same bits (plan 15).</b> The three dedups (the candidates' values, the kernels, and the
    ///         candidate-by-kernel grid of standardized cell bounds) used to sort with LINQ, the grid being up to
    ///         n_samples × n_kernels pairs: with ten integer parameters and a thousand kernels one call took 5 ms and
    ///         allocated 1.9 MB. <see cref="UniqueInverse" /> finds the same groups by hashing, in order of first
    ///         occurrence instead of sorted, and every buffer is rented. Nothing computed depends on that order: the
    ///         unique set and its size are the same,
    ///         <see cref="TruncNorm.LogGaussMass(ReadOnlySpan{double}, ReadOnlySpan{double}, Span{double})" /> is
    ///         elementwise and picks its <c>erf</c> path from the number of central elements alone, and each sum still
    ///         receives the same terms in the same order (the cell's log mass, then minus the domain's). The members of a
    ///         group differ at most in the sign of a zero, which no log mass depends on (<c>ndtr(±0)</c> is 0.5 either
    ///         way and both zeros take the same side of every comparison), so which member represents it does not
    ///         matter either.
    ///     </para>
    /// </remarks>
    internal static void AddDiscreteLogPdf(double[,] x, int column, BatchedNumericalDistributions dtn, double[] weightedLogPdf)
    {
        int nSamples = x.GetLength(0);
        double[] mu = dtn.Mu, sigma = dtn.Sigma;
        int nKernels = mu.Length;
        bool isLog = dtn.IsLog;
        double step = dtn.Step;
        // AdaptedLow / AdaptedHigh recompute a logarithm per access on the log kinds: the same value every time.
        double adaptedLow = dtn.AdaptedLow, adaptedHigh = dtn.AdaptedHigh;

        // Candidate- and kernel-sized buffers, cut from one rented array each: the column, its unique values (and the
        // 1-D dedup's second component, the same values again), the unique kernels, the domain's standardized bounds
        // and log mass per unique kernel; and the two inverse maps.
        double[] perCall = System.Buffers.ArrayPool<double>.Shared.Rent(Math.Max(3 * nSamples + 5 * nKernels, 1));
        int[] perCallIndexes = System.Buffers.ArrayPool<int>.Shared.Rent(Math.Max(nSamples + nKernels, 1));
        try
        {
            var xcol = new Span<double>(perCall, 0, nSamples);
            var xiUniq = new Span<double>(perCall, nSamples, nSamples);
            var xiUniqAgain = new Span<double>(perCall, 2 * nSamples, nSamples);
            var muUniq = new Span<double>(perCall, 3 * nSamples, nKernels);
            var sigmaUniq = new Span<double>(perCall, 3 * nSamples + nKernels, nKernels);
            var normLow = new Span<double>(perCall, 3 * nSamples + 2 * nKernels, nKernels);
            var normHigh = new Span<double>(perCall, 3 * nSamples + 3 * nKernels, nKernels);
            var norm = new Span<double>(perCall, 3 * nSamples + 4 * nKernels, nKernels);
            var xiInv = new Span<int>(perCallIndexes, 0, nSamples);
            var msInv = new Span<int>(perCallIndexes, nSamples, nKernels);

            for (int s = 0; s < nSamples; s++)
                xcol[s] = x[s, column];
            // np.unique of one array: a value paired with itself matches another pair exactly when the values compare
            // equal, so the pair dedup groups the values as the 1-D one did.
            int nXi = UniqueInverse(xcol, xcol, xiUniq, xiUniqAgain, xiInv);
            int nMs = UniqueInverse(mu, sigma, muUniq, sigmaUniq, msInv);

            // The standardized cell bounds of every (unique candidate, unique kernel) pair, row-major by candidate, their
            // dedup, and the log mass of each unique pair; sized by the dedups above, so rented only now.
            int gridSize = nXi * nMs;
            double[] grid = System.Buffers.ArrayPool<double>.Shared.Rent(Math.Max(5 * gridSize, 1));
            int[] gridIndexes = System.Buffers.ArrayPool<int>.Shared.Rent(Math.Max(gridSize, 1));
            try
            {
                var a = new Span<double>(grid, 0, gridSize);
                var b = new Span<double>(grid, gridSize, gridSize);
                var aUniq = new Span<double>(grid, 2 * gridSize, gridSize);
                var bUniq = new Span<double>(grid, 3 * gridSize, gridSize);
                var logMassUniq = new Span<double>(grid, 4 * gridSize, gridSize);
                var gridInv = new Span<int>(gridIndexes, 0, gridSize);
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
                int nUnique = UniqueInverse(a, b, aUniq, bUniq, gridInv);
                // Exactly the unique pairs, as _log_gauss_mass_unique passes them: their count picks the erf path.
                TruncNorm.LogGaussMass(aUniq[..nUnique], bUniq[..nUnique], logMassUniq);

                for (int s = 0; s < nSamples; s++)
                {
                    int row = xiInv[s] * nMs;
                    for (int k = 0; k < nKernels; k++)
                        weightedLogPdf[s * nKernels + k] += logMassUniq[gridInv[row + msInv[k]]];
                }
            }
            finally
            {
                System.Buffers.ArrayPool<double>.Shared.Return(grid);
                System.Buffers.ArrayPool<int>.Shared.Return(gridIndexes);
            }

            for (int qi = 0; qi < nMs; qi++)
            {
                normLow[qi] = (adaptedLow - muUniq[qi]) / sigmaUniq[qi];
                normHigh[qi] = (adaptedHigh - muUniq[qi]) / sigmaUniq[qi];
            }
            // Python skips the dedup here ("very unlikely to observe duplications"), so the count is the unique kernels'.
            TruncNorm.LogGaussMass(normLow[..nMs], normHigh[..nMs], norm);
            for (int s = 0; s < nSamples; s++)
            {
                for (int k = 0; k < nKernels; k++)
                    weightedLogPdf[s * nKernels + k] -= norm[msInv[k]];
            }
        }
        finally
        {
            System.Buffers.ArrayPool<double>.Shared.Return(perCall);
            System.Buffers.ArrayPool<int>.Shared.Return(perCallIndexes);
        }
    }

    /// <summary>
    ///     The unique (a, b) pairs of two equally long arrays, in order of first occurrence, and for each position the
    ///     index of its pair among them: Python's <c>_unique_inverse_2d(a, b)</c>, and <c>np.unique(x,
    ///     return_inverse=True)</c> when both arrays are <c>x</c>, except that those return the pairs sorted.
    /// </summary>
    /// <param name="a">The pairs' first components.</param>
    /// <param name="b">The pairs' second components; as long as <paramref name="a" />, and may be the same span.</param>
    /// <param name="uniqueA">
    ///     Receives the unique pairs' first components; at least as long as <paramref name="a" /> and overlapping neither
    ///     input.
    /// </param>
    /// <param name="uniqueB">Receives their second components; the same requirements as <paramref name="uniqueA" />.</param>
    /// <param name="inverse">
    ///     Receives, for each position of <paramref name="a" />, the index of its pair in <paramref name="uniqueA" /> and
    ///     <paramref name="uniqueB" />; at least as long as <paramref name="a" />.
    /// </param>
    /// <returns>The number of unique pairs, written to the front of <paramref name="uniqueA" /> and <paramref name="uniqueB" />.</returns>
    /// <exception cref="IndexOutOfRangeException"><paramref name="b" /> or an output is shorter than <paramref name="a" />.</exception>
    /// <remarks>
    ///     <para>
    ///         Two pairs are the same when both components compare equal with <c>==</c>, the test NumPy's version applies
    ///         to neighbors after sorting (<c>(a_order[1:] != a_order[:-1]) | (b_order[1:] != b_order[:-1])</c>): -0.0
    ///         equals 0.0, and a NaN equals nothing, so a pair holding a NaN is always a new one. The pair kept for a group
    ///         is its first occurrence, as with the stable LINQ sort this replaced. Only the order of the unique pairs
    ///         differs from Python's, and no caller computes anything that depends on it.
    ///     </para>
    ///     <para>
    ///         An open-addressing table with linear probing, at most half full, holds each unique pair's index plus one
    ///         (zero marks an empty slot). Pairs holding a NaN are never stored, because no later pair can match them, so
    ///         they cannot pile up in one probe chain either.
    ///     </para>
    /// </remarks>
    internal static int UniqueInverse(
        ReadOnlySpan<double> a, ReadOnlySpan<double> b, Span<double> uniqueA, Span<double> uniqueB, Span<int> inverse)
    {
        int n = a.Length;
        int capacity = (int)System.Numerics.BitOperations.RoundUpToPowerOf2((uint)Math.Max(2 * n, 16));
        int mask = capacity - 1;
        int[] table = System.Buffers.ArrayPool<int>.Shared.Rent(capacity);
        try
        {
            // A rented array holds a previous caller's entries: every slot must read as empty first.
            Array.Clear(table, 0, capacity);
            int count = 0;
            for (int i = 0; i < n; i++)
            {
                double ai = a[i], bi = b[i];
                if (double.IsNaN(ai) || double.IsNaN(bi))
                {
                    uniqueA[count] = ai;
                    uniqueB[count] = bi;
                    inverse[i] = count++;
                    continue;
                }
                int slot = PairHash(ai, bi) & mask;
                while (true)
                {
                    int entry = table[slot];
                    if (entry == 0)
                    {
                        uniqueA[count] = ai;
                        uniqueB[count] = bi;
                        inverse[i] = count++;
                        table[slot] = count;
                        break;
                    }
                    if (uniqueA[entry - 1] == ai && uniqueB[entry - 1] == bi)
                    {
                        inverse[i] = entry - 1;
                        break;
                    }
                    slot = (slot + 1) & mask;
                }
            }
            return count;
        }
        finally
        {
            System.Buffers.ArrayPool<int>.Shared.Return(table);
        }
    }

    /// <summary>A hash of a pair of doubles, equal for any two pairs whose components compare equal.</summary>
    /// <param name="a">The first component; not NaN.</param>
    /// <param name="b">The second component; not NaN.</param>
    /// <returns>A well-mixed 32-bit value, possibly negative; callers mask it to a power-of-two table size.</returns>
    private static int PairHash(double a, double b)
    {
        // -0.0 == 0.0, so the two zeros must hash alike: both hash as +0.0. Any other values that compare equal have
        // the same bits.
        ulong ka = a == 0.0 ? 0UL : (ulong)BitConverter.DoubleToInt64Bits(a);
        ulong kb = b == 0.0 ? 0UL : (ulong)BitConverter.DoubleToInt64Bits(b);
        // SplitMix64's finalizer over both keys: doubles of integers differ only in their high bits, which the
        // multiplications and shifts spread over the low bits the mask keeps.
        ulong h = (ka * 0x9E3779B97F4A7C15UL) ^ kb;
        h = (h ^ (h >> 30)) * 0xBF58476D1CE4E5B9UL;
        h = (h ^ (h >> 27)) * 0x94D049BB133111EBUL;
        return (int)(h ^ (h >> 31));
    }
}
