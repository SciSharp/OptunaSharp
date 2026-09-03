using System;
using System.Collections.Generic;
using System.Linq;
using OptunaSharp.Internal;

namespace OptunaSharp.Samplers.Tpe;

/// <summary>
///     Partial port of <c>optuna/samplers/_tpe/probability_distributions.py</c>. Only the batched
///     distribution kinds and the <c>log_pdf</c> path that <c>PedAnovaImportanceEvaluator</c>
///     depends on are ported here (categorical + discrete truncated-normal). The continuous
///     truncated-normal / log-normal kinds, and <c>sample</c>, land with the TPE sampler in
///     Phase 2.8.
/// </summary>
internal abstract class BatchedDistributions
{
}

/// <summary>Python <c>_BatchedCategoricalDistributions</c>.</summary>
internal sealed class BatchedCategoricalDistributions : BatchedDistributions
{
    /// <summary>Per-kernel categorical weights, shape <c>(n_kernels, n_choices)</c>.</summary>
    internal double[,] Weights { get; }

    internal BatchedCategoricalDistributions(double[,] weights) => Weights = weights;
}

/// <summary>Python <c>_BatchedDiscreteTruncNormDistributions</c>.</summary>
internal sealed class BatchedDiscreteTruncNormDistributions : BatchedDistributions
{
    internal double[] Mu { get; }
    internal double[] Sigma { get; }
    internal double Low { get; }
    internal double High { get; }
    internal double Step { get; }

    internal BatchedDiscreteTruncNormDistributions(double[] mu, double[] sigma, double low, double high, double step)
    {
        Mu = mu;
        Sigma = sigma;
        Low = low;
        High = high;
        Step = step;
    }

    internal double AdaptedLow => Low - Step / 2.0;
    internal double AdaptedHigh => High + Step / 2.0;
    internal bool IsLog => false;
}

/// <summary>Python <c>_MixtureOfProductDistribution</c>.</summary>
internal sealed class MixtureOfProductDistribution
{
    internal double[] Weights { get; }
    internal IReadOnlyList<BatchedDistributions> Distributions { get; }

    internal MixtureOfProductDistribution(double[] weights, IReadOnlyList<BatchedDistributions> distributions)
    {
        Weights = weights;
        Distributions = distributions;
    }

    /// <summary>Python <c>_MixtureOfProductDistribution.log_pdf(x)</c>, with <c>x</c> shape <c>(n_samples, dim)</c>.</summary>
    internal double[] LogPdf(double[,] x)
    {
        int nSamples = x.GetLength(0);
        int dim = x.GetLength(1);
        int nKernels = Weights.Length;
        var weightedLogPdf = new double[nSamples, nKernels];

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
                            weightedLogPdf[s, k] += Math.Log(cat.Weights[k, xi]);
                    }
                    break;
                }

                case BatchedDiscreteTruncNormDistributions dtn:
                {
                    // NOTE: PED-ANOVA's Scott estimator only ever produces the non-log discrete kind.
                    double step = dtn.Step;
                    var xcol = new double[nSamples];
                    for (int s = 0; s < nSamples; s++)
                        xcol[s] = x[s, i];

                    var (xiUniq, xiInv) = UniqueInverse1d(xcol);
                    var (muUniq, sigmaUniq, msInv) = UniqueInverse2d(dtn.Mu, dtn.Sigma);
                    int nXi = xiUniq.Length;
                    int nMs = muUniq.Length;

                    // left/right for the non-log discrete kind: xi_uniq ∓ step / 2.
                    var a = new double[nXi * nMs];
                    var b = new double[nXi * nMs];
                    for (int p = 0; p < nXi; p++)
                    {
                        double left = xiUniq[p] - step / 2.0;
                        double right = xiUniq[p] + step / 2.0;
                        for (int q = 0; q < nMs; q++)
                        {
                            a[p * nMs + q] = (left - muUniq[q]) / sigmaUniq[q];
                            b[p * nMs + q] = (right - muUniq[q]) / sigmaUniq[q];
                        }
                    }

                    var logMass = LogGaussMassUnique(a, b);  // shape (nXi, nMs), row-major.
                    for (int s = 0; s < nSamples; s++)
                    {
                        int row = xiInv[s] * nMs;
                        for (int k = 0; k < nKernels; k++)
                            weightedLogPdf[s, k] += logMass[row + msInv[k]];
                    }

                    var normLow = new double[nMs];
                    var normHigh = new double[nMs];
                    for (int q = 0; q < nMs; q++)
                    {
                        normLow[q] = (dtn.AdaptedLow - muUniq[q]) / sigmaUniq[q];
                        normHigh[q] = (dtn.AdaptedHigh - muUniq[q]) / sigmaUniq[q];
                    }
                    var norm = TruncNorm.LogGaussMass(normLow, normHigh);
                    for (int s = 0; s < nSamples; s++)
                    {
                        for (int k = 0; k < nKernels; k++)
                            weightedLogPdf[s, k] -= norm[msInv[k]];
                    }
                    break;
                }

                default:
                    throw new NotImplementedException(
                        "Continuous / log-scaled batched distributions are not ported yet " +
                        "(PORTING_PLAN Phase 2.8). PED-ANOVA only exercises the categorical and " +
                        "non-log discrete truncated-normal kinds.");
            }
        }

        // weighted_log_pdf += np.log(self.weights[np.newaxis]).
        for (int s = 0; s < nSamples; s++)
        {
            for (int k = 0; k < nKernels; k++)
                weightedLogPdf[s, k] += Math.Log(Weights[k]);
        }

        // np.log(np.exp(weighted_log_pdf - max_[:, None]).sum(axis=1)) + max_, with -inf max_ -> 0.
        var result = new double[nSamples];
        var expBuf = new double[nKernels];
        for (int s = 0; s < nSamples; s++)
        {
            double max = double.NegativeInfinity;
            for (int k = 0; k < nKernels; k++)
                max = Math.Max(max, weightedLogPdf[s, k]);
            if (double.IsNegativeInfinity(max))
                max = 0.0;
            for (int k = 0; k < nKernels; k++)
                expBuf[k] = Math.Exp(weightedLogPdf[s, k] - max);
            result[s] = Math.Log(MathUtil.PairwiseSum(expBuf)) + max;
        }
        return result;
    }

    /// <summary>Python <c>_MixtureOfProductDistribution.sample</c> — arrives with TPE (Phase 2.8).</summary>
    internal double[,] Sample(LazyRandomState rng, int batchSize)
        => throw new NotImplementedException(
            "_MixtureOfProductDistribution.sample is not ported yet (PORTING_PLAN Phase 2.8, TPE " +
            "sampler). PED-ANOVA only uses log_pdf.");

    /// <summary>
    ///     Python <c>_log_gauss_mass_unique(a, b)</c>: dedup the raveled (a, b) pairs, evaluate
    ///     <c>_log_gauss_mass</c> once per unique pair, then scatter back. This is not merely an
    ///     optimization here — it fixes the array size passed to <c>erf</c>, which switches between
    ///     the exact and the vectorized approximation at 2000 elements (see <see cref="Erf" />).
    /// </summary>
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
