using System;
using System.Collections.Generic;
using System.Linq;
using NumSharp;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Internal;
using OptunaSharp.Samplers;

namespace OptunaSharp.Samplers.Tpe;

/// <summary>Python <c>_ParzenEstimatorParameters</c> (a <c>NamedTuple</c>).</summary>
internal sealed class ParzenEstimatorParameters
{
    /// <summary>Weight assigned to the broad prior kernel.</summary>
    internal double PriorWeight { get; }
    /// <summary>Whether the minimum-bandwidth heuristic is enabled.</summary>
    internal bool ConsiderMagicClip { get; }
    /// <summary>Whether domain endpoints contribute to bandwidth selection.</summary>
    internal bool ConsiderEndpoints { get; }
    /// <summary>Callback computing one nonnegative weight per historical observation.</summary>
    internal Func<int, double[]> Weights { get; }

    /// <summary>Historical / unused field (Optuna keeps it "for historical reasons").</summary>
    internal bool Multivariate { get; }

    /// <summary>Stores kernel-prior, bandwidth, and observation-weight settings used by TPE estimators.</summary>
    /// <param name="priorWeight">Nonnegative weight assigned to the broad prior kernel.</param>
    /// <param name="considerMagicClip">Whether kernel widths receive the minimum-bandwidth heuristic.</param>
    /// <param name="considerEndpoints">Whether domain endpoints participate in bandwidth estimation.</param>
    /// <param name="weights">Callback returning one observation weight per historical sample.</param>
    /// <param name="multivariate">Whether joint-estimation settings are enabled; retained for upstream compatibility where unused.</param>
    internal ParzenEstimatorParameters(
        double priorWeight, bool considerMagicClip, bool considerEndpoints,
        Func<int, double[]> weights, bool multivariate)
    {
        PriorWeight = priorWeight;
        ConsiderMagicClip = considerMagicClip;
        ConsiderEndpoints = considerEndpoints;
        Weights = weights;
        Multivariate = multivariate;
    }
}

/// <summary>
///     Port of <c>optuna/samplers/_tpe/parzen_estimator.py: _ParzenEstimator</c> — one Gaussian
///     Mixture Model per parameter, fit to a set of trial observations, wrapped in a
///     <see cref="MixtureOfProductDistribution" />.
///
///     <para><b>[Misaligned] argsort tie order.</b> <c>compute_sigmas</c> calls <c>np.argsort(mus)</c>.
///     NumPy 2.x sorts float64 with x86-simd-sort, whose tie order is a SIMD-network permutation that
///     is not portably reproducible (it isn't stable, isn't classic quicksort/heapsort, and varies by
///     CPU ISA). OptunaSharp uses a <em>stable</em> sort. This is exact whenever the observation values
///     are <em>distinct</em> — so <b>every all-continuous-float study matches Optuna</b> (bounded only
///     by the <see cref="Erf" /> tolerance), as do categorical params (no argsort) and, in the
///     independent (<c>multivariate=false</c>) path, discrete params while the "above" set has &lt;25
///     observations (there the kernel weights are uniform, so the 1-D tie order cancels). The residual
///     divergence — <b>repeated discrete/int observations in a multivariate study</b> (where each
///     dimension's bandwidth is paired with a joint kernel, so the tie order does not cancel), or a
///     ≥25-observation independent study with ties — matches Optuna's algorithm exactly but can pick a
///     different sigma→kernel pairing, and is documented as an accepted divergence like the erf one.</para>
/// </summary>
internal sealed class ParzenEstimator
{
    private const double Eps = 1e-12;

    private readonly OrderedDict<string, BaseDistribution> _searchSpace;
    private readonly MixtureOfProductDistribution _mixtureDistribution;

    /// <summary>Builds a weighted mixture of product kernels from encoded historical observations.</summary>
    /// <param name="observations">Encoded historical values indexed by parameter name, with aligned observation rows.</param>
    /// <param name="searchSpace">Ordered parameter distributions defining the coordinates used by this operation.</param>
    /// <param name="parameters">Kernel construction and observation-weight configuration.</param>
    /// <param name="predeterminedWeights">Optional precomputed observation weights replacing the configured callback.</param>
    /// <exception cref="NumSharp.ValueError">The prior weight is negative, or the configured weight callback returns negative, non-finite, or all-zero weights.</exception>
    internal ParzenEstimator(
        IReadOnlyDictionary<string, double[]> observations,
        OrderedDict<string, BaseDistribution> searchSpace,
        ParzenEstimatorParameters parameters,
        double[]? predeterminedWeights = null)
    {
        if (parameters.PriorWeight < 0)
            throw new ValueError(
                "A non-negative value must be specified for prior_weight, " +
                $"but got {PythonRepr.FloatRepr(parameters.PriorWeight)}.");

        _searchSpace = searchSpace;

        // transformed_observations = np.array([observations[p] for p in search_space]).T — its row
        // count is the number of contributing trials (identical across params by construction).
        int nObs = searchSpace.Count == 0 ? 0 : observations[searchSpace.Keys.First()].Length;

        System.Diagnostics.Debug.Assert(predeterminedWeights is null || nObs == predeterminedWeights.Length);
        double[] weights = predeterminedWeights ?? CallWeightsFunc(parameters.Weights, nObs);

        if (nObs == 0)
        {
            weights = new[] { 1.0 };
        }
        else
        {
            var extended = new double[weights.Length + 1];
            Array.Copy(weights, extended, weights.Length);
            extended[^1] = parameters.PriorWeight;
            weights = extended;
        }
        double sum = MathUtil.PairwiseSum(weights);
        for (int i = 0; i < weights.Length; i++)
            weights[i] /= sum;

        var dists = new List<BatchedDistributions>(searchSpace.Count);
        foreach (var (param, dist) in searchSpace)
            dists.Add(CalculateDistributions(observations[param], dist, parameters));

        _mixtureDistribution = new MixtureOfProductDistribution(weights, dists);
    }

    /// <summary>Exposed for differential tests (mirrors Optuna asserting on <c>_mixture_distribution</c>).</summary>
    internal MixtureOfProductDistribution MixtureDistribution => _mixtureDistribution;

    /// <summary>Python <c>_ParzenEstimator.sample(rng, size)</c>.</summary>
    /// <param name="rng">Random stream whose draw order is shared with the reference algorithm.</param>
    /// <param name="size">Number of candidate samples to draw.</param>
    /// <returns>Sample arrays keyed by parameter name, with aligned candidate indices and original internal distribution scales.</returns>
    internal OrderedDict<string, double[]> Sample(LazyRandomState rng, int size)
    {
        var sampled = _mixtureDistribution.Sample(rng, size);
        return Untransform(sampled);
    }

    /// <summary>Python <c>_ParzenEstimator.log_pdf(samples_dict)</c>.</summary>
    /// <param name="samplesDict">Candidate values indexed by parameter name, with aligned sample rows.</param>
    /// <returns>One mixture log probability density per sample row.</returns>
    internal double[] LogPdf(IReadOnlyDictionary<string, double[]> samplesDict)
    {
        var transformed = Transform(samplesDict);
        return _mixtureDistribution.LogPdf(transformed);
    }

    /// <summary>Python <c>_transform</c>: stack each param's column into a <c>(n_samples, n_params)</c> array.</summary>
    /// <param name="samplesDict">Candidate values indexed by parameter name, with aligned sample rows.</param>
    /// <returns>A sample-by-parameter matrix with logarithms applied to log-scaled coordinates.</returns>
    private double[,] Transform(IReadOnlyDictionary<string, double[]> samplesDict)
    {
        int nParams = _searchSpace.Count;
        int nSamples = nParams == 0 ? 0 : samplesDict[_searchSpace.Keys.First()].Length;
        var result = new double[nSamples, nParams];
        int col = 0;
        foreach (var param in _searchSpace.Keys)
        {
            var arr = samplesDict[param];
            for (int s = 0; s < nSamples; s++)
                result[s, col] = arr[s];
            col++;
        }
        return result;
    }

    /// <summary>Python <c>_untransform</c>: split a <c>(n_samples, n_params)</c> array back into a dict.</summary>
    /// <param name="samplesArray">Candidate matrix with one sample per row and one parameter per column.</param>
    /// <returns>Candidate coordinates restored from kernel space and keyed by parameter name.</returns>
    private OrderedDict<string, double[]> Untransform(double[,] samplesArray)
    {
        int nSamples = samplesArray.GetLength(0);
        var result = new OrderedDict<string, double[]>(_searchSpace.Count);
        int col = 0;
        foreach (var param in _searchSpace.Keys)
        {
            var arr = new double[nSamples];
            for (int s = 0; s < nSamples; s++)
                arr[s] = samplesArray[s, col];
            result[param] = arr;
            col++;
        }
        return result;
    }

    /// <summary>Python <c>_calculate_distributions</c>.</summary>
    /// <param name="observations">Historical internal values for this single parameter.</param>
    /// <param name="searchSpace">Distribution selecting the categorical or numerical kernel family.</param>
    /// <param name="parameters">Kernel construction and observation-weight configuration.</param>
    /// <returns>The batched kernel family appropriate to the parameter distribution.</returns>
    /// <exception cref="System.ArgumentException">The distribution is neither categorical nor a supported float/integer distribution.</exception>
    private static BatchedDistributions CalculateDistributions(
        double[] observations, BaseDistribution searchSpace, ParzenEstimatorParameters parameters)
    {
        if (searchSpace is CategoricalDistribution cat)
            return CalculateCategoricalDistributions(observations, cat, parameters);
        return CalculateNumericalDistributions(observations, searchSpace, parameters);
    }

    /// <summary>Python <c>_calculate_categorical_distributions</c>.</summary>
    /// <param name="observations">Historical zero-based choice indices for this parameter.</param>
    /// <param name="searchSpace">Categorical distribution fixing the choice count and order.</param>
    /// <param name="parameters">Kernel construction and observation-weight configuration.</param>
    /// <returns>Categorical kernel probability rows for observed choices plus the prior.</returns>
    private static BatchedDistributions CalculateCategoricalDistributions(
        double[] observations, CategoricalDistribution searchSpace, ParzenEstimatorParameters parameters)
    {
        int nChoices = searchSpace.Choices.Count;
        if (observations.Length == 0)
        {
            var uniform = new double[1, nChoices];
            for (int c = 0; c < nChoices; c++)
                uniform[0, c] = 1.0 / nChoices;
            return new BatchedCategoricalDistributions(uniform);
        }

        int nKernels = observations.Length + 1;  // +1 for prior.
        var weights = new double[nKernels, nChoices];
        for (int k = 0; k < nKernels; k++)
        {
            for (int c = 0; c < nChoices; c++)
                weights[k, c] = parameters.PriorWeight / nKernels;
        }
        for (int j = 0; j < observations.Length; j++)
            weights[j, (int)observations[j]] += 1;

        var row = new double[nChoices];
        for (int k = 0; k < nKernels; k++)
        {
            for (int c = 0; c < nChoices; c++)
                row[c] = weights[k, c];
            double rowSum = MathUtil.PairwiseSum(row);
            double denom = rowSum == 0 ? 1 : rowSum;
            for (int c = 0; c < nChoices; c++)
                weights[k, c] /= denom;
        }
        return new BatchedCategoricalDistributions(weights);
    }

    /// <summary>Python <c>_calculate_numerical_distributions</c> (with the inner <c>compute_sigmas</c>).</summary>
    /// <param name="observations">Historical numeric values before any logarithmic transformation.</param>
    /// <param name="searchSpace">Float or integer distribution defining bounds, logarithmic scale, and quantization.</param>
    /// <param name="parameters">Kernel construction and observation-weight configuration.</param>
    /// <returns>Numerical kernels with centers, bandwidths, and truncation bounds.</returns>
    /// <exception cref="System.ArgumentException">The supplied distribution is not a supported float or integer distribution.</exception>
    private static BatchedDistributions CalculateNumericalDistributions(
        double[] observations, BaseDistribution searchSpace, ParzenEstimatorParameters parameters)
    {
        double low, high;
        bool log;
        double? step;
        switch (searchSpace)
        {
            case FloatDistribution f:
                low = f.Low;
                high = f.High;
                log = f.Log;
                step = f.Step;
                break;
            case IntDistribution i:
                low = i.Low;
                high = i.High;
                log = i.Log;
                step = i.Step;  // IntDistribution always has an integer step (never continuous).
                break;
            default:
                throw new ArgumentException($"Unexpected distribution type {searchSpace.GetType()}.");
        }

        if (step is not null)
        {
            low -= step.Value / 2.0;
            high += step.Value / 2.0;
        }

        // mus = observations (or their logs), then np.append(mus, [0.5 * (low + high)]) and
        // np.append(sigmas, [high - low]). Plan 15: the centers and the widths are written straight into the arrays that
        // carry the prior's slot, instead of into arrays of their own copied over afterwards: the same values, two
        // arrays per parameter fewer per estimator (three on the log kinds).
        int n = observations.Length;
        var musWithPrior = new double[n + 1];
        if (log)
        {
            for (int i = 0; i < n; i++)
                musWithPrior[i] = Math.Log(observations[i]);
            low = Math.Log(low);
            high = Math.Log(high);
        }
        else
        {
            Array.Copy(observations, musWithPrior, n);
        }
        ReadOnlySpan<double> mus = musWithPrior.AsSpan(0, n);

        var sigmasWithPrior = new double[n + 1];
        ComputeSigmas(mus, low, high, n, parameters, sigmasWithPrior.AsSpan(0, n));

        musWithPrior[^1] = 0.5 * (low + high);
        sigmasWithPrior[^1] = high - low;

        // The original (pre-log, pre-step-widen) low/high/step drive the batched kind.
        double origLow = searchSpace is IntDistribution il ? il.Low : ((FloatDistribution)searchSpace).Low;
        double origHigh = searchSpace is IntDistribution ih ? ih.High : ((FloatDistribution)searchSpace).High;

        if (step is null)
        {
            return log
                ? new BatchedTruncLogNormDistributions(musWithPrior, sigmasWithPrior, origLow, origHigh)
                : new BatchedTruncNormDistributions(musWithPrior, sigmasWithPrior, origLow, origHigh);
        }
        return log
            ? new BatchedDiscreteTruncLogNormDistributions(musWithPrior, sigmasWithPrior, origLow, origHigh, step.Value)
            : new BatchedDiscreteTruncNormDistributions(musWithPrior, sigmasWithPrior, origLow, origHigh, step.Value);
    }

    /// <summary>Python <c>compute_sigmas()</c> nested in <c>_calculate_numerical_distributions</c>.</summary>
    /// <param name="mus">Kernel centers in transformed numerical coordinates.</param>
    /// <param name="low">Lower endpoint of the numerical domain.</param>
    /// <param name="high">Upper endpoint of the numerical domain.</param>
    /// <param name="nObservations">Number of historical observations represented by the kernels.</param>
    /// <param name="parameters">Kernel construction and observation-weight configuration.</param>
    /// <returns>Per-observation kernel widths after endpoint and clipping heuristics.</returns>
    /// <remarks>
    ///     The unclipped widths come from <see cref="BinnedRawSigmas(double[], double, double, double, int, bool)" /> when it can prove they clip to exactly what Optuna's
    ///     sort gives (plan 14: the sort was a quarter of a 1,000-trial, 40-parameter TPE run), otherwise from
    ///     <see cref="SortedRawSigmas(double[], double, double, bool)" />, Optuna's sort-based computation. Both then go through the same clip.
    /// </remarks>
    internal static double[] ComputeSigmas(
        double[] mus, double low, double high, int nObservations, ParzenEstimatorParameters parameters)
    {
        var sigmas = new double[mus.Length];
        ComputeSigmas(mus, low, high, nObservations, parameters, sigmas);
        return sigmas;
    }

    /// <summary>
    ///     <see cref="ComputeSigmas(double[], double, double, int, ParzenEstimatorParameters)" /> into a caller's buffer:
    ///     the same widths, written to <paramref name="sigmas" /> (plan 15: the estimator passes the slots of its
    ///     prior-extended array, so no width array is copied over afterwards).
    /// </summary>
    /// <param name="mus">Kernel centers in transformed numerical coordinates.</param>
    /// <param name="low">Lower endpoint of the numerical domain.</param>
    /// <param name="high">Upper endpoint of the numerical domain.</param>
    /// <param name="nObservations">Number of historical observations represented by the kernels.</param>
    /// <param name="parameters">Kernel construction and observation-weight configuration.</param>
    /// <param name="sigmas">Receives one width per center; at least as long as <paramref name="mus" />, and not overlapping it.</param>
    /// <exception cref="IndexOutOfRangeException"><paramref name="sigmas" /> is shorter than <paramref name="mus" />.</exception>
    internal static void ComputeSigmas(
        ReadOnlySpan<double> mus, double low, double high, int nObservations, ParzenEstimatorParameters parameters,
        Span<double> sigmas)
    {
        int n = mus.Length;
        double maxsigma = high - low;
        double minsigma;
        int nBins = 0;
        if (parameters.ConsiderMagicClip)
        {
            int nKernels = nObservations + 1;  // +1 for prior.
            double divisor = Math.Min(100.0, 1.0 + nKernels);
            minsigma = (high - low) / divisor;
            // One bin per minsigma of the domain: (high - low) / minsigma is the divisor, a whole number from 3 to 100.
            nBins = (int)divisor;
        }
        else
        {
            minsigma = Eps;
        }

        // The binned path writes the widths only when it succeeds; otherwise the sort writes them.
        if (!(parameters.ConsiderMagicClip
              && BinnedRawSigmas(mus, low, high, minsigma, nBins, parameters.ConsiderEndpoints, sigmas)))
            SortedRawSigmas(mus, low, high, parameters.ConsiderEndpoints, sigmas);

        for (int t = 0; t < n; t++)
            sigmas[t] = MathUtil.Clip(sigmas[t], minsigma, maxsigma);
    }

    /// <summary>
    ///     Optuna's unclipped kernel widths, computed as <c>compute_sigmas</c> does: sort the centers, take each one's larger
    ///     distance to its sorted neighbors (the domain endpoints at both ends), and with <c>consider_endpoints=False</c> and
    ///     at least two centers drop the endpoint distance of the smallest and the largest.
    /// </summary>
    /// <param name="mus">Kernel centers in transformed numerical coordinates; not modified.</param>
    /// <param name="low">Lower endpoint of the numerical domain.</param>
    /// <param name="high">Upper endpoint of the numerical domain.</param>
    /// <param name="considerEndpoints">Whether the domain endpoints count as neighbors of the smallest and largest center.</param>
    /// <returns>One unclipped width per center, in the centers' order.</returns>
    internal static double[] SortedRawSigmas(double[] mus, double low, double high, bool considerEndpoints)
    {
        var sigmas = new double[mus.Length];
        SortedRawSigmas(mus, low, high, considerEndpoints, sigmas);
        return sigmas;
    }

    /// <summary>
    ///     <see cref="SortedRawSigmas(double[], double, double, bool)" /> into a caller's buffer: the same widths, in the
    ///     centers' order.
    /// </summary>
    /// <param name="mus">Kernel centers in transformed numerical coordinates; not modified.</param>
    /// <param name="low">Lower endpoint of the numerical domain.</param>
    /// <param name="high">Upper endpoint of the numerical domain.</param>
    /// <param name="considerEndpoints">Whether the domain endpoints count as neighbors of the smallest and largest center.</param>
    /// <param name="sigmas">Receives one unclipped width per center; at least as long as <paramref name="mus" />.</param>
    /// <exception cref="IndexOutOfRangeException"><paramref name="sigmas" /> is shorter than <paramref name="mus" />.</exception>
    internal static void SortedRawSigmas(
        ReadOnlySpan<double> mus, double low, double high, bool considerEndpoints, Span<double> sigmas)
    {
        int n = mus.Length;

        // sorted_indices = np.argsort(mus). Stable — exact for distinct mus. NumPy 2.x sorts float64
        // with x86-simd-sort, whose tie order is not portably reproducible; see the class remark for
        // the (documented) divergence this leaves on repeated numerical observations.
        var sortedIndices = StableArgsort(mus);

        // sorted_mus_with_endpoints = [low, *sorted_mus, high].
        var swe = new double[n + 2];
        swe[0] = low;
        for (int t = 0; t < n; t++)
            swe[t + 1] = mus[sortedIndices[t]];
        swe[^1] = high;

        // sorted_sigmas = maximum(swe[1:-1] - swe[0:-2], swe[2:] - swe[1:-1]).
        var sortedSigmas = new double[n];
        for (int t = 0; t < n; t++)
            sortedSigmas[t] = Math.Max(swe[t + 1] - swe[t], swe[t + 2] - swe[t + 1]);

        if (!considerEndpoints && swe.Length >= 4)
        {
            sortedSigmas[0] = swe[2] - swe[1];
            sortedSigmas[^1] = swe[^2] - swe[^3];
        }

        // sigmas = sorted_sigmas[np.argsort(sorted_indices)] — invert the (distinct) permutation.
        for (int t = 0; t < n; t++)
            sigmas[sortedIndices[t]] = sortedSigmas[t];
    }

    /// <summary>
    ///     Unclipped kernel widths that clip to exactly what <see cref="SortedRawSigmas(double[], double, double, bool)" /> clips to, found in O(n) without
    ///     sorting, or <see langword="null" /> when that cannot be proven for these centers (the caller then sorts).
    /// </summary>
    /// <param name="mus">Kernel centers in transformed numerical coordinates; not modified.</param>
    /// <param name="low">Lower endpoint of the numerical domain.</param>
    /// <param name="high">Upper endpoint of the numerical domain.</param>
    /// <param name="minsigma">The magic-clip floor every width is clipped up to; the bin width.</param>
    /// <param name="nBins">How many bins of width <paramref name="minsigma" /> cover the domain.</param>
    /// <param name="considerEndpoints">Whether the domain endpoints count as neighbors of the smallest and largest center.</param>
    /// <returns>One width per center, in the centers' order, or <see langword="null" />.</returns>
    /// <remarks>
    ///     <para>The idea is Rustuna's <c>binned_sigmas</c> (rustuna_core/src/parzen_estimator/model.rs): a width is the
    ///     larger distance to the two sorted neighbors, floored at <c>minsigma</c> by the clip, so a neighbor nearer than
    ///     <c>minsigma</c> cannot change it. Bin the centers at that width. A center that is neither the smallest nor the
    ///     largest of its bin has both sorted neighbors inside the bin, so its width clips to <c>minsigma</c>; only each
    ///     bin's extremes can have a farther neighbor, and that neighbor is the adjacent non-empty bin's extreme (or the
    ///     domain endpoint). Rustuna's version can differ from the sort in the last bits where rounding puts a center in
    ///     a neighboring bin; this one checks instead of trusting the binning.</para>
    ///     <para><b>Why the result is exact.</b> Bin assignment <c>floor((v - low) / minsigma)</c> is monotone in
    ///     <c>v</c> (IEEE subtraction, division by a positive constant and floor all are), so the bins are contiguous runs
    ///     of the sorted order, and equal centers share a bin. A bin's smallest center is taken at its first index and its
    ///     largest at its last index, which are the positions a stable sort puts first and last among equal values; the
    ///     neighbor distances are the same subtractions of the same operands the sort makes. Every bin is verified to span
    ///     at most <c>minsigma</c> (<c>max - min &lt;= minsigma</c>, computed); since IEEE subtraction is monotone in its
    ///     operands, every in-bin neighbor distance is then at most <c>minsigma</c> too, so leaving those distances out
    ///     changes no clipped width. Any bin wider than that, a center outside the domain or NaN, fewer than two
    ///     centers, or a degenerate domain returns <see langword="null" />. <c>TpeBinnedSigmasTests</c> compares the two
    ///     paths bit for bit over random, tied, clustered and bin-edge inputs.</para>
    /// </remarks>
    internal static double[]? BinnedRawSigmas(
        double[] mus, double low, double high, double minsigma, int nBins, bool considerEndpoints)
    {
        var sigmas = new double[mus.Length];
        return BinnedRawSigmas(mus, low, high, minsigma, nBins, considerEndpoints, sigmas) ? sigmas : null;
    }

    /// <summary>
    ///     <see cref="BinnedRawSigmas(double[], double, double, double, int, bool)" /> into a caller's buffer: true with
    ///     the widths written to <paramref name="sigmas" />, or false with <paramref name="sigmas" /> untouched when the
    ///     binned widths cannot be proven exact (the caller then sorts).
    /// </summary>
    /// <param name="mus">Kernel centers in transformed numerical coordinates; not modified.</param>
    /// <param name="low">Lower endpoint of the numerical domain.</param>
    /// <param name="high">Upper endpoint of the numerical domain.</param>
    /// <param name="minsigma">The magic-clip floor every width is clipped up to; the bin width.</param>
    /// <param name="nBins">How many bins of width <paramref name="minsigma" /> cover the domain.</param>
    /// <param name="considerEndpoints">Whether the domain endpoints count as neighbors of the smallest and largest center.</param>
    /// <param name="sigmas">Receives one width per center on success; at least as long as <paramref name="mus" />.</param>
    /// <returns>Whether the widths were written.</returns>
    /// <exception cref="IndexOutOfRangeException"><paramref name="sigmas" /> is shorter than <paramref name="mus" />.</exception>
    /// <remarks>Every check that can refuse runs before the first write, so a refusal leaves the buffer as it was.</remarks>
    internal static bool BinnedRawSigmas(
        ReadOnlySpan<double> mus, double low, double high, double minsigma, int nBins, bool considerEndpoints,
        Span<double> sigmas)
    {
        int n = mus.Length;
        if (n < 2 || nBins < 1 || nBins > 100 || !(high > low) || !(minsigma > 0) || !double.IsFinite(minsigma))
            return false;

        // Per bin: the smallest center and its first index, the largest and its last index (stable-sort positions).
        Span<double> binMin = stackalloc double[nBins];
        Span<double> binMax = stackalloc double[nBins];
        Span<int> binMinIndex = stackalloc int[nBins];
        Span<int> binMaxIndex = stackalloc int[nBins];
        binMin.Fill(double.PositiveInfinity);
        binMax.Fill(double.NegativeInfinity);
        binMinIndex.Fill(-1);
        binMaxIndex.Fill(-1);
        for (int i = 0; i < n; i++)
        {
            double v = mus[i];
            // Outside the domain (or NaN): the sort handles it; the bins assume every center lies in [low, high].
            if (!(v >= low && v <= high))
                return false;
            int bin = (int)Math.Min(Math.Floor((v - low) / minsigma), nBins - 1);
            if (v < binMin[bin])
            {
                binMin[bin] = v;
                binMinIndex[bin] = i;
            }
            if (v >= binMax[bin])
            {
                binMax[bin] = v;
                binMaxIndex[bin] = i;
            }
        }

        // The proof's condition: every in-bin neighbor distance is at most minsigma, which holds when each bin's span is.
        for (int b = 0; b < nBins; b++)
        {
            if (binMinIndex[b] >= 0 && !(binMax[b] - binMin[b] <= minsigma))
                return false;
        }

        // Every center not at its bin's extremes clips to minsigma; start everyone there and raise the extremes.
        sigmas.Slice(0, n).Fill(minsigma);

        // Each bin's smallest center: its sorted predecessor is the previous non-empty bin's largest (or `low`).
        bool firstBin = true;
        double previousMax = 0.0;
        for (int b = 0; b < nBins; b++)
        {
            int i = binMinIndex[b];
            if (i < 0)
                continue;
            if (!firstBin)
                sigmas[i] = Math.Max(sigmas[i], binMin[b] - previousMax);
            else if (considerEndpoints)
                sigmas[i] = Math.Max(sigmas[i], binMin[b] - low);
            firstBin = false;
            previousMax = binMax[b];
        }

        // Each bin's largest center: its sorted successor is the next non-empty bin's smallest (or `high`).
        bool lastBin = true;
        double nextMin = 0.0;
        for (int b = nBins - 1; b >= 0; b--)
        {
            int i = binMaxIndex[b];
            if (i < 0)
                continue;
            if (!lastBin)
                sigmas[i] = Math.Max(sigmas[i], nextMin - binMax[b]);
            else if (considerEndpoints)
                sigmas[i] = Math.Max(sigmas[i], high - binMax[b]);
            lastBin = false;
            nextMin = binMin[b];
        }
        return true;
    }

    /// <summary>
    ///     The stable argsort <see cref="ComputeSigmas(double[], double, double, int, ParzenEstimatorParameters)" /> uses: the permutation
    ///     <c>Enumerable.Range(0, n).OrderBy(i => values[i])</c> returns, with ties in index order, computed by
    ///     <see cref="Array.Sort{TKey, TValue}(TKey[], TValue[])" /> and an index sort of each run of equal keys.
    /// </summary>
    /// <param name="values">The values to order; not modified.</param>
    /// <returns>The indices of <paramref name="values" /> in ascending order of value, equal values by index.</returns>
    /// <remarks>
    ///     Why not the LINQ sort it replaces (Rustuna plan 13, item S6): it compares through a delegate and allocated
    ///     its key buffers per call, about 0.8 s of a 1,000-trial, 40-parameter TPE run. The result is the same
    ///     permutation: <see cref="Array.Sort{TKey, TValue}(TKey[], TValue[])" /> orders doubles as
    ///     <see cref="double.CompareTo(double)" /> does (NaN first, -0.0 equal to 0.0), which is what the LINQ sort's
    ///     default comparer used, and its only freedom, the order inside a run of equal keys, is undone by sorting each
    ///     run's indices.
    /// </remarks>
    internal static int[] StableArgsort(double[] values) => StableArgsort(values.AsSpan());

    /// <summary>
    ///     <see cref="StableArgsort(double[])" /> over a span (plan 15: the estimator's centers are a slice of its
    ///     prior-extended array): the same permutation, sorting a copy of the values as the array form sorted its clone.
    /// </summary>
    /// <param name="values">The values to order; not modified.</param>
    /// <returns>The indices of <paramref name="values" /> in ascending order of value, equal values by index.</returns>
    internal static int[] StableArgsort(ReadOnlySpan<double> values)
    {
        int n = values.Length;
        var keys = values.ToArray();
        var indices = new int[n];
        for (int i = 0; i < n; i++)
            indices[i] = i;
        Array.Sort(keys, indices);

        int runStart = 0;
        for (int i = 1; i <= n; i++)
        {
            // A run ends where the key changes by CompareTo, which groups NaNs together and -0.0 with 0.0.
            if (i == n || keys[i].CompareTo(keys[runStart]) != 0)
            {
                if (i - runStart > 1)
                    Array.Sort(indices, runStart, i - runStart);
                runStart = i;
            }
        }
        return indices;
    }

    /// <summary>Python <c>_ParzenEstimator._call_weights_func</c>.</summary>
    /// <param name="weightsFunc">Callback returning one nonnegative finite weight per observation.</param>
    /// <param name="n">Number of vector entries to produce.</param>
    /// <returns>Validated nonnegative finite weights aligned with observations.</returns>
    /// <exception cref="NumSharp.ValueError">Returned weights contain negative or non-finite values, or are all zero.</exception>
    /// <exception cref="System.Exception">The configured observation-weight callback fails; its exception propagates.</exception>
    private static double[] CallWeightsFunc(Func<int, double[]> weightsFunc, int n)
    {
        var wFull = weightsFunc(n);
        var w = wFull.Length > n ? wFull[..n] : wFull;

        // NOTE: the deprecated custom `weights` function is the only way to reach these guards
        // (the default `default_weights` never does); the array in the message is formatted
        // approximately rather than byte-for-byte with NumPy's ndarray repr.
        string ArrRepr(double[] a) => "[" + string.Join(" ", a.Select(PythonRepr.FloatRepr)) + "]";

        if (w.Any(x => x < 0))
            throw new ValueError(
                $"The `weights` function is not allowed to return negative values {ArrRepr(w)}. " +
                $"The argument of the `weights` function is {n}.");
        if (w.Length > 0 && MathUtil.PairwiseSum(w) <= 0)
            throw new ValueError(
                $"The `weight` function is not allowed to return all-zero values {ArrRepr(w)}." +
                $" The argument of the `weights` function is {n}.");
        if (!w.All(double.IsFinite))
            throw new ValueError(
                "The `weights`function is not allowed to return infinite or NaN values " +
                $"{ArrRepr(w)}. The argument of the `weights` function is {n}.");
        return w;
    }
}
