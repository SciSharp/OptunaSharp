// The high-dimensional-kernel inner optimizer for REMBO (Wang et al. 2016, §3.2). Where the
// low-dimensional kernel fits the inner GP on the latent y directly (and can waste effort on the large
// region of Y that projects onto the box boundary), the high-dimensional kernel measures GP similarity in
// the *projected* space p_X(A·y) — i.e. the real parameter space. Because p_X(A·y) is exactly the real
// parameter vector REMBO evaluates, this is simply a standard GP over the real (embedded) parameters whose
// acquisition is optimized over the reachable subspace {p_X(A·y) : y ∈ Y} by scoring projected candidate
// y's. It reuses the native GP stack unchanged (GPRegressor + Acqf.LogEi + GpSearchSpace), exactly the way
// TurboOptimizer does — no GP code is modified.

using System;
using System.Collections.Generic;
using NumSharp;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Internal;
using OptunaSharp.Samplers.ModelBased.Gp;

namespace OptunaSharp.Samplers.RandomEmbedding;

/// <summary>
///     REMBO's high-dimensional-kernel proposal step (§3.2): a Matérn-5/2 ARD GP fit on the real
///     (embedded) parameters of a run's completed trials, with the expected-improvement acquisition
///     maximized over the y-subspace by projecting a scrambled-Sobol candidate set of latent points
///     <c>y ∈ Y</c> through <c>p_X(A·y)</c> and scoring each. This is the variant the paper recommends when
///     the low-dimensional kernel over-explores the clipped boundary of the embedding box, and — because
///     the GP lives in the real parameter space — it also carries integer, log, step, and (via the GP's
///     categorical equality kernel) categorical parameters, i.e. the §3.2 categorical/Hamming kernel.
///
///     <para>It holds no state between calls: like <see cref="TurboOptimizer" />, each proposal refits the
///     GP from the supplied history, so a study resumes from storage with nothing serialized.</para>
/// </summary>
internal sealed class RandomEmbeddingGpOptimizer
{
    private const double Eps = 1e-10;

    private readonly RandomEmbeddingProjection _embedding;
    private readonly int _nStartupTrials;
    private readonly int? _numCandidates;
    private readonly double _minimumNoise;
    private readonly bool _deterministicObjective;

    /// <summary>Configures the high-dimensional-kernel GP proposer.</summary>
    /// <param name="embedding">The shared random embedding supplying the box <c>Y</c> and the projection.</param>
    /// <param name="nStartupTrials">Completed trials of a run required before the GP takes over from uniform
    ///     random sampling in <c>Y</c>.</param>
    /// <param name="numCandidates">Projected candidate points scored per proposal; <c>null</c> uses
    ///     <c>min(5000, max(1000, 200·d))</c> — cheap because scoring is only a GP posterior evaluation.</param>
    /// <param name="minimumNoise">Lower floor on the fitted GP observation-noise variance.</param>
    /// <param name="deterministicObjective">Treat the objective as noise-free (fix the GP noise at the floor).</param>
    internal RandomEmbeddingGpOptimizer(
        RandomEmbeddingProjection embedding, int nStartupTrials, int? numCandidates,
        double minimumNoise, bool deterministicObjective)
    {
        _embedding = embedding;
        _nStartupTrials = nStartupTrials;
        _numCandidates = numCandidates;
        _minimumNoise = minimumNoise;
        _deterministicObjective = deterministicObjective;
    }

    /// <summary>
    ///     Proposes the next latent point for one embedding. During startup it draws uniformly from the box
    ///     <c>Y = [−√d, √d]^d</c>; afterwards it fits the GP on the run's real parameter history and returns
    ///     the candidate <c>y</c> whose projection maximizes expected improvement.
    /// </summary>
    /// <param name="run">The embedding (run) index whose matrix projects the parameters.</param>
    /// <param name="embeddedSpace">The embedded real search space (float / integer / categorical); may be
    ///     empty during startup, when it is not consulted.</param>
    /// <param name="trainParams">The run's completed trials' external parameters, restricted to
    ///     <paramref name="embeddedSpace" />, one dictionary per training row.</param>
    /// <param name="trainMaxValues">The matching maximization-oriented objective values (already sign-flipped
    ///     for a minimizing study), one per training row.</param>
    /// <param name="rng">The per-trial random stream driving the candidate Sobol seed.</param>
    /// <returns>The proposed latent point <c>y ∈ Y</c> (length <see cref="RandomEmbeddingProjection.Dim" />).</returns>
    internal double[] Propose(
        int run,
        OrderedDict<string, BaseDistribution> embeddedSpace,
        IReadOnlyList<IReadOnlyDictionary<string, object?>> trainParams,
        double[] trainMaxValues,
        NumPyRandom rng)
    {
        int d = _embedding.Dim;

        // Startup (or nothing to model yet): cover the box uniformly so the GP has spread-out data to fit.
        if (trainParams.Count < _nStartupTrials || embeddedSpace.Count == 0)
            return RandomY(d, rng);

        var internalSpace = new GpSearchSpace(embeddedSpace);
        double[,] normalizedParams = internalSpace.GetNormalizedParams(trainParams);

        // Standardize the maximization-oriented values; the incumbent f0 is the best standardized value
        // (expected improvement measures improvement over it).
        double mean = MathUtil.Mean(trainMaxValues);
        double std = Math.Max(Eps, MathUtil.Std(trainMaxValues));
        int m = trainMaxValues.Length;
        var yStd = new double[m];
        double f0 = double.NegativeInfinity;
        for (int i = 0; i < m; i++)
        {
            yStd[i] = (trainMaxValues[i] - mean) / std;
            if (yStd[i] > f0) f0 = yStd[i];
        }

        GPRegressor gpr = GPRegressor.FitKernelParams(
            normalizedParams, yStd, internalSpace.IsCategorical, _minimumNoise, _deterministicObjective);

        int n = _numCandidates ?? Math.Min(5000, Math.Max(1000, 200 * d));
        double[,] candidatesY = SampleBox(d, n, rng);

        // Project each candidate y to its real (embedded) parameters and normalize into the GP's space —
        // this is the "search over {p_X(Ay)}" the paper describes, done with the same de-normalization the
        // GP was trained on so distances are measured in the real parameter space.
        var candidateExternal = new List<IReadOnlyDictionary<string, object?>>(n);
        var yc = new double[d];
        foreach (int i in Range(n))
        {
            for (int j = 0; j < d; j++) yc[j] = candidatesY[i, j];
            var external = new Dictionary<string, object?>(embeddedSpace.Count);
            foreach (var (name, dist) in embeddedSpace)
                external[name] = _embedding.ToExternal(run, name, dist, yc);
            candidateExternal.Add(external);
        }
        double[,] candidateNorm = internalSpace.GetNormalizedParams(candidateExternal);

        (double[] postMean, double[] postVar) = gpr.Posterior(candidateNorm);

        // argmax expected improvement (log-EI is monotone in EI). Non-finite scores (a candidate coinciding
        // with a training point can drive the posterior variance to zero) are skipped, never chosen.
        int best = 0;
        double bestAcq = double.NegativeInfinity;
        for (int i = 0; i < n; i++)
        {
            double acq = Acqf.LogEi(postMean[i], postVar[i], f0);
            if (double.IsFinite(acq) && acq > bestAcq)
            {
                bestAcq = acq;
                best = i;
            }
        }

        var result = new double[d];
        for (int j = 0; j < d; j++) result[j] = candidatesY[best, j];
        return result;
    }

    /// <summary>Draws one latent point uniformly from the box <c>Y = [−√d, √d]^d</c>.</summary>
    /// <param name="d">The embedding dimension.</param>
    /// <param name="rng">The random stream to draw from.</param>
    /// <returns>A uniform latent point in the box.</returns>
    private double[] RandomY(int d, NumPyRandom rng)
    {
        double h = _embedding.BoxHalfWidth;
        return rng.uniform(-h, h, new Shape(d)).ToArray<double>();
    }

    /// <summary>Generates a scrambled-Sobol candidate set in the box <c>Y = [−√d, √d]^d</c>.</summary>
    /// <param name="d">The embedding dimension.</param>
    /// <param name="n">The number of candidate points.</param>
    /// <param name="rng">The random stream supplying the Sobol seed.</param>
    /// <returns>An <c>(n, d)</c> matrix of candidate latent points spanning the box.</returns>
    private double[,] SampleBox(int d, int n, NumPyRandom rng)
    {
        // Mask to 31 bits: numpy's SeedSequence (behind scrambled Sobol) rejects negative entropy.
        int seed = (int)(rng.randint(0, int.MaxValue) & 0x7FFFFFFF);
        double[,] u = ScrambledSobol.Random(d, n, seed);
        double h = _embedding.BoxHalfWidth;
        for (int i = 0; i < n; i++)
            for (int j = 0; j < d; j++)
                u[i, j] = (2.0 * u[i, j] - 1.0) * h;   // [0,1] → [−√d, √d]
        return u;
    }

    /// <summary>Yields <c>0 … n−1</c> — a small readability helper for the candidate loop.</summary>
    /// <param name="n">The exclusive upper bound.</param>
    /// <returns>The integer sequence from zero to <paramref name="n" /> exclusive.</returns>
    private static IEnumerable<int> Range(int n)
    {
        for (int i = 0; i < n; i++) yield return i;
    }
}
