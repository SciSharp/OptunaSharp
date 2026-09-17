// The numeric core of REMBO — Random EMbedding Bayesian Optimization (Wang, Hutter, Zoghi, Matheson &
// de Freitas, JAIR 55 (2016) 361-387, doi:10.1613/jair.4806). This class owns the two purely geometric
// pieces of Algorithm 2: the random Gaussian embedding matrix A ∈ ℝ^{D×d} and the box-constrained
// projection g(y) = f(p_X(A·y)). It holds NO optimization state — the search itself is delegated to an
// inner Bayesian optimizer running in the low-dimensional latent box, driven by RandomEmbeddingSampler.
// REMBO is absent from Optuna, so — like TuRBO / LM-CMA / the NES family — this is a native
// convergence-validated port with no Optuna counterpart, on NumSharp's MT19937 (never System.Random).

using System;
using System.Collections.Concurrent;
using NumSharp;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Internal;

namespace OptunaSharp.Samplers.RandomEmbedding;

/// <summary>
///     The random-embedding geometry of REMBO (Wang et al. 2016): a fixed Gaussian matrix
///     <c>A ∈ ℝ^{D×d}</c> mapping a low-dimensional latent point <c>y</c> in the box
///     <c>Y = [−√d, √d]^d</c> onto the real parameters, with the paper's convex box projection
///     <c>p_X</c> applied before evaluation. It is the piece that lets Bayesian optimization run in
///     <c>d</c> dimensions while the study's real space has (possibly vastly) more — provided the
///     objective has low <b>effective</b> dimensionality (Definition 1 / Theorem 2).
///
///     <para><b>Why the rows are seeded per name, not stored.</b> The paper draws one matrix <c>A</c>
///     with i.i.d. <c>N(0,1)</c> entries up front. In a define-by-run library the real parameters are
///     discovered incrementally and their count <c>D</c> is not known when the sampler is constructed,
///     so instead of persisting a growing matrix this class generates each real parameter's row on
///     demand from a per-(run, name) seeded MT19937 stream. Each entry is still an independent
///     <c>N(0,1)</c> draw, so the resulting matrix is distributionally identical to the paper's — but
///     it is a pure function of <c>(base seed, run, parameter name)</c>, hence stable across parameter
///     discovery order, study resumption, and parallel workers, with no shared mutable matrix to guard.</para>
///
///     <para><b>Interleaved runs.</b> Section 3.1 raises REMBO's success rate by running several
///     independent embeddings (the box may miss the optimum with probability <c>δ</c> per embedding, so
///     <c>k</c> interleaved embeddings miss it with probability <c>δ^k</c>). Each run index selects a
///     different embedding via <see cref="RunForTrial" /> and its own family of projection rows; the
///     sampler assigns trials to runs round-robin.</para>
/// </summary>
internal sealed class RandomEmbeddingProjection
{
    // SplitMix64 constants — a strong finalizer that decorrelates the (baseSeed, run, nameHash) inputs
    // before they seed an MT19937 stream, so adjacent parameter names do not produce correlated rows
    // (raw MT19937 seeding is sensitive to near-identical seeds).
    private const ulong SplitMixGamma = 0x9E3779B97F4A7C15UL;
    private const ulong SplitMixMul1 = 0xBF58476D1CE4E5B9UL;
    private const ulong SplitMixMul2 = 0x94D049BB133111EBUL;
    private const uint RunSalt = 0x9E3779B1u;   // odd multiplier that spreads the (small) run index across the word

    // FNV-1a 64-bit constants — a deterministic string hash (String.GetHashCode is randomized per process
    // in .NET Core and would break cross-process/seed reproducibility of the embedding).
    private const ulong FnvOffset = 14695981039346656037UL;
    private const ulong FnvPrime = 1099511628211UL;

    private readonly uint _baseSeed;
    private readonly int _dim;            // d — the internal embedding dimension
    private readonly int _nEmbeddings;    // k — the number of interleaved independent embeddings
    private readonly double _boxHalfWidth; // √d — the half-width of the latent box Y = [−√d, √d]^d

    // Projection rows are immutable once drawn; caching keeps a run's matrix stable and avoids re-drawing
    // the same row on every trial. Keyed by (run, real parameter name).
    private readonly ConcurrentDictionary<(int Run, string Name), double[]> _rows = new();

    /// <summary>The internal embedding dimension <c>d</c> the inner optimizer actually searches over.</summary>
    internal int Dim => _dim;

    /// <summary>The number of interleaved independent embeddings <c>k</c> (Section 3.1); 1 is a single embedding.</summary>
    internal int EmbeddingCount => _nEmbeddings;

    /// <summary>The half-width <c>√d</c> of the latent box <c>Y = [−√d, √d]^d</c> the paper searches in.</summary>
    internal double BoxHalfWidth => _boxHalfWidth;

    /// <summary>Builds the embedding geometry; the matrix rows are materialized lazily and deterministically.</summary>
    /// <param name="baseSeed">Seed anchoring every projection row; a fixed value reproduces the whole embedding.</param>
    /// <param name="dim">The internal embedding dimension <c>d</c> (should be ≥ the guessed effective dimensionality).</param>
    /// <param name="nEmbeddings">The number of interleaved independent embeddings <c>k</c> (Section 3.1).</param>
    /// <exception cref="System.ArgumentException"><paramref name="dim" /> or <paramref name="nEmbeddings" /> is below one.</exception>
    internal RandomEmbeddingProjection(uint baseSeed, int dim, int nEmbeddings)
    {
        if (dim < 1)
            throw new ArgumentException("The embedding dimension must be at least 1.", nameof(dim));
        if (nEmbeddings < 1)
            throw new ArgumentException("The number of embeddings must be at least 1.", nameof(nEmbeddings));

        _baseSeed = baseSeed;
        _dim = dim;
        _nEmbeddings = nEmbeddings;
        _boxHalfWidth = Math.Sqrt(dim);
    }

    /// <summary>The latent parameter name for coordinate <paramref name="i" /> in the shadow study.</summary>
    /// <param name="i">The zero-based latent coordinate index.</param>
    /// <returns>A stable name (<c>z0</c>, <c>z1</c>, …) identifying the coordinate in the low-dimensional space.</returns>
    internal static string LatentName(int i) => $"z{i}";

    /// <summary>
    ///     The latent search space the inner optimizer explores: <c>d</c> continuous coordinates, each on
    ///     <c>[−√d, √d]</c> — the box <c>Y</c> the paper derives (Theorem 3, with the practical choice
    ///     <c>ε = log(d)/√d</c>).
    /// </summary>
    /// <returns>An ordered map of the <c>d</c> latent coordinate distributions in coordinate order.</returns>
    internal OrderedDict<string, BaseDistribution> LatentSearchSpace()
    {
        var space = new OrderedDict<string, BaseDistribution>(_dim);
        for (int i = 0; i < _dim; i++)
            space[LatentName(i)] = new FloatDistribution(-_boxHalfWidth, _boxHalfWidth);
        return space;
    }

    /// <summary>Maps a trial to one of the <c>k</c> interleaved embeddings (round-robin by trial number).</summary>
    /// <param name="trialNumber">The trial's zero-based number within the study.</param>
    /// <returns>The run index in <c>[0, k)</c> selecting the embedding this trial uses.</returns>
    internal int RunForTrial(int trialNumber)
    {
        if (_nEmbeddings == 1)
            return 0;
        // Guard against a negative number defensively (trial numbers are non-negative in practice).
        int r = trialNumber % _nEmbeddings;
        return r < 0 ? r + _nEmbeddings : r;
    }

    /// <summary>
    ///     The projection row <c>A_name ∈ ℝ^d</c> (a row of the embedding matrix) for one real parameter
    ///     under one embedding — <c>d</c> i.i.d. <c>N(0,1)</c> values, drawn once and cached.
    /// </summary>
    /// <param name="run">The embedding (run) index selecting which independent matrix to use.</param>
    /// <param name="name">The real parameter's name; its hash anchors this row's random stream.</param>
    /// <returns>The cached <c>d</c>-vector row; callers must not mutate it (the cache shares the instance).</returns>
    internal double[] ProjectionRow(int run, string name)
        => _rows.GetOrAdd((run, name), key =>
        {
            // A fresh MT19937 seeded from (baseSeed, run, name) makes this row a pure, reproducible
            // function of its identity — independent of when or on which worker it is first requested.
            var rng = np.random.RandomState();
            rng.seed(RowSeed(key.Run, key.Name));
            return rng.randn(_dim).ToArray<double>();
        });

    /// <summary>
    ///     Maps a latent point to one real parameter's external value: the paper's
    ///     <c>x = p_X(A·y)</c> restricted to this coordinate, then de-normalized through the parameter's
    ///     own distribution. Concretely it forms <c>raw = A_name · y</c>, applies the convex box
    ///     projection <c>p_X</c> (per-coordinate clip into the rescaled box <c>[−1, 1]</c>), and maps the
    ///     result back onto <c>[low, high]</c> honoring log scale, discrete step, and integer rounding.
    /// </summary>
    /// <param name="run">The embedding (run) index whose matrix row projects this parameter.</param>
    /// <param name="name">The real parameter's name (selects its projection row).</param>
    /// <param name="distribution">The parameter's distribution, supplying its bounds and scale.</param>
    /// <param name="latentY">The latent point <c>y ∈ Y</c>; its length must equal <see cref="Dim" />.</param>
    /// <returns>The external parameter value (a <see cref="double" /> for float parameters, a boxed <see cref="long" /> for integers).</returns>
    /// <exception cref="System.ArgumentException"><paramref name="latentY" /> is not <see cref="Dim" />-dimensional.</exception>
    internal object? ToExternal(int run, string name, BaseDistribution distribution, double[] latentY)
    {
        if (latentY.Length != _dim)
            throw new ArgumentException($"Latent point must have length {_dim}, but got {latentY.Length}.", nameof(latentY));

        double[] row = ProjectionRow(run, name);

        // raw = A_name · y  (one coordinate of A·y).
        double raw = 0.0;
        for (int j = 0; j < _dim; j++)
            raw += row[j] * latentY[j];

        // p_X: the box constraint X is the real box rescaled to [−1, 1]; the paper's projection
        // p_X(z) = argmin_{u∈X} ‖u − z‖ is per-coordinate clipping for a box. Rescale the clipped value
        // to the unit interval [0, 1] that SearchSpaceTransform consumes.
        double clipped = raw < -1.0 ? -1.0 : raw > 1.0 ? 1.0 : raw;
        double unit = (clipped + 1.0) * 0.5;

        // Reuse Optuna's canonical [0,1] → external de-normalization so log/step/int semantics match the
        // rest of the library exactly (this is what RandomSampler/CMA-ES use).
        var oneParam = new OrderedDict<string, BaseDistribution> { [name] = distribution };
        var transform = new SearchSpaceTransform(oneParam, transform01: true);
        var external = transform.Untransform(np.array(new[] { unit }));
        return external[name];
    }

    /// <summary>Folds <c>(base seed, run, name hash)</c> into a decorrelated 32-bit seed for one row's MT19937 stream.</summary>
    /// <param name="run">The embedding (run) index.</param>
    /// <param name="name">The real parameter name whose row is being seeded.</param>
    /// <returns>A 32-bit seed suitable for <c>RandomState.seed</c>.</returns>
    private uint RowSeed(int run, string name)
    {
        // Combine the three inputs, then run a SplitMix64 finalizer so even adjacent (run, name) pairs
        // map to well-separated MT19937 seeds — avoiding the correlation MT19937 shows for close seeds.
        ulong z = ((ulong)_baseSeed << 32) ^ ((ulong)(uint)run * RunSalt) ^ Fnv1a(name);
        z += SplitMixGamma;
        z = (z ^ (z >> 30)) * SplitMixMul1;
        z = (z ^ (z >> 27)) * SplitMixMul2;
        z ^= z >> 31;
        return (uint)(z & 0xFFFFFFFFUL);
    }

    /// <summary>Deterministic FNV-1a 64-bit hash of a parameter name (both bytes of each UTF-16 unit mixed).</summary>
    /// <param name="s">The parameter name to hash.</param>
    /// <returns>The 64-bit hash used to anchor the parameter's projection-row seed.</returns>
    private static ulong Fnv1a(string s)
    {
        ulong hash = FnvOffset;
        foreach (char c in s)
        {
            hash ^= (byte)(c & 0xFF);
            hash *= FnvPrime;
            hash ^= (byte)((c >> 8) & 0xFF);
            hash *= FnvPrime;
        }
        return hash;
    }
}
