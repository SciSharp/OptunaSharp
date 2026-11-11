// Deterministic per-generation Gaussian sampling shared by the Natural-Evolution-Strategy optimizers.
//
// The Optuna ask/tell protocol proposes ONE candidate per trial, spreading a generation of `popsize`
// members across `popsize` consecutive trials (see `CmaEsSampler`). For the ES family each candidate is
// `mean + scale · εₖ`, and — like the CMA backend — the population is reconstructed at tell time from the
// stored trial parameters, never from persisted ε. That makes the whole generation's ε a pure function of
// (optimizer seed, generation): every trial in a generation re-derives the SAME ε matrix from a freshly
// seeded MT19937 stream and simply indexes its own row. This file centralizes that derivation so the three
// optimizers agree bit-for-bit and stay byte-reproducible against a numpy `RandomState` reference.

using NumSharp;
using OptunaSharp.Internal;

namespace OptunaSharp.Samplers.EvolutionStrategy;

/// <summary>
///     Deterministic Gaussian-noise generation for the ES optimizers: a stable (seed, generation) →
///     seed mixer and a row-major standard-normal matrix draw. Keeping both here guarantees the three
///     optimizers derive identical noise for identical state, which is what lets tell reconstruct ε from
///     the stored candidates instead of persisting it.
/// </summary>
internal static class EsSampling
{
    /// <summary>
    ///     Mixes the optimizer's base seed with the generation index into a well-scrambled 32-bit seed
    ///     (a SplitMix64 finalizer). Using a mixer rather than <c>seed + generation</c> avoids the MT19937
    ///     weakness where near-sequential seeds yield correlated early outputs — successive generations must
    ///     draw independent noise, not shifted copies of the same stream.
    /// </summary>
    /// <param name="seed">The optimizer's persisted base seed (fixed for the life of one optimizer).</param>
    /// <param name="generation">Zero-based generation counter; distinct generations get uncorrelated streams.</param>
    /// <returns>A 32-bit seed suitable for <c>RandomState.seed</c>.</returns>
    internal static uint Combine(uint seed, int generation)
    {
        // SplitMix64 avalanche: fold the two inputs into a 64-bit word, run the standard finalizer, and
        // truncate to 32 bits (RandomState.seed accepts a uint). The golden-ratio increment de-correlates
        // adjacent generations even when the base seed is small.
        ulong z = ((ulong)seed << 32) ^ (uint)generation;
        z += 0x9E3779B97F4A7C15UL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        z ^= z >> 31;
        return (uint)(z & 0xFFFFFFFFUL);
    }

    /// <summary>
    ///     Draws a <paramref name="rows" />×<paramref name="cols" /> matrix of i.i.d. standard normals from
    ///     <paramref name="rng" />, filled row-major (row 0 first). This matches numpy's
    ///     <c>RandomState.standard_normal((rows, cols))</c> C-order fill exactly, so the ES candidates are
    ///     byte-reproducible against a pinned numpy reference.
    /// </summary>
    /// <param name="rng">A NumPy-compatible MT19937 stream; the caller seeds it before this call.</param>
    /// <param name="rows">Number of rows (individuals) to draw; must be ≥ 0.</param>
    /// <param name="cols">Number of columns (dimensions) per row; must be ≥ 0.</param>
    /// <returns>A jagged <c>rows × cols</c> matrix of standard normals in row-major draw order.</returns>
    internal static double[][] StandardNormalMatrix(NumPyRandom rng, int rows, int cols)
    {
        // One flat draw preserves numpy's C-order fill (verified byte-identical to np.random.randn); the
        // reshape below is a pure view split, so element order is untouched.
        double[] flat = NumPyDraws.Randn(rng, rows * cols);
        var matrix = new double[rows][];
        for (int i = 0; i < rows; i++)
        {
            var row = new double[cols];
            for (int j = 0; j < cols; j++)
                row[j] = flat[i * cols + j];
            matrix[i] = row;
        }
        return matrix;
    }
}
