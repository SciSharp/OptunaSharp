using System;
using NumSharp;

namespace OptunaSharp.Internal;

/// <summary>
///     Scrambled Halton sequence — port of <c>scipy.stats.qmc.Halton(scramble=True)</c>
///     (<c>scipy/stats/_qmc.py</c> + <c>_qmc_cy.pyx</c>), the last remaining scrambling combination of
///     <see cref="Samplers.QMCSampler" /> (PORTING_PLAN Phase 2.12 — the final native sliver). scipy
///     scrambles each per-dimension van der Corput sequence with <b>Owen's digit permutations</b>: for a
///     base <c>b</c> it draws <c>ceil(54/log2(b)) - 1</c> permutations of <c>[0, b)</c> from
///     <c>numpy.random.default_rng(seed)</c> (<c>_van_der_corput_permutations</c>, one
///     <c>rng.shuffle(arange(b))</c> per permutation) and, at each radical-inverse digit, replaces the
///     digit by its permuted value (<c>_cy_van_der_corput_scrambled</c>).
///
///     <para>This is bit-exact because NumSharp's <c>np.random.default_rng</c> reproduces NumPy's PCG64
///     stream byte-for-byte and <c>Generator.shuffle</c> ports NumPy's exact Fisher–Yates over
///     <c>random_interval</c> (mask-rejection through the buffered 32-bit words) — the primitive scipy's
///     shuffle consumes — while the digit expansion is deterministic double arithmetic in scipy's order.
///     Like <see cref="ScrambledSobol" /> the engine is reconstructed per call from <c>(dim, seed)</c>, so
///     it holds no state and parallel trials need no lock (the <c>sample_id</c> counter is already
///     serialized through the study system attr).</para>
/// </summary>
internal static class ScrambledHalton
{
    /// <summary>
    ///     The scrambled Halton point at sequence index <paramref name="n" /> in <paramref name="dim" />
    ///     dimensions, seeded by <paramref name="seed" /> —
    ///     <c>Halton(dim, seed, scramble=True).fast_forward(n).random(1)</c>. The per-dimension
    ///     permutations are drawn from one shared <c>default_rng(seed)</c> stream in dimension order, so a
    ///     dimension's coordinate depends on the earlier dimensions' permutation draws exactly as scipy's
    ///     single <c>self.rng</c> makes it (verified bit-exact vs scipy).
    /// </summary>
    /// <param name="dim">The number of coordinates in each point.</param>
    /// <param name="n">The zero-based sequence index of the scrambled Halton point.</param>
    /// <param name="seed">The seed controlling the reproducible random stream.</param>
    /// <returns>The requested reproducibly scrambled Halton point.</returns>
    internal static double[] Point(int dim, ulong n, int seed)
    {
        int[] primes = QuasiMonteCarlo.HaltonPrimes(dim);
        Generator rng = np.random.default_rng(seed);

        var point = new double[dim];
        for (int d = 0; d < dim; d++)
        {
            int b = primes[d];
            // scipy _van_der_corput_permutations: `count` shuffles of arange(b) on the shared stream.
            // count = ceil(54 / log2(b)) - 1 (the number of base-b digits below 2**-54).
            int count = (int)Math.Ceiling(54.0 / Math.Log2(b)) - 1;
            var perms = new long[count][];
            for (int j = 0; j < count; j++)
            {
                NDArray row = np.arange((long)b);  // int64 [0, b), the row `np.repeat(arange(b))` shuffles
                rng.shuffle(row);                   // in-place Fisher–Yates (random_interval), byte-exact
                perms[j] = row.ToArray<long>();
            }

            // scipy _cy_van_der_corput_scrambled at start_index=n: EXACTLY `count` permuted digits,
            // so trailing zero digits are permuted too (P[j][0] may be nonzero) — never stop at
            // quotient==0. Accumulated in scipy's order (add, then advance b2r and quotient).
            double r = 0.0;
            double b2r = 1.0 / b;
            ulong quotient = n;
            for (int j = 0; j < count; j++)
            {
                int remainder = (int)(quotient % (ulong)b);
                remainder = (int)perms[j][remainder];
                r += remainder * b2r;
                b2r /= b;
                quotient /= (ulong)b;
            }
            point[d] = r;
        }
        return point;
    }
}
