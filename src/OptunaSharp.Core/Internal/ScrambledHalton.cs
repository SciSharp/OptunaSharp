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
///     Like <see cref="ScrambledSobol" /> the sequence is a pure function of <c>(dim, seed)</c>: <see cref="Create" />
///     draws the permutations once into an immutable <see cref="Engine" />, which QMCSampler keeps across trials
///     and which parallel trials can share without a lock (the <c>sample_id</c> counter is already serialized
///     through the study system attr).</para>
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
    internal static double[] Point(int dim, ulong n, int seed) => Create(dim, seed).Point(n);

    /// <summary>
    ///     Builds the scrambled engine of <c>Halton(dim, scramble=True, seed)</c>: every dimension's Owen digit
    ///     permutations, drawn from one shared <c>default_rng(seed)</c> stream in dimension order as scipy draws them.
    /// </summary>
    /// <param name="dim">The number of coordinates in each point.</param>
    /// <param name="seed">The seed controlling the reproducible random stream.</param>
    /// <returns>An immutable engine that can be shared across threads and reused for any point index.</returns>
    /// <exception cref="NumSharp.ValueError">The dimension is outside the supported prime-base table.</exception>
    /// <remarks>
    ///     Drawing the permutations is nearly all the cost of a point (<c>Σ_d count_d</c> shuffles of <c>[0, b_d)</c>,
    ///     each through a NumSharp array), so a caller that needs several points of one sequence — QMCSampler asks for
    ///     one per trial — keeps the engine. The shuffle arrays are disposed as soon as they are copied out, so none is
    ///     left for the finalizer.
    /// </remarks>
    internal static Engine Create(int dim, int seed)
    {
        int[] primes = QuasiMonteCarlo.HaltonPrimes(dim);
        Generator rng = np.random.default_rng(seed);

        var permutations = new long[dim][][];
        for (int d = 0; d < dim; d++)
        {
            int b = primes[d];
            // scipy _van_der_corput_permutations: `count` shuffles of arange(b) on the shared stream.
            // count = ceil(54 / log2(b)) - 1 (the number of base-b digits below 2**-54).
            int count = (int)Math.Ceiling(54.0 / Math.Log2(b)) - 1;
            var perms = new long[count][];
            for (int j = 0; j < count; j++)
            {
                using NDArray row = np.arange((long)b);  // int64 [0, b), the row `np.repeat(arange(b))` shuffles
                rng.shuffle(row);                         // in-place Fisher–Yates (random_interval), byte-exact
                perms[j] = row.ToArray<long>();
            }
            permutations[d] = perms;
        }
        return new Engine(primes, permutations);
    }

    /// <summary>
    ///     A scrambled Halton sequence for one <c>(dim, seed)</c>: the bases and every dimension's digit permutations,
    ///     from which any point is computed directly.
    /// </summary>
    /// <remarks>
    ///     Immutable after construction, so one instance can serve concurrent callers. The digit expansion in
    ///     <see cref="Point" /> is the same double arithmetic, in the same order, as the per-call form it replaces,
    ///     so the points are bit-identical.
    /// </remarks>
    internal sealed class Engine
    {
        private readonly int[] _bases;
        private readonly long[][][] _permutations;

        /// <summary>Wraps finished permutations; the arrays are owned by the engine from here on.</summary>
        /// <param name="bases">The prime base of each dimension.</param>
        /// <param name="permutations">Each dimension's digit permutations, <c>[dim][count][base]</c>.</param>
        internal Engine(int[] bases, long[][][] permutations)
        {
            _bases = bases;
            _permutations = permutations;
        }

        /// <summary>The number of dimensions every point of this sequence has.</summary>
        internal int Dim => _bases.Length;

        /// <summary>The scrambled point at sequence index <paramref name="n" /> — <c>fast_forward(n)</c> then <c>random(1)</c>.</summary>
        /// <param name="n">The zero-based sequence index of the scrambled Halton point.</param>
        /// <returns>A new array of <see cref="Dim" /> coordinates the caller owns.</returns>
        internal double[] Point(ulong n)
        {
            var point = new double[_bases.Length];
            for (int d = 0; d < _bases.Length; d++)
            {
                int b = _bases[d];
                long[][] perms = _permutations[d];

                // scipy _cy_van_der_corput_scrambled at start_index=n: EXACTLY `count` permuted digits,
                // so trailing zero digits are permuted too (P[j][0] may be nonzero) — never stop at
                // quotient==0. Accumulated in scipy's order (add, then advance b2r and quotient).
                double r = 0.0;
                double b2r = 1.0 / b;
                ulong quotient = n;
                for (int j = 0; j < perms.Length; j++)
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
}
