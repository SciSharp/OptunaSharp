using System;
using NumSharp;

namespace OptunaSharp.Internal;

/// <summary>
///     Scrambled Sobol' sequence — port of <c>scipy.stats.qmc.Sobol(scramble=True)</c>
///     (<c>scipy/stats/_sobol.pyx</c> + <c>_qmc.py</c>), the QMC engine the GP sampler's acqf optimizer
///     draws its preliminary points from (PORTING_PLAN Phase 3.2 part 3c — closing the 2.12 scrambling
///     gap). scipy scrambles the direction numbers with a linear matrix scramble (LMS) plus a digital
///     shift, both seeded by <c>numpy.random.default_rng(seed)</c> (PCG64). This is bit-exact because
///     NumSharp's <c>np.random.default_rng</c>/<c>Generator.integers</c> reproduce NumPy's PCG64 stream
///     byte-for-byte and the scramble + Gray-code generation are pure integer bit manipulation.
/// </summary>
internal static class ScrambledSobol
{
    /// <summary>
    ///     The first <paramref name="n" /> points of the scrambled Sobol' sequence in <paramref name="d" />
    ///     dimensions, seeded by <paramref name="seed" /> — <c>Sobol(d, scramble=True, seed).random(n)</c>.
    /// </summary>
    /// <param name="d">The number of Sobol dimensions.</param>
    /// <param name="n">The number of elements or points to process.</param>
    /// <param name="seed">The seed controlling the reproducible random stream.</param>
    /// <returns>The scrambled Sobol sample matrix with rows for points and columns for dimensions.</returns>
    internal static double[,] Random(int d, int n, int seed) => Create(d, seed).Generate(n);

    /// <summary>
    ///     Builds the scrambled engine of <c>Sobol(d, scramble=True, seed)</c>: draws the digital shift and the
    ///     linear-matrix-scramble cube from <c>default_rng(seed)</c> in scipy's order and scrambles the direction
    ///     numbers. Every point of the sequence is then a pure function of this state.
    /// </summary>
    /// <param name="d">The number of Sobol dimensions.</param>
    /// <param name="seed">The seed controlling the reproducible random stream.</param>
    /// <returns>An immutable engine that can be shared across threads and reused for any point index.</returns>
    /// <exception cref="NumSharp.ValueError">The dimension is outside the supported direction-number table.</exception>
    /// <remarks>
    ///     Building is the expensive part — <c>d·bits²</c> rng draws and the <c>O(d·bits³)</c> scramble — so a caller
    ///     that needs several points of the same <c>(d, seed)</c> sequence (QMCSampler asks for one point per trial)
    ///     keeps the engine instead of rebuilding it for every point.
    /// </remarks>
    internal static Engine Create(int d, int seed)
    {
        int bits = QuasiMonteCarlo.SobolBits;  // 30
        uint[][] sv = QuasiMonteCarlo.SobolSv(d);
        Generator rng = np.random.default_rng(seed);

        // _scramble(): the digital shift then the lower-triangular LMS matrices, in this rng order.
        // shift[i] = Σ_j M[i,j]·2^j  with M = integers(2, size=(d, bits)).
        long[] shiftBits;
        using (NDArray drawn = rng.integers(2, null, new Shape(d, bits)))
            shiftBits = drawn.ToArray<long>();
        var shift = new uint[d];
        for (int i = 0; i < d; i++)
        {
            uint s = 0;
            for (int j = 0; j < bits; j++)
                if (shiftBits[i * bits + j] != 0)
                    s += 1u << j;
            shift[i] = s;
        }

        // ltm = tril(integers(2, size=(d, bits, bits))) — the full cube is drawn (upper entries too),
        // then masked to the lower triangle; _cscramble forces the diagonal to 1.
        long[] ltm;
        using (NDArray drawn = rng.integers(2, null, new Shape(d, bits, bits)))
            ltm = drawn.ToArray<long>();
        Cscramble(d, bits, ltm, sv);
        return new Engine(d, bits, shift, sv);
    }

    /// <summary>
    ///     A scrambled Sobol' sequence for one <c>(d, seed)</c>: the digital shift and the scrambled direction
    ///     numbers, from which any point is computed without generating the points before it.
    /// </summary>
    /// <remarks>
    ///     Immutable after construction, so one instance can serve concurrent callers; nothing it returns aliases its
    ///     state. Point <c>k</c> of the Gray-code construction is <c>shift XOR</c> the direction numbers of the bits set
    ///     in <c>gray(k) = k ^ (k &gt;&gt; 1)</c>, because step <c>k</c> XORs in the direction number of the lowest set bit
    ///     of <c>k</c>, which is exactly the bit by which <c>gray(k)</c> differs from <c>gray(k - 1)</c>. The integer
    ///     state is therefore the same as after stepping, and the conversion to double is the same multiplication, so
    ///     <see cref="Point" /> and <see cref="Generate" /> agree bit for bit.
    /// </remarks>
    internal sealed class Engine
    {
        private readonly int _bits;
        private readonly uint[] _shift;
        private readonly uint[][] _sv;
        private readonly double _scale;

        /// <summary>Wraps a finished scramble; the arrays are owned by the engine from here on.</summary>
        /// <param name="dim">The number of Sobol dimensions.</param>
        /// <param name="bits">The number of direction-number bits (scipy's 30).</param>
        /// <param name="shift">The digital shift, one integer per dimension; point 0 of the sequence.</param>
        /// <param name="sv">The scrambled direction numbers, <c>[dim][bits]</c>.</param>
        internal Engine(int dim, int bits, uint[] shift, uint[][] sv)
        {
            Dim = dim;
            _bits = bits;
            _shift = shift;
            _sv = sv;
            _scale = 1.0 / Math.Pow(2.0, bits);
        }

        /// <summary>The number of dimensions every point of this sequence has.</summary>
        internal int Dim { get; }

        /// <summary>
        ///     The first <paramref name="n" /> points — <c>random(n)</c> on a fresh engine: point 0 is the shift and
        ///     each later point XORs one direction number into the previous one (Gray code).
        /// </summary>
        /// <param name="n">The number of points to produce.</param>
        /// <returns>A new <c>[n, Dim]</c> matrix the caller owns.</returns>
        internal double[,] Generate(int n)
        {
            int d = Dim;
            var sample = new double[n, d];
            var quasi = (uint[])_shift.Clone();
            for (int j = 0; j < d; j++)
                sample[0, j] = quasi[j] * _scale;
            for (int k = 1; k < n; k++)
            {
                int l = LowZeroBit((uint)(k - 1));
                for (int j = 0; j < d; j++)
                {
                    quasi[j] ^= _sv[j][l - 1];
                    sample[k, j] = quasi[j] * _scale;
                }
            }
            return sample;
        }

        /// <summary>
        ///     The point at <paramref name="index" /> — <c>fast_forward(index)</c> then <c>random(1)</c> — computed
        ///     directly from <c>gray(index)</c> in <c>O(Dim·bits)</c> instead of by generating every earlier point.
        /// </summary>
        /// <param name="index">The zero-based sequence index; smaller than <c>2^bits</c>.</param>
        /// <returns>A new array of <see cref="Dim" /> coordinates in <c>[0, 1)</c>, bit-identical to row
        ///     <paramref name="index" /> of <see cref="Generate" />.</returns>
        /// <exception cref="NumSharp.ValueError">The index is past the last point of a <c>bits</c>-bit sequence.</exception>
        internal double[] Point(ulong index)
        {
            // The stepping form would index past the direction numbers here; scipy refuses to draw past 2^bits too.
            if (index >= 1UL << _bits)
                throw new ValueError($"The Sobol' sequence has only 2**{_bits} points; index {index} is past its end.");
            ulong gray = index ^ (index >> 1);
            int d = Dim;
            var point = new double[d];
            for (int j = 0; j < d; j++)
            {
                uint quasi = _shift[j];
                uint[] svj = _sv[j];
                // XOR is associative and commutative, so the order of the set bits does not matter.
                for (ulong g = gray, b = 0; g != 0; g >>= 1, b++)
                    if ((g & 1UL) != 0)
                        quasi ^= svj[b];
                point[j] = quasi * _scale;
            }
            return point;
        }
    }

    /// <summary>
    ///     scipy's <c>_cscramble</c> — the left linear matrix scramble applied in place to
    ///     <paramref name="sv" />. <paramref name="ltm" /> is the raw <c>(dim, bits, bits)</c> integer
    ///     cube (row-major); only its strict lower triangle is used and the diagonal is forced to 1.
    /// </summary>
    /// <param name="dim">The number of coordinates in each point.</param>
    /// <param name="bits">The number of direction-number bits used in the Sobol scramble.</param>
    /// <param name="ltm">The lower-triangular binary matrices defining the linear scramble.</param>
    /// <param name="sv">The Sobol direction-number vectors to scramble in place.</param>
    private static void Cscramble(int dim, int bits, long[] ltm, uint[][] sv)
    {
        for (int dd = 0; dd < dim; dd++)
        {
            for (int j = 0; j < bits; j++)
            {
                uint vdj = sv[dd][j];
                uint l = 1;
                uint t2 = 0;
                for (int p = bits - 1; p >= 0; p--)
                {
                    uint lsmdp = CdotPow2(ltm, dim, bits, dd, p);
                    uint t1 = 0;
                    for (int k = 0; k < bits; k++)
                        t1 += ((lsmdp >> k) & 1u) * ((vdj >> k) & 1u);
                    t1 %= 2;
                    t2 += t1 * l;
                    l *= 2;
                }
                sv[dd][j] = t2;
            }
        }
    }

    /// <summary>
    ///     scipy's <c>cdot_pow2</c> over row <paramref name="p" /> of the (tril-masked, unit-diagonal)
    ///     LMS matrix for dimension <paramref name="dd" />: <c>Σ_k a[k]·2^(bits-1-k)</c>.
    /// </summary>
    /// <param name="ltm">The lower-triangular binary matrices defining the linear scramble.</param>
    /// <param name="dim">The number of coordinates in each point.</param>
    /// <param name="bits">The number of direction-number bits used in the Sobol scramble.</param>
    /// <param name="dd">The dimension whose scramble matrix is being evaluated.</param>
    /// <param name="p">The bit position or cursor used by the operation.</param>
    /// <returns>The binary row packed into its corresponding direction-number integer.</returns>
    private static uint CdotPow2(long[] ltm, int dim, int bits, int dd, int p)
    {
        uint z = 0;
        for (int k = 0; k < bits; k++)
        {
            uint a;
            if (k < p)
                a = (uint)ltm[(dd * bits + p) * bits + k];  // strict lower triangle: random bit
            else if (k == p)
                a = 1u;                                       // diagonal forced to 1
            else
                a = 0u;                                       // upper triangle zeroed by tril
            if (a != 0u)
                z += 1u << (bits - 1 - k);
        }
        return z;
    }

    /// <summary>scipy's <c>low_0_bit</c>: the 1-based position of the right-most zero bit of <paramref name="x" />.</summary>
    /// <param name="x">The scalar argument.</param>
    /// <returns>The one-based position of the least significant zero bit.</returns>
    private static int LowZeroBit(uint x)
    {
        int i = 0;
        while ((x & (1u << i)) != 0)
            i++;
        return i + 1;
    }
}
