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
    internal static double[,] Random(int d, int n, int seed)
    {
        int bits = QuasiMonteCarlo.SobolBits;  // 30
        uint[][] sv = QuasiMonteCarlo.SobolSv(d);
        Generator rng = np.random.default_rng(seed);

        // _scramble(): the digital shift then the lower-triangular LMS matrices, in this rng order.
        // shift[i] = Σ_j M[i,j]·2^j  with M = integers(2, size=(d, bits)).
        long[] shiftBits = rng.integers(2, null, new Shape(d, bits)).ToArray<long>();
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
        long[] ltm = rng.integers(2, null, new Shape(d, bits, bits)).ToArray<long>();
        Cscramble(d, bits, ltm, sv);

        // Generate: point 0 is the shift; each subsequent point XORs a direction number (Gray code).
        double scale = 1.0 / Math.Pow(2.0, bits);
        var sample = new double[n, d];
        var quasi = (uint[])shift.Clone();
        for (int j = 0; j < d; j++)
            sample[0, j] = quasi[j] * scale;
        for (int k = 1; k < n; k++)
        {
            int l = LowZeroBit((uint)(k - 1));
            for (int j = 0; j < d; j++)
            {
                quasi[j] ^= sv[j][l - 1];
                sample[k, j] = quasi[j] * scale;
            }
        }
        return sample;
    }

    /// <summary>
    ///     scipy's <c>_cscramble</c> — the left linear matrix scramble applied in place to
    ///     <paramref name="sv" />. <paramref name="ltm" /> is the raw <c>(dim, bits, bits)</c> integer
    ///     cube (row-major); only its strict lower triangle is used and the diagonal is forced to 1.
    /// </summary>
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
    private static int LowZeroBit(uint x)
    {
        int i = 0;
        while ((x & (1u << i)) != 0)
            i++;
        return i + 1;
    }
}
