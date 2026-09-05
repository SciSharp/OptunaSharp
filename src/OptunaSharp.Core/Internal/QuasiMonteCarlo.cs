using System.Collections.Generic;
using System.Numerics;
using NumSharp;

namespace OptunaSharp.Internal;

/// <summary>
///     The non-scrambled quasi-Monte-Carlo engines behind <see cref="Samplers.QMCSampler" /> — the
///     Sobol' sequence (Gray-code evaluation of the Joe-Kuo direction numbers) and the Halton
///     sequence (per-dimension van der Corput radical inverse in the first primes). Both reproduce
///     <c>scipy.stats.qmc</c> bit-for-bit for <c>scramble=False</c> (verified over many dimensions and
///     indices).
///
///     <para><b>Scrambling lives in the sibling engines.</b> scipy applies scrambling with its internal
///     <c>numpy.random.default_rng</c> (PCG64); both randomizations are now ported bit-for-bit —
///     <see cref="ScrambledSobol" /> (the linear-matrix-scramble + digital shift, scipy's <c>_cscramble</c>)
///     and <see cref="ScrambledHalton" /> (the Owen digit permutations, scipy's
///     <c>_van_der_corput_permutations</c> + <c>_cy_van_der_corput_scrambled</c>). Both reuse this class's
///     shared tables (<see cref="SobolSv" />, <see cref="SobolBits" />, <see cref="HaltonPrimes" />), so
///     <see cref="Samplers.QMCSampler" /> now supports every <c>(qmc_type, scramble)</c> combination
///     (PORTING_PLAN Phase 2.12 complete).</para>
/// </summary>
internal static class QuasiMonteCarlo
{
    // Number of fixed-point bits behind each Sobol' coordinate, matching scipy.stats.qmc.Sobol.
    private const int Bits = 30;
    private const ulong Capacity = 1UL << Bits;

    /// <summary>The Sobol' fixed-point bit count (<c>scipy.stats.qmc.Sobol</c>'s default <c>bits=30</c>).</summary>
    internal const int SobolBits = Bits;

    /// <summary>
    ///     A fresh copy of the first <paramref name="dim" /> rows of the (unscrambled) Sobol' direction
    ///     numbers — scipy's <c>_sv</c> after <c>_initialize_v</c>. Returned as a copy so the scrambled
    ///     engine (<see cref="ScrambledSobol" />) can apply the LMS in place without touching the cache.
    /// </summary>
    internal static uint[][] SobolSv(int dim)
    {
        if (dim < 1 || dim > SobolDirectionNumbers.MaxDim)
            throw new ValueError($"Sobol' dimension must be in 1..={SobolDirectionNumbers.MaxDim}, got {dim}.");
        uint[][] sv = DirectionNumbers();
        var copy = new uint[dim][];
        for (int d = 0; d < dim; d++)
            copy[d] = (uint[])sv[d].Clone();
        return copy;
    }

    private static uint[][]? _sv;      // Sobol' direction numbers [MaxDim][Bits], left-aligned.
    private static int[]? _primes;     // The first MaxDim primes (Halton bases).

    // ===========================================================================================
    // Sobol'
    // ===========================================================================================

    /// <summary>The Sobol' point at sequence index <paramref name="n" /> (index 0 is the origin).</summary>
    internal static double[] SobolPoint(int dim, ulong n)
    {
        if (dim < 1 || dim > SobolDirectionNumbers.MaxDim)
            throw new ValueError($"Sobol' dimension must be in 1..={SobolDirectionNumbers.MaxDim}, got {dim}.");
        if (n >= Capacity)
            throw new ValueError($"Sobol' point index must be below 2**{Bits}={Capacity}, got {n}.");

        uint[][] sv = DirectionNumbers();
        var point = new double[dim];
        for (int d = 0; d < dim; d++)
        {
            // XOR the direction numbers selected by the set bits of the Gray code of n.
            ulong gray = n ^ (n >> 1);
            uint quasi = 0;
            int j = 0;
            while (gray != 0)
            {
                if ((gray & 1UL) == 1UL)
                    quasi ^= sv[d][j];
                gray >>= 1;
                j++;
            }
            point[d] = quasi * (1.0 / Capacity);
        }
        return point;
    }

    /// <summary>
    ///     scipy's <c>_initialize_v</c>: build the left-aligned direction numbers from the Joe-Kuo
    ///     polynomials + initial m-values. Computed once for all <see cref="SobolDirectionNumbers.MaxDim" />
    ///     dimensions (row d depends only on dimension d).
    /// </summary>
    private static uint[][] DirectionNumbers()
    {
        if (_sv is not null)
            return _sv;

        int dim = SobolDirectionNumbers.MaxDim;
        ushort[] vinit = SobolDirectionNumbers.Vinit();
        var sv = new uint[dim][];
        for (int i = 0; i < dim; i++)
            sv[i] = new uint[Bits];

        // Dimension 0 is the van der Corput sequence: m_i = 1 for every i (unshifted).
        for (int j = 0; j < Bits; j++)
            sv[0][j] = 1;

        int vpos = 0;  // Dimension 0 has degree 0 → consumes no m-values.
        for (int i = 1; i < dim; i++)
        {
            int p = SobolDirectionNumbers.Poly(i);
            int degree = BitLength(p) - 1;
            for (int j = 0; j < degree; j++)
                sv[i][j] = vinit[vpos++];
            for (int j = degree; j < Bits; j++)
            {
                uint newv = sv[i][j - degree];
                uint pow2 = 1;
                for (int k = 0; k < degree; k++)
                {
                    pow2 <<= 1;
                    if (((p >> (degree - 1 - k)) & 1) == 1)
                        newv ^= pow2 * sv[i][j - k - 1];
                }
                sv[i][j] = newv;
            }
        }

        // Left-align every m_j so the stored integer is the direction number times 2^Bits.
        for (int i = 0; i < dim; i++)
            for (int j = 0; j < Bits; j++)
                sv[i][j] <<= Bits - 1 - j;

        _sv = sv;
        return sv;
    }

    // ===========================================================================================
    // Halton
    // ===========================================================================================

    /// <summary>The Halton point at sequence index <paramref name="n" /> (index 0 is the origin).</summary>
    internal static double[] HaltonPoint(int dim, ulong n)
    {
        if (dim < 1 || dim > SobolDirectionNumbers.MaxDim)
            throw new ValueError($"Halton dimension must be in 1..={SobolDirectionNumbers.MaxDim}, got {dim}.");

        int[] primes = Primes();
        var point = new double[dim];
        for (int d = 0; d < dim; d++)
        {
            ulong b = (ulong)primes[d];
            // van der Corput radical inverse, accumulated low digit to high (scipy's order).
            double r = 0.0;
            double f = 1.0 / b;
            ulong ii = n;
            while (ii > 0)
            {
                r += (double)(ii % b) * f;
                ii /= b;
                f /= b;
            }
            point[d] = r;
        }
        return point;
    }

    /// <summary>
    ///     A fresh copy of the first <paramref name="dim" /> Halton bases (primes 2, 3, 5, …) — the
    ///     shared prime table behind the unscrambled <see cref="HaltonPoint" /> and the scrambled
    ///     <see cref="ScrambledHalton" /> (the analogue of <see cref="SobolSv" /> for the Halton engine).
    /// </summary>
    internal static int[] HaltonPrimes(int dim)
    {
        if (dim < 1 || dim > SobolDirectionNumbers.MaxDim)
            throw new ValueError($"Halton dimension must be in 1..={SobolDirectionNumbers.MaxDim}, got {dim}.");
        return Primes()[..dim];
    }

    /// <summary>The first <see cref="SobolDirectionNumbers.MaxDim" /> primes (Halton bases 2, 3, 5, …).</summary>
    private static int[] Primes()
    {
        if (_primes is not null)
            return _primes;

        var primes = new List<int>(SobolDirectionNumbers.MaxDim);
        int candidate = 2;
        while (primes.Count < SobolDirectionNumbers.MaxDim)
        {
            bool isPrime = true;
            for (int i = 0; i < primes.Count && (long)primes[i] * primes[i] <= candidate; i++)
            {
                if (candidate % primes[i] == 0)
                {
                    isPrime = false;
                    break;
                }
            }
            if (isPrime)
                primes.Add(candidate);
            candidate++;
        }
        _primes = primes.ToArray();
        return _primes;
    }

    private static int BitLength(int value) => value == 0 ? 0 : BitOperations.Log2((uint)value) + 1;
}
