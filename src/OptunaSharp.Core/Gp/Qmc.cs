using System;
using OptunaSharp.Internal;

namespace OptunaSharp.Gp;

/// <summary>
///     Port of <c>optuna/_gp/qmc.py: sample_from_normal_sobol</c> — the standard-normal Quasi-Monte-Carlo
///     base samples the multi-objective and q-batch GP acquisition functions integrate over (LogEHVI /
///     qLogEI / qLogEHVI, PORTING_PLAN Phase 3.2 part 3e). Optuna draws a <b>scrambled</b> Sobol'
///     sequence in <c>[0, 1)</c> with <c>torch.quasirandom.SobolEngine</c> and maps it to the standard
///     normal by <c>erfinv(2u − 1)·√2 = Φ⁻¹(u)</c>.
///
///     <para><b>Reproducing torch bit-closely (the QMC gate for the MO/q-batch acqfs).</b> This is
///     <em>not</em> scipy's scramble (that is <see cref="ScrambledSobol" />, used by the acqf optimizer's
///     preliminary sampling): torch's <c>SobolEngine</c> scrambles with its <b>own</b> RNG. Three facts,
///     all verified against live torch 2.x, make it reproducible:</para>
///     <list type="number">
///       <item>torch's <b>unscrambled</b> Sobol' sequence is byte-identical to scipy's, so the
///       unscrambled direction numbers are exactly <see cref="QuasiMonteCarlo.SobolSv" />.</item>
///       <item>torch's CPU generator is the classic MT19937 (<see cref="TorchMt19937" />), and
///       <c>randint(2, …)</c> is the low bit of each 32-bit draw; the shift (<c>dim × 30</c>) and the
///       lower-triangular scramble matrices (<c>dim × 30 × 30</c>) are drawn in that order, C-order.</item>
///       <item>the scramble is the standard unit-diagonal <b>left matrix scramble</b> (MSB-first) — the
///       same transform scipy's <c>_cscramble</c> applies, only fed torch's bits.</item>
///     </list>
///     <para>The final <c>Φ⁻¹</c> uses cephes <see cref="TruncNorm.Ndtri" /> (reference-quality); torch's
///     <c>erfinv</c> is machine-accurate too, so the samples agree to ~1e-15 — well inside the erf/ucrt
///     tolerance the acqfs already carry (the documented GP boundary), not bit-for-bit.</para>
/// </summary>
internal static class Qmc
{
    private const int Bits = 30;  // torch SobolEngine MAXBIT

    /// <summary>
    ///     <c>sample_from_normal_sobol(dim, n_samples, seed)</c> — the first <paramref name="nSamples" />
    ///     points of the standard-normal QMC sequence, shape <c>(n_samples, dim)</c>.
    /// </summary>
    internal static double[,] SampleFromNormalSobol(int dim, int nSamples, int seed)
    {
        double[,] sobol = ScrambledSobolDraw(dim, nSamples, seed);
        var result = new double[nSamples, dim];
        for (int i = 0; i < nSamples; i++)
            for (int j = 0; j < dim; j++)
                // 2*(u - 0.5) in [-1, 1], then erfinv(·)·√2 = Φ⁻¹(u) = ndtri(u).
                result[i, j] = TruncNorm.Ndtri(sobol[i, j]);
        return result;
    }

    /// <summary>
    ///     <c>torch.quasirandom.SobolEngine(dimension=dim, scramble=True, seed=seed).draw(n)</c> — the
    ///     scrambled Sobol' points in <c>[0, 1)</c>, shape <c>(n, dim)</c>.
    /// </summary>
    private static double[,] ScrambledSobolDraw(int dim, int n, int seed)
    {
        var mt = new TorchMt19937((uint)seed);

        // _scramble(): the digital shift bits (dim × Bits, C-order) then the lower-triangular scramble
        // matrices (dim × Bits × Bits, C-order). randint(2) == the low bit of each 32-bit draw.
        var shift = new uint[dim];
        for (int d = 0; d < dim; d++)
        {
            uint s = 0;
            for (int j = 0; j < Bits; j++)
                if (mt.NextBit() != 0u)
                    s |= 1u << j;
            shift[d] = s;
        }
        var ltm = new uint[dim][,];
        for (int d = 0; d < dim; d++)
        {
            var mat = new uint[Bits, Bits];
            for (int r = 0; r < Bits; r++)
                for (int c = 0; c < Bits; c++)
                    mat[r, c] = mt.NextBit();  // the full cube is drawn; only the lower triangle is used.
            ltm[d] = mat;
        }

        // Apply the LMS to the (torch == scipy) unscrambled direction numbers.
        uint[][] sv = QuasiMonteCarlo.SobolSv(dim);
        Scramble(dim, ltm, sv);

        // Gray-code generation: point 0 is the shift; each next point XORs one direction number.
        double scale = 1.0 / Math.Pow(2.0, Bits);
        var sample = new double[n, dim];
        var quasi = (uint[])shift.Clone();
        // torch computes the FIRST point (`_first_point = quasi / 2**MAXBIT`) with no explicit dtype, so
        // it lands in torch's default float32 and is then upcast — a ~1e-8 rounding on row 0 only. The
        // remaining points come from `_sobol_engine_draw(dtype=float64)` at full precision.
        for (int j = 0; j < dim; j++)
            sample[0, j] = (float)(quasi[j] * scale);
        for (int k = 1; k < n; k++)
        {
            int l = LowZeroBit((uint)(k - 1));
            for (int j = 0; j < dim; j++)
            {
                quasi[j] ^= sv[j][l - 1];
                sample[k, j] = quasi[j] * scale;
            }
        }
        return sample;
    }

    /// <summary>
    ///     torch's <c>_sobol_engine_scramble_</c> — the unit-diagonal left matrix scramble applied in
    ///     place to the direction numbers. Each direction number <c>v</c> is a <see cref="Bits" />-bit
    ///     value with its most-significant bit at <c>2^(Bits−1)</c>; the scrambled MSB-position <c>l</c>
    ///     bit is <c>v_l ⊕ XOR_{k&lt;l} ltm[d][l][k]·v_k</c> (the forced unit diagonal plus the strict
    ///     lower triangle).
    /// </summary>
    private static void Scramble(int dim, uint[][,] ltm, uint[][] sv)
    {
        for (int d = 0; d < dim; d++)
        {
            uint[,] mat = ltm[d];
            for (int i = 0; i < Bits; i++)
            {
                uint v = sv[d][i];
                uint outVal = 0;
                for (int l = 0; l < Bits; l++)
                {
                    uint r = (v >> (Bits - 1 - l)) & 1u;   // diagonal (unit)
                    for (int k = 0; k < l; k++)            // strict lower triangle
                        if (mat[l, k] != 0u)
                            r ^= (v >> (Bits - 1 - k)) & 1u;
                    outVal |= (r & 1u) << (Bits - 1 - l);
                }
                sv[d][i] = outVal;
            }
        }
    }

    /// <summary>The 1-based position of the right-most zero bit of <paramref name="x" /> (Gray-code index).</summary>
    private static int LowZeroBit(uint x)
    {
        int i = 0;
        while ((x & (1u << i)) != 0u)
            i++;
        return i + 1;
    }
}
