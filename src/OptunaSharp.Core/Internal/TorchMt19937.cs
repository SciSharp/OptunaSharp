namespace OptunaSharp.Internal;

/// <summary>
///     The classic MT19937 Mersenne-Twister with <c>init_genrand</c> seeding — a faithful reproduction
///     of PyTorch's CPU generator (<c>at::mt19937</c> in <c>ATen/core/MT19937RNGEngine.h</c>), which
///     <b>differs from NumPy's</b> MT19937 only in the seeding (NumPy uses <c>init_by_array</c>; torch,
///     like the reference implementation, uses the scalar <c>init_genrand</c>). The twist and tempering
///     are identical.
///
///     <para>Used to reproduce <c>torch.quasirandom.SobolEngine(scramble=True, seed=…)</c> bit-for-bit
///     for the GP QMC normal Sobol' sampler (<see cref="OptunaSharp.Gp.Qmc" />, PORTING_PLAN Phase 3.2
///     part 3e-3): <c>torch.randint(2, …, generator=g)</c> reduces to the low bit of a 32-bit draw
///     (<see cref="NextBit" />), verified against live torch.</para>
/// </summary>
internal sealed class TorchMt19937
{
    private const int N = 624;
    private const int M = 397;
    private const uint MatrixA = 0x9908b0dfu;
    private const uint UpperMask = 0x80000000u;
    private const uint LowerMask = 0x7fffffffu;

    private readonly uint[] _mt = new uint[N];
    private int _index;

    internal TorchMt19937(uint seed)
    {
        // init_genrand: s[0] = seed; s[j] = 1812433253 * (s[j-1] ^ (s[j-1] >> 30)) + j  (mod 2^32).
        _mt[0] = seed;
        for (int i = 1; i < N; i++)
            _mt[i] = 1812433253u * (_mt[i - 1] ^ (_mt[i - 1] >> 30)) + (uint)i;
        _index = N;
    }

    /// <summary>A tempered 32-bit output (<c>at::mt19937::operator()</c>).</summary>
    internal uint NextUInt32()
    {
        if (_index >= N)
            Generate();
        uint y = _mt[_index++];
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c5680u;
        y ^= (y << 15) & 0xefc60000u;
        y ^= y >> 18;
        return y;
    }

    /// <summary><c>torch.randint(2, …)</c> for a small range is <c>random() % 2</c> — the low bit of a 32-bit draw.</summary>
    internal uint NextBit() => NextUInt32() & 1u;

    private void Generate()
    {
        for (int i = 0; i < N; i++)
        {
            uint y = (_mt[i] & UpperMask) | (_mt[(i + 1) % N] & LowerMask);
            _mt[i] = _mt[(i + M) % N] ^ (y >> 1);
            if ((y & 1u) != 0)
                _mt[i] ^= MatrixA;
        }
        _index = 0;
    }
}
