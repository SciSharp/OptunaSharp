using System;
using System.Security.Cryptography;
using NumSharp;

namespace OptunaSharp.Samplers;

/// <summary>
///     Lazy Random State class. Port of <c>optuna/samplers/_lazy_random_state.py</c>.
///
///     This is a class to initialize the random state just before use to prevent duplication of
///     the same random state when a sampler instance is copied.
///
///     Every sampler consumes randomness through this wrapper over NumSharp's NumPy-exact
///     MT19937 <c>RandomState</c> — never <see cref="System.Random" /> — which is what makes
///     differential testing against Python bit-exact (plan §2 RNG discipline).
/// </summary>
public sealed class LazyRandomState
{
    private NumPyRandom? _rng;

    /// <summary>Creates a random-state holder, initializing immediately only when an explicit seed is supplied.</summary>
    /// <param name="seed">Seed for subsequent random draws; null uses the generator default when permitted.</param>
    public LazyRandomState(int? seed = null)
    {
        _rng = null;
        if (seed is not null)
            Rng.seed(seed.Value);
    }

    /// <summary>Allocates the NumPy-compatible random generator on first access, seeded from OS entropy.</summary>
    /// <remarks>
    ///     Python's <c>np.random.RandomState()</c> seeds itself from the operating system's entropy source, so two
    ///     unseeded samplers never share a stream. NumSharp's unseeded constructor seeds from
    ///     <see cref="System.Environment.TickCount" /> instead, whose resolution is 1–16 ms: unseeded samplers created
    ///     within one tick drew identical streams. That duplicated trials across worker processes or parallel CLI
    ///     <c>ask</c> calls started together (8 parallel shells repeated 11 and 15 of their 80 draws in two runs),
    ///     and across studies whose samplers were built at once. The seed now comes from
    ///     <see cref="RandomNumberGenerator" />, as the samplers' <c>ReseedRng</c> already does. An explicit seed
    ///     never reaches this path, so seeded runs are unchanged.
    /// </remarks>
    private void SetRng()
    {
        Span<byte> entropy = stackalloc byte[4];
        RandomNumberGenerator.Fill(entropy);
        var rng = np.random.RandomState();
        rng.seed(BitConverter.ToUInt32(entropy));
        _rng = rng;
    }

    /// <summary>The underlying <c>np.random.RandomState</c>, created on first use.</summary>
    public NumPyRandom Rng
    {
        get
        {
            if (_rng is null)
                SetRng();
            return _rng!;
        }
    }
}
