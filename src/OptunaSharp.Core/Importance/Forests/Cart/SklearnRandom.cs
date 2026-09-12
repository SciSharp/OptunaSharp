namespace OptunaSharp.Importance.Forests.Internal.Cart;

/// <summary>
///     Port of scikit-learn's splitter RNG: <c>sklearn/utils/_random.pxd: our_rand_r</c> (a 32-bit
///     XorShift, the "rand_r replacement") and <c>sklearn/tree/_utils.pyx: rand_int</c>. This is the
///     one piece with no numpy equivalent — numpy's MT19937 (reproduced byte-for-byte by NumSharp)
///     seeds it and drives the bagging, but the per-node feature shuffling inside the tree runs on
///     this generator. Verified bit-exact against sklearn's <c>_our_rand_r_py</c> hook.
/// </summary>
internal static class SklearnRandom
{
    /// <summary>The largest integer returned by scikit-learn's portable rand-r implementation.</summary>
    internal const int RandRMax = 2147483647;  // 2**31 - 1
    private const uint DefaultSeed = 1;

    /// <summary>Generate a pseudo-random uint32 and advance the seed (<c>our_rand_r</c>).</summary>
    /// <param name="seed">The seed controlling the reproducible random stream.</param>
    /// <returns>The next scikit-learn-compatible random integer, with the seed advanced in place.</returns>
    internal static uint OurRandR(ref uint seed)
    {
        // seed shouldn't ever be 0.
        if (seed == 0)
            seed = DefaultSeed;

        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;

        // Modulo 2**31 so the result never exceeds RAND_R_MAX (matches (uint)RAND_R_MAX + 1).
        return seed % 2147483648u;
    }

    /// <summary>Generate a random integer in <c>[low, high)</c> (<c>rand_int</c>).</summary>
    /// <param name="low">The lower endpoint of the allowed range.</param>
    /// <param name="high">The upper endpoint of the allowed range.</param>
    /// <param name="seed">The seed controlling the reproducible random stream.</param>
    /// <returns>A pseudorandom integer in the half-open interval from low to high.</returns>
    internal static int RandInt(int low, int high, ref uint seed)
        => low + (int)(OurRandR(ref seed) % (uint)(high - low));
}
