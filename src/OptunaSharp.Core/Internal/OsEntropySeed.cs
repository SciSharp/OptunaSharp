using NumSharp;

namespace OptunaSharp.Internal;

/// <summary>
///     numpy's two ways of giving an MT19937 <c>RandomState</c> fresh OS entropy, ported exactly: the unseeded
///     constructor <c>numpy.random.RandomState()</c>, and <c>RandomState.seed()</c> with no argument. Optuna uses both:
///     <c>LazyRandomState</c> builds an unseeded sampler's generator with the first, and every sampler's
///     <c>reseed_rng</c> calls the second, once per trial of an <c>n_jobs</c> worker.
/// </summary>
/// <remarks>
///     <para>
///         Both draw 128 bits of OS entropy through a <see cref="SeedSequence" /> and expand them into the whole
///         624-word key, as numpy 2.4.2's <c>MT19937.__init__</c> and <c>MT19937._legacy_seeding(None)</c> do
///         (<c>numpy/random/_mt19937.pyx</c>). Before this port each OS seed was one 32-bit word, passed to
///         <c>RandomState.seed(uint)</c>. One million reseeds then repeated 116 first draws, the birthday bound for
///         2^32 states; one million numpy reseeds repeat none (THREADING.md §6.3).
///     </para>
///     <para>
///         Two numpy quirks are kept, because they decide the first numbers drawn. The constructor leaves the
///         position at 623, the last value of its key-copy loop, so the first draw tempers the key's last word before
///         the first twist. <c>seed()</c> leaves the position where it was and only drops the cached Gaussian, so a
///         reseeded generator continues at its old position, into the new key. An explicit seed overwrites this
///         state at once (<c>LazyRandomState(seed)</c> calls <c>seed(seed)</c> on the new generator, as Python's
///         does), so seeded runs stay bit-identical to Python.
///     </para>
///     <para>Not thread-safe on one generator, like the rest of NumSharp's <c>RandomState</c>: callers hold the
///     sampler monitor, as every <c>ReseedRng</c> does (THREADING.md §6.3).</para>
/// </remarks>
internal static class OsEntropySeed
{
    /// <summary>The MT19937 key length, numpy's <c>RK_STATE_LEN</c>.</summary>
    private const int StateLength = 624;

    /// <summary>
    ///     The first key word numpy forces after the expansion ("MSB is 1; assuring non-zero initial array"): a key of
    ///     all zeros would make MT19937 output only zeros.
    /// </summary>
    private const uint FirstKeyWord = 0x80000000u;

    /// <summary>Python <c>numpy.random.RandomState()</c>: a new generator seeded from fresh OS entropy.</summary>
    /// <returns>A generator whose state is numpy's for 128 fresh bits of entropy.</returns>
    internal static NumPyRandom NewRandomState() => NewRandomState(new SeedSequence());

    /// <summary>
    ///     Python <c>numpy.random.RandomState(numpy.random.MT19937(entropy))</c>: the unseeded constructor's state for a
    ///     given entropy. Production passes a fresh <see cref="SeedSequence" />; tests pass a fixed one to compare with
    ///     numpy.
    /// </summary>
    /// <param name="entropy">The sequence whose first 624 generated words become the key.</param>
    /// <returns>A generator at numpy's starting position 623, with no cached Gaussian.</returns>
    internal static NumPyRandom NewRandomState(SeedSequence entropy)
    {
        // NumSharp's own unseeded constructor seeds from the clock; the state is replaced at once, so that seed never
        // produces a number.
        var rng = np.random.RandomState();
        rng.set_state(new NativeRandomState(KeyFrom(entropy), pos: StateLength - 1));
        return rng;
    }

    /// <summary>Python <c>RandomState.seed()</c> with no argument: reseeds <paramref name="rng" /> from fresh OS entropy.</summary>
    /// <param name="rng">The generator to reseed, in place.</param>
    internal static void Reseed(NumPyRandom rng) => Reseed(rng, new SeedSequence());

    /// <summary>
    ///     Python <c>RandomState.seed()</c> with no argument, for a given entropy: the key is replaced, the position is
    ///     kept, and the cached Gaussian is dropped.
    /// </summary>
    /// <param name="rng">The generator to reseed, in place.</param>
    /// <param name="entropy">The sequence whose first 624 generated words become the key.</param>
    internal static void Reseed(NumPyRandom rng, SeedSequence entropy)
    {
        // numpy's _legacy_seeding(None) writes the key and nothing else, and RandomState.seed then resets the Gaussian
        // cache; the position survives (measured on numpy 2.4.2: a generator at 24 is still at 24 after seed()).
        int position = rng.get_state().Pos;
        rng.set_state(new NativeRandomState(KeyFrom(entropy), position));
    }

    /// <summary>numpy's key for <paramref name="entropy" />: its first 624 generated words, with the first forced.</summary>
    /// <param name="entropy">The sequence to expand.</param>
    /// <returns>A fresh 624-word key.</returns>
    private static uint[] KeyFrom(SeedSequence entropy)
    {
        var key = (uint[])entropy.generate_state(StateLength);
        key[0] = FirstKeyWord;
        return key;
    }
}
