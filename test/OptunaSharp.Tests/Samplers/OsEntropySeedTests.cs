using System.Collections.Generic;
using System.Linq;
using AwesomeAssertions;
using NumSharp;
using OptunaSharp.Internal;
using OptunaSharp.Samplers;

namespace OptunaSharp.Tests.Samplers;

/// <summary>
///     Seeding from OS entropy is numpy's own (plan 10, item 2.1; THREADING.md §6.3). The unseeded
///     <c>RandomState()</c> and the no-argument <c>RandomState.seed()</c> expand 128 bits of entropy through a
///     <see cref="SeedSequence" /> into the whole MT19937 key, where OptunaSharp used to pass one 32-bit word to
///     <c>seed(uint)</c>.
/// </summary>
/// <remarks>
///     The expected values were produced by numpy 2.4.2 with the entropy injected, so they pin the exact
///     construction and not only its spread. For the constructor, numpy itself ran
///     <c>RandomState(MT19937(SeedSequence(e)))</c>, which executes the code <c>RandomState()</c> runs with a given
///     <c>SeedSequence</c> in place of a fresh one. <c>seed()</c> cannot take injected entropy, so its expected
///     draws come from numpy after its two effects were applied through <c>set_state</c>. The source
///     (<c>MT19937._legacy_seeding(None)</c> + <c>_reset_gauss</c>) writes the new key and drops the cached
///     Gaussian, and leaves the position alone, which was measured: a generator at position 24 is still at 24 after
///     <c>seed()</c>.
/// </remarks>
[TestClass]
public class OsEntropySeedTests
{
    /// <summary>The injected entropy of every oracle value below: numpy <c>SeedSequence([0x01234567, …])</c>.</summary>
    /// <returns>A fresh sequence over the four fixed words.</returns>
    private static SeedSequence FixedEntropy() => new(new uint[] { 0x01234567, 0x89ABCDEF, 0x0F1E2D3C, 0x4B5A6978 });

    /// <summary>
    ///     The constructor's state: numpy's key (first word forced to <c>0x80000000</c>, the rest from
    ///     <c>generate_state(624)</c>), its starting position 623, no cached Gaussian, and therefore numpy's draws.
    /// </summary>
    [TestMethod]
    public void NewRandomState_IsNumpysUnseededConstructor_ForAGivenEntropy()
    {
        var rng = OsEntropySeed.NewRandomState(FixedEntropy());

        var state = rng.get_state();
        state.Key[0].Should().Be(0x80000000u, "numpy forces the first key word so the key can never be all zero");
        state.Key.Skip(1).Take(3).Should().Equal(2647680151u, 3470823803u, 998631274u);
        state.Key[623].Should().Be(3752903327u);
        state.Pos.Should().Be(623, "numpy's constructor leaves the position at its key-copy loop's last index");
        state.HasGauss.Should().Be(0);

        rng.random_sample(4).ToArray<double>().Should().Equal(
            0.8813675928436056, 0.2663092905649186, 0.28729013059816444, 0.4493467611510169);
    }

    /// <summary>Integer and Gaussian draws from the constructed state also match numpy's.</summary>
    [TestMethod]
    public void NewRandomState_IntegerAndNormalDraws_MatchNumpy()
    {
        OsEntropySeed.NewRandomState(FixedEntropy()).randint(0, 1000, new Shape(5)).astype(np.int64).ToArray<long>()
            .Should().Equal(623L, 649L, 101L, 588L, 144L);

        OsEntropySeed.NewRandomState(FixedEntropy()).standard_normal(new Shape(3)).ToArray<double>()
            .Should().Equal(-0.34883538588396795, 0.5692759961867279, -0.4213580054829843);
    }

    /// <summary>
    ///     A reseed replaces the key, keeps the position and drops the cached Gaussian. The generator first draws five
    ///     doubles and one normal (position 14, a Gaussian cached), as in the numpy run.
    /// </summary>
    [TestMethod]
    public void Reseed_IsNumpysSeedWithNoArgument_KeepsThePositionAndDropsTheCachedGaussian()
    {
        var rng = PositionedGenerator();
        rng.get_state().Pos.Should().Be(14);
        rng.get_state().HasGauss.Should().Be(1);

        OsEntropySeed.Reseed(rng, FixedEntropy());

        var state = rng.get_state();
        state.Pos.Should().Be(14, "numpy's seed() writes the key and leaves the position");
        state.HasGauss.Should().Be(0, "RandomState.seed resets the Gaussian cache");
        state.Key[0].Should().Be(0x80000000u);
        state.Key.Skip(1).Take(3).Should().Equal(2647680151u, 3470823803u, 998631274u);
        rng.random_sample(3).ToArray<double>().Should().Equal(0.8312054665772319, 0.9433144982061366, 0.8181049259562833);
    }

    /// <summary>After a reseed, a normal draw comes from the new key rather than from the dropped cached value.</summary>
    [TestMethod]
    public void Reseed_NormalDrawsAfterwards_MatchNumpy()
    {
        var rng = PositionedGenerator();

        OsEntropySeed.Reseed(rng, FixedEntropy());

        rng.standard_normal(new Shape(2)).ToArray<double>().Should().Equal(-0.2984459376901813, 1.2697608727041028);
    }

    /// <summary>An unseeded <see cref="LazyRandomState" /> builds numpy's constructor state, from fresh entropy.</summary>
    [TestMethod]
    public void UnseededLazyRandomState_HasNumpysConstructorShape()
    {
        var state = new LazyRandomState().Rng.get_state();

        state.Key[0].Should().Be(0x80000000u);
        state.Pos.Should().Be(623);
        state.HasGauss.Should().Be(0);
    }

    /// <summary>A sampler's reseed goes through the same port: the key's first word is forced, the position kept.</summary>
    [TestMethod]
    public void SamplerReseed_KeepsItsGeneratorsPosition()
    {
        var sampler = new RandomSampler(seed: 3);
        sampler.Rng.Rng.random_sample(7);
        int position = sampler.Rng.Rng.get_state().Pos;

        sampler.ReseedRng();

        var state = sampler.Rng.Rng.get_state();
        state.Pos.Should().Be(position);
        state.Key[0].Should().Be(0x80000000u);
    }

    /// <summary>
    ///     300,000 reseeds of one generator, each followed by one draw, repeat no draw. With one 32-bit word per seed
    ///     the birthday bound predicts about 10.5 repeats (300,000² / 2³³), and the old code measured 116 in a million
    ///     reseeds against the bound's 116.4; with 128 bits a repeat is out of reach.
    /// </summary>
    [TestMethod]
    public void Reseed_ManyTimes_RepeatsNoDraw()
    {
        const int reseeds = 300_000;
        var rng = OsEntropySeed.NewRandomState();
        var seen = new HashSet<double>(reseeds);
        int repeats = 0;

        for (int i = 0; i < reseeds; i++)
        {
            OsEntropySeed.Reseed(rng);
            if (!seen.Add((double)rng.rand()))
                repeats++;
        }

        repeats.Should().Be(0);
    }

    /// <summary>An explicit seed still fixes the stream: <see cref="LazyRandomState" /> reseeds the constructed state.</summary>
    [TestMethod]
    public void SeededLazyRandomState_IsUnchanged_AndEqualsAPlainSeededGenerator()
    {
        new LazyRandomState(42).Rng.random_sample(3).ToArray<double>()
            .Should().Equal(np.random.RandomState(42).random_sample(3).ToArray<double>());
    }

    /// <summary>The generator the reseed oracle starts from: <c>RandomState(7)</c> after five doubles and one normal.</summary>
    /// <returns>A generator at position 14 with a cached Gaussian.</returns>
    private static NumPyRandom PositionedGenerator()
    {
        var rng = np.random.RandomState(7);
        rng.random_sample(5);
        rng.standard_normal();
        return rng;
    }
}
