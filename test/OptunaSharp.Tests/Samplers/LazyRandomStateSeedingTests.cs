using System.Linq;
using AwesomeAssertions;
using OptunaSharp.Samplers;

namespace OptunaSharp.Tests.Samplers;

/// <summary>
///     Unseeded generators draw independent streams however close together they are created, as numpy's unseeded
///     <c>RandomState</c> does: it seeds itself from OS entropy. NumSharp's unseeded generator seeds from
///     <c>Environment.TickCount</c>, so samplers built within one tick repeated each other's draws: 8 parallel
///     <c>optunasharp ask</c> shells on one journal repeated 11 and 15 of their 80 draws in two runs.
/// </summary>
[TestClass]
public class LazyRandomStateSeedingTests
{
    /// <summary>Sixteen unseeded generators created back to back: no two may produce the same first draw.</summary>
    [TestMethod]
    public void Unseeded_CreatedTogether_DrawDistinctStreams()
    {
        var states = Enumerable.Range(0, 16).Select(_ => new LazyRandomState()).ToList();

        var firstDraws = states.Select(s => (double)s.Rng.rand()).ToList();

        firstDraws.Should().OnlyHaveUniqueItems("an unseeded generator must not take its seed from the clock");
    }

    /// <summary>
    ///     The same through the public surface: unseeded samplers of separate studies, as separate worker processes
    ///     or CLI calls build them, must not suggest the same values.
    /// </summary>
    [TestMethod]
    public void UnseededRandomSamplers_CreatedTogether_SuggestDifferentValues()
    {
        var studies = Enumerable.Range(0, 8).Select(_ => Optuna.CreateStudy(sampler: new RandomSampler())).ToList();

        var xs = studies.Select(s => s.Ask().SuggestFloat("x", 0, 1)).ToList();

        xs.Should().OnlyHaveUniqueItems();
    }

    /// <summary>An explicit seed still fixes the stream: the entropy path runs only for an unseeded generator.</summary>
    [TestMethod]
    public void Seeded_StaysReproducible()
    {
        double a = (double)new LazyRandomState(42).Rng.rand();
        double b = (double)new LazyRandomState(42).Rng.rand();

        a.Should().Be(b);
    }
}
