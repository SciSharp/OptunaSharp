using AwesomeAssertions;
using OptunaSharp.Samplers.SmsEmoa;

namespace OptunaSharp.Tests.Samplers;

/// <summary>
///     Unit tests for <see cref="SmsEmoaSurvival" /> — SMS-EMOA's environmental survival (least
///     hypervolume contribution in the worst nondomination front, pymoo's
///     <c>LeastHypervolumeContributionSurvival</c>). These pin the survival rule in objective space,
///     independent of the sampler wiring: worse fronts go first, and within the one overflowing front
///     the least-exclusive-hypervolume individual is dropped.
/// </summary>
[TestClass]
public class SmsEmoaSurvivalTests
{
    [TestMethod]
    public void Survival_RemovesLeastHypervolumeContributor_InSingleFront()
    {
        // Four mutually nondominated points. P3 sits right beside P2, so its exclusive hypervolume
        // contribution is by far the smallest and it is the one removed (μ+1 → μ).
        var loss = new[]
        {
            new[] { 0.0, 5.0 },    // 0 — extreme (min f1)
            new[] { 5.0, 0.0 },    // 1 — extreme (min f2)
            new[] { 2.0, 2.0 },    // 2 — genuine middle point
            new[] { 2.01, 1.99 },  // 3 — near-duplicate of 2, least contribution
        };

        var survivors = SmsEmoaSurvival.Select(
            loss, nSurvive: 3, ideal: new[] { 0.0, 0.0 }, nadir: new[] { 5.0, 5.0 });

        survivors.Should().BeEquivalentTo(new[] { 0, 1, 2 });
    }

    [TestMethod]
    public void Survival_DropsWorstFrontsFirst()
    {
        // Fronts: {0} dominates all; {1} dominates {2,3}; {2,3} last. n_survive=2 keeps the two best
        // fronts whole — no hypervolume needed since the second front fits exactly.
        var loss = new[]
        {
            new[] { 0.0, 0.0 },   // 0 — front 0
            new[] { 1.0, 1.0 },   // 1 — front 1
            new[] { 1.0, 2.0 },   // 2 — front 2
            new[] { 2.0, 1.0 },   // 3 — front 2
        };

        var survivors = SmsEmoaSurvival.Select(
            loss, nSurvive: 2, ideal: new[] { 0.0, 0.0 }, nadir: new[] { 2.0, 2.0 });

        survivors.Should().BeEquivalentTo(new[] { 0, 1 });
    }

    [TestMethod]
    public void Survival_RemovesADuplicatePoint()
    {
        // A duplicate has zero exclusive contribution, so exactly one copy is dropped while the spread
        // endpoints survive.
        var loss = new[]
        {
            new[] { 0.0, 2.0 },   // 0
            new[] { 1.0, 1.0 },   // 1
            new[] { 2.0, 0.0 },   // 2
            new[] { 1.0, 1.0 },   // 3 — duplicate of 1
        };

        var survivors = SmsEmoaSurvival.Select(
            loss, nSurvive: 3, ideal: new[] { 0.0, 0.0 }, nadir: new[] { 2.0, 2.0 });

        survivors.Should().HaveCount(3);
        survivors.Should().Contain(0).And.Contain(2);
        (System.Array.IndexOf(survivors, 1) >= 0 ^ System.Array.IndexOf(survivors, 3) >= 0)
            .Should().BeTrue("exactly one of the duplicate pair survives");
    }

    [TestMethod]
    public void Survival_TrivialBounds()
    {
        var loss = new[] { new[] { 0.0, 1.0 }, new[] { 1.0, 0.0 } };
        SmsEmoaSurvival.Select(loss, 5, new[] { 0.0, 0.0 }, new[] { 1.0, 1.0 })
            .Should().BeEquivalentTo(new[] { 0, 1 }, "n_survive ≥ n keeps everyone");
        SmsEmoaSurvival.Select(loss, 0, new[] { 0.0, 0.0 }, new[] { 1.0, 1.0 })
            .Should().BeEmpty("n_survive 0 keeps no one");
    }
}
