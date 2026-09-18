using System;
using AwesomeAssertions;
using NumSharp;
using OptunaSharp.Samplers.Turbo;

namespace OptunaSharp.Tests.Samplers;

/// <summary>
///     Tests for <see cref="TurboMOptimizer" /> — the ONE mechanic TuRBO-m (Eriksson et al. 2019, §2) adds
///     over TuRBO-1: the cross-region Thompson competition (which region spends the next evaluation) and the
///     deterministic init-fill schedule that keeps all <c>m</c> regions live. Everything per-region is the
///     already-tested <see cref="TurboOptimizer" />; these gates cover only the new bandit layer, most
///     importantly the load-bearing <b>un-standardization</b> before the cross-region comparison.
/// </summary>
[TestClass]
public class TurboMOptimizerTests
{
    /// <summary>Builds one region's competition payload with a near-deterministic posterior (negligible diagonal
    ///     variance ⇒ the Thompson sample ≈ the mean), so a test can predict the winner from the means alone.</summary>
    /// <param name="slot">The region's slot index.</param>
    /// <param name="standardizedMean">The standardized joint posterior mean per candidate (also the candidate count).</param>
    /// <param name="scoreMean">The region's score-standardization center (un-standardization offset).</param>
    /// <param name="scoreStd">The region's score-standardization scale (un-standardization multiplier).</param>
    /// <param name="diagVar">The diagonal posterior variance (tiny by default ⇒ sample ≈ mean; raise it for stochasticity).</param>
    /// <returns>A ready-region payload for <see cref="TurboMOptimizer.SelectAcrossTrustRegions" />.</returns>
    private static TrustRegionSample Region(
        int slot, double[] standardizedMean, double scoreMean, double scoreStd, double diagVar = 1e-12)
    {
        int n = standardizedMean.Length;
        var candidates = new double[n, 1];
        for (int i = 0; i < n; i++) candidates[i, 0] = (i + 0.5) / n;   // distinct rows so the returned index is checkable
        var cov = new double[n, n];
        for (int i = 0; i < n; i++) cov[i, i] = diagVar;
        return new TrustRegionSample(slot, candidates, (double[])standardizedMean.Clone(), cov, scoreMean, scoreStd);
    }

    private static NumPyRandom Rng(uint seed)
    {
        var rng = np.random.RandomState();
        rng.seed(seed);
        return rng;
    }

    // ---- cross-region competition -----------------------------------------------------------------

    [TestMethod]
    public void SelectAcrossTrustRegions_PicksGlobalArgmaxAcrossRegions()
    {
        // §2: the winner is the single best (region, candidate) across the concatenated per-region samples.
        // Region 1's candidate 0 (=9) is the global maximum, so it wins.
        var r0 = Region(0, new[] { 1.0, 5.0, 2.0 }, scoreMean: 0.0, scoreStd: 1.0);
        var r1 = Region(1, new[] { 9.0, 3.0 }, scoreMean: 0.0, scoreStd: 1.0);
        var (region, candidate) = TurboMOptimizer.SelectAcrossTrustRegions(new[] { r0, r1 }, Rng(1u));
        region.Should().Be(1, "region 1 holds the globally best sampled candidate");
        candidate.Should().Be(0, "candidate 0 (=9) is that region's argmax");
    }

    [TestMethod]
    public void SelectAcrossTrustRegions_UnstandardizesBeforeComparing()
    {
        // The load-bearing detail (turbo_m.py `y_cand = mu + sigma * y_cand`): each region standardizes by its
        // OWN mean/std, so the cross-region comparison MUST un-standardize first. Region 0 looks better in
        // STANDARDIZED units (mean 10 vs 1) but region 1 wins in true objective units (100 + 1 = 101 vs 0 + 10).
        var r0 = Region(0, new[] { 10.0 }, scoreMean: 0.0, scoreStd: 1.0);     // un-standardized ≈ 10
        var r1 = Region(1, new[] { 1.0 }, scoreMean: 100.0, scoreStd: 1.0);    // un-standardized ≈ 101
        var (region, _) = TurboMOptimizer.SelectAcrossTrustRegions(new[] { r0, r1 }, Rng(1u));
        region.Should().Be(1,
            "the competition un-standardizes each region's draw to shared objective units before comparing");
    }

    [TestMethod]
    public void SelectAcrossTrustRegions_SingleRegion_ReturnsItsArgmax()
    {
        // With one region the competition degenerates to that region's Thompson argmax (TuRBO-1's pick).
        var r0 = Region(0, new[] { 1.0, 9.0, 2.0 }, scoreMean: 3.0, scoreStd: 2.0);
        var (region, candidate) = TurboMOptimizer.SelectAcrossTrustRegions(new[] { r0 }, Rng(5u));
        region.Should().Be(0);
        candidate.Should().Be(1, "index 1 (=9) is the single region's argmax");
    }

    [TestMethod]
    public void SelectAcrossTrustRegions_TieResolvesToFirstRegionThenCandidate()
    {
        // A zero score-scale collapses every candidate's un-standardized score to the region's scoreMean, so two
        // regions with the SAME scoreMean produce an exact global tie — resolved to the first region, first
        // candidate (numpy argmin's first-occurrence rule, in maximization form).
        var r0 = Region(0, new[] { 0.0, 0.0 }, scoreMean: 7.0, scoreStd: 0.0);
        var r1 = Region(1, new[] { 0.0, 0.0 }, scoreMean: 7.0, scoreStd: 0.0);
        var (region, candidate) = TurboMOptimizer.SelectAcrossTrustRegions(new[] { r0, r1 }, Rng(9u));
        region.Should().Be(0, "ties resolve to the first region (region-major scan)");
        candidate.Should().Be(0, "then the first candidate");
    }

    [TestMethod]
    public void SelectAcrossTrustRegions_SameSeed_IsReproducible()
    {
        // With real posterior variance the draw matters, yet a fixed seed reproduces the winner exactly (the
        // per-region z is consumed in a fixed region order).
        (int, int) Draw()
        {
            var r0 = Region(0, new[] { 0.0, 0.1, 0.2 }, scoreMean: 0.0, scoreStd: 1.0, diagVar: 1.0);
            var r1 = Region(1, new[] { 0.05, 0.15 }, scoreMean: 0.0, scoreStd: 1.0, diagVar: 1.0);
            return TurboMOptimizer.SelectAcrossTrustRegions(new[] { r0, r1 }, Rng(123u));
        }
        Draw().Should().Be(Draw(), "a fixed seed gives an identical competition outcome");
    }

    [TestMethod]
    public void SelectAcrossTrustRegions_Empty_Throws()
    {
        Action act = () => TurboMOptimizer.SelectAcrossTrustRegions(Array.Empty<TrustRegionSample>(), Rng(1u));
        act.Should().Throw<ArgumentException>("there is nothing to compete when no region is ready");
    }

    // ---- init-fill schedule -----------------------------------------------------------------------

    [TestMethod]
    public void NextInitSlot_ReturnsLowestSlotStillInitializing()
    {
        // Slot-by-slot fill / post-collapse re-init: the lowest slot short of nInit is filled next.
        TurboMOptimizer.NextInitSlot(new[] { 4, 4, 2, 4 }, nInit: 4).Should().Be(2);
        TurboMOptimizer.NextInitSlot(new[] { 0, 4, 4 }, nInit: 4).Should().Be(0, "the lowest short slot wins");
        TurboMOptimizer.NextInitSlot(new[] { 4, 2, 0 }, nInit: 4).Should().Be(1, "slot 1 is short before slot 2");
    }

    [TestMethod]
    public void NextInitSlot_AllReady_ReturnsMinusOne()
    {
        // Every slot has completed its nInit design ⇒ the competition runs instead of an init fill.
        TurboMOptimizer.NextInitSlot(new[] { 4, 5, 4 }, nInit: 4).Should().Be(-1);
    }

    [TestMethod]
    public void NextInitSlot_Empty_Throws()
    {
        Action act = () => TurboMOptimizer.NextInitSlot(Array.Empty<int>(), nInit: 4);
        act.Should().Throw<ArgumentException>("there must be at least one trust region");
    }
}
