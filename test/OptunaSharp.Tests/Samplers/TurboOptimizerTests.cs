using System;
using AwesomeAssertions;
using NumSharp;
using OptunaSharp.Internal;
using OptunaSharp.Samplers.Turbo;

namespace OptunaSharp.Tests.Samplers;

/// <summary>
///     Tests for the native <see cref="TurboOptimizer" /> — TuRBO's trust-region mechanics (Eriksson et al.
///     2019): the deterministic state fold (expand / shrink / restart), the length-scale-weighted trust
///     region box, the Sobol perturbation candidate set, and Thompson sampling over candidates. TuRBO has no
///     Optuna counterpart, so these are behavioral gates on the exact TR algebra the paper specifies.
/// </summary>
[TestClass]
public class TurboOptimizerTests
{
    /// <summary>An optimizer with the paper/BoTorch defaults (τ_succ = 3) and a small τ_fail for compact tests.</summary>
    private static TurboOptimizer Make(int tauFail = 4) =>
        new(lengthInit: 0.8, lengthMin: 1.0 / 128.0, lengthMax: 1.6, tauSucc: 3, tauFail: tauFail, successTol: 1e-3);

    // ---- state fold: expand / shrink / restart ---------------------------------------------------

    [TestMethod]
    public void Fold_InitOnly_KeepsInitialLength()
    {
        var s = Make().ReconstructState(new double[] { 0, 0 }, nInit: 2);
        s.Length.Should().Be(0.8);
        s.RestartStart.Should().Be(0);
        s.SuccessCounter.Should().Be(0);
        s.FailureCounter.Should().Be(0);
        s.RestartCount.Should().Be(0);
    }

    [TestMethod]
    public void Fold_ConsecutiveSuccesses_ExpandsLength()
    {
        // 2 init (best=0), then 3 strictly-improving scores == τ_succ → double 0.8 → 1.6.
        var s = Make().ReconstructState(new double[] { 0, 0, 1, 2, 3 }, nInit: 2);
        s.Length.Should().Be(1.6, "τ_succ improvements double the base length");
        s.SuccessCounter.Should().Be(0, "the counter resets after a resize");
    }

    [TestMethod]
    public void Fold_ExpansionIsCappedAtLengthMax()
    {
        // Start already at 1.0 (via ctor) and improve: 2·1.0 = 2.0 is capped to lengthMax = 1.6.
        var opt = new TurboOptimizer(1.0, 1.0 / 128.0, 1.6, tauSucc: 2, tauFail: 4, successTol: 1e-3);
        var s = opt.ReconstructState(new double[] { 0, 0, 1, 2 }, nInit: 2);
        s.Length.Should().Be(1.6, "expansion is capped at lengthMax");
    }

    [TestMethod]
    public void Fold_ConsecutiveFailures_ShrinksLength()
    {
        // 2 init (best=0), then 4 non-improving scores == τ_fail → halve 0.8 → 0.4.
        var s = Make().ReconstructState(new double[] { 0, 0, -1, -1, -1, -1 }, nInit: 2);
        s.Length.Should().Be(0.4, "τ_fail stalls halve the base length");
        s.FailureCounter.Should().Be(0, "the counter resets after a resize");
    }

    [TestMethod]
    public void Fold_CollapseTriggersRestart()
    {
        // 2 init + 28 failures → 7 halvings (0.8 → 0.00625 < 2⁻⁷) → discard & restart at lengthInit.
        var seq = new double[2 + 28];
        for (int i = 2; i < seq.Length; i++) seq[i] = -1;
        var s = Make().ReconstructState(seq, nInit: 2);
        s.RestartCount.Should().Be(1, "the region collapsed once");
        s.RestartStart.Should().Be(30, "the next trial begins a fresh restart");
        s.Length.Should().Be(0.8, "the restarted region resets to lengthInit");
    }

    [TestMethod]
    public void Fold_NonFiniteScore_NeverImproves()
    {
        // A blown-up (−∞) score after the init phase must count as a failure, not a spurious success.
        var s = Make().ReconstructState(new double[] { 0, 0, double.NegativeInfinity }, nInit: 2);
        s.FailureCounter.Should().Be(1);
        s.SuccessCounter.Should().Be(0);
    }

    // ---- trust-region box -------------------------------------------------------------------------

    [TestMethod]
    public void TrustRegionBounds_Isotropic_IsCenteredBox()
    {
        // half = 0.5·1·0.8 = 0.4 ⇒ [0.1, 0.9] up to a ULP (0.5 − 0.4 = 0.0999…8), so compare approximately.
        var (lb, ub) = Make().TrustRegionBounds(new[] { 0.5, 0.5, 0.5 }, new[] { 1.0, 1.0, 1.0 }, 0.8);
        for (int j = 0; j < 3; j++)
        {
            lb[j].Should().BeApproximately(0.1, 1e-12);
            ub[j].Should().BeApproximately(0.9, 1e-12);
        }
    }

    [TestMethod]
    public void TrustRegionBounds_Anisotropic_ScalesByLengthScaleWithUnitVolume()
    {
        // Length scales [4, 1] → geomean 2 → weights [2, 0.5] → side widths in ratio 4:1, product of the
        // per-dim weights is 1 so the box volume stays L^d.
        var (lb, ub) = Make().TrustRegionBounds(new[] { 0.5, 0.5 }, new[] { 4.0, 1.0 }, 0.4);
        double w0 = ub[0] - lb[0], w1 = ub[1] - lb[1];
        (w0 / w1).Should().BeApproximately(4.0, 1e-9, "sides scale linearly with the length scale");
    }

    [TestMethod]
    public void TrustRegionBounds_ClipsToUnitBox()
    {
        // A center near the edge with a wide region clips to [0, 1].
        var (lb, ub) = Make().TrustRegionBounds(new[] { 0.05 }, new[] { 1.0 }, 0.8);
        lb[0].Should().Be(0.0);
        ub[0].Should().Be(0.45);
    }

    // ---- candidate generation ---------------------------------------------------------------------

    [TestMethod]
    public void GenerateCandidates_LowDimension_AllInsideTrustRegion()
    {
        var opt = Make();
        var xCenter = new[] { 0.5, 0.5, 0.5 };
        var (lb, ub) = opt.TrustRegionBounds(xCenter, new[] { 1.0, 1.0, 1.0 }, 0.4);
        var rng = np.random.RandomState();
        rng.seed(123u);
        double[,] cand = opt.GenerateCandidates(xCenter, lb, ub, nCandidates: 64, rng);

        cand.GetLength(0).Should().Be(64);
        for (int i = 0; i < 64; i++)
            for (int j = 0; j < 3; j++)
                // d = 3 ⇒ prob_perturb = 1 ⇒ every coordinate is perturbed into the TR box.
                ((double)cand[i, j]).Should().BeInRange(lb[j], ub[j]);
    }

    [TestMethod]
    public void GenerateCandidates_HighDimension_MasksSomeCoordinatesAtCenter()
    {
        // d = 30 ⇒ prob_perturb = 20/30 ≈ 0.67, so most rows keep some coordinates exactly at the center,
        // and every row still perturbs at least one coordinate.
        int d = 30;
        var opt = Make();
        var xCenter = new double[d];
        for (int j = 0; j < d; j++) xCenter[j] = 0.5;
        var lengthScales = new double[d];
        for (int j = 0; j < d; j++) lengthScales[j] = 1.0;
        var (lb, ub) = opt.TrustRegionBounds(xCenter, lengthScales, 0.4);
        var rng = np.random.RandomState();
        rng.seed(7u);
        double[,] cand = opt.GenerateCandidates(xCenter, lb, ub, nCandidates: 50, rng);

        int rowsWithUnperturbed = 0;
        for (int i = 0; i < 50; i++)
        {
            int perturbed = 0, atCenter = 0;
            for (int j = 0; j < d; j++)
            {
                if (cand[i, j] == 0.5) atCenter++;
                else perturbed++;
            }
            perturbed.Should().BeGreaterThan(0, "every candidate perturbs at least one coordinate");
            if (atCenter > 0) rowsWithUnperturbed++;
        }
        rowsWithUnperturbed.Should().BeGreaterThan(0, "with prob_perturb < 1 some coordinates stay at the center");
    }

    [TestMethod]
    public void GenerateCandidates_SameSeed_IsReproducible()
    {
        var opt = Make();
        var xCenter = new[] { 0.5, 0.5 };
        var (lb, ub) = opt.TrustRegionBounds(xCenter, new[] { 1.0, 1.0 }, 0.4);
        double[,] Draw()
        {
            var rng = np.random.RandomState();
            rng.seed(42u);
            return opt.GenerateCandidates(xCenter, lb, ub, 32, rng);
        }
        double[,] a = Draw(), b = Draw();
        for (int i = 0; i < 32; i++)
            for (int j = 0; j < 2; j++)
                ((double)b[i, j]).Should().Be(a[i, j]);
    }

    // ---- Thompson sampling ------------------------------------------------------------------------

    [TestMethod]
    public void ThompsonArgmax_TinyVariance_PicksMeanArgmax()
    {
        // Huge mean gaps with negligible covariance ⇒ the argmax of the sample is the argmax of the mean.
        var mean = new[] { 0.0, 10.0, 20.0 };
        var cov = new double[3, 3];
        for (int i = 0; i < 3; i++) cov[i, i] = 1e-8;
        var rng = np.random.RandomState();
        rng.seed(1u);
        TurboOptimizer.ThompsonArgmax(mean, cov, rng).Should().Be(2);
    }

    [TestMethod]
    public void ThompsonArgmax_ZeroCovariance_JittersAndSucceeds()
    {
        // A singular (all-zero) covariance must not throw: the jitter ladder makes the Cholesky succeed and
        // the sample reduces to the mean.
        var mean = new[] { 0.0, 10.0, 20.0 };
        var cov = new double[3, 3];
        var rng = np.random.RandomState();
        rng.seed(1u);
        TurboOptimizer.ThompsonArgmax(mean, cov, rng).Should().Be(2);
    }

    [TestMethod]
    public void ThompsonArgmax_SameSeed_IsReproducible()
    {
        var mean = new[] { 0.0, 0.1, 0.2, 0.3 };
        var cov = new double[4, 4];
        for (int i = 0; i < 4; i++) cov[i, i] = 1.0;   // real variance ⇒ the draw matters
        int Draw()
        {
            var rng = np.random.RandomState();
            rng.seed(99u);
            return TurboOptimizer.ThompsonArgmax(mean, cov, rng);
        }
        Draw().Should().Be(Draw());
    }

    // ---- paper claims: trust-region geometry (§2) -------------------------------------------------

    [TestMethod]
    public void TrustRegionBounds_UnitVolume_ProductOfSidesEqualsLPowD()
    {
        // §2: side_i = λ_i·L/(Πλ)^{1/d}, so ∏ side_i = L^d exactly for any length scales — the box rescales
        // its SHAPE by the length scales but keeps total VOLUME at L^d. Center + L chosen so no side clips.
        double l = 0.2;
        var (lb, ub) = Make().TrustRegionBounds(new[] { 0.5, 0.5, 0.5 }, new[] { 4.0, 1.0, 2.0 }, l);
        double volume = 1.0;
        for (int j = 0; j < 3; j++) volume *= ub[j] - lb[j];
        volume.Should().BeApproximately(Math.Pow(l, 3), 1e-12, "the length-scale-weighted box has volume L^d");
    }

    [TestMethod]
    public void TrustRegionBounds_LargeLength_CoversWholeUnitBox()
    {
        // §2: "If L was large enough for the TR to contain the whole space, this would be equivalent to
        // running standard global BO." A large base length clips the region to the entire unit cube.
        var (lb, ub) = Make().TrustRegionBounds(new[] { 0.3, 0.7, 0.5 }, new[] { 1.0, 1.0, 1.0 }, length: 100.0);
        for (int j = 0; j < 3; j++)
        {
            lb[j].Should().Be(0.0);
            ub[j].Should().Be(1.0);
        }
    }

    // ---- paper claims: success is a STRICT improvement (§D) ---------------------------------------

    [TestMethod]
    public void Fold_StrictImprovement_CountsTieAsFailureAndTinyGainAsSuccess()
    {
        // §D: a "success" is a candidate that (strictly) improves the incumbent. With the paper's strict
        // margin (successTol = 0) any positive gain is a success and an exact tie is a failure; the BoTorch
        // tutorial's 1e-3 RELATIVE margin instead rejects a sub-0.1%·|best| gain. A +0.05% gain straddles the
        // two, so it proves both the strict default and the margin knob.
        var strict = new TurboOptimizer(0.8, 1.0 / 128.0, 1.6, tauSucc: 3, tauFail: 4, successTol: 0.0);
        var botorch = new TurboOptimizer(0.8, 1.0 / 128.0, 1.6, tauSucc: 3, tauFail: 4, successTol: 1e-3);

        // init [0, 1] ⇒ incumbent best = 1; then a +0.0005 gain (0.05% of |best|).
        var gain = new double[] { 0.0, 1.0, 1.0005 };
        strict.ReconstructState(gain, nInit: 2).SuccessCounter.Should()
            .Be(1, "the strict margin counts any positive improvement as a success");
        botorch.ReconstructState(gain, nInit: 2).FailureCounter.Should()
            .Be(1, "a 1e-3 relative margin rejects a 0.05% gain");

        // An exact tie never strictly improves x⋆, so it is a failure even under the strict margin.
        var tie = new double[] { 0.0, 1.0, 1.0 };
        strict.ReconstructState(tie, nInit: 2).FailureCounter.Should()
            .Be(1, "a tie does not strictly improve the incumbent");
    }

    // ---- paper claims: candidate perturbation rate min(20/d, 1) (§D) ------------------------------

    [TestMethod]
    public void GenerateCandidates_PerturbationFraction_MatchesProbability()
    {
        // §D: each coordinate takes the Sobol value with probability min(20/d, 1) and the center otherwise.
        // At d = 25 that is 0.8, so ~80% of coordinates across many candidates differ from the center.
        int d = 25;
        var opt = Make();
        var xCenter = new double[d];
        var ls = new double[d];
        for (int j = 0; j < d; j++) { xCenter[j] = 0.5; ls[j] = 1.0; }
        var (lb, ub) = opt.TrustRegionBounds(xCenter, ls, 0.4);
        var rng = np.random.RandomState();
        rng.seed(2024u);

        const int n = 400;
        double[,] cand = opt.GenerateCandidates(xCenter, lb, ub, n, rng);
        int perturbed = 0;
        for (int i = 0; i < n; i++)
            for (int j = 0; j < d; j++)
                if (cand[i, j] != 0.5) perturbed++;   // held-at-center coordinates equal 0.5 exactly

        ((double)perturbed / (n * d)).Should()
            .BeApproximately(20.0 / d, 0.03, "the empirical perturbation rate matches min(20/d, 1)");
    }

    // ---- paper claims: Thompson sampling draws the joint posterior (§E) ---------------------------

    [TestMethod]
    public void ThompsonArgmax_DrawsExactJointPosteriorSample()
    {
        // §E: TS draws ONE realization of the GP posterior over the candidate set — the JOINT sample
        // f = mean + L·z with cov = L·Lᵀ, z ~ N(0, I) — not independent per-candidate marginals. Replicate
        // the draw from the same seed (z = randn(n); L = chol(cov + 1e-10·I), the first jitter level a
        // well-conditioned cov needs) and confirm ThompsonArgmax returns argmax(mean + L·z) exactly.
        var mean = new[] { 0.0, 0.0, 0.0 };
        var cov = new[,]
        {
            { 1.0, 0.9, 0.8 },
            { 0.9, 1.0, 0.9 },
            { 0.8, 0.9, 1.0 },
        };
        const uint seed = 20240517u;

        var expectRng = np.random.RandomState();
        expectRng.seed(seed);
        double[] z = expectRng.randn(3).ToArray<double>();
        var jittered = (double[,])cov.Clone();
        for (int i = 0; i < 3; i++) jittered[i, i] += 1e-10;
        double[] lz = LinAlg.Dot(LinAlg.Cholesky(jittered), z);
        int expected = 0;
        for (int i = 1; i < 3; i++) if (mean[i] + lz[i] > mean[expected] + lz[expected]) expected = i;

        var rng = np.random.RandomState();
        rng.seed(seed);
        TurboOptimizer.ThompsonArgmax(mean, cov, rng).Should()
            .Be(expected, "the Thompson pick is the argmax of the exact joint posterior draw mean + L·z");
    }

    [TestMethod]
    public void ThompsonArgmax_IsStochasticNotGreedy()
    {
        // §E: TS is stochastic — with real posterior variance it does NOT always pick the highest-mean
        // candidate (that would be greedy exploitation), yet the higher-mean candidate still wins more often.
        var mean = new[] { 0.0, 0.15 };
        var cov = new[,] { { 1.0, 0.0 }, { 0.0, 1.0 } };
        int pick0 = 0, pick1 = 0;
        for (uint s = 1; s <= 300; s++)
        {
            var rng = np.random.RandomState();
            rng.seed(s);
            if (TurboOptimizer.ThompsonArgmax(mean, cov, rng) == 0) pick0++; else pick1++;
        }
        pick1.Should().BeGreaterThan(pick0, "the higher-mean candidate wins more often");
        pick0.Should().BeGreaterThan(0, "but Thompson sampling still explores the lower-mean candidate");
    }

    // ---- paper claims: trust-region center rule (§2) ----------------------------------------------

    [TestMethod]
    public void SelectCenterIndex_Deterministic_PicksBestObservation()
    {
        // §2 noise-free case: center on the best OBSERVATION, ignoring the posterior mean entirely.
        var observed = new[] { 1.0, 3.0, 2.0 };   // best observation is index 1
        var postMean = new[] { 5.0, 0.0, 0.0 };   // posterior mean would prefer index 0
        TurboOptimizer.SelectCenterIndex(observed, postMean, deterministic: true).Should().Be(1);
        TurboOptimizer.SelectCenterIndex(observed, null, deterministic: true).Should().Be(1);
    }

    [TestMethod]
    public void SelectCenterIndex_Noisy_PicksBestPosteriorMean()
    {
        // §2 noisy case: center on the best POSTERIOR MEAN, so a spuriously good single observation (index 1)
        // cannot anchor the region — the GP-smoothed mean (index 2) wins instead.
        var observed = new[] { 1.0, 9.0, 2.0 };   // index 1 is a lucky outlier
        var postMean = new[] { 1.1, 1.9, 2.1 };   // the GP mean prefers index 2
        TurboOptimizer.SelectCenterIndex(observed, postMean, deterministic: false).Should().Be(2);
    }

    [TestMethod]
    public void SelectCenterIndex_Noisy_RequiresPosteriorMean()
    {
        Action act = () => TurboOptimizer.SelectCenterIndex(new[] { 1.0, 2.0 }, null, deterministic: false);
        act.Should().Throw<ArgumentException>("the noisy center rule needs the posterior mean");
    }

    // ---- constructor validation -------------------------------------------------------------------

    [TestMethod]
    public void Constructor_RejectsIllOrderedLengths()
    {
        Action act = () => new TurboOptimizer(0.8, lengthMin: 1.0, lengthMax: 1.6, tauSucc: 3, tauFail: 4, successTol: 1e-3);
        act.Should().Throw<ArgumentException>("lengthInit must be ≥ lengthMin");
    }
}
