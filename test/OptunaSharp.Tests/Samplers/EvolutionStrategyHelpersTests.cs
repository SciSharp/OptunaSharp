using System;
using System.Linq;
using AwesomeAssertions;
using NumSharp;
using OptunaSharp.Samplers.EvolutionStrategy;

namespace OptunaSharp.Tests.Samplers;

/// <summary>
///     Tests for the shared Natural-Evolution-Strategy building blocks — the rank-based fitness-shaping
///     surrogates (<see cref="EsFitnessShaping" />) and the deterministic Gaussian sampling
///     (<see cref="EsSampling" />) — on which the OpenAI-ES, SNES, and xNES optimizers all rely. These are
///     pure, deterministic functions, so the assertions pin exact values (the properties the papers require:
///     centered ranks span <c>[-0.5, 0.5]</c>, NES utilities sum to zero and decrease with rank, and the
///     noise draw is byte-reproducible).
/// </summary>
[TestClass]
public class EvolutionStrategyHelpersTests
{
    // ---- centered ranks (OpenAI-ES) ---------------------------------------------------------------

    [TestMethod]
    public void CenteredRanks_MapsToSymmetricInterval()
    {
        // Scores [3,1,2,0]: best (3) → +0.5, worst (0) → −0.5, evenly spaced in between. The expected
        // values reproduce the implementation's arithmetic (rank/(n−1) − 0.5) so the comparison is exact
        // rather than off by a ULP (1.0/3.0 − 0.5 ≠ −1.0/6.0 in IEEE double).
        double[] y = EsFitnessShaping.CenteredRanks(new double[] { 3, 1, 2, 0 });
        y.Should().Equal(0.5, 1.0 / 3.0 - 0.5, 2.0 / 3.0 - 0.5, -0.5);
    }

    [TestMethod]
    public void CenteredRanks_SumToZero()
    {
        double[] y = EsFitnessShaping.CenteredRanks(new double[] { 5, -2, 0, 9, 1, 1 });
        y.Sum().Should().BeApproximately(0.0, 1e-12, "centered ranks are a built-in baseline (sum zero)");
    }

    [TestMethod]
    public void CenteredRanks_SingleElement_IsNeutralZero()
    {
        // A population of one carries no ordering information (the n−1 denominator would divide by zero).
        EsFitnessShaping.CenteredRanks(new double[] { 42.0 }).Should().Equal(0.0);
    }

    [TestMethod]
    public void CenteredRanks_NaNSinksToWorst()
    {
        // NaN must sort as the smallest (worst) score without poisoning the comparison.
        double[] y = EsFitnessShaping.CenteredRanks(new double[] { 1.0, double.NaN, 2.0 });
        y[1].Should().Be(-0.5, "NaN takes the worst centered rank");
        y[2].Should().Be(0.5, "the largest finite score takes the best centered rank");
    }

    // ---- NES utilities (SNES / xNES) --------------------------------------------------------------

    [TestMethod]
    public void NesUtilities_SumToZeroAndDecreaseWithRank()
    {
        double[] u = EsFitnessShaping.NesUtilities(8);
        u.Sum().Should().BeApproximately(0.0, 1e-12, "the utility baseline sums to zero");
        for (int i = 1; i < u.Length; i++)
            u[i].Should().BeLessThanOrEqualTo(u[i - 1], "utilities are monotonically decreasing in rank");
        u[0].Should().BeGreaterThan(0, "the best rank carries positive weight");
        u[^1].Should().BeLessThan(0, "the worst rank carries negative weight after the 1/λ shift");
    }

    [TestMethod]
    public void NesUtilities_OnlyBetterHalfIsPositiveBeforeBaseline()
    {
        // The raw max(0, …) weights zero out worse-than-median ranks; after the −1/λ shift those become
        // exactly −1/λ. For λ = 10 the bottom half should sit at −0.1.
        double[] u = EsFitnessShaping.NesUtilities(10);
        u[^1].Should().BeApproximately(-0.1, 1e-12, "a zero raw weight becomes −1/λ");
    }

    [TestMethod]
    public void NesUtilities_RejectsNonPositiveLambda()
    {
        Action act = () => EsFitnessShaping.NesUtilities(0);
        act.Should().Throw<ArgumentOutOfRangeException>();
    }

    // ---- deterministic Gaussian sampling ----------------------------------------------------------

    [TestMethod]
    public void Combine_IsDeterministicAndGenerationSensitive()
    {
        EsSampling.Combine(1234u, 5).Should().Be(EsSampling.Combine(1234u, 5), "the mixer is a pure function");
        EsSampling.Combine(1234u, 5).Should().NotBe(EsSampling.Combine(1234u, 6),
            "consecutive generations must get de-correlated seeds");
    }

    [TestMethod]
    public void StandardNormalMatrix_IsRowMajorAndByteExactWithNumpy()
    {
        // The reshape must be a pure row-major view split of one flat draw — so a rows×cols matrix equals
        // the same-seed flat randn(rows*cols) laid out row 0 first.
        var rng1 = np.random.RandomState();
        rng1.seed(12345u);
        double[][] m = EsSampling.StandardNormalMatrix(rng1, 2, 3);

        var rng2 = np.random.RandomState();
        rng2.seed(12345u);
        double[] flat = rng2.randn(6).ToArray<double>();
        m[0].Should().Equal(flat[0], flat[1], flat[2]);
        m[1].Should().Equal(flat[3], flat[4], flat[5]);

        // Tie the underlying MT19937 stream to numpy: np.random.RandomState(12345).randn(6)[0] ≈ −0.2047077.
        m[0][0].Should().BeApproximately(-0.20470766, 1e-7, "NumSharp randn is byte-compatible with numpy");
    }
}
