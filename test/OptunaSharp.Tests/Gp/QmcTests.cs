using System;
using AwesomeAssertions;
using OptunaSharp.Gp;
using OptunaSharp.Internal;

namespace OptunaSharp.Tests.Gp;

/// <summary>
///     Unit coverage for the GP QMC normal Sobol' sampler (PORTING_PLAN Phase 3.2 part 3e-3 —
///     <c>optuna/_gp/qmc.py</c>): the classic MT19937 (<see cref="TorchMt19937" />, torch's CPU
///     generator), the cephes inverse-normal-CDF <see cref="TruncNorm.Ndtri" />, and
///     <see cref="Qmc.SampleFromNormalSobol" />. <c>ndtri</c> scalars are pinned to
///     <c>scipy.special.ndtri</c>; the MT19937 bit stream is pinned to live torch's
///     <c>randint(2, …)</c>. Byte-tolerance parity of the full pipeline vs live torch lives in the
///     interop <c>GpNormalSobolParityTests</c>.
/// </summary>
[TestClass]
public class QmcTests
{
    [TestMethod]
    public void Ndtri_MatchesScipy()
    {
        TruncNorm.Ndtri(0.5).Should().Be(0.0);
        TruncNorm.Ndtri(0.7).Should().BeApproximately(0.5244005127080407, 1e-15);
        TruncNorm.Ndtri(0.3).Should().BeApproximately(-0.5244005127080409, 1e-15);
        TruncNorm.Ndtri(0.9).Should().BeApproximately(1.2815515655446004, 1e-15);
        TruncNorm.Ndtri(0.135335).Should().BeApproximately(-1.101520930808415, 1e-14);
        TruncNorm.Ndtri(0.05).Should().BeApproximately(-1.6448536269514729, 1e-14);
        TruncNorm.Ndtri(0.001).Should().BeApproximately(-3.090232306167813, 1e-13);
        TruncNorm.Ndtri(0.999).Should().BeApproximately(3.090232306167813, 1e-13);
    }

    [TestMethod]
    public void Ndtri_Endpoints_AreInfinite()
    {
        double.IsNegativeInfinity(TruncNorm.Ndtri(0.0)).Should().BeTrue();
        double.IsPositiveInfinity(TruncNorm.Ndtri(1.0)).Should().BeTrue();
        // ndtri is antisymmetric about 0.5.
        TruncNorm.Ndtri(0.25).Should().BeApproximately(-TruncNorm.Ndtri(0.75), 1e-15);
    }

    [TestMethod]
    public void TorchMt19937_LowBits_MatchLiveTorch_Seed42()
    {
        // Pinned from torch.randint(2, (20,), generator=Generator().manual_seed(42)).
        var expected = new uint[] { 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0 };
        var mt = new TorchMt19937(42);
        for (int i = 0; i < expected.Length; i++)
            mt.NextBit().Should().Be(expected[i], $"bit {i}");
    }

    [TestMethod]
    public void SampleFromNormalSobol_Deterministic_AndShaped()
    {
        double[,] a = Qmc.SampleFromNormalSobol(dim: 3, nSamples: 16, seed: 7);
        double[,] b = Qmc.SampleFromNormalSobol(dim: 3, nSamples: 16, seed: 7);
        a.GetLength(0).Should().Be(16);
        a.GetLength(1).Should().Be(3);
        for (int i = 0; i < 16; i++)
            for (int j = 0; j < 3; j++)
                a[i, j].Should().Be(b[i, j]);

        // A different seed gives a different scramble.
        double[,] c = Qmc.SampleFromNormalSobol(dim: 3, nSamples: 16, seed: 8);
        bool anyDifferent = false;
        for (int i = 0; i < 16 && !anyDifferent; i++)
            for (int j = 0; j < 3; j++)
                if (a[i, j] != c[i, j]) { anyDifferent = true; break; }
        anyDifferent.Should().BeTrue();
    }

    [TestMethod]
    public void SampleFromNormalSobol_IsApproximatelyStandardNormal()
    {
        // 512 QMC normal points in 1-D: the empirical mean/std should be close to N(0,1).
        double[,] s = Qmc.SampleFromNormalSobol(dim: 1, nSamples: 512, seed: 0);
        double mean = 0.0;
        for (int i = 0; i < 512; i++) mean += s[i, 0];
        mean /= 512;
        double var = 0.0;
        for (int i = 0; i < 512; i++) var += (s[i, 0] - mean) * (s[i, 0] - mean);
        var /= 512;
        mean.Should().BeApproximately(0.0, 0.05);
        Math.Sqrt(var).Should().BeApproximately(1.0, 0.1);
    }
}
