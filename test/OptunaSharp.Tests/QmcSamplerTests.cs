using System;
using System.Collections.Generic;
using System.Linq;
using AwesomeAssertions;
using NumSharp;
using OptunaSharp.Distributions;
using OptunaSharp.Internal;
using OptunaSharp.Samplers;

namespace OptunaSharp.Tests;

/// <summary>
///     Unit coverage for the QMC sampler (PORTING_PLAN Phase 2.12). The engine assertions are pinned
///     to values produced by <c>scipy.stats.qmc</c> (non-scrambled Sobol' / Halton), which the port
///     reproduces bit-for-bit; the sampler assertions mirror Optuna's <c>tests/samplers_tests/test_qmc.py</c>.
/// </summary>
[TestClass]
public class QmcSamplerTests
{
    // ---- engines: bit-exact vs scipy ----------------------------------------------------------

    [TestMethod]
    public void Sobol_FirstPoints_MatchScipy()
    {
        var expected = new[]
        {
            new[] { 0.0, 0.0, 0.0 }, new[] { 0.5, 0.5, 0.5 }, new[] { 0.75, 0.25, 0.25 },
            new[] { 0.25, 0.75, 0.75 }, new[] { 0.375, 0.375, 0.625 }, new[] { 0.875, 0.875, 0.125 },
            new[] { 0.625, 0.125, 0.875 }, new[] { 0.125, 0.625, 0.375 },
        };
        for (ulong n = 0; n < (ulong)expected.Length; n++)
            QuasiMonteCarlo.SobolPoint(3, n).Should().Equal(expected[n]);
    }

    [TestMethod]
    public void Halton_FirstPoints_MatchScipy()
    {
        var expected = new[]
        {
            new[] { 0.0, 0.0, 0.0 },
            new[] { 0.5, 1.0 / 3.0, 0.2 },
            new[] { 0.25, 2.0 / 3.0, 0.4 },
            new[] { 0.75, 1.0 / 9.0, 0.6000000000000001 },
            new[] { 0.125, 4.0 / 9.0, 0.8 },
        };
        for (ulong n = 0; n < (ulong)expected.Length; n++)
            QuasiMonteCarlo.HaltonPoint(3, n).Should().Equal(expected[n]);
    }

    [TestMethod]
    public void Sobol_HigherDimAndIndex_MatchScipy()
    {
        QuasiMonteCarlo.SobolPoint(5, 100).Should().Equal(
            0.4140625, 0.2578125, 0.7734375, 0.7265625, 0.8828125);
        QuasiMonteCarlo.SobolPoint(8, 255).Should().Equal(
            0.00390625, 0.99609375, 0.76953125, 0.57421875, 0.61328125, 0.98046875, 0.88671875, 0.17578125);
    }

    [TestMethod]
    public void Halton_HigherDimAndIndex_MatchScipy()
    {
        QuasiMonteCarlo.HaltonPoint(4, 50).Should().Equal(
            0.296875, 0.8641975308641974, 0.016, 0.14577259475218657);
        QuasiMonteCarlo.HaltonPoint(6, 99).Should().Equal(
            0.7734375, 0.07818930041152262, 0.9840000000000001, 0.14868804664723032, 0.0743801652892562, 0.6568047337278107);
    }

    [TestMethod]
    public void Sobol_InvalidDimensionOrIndex_Throws()
    {
        ((Action)(() => QuasiMonteCarlo.SobolPoint(0, 0))).Should().Throw<ValueError>();
        ((Action)(() => QuasiMonteCarlo.SobolPoint(1025, 0))).Should().Throw<ValueError>();
        ((Action)(() => QuasiMonteCarlo.SobolPoint(2, 1UL << 30))).Should().Throw<ValueError>();
        QuasiMonteCarlo.SobolPoint(1024, 0).Should().HaveCount(1024);  // MaxDim is valid.
    }

    // ---- sampler behavior ---------------------------------------------------------------------

    [TestMethod]
    public void Scramble_ThrowsNotImplemented()
    {
        var act = () => new QMCSampler(scramble: true);
        act.Should().Throw<NotImplementedException>().WithMessage("*scramble*");
    }

    [TestMethod]
    public void InvalidQmcType_ThrowsValueError()
    {
        var act = () => new QMCSampler(qmcType: "latin");
        act.Should().Throw<ValueError>().WithMessage("*halton*sobol*");
    }

    [TestMethod]
    public void Construction_EmitsExperimentalWarning()
    {
        string? captured = null;
        void Handler(string msg, WarningCategory cat)
        {
            if (cat == WarningCategory.ExperimentalWarning && msg.Contains("QMCSampler"))
                captured = msg;
        }

        OptunaWarnings.WarningRaised += Handler;
        try
        {
            _ = new QMCSampler(seed: 0);
        }
        finally
        {
            OptunaWarnings.WarningRaised -= Handler;
        }

        captured.Should().NotBeNull();
        captured.Should().Contain("experimental");
    }

    [DataTestMethod]
    [DataRow("sobol")]
    [DataRow("halton")]
    public void Reproducible_SameSeed_ProducesIdenticalTrials(string qmcType)
    {
        List<double> Run()
        {
            var study = Optuna.CreateStudy(sampler: new QMCSampler(qmcType: qmcType, seed: 3));
            study.Optimize(t =>
            {
                double x = t.SuggestFloat("x", -1, 1);
                long y = t.SuggestInt("y", -5, 5);
                return Math.Pow(x, 2.0) + y;
            }, nTrials: 16);
            return study.Trials.Select(t => (double)t.Params["x"]).ToList();
        }

        Run().Should().Equal(Run());
    }
}
