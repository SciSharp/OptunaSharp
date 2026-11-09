using System;
using System.Collections.Generic;
using System.Linq;
using AwesomeAssertions;
using NumSharp;
using OptunaSharp.Collections;
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
    public void ScrambledHalton_FirstPoints_MatchScipy()
    {
        // scipy.stats.qmc.Halton(3, seed=12345, scramble=True).fast_forward(n).random(1), n = 0..3.
        var expected = new[]
        {
            new[] { 0.1533356327231844, 0.5925589076600479, 0.8245656388940836 },
            new[] { 0.6533356327231844, 0.9258922409933812, 0.42456563889408355 },
            new[] { 0.4033356327231844, 0.25922557432671445, 0.024565638894083558 },
            new[] { 0.9033356327231844, 0.48144779654893666, 0.2245656388940835 },
        };
        for (ulong n = 0; n < (ulong)expected.Length; n++)
            ScrambledHalton.Point(3, n, 12345).Should().Equal(expected[n]);
    }

    [TestMethod]
    public void ScrambledHalton_HigherDimAndIndex_MatchScipy()
    {
        // scipy.stats.qmc.Halton(5, seed=12345, scramble=True).fast_forward(100).random(1).
        ScrambledHalton.Point(5, 100, 12345).Should().Equal(
            0.004898132723184401, 0.9917358623925582, 0.8325656388940836, 0.2329383215209358, 0.7500238228429962);
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
    public void HaltonScramble_Constructs()
    {
        // Scrambled Halton (Owen digit permutations) is now ported bit-exact (Internal/ScrambledHalton),
        // closing the last Phase 2.12 combination — construction no longer throws.
        var act = () => new QMCSampler(qmcType: "halton", scramble: true, seed: 0);
        act.Should().NotThrow();
    }

    [TestMethod]
    public void SobolScramble_Constructs()
    {
        var act = () => new QMCSampler(qmcType: "sobol", scramble: true, seed: 0);
        act.Should().NotThrow();
    }

    [TestMethod]
    public void ScrambledSobol_Trajectory_MatchesOptuna()
    {
        // Pinned to live optuna QMCSampler(qmc_type='sobol', scramble=True, seed=42): trial 0 is the
        // independent RandomSampler(42) draw, trials 1..7 the scrambled Sobol' relative samples.
        var study = Optuna.CreateStudy(
            sampler: new QMCSampler(qmcType: "sobol", scramble: true, seed: 42, warnIndependentSampling: false));
        study.Optimize(t =>
        {
            double x = t.SuggestFloat("x", -1, 1);
            long y = t.SuggestInt("y", -1, 3);
            t.SuggestCategorical("z", new CategoricalChoice[] { "a", "b", "c" });
            return Math.Pow(x, 2.0) + y;
        }, nTrials: 8);

        study.Trials.Select(t => (double)t.Params["x"]).Should().Equal(
            -0.250919762305275, -0.13794105127453804, 0.7396270707249641, 0.15583090111613274,
            -0.7262669429183006, -0.9132113959640265, 0.4677744638174772, 0.92768351174891);
    }

    [TestMethod]
    public void ScrambledSobol_SameSeed_ProducesIdenticalTrials()
    {
        List<double> Run()
        {
            var study = Optuna.CreateStudy(
                sampler: new QMCSampler(qmcType: "sobol", scramble: true, seed: 11, warnIndependentSampling: false));
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

    [TestMethod]
    public void ScrambledSobol_DiffersFromUnscrambled()
    {
        List<double> Run(bool scramble)
        {
            var study = Optuna.CreateStudy(
                sampler: new QMCSampler(qmcType: "sobol", scramble: scramble, seed: 5, warnIndependentSampling: false));
            study.Optimize(t => t.SuggestFloat("x", -1, 1), nTrials: 8);
            // Trial 0 is the (seed-shared) independent sampler for both; the QMC relative samples differ.
            return study.Trials.Skip(1).Select(t => (double)t.Params["x"]).ToList();
        }

        Run(scramble: true).Should().NotEqual(Run(scramble: false));
    }

    [TestMethod]
    public void ScrambledHalton_Trajectory_MatchesOptuna()
    {
        // Pinned to live optuna QMCSampler(qmc_type='halton', scramble=True, seed=42): trial 0 is the
        // independent RandomSampler(42) draw (identical to the scrambled-Sobol trajectory, shared seed),
        // trials 1..7 the scrambled Halton relative samples.
        var study = Optuna.CreateStudy(
            sampler: new QMCSampler(qmcType: "halton", scramble: true, seed: 42, warnIndependentSampling: false));
        study.Optimize(t =>
        {
            double x = t.SuggestFloat("x", -1, 1);
            long y = t.SuggestInt("y", -1, 3);
            t.SuggestCategorical("z", new CategoricalChoice[] { "a", "b", "c" });
            return Math.Pow(x, 2.0) + y;
        }, nTrials: 8);

        study.Trials.Select(t => (double)t.Params["x"]).Should().Equal(
            -0.250919762305275, 0.10261173898301967, -0.8973882610169803, 0.6026117389830197,
            -0.3973882610169803, 0.3526117389830197, -0.6473882610169803, 0.8526117389830197);
    }

    [TestMethod]
    public void ScrambledHalton_SameSeed_ProducesIdenticalTrials()
    {
        List<double> Run()
        {
            var study = Optuna.CreateStudy(
                sampler: new QMCSampler(qmcType: "halton", scramble: true, seed: 11, warnIndependentSampling: false));
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

    [TestMethod]
    public void ScrambledHalton_DiffersFromUnscrambled()
    {
        List<double> Run(bool scramble)
        {
            var study = Optuna.CreateStudy(
                sampler: new QMCSampler(qmcType: "halton", scramble: scramble, seed: 5, warnIndependentSampling: false));
            study.Optimize(t => t.SuggestFloat("x", -1, 1), nTrials: 8);
            // Trial 0 is the (seed-shared) independent sampler for both; the QMC relative samples differ.
            return study.Trials.Skip(1).Select(t => (double)t.Params["x"]).ToList();
        }

        Run(scramble: true).Should().NotEqual(Run(scramble: false));
    }

    [TestMethod]
    public void InvalidQmcType_ThrowsValueError()
    {
        var act = () => new QMCSampler(qmcType: "latin");
        act.Should().Throw<ValueError>().WithMessage("*halton*sobol*");
    }

    // [Misaligned] Optuna decorates QMCSampler with @experimental_class("3.0.0"), which raises an
    // ExperimentalWarning on construction. OptunaSharp targets the Optuna 5.x surface and treats this
    // sampler as stable, so no ExperimentalWarning is emitted — this guards that deliberate divergence.
    [TestMethod]
    [Misaligned]
    public void Construction_DoesNotEmitExperimentalWarning()
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

        captured.Should().BeNull();
    }

    /// <summary>
    ///     The six numerical distributions Optuna's <c>RelativeSamplerTestCase.test_sample_relative_numerical</c>
    ///     crosses for both parameters (<c>optuna/testing/pytest_samplers.py</c>).
    /// </summary>
    /// <returns>Fresh distribution instances in Optuna's parametrize order.</returns>
    private static BaseDistribution[] RelativeNumericalDistributions() => new BaseDistribution[]
    {
        new FloatDistribution(-1.0, 1.0),
        new FloatDistribution(1e-7, 1.0, log: true),
        new FloatDistribution(-10, 10, step: 0.5),
        new IntDistribution(3, 10),
        new IntDistribution(1, 100, log: true),
        new IntDistribution(3, 9, step: 2),
    };

    /// <summary>
    ///     Port of the coverage Optuna 5.0.0 restored for <c>QMCSampler</c> (#6799 adds it to
    ///     <c>TestRelativeSampler</c>): for every pair of numerical distributions, each relative sample lies
    ///     inside its distribution and keeps the distribution's numeric type. The first Sobol' point is the
    ///     origin, so a log float lands exactly on its lower edge — the case that came back one ULP below
    ///     <c>low</c> before <see cref="SearchSpaceTransform" /> clamped both ends.
    /// </summary>
    [TestMethod]
    public void RelativeSamples_StayInsideEveryNumericalDistribution()
    {
        foreach (BaseDistribution xDistribution in RelativeNumericalDistributions())
        foreach (BaseDistribution yDistribution in RelativeNumericalDistributions())
        {
            var searchSpace = new OrderedDict<string, BaseDistribution>
            {
                ["x"] = xDistribution,
                ["y"] = yDistribution,
            };
            var sampler = new QMCSampler(seed: 0, warnIndependentSampling: false);
            var study = Optuna.CreateStudy(sampler: sampler);

            // One completed trial over the same space, exactly like the Optuna test's ask/tell preamble.
            Trial first = study.Ask(searchSpace);
            study.Tell(first, Convert.ToDouble(first.Params["x"]) + Convert.ToDouble(first.Params["y"]));

            // Optuna samples 10 points for the bounds check plus one more for the type check; each call
            // advances the QMC sample id, so the first call is the Sobol' origin (every lower edge).
            for (int k = 0; k < 11; k++)
            {
                int trialId = study.Storage.CreateNewTrial(study.StudyId);
                OrderedDict<string, object?> sampled =
                    sampler.SampleRelative(study, study.Storage.GetTrial(trialId), searchSpace);
                foreach (var (name, distribution) in searchSpace)
                {
                    string because = $"{name} ~ {distribution}, sample {k}";
                    switch (distribution)
                    {
                        case FloatDistribution f:
                            sampled[name].Should().BeOfType<double>(because);
                            ((double)sampled[name]!).Should().BeInRange(f.Low, f.High, because);
                            break;
                        case IntDistribution i:
                            sampled[name].Should().BeOfType<long>(because);
                            ((long)sampled[name]!).Should().BeInRange(i.Low, i.High, because);
                            break;
                    }
                }
            }
        }
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

    /// <summary>
    ///     Plan 14: the sample-id attribute key is computed once per search space and reused while the same names come with
    ///     the very same distribution objects. A memoized key must always equal the key a fresh sampler computes: for the
    ///     same objects again (reused), for equal-by-value objects that print differently (Python equality makes
    ///     <c>True == 1</c>, but <c>str()</c> differs, and so must the key), for a reordered space, and back again.
    /// </summary>
    /// <param name="scramble">Whether the engine is scrambled (the seed then enters the key).</param>
    [TestMethod]
    [DataRow(false)]
    [DataRow(true)]
    public void QmcIdKey_Memoized_EqualsAFreshComputation(bool scramble)
    {
        var x = new FloatDistribution(-5, 5);
        var flags = new CategoricalDistribution(new CategoricalChoice[] { true, false });
        var ints = new CategoricalDistribution(new CategoricalChoice[] { 1L, 0L });
        flags.Equals(ints).Should().BeTrue("Python equality: True == 1 and False == 0");
        var spaceFlags = new OrderedDict<string, BaseDistribution> { ["x"] = x, ["c"] = flags };
        var spaceInts = new OrderedDict<string, BaseDistribution> { ["x"] = x, ["c"] = ints };
        var reordered = new OrderedDict<string, BaseDistribution> { ["c"] = flags, ["x"] = x };

        string Fresh(OrderedDict<string, BaseDistribution> space) => new QMCSampler(scramble: scramble, seed: 3).QmcIdKey(space);

        var memo = new QMCSampler(scramble: scramble, seed: 3);
        string first = memo.QmcIdKey(spaceFlags);
        first.Should().Be(Fresh(spaceFlags));
        memo.QmcIdKey(spaceFlags).Should().Be(first, "the same objects reuse the key");
        string ints2 = memo.QmcIdKey(spaceInts);
        ints2.Should().Be(Fresh(spaceInts));
        ints2.Should().NotBe(first, "equal choices that print differently give a different key, as in Optuna");
        memo.QmcIdKey(reordered).Should().Be(Fresh(reordered));
        memo.QmcIdKey(spaceFlags).Should().Be(first);
        first.Should().MatchRegex("^qmc:[0-9a-f]{64}$");
    }
}
