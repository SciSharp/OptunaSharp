using System;
using System.Collections.Generic;
using System.Linq;
using AwesomeAssertions;
using NumSharp;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Internal;
using OptunaSharp.Samplers;
using OptunaSharp.Samplers.Tpe;
using OptunaSharp.Storages;

namespace OptunaSharp.Tests;

/// <summary>
///     Unit coverage for the TPE sampler (PORTING_PLAN Phase 2.8). Numeric assertions are pinned to
///     values produced by running real Optuna / SciPy (the reference for
///     <c>_truncnorm</c>/<c>probability_distributions</c>); structural assertions mirror Optuna's own
///     <c>tests/samplers_tests/tpe_tests</c>. Values routed through <see cref="Erf" /> are asserted to
///     a small tolerance (the documented erf/ucrt divergence), everything else bit-for-bit.
/// </summary>
[TestClass]
public class TpeSamplerTests
{
    private const double ErfTol = 1e-12;

    // ---- numeric primitives (bit-exact vs Python) ---------------------------------------------

    [TestMethod]
    public void Expm1_MatchesNumpy()
    {
        MathUtil.Expm1(-0.005).Should().Be(-0.004987520807317687);
        MathUtil.Expm1(-0.7).Should().Be(-0.5034146962085905);
        MathUtil.Expm1(-1e-12).Should().Be(-9.999999999995e-13);
        MathUtil.Expm1(0.0).Should().Be(0.0);
    }

    [TestMethod]
    public void LogAddExp_MatchesNumpy()
    {
        MathUtil.LogAddExp(-1.0, -2.0).Should().Be(-0.6867383124817772);
        MathUtil.LogAddExp(-700.0, -701.0).Should().Be(-699.6867383124818);
        MathUtil.LogAddExp(-1e-5, -1e-5).Should().Be(0.6931371805599453);
        MathUtil.LogAddExp(0.0, 0.0).Should().Be(0.6931471805599453);
    }

    [TestMethod]
    public void NdtriExp_MatchesScipy_WithinErfTolerance()
    {
        // optuna _ndtri_exp — its Newton step reads _log_ndtr → the scalar erf, so it agrees with
        // Python to the erf/ucrt tolerance (a ULP or two), not bit-for-bit.
        var y = new[] { -1e-3, -0.5, -3.0, -8.0, -50.0 };
        var expected = new[]
        {
            3.0903807869170454, 0.27028802073873576, -1.646921720527715, -3.401192656144666, -9.674825283612357,
        };
        var got = TruncNorm.NdtriExp(y);
        for (int i = 0; i < y.Length; i++)
            got[i].Should().BeApproximately(expected[i], ErfTol);
    }

    [TestMethod]
    public void Ppf_MatchesScipy_WithinErfTolerance()
    {
        var q = new[] { 0.1, 0.5, 0.9, 0.01, 0.99 };
        var a = new[] { -2.0, -2.0, -2.0, -2.0, -2.0 };
        var b = new[] { 2.0, 2.0, 2.0, 2.0, 2.0 };
        var expected = new[]
        {
            -1.184032466693905, 1.2241960676768329e-16, 1.1840324666939057, -1.8480835395913313, 1.8480835395913333,
        };
        var got = TruncNorm.Ppf(q, a, b);
        for (int i = 0; i < q.Length; i++)
            got[i].Should().BeApproximately(expected[i], ErfTol);
    }

    // ---- mixture of product distributions (vs Optuna) -----------------------------------------

    [TestMethod]
    public void MixtureDiscreteSample_MatchesOptuna()
    {
        var d = new BatchedDiscreteTruncNormDistributions(
            new[] { -2.0, 5.0, -8.0, 0.0 }, new[] { 3.0, 4.0, 2.5, 20.0 }, low: -10.0, high: 10.0, step: 1.0);
        var mix = new MixtureOfProductDistribution(new[] { 0.25, 0.25, 0.25, 0.25 }, new BatchedDistributions[] { d });

        var samp = mix.Sample(new LazyRandomState(123), 8);
        var got = Enumerable.Range(0, 8).Select(s => samp[s, 0]).ToArray();
        got.Should().Equal(-8.0, 4.0, -3.0, -6.0, -8.0, -1.0, -2.0, -6.0);
    }

    [TestMethod]
    public void MixtureDiscreteLogPdf_MatchesOptuna_WithinErfTolerance()
    {
        var d = new BatchedDiscreteTruncNormDistributions(
            new[] { -2.0, 5.0, -8.0, 0.0 }, new[] { 3.0, 4.0, 2.5, 20.0 }, low: -10.0, high: 10.0, step: 1.0);
        var mix = new MixtureOfProductDistribution(new[] { 0.25, 0.25, 0.25, 0.25 }, new BatchedDistributions[] { d });

        var vals = new[] { -8.0, 4.0, -3.0, -6.0 };
        var x = new double[vals.Length, 1];
        for (int s = 0; s < vals.Length; s++)
            x[s, 0] = vals[s];
        var lp = mix.LogPdf(x);
        var expected = new[]
        {
            -2.7596702116764478, -3.143876666564482, -2.91939894600789, -2.80290457457915,
        };
        for (int s = 0; s < vals.Length; s++)
            lp[s].Should().BeApproximately(expected[s], ErfTol);
    }

    // ---- module defaults (bit-exact port of sampler.py) ---------------------------------------

    [TestMethod]
    public void DefaultGamma_MatchesOptuna()
    {
        TpeTrialSplit.DefaultGamma(0).Should().Be(0);
        TpeTrialSplit.DefaultGamma(10).Should().Be(1);
        TpeTrialSplit.DefaultGamma(21).Should().Be(3);
        TpeTrialSplit.DefaultGamma(300).Should().Be(25);  // capped at 25.
        TpeTrialSplit.DefaultGammaMultiObjective(300).Should().Be(30);  // uncapped.
    }

    [TestMethod]
    public void DefaultWeights_MatchesOptuna()
    {
        TpeTrialSplit.DefaultWeights(0).Should().BeEmpty();
        TpeTrialSplit.DefaultWeights(5).Should().Equal(1.0, 1.0, 1.0, 1.0, 1.0);
        TpeTrialSplit.DefaultWeights(25).Should().OnlyContain(w => w == 1.0);
        TpeTrialSplit.DefaultWeights(25).Should().HaveCount(25);

        var w30 = TpeTrialSplit.DefaultWeights(30);
        w30.Should().HaveCount(30);
        // First 5 form a ramp linspace(1/30, 1.0, 5); the last 25 are the flat ones.
        w30[0].Should().BeApproximately(1.0 / 30.0, 1e-15);
        w30[4].Should().Be(1.0);
        w30.Skip(5).Should().OnlyContain(w => w == 1.0);
    }

    // ---- structural behavior ------------------------------------------------------------------

    [TestMethod]
    public void Reproducible_SameSeed_ProducesIdenticalTrials()
    {
        List<double> Run()
        {
            var study = Optuna.CreateStudy(sampler: new TPESampler(seed: 5));
            study.Optimize(t =>
            {
                double x = t.SuggestFloat("x", -10, 10);
                long z = t.SuggestInt("z", -5, 5);
                return Math.Pow(x, 2.0) + z;
            }, nTrials: 25);
            return study.Trials.Select(t => (double)t.Params["x"]).ToList();
        }

        Run().Should().Equal(Run());
    }

    [TestMethod]
    public void FirstNStartupTrials_UseRandomSampling()
    {
        // Below n_startup_trials the TPE sampler delegates to RandomSampler(seed): the first
        // n_startup suggestions are bit-identical to a standalone RandomSampler(seed).
        var tpeStudy = Optuna.CreateStudy(sampler: new TPESampler(seed: 3, nStartupTrials: 8));
        tpeStudy.Optimize(t => Math.Pow(t.SuggestFloat("x", -10, 10), 2.0), nTrials: 8);

        var rndStudy = Optuna.CreateStudy(sampler: new RandomSampler(seed: 3));
        rndStudy.Optimize(t => Math.Pow(t.SuggestFloat("x", -10, 10), 2.0), nTrials: 8);

        for (int i = 0; i < 8; i++)
            ((double)tpeStudy.Trials[i].Params["x"]).Should().Be((double)rndStudy.Trials[i].Params["x"]);
    }

    [TestMethod]
    public void GroupWithoutMultivariate_ThrowsValueError()
    {
        var act = () => new TPESampler(group: true, multivariate: false);
        act.Should().Throw<ValueError>().WithMessage("*group*multivariate*");
    }

    [TestMethod]
    public void Group_EmitsExperimentalWarning()
    {
        // Port of Optuna's test_group_experimental_warning: constructing with group=True warns.
        string? captured = null;
        void Handler(string msg, WarningCategory cat)
        {
            if (cat == WarningCategory.ExperimentalWarning && msg.Contains("group"))
                captured = msg;
        }

        OptunaWarnings.WarningRaised += Handler;
        try
        {
            _ = new TPESampler(multivariate: true, group: true, seed: 0);
        }
        finally
        {
            OptunaWarnings.WarningRaised -= Handler;
        }

        captured.Should().NotBeNull();
        captured.Should().Contain("experimental");
    }

    [TestMethod]
    public void Group_InferRelativeSearchSpace_ReturnsUnionOfSubspaceParams()
    {
        // Port of Optuna's test_group: disjoint param sets across trials decompose into subspaces
        // {x} and {y, z}; the inferred relative search space is their union.
        var sampler = new TPESampler(multivariate: true, group: true, nStartupTrials: 0, seed: 0);
        var study = Optuna.CreateStudy(sampler: sampler);

        study.Optimize(t => t.SuggestInt("x", 0, 10), nTrials: 1);
        study.Optimize(t => t.SuggestInt("y", 0, 10) + t.SuggestFloat("z", -3, 3), nTrials: 1);

        var probe = study.Ask();
        var searchSpace = sampler.InferRelativeSearchSpace(study, probe.CachedFrozenTrial);
        searchSpace.Keys.Should().BeEquivalentTo(new[] { "x", "y", "z" });
    }

    [TestMethod]
    public void GroupWithDefaultMultivariate_SingleObjective_InfersAllParams()
    {
        // Port of Optuna's test_group_with_default_multivariate (single-objective): group=True with
        // multivariate left at its default still enables the multivariate TPE and infers {x, y}.
        var sampler = new TPESampler(group: true, nStartupTrials: 0, seed: 0);
        var study = Optuna.CreateStudy(sampler: sampler);

        for (int i = 0; i < 2; i++)
        {
            var trial = study.Ask();
            trial.SuggestFloat("x", 0, 10);
            trial.SuggestFloat("y", 0, 10);
            study.Tell(trial, 0.0);
        }

        var searchSpace = sampler.InferRelativeSearchSpace(study, study.Trials[^1]);
        searchSpace.Keys.Should().BeEquivalentTo(new[] { "x", "y" });
    }

    [TestMethod]
    public void GroupWithDefaultMultivariate_MultiObjective_WarnsAndInfersAllParams()
    {
        // Port of Optuna's test_group_with_default_multivariate (multi-objective): the multivariate
        // default (False for MO) is overridden by group=True, which emits a UserWarning.
        var sampler = new TPESampler(group: true, nStartupTrials: 0, seed: 0);
        var study = Optuna.CreateStudy(
            sampler: sampler,
            directions: new[] { StudyDirection.Minimize, StudyDirection.Minimize });

        bool warned = false;
        void Handler(string msg, WarningCategory cat)
        {
            if (cat == WarningCategory.UserWarning && msg.Contains("group"))
                warned = true;
        }

        OptunaWarnings.WarningRaised += Handler;
        try
        {
            for (int i = 0; i < 2; i++)
            {
                var trial = study.Ask();
                trial.SuggestFloat("x", 0, 10);
                trial.SuggestFloat("y", 0, 10);
                study.Tell(trial, new[] { 0.0, 0.0 });
            }
        }
        finally
        {
            OptunaWarnings.WarningRaised -= Handler;
        }

        warned.Should().BeTrue();
        var searchSpace = sampler.InferRelativeSearchSpace(study, study.Trials[^1]);
        searchSpace.Keys.Should().BeEquivalentTo(new[] { "x", "y" });
    }

    [TestMethod]
    public void Group_Reproducible_SameSeed_ProducesIdenticalTrials()
    {
        // The group-decomposed multivariate TPE is deterministic under a fixed seed.
        List<double> Run()
        {
            var study = Optuna.CreateStudy(
                sampler: new TPESampler(seed: 11, multivariate: true, group: true, nStartupTrials: 5));
            study.Optimize(t =>
            {
                var branch = (string)t.SuggestCategorical(
                    "c", new CategoricalChoice[] { "left", "right" }).Value!;
                return branch == "left"
                    ? t.SuggestFloat("y", -10, 10)
                    : t.SuggestInt("z", -10, 10);
            }, nTrials: 20);
            return study.Trials.Select(t => t.Values![0]).ToList();
        }

        Run().Should().Equal(Run());
    }

    [TestMethod]
    public void MultiObjective_TpeSampler_Reproducible_SameSeed()
    {
        // MO TPE splits below/above by non-domination rank + the HSSP tie-break (Phase 2.10). The
        // whole run must be deterministic under a fixed seed.
        List<double[]> Run()
        {
            var study = Optuna.CreateStudy(
                sampler: new TPESampler(seed: 4, nStartupTrials: 8),
                directions: new[] { StudyDirection.Minimize, StudyDirection.Minimize });
            study.Optimize((Func<Trial, IReadOnlyList<double>>)(t =>
            {
                double x = t.SuggestFloat("x", -5, 5);
                double y = t.SuggestFloat("y", -5, 5);
                return new[] { x * x + y, (x - 2) * (x - 2) + y * y };
            }), nTrials: 30);
            return study.Trials.Select(t => t.Values!.ToArray()).ToList();
        }

        var a = Run();
        var b = Run();
        a.Count.Should().Be(b.Count);
        for (int i = 0; i < a.Count; i++)
            a[i].Should().Equal(b[i]);
    }

    [TestMethod]
    public void MultiObjective3d_TpeSampler_CompletesThroughHsspTieBreak()
    {
        // Three objectives exercise the n-dimensional HSSP tie-break (was a NotImplementedException
        // before Phase 2.10). The study must run end-to-end and produce a non-empty Pareto front.
        var study = Optuna.CreateStudy(
            sampler: new TPESampler(seed: 1, nStartupTrials: 8),
            directions: Enumerable.Repeat(StudyDirection.Minimize, 3).ToArray());
        var act = () => study.Optimize((Func<Trial, IReadOnlyList<double>>)(t =>
        {
            double x = t.SuggestFloat("x", -5, 5);
            double y = t.SuggestFloat("y", -5, 5);
            return new[] { x * x + y, (x - 2) * (x - 2) + y * y, (x + 1) * (x + 1) + (y - 1) * (y - 1) };
        }), nTrials: 40);

        act.Should().NotThrow();
        study.Trials.Should().HaveCount(40);
        study.BestTrials.Should().NotBeEmpty();
    }

    [TestMethod]
    public void MultiObjective_InfersEmptyRelativeSearchSpace()
    {
        // multivariate defaults to False for multi-objective, so relative sampling is disabled.
        var study = Optuna.CreateStudy(
            sampler: new TPESampler(seed: 1),
            directions: new[] { StudyDirection.Minimize, StudyDirection.Minimize });
        var trial = study.Ask();
        study.Sampler.InferRelativeSearchSpace(study, trial.CachedFrozenTrial).Should().BeEmpty();
    }

    [TestMethod]
    public void DeprecatedConsiderPriorArgument_Warns()
    {
        string? captured = null;
        void Handler(string msg, WarningCategory cat)
        {
            if (msg.Contains("consider_prior"))
                captured = msg;
        }

        OptunaWarnings.WarningRaised += Handler;
        try
        {
            _ = new TPESampler(considerPrior: true, seed: 0);
        }
        finally
        {
            OptunaWarnings.WarningRaised -= Handler;
        }

        captured.Should().NotBeNull();
        captured.Should().Contain("has been deprecated in v4.3.0");
    }
}
