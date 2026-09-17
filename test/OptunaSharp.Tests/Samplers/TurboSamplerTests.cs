using System;
using System.Linq;
using AwesomeAssertions;
using NumSharp;
using OptunaSharp.Samplers;

namespace OptunaSharp.Tests.Samplers;

/// <summary>
///     Tests for the native <see cref="TurboSampler" /> (TuRBO-1, Eriksson et al. 2019). TuRBO has no
///     <c>cmaes</c>/Optuna/OptunaHub counterpart, so — like <c>DifferentialEvolutionSamplerTests</c> —
///     these are behavioral gates: convergence (including on the multimodal Ackley where the trust-region
///     restart strategy beats a single global GP), same-seed determinism, bounds, and the continuous-only
///     fallback for categorical parameters. Thresholds are calibrated to measured convergence (GP +
///     Thompson sampling is stochastic and tolerance-reproducible, not bit-exact). Small candidate sets and
///     dimensions keep the runs fast.
/// </summary>
[TestClass]
public class TurboSamplerTests
{
    private static double Sphere(Trial t, int d)
    {
        double s = 0;
        for (int i = 0; i < d; i++) { double x = t.SuggestFloat($"x{i}", -5, 5); s += x * x; }
        return s;
    }

    private static double Ackley(Trial t, int d)
    {
        double s1 = 0, s2 = 0;
        for (int i = 0; i < d; i++)
        {
            double x = t.SuggestFloat($"x{i}", -5, 5);
            s1 += x * x;
            s2 += Math.Cos(2 * Math.PI * x);
        }
        return -20 * Math.Exp(-0.2 * Math.Sqrt(s1 / d)) - Math.Exp(s2 / d) + 20 + Math.E;
    }

    [TestMethod]
    public void Sampler_MinimizesSphere()
    {
        var study = Optuna.CreateStudy(sampler: new TurboSampler(seed: 1, numCandidates: 150));
        study.Optimize(t => Sphere(t, 5), nTrials: 80);
        study.BestValue.Should().BeLessThan(1e-2, "TuRBO's local GP + trust region converges on the sphere");
    }

    [TestMethod]
    public void Sampler_MinimizesMultimodalAckley()
    {
        // Ackley is multimodal; a random search sits near ~5–6. The trust-region local model drives well
        // below that — the regime where TuRBO's restart strategy pays off over a single global surrogate.
        var study = Optuna.CreateStudy(sampler: new TurboSampler(seed: 1, numCandidates: 150));
        study.Optimize(t => Ackley(t, 5), nTrials: 120);
        study.BestValue.Should().BeLessThan(3.0, "TuRBO makes clear progress on the multimodal Ackley");
    }

    [TestMethod]
    public void Sampler_Maximize_Works()
    {
        var study = Optuna.CreateStudy(
            direction: StudyDirection.Maximize, sampler: new TurboSampler(seed: 1, numCandidates: 150));
        study.Optimize(t => -Sphere(t, 5), nTrials: 80);
        study.BestValue.Should().BeGreaterThan(-1e-2, "direction flipping lets TuRBO maximize −sphere");
    }

    [TestMethod]
    public void Sampler_SameSeed_IsReproducible()
    {
        double Run()
        {
            var study = Optuna.CreateStudy(sampler: new TurboSampler(seed: 7, numCandidates: 120));
            study.Optimize(t => Sphere(t, 4), nTrials: 60);
            return study.BestValue;
        }
        Run().Should().Be(Run(), "a fixed seed gives an identical trajectory on a platform");
    }

    [TestMethod]
    public void Sampler_RespectsBounds()
    {
        var study = Optuna.CreateStudy(sampler: new TurboSampler(seed: 3, numCandidates: 120));
        study.Optimize(t =>
        {
            double x = t.SuggestFloat("x", -2, 3);
            double y = t.SuggestFloat("y", 1, 4);
            return (x - 1) * (x - 1) + (y - 2) * (y - 2);
        }, nTrials: 80);

        foreach (var tr in study.Trials)
        {
            ((double)tr.Params["x"]!).Should().BeInRange(-2, 3);
            ((double)tr.Params["y"]!).Should().BeInRange(1, 4);
        }
    }

    [TestMethod]
    public void Sampler_MixedIntFloatStepLog_Works()
    {
        var study = Optuna.CreateStudy(sampler: new TurboSampler(seed: 3, numCandidates: 120));
        study.Optimize(t =>
        {
            double a = t.SuggestFloat("a", -5, 5);
            long b = t.SuggestInt("b", -8, 8);
            double c = t.SuggestFloat("c", -4, 4, step: 0.5);
            double d = t.SuggestFloat("d", 1e-3, 1e3, log: true);
            double logd = Math.Log10(d);
            return a * a + b * b + c * c + logd * logd;
        }, nTrials: 120);
        study.BestValue.Should().BeLessThan(3.0, "int/float/step/log dims are handled by the normalized TR box");
    }

    [TestMethod]
    public void Sampler_CategoricalParameter_FallsBackToIndependent()
    {
        // The trust-region box is continuous, so a categorical is sampled independently while the continuous
        // params drive TuRBO. The study must run and the continuous coordinates must converge.
        var study = Optuna.CreateStudy(sampler: new TurboSampler(seed: 3, numCandidates: 120));
        study.Optimize(t =>
        {
            string pick = t.SuggestCategorical("mode", new[] { "u", "v", "w" });
            double bonus = pick == "v" ? 0.0 : 1.0;
            double x = t.SuggestFloat("x", -5, 5);
            double y = t.SuggestFloat("y", -5, 5);
            return x * x + y * y + bonus;
        }, nTrials: 120);

        study.BestValue.Should().BeLessThan(1.5, "continuous params converge while the categorical is independent");
    }

    [TestMethod]
    public void Sampler_InitPhaseIsSpaceFilling_ThenTrustRegionTakesOver()
    {
        // The first 2·d trials of the first restart are the Sobol initialization; afterwards TuRBO proposes
        // from the local GP. The study should still converge well below random after the init budget.
        var study = Optuna.CreateStudy(sampler: new TurboSampler(seed: 5, numCandidates: 120));
        study.Optimize(t => Sphere(t, 4), nTrials: 60);
        study.Trials.Count.Should().Be(60);
        study.BestValue.Should().BeLessThan(1e-1, "TuRBO improves markedly past the 2·d init trials");
    }

    [TestMethod]
    public void Sampler_MultiObjective_Raises()
    {
        var study = Optuna.CreateStudy(
            directions: new[] { StudyDirection.Minimize, StudyDirection.Minimize },
            sampler: new TurboSampler(seed: 1));
        Action act = () => study.Optimize(t =>
        {
            double x = t.SuggestFloat("x", -5, 5);
            return new[] { x * x, (x - 2) * (x - 2) };
        }, nTrials: 5);
        act.Should().Throw<ValueError>("TuRBO-1 is single-objective only");
    }

    [TestMethod]
    public void Constructor_RejectsIllOrderedLengths()
    {
        Action act = () => new TurboSampler(lengthInit: 0.8, lengthMin: 1.0, lengthMax: 1.6);
        act.Should().Throw<ArgumentException>("lengthInit must be ≥ lengthMin");
    }
}
