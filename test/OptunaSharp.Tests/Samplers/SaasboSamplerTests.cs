using System;
using System.Linq;
using AwesomeAssertions;
using NumSharp;
using OptunaSharp.Samplers;

namespace OptunaSharp.Tests.Samplers;

/// <summary>
///     Tests for the native <see cref="SaasboSampler" /> (SAASBO, Eriksson &amp; Jankowiak 2021). SAASBO is
///     offered by Optuna only through the BoTorch integration, so — like <c>TurboSamplerTests</c> — these
///     are behavioral gates: convergence on an objective with sparse low-dimensional structure (the paper's
///     setting), same-seed determinism, bounds, mixed int/float handling, and the continuous-only fallback
///     for categoricals. Both inference paths (NUTS, MAP) are exercised. Budgets are kept deliberately small
///     (NUTS is <c>O(N³D)</c> per leapfrog step) so the suite stays fast; the paper-faithful defaults are far
///     heavier and validated separately for convergence in the primitive tests.
/// </summary>
[TestClass]
public class SaasboSamplerTests
{
    /// <summary>A 2-D sphere embedded in <paramref name="d" /> dimensions — the paper's "sparse low-dimensional
    ///     structure" regime, minimized at <c>x0 = 0.3, x1 = −1.2</c> with the remaining axes irrelevant.</summary>
    /// <param name="t">The trial supplying suggestions.</param>
    /// <param name="d">The ambient dimension (≥ 2).</param>
    /// <returns>The embedded sphere value.</returns>
    private static double EmbeddedSphere(Trial t, int d)
    {
        double a = t.SuggestFloat("x0", -5, 5);
        double b = t.SuggestFloat("x1", -5, 5);
        for (int i = 2; i < d; i++) t.SuggestFloat($"x{i}", -5, 5);
        return (a - 0.3) * (a - 0.3) + (b + 1.2) * (b + 1.2);
    }

    /// <summary>A light NUTS-SAASBO sampler for fast tests (paper defaults are far heavier).</summary>
    /// <param name="seed">The RNG seed.</param>
    /// <returns>A configured NUTS <see cref="SaasboSampler" /> with a small inference/acquisition budget.</returns>
    private static SaasboSampler LightNuts(int seed) => new(
        seed: seed, nStartupTrials: 8,
        numWarmup: 40, numSamples: 40, thinning: 8, maxTreeDepth: 5,
        numCandidates: 200, numRestarts: 2);

    /// <summary>A fast MAP-SAASBO sampler for tests.</summary>
    /// <param name="seed">The RNG seed.</param>
    /// <returns>A configured MAP <see cref="SaasboSampler" /> with a small Adam/acquisition budget.</returns>
    private static SaasboSampler LightMap(int seed) => new(
        seed: seed, nStartupTrials: 8, inferenceMethod: SaasboInferenceMethod.Map,
        mapSteps: 250, numCandidates: 200, numRestarts: 2);

    [TestMethod]
    public void Sampler_Nuts_MinimizesEmbeddedSphere()
    {
        // NUTS is the paper's default. On a 2-D objective embedded in 5 dimensions it should drive well below
        // a random baseline (~5–10 on this box) by exploiting the identified sparse subspace.
        var study = Optuna.CreateStudy(sampler: LightNuts(0));
        study.Optimize(t => EmbeddedSphere(t, 5), nTrials: 35);
        study.BestValue.Should().BeLessThan(0.5, "NUTS-SAASBO exploits the sparse 2-D structure");
    }

    [TestMethod]
    public void Sampler_Map_MinimizesEmbeddedSphere()
    {
        // The faster MAP path should also converge on the embedded objective.
        var study = Optuna.CreateStudy(sampler: LightMap(0));
        study.Optimize(t => EmbeddedSphere(t, 5), nTrials: 35);
        study.BestValue.Should().BeLessThan(0.5, "MAP-SAASBO also identifies the relevant subspace");
    }

    [TestMethod]
    public void Sampler_SameSeed_IsDeterministic()
    {
        // A fixed seed reproduces the whole trajectory on a platform (MT19937, never System.Random).
        double Best(int seed)
        {
            var s = Optuna.CreateStudy(sampler: new SaasboSampler(
                seed: seed, nStartupTrials: 6, numWarmup: 20, numSamples: 20, thinning: 5,
                maxTreeDepth: 4, numCandidates: 128, numRestarts: 2));
            s.Optimize(t => EmbeddedSphere(t, 4), nTrials: 18);
            return s.BestValue;
        }
        Best(42).Should().Be(Best(42), "same seed ⇒ identical trajectory");
    }

    [TestMethod]
    public void Sampler_DifferentSeed_ProducesDifferentTrajectory()
    {
        double[] Params(int seed)
        {
            var s = Optuna.CreateStudy(sampler: LightNuts(seed));
            s.Optimize(t => EmbeddedSphere(t, 4), nTrials: 18);
            // The last (BO-phase) trial's coordinates depend on the whole seeded chain.
            var last = s.Trials[^1];
            return new[] { Convert.ToDouble(last.Params["x0"]), Convert.ToDouble(last.Params["x1"]) };
        }
        Params(1).Should().NotBeEquivalentTo(Params(2), "different seeds explore differently");
    }

    [TestMethod]
    public void Sampler_RespectsBounds()
    {
        var study = Optuna.CreateStudy(sampler: LightNuts(3));
        study.Optimize(t =>
        {
            double x = t.SuggestFloat("x", -2, 3);
            double y = t.SuggestFloat("y", 1, 4);
            return (x - 1) * (x - 1) + (y - 2) * (y - 2);
        }, nTrials: 30);

        foreach (var tr in study.Trials)
        {
            ((double)tr.Params["x"]!).Should().BeInRange(-2, 3);
            ((double)tr.Params["y"]!).Should().BeInRange(1, 4);
        }
    }

    [TestMethod]
    public void Sampler_MixedIntFloatStepLog_Works()
    {
        // The SAAS GP box normalizes int/step/log dimensions like the shared GP stack; the acquisition
        // optimizes continuously and unnormalization rounds discrete coordinates.
        var study = Optuna.CreateStudy(sampler: LightMap(3));
        study.Optimize(t =>
        {
            double a = t.SuggestFloat("a", -5, 5);
            long b = t.SuggestInt("b", -8, 8);
            double c = t.SuggestFloat("c", -4, 4, step: 0.5);
            double d = t.SuggestFloat("d", 1e-3, 1e3, log: true);
            double logd = Math.Log10(d);
            return a * a + b * b + c * c + logd * logd;
        }, nTrials: 40);

        foreach (var tr in study.Trials)
        {
            ((long)tr.Params["b"]!).Should().BeInRange(-8, 8);
            ((double)tr.Params["d"]!).Should().BeInRange(1e-3, 1e3);
        }
        study.BestValue.Should().BeLessThan(20.0, "int/float/step/log dims are handled by the normalized GP box");
    }

    [TestMethod]
    public void Sampler_CategoricalParameter_FallsBackToIndependent()
    {
        // The SAAS GP box is continuous, so a categorical is sampled independently while the continuous params
        // drive SAASBO. The study must run and the continuous coordinates must converge.
        var study = Optuna.CreateStudy(sampler: LightNuts(3));
        study.Optimize(t =>
        {
            string pick = t.SuggestCategorical("mode", new[] { "u", "v", "w" });
            double bonus = pick == "v" ? 0.0 : 1.0;
            double x = t.SuggestFloat("x", -5, 5);
            double y = t.SuggestFloat("y", -5, 5);
            return x * x + y * y + bonus;
        }, nTrials: 35);

        study.BestValue.Should().BeLessThan(1.6, "continuous params converge while the categorical is independent");
    }

    [TestMethod]
    public void Sampler_StartupPhaseIsSobol_ThenGpTakesOver()
    {
        // The first nStartupTrials are the Sobol initialization; afterwards SAASBO fits the GP. The study
        // should complete and improve past the init budget.
        var study = Optuna.CreateStudy(sampler: LightNuts(5));
        study.Optimize(t => EmbeddedSphere(t, 4), nTrials: 30);
        study.Trials.Count.Should().Be(30);
        study.BestValue.Should().BeLessThan(0.5, "SAASBO improves markedly past the Sobol init trials");
    }

    [TestMethod]
    public void Sampler_MultiObjective_Raises()
    {
        var study = Optuna.CreateStudy(
            directions: new[] { StudyDirection.Minimize, StudyDirection.Minimize },
            sampler: LightNuts(1));
        Action act = () => study.Optimize(t =>
        {
            double x = t.SuggestFloat("x", -5, 5);
            return new[] { x * x, (x - 2) * (x - 2) };
        }, nTrials: 5);
        act.Should().Throw<ValueError>("SAASBO is single-objective only");
    }

    [TestMethod]
    public void Sampler_SingleValuedParameter_IsHandled()
    {
        // A single-valued distribution is sampled inside Trial (not modeled by the GP); the study must still run.
        var study = Optuna.CreateStudy(sampler: LightMap(2));
        study.Optimize(t =>
        {
            double fixedVal = t.SuggestFloat("fixed", 2.0, 2.0);   // single value
            double x = t.SuggestFloat("x", -5, 5);
            return (x - 1) * (x - 1) + fixedVal;
        }, nTrials: 25);

        study.Trials.Should().OnlyContain(tr => (double)tr.Params["fixed"]! == 2.0);
        study.BestValue.Should().BeLessThan(3.0);
    }

    [TestMethod]
    public void Constructor_RejectsInvalidArguments()
    {
        Action a1 = () => new SaasboSampler(nStartupTrials: 0);
        a1.Should().Throw<ArgumentException>("nStartupTrials must be at least 1");

        Action a2 = () => new SaasboSampler(alpha: 0.0);
        a2.Should().Throw<ArgumentException>("alpha must be positive");

        Action a3 = () => new SaasboSampler(thinning: 0);
        a3.Should().Throw<ArgumentException>("thinning must be at least 1");

        Action a4 = () => new SaasboSampler(noiseVariance: -1.0);
        a4.Should().Throw<ArgumentException>("noiseVariance must be positive");
    }
}
