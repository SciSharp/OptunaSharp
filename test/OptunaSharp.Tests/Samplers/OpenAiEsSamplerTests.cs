using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json;
using AwesomeAssertions;
using NumSharp;
using OptunaSharp.Samplers;
using OptunaSharp.Samplers.EvolutionStrategy;

namespace OptunaSharp.Tests.Samplers;

/// <summary>
///     Tests for the native <see cref="OpenEsOptimizer" /> (OpenAI-ES, Salimans et al. 2017) and its
///     <see cref="OpenAiEsSampler" />. OpenAI-ES has no <c>cmaes</c>/Optuna/OptunaHub counterpart, so — like
///     <c>DifferentialEvolutionSamplerTests</c> — these are behavioral gates: convergence, same-seed
///     determinism, optimizer-state round-trip fidelity, and antithetic-sampling structure. Thresholds are
///     calibrated to measured convergence; OpenAI-ES uses a single (slowly annealed) σ, so it settles to a
///     neighborhood of the optimum rather than to machine precision — a deliberately looser bar than the
///     per-coordinate SNES.
/// </summary>
[TestClass]
public class OpenAiEsSamplerTests
{
    private static double Sphere(double[] x)
    {
        double s = 0;
        foreach (double v in x) s += v * v;
        return s;
    }

    /// <summary>An OpenAI-ES optimizer with an annealed σ, suitable for the unbounded direct-drive tests.</summary>
    private static OpenEsOptimizer MakeOptimizer(double[] mean, uint seed) =>
        new(mean, sigma: 0.5, learningRate: 0.2, popsize: 8, mirrored: true,
            weightDecay: 0.0, sigmaDecay: 0.99, sigmaLimit: 1e-3, learningRateDecay: 1.0, learningRateLimit: 1e-3,
            beta1: 0.9, beta2: 0.999, adamEpsilon: 1e-8, seed: seed);

    /// <summary>Drive the optimizer as a batch ask/tell loop on an unbounded objective (no clamping).</summary>
    private static double Drive(IEvolutionStrategyOptimizer opt, Func<double[], double> f, int generations)
    {
        double best = double.PositiveInfinity;
        for (int g = 0; g < generations; g++)
        {
            var sols = new List<(double[] X, double Value)>(opt.PopulationSize);
            for (int i = 0; i < opt.PopulationSize; i++)
            {
                double[] x = opt.Ask(i);
                double v = f(x);
                if (v < best) best = v;
                sols.Add((x, v));
            }
            opt.Tell(sols);
        }
        return best;
    }

    // ---- optimizer: convergence, determinism, state ----------------------------------------------

    [TestMethod]
    public void Optimizer_Sphere_Converges()
    {
        double best = Drive(MakeOptimizer(new[] { 2.0, 2, 2 }, seed: 3), Sphere, generations: 300);
        best.Should().BeLessThan(1e-2, "OpenAI-ES with an annealed σ drives the sphere far below its start (f=12)");
    }

    [TestMethod]
    public void Optimizer_SameSeed_IsReproducible()
    {
        double a = Drive(MakeOptimizer(new[] { 1.0, -1, 2, 0.5 }, seed: 11), Sphere, generations: 60);
        double b = Drive(MakeOptimizer(new[] { 1.0, -1, 2, 0.5 }, seed: 11), Sphere, generations: 60);
        b.Should().Be(a, "same seed and start ⇒ identical trajectory");
    }

    [TestMethod]
    public void Optimizer_StateRoundTrip_PreservesEverything()
    {
        var opt = MakeOptimizer(new[] { 1.0, 2, 3, 4, 5 }, seed: 5);
        Drive(opt, Sphere, generations: 20);   // evolve mean, Adam moments, σ, generation

        var restored = OpenEsOptimizer.FromState(
            JsonSerializer.Deserialize<OpenEsState>(JsonSerializer.Serialize(opt.ToState()))!);

        restored.Dim.Should().Be(opt.Dim);
        restored.PopulationSize.Should().Be(opt.PopulationSize);
        restored.Generation.Should().Be(opt.Generation);
        restored.Sigma.Should().Be(opt.Sigma);

        // Every candidate of the next generation must be bit-identical — proving the mean, Adam moments,
        // and seed all round-tripped, not just the scalars.
        for (int i = 0; i < opt.PopulationSize; i++)
            restored.Ask(i).Should().Equal(opt.Ask(i));
    }

    [TestMethod]
    public void Optimizer_MirroredSampling_ProducesAntitheticPairs()
    {
        // With antithetic sampling, candidate k and candidate k+half are mirror images about the mean:
        // (mean + σε) + (mean − σε) = 2·mean.
        var opt = new OpenEsOptimizer(new[] { 0.5, 0.5 }, 0.1, 0.05, 6, mirrored: true,
            0, 1.0, 0.01, 1.0, 1e-3, 0.9, 0.999, 1e-8, seed: 9);
        double[] a0 = opt.Ask(0);
        double[] a3 = opt.Ask(3);   // half = 3
        (a0[0] + a3[0]).Should().BeApproximately(1.0, 1e-12);
        (a0[1] + a3[1]).Should().BeApproximately(1.0, 1e-12);
    }

    [TestMethod]
    public void Optimizer_RejectsOddMirroredPopulation()
    {
        Action act = () => new OpenEsOptimizer(new[] { 0.5, 0.5 }, 0.1, 0.05, 7, mirrored: true,
            0, 0.999, 0.01, 1.0, 1e-3, 0.9, 0.999, 1e-8, seed: 1);
        act.Should().Throw<ArgumentException>("antithetic sampling needs an even population to form pairs");
    }

    // ---- sampler: end-to-end through a Study ------------------------------------------------------

    [TestMethod]
    public void Sampler_MinimizesSphere()
    {
        var study = Optuna.CreateStudy(sampler: new OpenAiEsSampler(seed: 1));
        study.Optimize(t =>
        {
            double s = 0;
            for (int i = 0; i < 5; i++) { double x = t.SuggestFloat($"x{i}", -5, 5); s += x * x; }
            return s;
        }, nTrials: 800);

        study.BestValue.Should().BeLessThan(1.0, "OpenAI-ES settles well inside the random baseline (~2.3)");
    }

    [TestMethod]
    public void Sampler_Maximize_Works()
    {
        var study = Optuna.CreateStudy(
            direction: StudyDirection.Maximize, sampler: new OpenAiEsSampler(seed: 2));
        study.Optimize(t =>
        {
            double s = 0;
            for (int i = 0; i < 5; i++) { double x = t.SuggestFloat($"x{i}", -5, 5); s += x * x; }
            return -s;   // optimum 0, approached from below
        }, nTrials: 800);

        study.BestValue.Should().BeGreaterThan(-1.0, "direction flipping lets OpenAI-ES maximize −sphere");
    }

    [TestMethod]
    public void Sampler_SameSeed_IsReproducible()
    {
        double Run()
        {
            var study = Optuna.CreateStudy(sampler: new OpenAiEsSampler(seed: 7));
            study.Optimize(t =>
            {
                double s = 0;
                for (int i = 0; i < 4; i++) { double x = t.SuggestFloat($"x{i}", -5, 5); s += x * x; }
                return s;
            }, nTrials: 200);
            return study.BestValue;
        }

        Run().Should().Be(Run(), "a fixed seed gives an identical trajectory");
    }

    [TestMethod]
    public void Sampler_GenerationAttr_IsStamped()
    {
        var study = Optuna.CreateStudy(sampler: new OpenAiEsSampler(seed: 42));
        study.Optimize(t =>
        {
            double x = t.SuggestFloat("x", -5, 5);
            double y = t.SuggestFloat("y", -5, 5);
            return x * x + y * y;
        }, nTrials: 40);

        study.Trials.Any(tr => tr.SystemAttrs.ContainsKey("openaies:generation")).Should().BeTrue();
    }

    [TestMethod]
    public void Sampler_RespectsBounds()
    {
        var study = Optuna.CreateStudy(sampler: new OpenAiEsSampler(seed: 3));
        study.Optimize(t =>
        {
            double x = t.SuggestFloat("x", -2, 3);
            double y = t.SuggestFloat("y", 1, 4);
            return (x - 1) * (x - 1) + (y - 2) * (y - 2);
        }, nTrials: 200);

        foreach (var tr in study.Trials)
        {
            ((double)tr.Params["x"]!).Should().BeInRange(-2, 3);
            ((double)tr.Params["y"]!).Should().BeInRange(1, 4);
        }
    }

    [TestMethod]
    public void Sampler_CategoricalParameter_FallsBackToIndependent()
    {
        // Categorical params are outside the relative space → sampled independently, while the continuous
        // params drive OpenAI-ES. The study must run and the continuous coordinates must improve.
        var study = Optuna.CreateStudy(sampler: new OpenAiEsSampler(seed: 3));
        study.Optimize(t =>
        {
            string pick = t.SuggestCategorical("mode", new[] { "u", "v", "w" });
            double bonus = pick == "v" ? 0.0 : 1.0;
            double x = t.SuggestFloat("x", -5, 5);
            double y = t.SuggestFloat("y", -5, 5);
            return x * x + y * y + bonus;
        }, nTrials: 400);

        study.BestValue.Should().BeLessThan(2.0, "continuous params improve while the categorical is independent");
    }

    [TestMethod]
    public void Sampler_MultiObjective_Raises()
    {
        var study = Optuna.CreateStudy(
            directions: new[] { StudyDirection.Minimize, StudyDirection.Minimize },
            sampler: new OpenAiEsSampler(seed: 1));
        Action act = () => study.Optimize(t =>
        {
            double x = t.SuggestFloat("x", -5, 5);
            return new[] { x * x, (x - 2) * (x - 2) };
        }, nTrials: 5);
        act.Should().Throw<ValueError>("OpenAI-ES is single-objective only");
    }
}
