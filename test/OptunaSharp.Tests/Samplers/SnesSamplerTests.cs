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
///     Tests for the native <see cref="SnesOptimizer" /> (Separable NES, Wierstra et al. 2014) and its
///     <see cref="SnesSampler" />. SNES has no <c>cmaes</c>/Optuna/OptunaHub counterpart, so — like
///     <c>DifferentialEvolutionSamplerTests</c> — these are behavioral gates: convergence, same-seed
///     determinism, and optimizer-state round-trip fidelity. Because σ is adapted per coordinate, SNES
///     converges to high precision on separable problems, so its optimizer-level thresholds are tight (near
///     machine precision on the sphere).
/// </summary>
[TestClass]
public class SnesSamplerTests
{
    private static double Sphere(double[] x)
    {
        double s = 0;
        foreach (double v in x) s += v * v;
        return s;
    }

    /// <summary>A SNES optimizer with a broad initial σ, for the unbounded direct-drive tests.</summary>
    private static SnesOptimizer MakeOptimizer(double[] mean, uint seed, double sigma = 0.5)
    {
        int d = mean.Length;
        var sigmaVec = new double[d];
        for (int i = 0; i < d; i++) sigmaVec[i] = sigma;
        return new SnesOptimizer(mean, sigmaVec, etaMu: 1.0, etaSigma: SnesOptimizer.DefaultEtaSigma(d),
            popsize: SnesOptimizer.DefaultPopulationSize(d), seed: seed);
    }

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
        double best = Drive(MakeOptimizer(new[] { 2.0, 2, 2 }, seed: 4), Sphere, generations: 300);
        best.Should().BeLessThan(1e-8, "per-coordinate σ adaptation drives the separable sphere to high precision");
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
        Drive(opt, Sphere, generations: 20);   // evolve mean, per-coordinate σ, generation

        var restored = SnesOptimizer.FromState(
            JsonSerializer.Deserialize<SnesState>(JsonSerializer.Serialize(opt.ToState()))!);

        restored.Dim.Should().Be(opt.Dim);
        restored.PopulationSize.Should().Be(opt.PopulationSize);
        restored.Generation.Should().Be(opt.Generation);
        restored.SigmaSnapshot().Should().Equal(opt.SigmaSnapshot());

        for (int i = 0; i < opt.PopulationSize; i++)
            restored.Ask(i).Should().Equal(opt.Ask(i));
    }

    // ---- sampler: end-to-end through a Study ------------------------------------------------------

    [TestMethod]
    public void Sampler_MinimizesSphere()
    {
        var study = Optuna.CreateStudy(sampler: new SnesSampler(seed: 1));
        study.Optimize(t =>
        {
            double s = 0;
            for (int i = 0; i < 5; i++) { double x = t.SuggestFloat($"x{i}", -5, 5); s += x * x; }
            return s;
        }, nTrials: 500);

        study.BestValue.Should().BeLessThan(1e-2, "SNES converges tightly on the separable sphere");
    }

    [TestMethod]
    public void Sampler_Maximize_Works()
    {
        var study = Optuna.CreateStudy(
            direction: StudyDirection.Maximize, sampler: new SnesSampler(seed: 2));
        study.Optimize(t =>
        {
            double s = 0;
            for (int i = 0; i < 5; i++) { double x = t.SuggestFloat($"x{i}", -5, 5); s += x * x; }
            return -s;
        }, nTrials: 500);

        study.BestValue.Should().BeGreaterThan(-1e-2, "direction flipping lets SNES maximize −sphere");
    }

    [TestMethod]
    public void Sampler_SameSeed_IsReproducible()
    {
        double Run()
        {
            var study = Optuna.CreateStudy(sampler: new SnesSampler(seed: 7));
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
        var study = Optuna.CreateStudy(sampler: new SnesSampler(seed: 42));
        study.Optimize(t =>
        {
            double x = t.SuggestFloat("x", -5, 5);
            double y = t.SuggestFloat("y", -5, 5);
            return x * x + y * y;
        }, nTrials: 40);

        study.Trials.Any(tr => tr.SystemAttrs.ContainsKey("snes:generation")).Should().BeTrue();
    }

    [TestMethod]
    public void Sampler_MixedIntFloatStepLog_Works()
    {
        var study = Optuna.CreateStudy(sampler: new SnesSampler(seed: 2));
        study.Optimize(t =>
        {
            double a = t.SuggestFloat("a", -5, 5);
            long b = t.SuggestInt("b", -8, 8);
            double c = t.SuggestFloat("c", -4, 4, step: 0.5);
            double d = t.SuggestFloat("d", 1e-3, 1e3, log: true);
            double logd = Math.Log10(d);
            return a * a + b * b + c * c + logd * logd;
        }, nTrials: 500);
        study.BestValue.Should().BeLessThan(5.0, "a mixed int/float/step/log space is handled by the transform");
    }

    [TestMethod]
    public void Sampler_RespectsBounds()
    {
        var study = Optuna.CreateStudy(sampler: new SnesSampler(seed: 3));
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
        var study = Optuna.CreateStudy(sampler: new SnesSampler(seed: 3));
        study.Optimize(t =>
        {
            string pick = t.SuggestCategorical("mode", new[] { "u", "v", "w" });
            double bonus = pick == "v" ? 0.0 : 1.0;
            double x = t.SuggestFloat("x", -5, 5);
            double y = t.SuggestFloat("y", -5, 5);
            return x * x + y * y + bonus;
        }, nTrials: 400);

        study.BestValue.Should().BeLessThan(1.5, "continuous params converge while the categorical is independent");
    }

    [TestMethod]
    public void Sampler_MultiObjective_Raises()
    {
        var study = Optuna.CreateStudy(
            directions: new[] { StudyDirection.Minimize, StudyDirection.Minimize },
            sampler: new SnesSampler(seed: 1));
        Action act = () => study.Optimize(t =>
        {
            double x = t.SuggestFloat("x", -5, 5);
            return new[] { x * x, (x - 2) * (x - 2) };
        }, nTrials: 5);
        act.Should().Throw<ValueError>("SNES is single-objective only");
    }

    [TestMethod]
    public void Constructor_RejectsNonPositiveSigma()
    {
        Action act = () => new SnesSampler(sigma: 0.0);
        act.Should().Throw<ArgumentException>();
    }
}
