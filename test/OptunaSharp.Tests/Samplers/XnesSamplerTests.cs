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
///     Tests for the native <see cref="XnesOptimizer" /> (Exponential NES, Wierstra et al. 2014) and its
///     <see cref="XnesSampler" />. xNES has no <c>cmaes</c>/Optuna/OptunaHub counterpart, so — like
///     <c>DifferentialEvolutionSamplerTests</c> — these are behavioral gates: convergence, same-seed
///     determinism, and optimizer-state round-trip fidelity. Because xNES adapts a full covariance (via the
///     matrix exponential of the natural-gradient direction, computed with OpenBLAS <c>eigh</c>), it handles
///     the correlated Rosenbrock valley the diagonal SNES cannot. These tests exercise the OpenBLAS-backed
///     linear algebra (eigh / solve / dot).
/// </summary>
[TestClass]
public class XnesSamplerTests
{
    private static double Sphere(double[] x)
    {
        double s = 0;
        foreach (double v in x) s += v * v;
        return s;
    }

    /// <summary>An xNES optimizer starting isotropic (B = I), for the unbounded direct-drive tests.</summary>
    private static XnesOptimizer MakeOptimizer(double[] mean, uint seed, double sigma = 0.5)
    {
        int d = mean.Length;
        double eta = XnesOptimizer.DefaultEtaSigma(d);
        return new XnesOptimizer(mean, sigma, XnesOptimizer.Identity(d), etaMu: 1.0, etaSigma: eta, etaB: eta,
            popsize: XnesOptimizer.DefaultPopulationSize(d), seed: seed);
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
        double best = Drive(MakeOptimizer(new[] { 2.0, 2, 2 }, seed: 5), Sphere, generations: 300);
        best.Should().BeLessThan(1e-8, "full-covariance adaptation drives the sphere to high precision");
    }

    [TestMethod]
    public void Optimizer_SameSeed_IsReproducible()
    {
        double a = Drive(MakeOptimizer(new[] { 1.0, -1, 2 }, seed: 11), Sphere, generations: 60);
        double b = Drive(MakeOptimizer(new[] { 1.0, -1, 2 }, seed: 11), Sphere, generations: 60);
        b.Should().Be(a, "same seed and start ⇒ identical trajectory");
    }

    [TestMethod]
    public void Optimizer_StateRoundTrip_PreservesEverything()
    {
        var opt = MakeOptimizer(new[] { 1.0, 2, 3, 4 }, seed: 5);
        Drive(opt, Sphere, generations: 20);   // evolve mean, σ, the full transform B, generation

        var restored = XnesOptimizer.FromState(
            JsonSerializer.Deserialize<XnesState>(JsonSerializer.Serialize(opt.ToState()))!);

        restored.Dim.Should().Be(opt.Dim);
        restored.PopulationSize.Should().Be(opt.PopulationSize);
        restored.Generation.Should().Be(opt.Generation);
        restored.Sigma.Should().Be(opt.Sigma);

        // Every candidate of the next generation must be bit-identical — the transform B (d×d) round-trips
        // through its jagged JSON form, so B·s reproduces exactly.
        for (int i = 0; i < opt.PopulationSize; i++)
            restored.Ask(i).Should().Equal(opt.Ask(i));
    }

    // ---- sampler: end-to-end through a Study ------------------------------------------------------

    [TestMethod]
    public void Sampler_MinimizesSphere()
    {
        var study = Optuna.CreateStudy(sampler: new XnesSampler(seed: 1));
        study.Optimize(t =>
        {
            double s = 0;
            for (int i = 0; i < 3; i++) { double x = t.SuggestFloat($"x{i}", -5, 5); s += x * x; }
            return s;
        }, nTrials: 600);

        study.BestValue.Should().BeLessThan(1e-2, "xNES converges tightly on the sphere");
    }

    [TestMethod]
    public void Sampler_CorrelatedRosenbrock_Converges()
    {
        // Rosenbrock's curved valley is non-separable; full-covariance adaptation is what lets xNES follow
        // it (the diagonal SNES cannot). The global optimum is (1, 1) with value 0.
        var study = Optuna.CreateStudy(sampler: new XnesSampler(seed: 1));
        study.Optimize(t =>
        {
            double x = t.SuggestFloat("x", -5, 5);
            double y = t.SuggestFloat("y", -5, 5);
            return (x - 1) * (x - 1) + 100 * (y - x * x) * (y - x * x);
        }, nTrials: 600);

        study.BestValue.Should().BeLessThan(0.5, "xNES's full covariance follows the correlated Rosenbrock valley");
    }

    [TestMethod]
    public void Sampler_Maximize_Works()
    {
        var study = Optuna.CreateStudy(
            direction: StudyDirection.Maximize, sampler: new XnesSampler(seed: 2));
        study.Optimize(t =>
        {
            double s = 0;
            for (int i = 0; i < 3; i++) { double x = t.SuggestFloat($"x{i}", -5, 5); s += x * x; }
            return -s;
        }, nTrials: 600);

        study.BestValue.Should().BeGreaterThan(-1e-2, "direction flipping lets xNES maximize −sphere");
    }

    [TestMethod]
    public void Sampler_SameSeed_IsReproducible()
    {
        double Run()
        {
            var study = Optuna.CreateStudy(sampler: new XnesSampler(seed: 7));
            study.Optimize(t =>
            {
                double s = 0;
                for (int i = 0; i < 3; i++) { double x = t.SuggestFloat($"x{i}", -5, 5); s += x * x; }
                return s;
            }, nTrials: 200);
            return study.BestValue;
        }

        Run().Should().Be(Run(), "a fixed seed gives an identical trajectory");
    }

    [TestMethod]
    public void Sampler_GenerationAttr_IsStamped()
    {
        var study = Optuna.CreateStudy(sampler: new XnesSampler(seed: 42));
        study.Optimize(t =>
        {
            double x = t.SuggestFloat("x", -5, 5);
            double y = t.SuggestFloat("y", -5, 5);
            return x * x + y * y;
        }, nTrials: 40);

        study.Trials.Any(tr => tr.SystemAttrs.ContainsKey("xnes:generation")).Should().BeTrue();
    }

    [TestMethod]
    public void Sampler_HighDimension_ChunksOptimizerState()
    {
        // xNES stores a d×d transform B, so at d = 12 its serialized state exceeds the 2045-char attr limit
        // and must round-trip through multiple `xnes:optimizer:i` chunks each generation. Full-covariance
        // adaptation in 12-D is slow, so the convergence bar is loose — the point is that the chunked state
        // round-trips faithfully (a corrupted chunk would stall the search near the random baseline ~30).
        var study = Optuna.CreateStudy(sampler: new XnesSampler(seed: 4));
        study.Optimize(t =>
        {
            double s = 0;
            for (int i = 0; i < 12; i++) { double x = t.SuggestFloat($"x{i}", -5, 5); s += x * x; }
            return s;
        }, nTrials: 800);

        study.Trials.Any(tr => tr.SystemAttrs.ContainsKey("xnes:optimizer:1"))
            .Should().BeTrue("a 12-D covariance exceeds the 2045-char attr limit and is chunked");
        study.BestValue.Should().BeLessThan(15.0, "the chunked state round-trips and the sampler keeps converging");
    }

    [TestMethod]
    public void Sampler_RespectsBounds()
    {
        var study = Optuna.CreateStudy(sampler: new XnesSampler(seed: 3));
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
        var study = Optuna.CreateStudy(sampler: new XnesSampler(seed: 3));
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
            sampler: new XnesSampler(seed: 1));
        Action act = () => study.Optimize(t =>
        {
            double x = t.SuggestFloat("x", -5, 5);
            return new[] { x * x, (x - 2) * (x - 2) };
        }, nTrials: 5);
        act.Should().Throw<ValueError>("xNES is single-objective only");
    }

    [TestMethod]
    public void Constructor_RejectsNonPositiveSigma()
    {
        Action act = () => new XnesSampler(sigma: -1.0);
        act.Should().Throw<ArgumentException>();
    }
}
