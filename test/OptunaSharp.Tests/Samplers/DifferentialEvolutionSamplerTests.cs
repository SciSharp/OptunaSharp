using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json;
using AwesomeAssertions;
using OptunaSharp.Samplers;
using OptunaSharp.Samplers.Differential;

namespace OptunaSharp.Tests.Samplers;

/// <summary>
///     Tests for the native <see cref="De" /> Differential Evolution optimizer (Storn &amp; Price 1997) and
///     its <see cref="DifferentialEvolutionSampler" />. DE has no <c>cmaes</c>/Optuna/OptunaHub counterpart,
///     so — like <c>LmCmaSamplerTests</c> — these are behavioral gates: convergence, same-seed determinism,
///     optimizer-state round-trip fidelity, and the two selection schemes (classic one-to-one greedy and
///     the order-independent (μ+λ) elitist selection the JANUS host uses). Thresholds are calibrated to
///     measured convergence, not machine precision (DE is a global explorer, deliberately slower than CMA).
/// </summary>
[TestClass]
public class DifferentialEvolutionSamplerTests
{
    private static double Sphere(double[] x)
    {
        double s = 0;
        foreach (double v in x) s += v * v;
        return s;
    }

    /// <summary>A random initial population in the box [lo, hi]^d with its sphere values.</summary>
    private static De MakeSphereDe(int d, int np, int seed, DeStrategy strategy, double lo = -5, double hi = 5)
    {
        var rng = new Random(seed * 7 + 1);   // test scaffolding only — NOT the optimizer's RNG
        var pop = new double[np][];
        var val = new double[np];
        for (int i = 0; i < np; i++)
        {
            var row = new double[d];
            for (int j = 0; j < d; j++) row[j] = rng.NextDouble() * (hi - lo) + lo;
            pop[i] = row;
            val[i] = Sphere(row);
        }
        return new De(pop, val, f: 0.8, cr: 0.9, strategy: strategy, seed: seed);
    }

    /// <summary>Drive the optimizer as a batch ask/tell loop, clamping to the box (DE is unconstrained).</summary>
    private static double Drive(De opt, Func<double[], double> f, int generations, bool elitist = false,
        double lo = -5, double hi = 5)
    {
        double best = double.PositiveInfinity;
        int np = opt.PopulationSize;
        for (int g = 0; g < generations; g++)
        {
            var sols = new (double[] X, double Value)[np];
            for (int i = 0; i < np; i++)
            {
                double[] x = opt.AskCandidate(i);
                for (int j = 0; j < x.Length; j++) x[j] = Math.Min(Math.Max(x[j], lo), hi);
                double v = f(x);
                if (v < best) best = v;
                sols[i] = (x, v);
            }
            if (elitist) opt.TellElitist(sols); else opt.Tell(sols);
        }
        return best;
    }

    // ---- optimizer: convergence -------------------------------------------------------------------

    [TestMethod]
    public void Optimizer_Rand1_Sphere_Converges()
    {
        double best = Drive(MakeSphereDe(8, 12, seed: 1, DeStrategy.Rand1Bin), Sphere, generations: 500);
        best.Should().BeLessThan(1e-4, "DE/rand/1/bin drives the 8-D sphere down to a small value");
    }

    [TestMethod]
    public void Optimizer_Best1_Sphere_Converges()
    {
        double best = Drive(MakeSphereDe(8, 12, seed: 2, DeStrategy.Best1Bin), Sphere, generations: 500);
        best.Should().BeLessThan(1e-4, "DE/best/1/bin converges (the greedier strategy)");
    }

    [TestMethod]
    public void Optimizer_ElitistSelection_Sphere_Converges()
    {
        // The (μ+λ) elitist selection the JANUS host uses must also converge (order-independent selection).
        double best = Drive(MakeSphereDe(8, 12, seed: 3, DeStrategy.Rand1Bin), Sphere, generations: 500,
            elitist: true);
        best.Should().BeLessThan(1e-4, "the (μ+λ) elitist selection converges too");
    }

    // ---- optimizer: determinism & state round-trip ------------------------------------------------

    [TestMethod]
    public void Optimizer_SameSeed_IsReproducible()
    {
        double a = Drive(MakeSphereDe(6, 10, seed: 11, DeStrategy.Rand1Bin), Sphere, generations: 50);
        double b = Drive(MakeSphereDe(6, 10, seed: 11, DeStrategy.Rand1Bin), Sphere, generations: 50);
        b.Should().Be(a, "same seed and initial population ⇒ identical trajectory");
    }

    [TestMethod]
    public void Optimizer_StateRoundTrip_PreservesEverything()
    {
        var opt = MakeSphereDe(10, 12, seed: 5, DeStrategy.Rand1Bin);
        Drive(opt, Sphere, generations: 40);   // evolve the population/generation counter

        // Serialize → deserialize exactly as DifferentialEvolutionSampler stores optimizer state.
        string json = JsonSerializer.Serialize(opt.ToState());
        var restored = De.FromState(JsonSerializer.Deserialize<DeState>(json)!);

        restored.Dim.Should().Be(opt.Dim);
        restored.PopulationSize.Should().Be(opt.PopulationSize);
        restored.Generation.Should().Be(opt.Generation);

        // After an identical reseed, the next candidate must be bit-identical — proving the population and
        // values round-tripped, not just the scalars.
        opt.ReseedRng(999);
        restored.ReseedRng(999);
        restored.AskCandidate(0).Should().Equal(opt.AskCandidate(0));
    }

    [TestMethod]
    public void Optimizer_RejectsTooSmallPopulation()
    {
        Action act = () => new De(new[] { new double[2], new double[2], new double[2] }, new double[3]);
        act.Should().Throw<ArgumentException>("DE needs at least 4 members (target + 3 donors)");
    }

    // ---- sampler: end-to-end through a Study ------------------------------------------------------

    [TestMethod]
    public void Sampler_MinimizesSphere()
    {
        var study = Optuna.CreateStudy(sampler: new DifferentialEvolutionSampler(seed: 1));
        study.Optimize(t =>
        {
            double s = 0;
            for (int i = 0; i < 6; i++) { double x = t.SuggestFloat($"x{i}", -5, 5); s += x * x; }
            return s;
        }, nTrials: 700);

        study.BestValue.Should().BeLessThan(5e-2, "the sampler drives DE to a small sphere value");
    }

    [TestMethod]
    public void Sampler_Maximize_Works()
    {
        var study = Optuna.CreateStudy(
            direction: StudyDirection.Maximize, sampler: new DifferentialEvolutionSampler(seed: 2));
        study.Optimize(t =>
        {
            double s = 0;
            for (int i = 0; i < 6; i++) { double x = t.SuggestFloat($"x{i}", -5, 5); s += x * x; }
            return -s;   // optimum 0, approached from below
        }, nTrials: 700);

        study.BestValue.Should().BeGreaterThan(-5e-2, "direction flipping lets DE maximize -sphere");
    }

    [TestMethod]
    public void Sampler_SameSeed_IsReproducible()
    {
        double Run()
        {
            var study = Optuna.CreateStudy(sampler: new DifferentialEvolutionSampler(seed: 7));
            study.Optimize(t =>
            {
                double s = 0;
                for (int i = 0; i < 5; i++) { double x = t.SuggestFloat($"x{i}", -5, 5); s += x * x; }
                return s;
            }, nTrials: 150);
            return study.BestValue;
        }

        Run().Should().Be(Run(), "a fixed seed gives an identical trajectory");
    }

    [TestMethod]
    public void Sampler_GenerationAndIndexAttrs_AreAttributedPerTrial()
    {
        var study = Optuna.CreateStudy(sampler: new DifferentialEvolutionSampler(seed: 42));
        study.Optimize(t =>
        {
            double x = t.SuggestFloat("x", -5, 5);
            double y = t.SuggestFloat("y", -5, 5);
            return x * x + y * y;
        }, nTrials: 60);

        // Trials past the initial-population phase carry the optimizer's generation and target-index stamps
        // (the pairing DE's greedy selection needs).
        study.Trials.Any(tr => tr.SystemAttrs.ContainsKey("de:generation")).Should().BeTrue();
        study.Trials.Any(tr => tr.SystemAttrs.ContainsKey("de:index")).Should().BeTrue();
    }

    [TestMethod]
    public void Sampler_RespectsBounds()
    {
        var study = Optuna.CreateStudy(sampler: new DifferentialEvolutionSampler(seed: 3));
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
}
