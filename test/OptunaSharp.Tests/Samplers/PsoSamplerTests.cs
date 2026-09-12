using System;
using System.Linq;
using System.Text.Json;
using AwesomeAssertions;
using OptunaSharp.Samplers;
using OptunaSharp.Samplers.ParticleSwarm;

namespace OptunaSharp.Tests.Samplers;

/// <summary>
///     Tests for the native <see cref="Pso" /> Particle Swarm optimizer (Kennedy &amp; Eberhart 1995;
///     Shi &amp; Eberhart 1998 inertia weight) and its <see cref="PsoSampler" />. PSO has no
///     <c>cmaes</c>/Optuna/OptunaHub counterpart, so — like <c>DifferentialEvolutionSamplerTests</c> —
///     these are behavioral gates: convergence, same-seed determinism, optimizer-state round-trip
///     fidelity, direction flipping and bound respect. Thresholds are calibrated to measured convergence
///     (Plan 07 baseline), not machine precision.
/// </summary>
[TestClass]
public class PsoSamplerTests
{
    /// <summary>A sphere shifted so its optimum sits at 0.5 in every coordinate (inside the unit box the
    ///     optimizer works in, once the sampler's transform maps a real search space to [0,1]^d).</summary>
    private static double Sphere(double[] x)
    {
        double s = 0;
        foreach (double v in x) { double d = v - 0.5; s += d * d; }
        return s;
    }

    /// <summary>A random initial swarm in the unit box [0,1]^d with its sphere values.</summary>
    private static Pso MakeSphereSwarm(int d, int np, int seed)
    {
        var rng = new Random(seed * 7 + 1);   // test scaffolding only — NOT the optimizer's RNG
        var pop = new double[np][];
        var val = new double[np];
        for (int i = 0; i < np; i++)
        {
            var row = new double[d];
            for (int j = 0; j < d; j++) row[j] = rng.NextDouble();
            pop[i] = row;
            val[i] = Sphere(row);
        }
        return new Pso(pop, val, seed: seed);
    }

    /// <summary>Drive the optimizer as a batch ask/tell loop, clamping to the unit box (PSO is unconstrained).</summary>
    private static double Drive(Pso opt, int generations)
    {
        double best = double.PositiveInfinity;
        int np = opt.PopulationSize;
        for (int g = 0; g < generations; g++)
        {
            var sols = new (double[] X, double Value)[np];
            for (int i = 0; i < np; i++)
            {
                double[] x = opt.AskCandidate(i);
                for (int j = 0; j < x.Length; j++) x[j] = Math.Min(Math.Max(x[j], 0.0), 1.0);
                double v = Sphere(x);
                if (v < best) best = v;
                sols[i] = (x, v);
            }
            opt.Tell(sols);
        }
        return best;
    }

    // ---- optimizer: convergence -------------------------------------------------------------------

    [TestMethod]
    public void Optimizer_Sphere_Converges()
    {
        double best = Drive(MakeSphereSwarm(8, 12, seed: 1), generations: 500);
        best.Should().BeLessThan(1e-4, "the swarm drives the 8-D sphere down to a small value");
    }

    // ---- optimizer: determinism & state round-trip ------------------------------------------------

    [TestMethod]
    public void Optimizer_SameSeed_IsReproducible()
    {
        double a = Drive(MakeSphereSwarm(6, 10, seed: 11), generations: 50);
        double b = Drive(MakeSphereSwarm(6, 10, seed: 11), generations: 50);
        b.Should().Be(a, "same seed and initial swarm ⇒ identical trajectory");
    }

    [TestMethod]
    public void Optimizer_StateRoundTrip_PreservesEverything()
    {
        var opt = MakeSphereSwarm(10, 12, seed: 5);
        Drive(opt, generations: 40);   // evolve positions/velocities/bests/generation counter

        // Serialize → deserialize exactly as PsoSampler stores optimizer state.
        string json = JsonSerializer.Serialize(opt.ToState());
        var restored = Pso.FromState(JsonSerializer.Deserialize<PsoState>(json)!);

        restored.Dim.Should().Be(opt.Dim);
        restored.PopulationSize.Should().Be(opt.PopulationSize);
        restored.Generation.Should().Be(opt.Generation);

        // After an identical reseed, the next candidate must be bit-identical — proving positions,
        // velocities and personal/global bests round-tripped, not just the scalars.
        opt.ReseedRng(999);
        restored.ReseedRng(999);
        restored.AskCandidate(0).Should().Equal(opt.AskCandidate(0));
    }

    [TestMethod]
    public void Optimizer_RejectsTooSmallPopulation()
    {
        Action act = () => new Pso(new[] { new double[2] }, new double[1]);
        act.Should().Throw<ArgumentException>("PSO needs at least 2 particles");
    }

    [TestMethod]
    public void Optimizer_InertiaWeight_DecaysThenHolds()
    {
        var opt = MakeSphereSwarm(4, 6, seed: 2);   // defaults w0=0.9, w1=0.4, decay over 100 generations
        opt.InertiaWeight(0).Should().BeApproximately(0.9, 1e-12, "the schedule starts at w");
        opt.InertiaWeight(50).Should().BeApproximately(0.65, 1e-12, "it decays linearly to the midpoint");
        opt.InertiaWeight(100).Should().BeApproximately(0.4, 1e-12, "it reaches wMin at the horizon");
        opt.InertiaWeight(500).Should().BeApproximately(0.4, 1e-12, "it holds at wMin past the horizon");
    }

    // ---- sampler: end-to-end through a Study ------------------------------------------------------

    [TestMethod]
    public void Sampler_MinimizesSphere()
    {
        var study = Optuna.CreateStudy(sampler: new PsoSampler(seed: 1));
        study.Optimize(t =>
        {
            double s = 0;
            for (int i = 0; i < 6; i++) { double x = t.SuggestFloat($"x{i}", -5, 5); s += x * x; }
            return s;
        }, nTrials: 700);

        study.BestValue.Should().BeLessThan(1e-1, "the sampler drives the swarm to a small sphere value");
    }

    [TestMethod]
    public void Sampler_Maximize_Works()
    {
        var study = Optuna.CreateStudy(
            direction: StudyDirection.Maximize, sampler: new PsoSampler(seed: 2));
        study.Optimize(t =>
        {
            double s = 0;
            for (int i = 0; i < 6; i++) { double x = t.SuggestFloat($"x{i}", -5, 5); s += x * x; }
            return -s;   // optimum 0, approached from below
        }, nTrials: 700);

        study.BestValue.Should().BeGreaterThan(-1e-1, "direction flipping lets the swarm maximize -sphere");
    }

    [TestMethod]
    public void Sampler_SameSeed_IsReproducible()
    {
        double Run()
        {
            var study = Optuna.CreateStudy(sampler: new PsoSampler(seed: 7));
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
        var study = Optuna.CreateStudy(sampler: new PsoSampler(seed: 42));
        study.Optimize(t =>
        {
            double x = t.SuggestFloat("x", -5, 5);
            double y = t.SuggestFloat("y", -5, 5);
            return x * x + y * y;
        }, nTrials: 60);

        // Trials past the initial-swarm phase carry the optimizer's generation and target-index stamps
        // (the pairing the swarm update needs).
        study.Trials.Any(tr => tr.SystemAttrs.ContainsKey("pso:generation")).Should().BeTrue();
        study.Trials.Any(tr => tr.SystemAttrs.ContainsKey("pso:index")).Should().BeTrue();
    }

    [TestMethod]
    public void Sampler_RespectsBounds()
    {
        var study = Optuna.CreateStudy(sampler: new PsoSampler(seed: 3));
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
