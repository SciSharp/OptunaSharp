using System;
using System.Linq;
using AwesomeAssertions;
using OptunaSharp.Internal;
using OptunaSharp.Samplers;
using OptunaSharp.Samplers.Nsgaii;

namespace OptunaSharp.Tests.Samplers;

/// <summary>
///     Tests for the native <see cref="SmsEmoaSampler" /> port (SMS-EMOA — Beume, Naujoks &amp;
///     Emmerich, 2007; survival per pymoo's <c>SMSEMOA</c>). SMS-EMOA has no Optuna/OptunaHub
///     counterpart, so — like <c>LmCmaSamplerTests</c> / <c>JanusSamplerTests</c> — these are
///     behavioral gates: it drives the population's hypervolume toward the ZDT1 Pareto front, matches
///     or beats NSGA-II's crowding selection on achieved hypervolume, is deterministic under a fixed
///     seed, and keeps a spread of nondominated solutions.
/// </summary>
[TestClass]
public class SmsEmoaSamplerTests
{
    private static readonly double[] Ref = { 1.1, 1.1 };

    // ---- ZDT1 (2-objective, convex Pareto front f2 = 1 - sqrt(f1)) -------------------------------

    private static double[] Zdt1(double[] x)
    {
        double f1 = x[0];
        double g = 1.0;
        for (int i = 1; i < x.Length; i++) g += 9.0 * x[i] / (x.Length - 1);
        double f2 = g * (1.0 - Math.Sqrt(f1 / g));
        return new[] { f1, f2 };
    }

    private static Study RunZdt1(BaseSampler sampler, int nTrials, int dim = 4)
    {
        var study = Optuna.CreateStudy(
            directions: new[] { StudyDirection.Minimize, StudyDirection.Minimize }, sampler: sampler);
        study.Optimize(t =>
        {
            var x = new double[dim];
            for (int i = 0; i < dim; i++) x[i] = t.SuggestFloat($"x{i}", 0, 1);
            return (System.Collections.Generic.IReadOnlyList<double>)Zdt1(x);
        }, nTrials: nTrials);
        return study;
    }

    private static double AchievedHv(Study study)
    {
        var pts = MultiObjective.GetParetoFrontTrials(study)
            .Select(t => t.Values!.ToArray())
            .Where(v => v[0] <= Ref[0] && v[1] <= Ref[1])
            .ToArray();
        return pts.Length == 0 ? 0.0 : Hypervolume.ComputeHypervolume(pts, Ref);
    }

    /// <summary>Hypervolume of a dense sampling of the true ZDT1 front — the convergence target.</summary>
    private static double IdealFrontHv()
    {
        var pts = Enumerable.Range(0, 2001)
            .Select(i => i / 2000.0)
            .Select(t => new[] { t, 1.0 - Math.Sqrt(t) })
            .ToArray();
        return Hypervolume.ComputeHypervolume(pts, Ref);
    }

    // ---- convergence -----------------------------------------------------------------------------

    [TestMethod]
    public void Sampler_ConvergesTowardZdt1Front()
    {
        var study = RunZdt1(new SmsEmoaSampler(populationSize: 40, seed: 0), nTrials: 600);
        double achieved = AchievedHv(study), ideal = IdealFrontHv();
        achieved.Should().BeGreaterThan(0.85 * ideal,
            $"SMS-EMOA drives the population's hypervolume toward the ZDT1 front (achieved {achieved:g4}, ideal {ideal:g4})");
    }

    // ---- matches or beats NSGA-II on achieved hypervolume ----------------------------------------

    [TestMethod]
    public void Sampler_MatchesOrBeatsNsgaiiHypervolume()
    {
        // Same SBX + polynomial-mutation operators on both sides, so the comparison isolates the
        // survival rule: SMS-EMOA selects by hypervolume contribution and should match or beat
        // NSGA-II's crowding distance. Aggregated over seeds to absorb single-run noise.
        const int nTrials = 300, pop = 40;
        double smsSum = 0, nsgaSum = 0;
        for (int seed = 0; seed < 3; seed++)
        {
            smsSum += AchievedHv(RunZdt1(new SmsEmoaSampler(populationSize: pop, seed: seed), nTrials));
            nsgaSum += AchievedHv(RunZdt1(
                new NSGAIISampler(populationSize: pop, crossover: new SBXCrossover(),
                    mutation: new PolynomialMutation(), seed: seed), nTrials));
        }

        smsSum.Should().BeGreaterThanOrEqualTo(nsgaSum * 0.98,
            $"SMS-EMOA matches or beats NSGA-II in aggregate hypervolume (SMS Σ={smsSum:g4}, NSGA Σ={nsgaSum:g4})");
    }

    // ---- determinism -----------------------------------------------------------------------------

    [TestMethod]
    public void Sampler_SameSeed_IsReproducible()
    {
        double Run() => AchievedHv(RunZdt1(new SmsEmoaSampler(populationSize: 24, seed: 7), nTrials: 200));
        Run().Should().Be(Run(), "a fixed seed gives an identical trajectory");
    }

    // ---- spread ----------------------------------------------------------------------------------

    [TestMethod]
    public void Sampler_KeepsSpreadOfNondominatedSolutions()
    {
        var study = RunZdt1(new SmsEmoaSampler(populationSize: 40, seed: 1), nTrials: 400);
        MultiObjective.GetParetoFrontTrials(study).Count
            .Should().BeGreaterThan(5, "SMS-EMOA keeps a spread of nondominated solutions, not a single point");
    }
}
