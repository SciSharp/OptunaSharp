using System;
using System.Collections.Generic;
using System.Linq;
using AwesomeAssertions;
using NumSharp;
using OptunaSharp.Internal;
using OptunaSharp.Samplers;
using OptunaSharp.Samplers.Moead;
using OptunaSharp.Samplers.Nsgaii;

namespace OptunaSharp.Tests.Samplers;

/// <summary>
///     Tests for the native <see cref="MoeaDSampler" /> port (MOEA/D — Zhang &amp; Li, 2007; structure per
///     pymoo's <c>MOEAD</c>). MOEA/D has no Optuna/OptunaHub counterpart, so — like
///     <c>SmsEmoaSamplerTests</c> / <c>LmCmaSamplerTests</c> — these are behavioral gates: it drives the
///     population's hypervolume toward the ZDT1 Pareto front, beats NSGA-II's crowding selection on
///     achieved hypervolume, is deterministic under a fixed seed, keeps a spread of nondominated
///     solutions, and each scalarizing function converges. Thresholds are calibrated to measured
///     convergence, not machine precision.
/// </summary>
[TestClass]
public class MoeaDSamplerTests
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
            return (IReadOnlyList<double>)Zdt1(x);
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
        var study = RunZdt1(new MoeaDSampler(populationSize: 40, nNeighbors: 12, seed: 0), nTrials: 600);
        double achieved = AchievedHv(study), ideal = IdealFrontHv();
        achieved.Should().BeGreaterThan(0.85 * ideal,
            $"MOEA/D drives the population's hypervolume toward the ZDT1 front (achieved {achieved:g4}, ideal {ideal:g4})");
    }

    // ---- beats NSGA-II on achieved hypervolume ---------------------------------------------------

    [TestMethod]
    public void Sampler_BeatsNsgaiiHypervolume()
    {
        // Same SBX + polynomial-mutation operators on both sides, so the comparison isolates the
        // selection/replacement: MOEA/D's decomposition should match or beat NSGA-II's crowding distance.
        // Aggregated over seeds to absorb single-run noise.
        const int nTrials = 300, pop = 40;
        double moeadSum = 0, nsgaSum = 0;
        for (int seed = 0; seed < 3; seed++)
        {
            moeadSum += AchievedHv(RunZdt1(new MoeaDSampler(populationSize: pop, nNeighbors: 12, seed: seed), nTrials));
            nsgaSum += AchievedHv(RunZdt1(
                new NSGAIISampler(populationSize: pop, crossover: new SBXCrossover(),
                    mutation: new PolynomialMutation(), seed: seed), nTrials));
        }

        moeadSum.Should().BeGreaterThanOrEqualTo(nsgaSum * 0.98,
            $"MOEA/D matches or beats NSGA-II in aggregate hypervolume (MOEA/D Σ={moeadSum:g4}, NSGA Σ={nsgaSum:g4})");
    }

    // ---- determinism -----------------------------------------------------------------------------

    [TestMethod]
    public void Sampler_SameSeed_IsReproducible()
    {
        double Run() => AchievedHv(RunZdt1(new MoeaDSampler(populationSize: 40, nNeighbors: 12, seed: 7), nTrials: 300));
        double a = Run();
        a.Should().BeGreaterThan(0.0, "the determinism config must reach the reference region (a non-trivial value to compare)");
        Run().Should().Be(a, "a fixed seed gives an identical trajectory");
    }

    // ---- spread ----------------------------------------------------------------------------------

    [TestMethod]
    public void Sampler_KeepsSpreadOfNondominatedSolutions()
    {
        var study = RunZdt1(new MoeaDSampler(populationSize: 40, nNeighbors: 12, seed: 1), nTrials: 400);
        MultiObjective.GetParetoFrontTrials(study).Count
            .Should().BeGreaterThan(5, "MOEA/D keeps a spread of nondominated solutions, not a single point");
    }

    // ---- scalarizing functions -------------------------------------------------------------------

    [TestMethod]
    public void Sampler_Pbi_Converges()
    {
        var study = RunZdt1(
            new MoeaDSampler(populationSize: 40, nNeighbors: 12, scalarization: MoeadScalarization.Pbi, seed: 0),
            nTrials: 600);
        AchievedHv(study).Should().BeGreaterThan(0.6 * IdealFrontHv(),
            "the PBI decomposition also drives the population toward the ZDT1 front");
    }

    [TestMethod]
    public void Sampler_WeightedSum_Converges()
    {
        // Weighted sum cannot resolve a concave front but ZDT1 is convex, so it should still converge.
        var study = RunZdt1(
            new MoeaDSampler(populationSize: 40, nNeighbors: 12, scalarization: MoeadScalarization.WeightedSum, seed: 0),
            nTrials: 600);
        AchievedHv(study).Should().BeGreaterThan(0.6 * IdealFrontHv(),
            "the weighted-sum decomposition converges on the convex ZDT1 front");
    }

    // ---- single-objective fallback ---------------------------------------------------------------

    [TestMethod]
    public void Sampler_SingleObjective_FallsBackToRandom()
    {
        // On a single objective MOEA/D degenerates, so it falls back to the independent (random) sampler
        // and must still run and make progress rather than throw.
        var study = Optuna.CreateStudy(sampler: new MoeaDSampler(seed: 0));
        study.Optimize(t =>
        {
            double x = t.SuggestFloat("x", -5, 5), y = t.SuggestFloat("y", -5, 5);
            return x * x + y * y;
        }, nTrials: 80);
        study.BestValue.Should().BeLessThan(25.0, "the random fallback still explores the single-objective space");
    }

    // ---- validation ------------------------------------------------------------------------------

    [TestMethod]
    public void Constructor_RejectsInvalidArguments()
    {
        ((Action)(() => new MoeaDSampler(populationSize: 1))).Should().Throw<ValueError>();
        ((Action)(() => new MoeaDSampler(nNeighbors: 0))).Should().Throw<ValueError>();
        ((Action)(() => new MoeaDSampler(neighborSelectionProb: 1.5))).Should().Throw<ValueError>();
        ((Action)(() => new MoeaDSampler(nReplacements: 0))).Should().Throw<ValueError>();
    }
}
