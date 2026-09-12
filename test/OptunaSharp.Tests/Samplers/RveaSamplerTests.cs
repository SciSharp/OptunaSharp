using System;
using System.Collections.Generic;
using System.Linq;
using AwesomeAssertions;
using NumSharp;
using OptunaSharp.Internal;
using OptunaSharp.Samplers;
using OptunaSharp.Samplers.Nsgaii;

namespace OptunaSharp.Tests.Samplers;

/// <summary>
///     Tests for the native <see cref="RveaSampler" /> port (RVEA — Cheng et al., 2016; structure per
///     pymoo's <c>RVEA</c>). RVEA has no Optuna/OptunaHub counterpart, so — like <c>SmsEmoaSamplerTests</c>
///     / <c>LmCmaSamplerTests</c> — these are behavioral gates: it drives the population's hypervolume
///     toward the ZDT1 front, matches NSGA-II's crowding selection on 2 objectives (RVEA's strength is
///     many-objective), converges with a good spread on 3-objective DTLZ2, and is deterministic under a
///     fixed seed (through the reference-vector adaptation). Thresholds are calibrated to measured
///     convergence.
/// </summary>
[TestClass]
public class RveaSamplerTests
{
    private static readonly double[] Ref = { 1.1, 1.1 };

    // ---- ZDT1 (2-objective) ----------------------------------------------------------------------

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

    private static double IdealFrontHv()
    {
        var pts = Enumerable.Range(0, 2001)
            .Select(i => i / 2000.0)
            .Select(t => new[] { t, 1.0 - Math.Sqrt(t) })
            .ToArray();
        return Hypervolume.ComputeHypervolume(pts, Ref);
    }

    // ---- DTLZ2 (3-objective; the true Pareto front is the unit sphere ‖f‖ = 1) -------------------

    private static double[] Dtlz2(double[] x, int m)
    {
        int n = x.Length, k = n - m + 1;
        double g = 0.0;
        for (int i = n - k; i < n; i++) { double d = x[i] - 0.5; g += d * d; }
        var f = new double[m];
        for (int i = 0; i < m; i++)
        {
            double prod = 1.0 + g;
            for (int j = 0; j < m - 1 - i; j++) prod *= Math.Cos(x[j] * Math.PI / 2.0);
            if (i > 0) prod *= Math.Sin(x[m - 1 - i] * Math.PI / 2.0);
            f[i] = prod;
        }
        return f;
    }

    private static Study RunDtlz2(BaseSampler sampler, int nTrials, int dim = 7)
    {
        var study = Optuna.CreateStudy(
            directions: Enumerable.Repeat(StudyDirection.Minimize, 3).ToArray(), sampler: sampler);
        study.Optimize(t =>
        {
            var x = new double[dim];
            for (int i = 0; i < dim; i++) x[i] = t.SuggestFloat($"x{i}", 0, 1);
            return (IReadOnlyList<double>)Dtlz2(x, 3);
        }, nTrials: nTrials);
        return study;
    }

    // ---- convergence (2-objective) ---------------------------------------------------------------

    [TestMethod]
    public void Sampler_ConvergesTowardZdt1Front()
    {
        var study = RunZdt1(new RveaSampler(populationSize: 40, nMaxGenerations: 15, seed: 0), nTrials: 600);
        double achieved = AchievedHv(study), ideal = IdealFrontHv();
        achieved.Should().BeGreaterThan(0.80 * ideal,
            $"RVEA drives the population's hypervolume toward the ZDT1 front (achieved {achieved:g4}, ideal {ideal:g4})");
    }

    // ---- matches NSGA-II on achieved hypervolume (2-objective) -----------------------------------

    [TestMethod]
    public void Sampler_MatchesNsgaiiHypervolume()
    {
        // RVEA's edge is many-objective; on two objectives it should still match NSGA-II's crowding
        // selection under the same SBX + polynomial-mutation operators. Aggregated over seeds.
        const int nTrials = 300, pop = 40;
        double rveaSum = 0, nsgaSum = 0;
        for (int seed = 0; seed < 3; seed++)
        {
            rveaSum += AchievedHv(RunZdt1(new RveaSampler(populationSize: pop, nMaxGenerations: 15, seed: seed), nTrials));
            nsgaSum += AchievedHv(RunZdt1(
                new NSGAIISampler(populationSize: pop, crossover: new SBXCrossover(),
                    mutation: new PolynomialMutation(), seed: seed), nTrials));
        }

        rveaSum.Should().BeGreaterThanOrEqualTo(nsgaSum * 0.95,
            $"RVEA matches NSGA-II in aggregate hypervolume on 2 objectives (RVEA Σ={rveaSum:g4}, NSGA Σ={nsgaSum:g4})");
    }

    // ---- many-objective: DTLZ2 (3 objectives) ----------------------------------------------------

    [TestMethod]
    public void Sampler_ThreeObjective_ConvergesWithSpread()
    {
        var study = RunDtlz2(new RveaSampler(populationSize: 36, nMaxGenerations: 10, seed: 1), nTrials: 400);
        var front = MultiObjective.GetParetoFrontTrials(study).Select(t => t.Values!.ToArray()).ToArray();

        front.Length.Should().BeGreaterThan(20, "RVEA keeps a broad spread on the 3-objective DTLZ2 front");
        double avgDist = front.Average(v => Math.Abs(Math.Sqrt(v.Sum(a => a * a)) - 1.0));
        avgDist.Should().BeLessThan(0.25,
            $"the front approaches the DTLZ2 unit-sphere true front (avg |‖f‖-1| = {avgDist:g4})");
    }

    // ---- determinism (exercises reference-vector adaptation, which fires every 2 generations) ----

    [TestMethod]
    public void Sampler_SameSeed_IsReproducible()
    {
        double Run() => AchievedHv(RunZdt1(new RveaSampler(populationSize: 40, nMaxGenerations: 15, seed: 7), nTrials: 300));
        double a = Run();
        a.Should().BeGreaterThan(0.0, "the determinism config must reach the reference region (a non-trivial value to compare)");
        Run().Should().Be(a, "a fixed seed gives an identical trajectory, adaptation included");
    }

    // ---- spread (2-objective) --------------------------------------------------------------------

    [TestMethod]
    public void Sampler_KeepsSpreadOfNondominatedSolutions()
    {
        var study = RunZdt1(new RveaSampler(populationSize: 40, nMaxGenerations: 15, seed: 1), nTrials: 400);
        MultiObjective.GetParetoFrontTrials(study).Count
            .Should().BeGreaterThan(5, "RVEA keeps a spread of nondominated solutions, not a single point");
    }

    // ---- single-objective fallback ---------------------------------------------------------------

    [TestMethod]
    public void Sampler_SingleObjective_FallsBackToRandom()
    {
        var study = Optuna.CreateStudy(sampler: new RveaSampler(seed: 0));
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
        ((Action)(() => new RveaSampler(populationSize: 1))).Should().Throw<ValueError>();
        ((Action)(() => new RveaSampler(alpha: -1.0))).Should().Throw<ValueError>();
        ((Action)(() => new RveaSampler(adaptFreq: 2.0))).Should().Throw<ValueError>();
        ((Action)(() => new RveaSampler(nMaxGenerations: 0))).Should().Throw<ValueError>();
    }
}
