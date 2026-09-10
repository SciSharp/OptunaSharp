using System;
using System.Linq;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using OptunaSharp.Benchmark.VolleyEval;
using OptunaSharp.Benchmark.VolleyEval.Challenges;
using OptunaSharp.Logging;
using OptunaSharp.Samplers;

namespace OptunaSharp.Tests.Benchmark.VolleyEval;

/// <summary>
///     v0 Definition-of-Done smoke tests for the three-radical VolleyEval suite. Everything here is
///     deterministic (fixed seeds → fixed samplers), so the assertions use generous margins that the
///     validated numbers clear comfortably. The full eight-contender matrix (which adds the cubic GP) is
///     exercised by <c>volley run --assert</c>; these tests use the cheap subset so the suite stays fast.
/// </summary>
[TestClass]
public sealed class VolleyEvalTests
{
    [ClassInitialize]
    public static void Init(TestContext _)
    {
        // Silence Optuna's independent-sampling / experimental log lines during tests.
        OptunaLogging.DisableDefaultHandler();
    }

    // ---- Challenge correctness (the exact normalization bands) ----

    [DataTestMethod]
    [DataRow(14, 2)]
    [DataRow(12, 3)]
    [DataRow(16, 1)]
    public void NkLandscape_DpOptimum_MatchesBruteForce(int n, int k)
    {
        var nk = new NkLandscape(n: n, k: k, instanceId: 3);

        double brute = double.NegativeInfinity;
        var b = new int[n];
        for (int mask = 0; mask < (1 << n); mask++)
        {
            for (int i = 0; i < n; i++) b[i] = (mask >> i) & 1;
            brute = Math.Max(brute, nk.Fitness(b));
        }

        nk.OptimumF.Should().BeApproximately(brute, 1e-9,
            "the ring DP must equal the brute-force maximum over all 2^N bit-strings");
    }

    [TestMethod]
    public void Rosenbrock_OptimumIsExactlyReachable_AndBaselineIsLarge()
    {
        var rb = new Rosenbrock(dimension: 40, instanceId: 0);

        rb.Value(rb.OptimumLocation()).Should().BeApproximately(0.0, 1e-9,
            "f(x_opt) = 0 by construction (z = 1 at the shift)");
        rb.Optimum.Should().Be(0.0);
        rb.RandomBaseline.Should().BeGreaterThan(1000.0,
            "the uniform-sampling mean of a D=40 Rosenbrock is large and positive");
    }

    [TestMethod]
    public void Zdt1_TrueFrontHypervolume_MatchesNumericIntegration()
    {
        var z = new Zdt1(dimension: 30, instanceId: 0);

        // Dense Riemann sum of the dominated area of f2 = 1 - sqrt(f1), f1 in [0,1], to ref (1.1,1.1):
        //   ∫_0^1 (1.1 - (1 - sqrt(a))) da  +  strip a in [1, 1.1] of height 1.1.
        const int steps = 2_000_000;
        double r = 1.1, area = 0.0, dx = 1.0 / steps;
        for (int i = 0; i < steps; i++)
        {
            double a = (i + 0.5) * dx;
            area += (r - (1.0 - Math.Sqrt(a))) * dx;
        }
        area += (r - 1.0) * r; // the a in [1, 1.1] strip

        z.TrueFrontHypervolume().Should().BeApproximately(area, 1e-3);
        z.Optimum.Should().BeApproximately(0.87667, 1e-3);
    }

    [TestMethod]
    public void Zdt1_KnownPointObjectives_AreCorrect()
    {
        var z = new Zdt1(dimension: 10, instanceId: 0);
        var origin = new double[10]; // all zeros: f1 = 0, g = 1, f2 = 1

        var o = z.Objectives(origin);
        o[0].Should().Be(0.0);
        o[1].Should().BeApproximately(1.0, 1e-12);
    }

    // ---- Inert & DNF are first-class results ----

    [TestMethod]
    public void CmaFamily_IsInert_OnCategorical_Nk()
    {
        var nk = new NkLandscape(n: 24, k: 2, instanceId: 0, budget: 120);
        var ev = new Evaluator(new[] { 0, 1, 2 });

        foreach (var name in new[] { "cmaes", "lmcma" })
        {
            var cell = ev.EvaluateCell(nk, ContenderRegistry.Select(new[] { name })[0]);
            cell.Inert.Should().BeTrue($"{name} has no categorical move and must fall back bit-for-bit to RandomSampler");
            cell.Dnf.Should().BeFalse();
        }

        // TPE and QMC genuinely engage the categorical space — not inert.
        foreach (var name in new[] { "tpe", "qmc" })
        {
            var cell = ev.EvaluateCell(nk, ContenderRegistry.Select(new[] { name })[0]);
            cell.Inert.Should().BeFalse($"{name} handles categoricals and must diverge from the random fallback");
        }
    }

    [TestMethod]
    public void CmaFamily_Dnf_OnMultiObjective_Zdt1()
    {
        var z = new Zdt1(dimension: 10, instanceId: 0, budget: 50);
        var ev = new Evaluator(new[] { 0, 1 });

        foreach (var name in new[] { "cmaes", "lmcma" })
        {
            var cell = ev.EvaluateCell(z, ContenderRegistry.Select(new[] { name })[0]);
            cell.Dnf.Should().BeTrue($"{name} does not support multi-objective studies");
            cell.Error.Should().NotBeNull();
        }
    }

    [TestMethod]
    public void Gp_Runs_AndBeatsRandom_OnSmallRosenbrock()
    {
        var rb = new Rosenbrock(dimension: 6, instanceId: 0, budget: 40);
        var ev = new Evaluator(new[] { 0, 1 });

        var gp = ev.EvaluateCell(rb, ContenderRegistry.Select(new[] { "gp" })[0]);
        gp.Dnf.Should().BeFalse();
        gp.Median.Should().BeGreaterThan(0.2, "GP is a strong low-D optimizer and must clear the random baseline");
    }

    // ---- The three-radical thesis (the headline DoD) ----

    [TestMethod]
    public void ThreeRadical_ProfileMatrix_HasDistinctColumnWinners_AndPassesAllDodGates()
    {
        // Fast budgets (smaller than the CLI defaults) that still reproduce every gate deterministically.
        var challenges = new IChallenge[]
        {
            new Rosenbrock(dimension: 40, instanceId: 0, budget: 200),
            new NkLandscape(n: 24, k: 2, instanceId: 0, budget: 150),
            new Zdt1(dimension: 30, instanceId: 0, budget: 3000),
        };
        var ev = new Evaluator(new[] { 0, 1, 2 });

        var matrix = ev.Run(challenges, ContenderRegistry.Cheap);

        // Central claim: no two radicals share a champion.
        matrix.WinnersDistinct().Should().BeTrue(
            "the three challenges are orthogonal radicals — each has a different best sampler: {0}",
            string.Join(", ", matrix.Challenges.Select(c => $"{c.Name}={matrix.ColumnWinner(c.Name)}")));

        matrix.ColumnWinner("rosenbrock").Should().BeOneOf("cmaes", "lmcma");
        matrix.ColumnWinner("zdt1").Should().BeOneOf("nsga2", "nsga3");

        // Every applicable DoD gate passes (home>random for all; home>away for Rosenbrock & ZDT1).
        var gates = DodGate.Evaluate(matrix);
        gates.Should().OnlyContain(g => g.Passed,
            "DoD gates: {0}", string.Join("; ", gates.Select(g => $"{g.Name}={(g.Passed ? "PASS" : "FAIL")}")));
        DodGate.AllPassed(matrix).Should().BeTrue();
    }
}
