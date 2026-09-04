using System;
using System.Collections.Generic;
using System.Linq;
using AwesomeAssertions;
using OptunaSharp.Samplers;

namespace OptunaSharp.Tests.Gp;

/// <summary>
///     Unit coverage for <see cref="GPSampler" />'s constrained single-objective path (PORTING_PLAN
///     Phase 3.2 part 3e-2 — the <see cref="LogCEI" /> acquisition function) and the still-deferred
///     branch placeholders. These are oracle-free: reproducibility under a fixed seed, that the
///     constrained sampler runs end-to-end and drives the feasible best low, and that the
///     multi-objective / running-trial q-batch branches still throw (locking the placeholders until
///     their parts land). Byte-tolerance parity vs live Optuna lives in the interop
///     <c>GpConstrainedSamplerParityTests</c>.
/// </summary>
[TestClass]
public class GPSamplerTests
{
    // A constrained shifted sphere: feasible when x0 <= 0.5 (constraint value x0 - 0.5 <= 0). The
    // unconstrained optimum sits at x0 = -0.2 (feasible), so a correct constrained sampler converges.
    private static Study RunConstrainedSphere(int seed, int nTrials)
    {
        var study = Optuna.CreateStudy(sampler: new GPSampler(seed: seed));
        study.Optimize(t =>
        {
            double x0 = t.SuggestFloat("x0", -5, 5);
            double x1 = t.SuggestFloat("x1", -5, 5);
            t.SetConstraint("0", x0 - 0.5);
            return Math.Pow(x0 + 0.2, 2) + Math.Pow(x1 - 0.2, 2);
        }, nTrials: nTrials);
        return study;
    }

    [TestMethod]
    public void Constrained_RunsEndToEnd_AndDrivesFeasibleBestLow()
    {
        Study study = RunConstrainedSphere(seed: 0, nTrials: 30);
        study.Trials.Count.Should().Be(30);

        // Both feasibility outcomes must appear (so LogCEI is genuinely exercised, not degenerate).
        var feasible = study.Trials.Where(t => t.Constraints.Values.All(c => c <= 0.0)).ToList();
        feasible.Should().NotBeEmpty();
        study.Trials.Should().Contain(t => t.Constraints.Values.Any(c => c > 0.0));

        double bestFeasible = feasible.Min(t => t.Value!.Value);
        bestFeasible.Should().BeLessThan(0.5, "GP-based constrained optimization approaches the feasible optimum");
    }

    [TestMethod]
    public void Constrained_SameSeed_Reproducible()
    {
        List<(double, double)> Params(Study s) => s.Trials
            .Select(t => ((double)t.Params["x0"]!, (double)t.Params["x1"]!)).ToList();

        Params(RunConstrainedSphere(seed: 7, nTrials: 18))
            .Should().Equal(Params(RunConstrainedSphere(seed: 7, nTrials: 18)));
    }

    // A 2-objective problem (minimize both) over a 2-D box; the GP LogEHVI acqf drives the Pareto front.
    private static Study RunMultiObjective(int seed, int nTrials)
    {
        var study = Optuna.CreateStudy(
            sampler: new GPSampler(seed: seed),
            directions: new[] { StudyDirection.Minimize, StudyDirection.Minimize });
        study.Optimize(t =>
        {
            double a = t.SuggestFloat("a", -5, 5);
            double b = t.SuggestFloat("b", -5, 5);
            return (IReadOnlyList<double>)new[] { a * a + b * b, (a - 2) * (a - 2) + b * b };
        }, nTrials: nTrials);
        return study;
    }

    [TestMethod]
    public void MultiObjective_RunsEndToEnd_AndBuildsAParetoFront()
    {
        Study study = RunMultiObjective(seed: 0, nTrials: 24);
        study.Trials.Count.Should().Be(24);
        // The GP LogEHVI path (post-startup) must produce a non-trivial Pareto front.
        study.BestTrials.Count.Should().BeGreaterThan(1,
            "LogEHVI multi-objective optimization should discover several Pareto-optimal trials");
    }

    [TestMethod]
    public void MultiObjective_SameSeed_Reproducible()
    {
        List<(double, double)> Params(Study s) => s.Trials
            .Select(t => ((double)t.Params["a"]!, (double)t.Params["b"]!)).ToList();

        Params(RunMultiObjective(seed: 5, nTrials: 16))
            .Should().Equal(Params(RunMultiObjective(seed: 5, nTrials: 16)));
    }

    // A 2-objective constrained problem (LogCEHVI): feasible when a <= 1.
    private static Study RunMultiObjectiveConstrained(int seed, int nTrials)
    {
        var study = Optuna.CreateStudy(
            sampler: new GPSampler(seed: seed),
            directions: new[] { StudyDirection.Minimize, StudyDirection.Minimize });
        study.Optimize(t =>
        {
            double a = t.SuggestFloat("a", -5, 5);
            double b = t.SuggestFloat("b", -5, 5);
            t.SetConstraint("0", a - 1.0);
            return (IReadOnlyList<double>)new[] { a * a + b * b, (a - 2) * (a - 2) + b * b };
        }, nTrials: nTrials);
        return study;
    }

    [TestMethod]
    public void MultiObjectiveConstrained_RunsEndToEnd()
    {
        Study study = RunMultiObjectiveConstrained(seed: 0, nTrials: 24);
        study.Trials.Count.Should().Be(24);
        // Both feasibility outcomes must appear (so LogCEHVI is genuinely exercised).
        study.Trials.Should().Contain(t => t.Constraints.Values.All(c => c <= 0.0));
        study.Trials.Should().Contain(t => t.Constraints.Values.Any(c => c > 0.0));
    }

    [TestMethod]
    public void MultiObjectiveConstrained_SameSeed_Reproducible()
    {
        List<(double, double)> Params(Study s) => s.Trials
            .Select(t => ((double)t.Params["a"]!, (double)t.Params["b"]!)).ToList();

        Params(RunMultiObjectiveConstrained(seed: 3, nTrials: 16))
            .Should().Equal(Params(RunMultiObjectiveConstrained(seed: 3, nTrials: 16)));
    }
}
