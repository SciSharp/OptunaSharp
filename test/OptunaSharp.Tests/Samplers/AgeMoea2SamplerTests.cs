using System;
using System.Collections.Generic;
using System.Linq;
using AwesomeAssertions;
using OptunaSharp;
using OptunaSharp.Samplers;

namespace OptunaSharp.Tests.Samplers;

/// <summary>
///     Behavioral coverage for <see cref="AgeMoea2Sampler" /> (Layer 2 of the AGE-MOEA-2 port). Unlike
///     the Layer-1 <c>AgeMoea2Optimizer</c> (generation-by-generation bit-exact vs pymoo — see
///     <c>AgeMoea2OperatorParityTests</c>), the sampler brings AGE-MOEA-2's survival to Optuna's
///     define-by-run workflow with Optuna's own operators, so it has no pymoo oracle: it is validated
///     behaviorally (valid trials, generation bookkeeping, reproducibility under a seed, and that it
///     actually drives a two-objective front).
/// </summary>
[TestClass]
public class AgeMoea2SamplerTests
{
    /// <summary>min f0 = Σx², f1 = Σ(x-1)² over [0,1]³ — a convex two-objective trade-off.</summary>
    private static IReadOnlyList<double> TwoObjective(Trial t)
    {
        double f0 = 0, f1 = 0;
        for (int k = 0; k < 3; k++)
        {
            double x = t.SuggestFloat($"x{k}", 0.0, 1.0);
            f0 += x * x;
            double d = x - 1.0;
            f1 += d * d;
        }
        return new[] { f0, f1 };
    }

    [TestMethod]
    public void RunsAndAttributesGenerations()
    {
        var study = Optuna.CreateStudy(
            sampler: new AgeMoea2Sampler(populationSize: 10, seed: 42),
            directions: new[] { StudyDirection.Minimize, StudyDirection.Minimize });
        study.Optimize(TwoObjective, nTrials: 40);

        var trials = study.Trials;
        trials.Should().HaveCount(40);
        trials.Should().OnlyContain(t => t.State == TrialState.Complete, "every trial completes");

        // 10 trials per generation (population_size = 10), namespaced by the concrete sampler name.
        for (int gen = 0; gen < 4; gen++)
            trials.Skip(gen * 10).Take(10).Should().OnlyContain(
                t => (int)Convert.ToInt64(t.SystemAttrs["AgeMoea2Sampler:generation"]) == gen,
                $"generation {gen} attribution");
    }

    [TestMethod]
    public void DrivesTheParetoFront()
    {
        var study = Optuna.CreateStudy(
            sampler: new AgeMoea2Sampler(populationSize: 20, seed: 7),
            directions: new[] { StudyDirection.Minimize, StudyDirection.Minimize });
        study.Optimize(TwoObjective, nTrials: 300);

        // A genuine multi-objective front: several mutually non-dominated best trials.
        study.BestTrials.Count.Should().BeGreaterThan(1, "AGE-MOEA-2 maintains a spread Pareto front");

        var trials = study.Trials;
        // AGE-MOEA-2 balances the whole front rather than racing to the corners, but it should still
        // improve both extremes well beyond the initial random generation (population_size = 20).
        double gen0MinF0 = trials.Take(20).Min(t => t.Values![0]);
        double gen0MinF1 = trials.Take(20).Min(t => t.Values![1]);
        double minF0 = trials.Min(t => t.Values![0]);
        double minF1 = trials.Min(t => t.Values![1]);
        minF0.Should().BeLessThan(gen0MinF0, "the sampler improves the f0 extreme (x -> 0) beyond random");
        minF1.Should().BeLessThan(gen0MinF1, "the sampler improves the f1 extreme (x -> 1) beyond random");
        minF0.Should().BeLessThan(0.3, "the sampler approaches the f0 extreme");
        minF1.Should().BeLessThan(0.3, "the sampler approaches the f1 extreme");
    }

    [TestMethod]
    public void Reproducible_SameSeed_ProducesIdenticalTrials()
    {
        List<double[]> Run()
        {
            var study = Optuna.CreateStudy(
                sampler: new AgeMoea2Sampler(populationSize: 12, seed: 123),
                directions: new[] { StudyDirection.Minimize, StudyDirection.Minimize });
            study.Optimize(TwoObjective, nTrials: 60);
            return study.Trials
                .Select(t => new[] { (double)t.Params["x0"]!, (double)t.Params["x1"]!, (double)t.Params["x2"]! })
                .ToList();
        }

        var a = Run();
        var b = Run();
        a.Should().HaveCount(b.Count);
        for (int i = 0; i < a.Count; i++)
            for (int j = 0; j < 3; j++)
                BitConverter.DoubleToInt64Bits(a[i][j])
                    .Should().Be(BitConverter.DoubleToInt64Bits(b[i][j]), $"trial {i} x{j} reproducible under seed");
    }

    [TestMethod]
    public void RejectsPopulationSizeBelowTwo()
    {
        var act = () => new AgeMoea2Sampler(populationSize: 1);
        act.Should().Throw<NumSharp.ValueError>();
    }
}
