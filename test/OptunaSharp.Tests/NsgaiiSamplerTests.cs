using System;
using System.Collections.Generic;
using System.Linq;
using AwesomeAssertions;
using NumSharp;
using OptunaSharp.Distributions;
using OptunaSharp.Internal;
using OptunaSharp.Samplers;
using OptunaSharp.Samplers.Nsgaii;

namespace OptunaSharp.Tests;

/// <summary>
///     Unit coverage for the NSGA-II sampler (PORTING_PLAN Phase 2.13). The hardcoded sequences are
///     the exact output of Optuna 5.0.0.dev; the port reproduces them bit-for-bit. Validation/warning
///     behavior mirrors Optuna's <c>tests/samplers_tests/test_nsgaii.py</c>.
/// </summary>
[TestClass]
public class NsgaiiSamplerTests
{
    // ---- bit-exact reference sequence (real Optuna 5.0.0.dev, pop=4, seed=42) -----------------

    [TestMethod]
    public void DefaultSingleObjective_MatchesOptunaSequence()
    {
        var study = Optuna.CreateStudy(
            sampler: new NSGAIISampler(populationSize: 4, seed: 42), direction: StudyDirection.Minimize);
        study.Optimize(t =>
        {
            double x = t.SuggestFloat("x", -5, 5);
            long y = t.SuggestInt("y", 0, 10);
            t.SuggestCategorical("z", new CategoricalChoice[] { "a", "b", "c" });
            return Math.Pow(x - 2, 2) + y;
        }, nTrials: 12);

        var trials = study.Trials;
        double[] expectedX =
        {
            -1.254598811526375, -3.4400547966379733, -4.7941550570419755, -3.165954901465662,
            -1.254598811526375, -3.6050613934795814, -3.165954901465662, -3.4400547966379733,
            -1.254598811526375, -3.4400547966379733, -1.254598811526375, -3.165954901465662,
        };
        long[] expectedY = { 10, 0, 10, 3, 6, 3, 0, 10, 0, 0, 0, 10 };
        string[] expectedZ = { "a", "a", "a", "a", "a", "a", "a", "c", "c", "b", "a", "a" };

        for (int i = 0; i < 12; i++)
        {
            BitConverter.DoubleToInt64Bits((double)trials[i].Params["x"]!)
                .Should().Be(BitConverter.DoubleToInt64Bits(expectedX[i]), $"x at trial {i}");
            Convert.ToInt64(trials[i].Params["y"]).Should().Be(expectedY[i], $"y at trial {i}");
            CategoricalChoice.From(trials[i].Params["z"]).Value.Should().Be(expectedZ[i], $"z at trial {i}");
        }

        // Generation attribution: 4 per generation for population_size 4.
        trials.Take(4).Should().OnlyContain(t => (int)Convert.ToInt64(t.SystemAttrs["NSGAIISampler:generation"]) == 0);
        trials.Skip(4).Take(4).Should().OnlyContain(t => (int)Convert.ToInt64(t.SystemAttrs["NSGAIISampler:generation"]) == 1);
        trials.Skip(8).Take(4).Should().OnlyContain(t => (int)Convert.ToInt64(t.SystemAttrs["NSGAIISampler:generation"]) == 2);
    }

    // ---- reproducibility ----------------------------------------------------------------------

    [DataTestMethod]
    [DataRow("uniform")]
    [DataRow("blx")]
    [DataRow("sbx")]
    [DataRow("vsbx")]
    [DataRow("spx")]
    [DataRow("undx")]
    public void Reproducible_SameSeed_ProducesIdenticalTrials(string kind)
    {
        BaseCrossover MakeCx() => kind switch
        {
            "uniform" => new UniformCrossover(),
            "blx" => new BLXAlphaCrossover(),
            "sbx" => new SBXCrossover(),
            "vsbx" => new VSBXCrossover(),
            "spx" => new SPXCrossover(),
            "undx" => new UNDXCrossover(),
            _ => throw new ArgumentException(kind),
        };

        List<(double, double, double)> Run()
        {
            var study = Optuna.CreateStudy(
                sampler: new NSGAIISampler(populationSize: 6, seed: 3, crossover: MakeCx()),
                directions: new[] { StudyDirection.Minimize, StudyDirection.Maximize });
            study.Optimize(t =>
            {
                double a = t.SuggestFloat("a", -5, 5);
                double b = t.SuggestFloat("b", -5, 5);
                double c = t.SuggestFloat("c", -5, 5);
                return (IReadOnlyList<double>)new[] { a * a + b * b + c * c, a - b + c };
            }, nTrials: 18);
            return study.Trials
                .Select(t => ((double)t.Params["a"]!, (double)t.Params["b"]!, (double)t.Params["c"]!))
                .ToList();
        }

        Run().Should().Equal(Run());
    }

    [TestMethod]
    public void MultiObjectiveReproducible_SameSeed_ProducesIdenticalValues()
    {
        List<double[]> Run()
        {
            var study = Optuna.CreateStudy(
                sampler: new NSGAIISampler(populationSize: 5, seed: 1),
                directions: new[] { StudyDirection.Minimize, StudyDirection.Maximize });
            study.Optimize(t =>
            {
                double x = t.SuggestFloat("x", -5, 5);
                double y = t.SuggestFloat("y", -5, 5);
                return (IReadOnlyList<double>)new[] { x * x + y, -y };
            }, nTrials: 20);
            return study.Trials.Select(t => t.Values!.ToArray()).ToList();
        }

        var a = Run();
        var b = Run();
        a.Should().HaveCount(b.Count);
        for (int i = 0; i < a.Count; i++)
            a[i].Should().Equal(b[i]);
    }

    // ---- validation ---------------------------------------------------------------------------

    [TestMethod]
    public void PopulationSizeBelowTwo_Throws()
        => ((Action)(() => new NSGAIISampler(populationSize: 1)))
            .Should().Throw<ValueError>().WithMessage("*population_size*greater than or equal to 2*");

    [TestMethod]
    public void PopulationSizeBelowNParents_Throws()
        => ((Action)(() => new NSGAIISampler(populationSize: 2, crossover: new SPXCrossover())))
            .Should().Throw<ValueError>().WithMessage("*population size*greater than or equal to 3*");

    [DataTestMethod]
    [DataRow(1.5)]
    [DataRow(-0.1)]
    public void CrossoverProbOutOfRange_Throws(double prob)
        => ((Action)(() => new NSGAIISampler(crossoverProb: prob)))
            .Should().Throw<ValueError>().WithMessage("*crossover_prob*");

    [TestMethod]
    public void MutationProbOutOfRange_Throws()
        => ((Action)(() => new NSGAIISampler(mutationProb: 2.0)))
            .Should().Throw<ValueError>().WithMessage("*mutation_prob*");

    [TestMethod]
    public void SwappingProbOutOfRange_Throws()
        => ((Action)(() => new NSGAIISampler(swappingProb: -0.5)))
            .Should().Throw<ValueError>().WithMessage("*swapping_prob*");

    [TestMethod]
    public void UniformCrossover_InvalidSwappingProb_Throws()
        => ((Action)(() => new UniformCrossover(1.2)))
            .Should().Throw<ValueError>().WithMessage("*swapping_prob*");

    [TestMethod]
    public void SbxCrossover_NegativeEta_Throws()
        => ((Action)(() => new SBXCrossover(eta: -1.0)))
            .Should().Throw<ValueError>().WithMessage("*eta*");

    [TestMethod]
    public void PolynomialMutation_NegativeEta_Throws()
        => ((Action)(() => new PolynomialMutation(eta: -1.0)))
            .Should().Throw<ValueError>().WithMessage("*eta*non-negative*");

    // ---- warnings -----------------------------------------------------------------------------

    [TestMethod]
    public void ConstraintsFunc_EmitsFutureWarning()
    {
        string? captured = CaptureWarning(
            WarningCategory.FutureWarning,
            () => new NSGAIISampler(constraintsFunc: _ => Array.Empty<double>()));
        captured.Should().NotBeNull();
        captured.Should().Contain("constraints_func");
    }

    [TestMethod]
    public void ExperimentalAfterTrialStrategy_EmitsExperimentalWarning()
    {
        string? captured = CaptureWarning(
            WarningCategory.ExperimentalWarning,
            () => new NSGAIISampler(afterTrialStrategy: (_, _, _, _) => { }));
        captured.Should().NotBeNull();
        captured.Should().Contain("after_trial_strategy");
    }

    // ---- polynomial mutation degenerate case --------------------------------------------------

    [TestMethod]
    public void PolynomialMutation_ZeroWidth_ReturnsParam()
    {
        var rng = np.random.RandomState();
        var study = Optuna.CreateStudy(direction: StudyDirection.Minimize);
        double result = new PolynomialMutation().Mutation(3.5, rng, study, new[] { 2.0, 2.0 });
        result.Should().Be(3.5);
    }

    private static string? CaptureWarning(WarningCategory wanted, Action action)
    {
        string? captured = null;
        void Handler(string msg, WarningCategory cat)
        {
            if (cat == wanted && captured is null)
                captured = msg;
        }

        OptunaWarnings.WarningRaised += Handler;
        try
        {
            action();
        }
        finally
        {
            OptunaWarnings.WarningRaised -= Handler;
        }
        return captured;
    }
}
