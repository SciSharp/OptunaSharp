using System;
using System.Collections.Generic;
using System.Linq;
using AwesomeAssertions;
using NumSharp;
using OptunaSharp.Distributions;
using OptunaSharp.Internal;
using OptunaSharp.Samplers;
using OptunaSharp.Samplers.Nsgaii;
using OptunaSharp.Samplers.Nsgaiii;

namespace OptunaSharp.Tests;

/// <summary>
///     Unit coverage for the NSGA-III sampler (PORTING_PLAN Phase 2.14). NSGA-III reuses the whole
///     NSGA-II genetic machinery and swaps the elite selection for reference-point niching. The
///     hardcoded sequence is the exact output of Optuna 5.0.0.dev; the port reproduces it bit-for-bit.
///     Validation/warning behavior mirrors Optuna's <c>tests/samplers_tests/test_nsgaiii.py</c>.
/// </summary>
[TestClass]
public class NsgaiiiSamplerTests
{
    // ---- bit-exact reference sequence (real Optuna 5.0.0.dev, pop=4, seed=42) ------------------
    // For a single-objective study every non-domination rank holds one trial, so the borderline
    // front never overflows and NSGA-III fills by rank exactly like NSGA-II — this locks the shared
    // generation/child machinery under the NSGA-III sampler.

    [TestMethod]
    public void SingleObjective_MatchesOptunaSequence()
    {
        var study = Optuna.CreateStudy(
            sampler: new NSGAIIISampler(populationSize: 4, seed: 42), direction: StudyDirection.Minimize);
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

        trials.Take(4).Should().OnlyContain(t => (int)Convert.ToInt64(t.SystemAttrs["NSGAIIISampler:generation"]) == 0);
        trials.Skip(4).Take(4).Should().OnlyContain(t => (int)Convert.ToInt64(t.SystemAttrs["NSGAIIISampler:generation"]) == 1);
        trials.Skip(8).Take(4).Should().OnlyContain(t => (int)Convert.ToInt64(t.SystemAttrs["NSGAIIISampler:generation"]) == 2);
    }

    // ---- default reference points (combinations_with_replacement) ------------------------------

    [TestMethod]
    public void GenerateDefaultReferencePoint_TwoObjectives_MatchesOptuna()
    {
        var rp = NSGAIIIElitePopulationSelectionStrategy.GenerateDefaultReferencePoint(2, 3);
        ToJagged(rp).Should().BeEquivalentTo(new[]
        {
            new[] { 3.0, 0.0 }, new[] { 2.0, 1.0 }, new[] { 1.0, 2.0 }, new[] { 0.0, 3.0 },
        }, o => o.WithStrictOrdering());
    }

    [TestMethod]
    public void GenerateDefaultReferencePoint_ThreeObjectives_MatchesOptuna()
    {
        var rp = NSGAIIIElitePopulationSelectionStrategy.GenerateDefaultReferencePoint(3, 3);
        ToJagged(rp).Should().BeEquivalentTo(new[]
        {
            new[] { 3.0, 0.0, 0.0 }, new[] { 2.0, 1.0, 0.0 }, new[] { 2.0, 0.0, 1.0 },
            new[] { 1.0, 2.0, 0.0 }, new[] { 1.0, 1.0, 1.0 }, new[] { 1.0, 0.0, 2.0 },
            new[] { 0.0, 3.0, 0.0 }, new[] { 0.0, 2.0, 1.0 }, new[] { 0.0, 1.0, 2.0 },
            new[] { 0.0, 0.0, 3.0 },
        }, o => o.WithStrictOrdering());
    }

    [TestMethod]
    public void GenerateDefaultReferencePoint_DividingParameterTwo_MatchesOptuna()
    {
        var rp = NSGAIIIElitePopulationSelectionStrategy.GenerateDefaultReferencePoint(2, 2);
        ToJagged(rp).Should().BeEquivalentTo(new[]
        {
            new[] { 2.0, 0.0 }, new[] { 1.0, 1.0 }, new[] { 0.0, 2.0 },
        }, o => o.WithStrictOrdering());
    }

    // ---- reproducibility (exercises reference-point niching) -----------------------------------

    [TestMethod]
    public void MultiObjectiveReproducible_SameSeed_ProducesIdenticalValues()
    {
        List<double[]> Run()
        {
            var study = Optuna.CreateStudy(
                sampler: new NSGAIIISampler(populationSize: 5, seed: 1),
                directions: new[] { StudyDirection.Minimize, StudyDirection.Maximize });
            study.Optimize(t =>
            {
                double x = t.SuggestFloat("x", -5, 5);
                double y = t.SuggestFloat("y", -5, 5);
                return (IReadOnlyList<double>)new[] { x * x + y, -y };
            }, nTrials: 25);
            return study.Trials.Select(t => t.Values!.ToArray()).ToList();
        }

        var a = Run();
        var b = Run();
        a.Should().HaveCount(b.Count);
        for (int i = 0; i < a.Count; i++)
            a[i].Should().Equal(b[i]);
    }

    [DataTestMethod]
    [DataRow("uniform")]
    [DataRow("blx")]
    [DataRow("sbx")]
    [DataRow("spx")]
    public void Reproducible_SameSeed_ProducesIdenticalTrials(string kind)
    {
        BaseCrossover MakeCx() => kind switch
        {
            "uniform" => new UniformCrossover(),
            "blx" => new BLXAlphaCrossover(),
            "sbx" => new SBXCrossover(),
            "spx" => new SPXCrossover(),
            _ => throw new ArgumentException(kind),
        };

        List<(double, double, double)> Run()
        {
            var study = Optuna.CreateStudy(
                sampler: new NSGAIIISampler(populationSize: 6, seed: 3, crossover: MakeCx()),
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
    public void CustomReferencePoints_Reproducible()
    {
        var referencePoints = new double[,] { { 1.0, 0.0 }, { 0.5, 0.5 }, { 0.0, 1.0 } };

        List<double[]> Run()
        {
            var study = Optuna.CreateStudy(
                sampler: new NSGAIIISampler(populationSize: 5, seed: 7, referencePoints: referencePoints),
                directions: new[] { StudyDirection.Minimize, StudyDirection.Maximize });
            study.Optimize(t =>
            {
                double x = t.SuggestFloat("x", -5, 5);
                double y = t.SuggestFloat("y", -5, 5);
                return (IReadOnlyList<double>)new[] { x * x + y, -y };
            }, nTrials: 15);
            return study.Trials.Select(t => t.Values!.ToArray()).ToList();
        }

        var a = Run();
        var b = Run();
        for (int i = 0; i < a.Count; i++)
            a[i].Should().Equal(b[i]);
    }

    // ---- validation ---------------------------------------------------------------------------

    [TestMethod]
    public void PopulationSizeBelowTwo_Throws()
        => ((Action)(() => new NSGAIIISampler(populationSize: 1)))
            .Should().Throw<ValueError>().WithMessage("*population_size*greater than or equal to 2*");

    [TestMethod]
    public void PopulationSizeBelowNParents_Throws()
        => ((Action)(() => new NSGAIIISampler(populationSize: 2, crossover: new SPXCrossover())))
            .Should().Throw<ValueError>().WithMessage("*population size*greater than or equal to 3*");

    [TestMethod]
    public void WrongDimensionReferencePoints_Throws()
    {
        // 3-column reference points for a 2-objective study: the mismatch surfaces once the borderline
        // front overflows (the first child generation), propagating out of optimize.
        var referencePoints = new double[,] { { 1.0, 0.0, 0.0 }, { 0.0, 1.0, 0.0 }, { 0.0, 0.0, 1.0 } };
        var study = Optuna.CreateStudy(
            sampler: new NSGAIIISampler(populationSize: 5, seed: 0, referencePoints: referencePoints),
            directions: new[] { StudyDirection.Minimize, StudyDirection.Maximize });

        ((Action)(() => study.Optimize(t =>
                (IReadOnlyList<double>)new[] { Math.Pow(t.SuggestFloat("x", -5, 5), 2), -t.SuggestFloat("y", -5, 5) },
            nTrials: 12)))
            .Should().Throw<ValueError>().WithMessage("*dimension of reference points*same as the number of objectives*");
    }

    // ---- warnings -----------------------------------------------------------------------------

    // [Misaligned] Optuna decorates NSGAIIISampler with @experimental_class("3.2.0"), which raises an
    // ExperimentalWarning on construction. OptunaSharp targets the Optuna 5.x surface and treats this
    // sampler as stable, so no ExperimentalWarning is emitted — this guards that deliberate divergence.
    [TestMethod]
    [Misaligned]
    public void Constructor_DoesNotEmitExperimentalWarning()
    {
        string? captured = CaptureWarning(
            WarningCategory.ExperimentalWarning, () => new NSGAIIISampler(populationSize: 5));
        captured.Should().BeNull();
    }

    [TestMethod]
    public void ConstraintsFunc_EmitsFutureWarning()
    {
        string? captured = CaptureWarning(
            WarningCategory.FutureWarning,
            () => new NSGAIIISampler(constraintsFunc: _ => Array.Empty<double>()));
        captured.Should().NotBeNull();
        captured.Should().Contain("constraints_func");
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

    private static double[][] ToJagged(double[,] a)
    {
        int rows = a.GetLength(0), cols = a.GetLength(1);
        var result = new double[rows][];
        for (int i = 0; i < rows; i++)
        {
            result[i] = new double[cols];
            for (int j = 0; j < cols; j++)
                result[i][j] = a[i, j];
        }
        return result;
    }
}
