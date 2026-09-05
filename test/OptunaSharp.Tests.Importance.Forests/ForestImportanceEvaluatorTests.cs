using System;
using System.Collections.Generic;
using System.Linq;
using AwesomeAssertions;
using Microsoft.ML.Trainers.FastTree;
using NumSharp;
using OptunaSharp.Importance;
using OptunaSharp.Importance.Forests;
using OptunaSharp.Samplers;

namespace OptunaSharp.Tests.Importance.Forests;

/// <summary>
///     Behavioural tests for the forest-based importance evaluators — run across <b>both backends</b>:
///     the pure-managed, scikit-learn-bit-exact native CART (<see cref="FanovaImportanceEvaluator" /> /
///     <see cref="MeanDecreaseImpurityImportanceEvaluator" />, Core) and ML.NET's FastForest
///     (<see cref="FastForestFanovaImportanceEvaluator" /> /
///     <see cref="FastForestMeanDecreaseImpurityImportanceEvaluator" />, this satellite). These assert
///     the algorithm's invariants (reproducibility, hyperparameter effects, normalization, ordering,
///     single-value / degenerate handling, the multi-objective guard) — value parity vs Optuna is in
///     the interop suite. Adapted from Optuna's <c>tests/importance_tests/test_evaluator.py</c>.
/// </summary>
[TestClass]
public class ForestImportanceEvaluatorTests
{
    private static double Objective(Trial t)
    {
        double x = t.SuggestFloat("x", -10, 10);
        double y = t.SuggestFloat("y", -10, 10);
        long z = t.SuggestInt("z", 0, 10);
        return 10.0 * x * x + 0.001 * y + 0.0001 * z;
    }

    private static Study MakeStudy(int seed = 0, int nTrials = 60)
    {
        var study = Optuna.CreateStudy(sampler: new RandomSampler(seed));
        study.Optimize(Objective, nTrials: nTrials);
        return study;
    }

    private sealed record Backend(string Name, bool IsFanova, Func<int, int, int?, BaseImportanceEvaluator> Make);

    private static readonly Backend[] Backends =
    {
        new("native-fanova", true, (nt, md, s) => new FanovaImportanceEvaluator(nt, md, s)),
        new("native-mdi", false, (nt, md, s) => new MeanDecreaseImpurityImportanceEvaluator(nt, md, s)),
        new("fastforest-fanova", true, (nt, md, s) => new FastForestFanovaImportanceEvaluator(nt, md, s)),
        new("fastforest-mdi", false, (nt, md, s) => new FastForestMeanDecreaseImpurityImportanceEvaluator(nt, md, s)),
    };

    private static bool SameImportances(IReadOnlyDictionary<string, double> a, IReadOnlyDictionary<string, double> b)
    {
        if (a.Count != b.Count) return false;
        foreach (var (k, v) in a)
            if (!b.TryGetValue(k, out double w) || Math.Abs(v - w) > 1e-12) return false;
        return true;
    }

    [TestMethod]
    public void SameSeed_IsReproducible()
    {
        var study = MakeStudy(seed: 0);
        foreach (var b in Backends)
        {
            var a = b.Make(64, 64, 2).Evaluate(study);
            var c = b.Make(64, 64, 2).Evaluate(study);
            SameImportances(a, c).Should().BeTrue($"{b.Name} with the same seed must be reproducible");
        }
    }

    [TestMethod]
    public void DifferentSeed_Differs()
    {
        var study = MakeStudy(seed: 0);
        foreach (var b in Backends)
            SameImportances(b.Make(64, 64, 2).Evaluate(study), b.Make(64, 64, 3).Evaluate(study))
                .Should().BeFalse($"{b.Name} with different seeds should differ");
    }

    [TestMethod]
    public void NTrees_AffectsResult()
    {
        var study = MakeStudy(seed: 0);
        foreach (var b in Backends)
            SameImportances(b.Make(10, 64, 0).Evaluate(study), b.Make(30, 64, 0).Evaluate(study))
                .Should().BeFalse($"{b.Name}: n_trees should affect the result");
    }

    [TestMethod]
    public void MaxDepth_AffectsResult()
    {
        var study = MakeStudy(seed: 0);
        foreach (var b in Backends)
            SameImportances(b.Make(64, 1, 0).Evaluate(study), b.Make(64, 8, 0).Evaluate(study))
                .Should().BeFalse($"{b.Name}: max_depth should affect the result");
    }

    [TestMethod]
    public void Normalized_SumsToOne_AndDescending()
    {
        var study = MakeStudy(seed: 0);
        foreach (var b in Backends)
        {
            var imp = ImportanceFunctions.GetParamImportances(study, b.Make(64, 64, 0));
            imp.Values.Sum().Should().BeApproximately(1.0, 1e-9, b.Name);
            var values = imp.Values.ToList();
            for (int i = 1; i < values.Count; i++)
                values[i].Should().BeLessThanOrEqualTo(values[i - 1] + 1e-12, $"{b.Name} importances are descending");
        }
    }

    [TestMethod]
    public void DominantParam_RankedFirst()
    {
        var study = MakeStudy(seed: 0, nTrials: 80);
        foreach (var b in Backends)
            ImportanceFunctions.GetParamImportances(study, b.Make(64, 64, 0)).Keys.First()
                .Should().Be("x", $"{b.Name}: x dominates the objective (10*x^2)");
    }

    [TestMethod]
    public void MultiObjective_WithoutTarget_Throws()
    {
        var study = MakeMultiObjectiveStudy();
        foreach (var b in Backends)
        {
            var act = () => b.Make(64, 64, 0).Evaluate(study);
            act.Should().Throw<ValueError>().WithMessage("*multi-objective*", $"{b.Name} MO guard");
        }
    }

    [TestMethod]
    public void MultiObjective_WithTarget_Works()
    {
        var study = MakeMultiObjectiveStudy();
        foreach (var b in Backends)
        {
            var imp = ImportanceFunctions.GetParamImportances(study, b.Make(64, 64, 0), target: t => t.Values![0]);
            imp.Values.Sum().Should().BeApproximately(1.0, 1e-9, b.Name);
            imp.Keys.Should().Contain(new[] { "a", "b" });
        }
    }

    [TestMethod]
    public void SingleTrial_ReturnsZeros()
    {
        var study = MakeStudy(seed: 0, nTrials: 1);
        foreach (var b in Backends)
            b.Make(64, 64, 0).Evaluate(study).Values.Should().OnlyContain(v => v == 0.0, b.Name);
    }

    [TestMethod]
    public void EmptyParams_ReturnsEmpty()
    {
        var study = MakeStudy(seed: 0);
        foreach (var b in Backends)
            b.Make(64, 64, 0).Evaluate(study, @params: Array.Empty<string>()).Should().BeEmpty(b.Name);
    }

    [TestMethod]
    public void Fanova_SingleValueDistribution_IsZero()
    {
        var study = Optuna.CreateStudy(sampler: new RandomSampler(0));
        study.Optimize((Func<Trial, double>)(t =>
        {
            double x = t.SuggestFloat("x", -10, 10);
            long k = t.SuggestInt("k", 5, 5);  // single-valued
            return 10.0 * x * x + 0.0 * k;
        }), nTrials: 60);

        foreach (var b in Backends.Where(b => b.IsFanova))
        {
            var imp = b.Make(64, 64, 0).Evaluate(study);
            imp["k"].Should().Be(0.0, $"{b.Name}: a single-valued distribution has importance exactly 0");
            imp["x"].Should().BeGreaterThan(0.0, b.Name);
        }
    }

    // ---- Full-API options overload (FastForest only; the native Core evaluators have no ML.NET
    //      trainer options — they are the value-exact scikit-learn port). ----

    private static FastForestRegressionTrainer.Options FfOptions(int numberOfLeaves = 64, int minPerLeaf = 1, int seed = 0)
        => new()
        {
            NumberOfTrees = 64,
            NumberOfLeaves = numberOfLeaves,
            MinimumExampleCountPerLeaf = minPerLeaf,
            FeatureFraction = 1.0,
            NumberOfThreads = 1,
            Seed = seed,
            FeatureSelectionSeed = seed,
            AllowEmptyTrees = true,
            ShuffleLabels = false,
        };

    [TestMethod]
    public void Options_Overload_RanksNormalizesAndReproducible()
    {
        var study = MakeStudy(seed: 0, nTrials: 80);
        var evaluators = new (string Name, Func<BaseImportanceEvaluator> Make)[]
        {
            ("fanova", () => new FastForestFanovaImportanceEvaluator(FfOptions())),
            ("mdi", () => new FastForestMeanDecreaseImpurityImportanceEvaluator(FfOptions())),
        };
        foreach (var (name, make) in evaluators)
        {
            var imp = ImportanceFunctions.GetParamImportances(study, make());
            imp.Keys.First().Should().Be("x", $"options-{name}: x dominates the objective (10*x^2)");
            imp.Values.Sum().Should().BeApproximately(1.0, 1e-9, name);
            SameImportances(make().Evaluate(study), make().Evaluate(study))
                .Should().BeTrue($"options-{name}: a fixed options config (Seed + 1 thread) is reproducible");
        }
    }

    [TestMethod]
    public void Options_ExposesControlBeyondSimpleCtor()
    {
        var study = MakeStudy(seed: 0);
        // MinimumExampleCountPerLeaf is pinned to 1 by the simple (n_trees, max_depth, seed) ctor; via
        // the options overload it is free — a larger value coarsens the forest and changes importances.
        var fine = new FastForestFanovaImportanceEvaluator(FfOptions(minPerLeaf: 1)).Evaluate(study);
        var coarse = new FastForestFanovaImportanceEvaluator(FfOptions(minPerLeaf: 15)).Evaluate(study);
        SameImportances(fine, coarse).Should()
            .BeFalse("MinimumExampleCountPerLeaf (a knob the simple ctor pins to 1) must reach the fit");
    }

    [TestMethod]
    public void Configure_Overload_TweaksDefaultsRanksAndReproducible()
    {
        var study = MakeStudy(seed: 0, nTrials: 80);
        // Start from the reproducible defaults and tweak only what we care about.
        BaseImportanceEvaluator Make() => new FastForestFanovaImportanceEvaluator(
            o => { o.MinimumExampleCountPerLeaf = 15; o.Seed = 0; o.NumberOfThreads = 1; });

        var imp = ImportanceFunctions.GetParamImportances(study, Make());
        imp.Keys.First().Should().Be("x", "configure: x dominates the objective (10*x^2)");
        imp.Values.Sum().Should().BeApproximately(1.0, 1e-9);
        SameImportances(Make().Evaluate(study), Make().Evaluate(study)).Should().BeTrue("configure is reproducible");
        SameImportances(Make().Evaluate(study), new FastForestFanovaImportanceEvaluator(seed: 0).Evaluate(study))
            .Should().BeFalse("the configure tweak (MinimumExampleCountPerLeaf=15) must reach the fit");
    }

    [TestMethod]
    public void Options_UnsupportedColumns_ThrowClearly()
    {
        var study = MakeStudy(seed: 0);
        // The importance schema has only label + features; the trainer's other two column-name fields
        // (of four) cannot bind, so the evaluator rejects them with a clear message rather than a
        // cryptic ML.NET "column not found" deep in Fit.
        var weighted = FfOptions();
        weighted.ExampleWeightColumnName = "w";
        Action act1 = () => new FastForestFanovaImportanceEvaluator(weighted).Evaluate(study);
        act1.Should().Throw<ArgumentException>().WithMessage("*ExampleWeightColumnName*");

        var grouped = FfOptions();
        grouped.RowGroupColumnName = "g";
        Action act2 = () => new FastForestFanovaImportanceEvaluator(grouped).Evaluate(study);
        act2.Should().Throw<ArgumentException>().WithMessage("*RowGroupColumnName*");
    }

    private static Study MakeMultiObjectiveStudy()
    {
        var study = Optuna.CreateStudy(
            directions: new[] { StudyDirection.Minimize, StudyDirection.Minimize },
            sampler: new RandomSampler(0));
        study.Optimize((Func<Trial, IReadOnlyList<double>>)(t =>
        {
            double a = t.SuggestFloat("a", -3, 3);
            long b = t.SuggestInt("b", 0, 10);
            return new[] { 10.0 * a * a + 0.001 * b, (a - 1) * (a - 1) - b };
        }), nTrials: 60);
        return study;
    }
}
